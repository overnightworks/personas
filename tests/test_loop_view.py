"""The whole runner, a place that serves a view: when it is accepted, when a turn is due and what bounds the turns."""

from __future__ import annotations

import logging
import secrets
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply, completion, tool_call, tool_round
from agents.member import STANDING_URI, PlaceRef

from personas.config import load_config
from personas.decision import TURN_CANCELLED_CODE
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, say_in_the_call
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, READ, SAY_TOOL, active, invited, paused
from tests.support.realm import AGENT_NAME
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG, ToolsOnlyApp

pytestmark = pytest.mark.anyio

TABLE = PlaceRef(kind="table", id="table-1")
OTHER_TABLE = PlaceRef(kind="table", id="table-2")


def at_table(place: PlaceRef = TABLE):
    return active(place.id, place.kind, serves_view=True)


def a_state() -> dict[str, str]:
    return {"hand": secrets.token_hex(8)}


def waits(runner, revision: int, place: PlaceRef = TABLE) -> dict[str, str]:
    """The place's view shows this revision, which waits for the agent, and says what the state is."""
    state = a_state()
    runner.app.show_view(place.id, revision, state)
    return state


def told_states(runner) -> list[object]:
    return [opening.room["state"] for opening in runner.gateway.openings()]


async def seated_at_a_table_whose_view_waits(runner, revision: int = 1) -> dict[str, str]:
    runner.app.stand(SUBJECT, at_table())
    state = waits(runner, revision)
    runner.start()
    return state


async def wakes_at_the_same_revision(runner, state, count: int = 2) -> None:
    """The app signals its view again, unmoved, and the runner has read it again for each signal."""
    for _ in range(count):
        reads = runner.app.view_reads(TABLE)
        runner.app.show_view(TABLE.id, 1, state)
        await runner.app.until_view_read(TABLE, reads + 1)


@pytest.fixture
async def songmaker(runner):
    app = ToolsOnlyApp()
    async with app.serving() as url:
        runner.add_tools_only_app(app, url)
        yield app


async def test_an_invitation_to_a_view_place_at_an_app_that_binds_no_tools_is_not_accepted_and_nothing_is_read(runner):
    runner.files.text = runner.files.valid_text(runner.app_url, "mcp-app", binds_tools=False)
    runner.config = load_config(runner.files.write())
    invitation = invited(TABLE.id, "inv-3", kind=TABLE.kind, serves_view=True)
    runner.app.stand(SUBJECT, invitation)
    runner.start()
    await runner.app.until_standing_read(1)
    runner.app.stand(SUBJECT, invitation)
    await runner.app.until_standing_read(2)

    assert runner.app.accepted == []
    assert runner.app.view_reads(TABLE) == 0


async def test_an_invitation_to_a_place_that_serves_a_view_is_accepted_once_and_the_place_is_then_served(runner):
    told = []

    def seated_once_joined(change):
        if change.event.value == "joined":
            runner.app.stand(SUBJECT, at_table())
            told.append(waits(runner, 1))

    runner.on_report = seated_once_joined
    runner.app.stand(SUBJECT, invited(TABLE.id, "inv-3", kind=TABLE.kind, serves_view=True))
    runner.start()
    await runner.gateway.until_turns(1)

    assert [accept.invitation for accept in runner.app.accepted] == ["inv-3"]
    assert told_states(runner) == told
    assert runner.app.window_reads(TABLE.id) == 0


@pytest.mark.parametrize(
    "standings",
    [
        [invited(TABLE.id, "inv-3", kind=TABLE.kind)],
        [active(TABLE.id, TABLE.kind)],
    ],
    ids=["invited to a place that serves no view", "active in a place that serves no view"],
)
async def test_a_place_of_another_kind_that_serves_no_view_is_ignored_unread_and_unaccepted(runner, standings):
    runner.app.stand(SUBJECT, *standings)
    runner.start()
    await runner.app.until_standing_read(1)
    runner.app.stand(SUBJECT, *standings)
    await runner.app.until_standing_read(2)

    assert runner.app.accepted == []
    assert set(runner.app.reads) == {STANDING_URI}


async def test_a_view_that_does_not_wait_for_him_or_sits_in_a_paused_place_starts_no_turn(runner):
    runner.app.stand(SUBJECT, at_table(), paused(OTHER_TABLE.id, OTHER_TABLE.kind, serves_view=True))
    runner.app.show_view(TABLE.id, 1, a_state(), awaits_you=False)
    runner.app.show_view(OTHER_TABLE.id, 1, a_state())
    runner.start()
    await runner.app.until_view_read(TABLE, 1)
    runner.app.show_view(TABLE.id, 2, a_state(), awaits_you=False)
    await runner.app.until_view_read(TABLE, 2)
    state = waits(runner, 3)
    await runner.gateway.until_turns(1)

    assert told_states(runner) == [state]
    assert runner.app.view_reads(OTHER_TABLE) == 0


async def test_a_view_that_moves_to_a_revision_waiting_for_him_gets_one_turn_told_the_state_and_no_instruction(
    runner, songmaker
):
    runner.app.stand(SUBJECT, at_table())
    runner.app.show_view(TABLE.id, 1, a_state(), awaits_you=False)
    runner.start()
    await runner.app.until_view_read(TABLE, 1)
    state = waits(runner, 2)
    await runner.gateway.until_turns(1)

    opening = runner.gateway.openings()[0]
    assert opening.room == {
        "app": runner.config.apps[0].mcp_url,
        "kind": TABLE.kind,
        "id": TABLE.id,
        "you": AGENT_NAME,
        "awaits_you": True,
        "state": state,
        "history": {"older_dropped": False, "entries": []},
    }
    assert opening.system == runner.files.persona_prompt
    assert sorted(runner.gateway.offered_tools()) == sorted([SAY_TOOL, CREATE_SONG, GENERATE_SONG])
    assert len(runner.gateway.requests) == 1


@pytest.mark.parametrize(
    ("replies", "relay_fails"),
    [
        pytest.param(says(TABLE.id, ANSWER), False, id="the turn calls a tool that moves nothing"),
        pytest.param([completion("done")], False, id="the turn makes no call"),
        pytest.param([Reply(status=HTTPStatus.TOO_MANY_REQUESTS)], False, id="the gateway is busy"),
        pytest.param([BUDGET_SPENT], False, id="the budget is spent"),
        pytest.param(
            [tool_round(tool_call("call-0", SAY_TOOL, {"place": TABLE.id, "text": ANSWER}))],
            True,
            id="the relayed call fails",
        ),
    ],
)
async def test_a_turn_that_ends_and_leaves_its_revision_waiting_gets_no_second_turn_however_the_loop_wakes(
    runner, replies, relay_fails
):
    async def cuts(reply):
        await reply.open_stream()
        reply.cut()

    runner.gateway = Gateway(*replies)
    if relay_fails:
        runner.app.script(CALL, cuts)
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    await wakes_at_the_same_revision(runner, first)
    second = waits(runner, 2)
    await runner.gateway.until_turns(2)

    assert told_states(runner) == [first, second]


async def test_a_turn_that_moves_the_app_to_a_revision_that_waits_again_starts_that_revision_afresh(runner):
    runner.gateway = Gateway(completion("done"))
    ending = runner.gateway.hold_request(1)
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.gateway.until_requested(1)
    second = waits(runner, 2)
    ending.set()
    await runner.gateway.until_turns(2)

    assert told_states(runner) == [first, second]


async def test_two_view_places_that_wait_at_once_get_their_turns_one_at_a_time_in_the_order_of_the_standing(runner):
    runner.app.stand(SUBJECT, at_table(TABLE), at_table(OTHER_TABLE))
    first = waits(runner, 1, TABLE)
    second = waits(runner, 1, OTHER_TABLE)
    ending = runner.gateway.hold_request(1)
    runner.start()
    await runner.gateway.until_requested(1)
    reads = runner.app.reads[STANDING_URI]
    runner.app.stand(SUBJECT, at_table(TABLE), at_table(OTHER_TABLE))
    await runner.app.until_standing_read(reads + 1)
    assert runner.gateway.requests == []
    runner.app.show_view(TABLE.id, 2, a_state(), awaits_you=False)
    ending.set()
    await runner.gateway.until_turns(2)

    assert told_states(runner) == [first, second]


async def test_a_place_that_leaves_the_standing_during_its_turn_cancels_it_and_is_forgotten(runner):
    runner.gateway = Gateway(completion("done"))
    runner.gateway.hold_request(1)
    state = await seated_at_a_table_whose_view_waits(runner)
    await runner.gateway.until_requested(1)

    runner.app.stand(SUBJECT)
    await runner.reports.until("turn_ended")
    runner.app.stand(SUBJECT, at_table())
    await runner.gateway.until_turns(1)

    assert [code for event, code in runner.reports.events() if event == "turn_ended"] == [TURN_CANCELLED_CODE]
    assert told_states(runner) == [state]


async def test_a_place_that_leaves_and_comes_back_at_its_waiting_revision_gets_one_turn(runner):
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    standing_reads = runner.app.reads[STANDING_URI]
    runner.app.stand(SUBJECT)
    await runner.app.until_standing_read(standing_reads + 1)
    runner.app.stand(SUBJECT, at_table())
    await runner.gateway.until_turns(2)
    reads = runner.app.view_reads(TABLE)
    await runner.app.until_view_read(TABLE, reads + 1)
    second = waits(runner, 2)
    await runner.gateway.until_turns(3)

    assert told_states(runner) == [first, first, second]


async def test_a_view_he_cannot_read_starts_no_turn_and_fails_nothing_until_the_standing_shows_it_again(runner):
    runner.app.vanished.add(TABLE.id)
    await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 1)
    runner.app.vanished.discard(TABLE.id)
    state = waits(runner, 2)
    await runner.gateway.until_turns(1)

    assert told_states(runner) == [state]
    assert runner.reports.events() == [("started", None)]


async def test_a_reaction_after_a_turn_that_left_its_revision_waiting_starts_no_second_turn_at_that_revision(runner):
    async def cuts(reply):
        await reply.open_stream()
        reply.cut()

    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    reads = runner.app.view_reads(TABLE)
    runner.app.script(READ, cuts)
    runner.app.show_view(TABLE.id, 1, first)
    await runner.reports.until("resumed")
    await runner.app.until_view_read(TABLE, reads + 1)
    second = waits(runner, 2)
    await runner.gateway.until_turns(2)

    assert told_states(runner) == [first, second]


async def test_a_line_that_addresses_him_while_a_view_turn_runs_gets_its_turn_once_that_one_has_ended(runner):
    runner.gateway = Gateway(completion("done"), *says(PLACE_ID, ANSWER))
    view_turn = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    waits(runner, 1)
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.gateway.until_requested(1)
    say_in_the_call(runner, ADDRESSED)
    runner.app.show_view(TABLE.id, 2, a_state(), awaits_you=False)
    view_turn.set()
    await runner.app.said_count(1)

    assert [opening.room.get("state") is not None for opening in runner.gateway.openings()] == [True, False]
    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert len(runner.gateway.openings()) == 2


async def test_the_state_of_a_view_reaches_no_report_and_no_log_line(runner, caplog):
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    logged = caplog.text

    assert not [value for value in first.values() if value in logged]
    assert TABLE.id not in logged
    assert all(record.levelno < logging.ERROR for record in caplog.records)
