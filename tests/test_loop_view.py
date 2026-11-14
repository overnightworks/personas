"""The whole runner, a place that serves a view: when it is accepted, when a turn is due and what bounds the turns."""

from __future__ import annotations

import logging
import secrets
from dataclasses import replace

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import STANDING_URI, FigureRef, PlaceRef, place_view_uri, place_window_uri

from personas.config import Voice, load_config
from personas.decision import BACKOFF_BASE_SECONDS, NO_TURNS_CODE, TURN_CANCELLED_CODE
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, say_in_the_call
from tests.support.fakes import SweepClock
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.history import entry, history_of, said_and_taken, settled, view_entry, woke_at_table
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, READ, SAY_TOOL, active, formatted, invited, paused
from tests.support.realm import AGENT_NAME
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG
from tests.support.wire import cuts

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
    """The state of the view entry each turn is told as new, in the order of the turns."""
    return [
        each["state"]
        for opening in runner.gateway.openings()
        for each in opening.room["history"]["entries"]
        if each["entry"] == "view" and each["new"]
    ]


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


async def an_invited_table_at_an_app_that_binds_no_tools_read_twice(runner) -> None:
    runner.files.text = runner.files.valid_text(runner.app_url, "mcp-app", binds_tools=False)
    runner.config = load_config(runner.files.write())
    invitation = invited(TABLE.id, "inv-3", kind=TABLE.kind, serves_view=True)
    runner.app.stand(SUBJECT, invitation)
    runner.start()
    await runner.app.until_standing_read(1)
    runner.app.stand(SUBJECT, invitation)
    await runner.app.until_standing_read(2)


async def test_an_invitation_to_a_view_place_at_an_app_that_binds_no_tools_is_not_accepted_and_nothing_is_read(runner):
    await an_invited_table_at_an_app_that_binds_no_tools_read_twice(runner)

    assert runner.app.accepted == []
    assert runner.app.view_reads(TABLE) == 0


async def test_an_invitation_the_loop_leaves_out_is_reported_once_with_its_place_kind_and_a_reason_and_no_content(
    runner, caplog
):
    await an_invited_table_at_an_app_that_binds_no_tools_read_twice(runner)

    [left_out] = runner.reports.changes_of(Event.INVITATION_LEFT_OUT)
    assert (left_out.place_kind, left_out.code) == (TABLE.kind, NO_TURNS_CODE)
    logged = [line for line in formatted(caplog.get_records("call")).splitlines() if Event.INVITATION_LEFT_OUT in line]
    assert len(logged) == 1
    assert logged[0].endswith(f" code={NO_TURNS_CODE} place_kind={TABLE.kind}")
    assert not [value for value in (TABLE.id, "inv-3") if value in logged[0]]


async def test_a_table_that_serves_no_view_yet_is_accepted_with_figure_and_voice_and_his_turn_runs_once_it_is_active(
    runner,
):
    figure = FigureRef(url=f"https://figures.test/{secrets.token_hex(6)}.glb", sha256=secrets.token_hex(32))
    voice_name = f"voice-{secrets.token_hex(4)}"
    runner.config = replace(
        runner.config, agents=(replace(runner.config.agents[0], figure=figure, voice=Voice(voice_name)),)
    )
    runner.app.stand(SUBJECT, invited(TABLE.id, "inv-3", kind=TABLE.kind))
    runner.start()
    await runner.reports.until(Event.JOINED)
    [(_, wire)] = runner.app.accepted_on_the_wire
    runner.app.stand(SUBJECT, at_table())
    told = waits(runner, 1)
    await runner.gateway.until_turns(1)

    assert wire["figure"] == {"url": figure.url, "sha256": figure.sha256}
    assert wire["voice"] == voice_name
    assert told_states(runner) == [told]
    assert runner.app.window_reads(TABLE.id, TABLE.kind) > 0


async def test_an_invitation_to_a_place_that_serves_a_view_is_accepted_once_and_the_place_is_then_served(runner):
    told = []

    def seated_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, at_table())
            told.append(waits(runner, 1))

    runner.on_report = seated_once_joined
    runner.app.stand(SUBJECT, invited(TABLE.id, "inv-3", kind=TABLE.kind, serves_view=True))
    runner.start()
    await runner.gateway.until_turns(1)

    assert [accept.invitation for accept in runner.app.accepted] == ["inv-3"]
    assert told_states(runner) == told
    assert runner.app.window_reads(TABLE.id, TABLE.kind) > 0


async def test_an_active_place_of_another_kind_that_serves_no_view_is_ignored_and_unread(runner):
    runner.app.stand(SUBJECT, active(TABLE.id, TABLE.kind))
    runner.start()
    await runner.app.until_standing_read(1)
    runner.app.stand(SUBJECT, active(TABLE.id, TABLE.kind))
    await runner.app.until_standing_read(2)

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
    first_state = a_state()
    runner.app.show_view(TABLE.id, 1, first_state, awaits_you=False)
    runner.start()
    await runner.app.until_view_read(TABLE, 1)
    state = waits(runner, 2)
    await runner.gateway.until_turns(1)

    app_url = runner.config.apps[0].mcp_url
    opening = runner.gateway.openings()[0]
    assert opening.room == {
        "you": AGENT_NAME,
        "history": history_of(
            [
                view_entry(app_url, first_state, revision=1, awaits_you=False),
                view_entry(app_url, state, revision=2, new=True),
            ]
        ),
        "woke": woke_at_table(app_url),
    }
    assert opening.system == runner.files.persona_prompt
    assert sorted(runner.gateway.offered_tools()) == sorted([SAY_TOOL, CREATE_SONG, GENERATE_SONG])
    assert len(runner.gateway.requests) == 1


@pytest.mark.parametrize(
    ("replies", "relay_fails"),
    [
        pytest.param(says(TABLE.id, ANSWER), False, id="the turn calls a tool that moves nothing"),
        pytest.param([completion("done")], False, id="the turn makes no call"),
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
    runner.gateway = Gateway(*replies)
    if relay_fails:
        runner.app.script(CALL, cuts)
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    await wakes_at_the_same_revision(runner, first)
    second = waits(runner, 2)
    await runner.gateway.until_turns(2)

    assert told_states(runner) == [first, second]


async def test_a_view_turn_the_budget_ended_runs_again_at_the_same_revision_after_the_pause_with_no_new_event(runner):
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, completion("done"))
    await seated_at_a_table_whose_view_waits(runner)
    await runner.clock.until_waiting(1)
    held_requests = len(runner.gateway.requests)
    runner.clock.open_gate()
    await runner.gateway.until_turns(2, why="view")

    assert held_requests == 1
    assert [opening.room["woke"] for opening in runner.gateway.openings()] == [
        woke_at_table(runner.config.apps[0].mcp_url)
    ] * 2
    assert len({opening.room["history"]["entries"][-1]["revision"] for opening in runner.gateway.openings()}) == 1


async def test_a_newer_revision_during_the_pause_replaces_the_one_that_was_due_and_does_not_end_the_pause(runner):
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, completion("done"))
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.clock.until_waiting(1)
    reads = runner.app.view_reads(TABLE)
    newer = waits(runner, 2)
    await runner.app.until_view_read(TABLE, reads + 1)
    held_requests = len(runner.gateway.requests)
    runner.clock.open_gate()
    await runner.reports.until_a_turn_completed()

    assert held_requests == 1
    assert told_states(runner) == [first, newer]
    assert len(runner.gateway.requests) == 2


async def test_a_place_left_during_the_pause_gets_no_turn_and_the_next_place_follows(runner):
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, completion("done"))
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.clock.until_waiting(1)
    runner.app.stand(SUBJECT, at_table(OTHER_TABLE))
    other = waits(runner, 1, OTHER_TABLE)
    await runner.reports.until_a_turn_completed()

    assert told_states(runner) == [first, other]
    assert len(runner.gateway.requests) == 2


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
    await runner.reports.until(Event.TURN_ENDED)
    runner.app.stand(SUBJECT, at_table())
    await runner.gateway.until_turns(1)

    assert [code for event, code in runner.reports.reported() if event is Event.TURN_ENDED] == [TURN_CANCELLED_CODE]
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
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_reaction_after_a_turn_that_left_its_revision_waiting_starts_no_second_turn_at_that_revision(runner):
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    reads = runner.app.view_reads(TABLE)
    runner.app.script(READ, cuts)
    runner.app.show_view(TABLE.id, 1, first)
    await runner.reports.until(Event.RESUMED)
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

    assert [opening.room["woke"]["why"] for opening in runner.gateway.openings()] == ["view", "lines"]
    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert len(runner.gateway.openings()) == 2


async def test_the_state_of_a_view_reaches_no_report_and_no_log_line(runner, caplog):
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.app.until_view_read(TABLE, 2)
    logged = caplog.text

    assert not [value for value in first.values() if value in logged]
    assert TABLE.id not in logged
    assert all(record.levelno < logging.ERROR for record in caplog.records)


def joined_at(runner, place: PlaceRef, *, said: tuple[str, ...] = (), revision: int = 1, awaits_you: bool = False):
    """The runner accepts the invitation to the place, which then serves a view at this revision, with the lines said
    before it is read; the states shown are collected in the list returned."""
    shown: list[dict[str, str]] = []

    def seated_once_joined(change):
        if change.event is Event.JOINED:
            for text in said:
                runner.app.speak(place.id, PERSON, "Ada", text)
            runner.app.stand(SUBJECT, at_table(place))
            state = a_state()
            runner.app.show_view(place.id, revision, state, awaits_you=awaits_you)
            shown.append(state)

    runner.on_report = seated_once_joined
    runner.app.stand(SUBJECT, invited(place.id, "inv-3", kind=place.kind, serves_view=True))
    runner.start()
    return shown


CALL_WITH_A_VIEW = PlaceRef(kind="call", id="room-3")
PLACES_WITH_A_VIEW = [
    pytest.param(TABLE, id="a table"),
    pytest.param(CALL_WITH_A_VIEW, id="a call that serves a view"),
]


@pytest.mark.parametrize("place", PLACES_WITH_A_VIEW)
async def test_the_lines_of_a_place_enter_his_history_before_the_view_of_that_wake_and_only_the_view_is_new(
    runner, place
):
    said = ("first", "second")
    shown = joined_at(runner, place, said=said, awaits_you=True)
    await runner.gateway.until_turns(1, why="view")
    await settled(runner.app)
    [state] = shown

    (opening,) = runner.gateway.openings()
    app = runner.config.apps[0].mcp_url
    assert opening.room["history"] == history_of(
        [
            *(entry(app, place.id, text, seq=seq, place_kind=place.kind) for seq, text in enumerate(said, 1)),
            {**view_entry(app, state, revision=1, new=True), "place": {"kind": place.kind, "id": place.id}},
        ]
    )
    assert opening.room["woke"]["why"] == "view"


async def test_the_lines_said_between_his_accept_and_the_first_view_are_told_though_the_place_showed_active_without_it(
    runner,
):
    runner.app.stand(SUBJECT, invited(TABLE.id, "inv-3", kind=TABLE.kind))
    runner.start()
    await runner.reports.until(Event.JOINED)
    runner.app.stand(SUBJECT, active(TABLE.id, TABLE.kind))
    await settled(runner.app)
    runner.app.speak(TABLE.id, PERSON, "Ada", "said", spoken=False)
    runner.app.speak(TABLE.id, PERSON, "Ada", "acted", act=True)
    assert runner.gateway.openings() == []
    assert runner.app.window_reads(TABLE.id, TABLE.kind) == 0
    assert runner.app.view_reads(TABLE) == 0

    runner.app.stand(SUBJECT, at_table())
    state = waits(runner, 1)
    await runner.gateway.until_turns(1, why="view")
    await settled(runner.app)

    (opening,) = runner.gateway.openings()
    app = runner.config.apps[0].mcp_url
    assert opening.room["history"] == history_of(
        [
            entry(app, TABLE.id, "said", seq=1, place_kind=TABLE.kind),
            entry(app, TABLE.id, "acted", seq=2, place_kind=TABLE.kind, act=True),
            {**view_entry(app, state, revision=1, new=True), "place": {"kind": TABLE.kind, "id": TABLE.id}},
        ]
    )


@pytest.mark.parametrize("act", [False, True], ids=["a line said", "an act line"])
async def test_a_line_of_someone_else_at_a_place_with_a_view_starts_no_turn_and_is_told_with_the_next_one(runner, act):
    joined_at(runner, TABLE)
    await runner.app.until_view_read(TABLE)
    await said_and_taken(runner.app, TABLE.id, "a remark", act=act, place_kind=TABLE.kind)
    assert runner.gateway.openings() == []
    waits(runner, 2)
    await runner.gateway.until_turns(1, why="view")

    (opening,) = runner.gateway.openings()
    [told] = [each for each in opening.room["history"]["entries"] if each["entry"] == "line"]
    assert (told["text"], told.get("act", False), told["new"]) == ("a remark", act, False)
    assert [each["new"] for each in opening.room["history"]["entries"]] == [False, False, True]


async def test_a_view_place_with_an_empty_window_is_read_view_first_then_window_with_no_extra_standing_read(runner):
    state = await seated_at_a_table_whose_view_waits(runner)
    await runner.gateway.until_turns(1, why="view")

    assert told_states(runner) == [state]
    assert runner.app.read_order[:3] == [STANDING_URI, place_view_uri(TABLE), place_window_uri(TABLE)]


async def test_an_app_that_lists_the_window_and_answers_its_read_as_unknown_costs_no_extra_turn_and_no_failure(runner):
    runner.app.without_window.add(TABLE.id)
    first = await seated_at_a_table_whose_view_waits(runner)
    await runner.gateway.until_turns(1, why="view")
    await wakes_at_the_same_revision(runner, first)
    second = waits(runner, 2)
    await runner.gateway.until_turns(2, why="view")

    assert told_states(runner) == [first, second]
    assert runner.app.read_order[:4] == [STANDING_URI, place_view_uri(TABLE), place_window_uri(TABLE), STANDING_URI]
    assert runner.reports.changes_of(Event.BACKOFF) == []


async def test_a_window_that_fails_at_a_view_place_backs_off_and_the_resume_gives_one_view_entry_and_one_turn(runner):
    runner.app.failing_windows.add(TABLE.id)
    state = await seated_at_a_table_whose_view_waits(runner)
    await runner.reports.until(Event.BACKOFF)
    assert runner.gateway.openings() == []
    runner.app.failing_windows.discard(TABLE.id)
    await runner.gateway.until_turns(1, why="view")
    await settled(runner.app)

    (opening,) = runner.gateway.openings()
    app = runner.config.apps[0].mcp_url
    assert opening.room["history"] == history_of([view_entry(app, state, revision=1, new=True)])


async def test_the_lines_a_view_place_held_before_a_restart_are_never_taken_and_the_ones_said_after_it_are(runner):
    runner.app.stand(SUBJECT, at_table())
    runner.app.speak(TABLE.id, PERSON, "Ada", "before the restart")
    runner.app.show_view(TABLE.id, 1, a_state(), awaits_you=False)
    runner.start()
    await runner.app.until_window_read(TABLE.id, kind=TABLE.kind)
    await settled(runner.app)
    await said_and_taken(runner.app, TABLE.id, "after the restart", place_kind=TABLE.kind)
    waits(runner, 2)
    await runner.gateway.until_turns(1, why="view")

    (opening,) = runner.gateway.openings()
    assert [each["text"] for each in opening.room["history"]["entries"] if each["entry"] == "line"] == [
        "after the restart"
    ]
