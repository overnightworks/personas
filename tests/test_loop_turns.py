"""The whole runner, one turn at a time: a turn of an agent runs across all his apps one after the other, each place
that has one due gets it in the order it became due, and a line that is not his own makes one due."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply, completion
from agents.turn import Principal, person_fingerprint

from personas.decision import TURN_CANCELLED_CODE
from tests.support.call import ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.history import (
    OTHER_CALL_ID,
    REMARK,
    SPEAKER,
    TABLE,
    WindowReturns,
    at_table,
    said_and_taken,
    settled,
    url_of,
    view_turns,
    waits,
    written_turns,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, ISSUER, LIMIT_SECONDS, READ, active, paused
from tests.support.transport import ClosingHold
from tests.support.voice import HeldSleeps, in_a_voice_call_and_at_a_table

pytestmark = pytest.mark.anyio

AGAIN = "And once more."
SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME = "agent-bach", "agent-subject-2", "Bach"


def reported(runner) -> list[tuple[str, str | None]]:
    """What the loops reported besides the start of each."""
    return [each for each in runner.reports.events() if each != ("started", None)]


def turn_order(runner) -> list[tuple[int, str]]:
    """Each turn in the order it was asked of the model: the index of its app and why it ran."""
    apps = [each.mcp_url for each in runner.config.apps]
    return [(apps.index(each.room["woke"]["app"]), each.room["woke"]["why"]) for each in runner.gateway.turns()]


async def at_a_table_turn_that_waits_for_the_model(runner, other):
    """His table turn at the second app waits for the model; the call of the first app has been read once."""
    runner.gateway = Gateway(completion("done"))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    waits(other, 1)
    await runner.gateway.until_requested(1)
    return running


async def a_call_turn_that_waits_for_the_model(runner, other):
    """His call turn at the first app waits for the model; the call of the second app has been read once."""
    runner.gateway = Gateway(completion("done"))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, active(OTHER_CALL_ID), at_table())
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await other.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    return running


async def a_call_turn_beside_a_table_that_waits_for_the_model(runner):
    """His call turn waits for the model; the call and the table it has are both at the first app."""
    runner.gateway = Gateway(completion("done"))
    running = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    return running


async def test_a_line_that_does_not_name_him_gives_one_turn_told_it_as_new_at_its_place(runner):
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "Hello everyone", author=PERSON)
    await runner.gateway.until_turns(1, why="lines")

    (turn,) = runner.gateway.openings()
    assert turn.new_seqs == [1]
    assert turn.room["woke"]["place"] == {"kind": "call", "id": PLACE_ID}


async def test_a_turn_that_calls_nothing_says_nothing_reports_nothing_and_gives_no_further_turn(runner):
    await restarted_in_the_call(runner)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_turns(1, why="lines")
    await settled(runner.app)
    await settled(runner.app)

    assert runner.app.said == []
    assert reported(runner) == []
    assert len(runner.gateway.requests) == 1
    assert [each["entry"] for each in runner.gateway.openings()[0].room["history"]["entries"]] == ["line"]


async def test_a_line_said_in_his_call_while_his_table_turn_runs_is_taken_at_once_and_gets_its_turn_after_it(
    runner, other
):
    running = await at_a_table_turn_that_waits_for_the_model(runner, other)

    await said_and_taken(runner.app, PLACE_ID, REMARK)

    assert runner.gateway.arrivals == 1
    running.set()
    await runner.gateway.until_turns(2)
    assert turn_order(runner) == [(1, "view"), (0, "lines")]
    table_turn, call_turn = runner.gateway.openings()
    assert [each["entry"] for each in table_turn.room["history"]["entries"]] == ["view"]
    assert call_turn.new_seqs == [1]
    assert [each["entry"] for each in call_turn.room["history"]["entries"]] == ["view", "line"]


async def test_a_table_that_starts_waiting_at_another_app_during_his_call_turn_goes_before_a_line_said_after_it(
    runner, other
):
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    waits(other, 1)
    await settled(other)
    await said_and_taken(runner.app, PLACE_ID, AGAIN)

    running.set()
    await runner.gateway.until_turns(3)

    assert turn_order(runner) == [(0, "lines"), (1, "view"), (0, "lines")]
    _, table_turn, call_turn = runner.gateway.openings()
    assert [(each["entry"], each["new"]) for each in table_turn.room["history"]["entries"]] == [
        ("line", False),
        ("view", True),
        ("line", False),
    ]
    assert call_turn.new_seqs == [2]
    assert [(each["entry"], each["new"]) for each in call_turn.room["history"]["entries"]] == [
        ("line", False),
        ("view", False),
        ("line", True),
    ]


async def test_lines_said_in_three_reads_during_another_apps_turn_are_told_together_in_one_turn_after_it(runner, other):
    running = await at_a_table_turn_that_waits_for_the_model(runner, other)
    for number in (1, 2, 3):
        await said_and_taken(runner.app, PLACE_ID, f"{REMARK} {number}")

    running.set()
    await runner.gateway.until_turns(1, why="lines")
    await settled(runner.app)
    await settled(runner.app)

    (written,) = written_turns(runner)
    assert written.new_seqs == [1, 2, 3]
    assert len(runner.gateway.requests) == 2


async def test_a_line_in_his_own_call_while_his_call_turn_runs_enters_the_history_and_is_told_in_the_next_turn(
    runner, other
):
    running = await a_call_turn_that_waits_for_the_model(runner, other)

    await said_and_taken(runner.app, PLACE_ID, AGAIN)

    running.set()
    await runner.gateway.until_written_turn(PLACE_ID, 2)
    first, second = written_turns(runner)
    assert [(each["seq"], each["new"]) for each in first.room["history"]["entries"] if each["entry"] == "line"] == [
        (1, True)
    ]
    assert second.new_seqs == [2]


async def test_the_place_that_waited_longest_goes_first_when_a_table_starts_waiting_before_a_line_is_said(runner):
    running = await a_call_turn_beside_a_table_that_waits_for_the_model(runner)
    waits(runner.app, 1)
    await settled(runner.app)
    await said_and_taken(runner.app, PLACE_ID, AGAIN)

    running.set()
    await runner.gateway.until_turns(3)

    assert turn_order(runner) == [(0, "lines"), (0, "view"), (0, "lines")]
    last = runner.gateway.openings()[2]
    assert last.new_seqs == [2]
    assert [each["entry"] for each in last.room["history"]["entries"]] == ["line", "view", "line"]


async def test_the_place_that_waited_longest_goes_first_when_a_line_is_read_before_a_table_starts_waiting(runner):
    running = await a_call_turn_beside_a_table_that_waits_for_the_model(runner)
    await said_and_taken(runner.app, PLACE_ID, AGAIN)
    state = waits(runner.app, 1)
    await settled(runner.app)

    running.set()
    await runner.gateway.until_turns(3)

    assert turn_order(runner) == [(0, "lines"), (0, "lines"), (0, "view")]
    second = runner.gateway.openings()[1]
    assert second.new_seqs == [2]
    assert [each.get("state") for each in second.room["history"]["entries"]] == [None, None, state]


async def test_a_table_that_moves_to_a_revision_that_does_not_wait_before_his_turn_frees_gets_no_turn(runner):
    running = await a_call_turn_beside_a_table_that_waits_for_the_model(runner)
    waits(runner.app, 1)
    await settled(runner.app)
    runner.app.show_view(TABLE.id, 2, {"hand": "none"}, awaits_you=False)
    await settled(runner.app)

    running.set()
    await runner.gateway.until_turns(1, why="lines")
    await settled(runner.app)
    await settled(runner.app)

    assert view_turns(runner) == []


async def test_a_table_that_waits_at_several_revisions_during_a_turn_gets_one_turn_at_the_last_in_its_place(runner):
    running = await a_call_turn_beside_a_table_that_waits_for_the_model(runner)
    waits(runner.app, 1)
    await settled(runner.app)
    await said_and_taken(runner.app, PLACE_ID, AGAIN)
    state = waits(runner.app, 2)
    await settled(runner.app)
    await settled(runner.app)

    running.set()
    await runner.gateway.until_turns(3)
    await settled(runner.app)

    assert turn_order(runner) == [(0, "lines"), (0, "view"), (0, "lines")]
    (table,) = view_turns(runner)
    assert [
        (each["revision"], each["new"]) for each in table.room["history"]["entries"] if each["entry"] == "view"
    ] == [
        (1, False),
        (2, True),
    ]
    assert table.room["history"]["entries"][-1]["state"] == state


async def test_his_turn_does_not_wait_for_another_agents_turn(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    runner.gateway = Gateway(completion("done"))
    running = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(OTHER_CALL_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    runner.app.speak(OTHER_CALL_ID, PERSON, SPEAKER, REMARK)
    await runner.gateway.until_turns(1, why="lines")

    (second,) = runner.gateway.openings()
    assert second.room["woke"]["place"]["id"] == OTHER_CALL_ID
    running.set()


async def test_two_agents_in_one_call_each_say_once_and_then_every_turn_ends_silent_and_no_more_start(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    runner.gateway = Gateway(completion("done"))
    runner.gateway.script_for(person_of(SUBJECT), *says(PLACE_ID, ANSWER))
    runner.gateway.script_for(person_of(SECOND_SUBJECT), *says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, REMARK)
    await runner.app.said_count(2)
    seq_of = {line.author: line.seq for line in runner.app.lines[PLACE_ID]}
    await runner.gateway.until_written_turn(PLACE_ID, seq_of[SUBJECT], user=person_of(SECOND_SUBJECT))
    await runner.gateway.until_written_turn(PLACE_ID, seq_of[SECOND_SUBJECT], user=person_of(SUBJECT))
    for _ in range(3):
        await settled(runner.app)
    turns = len(runner.gateway.requests)
    for _ in range(2):
        await settled(runner.app)

    assert sorted(who for who, _, _ in runner.app.said) == sorted([SUBJECT, SECOND_SUBJECT])
    assert len(runner.gateway.requests) == turns


def person_of(subject: str) -> str:
    return person_fingerprint(Principal(iss=ISSUER, sub=subject))


async def cuts(reply):
    await reply.open_stream()
    reply.cut()


async def answers_with(status):
    async def answer(reply):
        await reply.status(status)

    return answer


async def test_a_relayed_call_that_fails_at_one_app_ends_its_turn_and_the_turn_due_at_another_app_follows(
    runner, other
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER)[:1], completion("done"))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, active(OTHER_CALL_ID))
    await restarted_in_the_call(runner)
    await other.until_window_read(OTHER_CALL_ID)
    runner.app.script(CALL, cuts)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)

    running.set()
    await runner.gateway.until_turns(2, why="lines")

    assert turn_order(runner) == [(0, "lines"), (1, "lines")]
    await runner.reports.until("resumed")
    assert reported(runner) == [("backoff", "network"), ("resumed", None)]
    assert runner.gateway.round_sizes() == [2, 2]


async def test_while_a_cancelled_turn_still_unwinds_no_turn_of_his_starts_at_another_app_until_it_ended(runner, other):
    unwinding = ClosingHold()
    runner.gateway = Gateway(completion("done"))
    runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    other.stand(SUBJECT, active(OTHER_CALL_ID))
    runner.start(open_session=unwinding.opener(of_url=runner.app_url))
    await runner.app.until_window_read(PLACE_ID)
    await other.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)

    try:
        runner.app.stand(SUBJECT)
        async with asyncio.timeout(LIMIT_SECONDS):
            await unwinding.closing.wait()
        await said_and_taken(other, OTHER_CALL_ID, AGAIN)

        assert runner.gateway.requests == []
        assert runner.gateway.arrivals == 1
        unwinding.release.set()
        await runner.gateway.until_turns(1, why="lines")
    finally:
        unwinding.release.set()

    assert turn_order(runner) == [(1, "lines")]
    assert reported(runner) == [("turn_ended", TURN_CANCELLED_CODE)]


@pytest.mark.parametrize("leaves", ["gone", "paused"], ids=["the place is gone", "the place is paused"])
async def test_a_place_that_leaves_before_its_turn_gets_none_and_a_younger_one_elsewhere_goes_when_the_slot_frees(
    runner, other, leaves
):
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)
    other.stand(SUBJECT, *([] if leaves == "gone" else [paused(OTHER_CALL_ID)]))
    await settled(other)
    await said_and_taken(runner.app, PLACE_ID, "A younger line.")

    running.set()
    await runner.gateway.until_written_turn(PLACE_ID, 2)

    assert turn_order(runner) == [(0, "lines"), (0, "lines")]


@pytest.mark.parametrize("leaves", ["gone", "paused"], ids=["the place is gone", "the place is paused"])
async def test_a_place_that_leaves_while_the_slot_is_free_lets_a_younger_due_place_at_another_app_go_at_once(
    runner, other, leaves, monkeypatch
):
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)
    await said_and_taken(runner.app, PLACE_ID, "A younger line.")
    other.vanished.add(OTHER_CALL_ID)
    held_read = other.hold_next_window_read()
    returns = WindowReturns(monkeypatch, PLACE_ID)

    running.set()
    await other.until_window_read_held()
    await returns.until(1)
    other.stand(SUBJECT, *([] if leaves == "gone" else [paused(OTHER_CALL_ID)]))
    held_read.set()
    await runner.gateway.until_written_turn(PLACE_ID, 2)
    await settled(other)
    await settled(other)

    assert turn_order(runner) == [(0, "lines"), (0, "lines")]


@pytest.mark.parametrize(
    "status", [HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.FORBIDDEN], ids=["it backs off", "it suspends"]
)
async def test_a_loop_that_backs_off_or_suspends_drops_its_due_places_and_holds_up_no_place_of_his_other_apps(
    runner, other, status
):
    sleeps = HeldSleeps(runner)
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)
    table = waits(other, 1)
    await settled(other)
    other.script(READ, await answers_with(status))
    other.stand(SUBJECT, active(OTHER_CALL_ID), at_table())
    await sleeps.until_began(1)
    await said_and_taken(runner.app, PLACE_ID, "A younger line.")

    running.set()
    await runner.gateway.until_written_turn(PLACE_ID, 2)
    assert turn_order(runner) == [(0, "lines"), (0, "lines")]
    sleeps.release.set()
    await runner.gateway.until_turns(1, why="view")

    (turn,) = view_turns(runner)
    assert turn.room["woke"]["app"] == url_of(runner, 1)
    assert [each.get("state") for each in turn.room["history"]["entries"] if each["entry"] == "view"] == [table]


@pytest.mark.parametrize(
    "reply",
    [Reply(status=HTTPStatus.TOO_MANY_REQUESTS), BUDGET_SPENT],
    ids=["rate limited", "budget spent"],
)
async def test_a_turn_that_ends_short_is_reported_and_the_turn_due_at_another_app_follows(runner, other, reply):
    runner.gateway = Gateway(reply, completion("done"))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, active(OTHER_CALL_ID))
    await restarted_in_the_call(runner)
    await other.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)

    running.set()
    await runner.gateway.until_turns(2, why="lines")

    assert turn_order(runner) == [(0, "lines"), (1, "lines")]
    assert [event for event, _ in reported(runner)] == ["turn_ended"]


async def test_a_loop_that_renews_its_token_keeps_its_due_place_and_that_place_goes_before_a_younger_one(runner, other):
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)
    other.script(READ, await answers_with(HTTPStatus.UNAUTHORIZED))
    other.stand(SUBJECT, active(OTHER_CALL_ID), at_table())
    await settled(other)
    await said_and_taken(runner.app, PLACE_ID, "A younger line.")

    running.set()
    await runner.gateway.until_turns(3, why="lines")

    assert turn_order(runner) == [(0, "lines"), (1, "lines"), (0, "lines")]
    assert runner.gateway.openings()[1].new_seqs == [1]
    assert reported(runner) == []


async def test_a_call_whose_window_the_app_lost_keeps_its_due_and_its_turn_runs_before_a_younger_place(runner, other):
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    await said_and_taken(other, OTHER_CALL_ID, AGAIN)
    other.vanished.add(OTHER_CALL_ID)
    other.stand(SUBJECT, active(OTHER_CALL_ID), at_table())
    await settled(other)
    await said_and_taken(runner.app, PLACE_ID, "A younger line.")

    running.set()
    await runner.gateway.until_turns(3, why="lines")

    assert turn_order(runner) == [(0, "lines"), (1, "lines"), (0, "lines")]
    assert runner.gateway.openings()[1].new_seqs == [1]


async def test_a_table_whose_view_the_app_no_longer_gives_loses_its_due_and_a_younger_place_goes_instead(runner, other):
    running = await a_call_turn_that_waits_for_the_model(runner, other)
    waits(other, 1)
    await settled(other)
    other.vanished.add(TABLE.id)
    other.stand(SUBJECT, active(OTHER_CALL_ID), at_table())
    await settled(other)
    await said_and_taken(runner.app, PLACE_ID, AGAIN)

    running.set()
    await runner.gateway.until_written_turn(PLACE_ID, 2)
    await settled(other)
    await settled(other)

    assert turn_order(runner) == [(0, "lines"), (0, "lines")]


async def test_a_line_heard_at_a_voice_place_is_heard_while_his_table_turn_of_that_app_runs_and_makes_no_turn_due(
    runner,
):
    await in_a_voice_call_and_at_a_table(runner, completion("table"), completion(ANSWER))
    running = runner.gateway.hold_request(1)
    waits(runner.app, 1)
    await runner.gateway.until_requested(1)
    say_in_the_call(runner, "Anyone there?", spoken=True)
    await runner.app.spoken_count(1)

    assert runner.gateway.arrivals == 2
    running.set()
    await runner.gateway.until_turns(1, why="view")
    await settled(runner.app)
    await settled(runner.app)
    assert written_turns(runner) == []
