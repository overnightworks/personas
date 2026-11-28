"""The whole runner: a pass that makes a place due writes `turn.fell_due` once, in the trace of what made it due."""

from __future__ import annotations

import secrets

import pytest
from agents.backend.fake_gateway import completion

from personas.decision import BACKOFF_BASE_SECONDS
from personas.reporting import Event
from tests.support.app_driving import said_and_taken, settled, shows
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import SweepClock
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.history import OTHER_CALL_ID, REMARK, SPEAKER, TABLE, at_table
from tests.support.hopin import a_traceparent, trace_id_of
from tests.support.logs import written_by_the_runner
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import a_voice_call, holding_sentence_ends

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

HOLD_ID = "hold-1"
FIELDS = {"agent_id", "app_id", "occasion", "spoken", "read_ms"}
LOGLINE_KEYS = {"event", "level", "logger", "service", "ts", "version", "trace_id"}


def fell_due(written) -> list[dict[str, object]]:
    return [line for line in written_by_the_runner(written) if line["event"] == "turn.fell_due"]


def event_keys(line: dict[str, object]) -> set[str]:
    """The keys of the line beside those every runner line carries."""
    return set(line) - LOGLINE_KEYS


async def test_a_line_that_makes_a_place_due_writes_one_event_in_the_trace_its_turn_wait_ends_in(runner, written):
    """LOG-37."""
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    traceparent = a_traceparent()
    say_in_the_call(runner, ADDRESSED, traceparent=traceparent)
    await runner.reports.until_a_turn_ended()

    (line,) = fell_due(written)
    (wait,) = [each for each in written_by_the_runner(written) if each["event"] == "turn_wait.ended"]
    assert line["trace_id"] == wait["trace_id"] == trace_id_of(traceparent)
    assert event_keys(line) == FIELDS
    assert (line["occasion"], line["spoken"]) == ("line", False)
    assert isinstance(line["read_ms"], int)
    assert line["read_ms"] >= 0


async def test_a_view_revision_that_waits_for_him_writes_one_event_in_the_trace_the_view_moved_in(runner, written):
    """LOG-37."""
    runner.gateway = Gateway(*says(TABLE.id, ANSWER))
    runner.app.stand(SUBJECT, active(TABLE.id, TABLE.kind, serves_view=True))
    traceparent = a_traceparent()
    runner.app.show_view(TABLE.id, 1, {"hand": secrets.token_hex(8)}, traceparent=traceparent)
    runner.start()
    await runner.reports.until_a_turn_ended()

    (line,) = fell_due(written)
    assert (line["occasion"], line["trace_id"]) == ("view", trace_id_of(traceparent))
    assert event_keys(line) == FIELDS
    assert line["spoken"] is False


async def test_a_hold_that_makes_a_place_due_writes_one_event_with_no_trace(runner, written):
    """LOG-37."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(ANSWER), completion("[silent]"))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed()
    before = len(fell_due(written))

    runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", "wait a moment"))
    await runner.reports.until(Event.HOLD_DECIDED)
    await runner.reports.until_a_turn_completed(count=2)

    (hold,) = [line for line in fell_due(written) if line["occasion"] == "hold"]
    assert len(fell_due(written)) == before + 1
    assert event_keys(hold) == FIELDS
    assert hold["spoken"] is True
    assert not hold.get("trace_id")


async def test_a_wake_that_makes_nothing_due_writes_no_event(runner, written):
    """LOG-37."""
    await restarted_in_the_call(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await settled(runner.app)

    assert fell_due(written) == []


async def a_second_line_joins_the_place_that_is_due(runner, other):
    """His turn at the first app waits for the model while a line makes the call of the second app due; a second line
    joins it. The places that fell due: the first app's call and the second app's."""
    runner.gateway = Gateway(completion("done"))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, active(OTHER_CALL_ID), at_table())
    await restarted_in_the_call(runner)
    await other.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    await said_and_taken(other, OTHER_CALL_ID, "One line.")
    await said_and_taken(other, OTHER_CALL_ID, "A second line.")
    running.set()
    await runner.gateway.until_turns(2, why="lines")
    await runner.reports.until_a_turn_ended(count=2)
    return 2


async def a_marked_line_of_another_agent_is_taken(runner, other):
    await restarted_in_the_call(runner)
    await said_and_taken(
        runner.app, PLACE_ID, "First sentence.", author="agent-2", name=SPEAKER, spoken=True, continues=True
    )
    await settled(runner.app)
    return 0


async def a_turn_without_an_answer_makes_its_place_due_again(runner, other):
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, REMARK)
    await runner.clock.until_waiting(1)
    runner.clock.open_gate()
    await runner.reports.until_a_turn_completed()
    return 1


async def a_view_that_awaits_him_is_read_again_while_its_place_is_due(runner, other):
    """His turn at the call waits for the model while a view revision that awaits him makes the table due; the view is
    read again, moved but not to another revision. The places that fell due: the call and the table."""
    runner.gateway = Gateway(completion("done"), completion("done"))
    running = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID), active(TABLE.id, TABLE.kind, serves_view=True))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, REMARK)
    await runner.gateway.until_requested(1)
    await shows(runner.app, 1, awaits_you=True)
    await shows(runner.app, 1, awaits_you=True)
    running.set()
    await runner.gateway.until_turns(2)
    await runner.reports.until_a_turn_ended(count=2)
    return 2


@pytest.mark.parametrize(
    "arranged",
    [
        a_second_line_joins_the_place_that_is_due,
        a_marked_line_of_another_agent_is_taken,
        a_turn_without_an_answer_makes_its_place_due_again,
        a_view_that_awaits_him_is_read_again_while_its_place_is_due,
    ],
    ids=[
        "a second line joins a due place",
        "a marked line",
        "a turn due again after one without an answer",
        "a view that awaits him read again while its place is due",
    ],
)
async def test_a_pass_that_makes_no_place_newly_due_writes_no_event(runner, other, written, arranged):
    """LOG-37."""
    fell = await arranged(runner, other)

    assert len(fell_due(written)) == fell
