"""The whole runner: a pass that makes a place due writes `turn.fell_due` once, in the trace of what made it due."""

from __future__ import annotations

import secrets

import pytest
from agents.backend.fake_gateway import completion

from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.history import TABLE, settled
from tests.support.hopin import a_traceparent, trace_id_of
from tests.support.logs import written_by_the_runner
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import a_voice_call, holding_sentence_ends

pytestmark = pytest.mark.anyio

HOLD_ID = "hold-1"
FIELDS = {"agent_id", "app_id", "occasion", "spoken", "read_ms"}


def fell_due(written) -> list[dict[str, object]]:
    return [line for line in written_by_the_runner(written) if line["event"] == "turn.fell_due"]


def of_the_event(line: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in line.items() if key in FIELDS}


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
    assert set(of_the_event(line)) == FIELDS
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
    assert hold["spoken"] is True
    assert not hold.get("trace_id")


async def test_a_wake_that_makes_nothing_due_writes_no_event(runner, written):
    await restarted_in_the_call(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await settled(runner.app)

    assert fell_due(written) == []
