"""The whole runner: every way a turn ends writes one `turn.ended` line in a trace of the turn's own."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply, completion
from agents.turn import RejectionCode

from personas.decision import NETWORK_FAILURE_CODE
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import CANCELLED_TURN, COMPLETED_TURN, refused_turn
from tests.support.gateway import Gateway, says
from tests.support.hopin import CALL, LISTEN
from tests.support.logs import lines_of, written_by_the_runner
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.voice import TWO_SENTENCES, a_voice_call, ended_by, holding_sentence_ends
from tests.support.wire import cuts

pytestmark = pytest.mark.anyio

SILENT = "[silent]"
_COUNTS = ("sentence_count", "silent_count", "tool_call_count")


def the_turn_ended_line(caplog) -> dict[str, object]:
    (ended,) = lines_of(caplog.get_records("call"), "turn.ended")
    return ended


def counted(line: dict[str, object]) -> tuple[object, ...]:
    return tuple(line[key] for key in _COUNTS)


async def a_written_turn(runner, *replies) -> None:
    runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()


async def a_spoken_turn(runner, *replies) -> None:
    await a_voice_call(runner, *replies)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until_a_turn_ended()


async def test_a_written_turn_that_called_a_tool_and_answered_says_its_call_and_no_sentence_or_first_output(
    runner, caplog
):
    await a_written_turn(runner, *says(PLACE_ID, "Here."))

    line = the_turn_ended_line(caplog)
    assert (line["outcome"], counted(line)) == ("ok", (0, 0, 1))
    assert "first_output_ms" not in line
    assert isinstance(line["duration_ms"], float)


async def test_a_spoken_turn_counts_the_sentences_it_sent_and_marks_when_the_first_one_went_out(runner, caplog):
    """LOG-29."""
    await a_spoken_turn(runner, completion(TWO_SENTENCES))

    line = the_turn_ended_line(caplog)
    assert (line["outcome"], counted(line)) == ("ok", (2, 0, 0))
    assert isinstance(line["first_output_ms"], float)
    assert line["first_output_ms"] <= line["duration_ms"]


async def test_a_silent_turn_counts_its_silent_round_and_has_no_sentence_or_first_output(runner, caplog):
    """LOG-29."""
    await a_spoken_turn(runner, completion(SILENT))

    line = the_turn_ended_line(caplog)
    assert (line["outcome"], counted(line)) == ("ok", (0, 1, 0))
    assert "first_output_ms" not in line


async def test_a_turn_the_gateway_refused_ends_refused_with_the_code_as_its_reason(runner, caplog):
    """LOG-28."""
    await a_written_turn(runner, Reply(status=HTTPStatus.TOO_MANY_REQUESTS))

    assert runner.reports.turn_endings == [refused_turn(RejectionCode.BUSY)]
    assert counted(the_turn_ended_line(caplog)) == (0, 0, 0)


async def test_a_turn_whose_relayed_call_failed_ends_failed_with_a_closed_reason_at_error_level(runner, caplog):
    """LOG-28."""
    runner.gateway = Gateway(*says(PLACE_ID, "Here.")[:1], *says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    runner.app.script(CALL, cuts)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()
    await runner.app.log.until(LISTEN, 2)

    line = the_turn_ended_line(caplog)
    (record,) = [each for each in caplog.get_records("call") if each.getMessage() == "turn.ended"]
    assert (line["outcome"], line["reason"], record.levelname) == ("failed", NETWORK_FAILURE_CODE, "ERROR")


async def test_a_turn_whose_place_left_before_its_reply_ended_ends_cancelled_without_a_reason(runner, caplog):
    """LOG-28."""
    await a_voice_call(runner, completion(TWO_SENTENCES))
    runner.gateway.pause_request(1, "First sentence. Second sen", "tence.")
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.gateway.until_requested(1)

    runner.app.stand(SUBJECT)
    await runner.reports.until_a_turn_ended()

    line = the_turn_ended_line(caplog)
    assert (runner.reports.turn_endings, "reason" in line) == ([CANCELLED_TURN], False)


async def test_a_turn_whose_reply_ended_ends_ok_with_the_sentences_it_released_though_none_was_answered(runner, caplog):
    """TURNS-25, LOG-28."""
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    await runner.reports.until_a_turn_ended()

    line = the_turn_ended_line(caplog)
    assert (runner.reports.turn_endings, line["sentence_count"], "reason" in line) == ([COMPLETED_TURN], 2, False)


async def test_a_turn_the_runner_cancelled_ends_cancelled_though_the_loop_did_not_ask_for_it(runner):
    runner.gateway = Gateway(completion("done"))
    runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    runner.task.cancel()
    await ended_by(runner.task)

    assert runner.reports.turn_endings == [CANCELLED_TURN]


async def test_every_line_a_turn_writes_shares_its_trace_and_no_line_outside_it_carries_one(written, runner):
    """LOG-4."""
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()

    lines = written_by_the_runner(written)
    in_the_turn = [line for line in lines if line["trace_id"] is not None]
    # The listing runs beside the claim's write and the thinking statement, so its line stands among theirs.
    events = [line["event"] for line in in_the_turn]
    assert [event for event in events if event != "tool_list.ended"] == [
        "turn_wait.ended",
        "claim_write.ended",
        "thinking.ended",
        "turn_setup.ended",
        "turns.paused",
        "turn.ended",
    ]
    assert events.count("tool_list.ended") == 1
    assert len({line["trace_id"] for line in in_the_turn}) == 1
    assert {line["event"] for line in lines} > {"loop.started", "turn.ended"}
    assert {line["trace_id"] for line in lines if line not in in_the_turn} == {None}


async def test_each_turn_has_a_trace_of_its_own(written, runner):
    runner.gateway = Gateway(*says(PLACE_ID, "One.", "Two."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.reports.until_a_turn_completed(count=2)

    traces = [line["trace_id"] for line in written_by_the_runner(written) if line["event"] == "turn.ended"]
    assert len(traces) == 2
    assert None not in traces
    assert traces[0] != traces[1]
    assert runner.reports.turn_endings == [COMPLETED_TURN, COMPLETED_TURN]
