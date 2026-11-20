"""The whole runner: the steps of a turn each write one `.ended` line in the turn's trace, and the requests of the turn
carry that trace to the app."""

from __future__ import annotations

import json
from http import HTTPStatus
from urllib.parse import urlsplit

import pytest
from agents.backend.fake_gateway import Reply

from personas.decision import NETWORK_FAILURE_CODE, NO_ANSWER_REASON
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import FakeClock
from tests.support.gateway import Gateway, says
from tests.support.hopin import LIST_TOOLS, LISTEN, a_traceparent, trace_id_of
from tests.support.logs import lines_of, written_by_the_runner
from tests.support.wire import cuts, hangs

pytestmark = pytest.mark.anyio

TOKEN_RAN_OUT_SECONDS = 10_000.0
SHORT_CALL_SECONDS = 0.3

STEPS = (
    "turn_wait.ended",
    "claim_write.ended",
    "thinking.ended",
    "tool_list.ended",
    "turn_setup.ended",
)
SEQUENTIAL_STEPS = ("turn_wait.ended", "claim_write.ended", "thinking.ended", "turn_setup.ended")


async def a_written_turn(runner, traceparent: str | None = None) -> None:
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, traceparent=traceparent)
    await runner.reports.until_a_turn_ended()


async def test_each_step_of_a_turn_writes_one_line_with_its_duration_and_outcome_in_the_turns_trace(runner, written):
    """LOG-13."""
    await a_written_turn(runner)

    lines = written_by_the_runner(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    for step in STEPS:
        (line,) = [each for each in lines if each["event"] == step]
        assert (line["outcome"], line["trace_id"]) == ("ok", ended["trace_id"])
        assert isinstance(line["duration_ms"], float)
    assert ended["trace_id"] is not None


async def test_the_waits_of_a_turn_run_in_sequence_from_its_wait_to_its_setup(runner, written):
    """LOG-33."""
    await a_written_turn(runner)

    events = [line["event"] for line in written_by_the_runner(written)]
    assert [event for event in events if event in SEQUENTIAL_STEPS] == list(SEQUENTIAL_STEPS)


async def test_every_step_names_the_agent_and_the_app_it_is_at(runner, written):
    await a_written_turn(runner)

    lines = written_by_the_runner(written)
    (ended,) = [each for each in lines if each["event"] == "turn.ended"]
    for step in STEPS:
        (line,) = [each for each in lines if each["event"] == step]
        assert (line["agent_id"], line["app_id"]) == (runner.agent.client_id, ended["app_id"])


async def test_the_setup_of_a_turn_begins_once_its_tool_list_ended(runner, written):
    """LOG-13."""
    await a_written_turn(runner)

    events = [line["event"] for line in written_by_the_runner(written)]
    assert events.index("tool_list.ended") < events.index("turn_setup.ended")


def the_app_refusing_the_tool_list(runner):
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = LIST_TOOLS


async def test_a_failed_step_ends_failed_with_a_closed_reason(runner, caplog):
    """LOG-13."""
    runner.gateway = Gateway(Reply(status=HTTPStatus.OK))
    await restarted_in_the_call(runner)
    the_app_refusing_the_tool_list(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()
    await runner.app.log.until(LISTEN, 2)

    (failed,) = lines_of(caplog.get_records("call"), "tool_list.ended")
    (failed_record,) = [each for each in caplog.get_records("call") if each.getMessage() == "tool_list.ended"]
    assert failed_record.levelname == "ERROR"
    assert (failed["outcome"], failed["reason"]) == ("failed", NO_ANSWER_REASON)
    assert failed["error_type"]
    assert lines_of(caplog.get_records("call"), "turn_setup.ended") == []


async def test_a_tool_list_the_turns_own_app_does_not_answer_in_time_ends_failed_and_the_model_round_starts(
    runner, caplog
):
    """LOG-13."""
    runner.bound_listening_calls_to(SHORT_CALL_SECONDS)
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    # The first listing is the turn's own; the second is the one the client makes before the turn's thinking statement.
    runner.app.script(LIST_TOOLS, hangs, hangs)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()

    (failed,) = lines_of(caplog.get_records("call"), "tool_list.ended")
    (setup,) = lines_of(caplog.get_records("call"), "turn_setup.ended")
    assert (failed["outcome"], failed["reason"]) == ("failed", NETWORK_FAILURE_CODE)
    assert setup["outcome"] == "ok"
    assert runner.gateway.arrivals == 1


async def test_a_tool_list_the_turns_own_app_loses_its_stream_in_ends_the_turn_before_its_setup(runner, caplog):
    """LOG-13."""
    runner.bound_listening_calls_to(SHORT_CALL_SECONDS)
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    # The first listing is the turn's own; the second is the one the client makes before the turn's thinking statement.
    runner.app.script(LIST_TOOLS, cuts, hangs)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()
    await runner.app.log.until(LISTEN, 2)

    (failed,) = lines_of(caplog.get_records("call"), "tool_list.ended")
    assert (failed["outcome"], failed["reason"]) == ("failed", NETWORK_FAILURE_CODE)
    assert lines_of(caplog.get_records("call"), "turn_setup.ended") == []


async def test_every_request_of_a_turn_carries_its_trace_and_the_requests_of_the_loop_carry_none(runner, written):
    """LOG-4."""
    await a_written_turn(runner)

    (ended,) = [line for line in written_by_the_runner(written) if line["event"] == "turn.ended"]
    in_the_turn = [header for method, header in runner.app.traceparents if method == LIST_TOOLS]
    outside_the_turn = [header for method, header in runner.app.traceparents if method == LISTEN]
    assert in_the_turn
    assert {header.split("-")[1] for header in in_the_turn if header is not None} == {ended["trace_id"]}
    assert None not in in_the_turn
    assert outside_the_turn
    assert set(outside_the_turn) == {None}


async def test_a_turn_woken_by_a_line_writes_its_wait_its_steps_and_its_end_in_the_trace_the_line_entered_in(
    runner, written
):
    """LOG-4, LOG-27."""
    traceparent = a_traceparent()
    await a_written_turn(runner, traceparent)

    lines = written_by_the_runner(written)
    for event in ("turn.ended", *STEPS):
        (line,) = [each for each in lines if each["event"] == event]
        assert line["trace_id"] == trace_id_of(traceparent)


async def test_the_requests_of_a_turn_woken_by_a_line_pass_the_trace_the_line_entered_in_to_the_app_and_the_gateway(
    runner,
):
    traceparent = a_traceparent()
    await a_written_turn(runner, traceparent)

    to_the_app = [header for method, header in runner.app.traceparents if method == LIST_TOOLS]
    to_the_gateway = [request.headers.get("traceparent") for request in runner.gateway.requests]
    assert to_the_app
    assert to_the_gateway
    assert {None if header is None else trace_id_of(header) for header in to_the_app + to_the_gateway} == {
        trace_id_of(traceparent)
    }


@pytest.mark.parametrize(
    "traceparent",
    [pytest.param(None, id="none"), pytest.param("not-a-traceparent", id="a malformed one")],
)
async def test_a_turn_woken_by_a_line_without_a_trace_it_can_continue_opens_a_trace_of_its_own(
    runner, written, traceparent
):
    """LOG-4."""
    await a_written_turn(runner, traceparent)

    lines = written_by_the_runner(written)
    traces = {line["trace_id"] for line in lines if line["event"] in {"turn.ended", *STEPS}}
    (trace_id,) = traces
    assert trace_id is not None


def all_written_lines(written):
    return [json.loads(line) for line in written.getvalue().splitlines()]


async def test_the_gateway_request_of_a_turn_and_its_tool_call_carry_the_trace_of_the_turn(runner, written):
    await a_written_turn(runner)

    lines = all_written_lines(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    tool_calls = [line for line in lines if line["event"] == "tool_call.ended"]
    headers = [request.headers.get("traceparent") for request in runner.gateway.requests]
    assert headers
    assert None not in headers
    assert {header.split("-")[1] for header in headers if header is not None} == {ended["trace_id"]}
    assert tool_calls
    assert {line["trace_id"] for line in tool_calls} == {ended["trace_id"]}


async def test_the_gateway_request_line_agents_writes_for_a_turn_carries_the_trace_of_the_turn(runner, written):
    """LOG-4."""
    await a_written_turn(runner)

    lines = all_written_lines(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    requests = [line for line in lines if line["event"] == "model_request.ended"]
    assert requests
    assert {line["trace_id"] for line in requests} == {ended["trace_id"]}


async def a_turn_with_another_apps_tools(runner):
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()


@pytest.mark.parametrize(("step", "count"), [("session_open.ended", 1), ("tool_list.ended", 2)])
async def test_a_turn_that_lists_another_apps_tools_writes_each_step_of_it_in_its_trace(
    runner, songmaker, written, step, count
):
    await a_turn_with_another_apps_tools(runner)

    lines = written_by_the_runner(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    at = [line for line in lines if line["event"] == step]
    assert len(at) == count
    assert {line["trace_id"] for line in at} == {ended["trace_id"]}


async def test_a_turn_lists_the_tools_of_each_app_once(runner, songmaker, written):
    await a_turn_with_another_apps_tools(runner)

    listed = [line for line in written_by_the_runner(written) if line["event"] == "tool_list.ended"]
    assert {line["app_id"] for line in listed} == {urlsplit(app.mcp_url).hostname for app in runner.config.apps}


async def test_a_turn_requests_its_token_in_its_trace(runner, songmaker, written):
    await a_turn_with_another_apps_tools(runner)

    lines = written_by_the_runner(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    in_the_turn = [line for line in lines if line["event"] == "token_request.ended" and line["trace_id"] is not None]
    assert [line["trace_id"] for line in in_the_turn] == [ended["trace_id"]]
    assert [header.split("-")[1] for header in runner.realm.traceparents if header is not None] == [ended["trace_id"]]


async def test_a_failed_listing_of_another_app_ends_its_tool_list_line_failed_and_the_turn_goes_on(
    runner, songmaker, written
):
    songmaker.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    songmaker.refused_method = LIST_TOOLS
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()

    lines = written_by_the_runner(written)
    listed = [line for line in lines if line["event"] == "tool_list.ended"]
    assert sorted(str(line["outcome"]) for line in listed) == ["failed", "ok"]
    assert [line["event"] for line in lines].count("session_open.ended") == 1


async def test_a_failed_step_of_another_app_is_written_as_a_handled_failure(runner, songmaker, caplog):
    songmaker.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    songmaker.refused_method = LIST_TOOLS
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()

    levels = [
        record.levelname
        for record in caplog.get_records("call")
        if record.getMessage() == "tool_list.ended" and str(record.outcome) == "failed"
    ]
    assert levels == ["WARNING"]


async def test_a_turn_due_again_after_a_pause_waits_from_the_pause_end(runner, caplog):
    """LOG-27."""
    passed_after_pause = 5.0

    class PausingClock(FakeClock):
        async def sleep(self, seconds: float) -> None:
            await super().sleep(seconds)
            self.time += passed_after_pause

    runner.clock = PausingClock()
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()

    waits = [line["duration_ms"] for line in lines_of(caplog.get_records("call"), "turn_wait.ended")]
    assert len(waits) == 2
    assert waits[1] == pytest.approx(passed_after_pause * 1000, abs=500)
