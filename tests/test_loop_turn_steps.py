"""The whole runner: the steps of a turn each write one `.ended` line in the turn's trace, and the requests of the turn
carry that trace to the app."""

from __future__ import annotations

from http import HTTPStatus
from urllib.parse import urlsplit

import pytest
from agents.backend.fake_gateway import Reply

from personas.decision import NO_ANSWER_REASON
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import LIST_TOOLS, LISTEN, lines_of, written_by_the_runner

pytestmark = pytest.mark.anyio

STEPS = ("turn_wait.ended", "token_hold.ended", "session_open.ended", "tool_list.ended", "turn_setup.ended")


async def a_written_turn(runner) -> None:
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()


async def test_each_step_of_a_turn_writes_one_line_with_its_duration_and_outcome_in_the_turns_trace(runner, written):
    await a_written_turn(runner)

    lines = written_by_the_runner(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    for step in STEPS:
        (line,) = [each for each in lines if each["event"] == step]
        assert (line["outcome"], line["trace_id"]) == ("ok", ended["trace_id"])
        assert isinstance(line["duration_ms"], float)
    assert ended["trace_id"] is not None


async def test_every_step_names_the_agent_and_the_app_it_is_at(runner, written):
    await a_written_turn(runner)

    lines = written_by_the_runner(written)
    (ended,) = [each for each in lines if each["event"] == "turn.ended"]
    for step in STEPS:
        (line,) = [each for each in lines if each["event"] == step]
        assert (line["agent_id"], line["app_id"]) == (runner.agent.client_id, ended["app_id"])


async def test_the_setup_of_a_turn_contains_the_steps_it_waits_for(runner, caplog):
    await a_written_turn(runner)

    duration = {step: lines_of(caplog.get_records("call"), step)[0]["duration_ms"] for step in STEPS}
    assert duration["turn_setup.ended"] >= max(
        duration["token_hold.ended"], duration["session_open.ended"], duration["tool_list.ended"]
    )


async def test_a_failed_step_ends_failed_with_a_closed_reason(runner, caplog):
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = LIST_TOOLS
    runner.gateway = Gateway(Reply(status=HTTPStatus.OK))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()
    await runner.app.log.until(LISTEN, 2)

    (listed,) = lines_of(caplog.get_records("call"), "tool_list.ended")
    (listed_record,) = [each for each in caplog.get_records("call") if each.getMessage() == "tool_list.ended"]
    assert listed_record.levelname == "ERROR"
    (setup,) = lines_of(caplog.get_records("call"), "turn_setup.ended")
    assert (listed["outcome"], listed["reason"]) == ("failed", NO_ANSWER_REASON)
    assert (setup["outcome"], setup["reason"]) == ("failed", NO_ANSWER_REASON)


async def test_every_request_of_a_turn_carries_its_trace_and_the_requests_of_the_loop_carry_none(runner, written):
    await a_written_turn(runner)

    (ended,) = [line for line in written_by_the_runner(written) if line["event"] == "turn.ended"]
    in_the_turn = [header for method, header in runner.app.traceparents if method == LIST_TOOLS]
    outside_the_turn = [header for method, header in runner.app.traceparents if method == LISTEN]
    assert in_the_turn
    assert {header.split("-")[1] for header in in_the_turn if header is not None} == {ended["trace_id"]}
    assert None not in in_the_turn
    assert outside_the_turn
    assert set(outside_the_turn) == {None}


async def test_a_turn_that_lists_another_apps_tools_writes_the_steps_of_each_app_and_its_token_request_in_its_trace(
    runner, songmaker, written
):
    runner.gateway = Gateway(*says(PLACE_ID, "Here."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()

    lines = written_by_the_runner(written)
    (ended,) = [line for line in lines if line["event"] == "turn.ended"]
    for step in ("session_open.ended", "tool_list.ended"):
        at = [line for line in lines if line["event"] == step]
        assert len(at) == 2
        assert {line["trace_id"] for line in at} == {ended["trace_id"]}
        assert {line["app_id"] for line in at} == {urlsplit(app.mcp_url).hostname for app in runner.config.apps}
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
    assert [line["event"] for line in lines].count("session_open.ended") == 2


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
