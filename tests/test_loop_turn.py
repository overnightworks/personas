"""The whole runner, the turn: which lines start one, how it ends and what its failures lead to."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import anyio
import pytest
from agent_runtime_client.backend.fake_gateway import Reply
from agent_runtime_client.member import (
    STANDING_URI,
    place_window_uri,
)
from agent_runtime_client.tools import ToolOutcome
from agent_runtime_client.turn import person_fingerprint
from mcp_types import (
    INVALID_PARAMS,
    ListToolsResult,
)

from agents import app_session
from agents.config import load_config
from agents.decision import TOOL_NOT_DECLARED_CODE
from agents.failures import (
    Unreachable,
    leaves_of,
)
from agents.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import (
    CALL,
    DISCOVER,
    LIMIT_SECONDS,
    LIST_TOOLS,
    LISTEN,
    READ,
    active,
    formatted,
    invited,
)
from tests.support.transport import transport_closing_with_late_answers

pytestmark = pytest.mark.anyio


async def test_an_addressed_line_gets_exactly_one_say_under_the_agents_own_fingerprint(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)

    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert len(runner.gateway.requests) == 2
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1]]
    assert runner.gateway.openings()[0].system == runner.files.persona_prompt
    assert runner.gateway.users() == {person_fingerprint(runner.app_principal)}
    assert runner.reports.events() == [("started", None)]


async def test_his_own_line_naming_him_starts_no_turn(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, "Mozart will say it himself", author=SUBJECT, name="Mozart")
    read = runner.app.hold_next_window_read()
    runner.start()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_an_unaddressed_line_makes_no_gateway_call(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, "Hello everyone")
    read = runner.app.hold_next_window_read()
    runner.start()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_after_his_own_accept_reading_starts_at_seq_1(runner):
    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, ADDRESSED)

    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.on_report = addressed_once_joined
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    runner.start()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1]]


async def test_a_place_he_joins_again_is_read_from_seq_1_of_the_new_window_and_nothing_of_the_old_one_is_answered(
    runner,
):
    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, f"{ADDRESSED} In the new window.")

    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Once more."))
    runner.on_report = addressed_once_joined
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)

    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.app.said_count(2)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [1]]


async def test_a_duplicate_pulse_or_a_re_read_starts_no_second_turn_for_a_line(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Another answer."))
    closing_round = runner.gateway.hold_request(2)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(2)
    for uri in (STANDING_URI, place_window_uri(CALL_PLACE), STANDING_URI, place_window_uri(CALL_PLACE)):
        runner.app.changes.changed(runner.app_principal, uri)
    read = runner.app.hold_next_window_read()
    closing_round.set()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    read.set()
    await runner.app.said_count(2)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [3]]


async def test_a_line_said_while_a_turn_ran_is_answered_once_that_turn_ends_though_nothing_else_wakes_the_loop(runner):
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER))
    first_round = runner.gateway.hold_request(1)

    def lets_the_turn_end_once_a_reconciliation_has_passed_it_by(change):
        if change.event is Event.JOINED:
            first_round.set()

    runner.on_report = lets_the_turn_end_once_a_reconciliation_has_passed_it_by
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    runner.app.standings[SUBJECT] = [active(PLACE_ID), invited("room-9", "inv-3")]
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)
    await runner.gateway.until_requested(3)

    assert runner.gateway.round_sizes() == [2, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert runner.reports.events()[:3] == [("started", None), ("joined", None), ("turn_ended", "busy")]


async def test_one_turn_runs_at_a_time_and_a_line_said_meanwhile_is_answered_after_a_turn_that_spoke(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Another answer."))
    first_round = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    window_reads = runner.app.window_reads(PLACE_ID)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.until_standing_read(runner.app.reads[STANDING_URI] + 1)
    later_wake = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()
    assert runner.app.window_reads(PLACE_ID) == window_reads
    later_wake.set()
    first_round.set()
    await runner.app.said_count(2)
    await runner.gateway.until_requested(4)

    assert runner.gateway.round_sizes() == [2, 4, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2, 3]]


async def a_turn_ended_short(runner, code):
    """Runs the runner on one addressed line until its turn reported this code."""
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")

    assert runner.reports.events() == [("started", None), ("turn_ended", code)]
    assert runner.app.said == []
    assert runner.clock.slept == []
    assert not runner.task.done()
    assert runner.app.log.count(LISTEN) == 1


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (Reply(status=HTTPStatus.TOO_MANY_REQUESTS, headers={"Retry-After": "3"}), "busy"),
        (Reply(status=HTTPStatus.FORBIDDEN), "provider_unavailable"),
        (Reply(status=HTTPStatus.SERVICE_UNAVAILABLE), "provider_unavailable"),
    ],
    ids=["rate limited", "refused", "failing"],
)
async def test_a_gateway_refusal_is_silent_logs_one_code_and_leaves_the_lifecycle_untouched(runner, reply, code):
    runner.gateway = Gateway(reply)

    await a_turn_ended_short(runner, code)

    assert len(runner.gateway.requests) == 1


async def test_a_key_the_gateway_cannot_use_ends_the_turn_before_any_gateway_call(runner):
    runner.files.key_path.write_text("a key with spaces\n", encoding="utf-8")

    await a_turn_ended_short(runner, "provider_unavailable")

    assert runner.gateway.requests == []


async def test_a_missing_api_group_is_refused_with_no_gateway_call(runner):
    runner.realm.groups = []

    await a_turn_ended_short(runner, "not_enabled")

    assert runner.gateway.requests == []


async def test_a_tool_that_answers_gone_shows_the_model_that_and_leaves_the_lifecycle_untouched(runner):
    gone = ToolOutcome("the place is gone", is_error=True)
    runner.app.say_outcomes.append(gone)
    runner.app.content.add(gone.content)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(2)

    assert gone.content in runner.gateway.requests[1].content.decode()
    assert runner.app.said == []
    assert runner.reports.events() == [("started", None)]
    assert runner.clock.slept == []


async def test_a_bound_tool_the_app_declares_no_more_ends_the_turn_with_a_code(runner):
    async def lists_no_tools(reply):
        await reply.result(ListToolsResult(tools=[]))

    runner.app.script(LIST_TOOLS, lists_no_tools)

    await a_turn_ended_short(runner, TOOL_NOT_DECLARED_CODE)

    assert runner.gateway.requests == []


async def cuts(reply):
    await reply.open_stream()
    reply.cut()


async def refuses_the_token(reply):
    await reply.status(HTTPStatus.UNAUTHORIZED)


async def answers_wrongly(reply):
    await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)


@pytest.mark.parametrize(
    ("method", "fails", "events"),
    [
        (DISCOVER, cuts, [("started", None), ("backoff", "network"), ("resumed", None)]),
        (LIST_TOOLS, cuts, [("started", None), ("backoff", "network"), ("resumed", None)]),
        (DISCOVER, refuses_the_token, [("started", None)]),
        (LIST_TOOLS, refuses_the_token, [("started", None)]),
    ],
    ids=["discovery cut", "tool listing cut", "discovery refuses the token", "tool listing refuses the token"],
)
async def test_a_turn_whose_session_fails_before_the_model_is_asked_leads_to_one_lifecycle_reaction_and_no_second_turn(
    runner, method, fails, events
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(method, fails)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert runner.reports.events() == events
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]
    assert len(runner.realm.issued) == (2 if fails is refuses_the_token else 1)


@pytest.mark.parametrize(
    "echo", [anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"]
)
async def test_a_turn_ended_short_beside_a_late_answer_is_reported_and_costs_one_backoff_and_no_second_turn(
    runner, monkeypatch, echo
):
    failing = transport_closing_with_late_answers(echo=echo, only_close=1)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("resumed")
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.events() == [
        ("started", None),
        ("turn_ended", "busy"),
        ("backoff", "network"),
        ("resumed", None),
    ]
    assert not runner.task.done()
    assert len(runner.gateway.requests) == 1


@pytest.mark.parametrize(
    "echo", [anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"]
)
async def test_a_turn_failure_cancelling_a_lifecycle_that_unwinds_with_a_late_answer_gives_one_reaction(
    runner, monkeypatch, echo
):
    failing = transport_closing_with_late_answers(echo=echo, only_close=2)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(DISCOVER, cuts)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def relayed_say_fails(runner, fails):
    """An addressed line whose model calls ``say`` and whose first relayed call the app answers with ``fails``."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER)[:1], *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(CALL, fails)
    say_in_the_call(runner, ADDRESSED)


@pytest.mark.parametrize(
    ("fails", "events", "issued"),
    [
        (cuts, [("started", None), ("backoff", "network"), ("resumed", None)], 1),
        (refuses_the_token, [("started", None)], 2),
        (answers_wrongly, [("started", None), ("suspended", f"jsonrpc_{INVALID_PARAMS}"), ("resumed", None)], 2),
    ],
    ids=["transport", "401", "protocol error"],
)
async def test_a_relayed_call_that_fails_ends_the_turn_after_one_round_and_leads_to_one_lifecycle_reaction(
    runner, fails, events, issued
):
    await relayed_say_fails(runner, fails)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(3)

    assert runner.reports.events() == events
    assert runner.gateway.round_sizes() == [2, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert len(runner.realm.issued) == issued


async def test_a_relayed_call_beside_a_late_answer_gives_one_reaction_and_no_turn_report(runner, monkeypatch):
    failing = transport_closing_with_late_answers(echo=anyio.BrokenResourceError, only_close=1)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    await relayed_say_fails(runner, cuts)
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]
    assert len(runner.gateway.requests) == 1
    assert not runner.task.done()


async def test_a_relayed_call_that_fails_beside_a_defect_of_the_turns_teardown_ends_the_runner_loud_with_both(
    runner, monkeypatch
):
    failing = transport_closing_with_late_answers(echo=RuntimeError, only_close=1)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    await relayed_say_fails(runner, cuts)
    assert runner.task is not None
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(runner.task, return_exceptions=True)

    assert sorted(type(leaf).__name__ for leaf in leaves_of(ended)) == [RuntimeError.__name__, Unreachable.__name__]
    assert runner.reports.events() == [("started", None)]
    assert len(runner.gateway.requests) == 1


async def hangs(reply):
    await asyncio.Event().wait()


async def test_a_deadline_during_a_relayed_call_ends_the_turn_as_limit_exceeded_and_gives_no_reaction(runner):
    runner.files.persona_path.write_text(runner.files.persona_text(max_duration_seconds=1), encoding="utf-8")
    runner.config = load_config(runner.files.write())
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(CALL, hangs)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(2)

    assert runner.reports.events() == [("started", None), ("turn_ended", "limit_exceeded")]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert runner.clock.slept == []
    assert runner.task is not None
    assert not runner.task.done()


async def test_a_window_the_app_lost_is_skipped_and_is_no_suspension(runner):
    runner.gateway = Gateway(*says("room-8", ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID), active("room-8"))
    runner.app.vanished.add(PLACE_ID)
    runner.start()
    await runner.app.until_window_read("room-8")
    say_in_the_call(runner, ADDRESSED)
    runner.app.speak("room-8", PERSON, "Ada", ADDRESSED)
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, "room-8", ANSWER)]
    assert runner.reports.events() == [("started", None)]


async def test_a_refused_and_an_answered_turn_log_a_code_and_no_secret_or_content(runner, caplog):
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    logged = formatted(caplog.get_records("call"))
    assert "turn_ended" in logged
