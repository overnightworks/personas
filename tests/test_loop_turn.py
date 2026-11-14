"""The whole runner, the turn: which lines start one, how it ends and what its failures lead to."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import anyio
import pytest
from agents.backend.fake_gateway import Reply, completion
from agents.member import (
    STANDING_URI,
    place_window_uri,
)
from agents.tools import ToolOutcome
from agents.turn import FailureCode, RejectionCode, person_fingerprint
from mcp_types import (
    INVALID_PARAMS,
    ListToolsResult,
)

from personas import app_session
from personas.config import load_config
from personas.decision import BACKOFF_BASE_SECONDS, NETWORK_FAILURE_CODE, TURN_CANCELLED_CODE, jsonrpc_code
from personas.failures import (
    Unreachable,
    leaves_of,
)
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import SweepClock
from tests.support.gateway import BUDGET_SPENT, Gateway, says, says_after_a_hold
from tests.support.history import act_entry, unanswered, url_of, woke_at_lines
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import (
    CALL,
    DISCOVER,
    LIMIT_SECONDS,
    LIST_TOOLS,
    LISTEN,
    READ,
    SAY_TOOL,
    active,
    formatted,
    invited,
)
from tests.support.transport import transport_closing_with_late_answers
from tests.support.wire import cuts, hangs

pytestmark = pytest.mark.anyio

REFUSAL_FREE_ROUND = 3


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
    assert runner.reports.reported() == [(Event.STARTED, None)]


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
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [1, 2]]
    assert runner.reports.reported()[:3] == [
        (Event.STARTED, None),
        (Event.JOINED, None),
        (Event.TURN_ENDED, RejectionCode.BUSY.value),
    ]


async def test_a_line_said_while_a_turn_ran_is_read_at_once_and_answered_after_a_turn_that_spoke(runner):
    runner.gateway = Gateway(*says_after_a_hold(PLACE_ID, ANSWER), *says(PLACE_ID, "Another answer."))
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
    assert runner.app.window_reads(PLACE_ID) > window_reads
    later_wake.set()
    first_round.set()
    await runner.app.said_count(2)
    await runner.gateway.until_requested(5)

    assert runner.gateway.round_sizes() == [2, 4, 6, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]


async def test_every_line_addressed_to_him_gets_its_own_turn_one_after_the_other(runner):
    runner.gateway = Gateway(
        Reply(status=HTTPStatus.TOO_MANY_REQUESTS), BUDGET_SPENT, Reply(status=HTTPStatus.TOO_MANY_REQUESTS)
    )
    await restarted_in_the_call(runner)
    for round_number in range(1, 4):
        say_in_the_call(runner, f"{ADDRESSED} Round {round_number}.")
        await runner.reports.until(Event.TURN_ENDED, count=round_number)

    assert len(runner.gateway.openings()) == 3
    assert [code for event, code in runner.reports.reported() if event is Event.TURN_ENDED] == [
        RejectionCode.BUSY.value,
        RejectionCode.BUDGET_SPENT.value,
        RejectionCode.BUSY.value,
    ]


async def a_turn_ended_short(runner, code):
    """Runs the runner on one addressed line until its turn reported this code and began the wait before its retry."""
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.clock.until_waiting(1)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, code), (Event.PAUSED, code)]
    assert runner.app.said == []
    assert not runner.task.done()
    assert runner.app.log.count(LISTEN) == 1


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (Reply(status=HTTPStatus.TOO_MANY_REQUESTS, headers={"Retry-After": "3"}), RejectionCode.BUSY.value),
        (Reply(status=HTTPStatus.FORBIDDEN), FailureCode.PROVIDER_UNAVAILABLE.value),
        (Reply(status=HTTPStatus.SERVICE_UNAVAILABLE), FailureCode.PROVIDER_UNAVAILABLE.value),
    ],
    ids=["rate limited", "refused", "failing"],
)
async def test_a_gateway_refusal_is_silent_logs_one_code_and_leaves_the_lifecycle_untouched(runner, reply, code):
    runner.gateway = Gateway(reply)

    await a_turn_ended_short(runner, code)

    assert len(runner.gateway.requests) == 1


async def test_a_turn_refused_for_a_spent_budget_waits_the_backoff_base_before_his_next_turn_and_logs_the_pause_once(
    runner,
):
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, f"{ADDRESSED} First.")
    await runner.clock.until_waiting(1)
    for each in range(3):
        say_in_the_call(runner, f"{ADDRESSED} Later {each}.")
    await runner.app.until_window_read(PLACE_ID, count=2)
    held_requests = len(runner.gateway.requests)
    runner.clock.open_gate()
    await runner.app.said_count(1)

    assert held_requests == 1
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [1, 2, 3, 4]]
    assert [
        (change.event, change.code, change.seconds) for change in runner.reports.changes if change.event is Event.PAUSED
    ] == [(Event.PAUSED, RejectionCode.BUDGET_SPENT.value, BACKOFF_BASE_SECONDS)]


async def test_a_lines_turn_the_gateway_failed_runs_again_after_the_wait_with_its_seqs_in_its_woke(runner):
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.clock.until_waiting(1)
    held_requests = len(runner.gateway.requests)
    runner.clock.open_gate()
    await runner.app.said_count(1)

    assert held_requests == 1
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [1]]
    assert [opening.room["woke"] for opening in runner.gateway.openings()] == [
        woke_at_lines(url_of(runner, 0), PLACE_ID)
    ] * 2


async def test_the_wait_after_a_spent_budget_doubles_and_starts_over_after_a_turn_that_is_not_refused_for_it(runner):
    runner.gateway = Gateway(BUDGET_SPENT, BUDGET_SPENT, *says(PLACE_ID, ANSWER), BUDGET_SPENT)
    await restarted_in_the_call(runner)
    pauses = 0
    for round_number in range(1, 5):
        say_in_the_call(runner, f"{ADDRESSED} Round {round_number}.")
        if round_number == REFUSAL_FREE_ROUND:
            await runner.reports.until_a_turn_completed()
        else:
            pauses += 1
            await runner.reports.until(Event.PAUSED, count=pauses)

    assert runner.clock.slept == [BACKOFF_BASE_SECONDS, 2 * BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS]


async def test_a_key_the_gateway_cannot_use_ends_the_turn_before_any_gateway_call(runner):
    runner.files.key_path.write_text("a key with spaces\n", encoding="utf-8")

    await a_turn_ended_short(runner, FailureCode.PROVIDER_UNAVAILABLE.value)

    assert runner.gateway.requests == []


async def test_a_missing_api_group_is_refused_with_no_gateway_call(runner):
    runner.realm.groups = []

    await a_turn_ended_short(runner, RejectionCode.NOT_ENABLED.value)

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
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert runner.clock.slept == []


async def test_a_turn_binds_the_tools_the_app_lists_right_now_and_a_missing_one_is_no_failure(runner):
    async def lists_no_tools(reply):
        await reply.result(ListToolsResult(tools=[]))

    runner.gateway = Gateway(completion("done"), completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    runner.app.script(LIST_TOOLS, lists_no_tools)
    say_in_the_call(runner, f"{ADDRESSED} Once more.")
    await runner.reports.until_a_turn_completed(count=2)

    assert runner.gateway.offered_tools(1) == [SAY_TOOL]
    assert runner.gateway.offered_tools(2) == []
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def refuses_the_token(reply):
    await reply.status(HTTPStatus.UNAUTHORIZED)


async def answers_wrongly(reply):
    await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)


@pytest.mark.parametrize(
    ("method", "fails", "events"),
    [
        (DISCOVER, cuts, [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE), (Event.RESUMED, None)]),
        (LIST_TOOLS, cuts, [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE), (Event.RESUMED, None)]),
        (DISCOVER, refuses_the_token, [(Event.STARTED, None)]),
        (LIST_TOOLS, refuses_the_token, [(Event.STARTED, None)]),
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

    assert runner.reports.reported() == events
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
    await runner.reports.until(Event.RESUMED)
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.TURN_ENDED, RejectionCode.BUSY.value),
        (Event.PAUSED, RejectionCode.BUSY.value),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
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

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
    ]
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
        (cuts, [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE), (Event.RESUMED, None)], 1),
        (refuses_the_token, [(Event.STARTED, None)], 2),
        (
            answers_wrongly,
            [(Event.STARTED, None), (Event.SUSPENDED, jsonrpc_code(INVALID_PARAMS)), (Event.RESUMED, None)],
            2,
        ),
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

    assert runner.reports.reported() == events
    assert runner.gateway.round_sizes() == [2, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert len(runner.realm.issued) == issued


async def test_a_relayed_call_beside_a_late_answer_gives_one_reaction_and_no_turn_report(runner, monkeypatch):
    failing = transport_closing_with_late_answers(echo=anyio.BrokenResourceError, only_close=1)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    await relayed_say_fails(runner, cuts)
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
    ]
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
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert len(runner.gateway.requests) == 1


async def test_a_deadline_during_a_relayed_call_ends_the_turn_as_limit_exceeded_and_gives_no_reaction(runner):
    runner.files.persona_path.write_text(runner.files.persona_text(max_duration_seconds=1), encoding="utf-8")
    runner.config = load_config(runner.files.write())
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(CALL, hangs)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(2)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.TURN_ENDED, FailureCode.LIMIT_EXCEEDED.value),
        (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value),
    ]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [1, 2]]
    assert runner.gateway.openings()[1].room["history"]["entries"][1] == act_entry(
        url_of(runner, 0),
        "say",
        {"place": PLACE_ID, "text": ANSWER},
        unanswered(TURN_CANCELLED_CODE),
        seconds_ago=int(BACKOFF_BASE_SECONDS),
    )
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
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_refused_and_an_answered_turn_log_a_code_and_no_secret_or_content(runner, caplog):
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    logged = formatted(caplog.get_records("call"))
    assert Event.TURN_ENDED.value in logged
