"""The whole runner, the turn: which lines start one, how it ends and what its failures lead to."""

from __future__ import annotations

import asyncio
from dataclasses import replace
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
from tests.support.fakes import SweepClock, failed_turn, refused_turn
from tests.support.gateway import BUDGET_SPENT, Gateway, says, says_after_a_hold
from tests.support.history import act_entry, after_start, unanswered, url_of, woke_at_lines
from tests.support.hopin import CALL, DISCOVER, LIMIT_SECONDS, LIST_TOOLS, LISTEN, READ, SAY_TOOL
from tests.support.logs import lines_of
from tests.support.realm import LIFETIME_SECONDS
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, invited
from tests.support.transport import transport_closing_with_late_answers
from tests.support.wire import cuts, hangs

pytestmark = pytest.mark.anyio


async def test_an_addressed_line_gets_exactly_one_say_under_the_agents_own_fingerprint(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)

    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert len(runner.gateway.requests) == 2
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1]]
    assert runner.gateway.openings()[0].system == runner.files.persona_prompt
    assert runner.gateway.users() == {person_fingerprint(runner.app_principal)}
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_his_own_line_naming_him_starts_no_turn(runner):
    """TURNS-5."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, "Mozart will say it himself", author=SUBJECT, name="Mozart")
    read = runner.app.hold_next_window_read()
    runner.start()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    read.set()
    await runner.app.said_count(1)

    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[2]]


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

    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[2]]


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

    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1]]


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

    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [1]]


async def test_a_duplicate_pulse_or_a_re_read_starts_no_second_turn_for_a_line(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Another answer."))
    closing_round = runner.gateway.hold_request(2)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(2)
    for uri in (STANDING_URI, place_window_uri(CALL_PLACE), STANDING_URI, place_window_uri(CALL_PLACE)):
        await runner.app.changes.changed(runner.app_principal, uri)
    read = runner.app.hold_next_window_read()
    closing_round.set()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    read.set()
    await runner.app.said_count(2)

    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [3]]


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

    assert runner.gateway.round_sizes() == [3, 3, 5]
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [1, 2]]
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported()[:2] == [(Event.STARTED, None), (Event.JOINED, None)]
    assert runner.reports.turn_endings == [refused_turn(RejectionCode.BUSY)]


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

    assert runner.gateway.round_sizes() == [3, 5, 7, 4, 6]
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [2]]


async def test_every_line_addressed_to_him_gets_its_own_turn_one_after_the_other(runner):
    runner.gateway = Gateway(
        Reply(status=HTTPStatus.TOO_MANY_REQUESTS), BUDGET_SPENT, Reply(status=HTTPStatus.TOO_MANY_REQUESTS)
    )
    await restarted_in_the_call(runner)
    for round_number in range(1, 4):
        say_in_the_call(runner, f"{ADDRESSED} Round {round_number}.")
        await runner.reports.until_a_turn_ended(count=round_number)

    assert len(runner.gateway.openings()) == 3
    assert runner.reports.turn_endings == [
        refused_turn(RejectionCode.BUSY),
        refused_turn(RejectionCode.BUDGET_SPENT),
        refused_turn(RejectionCode.BUSY),
    ]


async def test_two_turns_in_a_row_open_no_session_besides_the_loops(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER), *says(PLACE_ID, "Again."))
    await restarted_in_the_call(runner)
    for count in (1, 2):
        say_in_the_call(runner, f"{ADDRESSED} Round {count}.")
        await runner.app.said_count(count)
        await runner.reports.until_a_turn_ended(count=count)

    assert runner.app.log.count(DISCOVER) == 1
    assert runner.app.log.count(LISTEN) == 1


async def test_a_turn_lists_the_tools_of_its_app_with_the_token_the_loop_holds_when_it_lists(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)

    async def passes_renewal_due(reply):
        runner.clock.time += LIFETIME_SECONDS
        await reply.result(ListToolsResult(tools=[]))

    runner.app.script(LIST_TOOLS, passes_renewal_due)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    *earlier, last = runner.app.list_bearers
    assert last not in earlier
    assert last == runner.realm.issued[-1]


async def a_turn_ended_short(runner, code):
    """Runs the runner on one addressed line until its turn reported this code and began the wait before its retry."""
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.clock.until_waiting(1)

    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.PAUSED, code)]
    assert [each.reason for each in runner.reports.turn_endings] == [code]
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


async def test_a_turns_backend_is_closed_once_the_turn_answered(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    async with asyncio.timeout(LIMIT_SECONDS):
        await runner.backends[0].closed.wait()


async def test_a_turns_backend_is_closed_once_the_gateway_failed_the_turn(runner):
    runner.gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE))

    await a_turn_ended_short(runner, FailureCode.PROVIDER_UNAVAILABLE.value)

    assert runner.backends[0].closed.is_set()


async def test_a_turn_refused_for_a_spent_budget_waits_the_backoff_base_before_his_next_turn_and_logs_the_pause_once(
    runner,
):
    """TURNS-37, LOG-15."""
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
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [1, 2, 3, 4]]
    assert [
        (change.event, change.code, change.seconds) for change in runner.reports.changes if change.event is Event.PAUSED
    ] == [(Event.PAUSED, RejectionCode.BUDGET_SPENT.value, BACKOFF_BASE_SECONDS)]


async def test_a_lines_turn_the_gateway_failed_runs_again_after_the_wait_with_its_seqs_in_its_woke(runner):
    """TURNS-38."""
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.clock.until_waiting(1)
    held_requests = len(runner.gateway.requests)
    runner.clock.open_gate()
    await runner.app.said_count(1)

    assert held_requests == 1
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [1]]
    assert [opening.room["woke"] for opening in runner.gateway.openings()] == [
        woke_at_lines(url_of(runner, 0), PLACE_ID, [1])
    ] * 2


async def test_the_wait_after_a_spent_budget_doubles_and_starts_over_after_a_turn_that_is_not_refused_for_it(runner):
    runner.gateway = Gateway(BUDGET_SPENT, BUDGET_SPENT, *says(PLACE_ID, ANSWER), BUDGET_SPENT)
    await restarted_in_the_call(runner)

    say_in_the_call(runner, f"{ADDRESSED} First.")
    await runner.app.said_count(1)
    say_in_the_call(runner, f"{ADDRESSED} Second.")
    await runner.reports.until(Event.PAUSED, count=3)

    assert [change.seconds for change in runner.reports.changes if change.event is Event.PAUSED] == [
        BACKOFF_BASE_SECONDS,
        2 * BACKOFF_BASE_SECONDS,
        BACKOFF_BASE_SECONDS,
    ]


async def test_a_pause_outlives_the_leaving_of_its_turns_place_and_holds_the_turn_of_another_place_until_it_is_over(
    runner,
):
    other_place = "room-8"
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, *says(other_place, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.clock.until_waiting(1)
    runner.app.stand(SUBJECT, active(other_place))
    await runner.app.until_window_read(other_place)
    runner.app.speak(other_place, PERSON, "Ada", ADDRESSED)
    await runner.app.until_window_read(other_place, count=2)
    for later_read in range(3, 5):
        runner.app.speak(other_place, PERSON, "Ada", "Hm.")
        await runner.app.until_window_read(other_place, count=later_read)
    assert len(runner.gateway.requests) == 1
    runner.clock.open_gate()
    await runner.app.said_count(1)

    assert [place for _, place, _ in runner.app.said] == [other_place]


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
    """TURNS-18."""

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


async def lists_nothing(reply):
    await reply.result(ListToolsResult(tools=[]))


@pytest.mark.parametrize(
    ("fails", "events"),
    [
        (cuts, [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE), (Event.RESUMED, None)]),
        (refuses_the_token, [(Event.STARTED, None)]),
    ],
    ids=["tool listing cut", "tool listing refuses the token"],
)
async def test_a_turn_whose_listing_fails_before_the_model_is_asked_leads_to_one_lifecycle_reaction_and_no_second_turn(
    runner, fails, events
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    # The loop's session lists the tools first, to check the answer of the turn's thinking.
    runner.app.script(LIST_TOOLS, lists_nothing, fails)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert runner.reports.reported() == events
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[2]]
    assert len(runner.realm.issued) == (2 if fails is refuses_the_token else 1)


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
    assert runner.gateway.round_sizes() == [3, 3, 5]
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert len(runner.realm.issued) == issued


async def test_a_relayed_call_beside_a_late_answer_gives_one_reaction_and_no_pause(runner, monkeypatch):
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
    await runner.reports.until_a_turn_ended()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(2)

    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value),
    ]
    assert runner.reports.turn_endings == [failed_turn(FailureCode.LIMIT_EXCEEDED)]
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[1], [1, 2]]
    assert runner.gateway.openings()[1].room["history"]["entries"][1] == act_entry(
        url_of(runner, 0),
        "say",
        {"place": PLACE_ID, "text": ANSWER},
        unanswered(TURN_CANCELLED_CODE),
        at=after_start(0),
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
    """LOG-3."""
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert "refused" in [line["outcome"] for line in lines_of(caplog.get_records("call"), "turn.ended")]


async def test_the_started_line_names_the_model_each_agent_runs_on(runner, caplog):
    """LOG-6."""
    runner.config = replace(
        runner.config, agents=(replace(runner.config.agents[0], persona=replace_model(runner, "text/sonnet-low")),)
    )

    await restarted_in_the_call(runner)

    started = lines_of(caplog.get_records("call"), "loop.started")
    assert started
    assert all(line["model"] == "text/sonnet-low" for line in started)


def replace_model(runner, model):
    return runner.config.agents[0].persona.model_copy(update={"model": model})
