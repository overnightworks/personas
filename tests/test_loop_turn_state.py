"""The whole runner, what a turn states at its place: thinking while it runs, and what its end calls for; and what
his seats are told once the loop stops taking turns in the app, recovers or stops."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import Reply
from agents.member import (
    SET_TURN_STATE_TOOL,
    PlaceRef,
    SetTurnStateArguments,
    SetTurnStateRefusal,
    SetTurnStateRefused,
)
from agents.tools import ToolOutcome
from agents.turn import StopOutcome
from mcp_types import INVALID_PARAMS

from personas.decision import INVALID_PARAMS_REASON, NETWORK_FAILURE_CODE, NO_ANSWER_REASON
from personas.failures import leaves_of
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.hopin import CALL, LIMIT_SECONDS, READ, SAY_TOOL
from tests.support.logs import lines_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, paused
from tests.support.transport import opener_whose_request_raises
from tests.support.voice import ended_by
from tests.support.wire import hangs

if TYPE_CHECKING:
    from tests.support.wire import Reply as AppReply

pytestmark = pytest.mark.anyio

SHORT_CALL_SECONDS = 0.3
THINKING = SetTurnStateArguments(place=CALL_PLACE, thinking=True)


def ended(stopped: StopOutcome | None) -> SetTurnStateArguments:
    return SetTurnStateArguments(place=CALL_PLACE, stopped=stopped)


def statements(runner) -> list[SetTurnStateArguments]:
    return [statement for subject, statement in runner.app.turn_states if subject == SUBJECT]


async def a_turn_in_its_first_round(runner, *replies: Reply):
    """The runner restarted in the call; an addressed line started a turn whose first gateway round waits until the
    returned event is set."""
    runner.gateway = Gateway(*replies)
    first_round = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    return first_round


async def test_a_turn_states_thinking_at_its_place_before_its_first_round_and_its_model_never_sees_the_tool(runner):
    """TURNS-43, TURNS-18."""
    first_round = await a_turn_in_its_first_round(runner, *says(PLACE_ID, ANSWER))

    assert statements(runner) == [THINKING]
    first_round.set()
    await runner.app.said_count(1)
    assert runner.gateway.offered_tools(1) == [SAY_TOOL]


@pytest.mark.parametrize(
    ("replies", "stopped"),
    [
        (says(PLACE_ID, ANSWER), None),
        ([BUDGET_SPENT], StopOutcome.BUDGET_SPENT),
        ([Reply(status=HTTPStatus.TOO_MANY_REQUESTS)], StopOutcome.BUSY),
        ([Reply(status=HTTPStatus.INTERNAL_SERVER_ERROR)], StopOutcome.UNAVAILABLE),
    ],
    ids=["answered", "budget spent", "busy", "failed"],
)
async def test_a_turn_ends_its_thinking_with_none_after_an_answer_or_its_stops_outcome(runner, replies, stopped):
    """TURNS-43."""
    runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()
    await runner.app.until_turn_states(2)

    assert statements(runner)[:2] == [THINKING, ended(stopped)]


@pytest.mark.parametrize(
    ("replies", "turns", "earlier", "restated"),
    [
        (says(PLACE_ID, ANSWER), 1, [], None),
        (
            [Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER)],
            2,
            [THINKING, ended(StopOutcome.BUSY)],
            StopOutcome.BUSY,
        ),
    ],
    ids=["no earlier end", "an earlier busy end"],
)
async def test_a_cancelled_turn_ends_its_thinking_with_the_last_turn_end_at_its_seat(
    runner, replies, turns, earlier, restated
):
    """TURNS-43."""
    runner.gateway = Gateway(*replies)
    runner.gateway.hold_request(turns)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(turns)

    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.reports.until_a_turn_ended(count=turns)
    await runner.app.until_turn_states(len(earlier) + 2)

    assert statements(runner) == [*earlier, THINKING, ended(restated)]


async def test_a_turn_cancelled_because_his_seat_ended_states_no_end(runner):
    """TURNS-43."""
    await a_turn_in_its_first_round(runner, *says(PLACE_ID, ANSWER))

    runner.app.stand(SUBJECT)
    await runner.reports.until_a_turn_ended()
    # Statements go out in the order queued, so a next turn's thinking arrives after any end the first one queued.
    seated_again_read = runner.app.hold_next_window_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    seated_again_read.set()
    await runner.app.until_turn_states(2)

    assert statements(runner) == [THINKING, THINKING]
    assert runner.reports.reported(started=False) == []


async def test_a_turn_cancelled_while_its_thinking_waits_still_ends_its_thinking(runner):
    """TURNS-43."""
    entered, released = asyncio.Event(), asyncio.Event()

    async def fails_once_released(reply: AppReply) -> None:
        entered.set()
        await released.wait()
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.script(SET_TURN_STATE_TOOL, fails_once_released)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()

    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.reports.until_a_turn_ended()
    released.set()
    await runner.app.until_turn_states(1)

    assert statements(runner) == [ended(None)]
    assert runner.gateway.arrivals == 0


def refuses_the_thinking(runner) -> None:
    refusal = SetTurnStateRefused(refused=SetTurnStateRefusal.NOT_HELD)
    runner.app.turn_state_outcomes.append(ToolOutcome(refusal.model_dump_json(), is_error=True))


def fails_every_statement(runner) -> None:
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = CALL
    runner.app.refused_tool = SET_TURN_STATE_TOOL


@pytest.mark.parametrize(
    ("breaks", "reported"),
    [
        (refuses_the_thinking, [(Event.TURN_STATE_FAILED, SetTurnStateRefusal.NOT_HELD.value)]),
        (fails_every_statement, [(Event.TURN_STATE_FAILED, NO_ANSWER_REASON)] * 2),
    ],
    ids=["refused", "failed"],
)
async def test_a_statement_the_app_refuses_or_fails_is_logged_and_the_turn_goes_on(runner, breaks, reported):
    """TURNS-43, LOG-34."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    breaks(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.reports.until_a_turn_completed()
    await runner.reports.until(Event.TURN_STATE_FAILED, count=len(reported))

    assert runner.reports.reported(started=False) == reported


UNAVAILABLE = ended(StopOutcome.UNAVAILABLE)
EARLIER_ENDS = pytest.mark.parametrize(
    ("replies", "earlier", "last"),
    [
        ([], [], None),
        ([Reply(status=HTTPStatus.TOO_MANY_REQUESTS)], [THINKING, ended(StopOutcome.BUSY)], StopOutcome.BUSY),
    ],
    ids=["no turn end", "a busy turn end"],
)


async def in_the_call_after(runner, replies) -> None:
    """The runner restarted in the call, and the turn each reply answers has ended and stated its end."""
    if replies:
        runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)
    for turn in range(1, len(replies) + 1):
        say_in_the_call(runner, f"{ADDRESSED} {turn}")
        await runner.reports.until_a_turn_ended(count=turn)
    await runner.app.until_turn_states(2 * len(replies))


def fails_the_next_read(runner) -> None:
    async def fails(reply: AppReply) -> None:
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    runner.app.script(READ, fails)


async def a_turn_after_it(runner) -> None:
    """A turn that answers runs and states its thinking and its end, after whatever the loop queued before it."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()


def fails_the_window_reads(runner) -> None:
    runner.app.failing_windows.add(PLACE_ID)


def answers_the_window_reads(runner) -> None:
    runner.app.failing_windows.clear()


def forbids_every_read(runner) -> None:
    runner.app.refuse_with = HTTPStatus.FORBIDDEN
    runner.app.refused_method = READ


def answers_every_read(runner) -> None:
    runner.app.refuse_with = None


@EARLIER_ENDS
@pytest.mark.parametrize(
    ("breaks", "recovers"),
    [(fails_the_window_reads, answers_the_window_reads), (forbids_every_read, answers_every_read)],
    ids=["a backoff", "a suspension"],
)
async def test_a_loop_that_stops_taking_turns_states_unavailable_and_the_next_full_pass_his_last_end(
    runner, replies, earlier, last, breaks, recovers
):
    """TURNS-44."""
    await in_the_call_after(runner, replies)

    breaks(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_turn_states(len(earlier) + 1)
    recovers(runner)
    await runner.app.until_turn_states(len(earlier) + 2)

    assert statements(runner) == [*earlier, UNAVAILABLE, ended(last)]


async def test_unavailable_goes_to_no_further_seat_once_the_app_does_not_answer_one(runner):
    """TURNS-44."""
    other_place = PlaceRef(kind="call", id="room-8")
    held = (active(PLACE_ID), active(other_place.id))

    async def fails(reply: AppReply) -> None:
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    runner.app.stand(SUBJECT, *held)
    runner.start()
    await runner.app.until_window_read(other_place.id)
    runner.app.script(SET_TURN_STATE_TOOL, fails)
    forbids_every_read(runner)
    runner.app.stand(SUBJECT, *held)
    await runner.reports.until(Event.TURN_STATE_FAILED)
    answers_every_read(runner)
    await runner.app.until_turn_states(2)

    assert statements(runner) == [
        SetTurnStateArguments(place=CALL_PLACE, stopped=None),
        SetTurnStateArguments(place=other_place, stopped=None),
    ]


async def test_a_defect_while_stating_unavailable_ends_the_runner_loud(runner):
    defect = ValueError("a defect")
    runner.open_session = opener_whose_request_raises("set_turn_state", defect)
    await restarted_in_the_call(runner)

    forbids_every_read(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    ended_as = await ended_by(runner.task)

    assert [leaf for leaf in leaves_of(ended_as) if leaf is defect] == [defect]


@EARLIER_ENDS
async def test_a_blip_that_recovers_on_its_first_retry_states_nothing(runner, replies, earlier, last):
    """TURNS-44."""
    await in_the_call_after(runner, replies)
    reads = runner.app.window_reads(PLACE_ID)

    fails_the_next_read(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.reports.until(Event.RESUMED)
    await runner.app.until_window_read(PLACE_ID, reads + 1)
    await a_turn_after_it(runner)
    await runner.app.until_turn_states(len(earlier) + 2)

    assert statements(runner) == [*earlier, THINKING, ended(None)]


async def test_a_blip_while_a_turns_end_is_answered_backs_off_told_by_the_reads_answer_and_states_no_unavailable(
    runner, caplog
):
    """TURNS-44, LOG-9: the answer to the statement beside the failed read does not tell how the read failed."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    await runner.app.until_turn_states(2)

    fails_the_next_read(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.reports.until(Event.RESUMED)
    await a_turn_after_it(runner)
    await runner.app.until_turn_states(4)

    reported = {event for event, _ in runner.reports.reported(started=False)}
    assert (Event.BACKOFF in reported, Event.SUSPENDED in reported) == (True, False)
    (backed_off,) = lines_of(caplog.records, "loop.backed_off")
    assert (backed_off.get("status"), backed_off.get("mcp_method")) == (HTTPStatus.SERVICE_UNAVAILABLE, READ)
    assert UNAVAILABLE not in statements(runner)


async def test_a_statement_lost_with_its_session_is_stated_again_after_the_next_full_pass(runner):
    """TURNS-44."""
    entered, released = asyncio.Event(), asyncio.Event()

    async def held_until_released(reply: AppReply) -> None:
        entered.set()
        await released.wait()

    first_round = await a_turn_in_its_first_round(runner, *says(PLACE_ID, ANSWER))
    await runner.app.until_turn_states(1)
    runner.app.script(SET_TURN_STATE_TOOL, held_until_released)
    first_round.set()
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()

    fails_the_next_read(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_turn_states(2)
    released.set()

    assert statements(runner) == [THINKING, ended(None)]


@EARLIER_ENDS
async def test_the_runners_stop_ends_the_thinking_of_the_turn_it_cuts_with_his_last_end(runner, replies, earlier, last):
    """TURNS-43."""
    await in_the_call_after(runner, replies)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.gateway.hold_request(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    runner.task.cancel()
    await ended_by(runner.task)

    assert statements(runner) == [*earlier, THINKING, ended(last)]


async def test_a_stops_statement_that_does_not_go_out_within_its_margin_is_logged_and_the_agent_still_ends(runner):
    """TURNS-43, LOG-34."""
    released = asyncio.Event()

    async def held_until_released(reply: AppReply) -> None:
        await released.wait()

    runner.config = replace(runner.config, ownership=replace(runner.config.ownership, cancel_margin_seconds=0.05))
    await a_turn_in_its_first_round(runner, *says(PLACE_ID, ANSWER))
    await runner.app.until_turn_states(1)
    runner.app.script(SET_TURN_STATE_TOOL, held_until_released)

    runner.task.cancel()
    try:
        ended_as = await ended_by(runner.task)
    finally:
        released.set()

    assert isinstance(ended_as, asyncio.CancelledError)
    assert runner.reports.reported(started=False).count((Event.TURN_STATE_FAILED, NETWORK_FAILURE_CODE)) == 1


@pytest.mark.parametrize("status", [HTTPStatus.BAD_REQUEST, HTTPStatus.OK], ids=["error status", "success status"])
async def test_a_seat_whose_app_answers_an_unknown_tool_gets_no_more_statements_after_one_line(runner, status):
    """TURNS-43, LOG-34."""

    async def unknown_tool(reply: AppReply) -> None:
        await reply.error(INVALID_PARAMS, status=status)

    runner.app.script(SET_TURN_STATE_TOOL, unknown_tool)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    for turn in (1, 2):
        say_in_the_call(runner, f"{ADDRESSED} {turn}")
        await runner.reports.until_a_turn_completed(count=turn)

    assert runner.reports.reported(started=False).count((Event.TURN_STATE_FAILED, INVALID_PARAMS_REASON)) == 1
    assert (runner.app.log.count(SET_TURN_STATE_TOOL), statements(runner)) == (1, [])


async def test_a_statement_the_app_does_not_answer_in_time_is_logged_as_failed_and_the_turns_round_starts(runner):
    """TURNS-43, LOG-34."""
    runner.bound_listening_calls_to(SHORT_CALL_SECONDS)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(SET_TURN_STATE_TOOL, hangs)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.reports.until(Event.TURN_STATE_FAILED)

    assert runner.reports.reported(started=False)[0] == (Event.TURN_STATE_FAILED, NETWORK_FAILURE_CODE)
    assert runner.gateway.arrivals == 1


async def test_a_backoff_whose_window_reads_keep_failing_logs_resumed_once_a_full_pass_went_through(runner):
    """LOG-8, LOG-9."""
    await restarted_in_the_call(runner)
    fails_the_window_reads(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_turn_states(1)
    answers_the_window_reads(runner)
    await runner.app.until_turn_states(2)

    assert [event for event, _ in runner.reports.reported(started=False)] == [Event.BACKOFF, Event.RESUMED]
