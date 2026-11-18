"""The whole runner, what a turn states at its place: thinking while it runs, and what its end calls for."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import Reply
from agents.member import (
    SET_TURN_STATE_TOOL,
    SetTurnStateArguments,
    SetTurnStateRefusal,
    SetTurnStateRefused,
)
from agents.tools import ToolOutcome
from agents.turn import StopOutcome

from personas.decision import NETWORK_FAILURE_CODE, status_code
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.hopin import CALL, LIMIT_SECONDS, SAY_TOOL
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, paused
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
        (fails_every_statement, [(Event.TURN_STATE_FAILED, status_code(HTTPStatus.SERVICE_UNAVAILABLE))] * 2),
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
