"""The whole runner, a turn that ends with its place: when it is cancelled, what it reports and when its slot frees."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from agents.member import STANDING_URI, place_window_uri
from agents.member_outcomes import Unreachable, status_code
from agents.turn import StopOutcome

from personas.failures import leaves_of
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import CANCELLED_TURN, COMPLETED_TURN, TurnEnded
from tests.support.gateway import Gateway, says
from tests.support.hopin import CALL, DISCOVER, LIMIT_SECONDS, LIST_TOOLS, LISTEN, READ
from tests.support.openers import ClosingHold, opener_whose_close_adds_a_late_answer, opener_whose_request_raises
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, paused
from tests.support.voice import a_turn_whose_speak_is_in_flight
from tests.support.wire import hangs, hangs_once_entered, lost_on_demand

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

OTHER_PLACE_ID = "room-8"


async def a_turn_in_its_first_round(runner):
    """The runner restarted in the call; an addressed line started a turn whose first gateway round waits."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)


def without_lost_statements(runner):
    """What the runner reported but for the turn's end statement the session's close may cut, which logs it as lost."""
    return [report for report in runner.reports.reported() if report[0] is not Event.TURN_STATE_LOST]


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_a_turn_whose_place_left_the_standing_is_cancelled_with_no_further_round_or_call_and_reported_once(
    runner,
):
    await a_turn_in_its_first_round(runner)

    runner.app.stand(SUBJECT)

    await runner.reports.until_a_turn_ended()

    assert runner.reports.reported() == [(Event.STARTED, None)]

    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert runner.gateway.requests == []
    assert runner.app.said == []
    assert not runner.task.done()


async def test_a_turn_paused_mid_way_is_cancelled_and_after_the_resume_the_old_lines_are_not_answered(runner):
    await a_turn_in_its_first_round(runner)
    say_in_the_call(runner, f"{ADDRESSED} While the turn ran.")

    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.reports.until_a_turn_ended()
    say_in_the_call(runner, f"{ADDRESSED} During the pause.")
    resumed_read = runner.app.hold_next_window_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    resumed_read.set()
    await runner.app.said_count(1)

    await runner.reports.until_a_turn_ended()

    assert runner.reports.reported() == [(Event.STARTED, None)]

    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert [opening.woke_seqs for opening in runner.gateway.openings()] == [[4]]


async def test_while_a_cancelled_turn_still_unwinds_a_new_addressed_line_starts_no_second_turn_until_it_ended(
    runner,
):
    unwinding = ClosingHold()
    runner.open_session = unwinding.opener_holding_cancelled("declared_tools")
    runner.gateway = Gateway(*says(OTHER_PLACE_ID, "There."))
    runner.app.stand(SUBJECT, active(PLACE_ID), active(OTHER_PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_window_read(OTHER_PLACE_ID)
    say_in_the_call(runner, ADDRESSED)

    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await unwinding.closing.wait()
        await runner.app.until_turn_states(1)
        runner.app.stand(SUBJECT, active(OTHER_PLACE_ID))
        async with asyncio.timeout(LIMIT_SECONDS):
            await unwinding.cancelled.wait()
        window_reads = runner.app.window_reads(OTHER_PLACE_ID)
        runner.app.speak(OTHER_PLACE_ID, PERSON, "Ada", ADDRESSED)
        await runner.app.until_standing_read(runner.app.reads[STANDING_URI] + 1)
        later_read = runner.app.hold_next_read()
        runner.app.stand(SUBJECT, active(OTHER_PLACE_ID))
        await runner.app.until_read_held()

        assert runner.app.window_reads(OTHER_PLACE_ID) > window_reads
        assert runner.gateway.requests == []
        unwinding.release.set()
        later_read.set()
        await runner.app.said_count(1)
    finally:
        unwinding.release.set()

    assert runner.app.said == [(SUBJECT, OTHER_PLACE_ID, "There.")]
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert runner.reports.turn_endings == [CANCELLED_TURN]


async def test_a_runner_cancelled_during_a_turn_ends_cancelled_and_reports_nothing_for_the_turn(runner):
    await a_turn_in_its_first_round(runner)

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    assert without_lost_statements(runner) == [(Event.STARTED, None)]


async def test_a_turns_backend_is_closed_once_the_turn_was_cancelled(runner):
    await a_turn_in_its_first_round(runner)

    runner.app.stand(SUBJECT)
    await runner.reports.until_a_turn_ended()

    assert runner.backends[0].closed.is_set()


async def test_a_turns_backend_is_closed_once_the_runner_was_cancelled(runner):
    await a_turn_in_its_first_round(runner)

    runner.task.cancel()
    await ended_by(runner.task)

    assert runner.backends[0].closed.is_set()


SESSION_SEAMS = pytest.mark.parametrize(
    "seam", ["invoke", "declared_tools"], ids=["a relayed call", "the declarations"]
)


async def a_turn_whose_request_fails_as_its_place_leaves(runner, *, seam, error):
    """A turn whose app seam waits and raises ``error`` in place of the cancellation that reaches it; the place is
    then removed."""
    arrived = asyncio.Event()
    runner.open_session = opener_whose_request_raises(seam, error, arrived=arrived, when_cancelled=True)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    async with asyncio.timeout(LIMIT_SECONDS):
        await arrived.wait()
    await runner.app.until_turn_states(1)
    runner.app.stand(SUBJECT)


@SESSION_SEAMS
async def test_a_defect_in_a_turns_request_whose_place_then_leaves_ends_the_runner_loud_without_a_cancel(runner, seam):
    defect = ValueError("a defect")
    await a_turn_whose_request_fails_as_its_place_leaves(runner, seam=seam, error=defect)

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]


@SESSION_SEAMS
async def test_a_call_failure_in_a_turns_request_whose_place_then_leaves_ends_as_the_cancellation(runner, seam):
    await a_turn_whose_request_fails_as_its_place_leaves(runner, seam=seam, error=Unreachable("down"))
    await runner.reports.until_a_turn_ended()
    later_read = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()

    await runner.reports.until_a_turn_ended()

    assert runner.reports.reported() == [(Event.STARTED, None)]

    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert not runner.task.done()
    later_read.set()


def stopped_outcomes(runner) -> list[StopOutcome | None]:
    return [statement.stopped for _, statement in runner.app.turn_states]


async def a_turn_whose_call_is_in_flight(runner, said: asyncio.Event | None = None):
    """The runner restarted in the call; the turn's relayed ``say`` reached the app, which never answers."""
    runner.app.script(CALL, hangs)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    if said is not None:
        said.set()
    await runner.app.log.until(CALL, 1)


@dataclass(frozen=True)
class InFlight:
    """A request to the app that is open and how the turn it belongs to ends: a call the turn relays ends with it, a
    sentence the sender sends outlives it."""

    arrange: Callable[[asyncio.Event | None], Awaitable[None]]
    ending: TurnEnded


@pytest.fixture(params=["an own-app call", "a speak"])
def in_flight(request, runner):
    """Arranges a turn whose request to the app is in flight: a call it relays, or a sentence its sender sends."""
    if request.param == "a speak":

        async def sentence_sent(said=None):
            await a_turn_whose_speak_is_in_flight(runner, said)
            await runner.reports.until_a_turn_ended()

        return InFlight(sentence_sent, COMPLETED_TURN)
    return InFlight(lambda said=None: a_turn_whose_call_is_in_flight(runner, said), CANCELLED_TURN)


async def test_a_runner_stopped_while_a_request_of_a_turn_is_in_flight_ends_the_turn_as_it_ends_that_request(
    runner, in_flight
):
    await in_flight.arrange(None)

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    assert runner.reports.turn_endings == [in_flight.ending]
    assert StopOutcome.UNAVAILABLE not in stopped_outcomes(runner)


async def test_a_runner_stopped_while_a_turns_own_app_listing_is_in_flight_ends_the_turn_cancelled_with_its_place_due(
    runner,
):
    entered = asyncio.Event()
    runner.app.script(LIST_TOOLS, hangs_once_entered(entered), hangs_once_entered(entered))
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert StopOutcome.UNAVAILABLE not in stopped_outcomes(runner)
    kept = runner.histories.turns[runner.config.agents[0].client_id]
    assert [due.place for due in kept.due] == [CALL_PLACE]


async def test_a_lost_stream_while_a_turns_request_is_in_flight_ends_the_turn_as_that_request_does_and_backs_off_once(
    runner, in_flight
):
    said, lost = asyncio.Event(), asyncio.Event()
    runner.app.script(LISTEN, lost_on_demand(said, lost, place_window_uri(CALL_PLACE)))
    await in_flight.arrange(said)

    lost.set()
    await runner.reports.until(Event.BACKOFF)
    await runner.reports.until_a_turn_ended()

    assert runner.reports.turn_endings == [in_flight.ending]
    assert StopOutcome.UNAVAILABLE not in stopped_outcomes(runner)
    assert [event for event, _ in runner.reports.reported(started=False)].count(Event.BACKOFF) == 1


async def a_turn_opening_a_tools_only_session(runner, songmaker, *, only_close):
    """A turn whose ``server/discover`` at the tools-only app is held, the closes of its sessions leaving with late
    answers alone. The returned event releases the held request."""
    arrived, release = asyncio.Event(), asyncio.Event()

    async def held(reply):
        arrived.set()
        await release.wait()
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    runner.open_session = opener_whose_close_adds_a_late_answer(only_close=only_close)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    songmaker.script(DISCOVER, held)
    say_in_the_call(runner, ADDRESSED)
    async with asyncio.timeout(LIMIT_SECONDS):
        await arrived.wait()
    await runner.app.until_turn_states(1)
    return release


async def test_a_removal_that_reaches_a_turn_while_a_tools_only_session_opens_with_late_answers_alone_ends_it_cancelled(
    runner, songmaker
):
    release = await a_turn_opening_a_tools_only_session(runner, songmaker, only_close=1)
    try:
        runner.app.stand(SUBJECT)
        await runner.reports.until_a_turn_ended()
        later_read = runner.app.hold_next_read()
        runner.app.stand(SUBJECT, active(PLACE_ID))
        await runner.app.until_read_held()

        await runner.reports.until_a_turn_ended()

        assert runner.reports.reported() == [(Event.STARTED, None)]

        assert runner.reports.turn_endings == [CANCELLED_TURN]
        assert not runner.task.done()
        later_read.set()
    finally:
        release.set()


async def test_a_runner_cancelled_while_a_turns_tools_only_session_opens_with_late_answers_alone_ends_cancelled(
    runner, songmaker
):
    release = await a_turn_opening_a_tools_only_session(runner, songmaker, only_close=2)
    try:
        runner.task.cancel()

        assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    finally:
        release.set()


async def test_a_lifecycle_refusal_that_reaches_a_turn_while_a_tools_only_session_opens_backs_off_once(
    runner, songmaker
):
    async def refused(reply):
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    release = await a_turn_opening_a_tools_only_session(runner, songmaker, only_close=2)
    try:
        runner.app.script(READ, refused)
        runner.app.stand(SUBJECT, active(PLACE_ID))
        await runner.reports.until(Event.RESUMED)

        assert without_lost_statements(runner) == [
            (Event.STARTED, None),
            (Event.BACKOFF, status_code(503)),
            (Event.RESUMED, None),
        ]
        assert not runner.task.done()
    finally:
        release.set()
