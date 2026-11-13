"""The whole runner, a turn that ends with its place: when it is cancelled, what it reports and when its slot frees."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus

import anyio
import pytest
from agents.backend.fake_gateway import Reply
from agents.member import STANDING_URI
from mcp_types import ListToolsResult

from personas import app_session
from personas.decision import TOOL_NOT_DECLARED_CODE, TURN_CANCELLED_CODE, status_code
from personas.failures import Unreachable, leaves_of
from personas.reporting import Event
from personas.task_record import UnrequestedCancellationError
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import DISCOVER, LIMIT_SECONDS, LIST_TOOLS, READ, active, paused
from tests.support.transport import (
    ClosingHold,
    transport_closing_with_late_answers,
    transport_whose_close_loses_a_child,
    transport_whose_session_loses_a_child,
)

pytestmark = pytest.mark.anyio

OTHER_PLACE_ID = "room-8"
LATE_ANSWERS = pytest.mark.parametrize(
    "echo", [anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"]
)


async def a_turn_in_its_first_round(runner):
    """The runner restarted in the call; an addressed line started a turn whose first gateway round waits."""
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_a_turn_whose_place_left_the_standing_is_cancelled_with_no_further_round_or_call_and_reported_once(
    runner,
):
    await a_turn_in_its_first_round(runner)

    runner.app.stand(SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert runner.gateway.requests == []
    assert runner.app.said == []
    assert not runner.task.done()


async def test_a_turn_paused_mid_way_is_cancelled_and_after_the_resume_the_old_lines_are_not_answered(runner):
    await a_turn_in_its_first_round(runner)
    say_in_the_call(runner, f"{ADDRESSED} While the turn ran.")

    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.reports.until(Event.TURN_ENDED)
    say_in_the_call(runner, f"{ADDRESSED} During the pause.")
    resumed_read = runner.app.hold_next_window_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    resumed_read.set()
    await runner.app.said_count(1)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[4]]


@LATE_ANSWERS
async def test_a_cancelled_turn_whose_session_leaves_with_late_answers_alone_gives_no_lifecycle_reaction(
    runner, monkeypatch, echo
):
    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_closing_with_late_answers(echo=echo, only_close=1)
    )
    await a_turn_in_its_first_round(runner)

    runner.app.stand(SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)
    later_read = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert runner.app.log.count("subscriptions/listen") == 1
    assert not runner.task.done()
    later_read.set()


async def test_while_a_cancelled_turn_still_unwinds_a_new_addressed_line_starts_no_second_turn_until_it_ended(
    runner, monkeypatch
):
    unwinding = ClosingHold()
    monkeypatch.setattr(app_session, "streamable_http_client", unwinding.transport())
    runner.gateway = Gateway(*says(OTHER_PLACE_ID, "There."))
    runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID), active(OTHER_PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_window_read(OTHER_PLACE_ID)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    try:
        runner.app.stand(SUBJECT, active(OTHER_PLACE_ID))
        async with asyncio.timeout(LIMIT_SECONDS):
            await unwinding.closing.wait()
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
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]


async def test_a_turn_that_reported_its_end_and_meets_the_cancel_in_its_teardown_reports_no_cancel(runner, monkeypatch):
    async def lists_no_tools(reply):
        await reply.result(ListToolsResult(tools=[]))

    closing = ClosingHold()
    monkeypatch.setattr(app_session, "streamable_http_client", closing.transport())
    runner.app.script(LIST_TOOLS, lists_no_tools)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.closing.wait()
        removal_read = runner.app.hold_next_read()
        runner.app.stand(SUBJECT)
        await runner.app.until_read_held()
        removal_read.set()
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
        await runner.app.until_standing_read(runner.app.reads[STANDING_URI] + 1)
    finally:
        closing.release.set()

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TOOL_NOT_DECLARED_CODE)]
    assert not runner.task.done()


async def test_a_runner_cancelled_during_a_turn_ends_cancelled_and_reports_nothing_for_the_turn(runner):
    await a_turn_in_its_first_round(runner)

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    assert runner.reports.reported() == [(Event.STARTED, None)]


@LATE_ANSWERS
async def test_a_runner_cancelled_while_its_lifecycle_session_unwinds_with_late_answers_alone_ends_cancelled(
    runner, monkeypatch, echo
):
    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_closing_with_late_answers(echo=echo, only_close=1)
    )
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read()

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)


SESSION_SEAMS = pytest.mark.parametrize(
    "seam", ["invoke", "declared_tools"], ids=["a relayed call", "the declarations"]
)


async def a_turn_removed_while_its_session_closes(runner, monkeypatch, *, seam=None, error=None, gateway=None):
    """A turn whose app seam raises ``error`` or whose gateway answers as scripted, its session's close held.

    The place is then removed.
    """

    async def raise_it(self, *arguments):
        raise error

    closing = ClosingHold()
    monkeypatch.setattr(app_session, "streamable_http_client", closing.transport())
    if seam is not None:
        monkeypatch.setattr(f"personas.app_session._McpAppSession.{seam}", raise_it)
    runner.gateway = gateway or Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.closing.wait()
        runner.app.stand(SUBJECT)
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
    finally:
        closing.release.set()


@SESSION_SEAMS
async def test_a_defect_in_a_turns_session_body_whose_place_then_leaves_ends_the_runner_loud_without_a_cancel(
    runner, monkeypatch, seam
):
    defect = ValueError("a defect")
    await a_turn_removed_while_its_session_closes(runner, monkeypatch, seam=seam, error=defect)

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]


@SESSION_SEAMS
async def test_a_call_failure_in_a_turns_session_body_whose_place_then_leaves_ends_as_the_cancellation(
    runner, monkeypatch, seam
):
    await a_turn_removed_while_its_session_closes(runner, monkeypatch, seam=seam, error=Unreachable("down"))
    await runner.reports.until(Event.TURN_ENDED)
    later_read = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert not runner.task.done()
    later_read.set()


async def test_a_gateway_stop_in_a_turns_session_body_whose_place_then_leaves_ends_as_the_cancellation(
    runner, monkeypatch
):
    gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE))
    await a_turn_removed_while_its_session_closes(runner, monkeypatch, gateway=gateway)
    await runner.reports.until(Event.TURN_ENDED)
    later_read = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert not runner.task.done()
    later_read.set()


@asynccontextmanager
async def a_turn_opening_its_session(runner, monkeypatch, *, only_close, echo):
    """A turn whose ``server/discover`` is held, the closes of its session leaving with late answers alone."""
    arrived, release = asyncio.Event(), asyncio.Event()

    async def held(reply):
        arrived.set()
        await release.wait()
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_closing_with_late_answers(echo=echo, only_close=only_close)
    )
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.script(DISCOVER, held)
    say_in_the_call(runner, ADDRESSED)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await arrived.wait()
        yield
    finally:
        release.set()


@LATE_ANSWERS
async def test_a_removal_that_reaches_a_turn_while_its_session_opens_with_late_answers_alone_ends_the_turn_cancelled(
    runner, monkeypatch, echo
):
    async with a_turn_opening_its_session(runner, monkeypatch, only_close=1, echo=echo):
        runner.app.stand(SUBJECT)
        await runner.reports.until(Event.TURN_ENDED)
        later_read = runner.app.hold_next_read()
        runner.app.stand(SUBJECT, active(PLACE_ID))
        await runner.app.until_read_held()

        assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
        assert not runner.task.done()
        later_read.set()


@LATE_ANSWERS
async def test_a_runner_cancelled_while_a_turn_session_opens_with_late_answers_alone_ends_cancelled(
    runner, monkeypatch, echo
):
    async with a_turn_opening_its_session(runner, monkeypatch, only_close=2, echo=echo):
        runner.task.cancel()

        assert isinstance(await ended_by(runner.task), asyncio.CancelledError)


@LATE_ANSWERS
async def test_a_lifecycle_refusal_that_reaches_a_turn_while_its_session_opens_with_late_answers_alone_backs_off_once(
    runner, monkeypatch, echo
):
    async def refused(reply):
        await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)

    async with a_turn_opening_its_session(runner, monkeypatch, only_close=2, echo=echo):
        runner.app.script(READ, refused)
        runner.app.stand(SUBJECT, active(PLACE_ID))
        await runner.reports.until(Event.RESUMED)

        assert runner.reports.reported() == [
            (Event.STARTED, None),
            (Event.BACKOFF, status_code(503)),
            (Event.RESUMED, None),
        ]
        assert not runner.task.done()


async def test_a_turn_whose_session_anyio_cancelled_with_late_answers_alone_while_nobody_cancelled_it_ends_loud(
    runner, monkeypatch
):
    failing = asyncio.Event()
    monkeypatch.setattr(app_session, "streamable_http_client", transport_whose_session_loses_a_child(failing))
    await a_turn_in_its_first_round(runner)

    failing.set()
    ended = await ended_by(runner.task)

    (leaf,) = leaves_of(ended)
    assert isinstance(leaf, UnrequestedCancellationError)
    assert isinstance(leaf.__cause__, asyncio.CancelledError)


async def test_a_defect_in_a_turns_session_body_whose_close_loses_a_child_ends_the_runner_loud(runner, monkeypatch):
    defect = ValueError("a defect")

    async def raise_it(self, *arguments):
        raise defect

    monkeypatch.setattr("personas.app_session._McpAppSession.invoke", raise_it)
    lost = transport_whose_close_loses_a_child(OSError, nth=2, of_url=runner.app_url)
    monkeypatch.setattr(app_session, "streamable_http_client", lost)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    ended = await ended_by(runner.task)

    assert sorted(type(leaf).__name__ for leaf in leaves_of(ended)) == ["Unreachable", "ValueError"]
    assert [leaf for leaf in leaves_of(ended) if leaf is defect] == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]
