"""The whole runner, a turn that ends with its place: when it is cancelled, what it reports and when its slot frees."""

from __future__ import annotations

import asyncio

import anyio
import pytest
from agent_runtime_client.member import STANDING_URI
from mcp_types import ListToolsResult

from agents import app_session
from agents.decision import TOOL_NOT_DECLARED_CODE, TURN_CANCELLED_CODE
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import LIMIT_SECONDS, LIST_TOOLS, active, paused
from tests.support.transport import ClosingHold, transport_closing_with_late_answers

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
    await runner.reports.until("turn_ended")

    assert runner.reports.events() == [("started", None), ("turn_ended", TURN_CANCELLED_CODE)]
    assert runner.gateway.requests == []
    assert runner.app.said == []
    assert not runner.task.done()


async def test_a_turn_paused_mid_way_is_cancelled_and_after_the_resume_the_old_lines_are_not_answered(runner):
    await a_turn_in_its_first_round(runner)
    say_in_the_call(runner, f"{ADDRESSED} While the turn ran.")

    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.reports.until("turn_ended")
    say_in_the_call(runner, f"{ADDRESSED} During the pause.")
    resumed_read = runner.app.hold_next_window_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    resumed_read.set()
    await runner.app.said_count(1)

    assert runner.reports.events() == [("started", None), ("turn_ended", TURN_CANCELLED_CODE)]
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
    await runner.reports.until("turn_ended")
    later_read = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()

    assert runner.reports.events() == [("started", None), ("turn_ended", TURN_CANCELLED_CODE)]
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

        assert runner.app.window_reads(OTHER_PLACE_ID) == window_reads
        assert runner.gateway.requests == []
        unwinding.release.set()
        later_read.set()
        await runner.app.said_count(1)
    finally:
        unwinding.release.set()

    assert runner.app.said == [(SUBJECT, OTHER_PLACE_ID, "There.")]
    assert runner.reports.events() == [("started", None), ("turn_ended", TURN_CANCELLED_CODE)]


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

    assert runner.reports.events() == [("started", None), ("turn_ended", TOOL_NOT_DECLARED_CODE)]
    assert not runner.task.done()


async def test_a_runner_cancelled_during_a_turn_ends_cancelled_and_reports_nothing_for_the_turn(runner):
    await a_turn_in_its_first_round(runner)

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    assert runner.reports.events() == [("started", None)]


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
