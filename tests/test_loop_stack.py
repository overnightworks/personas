"""The whole runner, in process: the real loop and SDK session against a realm and an app that answer for real."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import pytest
from mcp_types import INVALID_PARAMS

from agents.decision import SUSPENDED_RETRY_SECONDS
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import active, invited

pytestmark = pytest.mark.anyio

LIMIT_SECONDS = 10
REFUSALS_BEFORE_THE_FIX = 3


async def next_wait(runner):
    return await asyncio.wait_for(runner.app.changes.started.get(), LIMIT_SECONDS)


async def test_one_wait_is_in_flight_with_the_cursor_carried_and_an_invitation_is_accepted_once(runner):
    runner.start()
    assert await next_wait(runner) is None
    first_cursor = await next_wait(runner)
    assert first_cursor is not None

    runner.app.stand(SUBJECT, invited("room-7", "inv-3", epoch=2))
    await runner.reports.until("joined")
    runner.app.stand(SUBJECT, active("room-7"))
    carried_after_change = await next_wait(runner)

    assert carried_after_change not in (None, first_cursor)
    assert runner.app.changes.most_in_flight == 1
    assert runner.app.accepted == [{"room": "room-7", "invitation": "inv-3", "epoch": 2}]
    assert runner.reports.events() == [("started", None), ("joined", None)]


async def drain(queue, count):
    for _ in range(count):
        await asyncio.wait_for(queue.get(), LIMIT_SECONDS)


async def test_a_disabled_client_suspends_the_loop_before_any_app_call_and_a_granted_token_resumes_it(runner):
    runner.realm.refusing = True
    runner.start()
    await drain(runner.realm.refusals, REFUSALS_BEFORE_THE_FIX)

    assert runner.app.requests_seen == 0
    assert runner.reports.events() == [("started", None), ("suspended", "http_401")]
    assert set(runner.clock.slept) == {SUSPENDED_RETRY_SECONDS}

    runner.realm.refusing = False
    await runner.reports.until("resumed")
    runner.app.stand(SUBJECT, active("room-7"))
    await next_wait(runner)
    await next_wait(runner)
    assert runner.app.accepted == []

    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    await runner.reports.until("joined")
    assert len(runner.app.accepted) == 1


async def test_an_app_that_refuses_with_403_suspends_the_loop_until_it_answers_again(runner):
    runner.app.refuse_with = HTTPStatus.FORBIDDEN
    runner.start()
    await drain(runner.app.refusals, REFUSALS_BEFORE_THE_FIX)

    assert runner.reports.events() == [("started", None), ("suspended", "http_403")]

    runner.app.refuse_with = None
    await runner.reports.until("resumed")
    assert runner.reports.events() == [("started", None), ("suspended", "http_403"), ("resumed", None)]


async def test_a_token_the_app_does_not_accept_is_renewed_once_and_then_suspends_the_loop(runner):
    runner.realm.reaches_app = False
    runner.start()
    await drain(runner.app.refusals, 2 * REFUSALS_BEFORE_THE_FIX)

    assert runner.reports.events() == [("started", None), ("suspended", "http_401")]
    assert len(runner.realm.issued) > 2


async def test_a_wait_tool_that_answers_an_error_suspends_the_loop_until_it_answers_standings_again(runner):
    runner.app.standings_fail = True
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", "wait_error")]
    assert set(runner.clock.slept) == {SUSPENDED_RETRY_SECONDS}

    runner.app.standings_fail = False
    await runner.reports.until("resumed")
    assert runner.reports.events() == [("started", None), ("suspended", "wait_error"), ("resumed", None)]


async def test_an_app_without_the_accept_tool_suspends_the_loop_with_the_protocol_error_code(runner):
    runner.app.offers_accept = False
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", f"jsonrpc_{INVALID_PARAMS}")]
    assert runner.app.accepted == []


@pytest.mark.parametrize("turns_before_stop", range(0, 40, 3))
async def test_a_runner_stopped_while_it_connects_ends(runner, turns_before_stop):
    runner.realm.reaches_app = False
    runner.start()
    for _ in range(turns_before_stop):
        await asyncio.sleep(0)

    runner.task.cancel()

    async with asyncio.timeout(LIMIT_SECONDS):
        await asyncio.gather(runner.task, return_exceptions=True)
    assert runner.task.done()
