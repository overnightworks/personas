"""The runner over several loops: what a loop that ends cancelled means, and what stays as it is."""

from __future__ import annotations

import asyncio

import pytest

from agents import app_session
from agents.failures import leaves_of
from agents.loop import UnrequestedCancellationError
from tests.support.call import PLACE_ID
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import LIMIT_SECONDS, active
from tests.support.transport import transport_cancelled_once

pytestmark = pytest.mark.anyio

SECOND_AGENT = "agent-ada"
SECOND_SUBJECT = "ada-subject"


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_a_loop_cancelled_by_nobody_ends_the_runner_loud_and_adds_no_other_leaf(runner, monkeypatch):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    monkeypatch.setattr(app_session, "streamable_http_client", transport_cancelled_once())
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))

    runner.start()
    ended = await ended_by(runner.task)

    assert isinstance(ended, ExceptionGroup)
    assert [type(leaf) for leaf in leaves_of(ended)] == [UnrequestedCancellationError]


async def test_a_runner_cancelled_with_several_loops_running_ends_cancelled_and_adds_no_defect(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID, 1)
    await runner.app.until_standing_read(2)

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
