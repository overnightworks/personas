from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from agents.loop import AgentLoop, Boundaries
from tests.support.fakes import FakeClock, FakeSession, FakeTokens, Reports, app_target, credentials, grant

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_runtime_client.wait import WaitAnswer

    from agents.decision import Refusal
    from agents.failures import CallFailure
    from agents.tokens import GrantedToken

SCENARIO_LIMIT_SECONDS = 10


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Scenario:
    """The loop of one agent in one app over scripted boundaries."""

    def __init__(self, tokens: FakeTokens, session: FakeSession, clock: FakeClock, reports: Reports) -> None:
        self.tokens = tokens
        self.session = session
        self.clock = clock
        self.reports = reports
        self.loop = AgentLoop(credentials(), app_target(), Boundaries(tokens, session.open, clock, reports))

    async def play(self) -> None:
        """Runs the loop until the script is spent, which is when a boundary blocks, then stops it."""
        task = asyncio.create_task(self.loop.run())
        spent = [asyncio.create_task(event.wait()) for event in (self.tokens.idle, self.session.idle)]
        try:
            await asyncio.wait([task, *spent], timeout=SCENARIO_LIMIT_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                task.result()
            assert spent[0].done() or spent[1].done(), "the scenario did not reach the end of its script"
        finally:
            for each in (task, *spent):
                each.cancel()


@pytest.fixture
def scenario() -> Callable[..., Scenario]:
    def build(
        waits: list[WaitAnswer | CallFailure],
        *,
        tokens: list[GrantedToken | CallFailure] | None = None,
        accepts: list[Refusal | CallFailure | None] | None = None,
        seconds_per_wait: float = 0.0,
    ) -> Scenario:
        clock = FakeClock()
        granted = FakeTokens(clock, list(tokens) if tokens is not None else [grant() for _ in range(5)])
        session = FakeSession(clock, list(waits), list(accepts or []), seconds_per_wait)
        return Scenario(granted, session, clock, Reports())

    return build
