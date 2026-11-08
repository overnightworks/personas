from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING

import httpx2
import pytest

from agents.app_session import open_mcp_session
from agents.config import AppTarget, Config
from agents.loop import AgentLoop, Boundaries, run_all
from agents.main import QUIET_LIBRARIES, configure_logging
from agents.reporting import log_state_change
from agents.tokens import HttpTokenSource
from tests.support.fakes import FakeClock, FakeSession, FakeTokens, Reports, app_target, credentials, grant
from tests.support.hopin import AGENT_SUBJECT, APP_URL, FakeApp
from tests.support.realm import TOKEN_ENDPOINT, FakeRealm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from agent_runtime_client.wait import WaitAnswer

    from agents.decision import Refusal
    from agents.failures import CallFailure
    from agents.reporting import StateChange
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
        seconds_per_accept: float = 0.0,
    ) -> Scenario:
        clock = FakeClock()
        granted = FakeTokens(clock, list(tokens) if tokens is not None else [grant() for _ in range(5)])
        session = FakeSession(clock, list(waits), list(accepts or []), seconds_per_wait, seconds_per_accept)
        return Scenario(granted, session, clock, Reports())

    return build


SCOPE = "mcp-app"
SCENARIO_CONTENT = ("room-7", "inv-3", "Mozart")


class Harness:
    """The real runner (loop, token adapter, SDK session) against a fake realm and app, in process."""

    def __init__(self) -> None:
        self.app = FakeApp()
        self.agent = credentials()
        self.realm = FakeRealm(self.app, self.agent, AGENT_SUBJECT, SCOPE)
        self.clock = FakeClock()
        self.reports = Reports()
        self.token_client = httpx2.AsyncClient(transport=self.realm.transport())
        self.task: asyncio.Task[object] | None = None

    def _report(self, change: StateChange) -> None:
        self.reports(change)
        log_state_change(change)

    def start(self) -> None:
        boundaries = Boundaries(
            tokens=HttpTokenSource(TOKEN_ENDPOINT, self.token_client),
            open_session=partial(open_mcp_session, transport=self.app.asgi()),
            clock=self.clock,
            report=self._report,
        )
        config = Config(TOKEN_ENDPOINT, (AppTarget(APP_URL, SCOPE),), (self.agent,))
        self.task = asyncio.create_task(run_all(config, boundaries))

    def secrets_and_content(self) -> list[str]:
        return [self.agent.client_secret.reveal(), *self.realm.issued, *SCENARIO_CONTENT]


@pytest.fixture
def library_loggers() -> Iterator[None]:
    """The levels of the loggers `configure_logging` quiets, put back at the test's end."""
    loggers = [logging.getLogger(name) for name in QUIET_LIBRARIES]
    levels = [each.level for each in loggers]
    yield
    for each, level in zip(loggers, levels, strict=True):
        each.setLevel(level)


@pytest.fixture
async def runner(caplog: pytest.LogCaptureFixture, library_loggers: None) -> AsyncIterator[Harness]:
    """The runner's world; at its end no log line holds a secret, a token or the content of a tool."""
    configure_logging()
    caplog.set_level(logging.DEBUG)
    harness = Harness()
    yield harness
    if harness.task is not None:
        harness.task.cancel()
        await asyncio.gather(harness.task, return_exceptions=True)
    await harness.token_client.aclose()
    logged = "\n".join(record.getMessage() for record in caplog.get_records("call"))
    assert not harness.reports.changes or "client_id=" in logged
    for held in harness.secrets_and_content():
        assert held not in logged
