from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING

import httpx2
import pytest
from agent_runtime_client.turn import Principal

from agents.app_session import READ_SECONDS, open_mcp_session
from agents.config import AppTarget, Config
from agents.loop import AgentLoop, Boundaries, run_all
from agents.main import QUIET_LIBRARIES, configure_logging
from agents.readiness import FileReadyMarker
from agents.reporting import log_state_change
from agents.tokens import HttpTokenSource
from tests.support.configfiles import Files
from tests.support.fakes import (
    FakeClock,
    FakeReadyMarker,
    FakeSession,
    FakeTokens,
    Reports,
    app_target,
    credentials,
    grant,
)
from tests.support.hopin import AGENT_SUBJECT, ISSUER, FakeApp, formatted
from tests.support.realm import TOKEN_ENDPOINT, FakeRealm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from pathlib import Path

    from agent_runtime_client.member import MemberStanding

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
        self.ready = FakeReadyMarker()
        boundaries = Boundaries(tokens, session.open, clock, reports, self.ready)
        self.loop = AgentLoop(credentials(), app_target(), boundaries)

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
        reads: list[MemberStanding | Exception],
        *,
        tokens: list[GrantedToken | CallFailure] | None = None,
        accepts: list[Refusal | CallFailure | None] | None = None,
        seconds_per_read: float = 0.0,
        seconds_per_accept: float = 0.0,
        graceful_ends: int = 0,
    ) -> Scenario:
        clock = FakeClock()
        granted = FakeTokens(clock, list(tokens) if tokens is not None else [grant() for _ in range(5)])
        session = FakeSession(
            clock, list(reads), list(accepts or []), seconds_per_read, seconds_per_accept, graceful_ends
        )
        return Scenario(granted, session, clock, Reports())

    return build


SCOPE = "mcp-app"


class Harness:
    """The real runner (loop, token adapter, SDK session) against a fake realm and an app on a loopback socket."""

    def __init__(self, ready_file: Path, app: FakeApp, app_url: str) -> None:
        self.ready_file = ready_file
        self.app_url = app_url
        self.on_report: Callable[[StateChange], None] = lambda change: None
        self.app = app
        self.app_principal = Principal(iss=ISSUER, sub=AGENT_SUBJECT)
        self.agent = credentials()
        self.realm = FakeRealm(self.app, self.agent, AGENT_SUBJECT, SCOPE)
        self.clock = FakeClock()
        self.reports = Reports()
        self.token_client = httpx2.AsyncClient(transport=self.realm.transport())
        self.task: asyncio.Task[object] | None = None

    def _report(self, change: StateChange) -> None:
        self.reports(change)
        log_state_change(change)
        self.on_report(change)

    def start(self, *, stream_seconds: float = READ_SECONDS) -> None:
        boundaries = Boundaries(
            tokens=HttpTokenSource(TOKEN_ENDPOINT, self.token_client),
            open_session=partial(open_mcp_session, stream_seconds=stream_seconds),
            clock=self.clock,
            report=self._report,
            ready=FileReadyMarker(self.ready_file),
        )
        config = Config(TOKEN_ENDPOINT, self.ready_file, (AppTarget(self.app_url, SCOPE),), (self.agent,))
        self.task = asyncio.create_task(run_all(config, boundaries))

    def secrets_and_content(self) -> list[str]:
        return [self.agent.client_secret.reveal(), *self.realm.issued, *self.app.kept_out_of_logs()]


@pytest.fixture
def library_loggers() -> Iterator[None]:
    """The levels of the loggers `configure_logging` quiets, put back at the test's end."""
    loggers = [logging.getLogger(name) for name in QUIET_LIBRARIES]
    levels = [each.level for each in loggers]
    yield
    for each, level in zip(loggers, levels, strict=True):
        each.setLevel(level)


@pytest.fixture
async def runner(caplog: pytest.LogCaptureFixture, library_loggers: None, tmp_path: Path) -> AsyncIterator[Harness]:
    """The runner's world; at its end no log line holds a secret, a token or the content of a tool."""
    configure_logging()
    caplog.set_level(logging.DEBUG)
    app = FakeApp()
    async with app.serving() as app_url:
        harness = Harness(tmp_path / "ready", app, app_url)
        yield harness
        if harness.task is not None:
            harness.task.cancel()
            await asyncio.gather(harness.task, return_exceptions=True)
    await harness.token_client.aclose()
    logged = formatted(caplog.get_records("call"))
    assert not harness.reports.changes or "client_id=" in logged
    for held in harness.secrets_and_content():
        assert held not in logged


@pytest.fixture
def files(tmp_path: Path) -> Files:
    return Files(tmp_path)
