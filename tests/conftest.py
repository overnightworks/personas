from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import httpx2
import pytest
from agents.agent_file import load_agent
from agents.backend.fake_gateway import completion
from agents.backend.gateway import GatewayBackend
from agents.turn import Principal

from personas.app_session import READ_SECONDS, open_mcp_session
from personas.config import DEFAULT_HISTORY_BYTES, Voice, load_config
from personas.loop import AgentLoop, Boundaries
from personas.main import QUIET_LIBRARIES, SweepingBoundaries, configure_logging, run_with_sweep
from personas.mind import Mind
from personas.readiness import FileReadyMarker
from personas.reporting import log_state_change
from personas.secret import Secret
from personas.tokens import HttpTokenSource
from tests.support.configfiles import MAX_TOOL_CALLS, PERSONA_MAX_DURATION_SECONDS, Files
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
from tests.support.gateway import Gateway
from tests.support.hopin import AGENT_SUBJECT, ISSUER, PLAY_TOOL, FakeApp, formatted
from tests.support.realm import TOKEN_ENDPOINT, FakeRealm
from tests.support.songmaker import CALLS, CREATE_SONG, GENERATE_SONG, ToolsOnlyApp
from tests.support.songmaker import SCOPE as TOOLS_ONLY_SCOPE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Iterator
    from pathlib import Path

    from agents.backend import AgentBackend
    from agents.member import MemberStanding
    from agents.tools import HandTool

    from personas.app_session import SessionOpener
    from personas.config import MemoryTarget
    from personas.decision import Refusal
    from personas.failures import CallFailure
    from personas.remembering import LineStore
    from personas.reporting import StateChange
    from personas.sweep import SweptStore
    from personas.tokens import GrantedToken

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
        self.loop = AgentLoop(credentials(), app_target(), boundaries, Mind(DEFAULT_HISTORY_BYTES), (app_target(),))

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

    def __init__(self, files: Files, app: FakeApp, app_url: str) -> None:
        files.text = files.valid_text(app_url, SCOPE)
        self.config = load_config(files.write())
        self.files = files
        self.ready_file = files.ready_path
        self.app_url = app_url
        self.on_report: Callable[[StateChange], None] = lambda change: None
        self.app = app
        self.app_principal = Principal(iss=ISSUER, sub=AGENT_SUBJECT)
        self.agent = self.config.agents[0]
        self.gateway = Gateway(completion("done"))
        self.realm = FakeRealm(self.app, self.agent, AGENT_SUBJECT, SCOPE)
        self.clock = FakeClock()
        self.reports = Reports()
        self.token_client = httpx2.AsyncClient(transport=self.realm.transport())
        self.task: asyncio.Task[object] | None = None
        self.tools_only_apps: list[ToolsOnlyApp] = []
        self.other_apps: list[FakeApp] = []
        self.lines: LineStore | None = None
        self.swept: SweptStore | None = None

    def remember_in(self, store: LineStore, target: MemoryTarget) -> None:
        """Gives the runner a memory: the store its loops keep and read lines through, and the table that names it."""
        self.lines = store
        self.swept = store
        self.config = replace(self.config, memory=target)

    def add_tools_only_app(self, app: ToolsOnlyApp, url: str) -> None:
        """Binds the tools of an app that listens to nothing and lets the agent's tokens for its scope into it."""
        self.files.text += (
            f'[[apps]]\nmcp_url = "{url}"\nscope = "{TOOLS_ONLY_SCOPE}"\nlistens = false\n'
            f'tools = ["{CREATE_SONG}", "{GENERATE_SONG}"]\nmax_tool_calls = {CALLS}\n'
        )
        self.config = load_config(self.files.write())
        self.realm.add_app(TOOLS_ONLY_SCOPE, app)
        self.tools_only_apps.append(app)

    def add_listening_app(self, app: FakeApp, url: str, scope: str) -> None:
        """Binds the line tool of one more app the agents listen to, and lets their tokens for its scope into it.

        It reloads the configuration, so it comes before the other changes of a scenario.
        """
        self.files.text += (
            f'[[apps]]\nmcp_url = "{url}"\nscope = "{scope}"\ntools = ["{app.line_tool}"]\n'
            f"max_tool_calls = {MAX_TOOL_CALLS}\n"
        )
        self.config = load_config(self.files.write())
        self.realm.add_app(scope, app)
        self.other_apps.append(app)

    def give_voice(self, max_duration_seconds: int = PERSONA_MAX_DURATION_SECONDS) -> None:
        """Turns every agent's voice on and gives the persona this longest duration."""
        self.files.persona_path.write_text(self.files.persona_text(max_duration_seconds), encoding="utf-8")
        persona = load_agent(self.files.persona_path, [])
        self.config = replace(
            self.config, agents=tuple(replace(each, persona=persona, voice=Voice(None)) for each in self.config.agents)
        )

    def bound_tools_only_calls_to(self, seconds: float) -> None:
        """Gives every tools-only app this call time; the amounts of a scenario are its own."""
        self.config = replace(
            self.config,
            apps=tuple(replace(each, call_seconds=seconds) if not each.listens else each for each in self.config.apps),
        )

    def add_app(self) -> None:
        """A second app of the same agents; it answers at the same address as the first."""
        self.config = replace(self.config, apps=(*self.config.apps, *self.config.apps))

    def add_agent(self, client_id: str, subject: str, name: str | None = None) -> None:
        """A second agent with the first one's persona, its own client and its own subject in the app.

        Its token names it ``name``, or the first agent's name.
        """
        second = replace(self.config.agents[0], client_id=client_id, client_secret=Secret(secrets.token_hex(16)))
        self.config = replace(self.config, agents=(*self.config.agents, second))
        self.realm.add_client(second, subject, name)

    def _report(self, change: StateChange) -> None:
        self.reports(change)
        log_state_change(change)
        self.on_report(change)

    def start(self, *, stream_seconds: float = READ_SECONDS, open_session: SessionOpener | None = None) -> None:
        boundaries = SweepingBoundaries(
            tokens=HttpTokenSource(TOKEN_ENDPOINT, self.token_client),
            open_session=open_session or partial(open_mcp_session, stream_seconds=stream_seconds),
            clock=self.clock,
            report=self._report,
            ready=FileReadyMarker(self.ready_file),
            backend=self._backend,
            lines=self.lines,
            swept=self.swept,
        )
        self.task = asyncio.create_task(run_with_sweep(self.config, boundaries))

    def _backend(self, tools: Iterable[HandTool]) -> AgentBackend:
        assert self.config.gateway is not None
        return GatewayBackend(self.config.gateway, tools=tools, transport=self.gateway)

    def secrets_and_content(self) -> list[str]:
        return [
            *(agent.client_secret.reveal() for agent in self.config.agents),
            self.files.gateway_key,
            self.files.password,
            self.files.persona_prompt,
            *(held for agent in self.config.agents if agent.figure for held in (agent.figure.url, agent.figure.sha256)),
            *self.realm.issued,
            *self.app.kept_out_of_logs(),
            *(held for each in [*self.tools_only_apps, *self.other_apps] for held in each.kept_out_of_logs()),
        ]


@pytest.fixture
async def songmaker(runner: Harness) -> AsyncIterator[ToolsOnlyApp]:
    """An app that listens to nothing and binds the song tools, served and bound into the runner's configuration."""
    app = ToolsOnlyApp()
    async with app.serving() as url:
        runner.add_tools_only_app(app, url)
        yield app


@pytest.fixture
def library_loggers() -> Iterator[None]:
    """The levels of the loggers `configure_logging` quiets, put back at the test's end."""
    loggers = [logging.getLogger(name) for name in QUIET_LIBRARIES]
    levels = [each.level for each in loggers]
    yield
    for each, level in zip(loggers, levels, strict=True):
        each.setLevel(level)


@pytest.fixture
async def runner(caplog: pytest.LogCaptureFixture, library_loggers: None, files: Files) -> AsyncIterator[Harness]:
    """The runner's world; at its end no log line holds a secret, a token or the content of a tool."""
    configure_logging()
    caplog.set_level(logging.DEBUG)
    app = FakeApp()
    async with app.serving() as app_url:
        harness = Harness(files, app, app_url)
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


@pytest.fixture(scope="session")
def postgres_template() -> Iterator[str]:
    """The name of the migrated template database; the first test that asks creates it.

    The support module imports the database library, which the tests of the loop never load.
    """
    from tests.support.postgres import migrated_template

    with migrated_template() as name:
        yield name


@pytest.fixture
def database(postgres_template: str) -> Iterator[MemoryTarget]:
    """A migrated database of this test's own."""
    from tests.support.postgres import created_database, target_of

    with created_database(postgres_template) as name:
        yield target_of(name)


@pytest.fixture
def unmigrated_database() -> Iterator[MemoryTarget]:
    from tests.support.postgres import created_database, target_of

    with created_database() as name:
        yield target_of(name)


@pytest.fixture
async def other(runner):
    """A second app the agent listens to."""
    app = FakeApp(line_tool=PLAY_TOOL)
    async with app.serving() as url:
        runner.add_listening_app(app, url, "mcp-other")
        yield app
