"""The entry point's logging: only the runner's own state changes reach a log line."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import anyio
import httpx2
import pytest
from agents.backend.gateway import GatewayBackend
from agents.turn_events import FinalEvent
from logline import current_traceparent, unit_trace

from personas.app_session import open_app_session
from personas.config import CONFIG_ENVIRONMENT_VARIABLE, load_config
from personas.failures import STORE_UNREACHABLE_CODE, SchemaNotAtHeadError, StoreUnavailable, leaves_of
from personas.main import (
    CONFIGURATION_ERROR_EXIT_CODE,
    READINESS_ERROR_EXIT_CODE,
    STORE_ERROR_EXIT_CODE,
    SchemaChecked,
    SweepingBoundaries,
    configure_logging,
    main,
    real_boundaries,
)
from personas.memory import PostgresLineStore
from personas.readiness import FileReadyMarker
from personas.reporting import Event, log_state_change
from personas.tokens import HttpTokenSource
from tests.support.fakes import FakeClock, credentials
from tests.support.gateway import Gateway, says
from tests.support.hopin import FakeApp, trace_id_of
from tests.support.ownership import AllAgentsOwned
from tests.support.realm import FakeRealm
from tests.support.standings import AGENT_SUBJECT, active, invited

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Sequence
    from pathlib import Path
    from typing import NoReturn

    from agents.agent_file import TurnRequest
    from agents.backend import TurnEvent, TurnPerson
    from agents.tools import HandTool

    from personas.config import AppTarget, Config
    from personas.entries import WindowRead
    from personas.ownership import ServeAgent
    from personas.readiness import ReadyMarker
    from personas.remembering import LineStore
    from personas.reporting import StateChange
    from personas.secret import Secret
    from tests.support.configfiles import Files


def test_every_line_the_entry_point_configures_is_one_json_object_of_the_runner_on_stdout(
    restored_logging, capsys: pytest.CaptureFixture[str]
):
    """LOG-1."""
    configure_logging()

    logging.getLogger("personas").info("loop.started", extra={"agent_id": "mozart"})

    line = json.loads(capsys.readouterr().out)
    assert (line["service"], line["event"], line["agent_id"]) == ("personas", "loop.started", "mozart")


@pytest.mark.parametrize(
    ("error", "level"),
    [
        pytest.param(anyio.ClosedResourceError(), "info", id="closed stream"),
        pytest.param(anyio.BrokenResourceError(), "info", id="broken stream"),
        pytest.param(ValueError("not json"), "error", id="a real parse failure"),
    ],
)
def test_the_mcp_clients_error_for_a_closed_stream_is_written_at_info_and_any_other_stays_at_error(
    restored_logging, capsys: pytest.CaptureFixture[str], error, level
):
    """LOG-1."""
    configure_logging()

    try:
        raise error
    except Exception:
        logging.getLogger("mcp.client.streamable_http").exception("sse_message.failed")

    assert json.loads(capsys.readouterr().out)["level"] == level


@pytest.mark.parametrize("arguments", [[], ["migrate"], ["memory", "list"]], ids=["runner", "migrate", "memory"])
def test_a_log_level_logline_refuses_is_one_plain_line_on_stderr_and_the_configuration_exit_code_of_every_command(
    files, restored_logging, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arguments
):
    """LOG-1."""
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))
    monkeypatch.setenv("LOG_LEVEL", "loud")

    with pytest.raises(SystemExit) as refused:
        main(arguments=arguments)

    assert refused.value.code == CONFIGURATION_ERROR_EXIT_CODE
    error = capsys.readouterr().err
    assert error.startswith("personas: LOG_LEVEL is 'loud'")
    assert error.count("\n") == 1


def test_an_uncaught_exception_after_the_entry_point_configures_is_one_json_crash_line_on_stdout(
    restored_logging, capsys: pytest.CaptureFixture[str]
):
    """LOG-1."""
    configure_logging()

    try:
        raise LookupError
    except LookupError as crash:
        sys.excepthook(type(crash), crash, crash.__traceback__)

    written = capsys.readouterr()
    line = json.loads(written.out)
    assert (line["service"], line["level"], line["event"], line["error_type"]) == (
        "personas",
        "critical",
        "process.crashed",
        "LookupError",
    )
    assert "LookupError" in line["exception"]
    assert written.err == ""


class Stop(Exception):  # noqa: N818 - a test's way to end a runner whose loops never end
    """Raised by the report once an agent joined, which is after its first read answered."""


class StopsAfterTheAnswer:
    """The gateway backend, which raises ``Stop`` once the turn has its final answer."""

    def __init__(self, inner: GatewayBackend) -> None:
        self._inner = inner

    async def __aenter__(self) -> StopsAfterTheAnswer:
        return self

    async def __aexit__(self, *error: object) -> None:
        await self._inner.aclose()

    async def run(self, request: TurnRequest, person: TurnPerson) -> AsyncIterator[TurnEvent]:
        async for event in self._inner.run(request, person):
            yield event
            if isinstance(event, FinalEvent):
                raise Stop


class NotingTheMarker:
    """The ownership that serves every agent and notes whether the marker existed when it started."""

    def __init__(self, inner: AllAgentsOwned, marker: Path, noted: list[bool]) -> None:
        self._inner = inner
        self._marker = marker
        self._noted = noted

    async def serve(
        self, client_ids: Sequence[str], serve_agent: ServeAgent, contending: Callable[[], None]
    ) -> NoReturn:
        self._noted.append(self._marker.exists())
        await self._inner.serve(client_ids, serve_agent, contending)


class World:
    """The real entry point, loops and adapters over a fake realm and app that answer for real."""

    def __init__(self, files: Files) -> None:
        self.files = files
        self.app = FakeApp()
        self.ownership = AllAgentsOwned()
        self.marker_when_ownership_started: list[bool] = []
        self.gateway = Gateway(*says("room-7", files.persona_prompt))
        self.issued: list[str] = []
        self.app.stand(AGENT_SUBJECT, invited("room-7", "inv-3"))
        self.on_joined: Callable[[], None] = self._stop
        self.lines: LineStore | None = None
        self.schema: SchemaChecked | None = None
        self.events: list[Event] = []

    @staticmethod
    def _stop() -> None:
        raise Stop

    def _report(self, change: StateChange) -> None:
        log_state_change(change)
        self.events.append(change.event)
        if change.event is Event.JOINED:
            self.on_joined()

    @asynccontextmanager
    async def compose(self, config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
        realm = FakeRealm(self.app, config.agents[0], AGENT_SUBJECT, config.apps[0].scope)
        self.issued = realm.issued

        def gateway_backend(tools: Iterable[HandTool]) -> StopsAfterTheAnswer:
            assert config.gateway is not None
            return StopsAfterTheAnswer(GatewayBackend(config.gateway, tools=tools, transport=self.gateway))

        def open_at_the_app(app: AppTarget, bearer: Callable[[], Secret], agent_id: str):
            return open_app_session(replace(app, mcp_url=app_url), bearer, agent_id)

        async with (
            self.app.serving() as app_url,
            httpx2.AsyncClient(transport=realm.transport()) as token_client,
        ):
            yield SweepingBoundaries(
                tokens=HttpTokenSource(config.token_endpoint, token_client),
                open_session=open_at_the_app,
                clock=FakeClock(),
                report=self._report,
                ready=ready,
                backend=gateway_backend,
                lines=self.lines,
                schema=self.schema,
                ownership=NotingTheMarker(self.ownership, self.files.ready_path, self.marker_when_ownership_started),
            )

    def start(self) -> None:
        main(self.compose)


@pytest.fixture
def world(files: Files, monkeypatch: pytest.MonkeyPatch, restored_logging: None) -> World:
    files.write()
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.config_path))
    return World(files)


def test_a_marker_left_from_an_earlier_run_is_gone_before_the_agents_are_contended_for(world):
    world.files.ready_path.touch()

    with pytest.raises(ExceptionGroup):
        world.start()

    assert world.marker_when_ownership_started == [False]


def test_the_marker_is_created_when_the_runner_contends_for_its_agents_and_stays(world):
    with pytest.raises(ExceptionGroup):
        world.start()

    assert world.files.ready_path.exists()


class AnswersOnItsThirdAsk:
    """A memory database that cannot be reached at the first two checks of its schema and is at its head at the
    third."""

    def __init__(self) -> None:
        self.asked = 0

    async def require_head(self) -> None:
        self.asked += 1
        if self.asked <= 2:
            raise StoreUnavailable(STORE_UNREACHABLE_CODE)


def test_a_runner_whose_database_cannot_be_reached_reports_it_once_and_starts_once_the_schema_answers(world):
    world.schema = AnswersOnItsThirdAsk()

    with pytest.raises(ExceptionGroup):
        world.start()

    assert [each for each in world.events if each in {Event.STORE_FAILED, Event.STORE_RESUMED}] == [
        Event.STORE_FAILED,
        Event.STORE_RESUMED,
    ]
    assert world.files.ready_path.exists()


class StoreBehindHead:
    """A store whose database is behind the newest migration, which it finds out at its first use."""

    BEHIND = "memory: the database is at revision none but personas needs head; run personas migrate"

    async def keep(self, read: WindowRead) -> None:
        raise SchemaNotAtHeadError(self.BEHIND)


def test_a_runner_whose_memory_is_behind_its_head_stops_with_one_line_naming_memory(world, capsys):
    def addressed_once_joined() -> None:
        world.app.stand(AGENT_SUBJECT, active("room-7"))
        world.app.speak("room-7", "visitor-subject", "Ada", "Mozart, are you there?")

    world.on_joined = addressed_once_joined
    world.lines = StoreBehindHead()

    with pytest.raises(SystemExit) as stopped:
        world.start()

    assert stopped.value.code == STORE_ERROR_EXIT_CODE
    assert stopped.value.__cause__ is None
    error = capsys.readouterr().err
    assert error.startswith("personas: memory:")
    assert error.count("\n") == 1


def test_a_marker_that_cannot_be_removed_stops_the_start_and_names_its_path(world, capsys):
    world.files.ready_path.mkdir()
    (world.files.ready_path / "held").touch()

    with pytest.raises(SystemExit) as stopped:
        world.start()

    assert stopped.value.code == READINESS_ERROR_EXIT_CODE
    assert str(world.files.ready_path) in capsys.readouterr().err
    assert world.marker_when_ownership_started == []


def test_a_marker_that_cannot_be_created_stops_the_runner_and_names_its_path(files, world, capsys):
    files.text = files.text.replace(str(files.ready_path), str(files.ready_path / "missing-directory" / "ready"))
    files.write()

    with pytest.raises(SystemExit) as stopped:
        world.start()

    assert stopped.value.code == READINESS_ERROR_EXIT_CODE
    assert "missing-directory" in capsys.readouterr().err


def test_a_marker_that_cannot_be_created_ends_the_runner_with_its_exit_code_though_other_loops_are_cancelled(
    files, world, capsys
):
    first_app = files.text[files.text.index("[[apps]]") : files.text.index("[[agents]]")]
    second_app = first_app.replace("hopin.test", "second.hopin.test")
    files.text = files.text.replace(str(files.ready_path), str(files.ready_path / "missing-directory" / "ready"))
    files.text = files.text.replace("[[agents]]", f"{second_app}[[agents]]")
    files.write()
    world.app.hold_next_read()

    with pytest.raises(SystemExit) as stopped:
        world.start()

    assert stopped.value.code == READINESS_ERROR_EXIT_CODE
    assert "missing-directory" in capsys.readouterr().err


def test_a_sigterm_that_cancels_the_runner_ends_the_entry_point_without_raising(world):
    world.on_joined = lambda: None

    @asynccontextmanager
    async def compose_then_stop(config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
        async with world.compose(config, ready) as boundaries:
            os.kill(os.getpid(), signal.SIGTERM)
            yield boundaries

    main(compose_then_stop)


SWEEP_MINUTES = 7.0


class SweepEndedBySigterm:
    """A store and a clock that send the runner a SIGTERM inside a sweep, or inside the wait after it, and then wait."""

    def __init__(self, *, in_a_sweep: bool) -> None:
        self._in_a_sweep = in_a_sweep
        self.time = FakeClock()

    async def erase_before(self, first_read_before) -> int:
        if self._in_a_sweep:
            await self._terminated()
        return 0

    def now(self) -> float:
        return self.time.now()

    def wall(self):
        return self.time.wall()

    async def sleep(self, seconds: float) -> None:
        if seconds == SWEEP_MINUTES * 60:
            await self._terminated()
        await self.time.sleep(seconds)

    @staticmethod
    async def _terminated() -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.Event().wait()


@pytest.mark.parametrize("in_a_sweep", [False, True], ids=["in the wait", "in a sweep"])
def test_a_sigterm_in_a_sweep_or_in_its_wait_ends_the_entry_point_without_raising(world, in_a_sweep):
    world.on_joined = lambda: None
    world.files.text += world.files.memory_table(sweep_minutes=str(SWEEP_MINUTES))
    world.files.write()
    sweeping = SweepEndedBySigterm(in_a_sweep=in_a_sweep)

    @asynccontextmanager
    async def compose_sweeping(config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
        async with world.compose(config, ready) as boundaries:
            yield replace(boundaries, lines=sweeping, swept=sweeping, clock=sweeping)

    main(compose_sweeping)


def test_the_real_boundaries_hand_the_loops_the_marker_they_were_given(files):
    files.write()
    config = load_config({CONFIG_ENVIRONMENT_VARIABLE: str(files.config_path)})
    marker = FileReadyMarker(config.ready_file)

    async def boundaries_ready() -> ReadyMarker:
        async with real_boundaries(config, marker) as boundaries:
            return boundaries.ready

    assert asyncio.run(boundaries_ready()) is marker


def test_the_real_boundaries_send_the_current_trace_to_the_realm_and_none_outside_one(files, restored_logging):
    files.write()
    seen: list[str | None] = []

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = (await reader.readuntil(b"\r\n\r\n")).decode().lower()
        length = int(head.split("content-length: ")[1].split("\r\n")[0])
        await reader.readexactly(length)
        seen.append(next((line.split(": ")[1] for line in head.split("\r\n") if line.startswith("traceparent")), None))
        body = b'{"access_token": "t", "expires_in": 5}'
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s" % (len(body), body)
        )
        await writer.drain()
        writer.close()

    async def fetched_twice() -> str | None:
        async with await asyncio.start_server(answer, "127.0.0.1", 0) as realm:
            port = realm.sockets[0].getsockname()[1]
            config = replace(
                load_config({CONFIG_ENVIRONMENT_VARIABLE: str(files.config_path)}),
                token_endpoint=f"http://127.0.0.1:{port}/token",
            )
            async with real_boundaries(config, FileReadyMarker(config.ready_file)) as boundaries:
                agent = credentials()
                await boundaries.tokens.fetch(agent, "mcp-app")
                with unit_trace(None):
                    inside = current_traceparent()
                    await boundaries.tokens.fetch(agent, "mcp-app")
                return inside

    inside = asyncio.run(fetched_twice())

    assert inside is not None
    assert [seen[0], seen[1] and trace_id_of(seen[1])] == [None, trace_id_of(inside)]


def test_an_addressed_line_gets_one_say_through_the_entry_point_and_nothing_secret_or_said_reaches_the_log(
    world, restored_logging, capsys
):
    answer = world.files.persona_prompt

    def addressed_once_joined() -> None:
        world.app.stand(AGENT_SUBJECT, active("room-7"))
        world.app.speak("room-7", "visitor-subject", "Ada", "Mozart, are you there?")

    world.on_joined = addressed_once_joined

    with pytest.raises(ExceptionGroup) as ended:
        world.start()

    assert [type(leaf) for leaf in leaves_of(ended.value)] == [Stop]
    assert world.app.said == [(AGENT_SUBJECT, "room-7", answer)]
    logged = capsys.readouterr().out
    assert "loop.started" in [json.loads(line)["event"] for line in logged.splitlines()]
    for held in (world.files.secret, world.files.gateway_key, *world.issued, *world.app.kept_out_of_logs()):
        assert held not in logged


@pytest.mark.parametrize("binds_tools", [True, False], ids=["tools bound", "lifecycle only"])
def test_the_real_boundaries_run_turns_through_the_gateway_exactly_when_the_configuration_has_one(
    files, restored_logging, binds_tools
):
    files.text = files.valid_text(binds_tools=binds_tools)
    config = load_config(files.write())

    async def composed_backend() -> object:
        async with real_boundaries(config, FileReadyMarker(config.ready_file)) as boundaries:
            return None if boundaries.backend is None else boundaries.backend(())

    backend = asyncio.run(composed_backend())

    assert isinstance(backend, GatewayBackend) is binds_tools


class ReadyTooEarlyError(Exception):
    """The runner announced it is ready."""


class MarkerThatMustStayAbsent:
    """A file marker that stops the runner when it is announced, so a runner that would go on ends."""

    def __init__(self, marker: FileReadyMarker) -> None:
        self._marker = marker

    def clear(self) -> None:
        self._marker.clear()

    def announce(self) -> None:
        self._marker.announce()
        raise ReadyTooEarlyError


@pytest.mark.postgres
@pytest.mark.parametrize("has_memory", [True, False], ids=["memory table", "no memory table"])
def test_a_runner_whose_database_is_behind_its_head_stops_before_it_writes_its_ready_marker(
    files, database, monkeypatch, capsys, has_memory, restored_logging
):
    from tests.support.postgres import run_sql

    if has_memory:
        files.text += files.memory_table(lines_keep_days="inf")
    files.use_database(database)
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, files.write()[CONFIG_ENVIRONMENT_VARIABLE])
    run_sql(database, "UPDATE alembic_version SET version_num = '0001'")

    @asynccontextmanager
    async def real_with_a_watched_marker(config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
        async with real_boundaries(config, MarkerThatMustStayAbsent(FileReadyMarker(config.ready_file))) as boundaries:
            yield boundaries

    with pytest.raises(SystemExit) as stopped:
        main(real_with_a_watched_marker)

    assert stopped.value.code == STORE_ERROR_EXIT_CODE
    assert capsys.readouterr().err.startswith("personas: memory:")
    assert not files.ready_path.exists()


class WarnsAfterTheSchemaCheck:
    """The real schema check, after which alembic warns and the runner stops."""

    def __init__(self, inner: SchemaChecked) -> None:
        self._inner = inner

    async def require_head(self) -> None:
        await self._inner.require_head()
        logging.getLogger("alembic.runtime.migration").warning("migration.warned")
        raise Stop


@pytest.mark.postgres
def test_a_runner_with_a_memory_writes_no_info_line_of_alembic_and_its_warning_still_reaches_stdout(
    files, database, monkeypatch, restored_logging, capsys
):
    """LOG-1."""
    files.text += files.memory_table()
    files.use_database(database)
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, files.write()[CONFIG_ENVIRONMENT_VARIABLE])

    @asynccontextmanager
    async def real_with_a_warning(config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
        async with real_boundaries(config, ready) as boundaries:
            assert boundaries.schema is not None
            yield replace(boundaries, schema=WarnsAfterTheSchemaCheck(boundaries.schema))

    with pytest.raises(Stop):
        main(real_with_a_warning)

    alembic_lines = [
        line for line in map(json.loads, capsys.readouterr().out.splitlines()) if line["logger"].startswith("alembic")
    ]
    assert [(line["level"], line["event"]) for line in alembic_lines] == [("warning", "migration.warned")]


@pytest.mark.parametrize("has_memory", [True, False], ids=["memory table", "no memory table"])
def test_the_real_boundaries_hold_a_store_exactly_when_the_configuration_has_a_memory_table(files, has_memory):
    """CONFIG-33."""
    if has_memory:
        files.text += files.memory_table()
    config = load_config(files.write())

    async def composed_store() -> tuple[object, object]:
        async with real_boundaries(config, FileReadyMarker(config.ready_file)) as boundaries:
            return boundaries.lines, boundaries.swept

    store, swept = asyncio.run(composed_store())

    assert isinstance(store, PostgresLineStore) is has_memory
    assert (swept is store) if has_memory else (swept is None)
