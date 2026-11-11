"""The entry point's logging: only the runner's own state changes reach a log line."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import httpx2
import pytest
from agents.backend.gateway import GatewayBackend
from agents.turn_events import FinalEvent

from personas.app_session import open_mcp_session
from personas.config import CONFIG_ENVIRONMENT_VARIABLE, load_config
from personas.failures import leaves_of
from personas.main import (
    QUIET_LIBRARIES,
    READINESS_ERROR_EXIT_CODE,
    SweepingBoundaries,
    configure_logging,
    main,
    real_boundaries,
)
from personas.memory import PostgresLineStore
from personas.readiness import FileReadyMarker
from personas.reporting import Event, log_state_change
from personas.tokens import HttpTokenSource
from tests.support.fakes import FakeClock
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT, FakeApp, active, formatted, invited
from tests.support.realm import FakeRealm

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable

    from agents.agent_file import TurnRequest
    from agents.backend import AgentBackend, TurnEvent, TurnPerson
    from agents.tools import HandTool

    from personas.config import AppTarget, Config
    from personas.readiness import ReadyMarker
    from personas.reporting import StateChange
    from personas.secret import Secret
    from tests.support.configfiles import Files


@pytest.mark.parametrize("library", QUIET_LIBRARIES)
def test_an_error_record_of_a_library_the_runner_uses_reaches_no_log_line(
    library, library_loggers, caplog: pytest.LogCaptureFixture
):
    configure_logging()
    caplog.set_level(logging.DEBUG)

    logging.getLogger(f"{library}.client.streamable_http").error("a response body")

    assert caplog.records == []


def test_an_error_record_of_the_gateway_backend_reaches_no_log_line(library_loggers, caplog: pytest.LogCaptureFixture):
    configure_logging()
    caplog.set_level(logging.DEBUG)

    logging.getLogger("agents.backend.gateway").error("tool say raised Unreachable")

    assert caplog.records == []


class Stop(Exception):  # noqa: N818 - a test's way to end a runner whose loops never end
    """Raised by the report once an agent joined, which is after its first read answered."""


class StopsAfterTheAnswer:
    """The gateway backend, which raises ``Stop`` once the turn has its final answer."""

    def __init__(self, inner: AgentBackend) -> None:
        self._inner = inner

    async def run(self, request: TurnRequest, person: TurnPerson) -> AsyncIterator[TurnEvent]:
        async for event in self._inner.run(request, person):
            yield event
            if isinstance(event, FinalEvent):
                raise Stop


class World:
    """The real entry point, loops and adapters over a fake realm and app that answer for real."""

    def __init__(self, files: Files) -> None:
        self.files = files
        self.app = FakeApp()
        self.tokens_requested_with_marker: list[bool] = []
        self.gateway = Gateway(*says("room-7", files.persona_prompt))
        self.issued: list[str] = []
        self.app.stand(AGENT_SUBJECT, invited("room-7", "inv-3"))
        self.on_joined: Callable[[], None] = self._stop

    @staticmethod
    def _stop() -> None:
        raise Stop

    def _report(self, change: StateChange) -> None:
        log_state_change(change)
        if change.event is Event.JOINED:
            self.on_joined()

    @asynccontextmanager
    async def compose(self, config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
        realm = FakeRealm(self.app, config.agents[0], AGENT_SUBJECT, config.apps[0].scope)
        self.issued = realm.issued

        def gateway_backend(tools: Iterable[HandTool]) -> AgentBackend:
            assert config.gateway is not None
            return StopsAfterTheAnswer(GatewayBackend(config.gateway, tools=tools, transport=self.gateway))

        async def note_request(request: httpx2.Request) -> None:
            self.tokens_requested_with_marker.append(self.files.ready_path.exists())

        def open_at_the_app(app: AppTarget, token: Secret):
            return open_mcp_session(replace(app, mcp_url=app_url), token)

        async with (
            self.app.serving() as app_url,
            httpx2.AsyncClient(transport=realm.transport(), event_hooks={"request": [note_request]}) as token_client,
        ):
            yield SweepingBoundaries(
                tokens=HttpTokenSource(config.token_endpoint, token_client),
                open_session=open_at_the_app,
                clock=FakeClock(),
                report=self._report,
                ready=ready,
                backend=gateway_backend,
            )

    def start(self) -> None:
        main(self.compose)


@pytest.fixture
def world(files: Files, monkeypatch: pytest.MonkeyPatch) -> World:
    files.write()
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.config_path))
    return World(files)


def test_a_marker_left_from_an_earlier_run_is_gone_before_the_first_token_request(world):
    world.files.ready_path.touch()

    with pytest.raises(ExceptionGroup):
        world.start()

    assert world.tokens_requested_with_marker[0] is False


def test_the_marker_is_created_when_a_loop_is_served_and_stays(world):
    with pytest.raises(ExceptionGroup):
        world.start()

    assert world.files.ready_path.exists()


def test_a_marker_that_cannot_be_removed_stops_the_start_and_names_its_path(world, capsys):
    world.files.ready_path.mkdir()
    (world.files.ready_path / "held").touch()

    with pytest.raises(SystemExit) as stopped:
        world.start()

    assert stopped.value.code == READINESS_ERROR_EXIT_CODE
    assert str(world.files.ready_path) in capsys.readouterr().err
    assert world.tokens_requested_with_marker == []


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


def test_an_addressed_line_gets_one_say_through_the_entry_point_and_nothing_secret_or_said_reaches_the_log(
    world, library_loggers, caplog
):
    answer = world.files.persona_prompt

    def addressed_once_joined() -> None:
        world.app.stand(AGENT_SUBJECT, active("room-7"))
        world.app.speak("room-7", "visitor-subject", "Ada", "Mozart, are you there?")

    world.on_joined = addressed_once_joined
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ExceptionGroup) as ended:
        world.start()

    assert [type(leaf) for leaf in leaves_of(ended.value)] == [Stop]
    assert world.app.said == [(AGENT_SUBJECT, "room-7", answer)]
    logged = formatted(caplog.get_records("call"))
    assert "started" in logged
    for held in (world.files.secret, world.files.gateway_key, *world.issued, *world.app.kept_out_of_logs()):
        assert held not in logged


@pytest.mark.parametrize("binds_tools", [True, False], ids=["tools bound", "lifecycle only"])
def test_the_real_boundaries_run_turns_through_the_gateway_exactly_when_the_configuration_has_one(
    files, library_loggers, binds_tools
):
    files.text = files.valid_text(binds_tools=binds_tools)
    config = load_config(files.write())

    async def composed_backend() -> object:
        async with real_boundaries(config, FileReadyMarker(config.ready_file)) as boundaries:
            return None if boundaries.backend is None else boundaries.backend(())

    backend = asyncio.run(composed_backend())

    assert isinstance(backend, GatewayBackend) is binds_tools


@pytest.mark.parametrize("has_memory", [True, False], ids=["memory table", "no memory table"])
def test_the_real_boundaries_hold_a_store_exactly_when_the_configuration_has_a_memory_table(files, has_memory):
    if has_memory:
        files.text += files.memory_table()
    config = load_config(files.write())

    async def composed_store() -> tuple[object, object]:
        async with real_boundaries(config, FileReadyMarker(config.ready_file)) as boundaries:
            return boundaries.lines, boundaries.swept

    store, swept = asyncio.run(composed_store())

    assert isinstance(store, PostgresLineStore) is has_memory
    assert (swept is store) if has_memory else (swept is None)
