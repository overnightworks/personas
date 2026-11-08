"""The entry point's logging: only the runner's own state changes reach a log line."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import httpx2
import pytest

from agents.app_session import open_mcp_session
from agents.config import CONFIG_ENVIRONMENT_VARIABLE, load_config
from agents.loop import Boundaries
from agents.main import QUIET_LIBRARIES, READINESS_ERROR_EXIT_CODE, configure_logging, main, real_boundaries
from agents.readiness import FileReadyMarker
from agents.reporting import Event
from agents.tokens import HttpTokenSource
from tests.support.fakes import FakeClock
from tests.support.hopin import AGENT_SUBJECT, FakeApp, invited
from tests.support.realm import FakeRealm

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from agents.config import AppTarget, Config
    from agents.readiness import ReadyMarker
    from agents.reporting import StateChange
    from agents.secret import Secret
    from tests.support.configfiles import Files


@pytest.mark.parametrize("library", QUIET_LIBRARIES)
def test_an_error_record_of_a_library_the_runner_uses_reaches_no_log_line(
    library, library_loggers, caplog: pytest.LogCaptureFixture
):
    configure_logging()
    caplog.set_level(logging.DEBUG)

    logging.getLogger(f"{library}.client.streamable_http").error("a response body")

    assert caplog.records == []


class Stop(Exception):  # noqa: N818 - a test's way to end a runner whose loops never end
    """Raised by the report once an agent joined, which is after its first read answered."""


class World:
    """The real entry point, loops and adapters over a fake realm and app that answer for real."""

    def __init__(self, files: Files) -> None:
        self.files = files
        self.app = FakeApp()
        self.tokens_requested_with_marker: list[bool] = []
        self.app.stand(AGENT_SUBJECT, invited("room-7", "inv-3"))

    def _report(self, change: StateChange) -> None:
        if change.event is Event.JOINED:
            raise Stop

    @asynccontextmanager
    async def compose(self, config: Config, ready: ReadyMarker) -> AsyncGenerator[Boundaries]:
        realm = FakeRealm(self.app, config.agents[0], AGENT_SUBJECT, config.apps[0].scope)

        async def note_request(request: httpx2.Request) -> None:
            self.tokens_requested_with_marker.append(self.files.ready_path.exists())

        def open_at_the_app(app: AppTarget, token: Secret):
            return open_mcp_session(replace(app, mcp_url=app_url), token)

        async with (
            self.app.serving() as app_url,
            httpx2.AsyncClient(transport=realm.transport(), event_hooks={"request": [note_request]}) as token_client,
        ):
            yield Boundaries(
                tokens=HttpTokenSource(config.token_endpoint, token_client),
                open_session=open_at_the_app,
                clock=FakeClock(),
                report=self._report,
                ready=ready,
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


def test_the_real_boundaries_hand_the_loops_the_marker_they_were_given(files):
    files.write()
    config = load_config({CONFIG_ENVIRONMENT_VARIABLE: str(files.config_path)})
    marker = FileReadyMarker(config.ready_file)

    async def boundaries_ready() -> ReadyMarker:
        async with real_boundaries(config, marker) as boundaries:
            return boundaries.ready

    assert asyncio.run(boundaries_ready()) is marker
