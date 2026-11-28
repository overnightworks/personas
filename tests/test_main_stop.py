"""The console script's stop: a SIGTERM ends a runner that is backing off with exit code 0, and a stop that outlasts
its deadline is logged and ends the process with its own exit code."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

import personas
from personas.config import CONFIG_ENVIRONMENT_VARIABLE, OwnershipTimings
from personas.failures import STORE_UNREACHABLE_CODE
from personas.main import STOP_CUT_EXIT_CODE, main
from tests.support.configfiles import Files
from tests.support.pods import SECOND_AGENT, serving_pods, until
from tests.support.postgres import Forwarder

if TYPE_CHECKING:
    from typing import TextIO

    from personas.config import Config, DatabaseTarget
    from personas.main import SweepingBoundaries
    from personas.readiness import ReadyMarker

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

WAIT_SECONDS = 30
BACKOFF_EVENT = "loop.backed_off"
STOP_CUT_EVENT = "runner.stop_cut"
ENTRY_POINT_CALL = "from personas.main import main; main()"
"""What the console script `personas` runs; the test environment does not install the project's scripts."""
FAST_OWNERSHIP_TABLE = (
    "[ownership]\ncontend_seconds = 0.2\nheartbeat_timeout_seconds = 1.0\ncancel_margin_seconds = 0.5\n"
    "grace_seconds = 2.0\n"
)
"""The runner holds its agents only after the grace, which the default would stretch this test by."""
REALM_TOKEN_ENDPOINT = "https://realm.test/protocol/openid-connect/token"
AGENT = "agent-mozart"
TIMINGS = OwnershipTimings(
    contend_seconds=0.5, heartbeat_timeout_seconds=1.0, cancel_margin_seconds=0.5, grace_seconds=3.0, stop_seconds=1.0
)
SLACK_SECONDS = 3.0
"""What a loaded machine may add between a signal and the end of the process it reaches."""


def _refused_token_endpoint() -> str:
    """An endpoint on a port nothing listens on, so every token call is refused."""
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{unused.getsockname()[1]}/token"


def _flag_when_logged(stream: TextIO, event: str, seen: threading.Event) -> None:
    for line in stream:
        if json.loads(line)["event"] == event:
            seen.set()


@pytest.mark.postgres
def test_a_sigterm_ends_a_backing_off_runner_with_exit_code_0(tmp_path: Path, database: DatabaseTarget):
    files = Files(tmp_path)
    files.use_database(database)
    files.text = files.text.replace(REALM_TOKEN_ENDPOINT, _refused_token_endpoint()) + FAST_OWNERSHIP_TABLE
    environment = {**os.environ, **files.write(), "PYTHONPATH": str(Path(personas.__file__).parent.parent)}
    runner = subprocess.Popen(  # noqa: S603 - the interpreter running the tests
        [sys.executable, "-c", ENTRY_POINT_CALL], env=environment, stdout=subprocess.PIPE, text=True
    )
    assert runner.stdout is not None
    backing_off = threading.Event()
    watcher = threading.Thread(target=_flag_when_logged, args=(runner.stdout, BACKOFF_EVENT, backing_off), daemon=True)
    watcher.start()
    try:
        assert backing_off.wait(WAIT_SECONDS), "the runner never logged a backoff"
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(WAIT_SECONDS) == 0
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
        watcher.join(WAIT_SECONDS)


def _ownership_table(timings: OwnershipTimings) -> str:
    return "[ownership]\n" + "".join(f"{name} = {seconds}\n" for name, seconds in vars(timings).items())


@pytest.mark.postgres
async def test_a_sigterm_while_the_lock_connection_unwinds_from_a_dark_database_ends_the_runner_within_its_deadline(
    files: Files, database: DatabaseTarget
):
    forwarder = Forwarder(database)
    async with forwarder.serving() as through:
        files.use_database(through)
        async with serving_pods(files, through, _ownership_table(TIMINGS)) as pods:
            runner = pods.start("first")
            await until(
                lambda: bool(runner.lines("loop.started", AGENT) and runner.lines("loop.started", SECOND_AGENT)),
                "the runner serves both agents",
            )
            forwarder.go_dark()
            await until(
                lambda: any(line.get("reason") == STORE_UNREACHABLE_CODE for _, line in runner.lines("store.failed")),
                "a lock round was given up",
            )

            sent = time.monotonic()
            exit_code = await runner.terminate()

            assert time.monotonic() - sent <= TIMINGS.stop_seconds + SLACK_SECONDS
            assert exit_code == STOP_CUT_EXIT_CODE
            assert len(runner.lines(STOP_CUT_EVENT)) == 1


class _OpeningThatHangsOnceCancelled(AbstractAsyncContextManager["SweepingBoundaries"]):
    """Boundaries that send the runner a SIGTERM while they open, and whose unwinding from it never ends."""

    async def __aenter__(self) -> NoReturn:
        os.kill(os.getpid(), signal.SIGTERM)
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.Event().wait()
        raise AssertionError("the SIGTERM never arrived")

    async def __aexit__(self, *_: object) -> None:
        raise AssertionError("boundaries that never opened were closed")


def _hanging_once_cancelled(_config: Config, _ready: ReadyMarker) -> _OpeningThatHangsOnceCancelled:
    return _OpeningThatHangsOnceCancelled()


def _exit_raising(code: int) -> NoReturn:
    raise SystemExit(code)


def test_a_stop_that_outlasts_its_deadline_is_logged_once_and_ends_the_entry_point_with_its_exit_code(
    files: Files, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], restored_logging: None
):
    """CONFIG-32, LOG-25."""
    files.text += _ownership_table(TIMINGS)
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, files.write()[CONFIG_ENVIRONMENT_VARIABLE])
    monkeypatch.setattr(os, "_exit", _exit_raising)
    started = time.monotonic()

    with pytest.raises(SystemExit) as ended:
        main(_hanging_once_cancelled)

    assert time.monotonic() - started <= TIMINGS.stop_seconds + SLACK_SECONDS
    assert ended.value.code == STOP_CUT_EXIT_CODE
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events.count(STOP_CUT_EVENT) == 1
