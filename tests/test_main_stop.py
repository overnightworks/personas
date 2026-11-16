"""The console script's stop: a SIGTERM ends a runner that is backing off with exit code 0."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import personas
from tests.support.configfiles import Files

if TYPE_CHECKING:
    from typing import TextIO

    from personas.config import DatabaseTarget

WAIT_SECONDS = 30
BACKOFF_EVENT = "loop.backed_off"
ENTRY_POINT_CALL = "from personas.main import main; main()"
"""What the console script `personas` runs; the test environment does not install the project's scripts."""
FAST_OWNERSHIP_TABLE = (
    "[ownership]\ncontend_seconds = 0.2\nheartbeat_timeout_seconds = 1.0\ncancel_margin_seconds = 0.5\n"
    "grace_seconds = 2.0\n"
)
"""The runner holds its agents only after the grace, which the default would stretch this test by."""
REALM_TOKEN_ENDPOINT = "https://realm.test/protocol/openid-connect/token"


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
