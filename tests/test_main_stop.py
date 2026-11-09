"""The console script's stop: a SIGTERM ends a runner that is backing off with exit code 0."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from tests.support.configfiles import Files

if TYPE_CHECKING:
    from typing import TextIO

WAIT_SECONDS = 30
REALM_TOKEN_ENDPOINT = "https://realm.test/protocol/openid-connect/token"


def _refused_token_endpoint() -> str:
    """An endpoint on a port nothing listens on, so every token call is refused."""
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{unused.getsockname()[1]}/token"


def _flag_when_logged(stream: TextIO, needle: str, seen: threading.Event) -> None:
    for line in stream:
        if needle in line:
            seen.set()


def test_a_sigterm_ends_a_backing_off_runner_with_exit_code_0(tmp_path: Path):
    files = Files(tmp_path)
    files.text = files.text.replace(REALM_TOKEN_ENDPOINT, _refused_token_endpoint())
    console_script = shutil.which("agents", path=str(Path(sys.executable).parent))
    assert console_script is not None
    runner = subprocess.Popen(  # noqa: S603 - the repository's own console script
        [console_script], env={**os.environ, **files.write()}, stderr=subprocess.PIPE, text=True
    )
    assert runner.stderr is not None
    backing_off = threading.Event()
    watcher = threading.Thread(target=_flag_when_logged, args=(runner.stderr, "backoff", backing_off), daemon=True)
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
