"""Runner pods as separate processes: the app, realm and gateway served on loopback for them, and a handle on each."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
from agents.backend.fake_gateway import completion

import personas
from personas.config import CONFIG_ENVIRONMENT_VARIABLE, load_config
from tests.support.configfiles import GATEWAY_ORIGIN, Files
from tests.support.gateway import Gateway
from tests.support.hopin import AGENT_SUBJECT, CALL, FakeApp
from tests.support.loopback import serving
from tests.support.realm import TOKEN_ENDPOINT, FakeRealm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from typing import Any

    from agents.backend.fake_gateway import Reply

    from personas.config import DatabaseTarget

SCOPE = "mcp-hopin"
SECOND_AGENT = "agent-ada"
SECOND_SUBJECT = "ada-subject"
ENTRY_POINT_CALL = "from personas.main import command_line; command_line()"
"""What the console script `personas` runs; the test environment does not install the project's scripts."""
POLL_SECONDS = 0.05
PROCESS_LIMIT_SECONDS = 60
"""A process imports the runner and talks to Postgres before it logs anything, which a loaded machine slows."""


class TracedApp(FakeApp):
    """The app, which notes which bearer made each call of a tool."""

    async def _answer(
        self, scope: dict[str, Any], receive: Any, send: Any, body: bytes, document: dict[str, Any]
    ) -> None:
        if document.get("method") == CALL:
            headers = {name.decode(): value.decode() for name, value in scope["headers"]}
            bearer = headers.get("authorization", "").removeprefix("Bearer ")
            self.calls_by.append((bearer, (document.get("params") or {}).get("name", "")))
        await super()._answer(scope, receive, send, body, document)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.calls_by: list[tuple[str, str]] = []
        """The bearer and the tool of each call, in the order the calls arrived."""


class LabelledRealm(FakeRealm):
    """The realm, which notes the pod each token was issued to: the first segment of the path it was asked on."""

    def __init__(self, *arguments: Any, **keywords: Any) -> None:
        super().__init__(*arguments, **keywords)
        self.pod_of_token: dict[str, str] = {}

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        response = super()._answer(request)
        if response.status_code == httpx2.codes.OK:
            self.pod_of_token[json.loads(response.content)["access_token"]] = request.url.path.split("/")[1]
        return response


async def _read(receive: Any) -> bytes:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            return body


class _Hop(httpx2.AsyncBaseTransport):
    """A transport that hands each request to whatever its target is at that moment."""

    def __init__(self, target: httpx2.AsyncBaseTransport) -> None:
        self.target = target

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self.target.handle_async_request(request)


def _over_http(transport: httpx2.AsyncBaseTransport, origin: str) -> Any:
    """An ASGI app that answers each request through an in-process transport."""

    async def answer(scope: dict[str, Any], receive: Any, send: Any) -> None:
        request = httpx2.Request(
            scope["method"],
            f"{origin}{scope['path']}",
            headers=[(name.decode(), value.decode()) for name, value in scope["headers"]],
            content=await _read(receive),
        )
        response = await transport.handle_async_request(request)
        content = await response.aread()
        await send({"type": "http.response.start", "status": response.status_code, "headers": response.headers.raw})
        await send({"type": "http.response.body", "body": content})

    return answer


@dataclass
class Runner:
    """One runner process: what it logged and when, and the ways to end it."""

    label: str
    process: subprocess.Popen[str]
    ready_file: Path
    logged: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            with self._lock:
                self.logged.append((time.monotonic(), json.loads(line)))

    def lines(self, event: str, agent_id: str | None = None) -> list[tuple[float, dict[str, Any]]]:
        with self._lock:
            return [
                (at, line)
                for at, line in self.logged
                if line["event"] == event and agent_id in (None, line.get("agent_id"))
            ]

    def is_ready(self) -> bool:
        return self.ready_file.exists()

    async def kill(self) -> float:
        """SIGKILLs the process and returns the monotonic time it was sent at."""
        sent = time.monotonic()
        self.process.kill()
        await asyncio.to_thread(self.process.wait)
        return sent

    async def terminate(self) -> int:
        """SIGTERMs the process and returns its exit code; the app and the gateway it may still ask answer meanwhile."""
        self.process.send_signal(signal.SIGTERM)
        return await asyncio.to_thread(self.process.wait, PROCESS_LIMIT_SECONDS)


async def until(condition: Callable[[], bool], why: str) -> None:
    """Returns once the condition holds; fails after the limit, naming what was waited for."""
    try:
        async with asyncio.timeout(PROCESS_LIMIT_SECONDS):
            while not condition():  # noqa: ASYNC110 - what is waited for happens in other processes, read on threads
                await asyncio.sleep(POLL_SECONDS)
    except TimeoutError:
        raise AssertionError(f"never happened: {why}") from None


@dataclass
class Pods:
    """The world of the pods: one app, one realm and one gateway, one database, and the runners started against it."""

    files: Files
    database: DatabaseTarget
    app: TracedApp
    realm: LabelledRealm
    gateway: Gateway
    gateway_hop: _Hop
    realm_origin: str
    gateway_origin: str
    ownership_table: str
    runners: dict[str, Runner] = field(default_factory=dict)

    def start(self, label: str) -> Runner:
        directory = self.files.config_path.parent / label
        directory.mkdir()
        ready_file = directory / "ready"
        text = (
            self.files.text.replace(TOKEN_ENDPOINT, f"{self.realm_origin}/{label}/token")
            .replace(GATEWAY_ORIGIN, self.gateway_origin)
            .replace(str(self.files.ready_path), str(ready_file))
        )
        configuration = directory / "personas.toml"
        configuration.write_text(text + self.ownership_table, encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603 - the interpreter running the tests
            [sys.executable, "-c", ENTRY_POINT_CALL],
            env=_environment(configuration),
            stdout=subprocess.PIPE,
            stderr=(directory / "stderr").open("w"),
            text=True,
        )
        self.runners[label] = Runner(label, process, ready_file)
        return self.runners[label]

    async def command(self, label: str, *arguments: str) -> int:
        """Runs a command of ``personas`` with the configuration of the pod started as ``label``; its exit code."""
        configuration = self.files.config_path.parent / label / "personas.toml"
        process = await asyncio.to_thread(
            subprocess.run,  # noqa: S603 - the interpreter running the tests
            [sys.executable, "-c", ENTRY_POINT_CALL, *arguments],
            env=_environment(configuration),
            capture_output=True,
            timeout=PROCESS_LIMIT_SECONDS,
            check=False,
        )
        return process.returncode

    def script_gateway(self, *replies: Reply) -> None:
        """The gateway answers from now on with these replies, in turn, the last one repeating."""
        self.gateway = Gateway(*replies)
        self.gateway_hop.target = self.gateway

    def pod_of(self, bearer: str) -> str:
        return self.realm.pod_of_token[bearer]

    def calls_of(self, label: str, tool: str) -> int:
        """How many calls of the tool the pod made."""
        return sum(1 for bearer, name in self.app.calls_by if name == tool and self.pod_of(bearer) == label)

    def stop_all(self) -> None:
        for runner in self.runners.values():
            if runner.process.poll() is None:
                runner.process.kill()
                runner.process.wait()


def _environment(configuration: Path) -> dict[str, str]:
    return {
        **os.environ,
        CONFIG_ENVIRONMENT_VARIABLE: str(configuration),
        "PYTHONPATH": str(Path(personas.__file__).parent.parent),
    }


@contextmanager
def _stopped_at_the_end(pods: Pods) -> Iterator[Pods]:
    try:
        yield pods
    finally:
        pods.stop_all()


@asynccontextmanager
async def serving_pods(files: Files, database: DatabaseTarget, ownership_table: str) -> AsyncIterator[Pods]:
    """The pods' world, served until the end; the processes started in it are killed then."""
    files.text += (
        f'[[agents]]\nclient_id = "{SECOND_AGENT}"\nsecret_file = "{files.secret_path}"\n'
        f'persona_file = "{files.persona_path}"\n'
    )
    config = load_config(files.write())
    app = TracedApp()
    realm = LabelledRealm(app, config.agents[0], AGENT_SUBJECT, SCOPE)
    realm.add_client(config.agents[1], SECOND_SUBJECT)
    gateway = Gateway(completion("done"))
    hop = _Hop(gateway)
    async with (
        app.serving() as app_url,
        serving(_over_http(realm.transport(), "http://realm.test")) as realm_origin,
        serving(_over_http(hop, "http://gateway.test")) as gateway_origin,
    ):
        files.text = files.text.replace("https://hopin.test/mcp", app_url)
        pods = Pods(files, database, app, realm, gateway, hop, realm_origin, gateway_origin, ownership_table)
        with _stopped_at_the_end(pods):
            yield pods
