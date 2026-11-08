"""Scripted stand-ins for the runner's boundaries, for tests that judge the loop alone."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_runtime_client.wait import Standing, WaitAnswer

from agents.config import AgentCredentials, AppTarget
from agents.decision import Accept, Refusal
from agents.failures import CallFailure
from agents.secret import Secret
from agents.tokens import GrantedToken

if TYPE_CHECKING:
    from agents.reporting import StateChange


class FakeClock:
    """Time that moves only when told, and sleeps that return at once."""

    def __init__(self) -> None:
        self.time = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.time

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.time += seconds
        await asyncio.sleep(0)


def credentials(client_id: str = "agent-mozart") -> AgentCredentials:
    return AgentCredentials(client_id, Secret(secrets.token_hex(16)))


def app_target() -> AppTarget:
    return AppTarget("https://app.test/mcp", "mcp-app")


def answer(cursor: str, *standings: Standing) -> WaitAnswer:
    return WaitAnswer(cursor=cursor, standings=standings)


def grant(lifetime: float = 300.0) -> GrantedToken:
    return GrantedToken(Secret(secrets.token_hex(16)), lifetime)


@dataclass
class FakeTokens:
    """Hands out scripted grants or failures, then blocks, which ends a test's scenario."""

    clock: FakeClock
    script: list[GrantedToken | CallFailure]
    fetched: int = 0
    expires_at: dict[Secret, float] = field(default_factory=dict[Secret, float])
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    async def fetch(self, agent: AgentCredentials, scope: str) -> GrantedToken:
        if not self.script:
            self.idle.set()
            await asyncio.Event().wait()
        self.fetched += 1
        step = self.script.pop(0)
        if isinstance(step, CallFailure):
            raise step
        self.expires_at[step.value] = self.clock.now() + step.lifetime_seconds
        return step


@dataclass
class FakeSession:
    """Answers each wait and accept from a script and notes their bearer and time; blocking ends a scenario."""

    clock: FakeClock
    waits: list[WaitAnswer | CallFailure]
    accepts: list[Refusal | CallFailure | None] = field(default_factory=list[Refusal | CallFailure | None])
    seconds_per_wait: float = 0.0
    seconds_per_accept: float = 0.0
    bearer: Secret | None = None
    opened: int = 0
    cursors: list[str | None] = field(default_factory=list[str | None])
    accepted: list[Accept] = field(default_factory=list[Accept])
    bearers_with_time: list[tuple[Secret, float]] = field(default_factory=list[tuple[Secret, float]])
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    def use_token(self, token: Secret) -> None:
        self.bearer = token

    async def wait(self, after: str | None) -> WaitAnswer:
        assert self.bearer is not None
        self.bearers_with_time.append((self.bearer, self.clock.now()))
        self.cursors.append(after)
        if not self.waits:
            self.idle.set()
            await asyncio.Event().wait()
        self.clock.time += self.seconds_per_wait
        step = self.waits.pop(0)
        if isinstance(step, CallFailure):
            raise step
        return step

    async def accept(self, accept: Accept) -> Refusal | None:
        assert self.bearer is not None
        self.bearers_with_time.append((self.bearer, self.clock.now()))
        self.clock.time += self.seconds_per_accept
        self.accepted.append(accept)
        step = self.accepts.pop(0) if self.accepts else None
        if isinstance(step, CallFailure):
            raise step
        return step

    @asynccontextmanager
    async def open(self, app: AppTarget, token: Secret):
        self.opened += 1
        self.bearer = token
        yield self


class FakeReadyMarker:
    """Notes how often the marker was asked for; the marker itself is a file the real adapter owns."""

    def __init__(self) -> None:
        self.cleared = 0
        self.announced = 0

    def clear(self) -> None:
        self.cleared += 1

    def announce(self) -> None:
        self.announced += 1


class Reports:
    """The state changes a loop reported, in order, and a way to wait for one."""

    def __init__(self) -> None:
        self.changes: list[StateChange] = []
        self._added = asyncio.Event()

    def __call__(self, change: StateChange) -> None:
        self.changes.append(change)
        self._added.set()

    def events(self) -> list[tuple[str, str | None]]:
        return [(change.event.value, change.code) for change in self.changes]

    async def until(self, event: str, *, limit_seconds: float = 10.0) -> None:
        """Returns once a change of this event has been reported; fails after the limit."""
        async with asyncio.timeout(limit_seconds):
            while not any(change.event.value == event for change in self.changes):
                self._added.clear()
                await self._added.wait()
