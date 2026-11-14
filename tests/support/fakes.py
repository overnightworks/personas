"""Scripted stand-ins for the runner's boundaries, for tests that judge the loop alone."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

from agents.member import MemberStanding, PlaceEntry, PlaceWindow, Standing

from personas.config import AgentCredentials, AppTarget, MemoryTarget
from personas.decision import Accept, Refusal
from personas.failures import CallFailure
from personas.reporting import Event
from personas.secret import Secret
from personas.tokens import GrantedToken
from tests.support.hopin import GENERATION, ISSUER
from tests.support.realm import AGENT_NAME, API_GROUP, jwt_of

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from personas.reporting import StateChange


WALL_START = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)


class FakeClock:
    """Time that moves only when told, and sleeps that return at once."""

    def __init__(self) -> None:
        self.time = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.time

    def wall(self) -> datetime:
        return WALL_START + timedelta(seconds=self.time)

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.time += seconds
        await asyncio.sleep(0)


class SweepClock(FakeClock):
    """A fake clock whose sleep of one length waits for ``open_gate``; every other sleep returns at once."""

    def __init__(self, gated_seconds: float) -> None:
        super().__init__()
        self._gated_seconds = gated_seconds
        self._waiting = 0
        self._arrived = asyncio.Event()
        self._opened = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        if seconds != self._gated_seconds:
            await super().sleep(seconds)
            return
        self._waiting += 1
        self._arrived.set()
        await self._opened.wait()
        self._opened.clear()
        self.time += seconds

    async def until_waiting(self, count: int, *, limit_seconds: float = 10.0) -> None:
        """Returns once the gated sleep has been entered this many times."""
        async with asyncio.timeout(limit_seconds):
            while self._waiting < count:
                self._arrived.clear()
                await self._arrived.wait()

    def open_gate(self) -> None:
        self._opened.set()


def credentials(client_id: str = "agent-mozart") -> AgentCredentials:
    return AgentCredentials(client_id, Secret(secrets.token_hex(16)))


def memory_target() -> MemoryTarget:
    """A memory table naming a database no test reaches, for a loop whose store is scripted."""
    return MemoryTarget("db.test", 5432, "personas", "mozart", Secret(secrets.token_hex(16)))


def app_target() -> AppTarget:
    return AppTarget("https://app.test/mcp", "mcp-app")


def read_of(*standings: Standing) -> MemberStanding:
    return MemberStanding(standings=standings)


def line(
    seq: int, text: str, *, own: bool = False, spoken: bool = False, act: bool = False, name: str = "Ada"
) -> PlaceEntry:
    return PlaceEntry(seq=seq, kind="message", name=name, text=text, own=own, spoken=spoken, cut=False, act=act)


def window_of(*lines: PlaceEntry, generation: str = GENERATION) -> PlaceWindow:
    return PlaceWindow(generation=generation, starts_at=lines[0].seq if lines else 1, entries=lines)


def grant(lifetime: float = 300.0) -> GrantedToken:
    return GrantedToken(Secret(secrets.token_hex(16)), lifetime)


def jwt_grant(lifetime: float = 300.0) -> GrantedToken:
    """A grant of a token that carries the facts of an agent named Mozart."""
    claims = {"iss": ISSUER, "sub": secrets.token_hex(8), "groups": [API_GROUP], "agent_name": AGENT_NAME}
    return GrantedToken(Secret(jwt_of(claims)), lifetime)


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


class _Wakes:
    """One wake for each read still scripted, then silence: the scenario has reached its end."""

    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def __aiter__(self) -> _Wakes:
        return self

    async def __anext__(self) -> None:
        if self._session.graceful_ends:
            self._session.graceful_ends -= 1
            raise StopAsyncIteration
        if self._session.reads:
            return
        self._session.idle.set()
        await asyncio.Event().wait()


@dataclass
class FakeSession:
    """Answers each read and accept from a script and notes their bearer and time; blocking ends a scenario.

    The first ``graceful_ends`` streams end at once, as an app ends a stream at its bound.
    """

    clock: FakeClock
    reads: list[MemberStanding | Exception]
    accepts: list[Refusal | CallFailure | None] = field(default_factory=list[Refusal | CallFailure | None])
    seconds_per_read: float = 0.0
    seconds_per_accept: float = 0.0
    graceful_ends: int = 0
    bearer: Secret | None = None
    entered_with_time: list[tuple[Secret, float]] = field(default_factory=list[tuple[Secret, float]])
    opened_with: list[Secret] = field(default_factory=list[Secret])
    accepted: list[Accept] = field(default_factory=list[Accept])
    bearers_with_time: list[tuple[Secret, float]] = field(default_factory=list[tuple[Secret, float]])
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    def use_token(self, token: Secret) -> None:
        self.bearer = token

    @property
    def opened(self) -> int:
        return len(self.opened_with)

    @asynccontextmanager
    async def changes(self) -> AsyncIterator[AsyncIterator[object]]:
        assert self.bearer is not None
        self.entered_with_time.append((self.bearer, self.clock.now()))
        yield _Wakes(self)

    async def read_standing(self) -> MemberStanding:
        assert self.bearer is not None
        self.bearers_with_time.append((self.bearer, self.clock.now()))
        if not self.reads:
            self.idle.set()
            await asyncio.Event().wait()
        self.clock.time += self.seconds_per_read
        step = self.reads.pop(0)
        if isinstance(step, Exception):
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
        self.opened_with.append(token)
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


class Report(NamedTuple):
    """One reported state change as a scenario reads it: its event, its code and the address of the app it names."""

    event: Event
    code: str | None
    app: str | None


class Reports:
    """The state changes a loop reported, in order, and a way to wait for one.

    A turn that ended with its final event is reported with no code; ``reported`` and ``until`` leave it out, so a
    scenario reads what went wrong or was cut short, and ``until_a_turn_completed`` waits for it.
    """

    def __init__(self) -> None:
        self.changes: list[StateChange] = []
        self._added = asyncio.Event()

    def __call__(self, change: StateChange) -> None:
        self.changes.append(change)
        self._added.set()

    @staticmethod
    def _completes_a_turn(change: StateChange) -> bool:
        return change.event is Event.TURN_ENDED and change.code is None

    @property
    def without_completed_turns(self) -> list[StateChange]:
        return [change for change in self.changes if not self._completes_a_turn(change)]

    @overload
    def reported(self, *, apps: Literal[False] = False, started: bool = True) -> list[tuple[Event, str | None]]: ...

    @overload
    def reported(self, *, apps: Literal[True], started: bool = True) -> list[Report]: ...

    def reported(self, *, apps: bool = False, started: bool = True) -> list[tuple[Event, str | None]] | list[Report]:
        """Each reported change as its event and code, with the address of the app it names when ``apps``."""
        reports = [
            Report(change.event, change.code, change.app)
            for change in self.without_completed_turns
            if started or change.event is not Event.STARTED
        ]
        return reports if apps else [(each.event, each.code) for each in reports]

    def of_the_sweep(self) -> list[tuple[Event, str | None]]:
        """The changes that name no agent, the sweep's own, as their event and code."""
        return [(change.event, change.code) for change in self.without_completed_turns if change.client_id is None]

    def lines_removed(self) -> list[int | None]:
        """The line count of each sweep that finished."""
        return [change.lines for change in self.changes if change.event is Event.SWEPT]

    @property
    def is_empty(self) -> bool:
        return not self.changes

    def changes_of(self, event: Event) -> list[StateChange]:
        """Each change of this event."""
        return [change for change in self.changes if change.event is event]

    def codes_of(self, event: Event) -> list[str | None]:
        """The code of each change of this event."""
        return [change.code for change in self.changes if change.event is event]

    async def until(self, event: Event, *, count: int = 1, limit_seconds: float = 10.0) -> None:
        """Returns once this many changes of this event have been reported; fails after the limit."""
        async with asyncio.timeout(limit_seconds):
            while sum(change.event is event for change in self.without_completed_turns) < count:
                self._added.clear()
                await self._added.wait()

    async def until_a_turn_completed(self, *, count: int = 1, limit_seconds: float = 10.0) -> None:
        async with asyncio.timeout(limit_seconds):
            while sum(self._completes_a_turn(change) for change in self.changes) < count:
                self._added.clear()
                await self._added.wait()
