"""Which pod serves which agent: a pod serves an agent only while it holds the agent's lock, and only after a grace.

Several pods run side by side against one database. Each holds one connection of its own and, on it, one lock per
agent it serves, so a pod that dies frees its agents at once. A pod that cannot confirm its locks stops serving, and a
pod that takes an agent waits longer than the pod it takes over from needs to notice and stop, so no two pods speak
for one agent. ``config.py`` refuses a grace that is not longer than that.

Taking an agent renews his generation, which fences every write of his history and his turns: a pod that wrote under an
older one has lost him, and stops serving him.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Protocol

from personas.failures import (
    STORE_DEFECT_CODE,
    STORE_UNREACHABLE_CODE,
    OrdinalReusedError,
    StaleGenerationError,
    StoreUnavailable,
)
from personas.reporting import Event, StateChange, StoreOutage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence
    from typing import NoReturn

    from personas.clock import Clock
    from personas.config import OwnershipTimings
    from personas.entries import HistoryDelta, KeptHistory, KeptTurns
    from personas.reporting import Reporter


LOST_HOLD_CODE = "hold_lost"
"""The code of an agent whose lock this pod no longer holds."""
STALE_GENERATION_CODE = "generation_stale"
"""The code of an agent whose history refused this pod's write: another owner has taken him since."""


class AgentHistories(Protocol):
    """Where each agent's history is kept, fenced by the generation of his owner."""

    async def take_over(self, client_id: str) -> int:
        """Renews the agent's generation and returns it."""
        ...

    async def load(self, client_id: str) -> KeptHistory:
        """The agent's history and turns as the store keeps them."""
        ...

    async def write(self, client_id: str, generation: int, delta: HistoryDelta | None, turns: KeptTurns) -> None:
        """Writes the delta, if any, and the turns in one transaction while ``generation`` is current, or raises
        ``StaleGenerationError``."""
        ...


@dataclass(frozen=True)
class Owned:
    """An agent this pod took over: his history and turns as the store kept them then, and the write of what they gain,
    fenced by the generation taken."""

    client_id: str
    history: KeptHistory
    write: Callable[[HistoryDelta | None, KeptTurns], Awaitable[None]]


async def taken_over(histories: AgentHistories, client_id: str) -> Owned:
    """Takes the agent over and loads his history after it, so every write before the takeover is in what loads."""
    generation = await histories.take_over(client_id)
    history = await histories.load(client_id)
    return Owned(client_id, history, partial(histories.write, client_id, generation))


type ServeAgent = Callable[[Owned], Awaitable[NoReturn]]
"""Runs everything one agent does, until it is cancelled or a write of his history is refused."""


class _Stop:
    """The moment an agent's stop began, from which the cancel margin counts, and whether a write of his is in
    flight: such a write is bounded and reported by the agent himself, so a slow store is not his stop's overrun."""

    def __init__(self) -> None:
        self.began = asyncio.Event()
        self._writes_in_flight = 0
        self._write_ended = asyncio.Event()

    @asynccontextmanager
    async def writing(self) -> AsyncGenerator[None]:
        self._writes_in_flight += 1
        try:
            yield
        finally:
            self._writes_in_flight -= 1
            self._write_ended.set()

    @property
    def write_in_flight(self) -> bool:
        return self._writes_in_flight > 0

    async def until_no_write_in_flight(self) -> None:
        while self._writes_in_flight:
            self._write_ended.clear()
            await self._write_ended.wait()


@dataclass(frozen=True)
class _Serving:
    """An agent this pod serves, and his stop."""

    task: asyncio.Task[None]
    stop: _Stop


class LockConnectionLost(Exception):  # noqa: N818 - named for the state of the connection
    """The lock connection ended or did not answer in time: the locks it held are gone or no longer known to be held."""


class AgentLocks(Protocol):
    """The pod's one lock connection."""

    async def hold(self, client_ids: Sequence[str]) -> frozenset[str]:
        """One bounded round: confirms the locks held, takes each free one and returns the agents held now.

        ``LockConnectionLost`` when the connection broke or did not answer in time; the next round opens a new one.
        """
        ...


class Ownership(Protocol):
    """What the runner asks to serve its agents."""

    async def serve(
        self, client_ids: Sequence[str], serve_agent: ServeAgent, contending: Callable[[], None]
    ) -> NoReturn:
        """Serves each agent while it is owned, calling ``contending`` once the first contention went through."""
        ...


class LockedOwnership:
    """Serves each agent its pod holds the lock of, after the grace; stops an agent the moment the hold is lost.

    Each round runs at ``contend_seconds`` from the start of the one before, so a hold that fails is noticed within
    ``contend_seconds`` plus the heartbeat timeout of the last round that confirmed it. Stopping cancels the agent's
    loops and the turns under them and waits until they ended; a stop that outlasts the cancel margin is reported,
    since the grace of a taker counts on it, whether the hold was lost, a write was refused or the pod stops. An agent
    whose takeover failed or whose write was refused stops, and is taken again in the next round that holds him.
    """

    def __init__(
        self,
        locks: AgentLocks,
        histories: AgentHistories,
        clock: Clock,
        timings: OwnershipTimings,
        report: Reporter,
    ) -> None:
        self._locks = locks
        self._histories = histories
        self._clock = clock
        self._timings = timings
        self._report = report

    async def serve(
        self, client_ids: Sequence[str], serve_agent: ServeAgent, contending: Callable[[], None]
    ) -> NoReturn:
        serving: dict[str, _Serving] = {}
        outage = StoreOutage(lambda event, code: self._report(StateChange(event, None, None, code)))
        contended = False
        async with asyncio.TaskGroup() as group:
            try:
                while True:
                    started = self._clock.now()
                    try:
                        held = await self._locks.hold(client_ids)
                    except LockConnectionLost:
                        outage.failed(STORE_UNREACHABLE_CODE)
                        held = frozenset[str]()
                    else:
                        outage.answered()
                        if not contended:
                            contended = True
                            contending()
                    for each in [each for each, agent in serving.items() if agent.task.done()]:
                        del serving[each]
                    await self._stop(serving, [each for each in serving if each not in held])
                    for each in client_ids:
                        if each in held and each not in serving:
                            self._report(StateChange(Event.TAKEN, each, None))
                            stop = _Stop()
                            task = group.create_task(self._serve_after_grace(each, serve_agent, stop))
                            serving[each] = _Serving(task, stop)
                    await self._clock.sleep(max(0.0, started + self._timings.contend_seconds - self._clock.now()))
            finally:
                await self._cancelled_until_ended(serving)

    async def _serve_after_grace(self, client_id: str, serve_agent: ServeAgent, stop: _Stop) -> None:
        overrun = asyncio.create_task(self._overran_after(client_id, stop))
        try:
            await self._clock.sleep(self._timings.grace_seconds)
            try:
                owned = await taken_over(self._histories, client_id)
            except StoreUnavailable as outage:
                self._report(StateChange(Event.LOST, client_id, None, outage.code))
                return
            try:
                await serve_agent(replace(owned, write=partial(_noting_refusal, owned.write, stop)))
            except* StaleGenerationError:
                self._report(StateChange(Event.LOST, client_id, None, STALE_GENERATION_CODE))
            except* OrdinalReusedError:
                self._report(StateChange(Event.LOST, client_id, None, STORE_DEFECT_CODE))
        finally:
            overrun.cancel()

    async def _overran_after(self, client_id: str, stop: _Stop) -> None:
        """Reports the agent once his stop outlasts the cancel margin, whatever began it and however many causes
        followed; cancelled once he ended.

        A write in flight when the margin ends is the store's delay, not his: he has the margin again once it ended,
        since the write often ends in a task of his whose end reaches him, and the flush of his stop, only later.
        """
        await stop.began.wait()
        await self._clock.sleep(self._timings.cancel_margin_seconds)
        while stop.write_in_flight:
            await stop.until_no_write_in_flight()
            await self._clock.sleep(self._timings.cancel_margin_seconds)
        self._report(StateChange(Event.STOP_OVERRAN, client_id, None))

    async def _stop(self, serving: dict[str, _Serving], client_ids: Iterable[str]) -> None:
        """Cancels the agents and waits until their loops and turns ended; the failure of one is the group's."""
        stopped = {each: serving.pop(each) for each in client_ids}
        await self._cancelled_until_ended(stopped)
        for each in stopped:
            self._report(StateChange(Event.LOST, each, None, LOST_HOLD_CODE))

    async def _cancelled_until_ended(self, agents: dict[str, _Serving]) -> None:
        """Begins each agent's stop, cancels him and waits until he ended."""
        for agent in agents.values():
            agent.stop.began.set()
            agent.task.cancel()
        if agents:
            await asyncio.wait([agent.task for agent in agents.values()])


async def _noting_refusal(
    write: Callable[[HistoryDelta | None, KeptTurns], Awaitable[None]],
    stop: _Stop,
    delta: HistoryDelta | None,
    turns: KeptTurns,
) -> None:
    """Writes the delta and the turns while ``stop`` knows it is in flight; a refusal begins the agent's stop."""
    try:
        async with stop.writing():
            await write(delta, turns)
    except StaleGenerationError, OrdinalReusedError:
        stop.began.set()
        raise
