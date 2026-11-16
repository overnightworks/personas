"""Which pod serves which agent: a pod serves an agent only while it holds the agent's lock, and only after a grace.

Several pods run side by side against one database. Each holds one connection of its own and, on it, one lock per
agent it serves, so a pod that dies frees its agents at once. A pod that cannot confirm its locks stops serving, and a
pod that takes an agent waits longer than the pod it takes over from needs to notice and stop, so no two pods speak
for one agent. ``config.py`` refuses a grace that is not longer than that.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from personas.failures import STORE_UNREACHABLE_CODE
from personas.reporting import Event, StateChange, StoreOutage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence
    from typing import NoReturn

    from personas.clock import Clock
    from personas.config import OwnershipTimings
    from personas.reporting import Reporter

type ServeAgent = Callable[[str], Awaitable[NoReturn]]
"""Runs everything one agent does, until it is cancelled."""

LOST_HOLD_CODE = "hold_lost"
"""The code of an agent whose lock this pod no longer holds."""


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
    loops and the turns under them and waits until they ended.
    """

    def __init__(self, locks: AgentLocks, clock: Clock, timings: OwnershipTimings, report: Reporter) -> None:
        self._locks = locks
        self._clock = clock
        self._timings = timings
        self._report = report

    async def serve(
        self, client_ids: Sequence[str], serve_agent: ServeAgent, contending: Callable[[], None]
    ) -> NoReturn:
        serving: dict[str, asyncio.Task[None]] = {}
        outage = StoreOutage(lambda event, code: self._report(StateChange(event, None, None, code)))
        contended = False
        async with asyncio.TaskGroup() as group:
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
                await self._stop(serving, [each for each in serving if each not in held])
                for each in client_ids:
                    if each in held and each not in serving:
                        self._report(StateChange(Event.TAKEN, each, None))
                        serving[each] = group.create_task(self._serve_after_grace(each, serve_agent))
                await self._clock.sleep(max(0.0, started + self._timings.contend_seconds - self._clock.now()))

    async def _serve_after_grace(self, client_id: str, serve_agent: ServeAgent) -> None:
        await self._clock.sleep(self._timings.grace_seconds)
        await serve_agent(client_id)

    async def _stop(self, serving: dict[str, asyncio.Task[None]], client_ids: Iterable[str]) -> None:
        """Cancels the agents and waits until their loops and turns ended; the failure of one is the group's."""
        stopped = {each: serving.pop(each) for each in client_ids}
        for task in stopped.values():
            task.cancel()
        if stopped:
            await asyncio.wait(stopped.values())
        for each in stopped:
            self._report(StateChange(Event.LOST, each, None, LOST_HOLD_CODE))
