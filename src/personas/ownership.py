"""Which pod serves which agent: a pod serves an agent only while it holds the agent's lock, and only after a grace.

Several pods run side by side against one database. Each holds one connection of its own and, on it, one lock per
agent it serves, so a pod that dies frees its agents at once. A pod that cannot confirm its locks stops serving, and a
pod that takes an agent waits longer than the pod it takes over from needs to notice and stop, so no two pods speak
for one agent. ``config.py`` refuses a grace that is not longer than that.

Taking an agent renews his generation, which fences every write of his history: a pod that wrote under an older one
has lost him, and stops serving him.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Protocol

from personas.failures import STORE_UNREACHABLE_CODE, StaleGenerationError, StoreUnavailable
from personas.reporting import Event, StateChange, StoreOutage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence
    from typing import NoReturn

    from personas.clock import Clock
    from personas.config import OwnershipTimings
    from personas.entries import HistoryDelta, KeptHistory
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
        """The agent's history as the store keeps it."""
        ...

    async def append(self, client_id: str, generation: int, delta: HistoryDelta) -> None:
        """Writes the delta while ``generation`` is current, or raises ``StaleGenerationError``."""
        ...


@dataclass(frozen=True)
class Owned:
    """An agent this pod took over: his history as the store kept it then, and the write of what it gains, fenced by
    the generation taken."""

    client_id: str
    history: KeptHistory
    write: Callable[[HistoryDelta], Awaitable[None]]


async def taken_over(histories: AgentHistories, client_id: str) -> Owned:
    """Takes the agent over and loads his history after it, so every write before the takeover is in what loads."""
    generation = await histories.take_over(client_id)
    history = await histories.load(client_id)
    return Owned(client_id, history, partial(histories.append, client_id, generation))


type ServeAgent = Callable[[Owned], Awaitable[NoReturn]]
"""Runs everything one agent does, until it is cancelled or a write of his history is refused."""


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
        serving: dict[str, asyncio.Task[None]] = {}
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
                    for each in [each for each, task in serving.items() if task.done()]:
                        del serving[each]
                    await self._stop(serving, [each for each in serving if each not in held])
                    for each in client_ids:
                        if each in held and each not in serving:
                            self._report(StateChange(Event.TAKEN, each, None))
                            serving[each] = group.create_task(self._serve_after_grace(each, serve_agent))
                    await self._clock.sleep(max(0.0, started + self._timings.contend_seconds - self._clock.now()))
            finally:
                await self._cancelled_until_ended(serving)

    async def _serve_after_grace(self, client_id: str, serve_agent: ServeAgent) -> None:
        await self._clock.sleep(self._timings.grace_seconds)
        try:
            owned = await taken_over(self._histories, client_id)
        except StoreUnavailable as outage:
            self._report(StateChange(Event.LOST, client_id, None, outage.code))
            return
        refused = asyncio.Event()
        overrun = asyncio.create_task(self._overran_after(client_id, refused))
        try:
            await serve_agent(replace(owned, write=partial(_noting_refusal, owned.write, refused)))
        except* StaleGenerationError:
            self._report(StateChange(Event.LOST, client_id, None, STALE_GENERATION_CODE))
        finally:
            overrun.cancel()

    async def _overran_after(self, client_id: str, refused: asyncio.Event) -> None:
        """Reports the agent once his stop after a refused write outlasts the cancel margin; cancelled once he ended."""
        await refused.wait()
        await self._clock.sleep(self._timings.cancel_margin_seconds)
        self._report(StateChange(Event.STOP_OVERRAN, client_id, None))

    async def _stop(self, serving: dict[str, asyncio.Task[None]], client_ids: Iterable[str]) -> None:
        """Cancels the agents and waits until their loops and turns ended; the failure of one is the group's."""
        stopped = {each: serving.pop(each) for each in client_ids}
        await self._cancelled_until_ended(stopped)
        for each in stopped:
            self._report(StateChange(Event.LOST, each, None, LOST_HOLD_CODE))

    async def _cancelled_until_ended(self, agents: dict[str, asyncio.Task[None]]) -> None:
        """Cancels the agents and waits until they ended, reporting each stop that overran."""
        for task in agents.values():
            task.cancel()
        if agents:
            await self._ended(agents)

    async def _ended(self, stopped: dict[str, asyncio.Task[None]]) -> None:
        """Waits until the stopped agents ended, reporting each that is still running once the cancel margin passed."""
        ended = asyncio.ensure_future(asyncio.wait(stopped.values()))
        margin = asyncio.ensure_future(self._clock.sleep(self._timings.cancel_margin_seconds))
        try:
            await asyncio.wait((ended, margin), return_when=asyncio.FIRST_COMPLETED)
            for each, task in stopped.items():
                if not task.done():
                    self._report(StateChange(Event.STOP_OVERRAN, each, None))
            await ended
        finally:
            margin.cancel()
            ended.cancel()


async def _noting_refusal(
    write: Callable[[HistoryDelta], Awaitable[None]], refused: asyncio.Event, delta: HistoryDelta
) -> None:
    """Writes the delta; a refusal sets ``refused``, since the agent's stop begins with it."""
    try:
        await write(delta)
    except StaleGenerationError:
        refused.set()
        raise
