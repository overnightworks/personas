"""The longest duration of one turn, which every wait of it ends at."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.turn import FailureCode

from personas.decision import remaining_seconds

if TYPE_CHECKING:
    from personas.clock import Clock


class TurnBound:
    """The one bound of every wait of a turn: its longest duration from its start. A wait that has not ended by then
    ends the turn as ``limit_exceeded``, a tie going to the bound. The sleep it waits on is made once, and ``close``
    ends it with the turn.
    """

    def __init__(self, clock: Clock, max_duration_seconds: float) -> None:
        self._clock = clock
        self._started = clock.now()
        self._max_duration_seconds = max_duration_seconds
        self._deadline: asyncio.Task[None] | None = None

    def seconds_left(self) -> float:
        return remaining_seconds(self._clock.now(), self._started, self._max_duration_seconds)

    def ends_at(self) -> float:
        """When the turn's longest duration ends, on the clock's monotonic scale."""
        return self._started + self._max_duration_seconds

    def deadline(self) -> asyncio.Task[None]:
        """The task that ends when the turn's duration has passed, made once for the whole turn."""
        if self._deadline is None:
            self._deadline = asyncio.create_task(self._clock.sleep(self.seconds_left()))
        return self._deadline

    def stop_if_deadline_done(self) -> None:
        """Raises the turn's end when its deadline task has run out, which a wait that has not made one yet cannot."""
        if self._deadline is not None and self._deadline.done():
            self._deadline.result()
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)

    def close(self) -> None:
        """The turn is over: its sleep ends."""
        if self._deadline is not None:
            self._deadline.cancel()

    @staticmethod
    async def first_of(taken: asyncio.Event, *ends: asyncio.Task[object]) -> None:
        """Returns once ``taken`` is set or one of ``ends`` is done, whichever comes first."""
        woken = asyncio.create_task(taken.wait())
        try:
            await asyncio.wait({*ends, woken}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            woken.cancel()
