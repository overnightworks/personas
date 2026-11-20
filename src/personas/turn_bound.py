"""The longest duration of one turn, and the waits for the lines he still owes that end at it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.turn import FailureCode

from personas.decision import past_deadline, remaining_seconds

if TYPE_CHECKING:
    from collections.abc import Callable

    from personas.clock import Clock
    from personas.decision import AppPlace
    from personas.mind import Mind
    from personas.speech import OwedLines


class TurnBound:
    """The one bound of every wait of a turn: its longest duration from its start. A wait that has not ended by then
    ends the turn as ``limit_exceeded``, a tie going to the bound.

    A wait for the lines he still owes also ends at the time the turn that sent them was over, which may be an earlier
    turn: the lines are then given up and reported missing by the first wait to find them so. The sleeps it waits on are
    made once, and ``close`` ends them with the turn.
    """

    def __init__(
        self,
        clock: Clock,
        max_duration_seconds: float,
        mind: Mind,
        report_lines_missing: Callable[[int], None],
    ) -> None:
        self._clock = clock
        self._started = clock.now()
        self._max_duration_seconds = max_duration_seconds
        self._mind = mind
        self._report_lines_missing = report_lines_missing
        self._deadline: asyncio.Task[None] | None = None
        self._lapse: asyncio.Task[None] | None = None

    @property
    def ends_at(self) -> float:
        """The time on the clock's monotonic scale by which the turn is over."""
        return self._started + self._max_duration_seconds

    def seconds_left(self) -> float:
        return remaining_seconds(self._clock.now(), self._started, self._max_duration_seconds)

    def past(self) -> bool:
        return past_deadline(self._clock.now(), self._started, self._max_duration_seconds)

    def stop_if_past(self) -> None:
        if self.past():
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)

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
        """The turn is over: its sleeps end."""
        if self._deadline is not None:
            self._deadline.cancel()
        if self._lapse is not None:
            self._lapse.cancel()

    @staticmethod
    async def first_of(taken: asyncio.Event, *ends: asyncio.Task[object]) -> None:
        """Returns at the mind's next take or once one of ``ends`` is done, whichever comes first."""
        woken = asyncio.create_task(taken.wait())
        try:
            await asyncio.wait({*ends, woken}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            woken.cancel()

    async def until_lines(
        self, enough: Callable[[int], bool], at: AppPlace | None = None, *, held_ends: bool = False
    ) -> OwedLines:
        """Returns the lines he still owes at the place, at every place with none, once ``enough`` holds for their
        count, judged at every take of the mind; with ``held_ends`` also once a hold stands over one of them, which
        enters only after it is decided. The turn's bound ends the turn first when it passes before."""
        while True:
            taken = self._mind.next_take()
            self.stop_if_deadline_done()
            owed = self._mind.lines_owed(self._clock.now(), self.ends_at, at)
            if owed.dropped:
                self._report_lines_missing(owed.dropped)
            if enough(owed.count) or (held_ends and owed.held is not None):
                return owed
            self.stop_if_past()
            ends = [self.deadline()]
            if (lapse := self._lapse_of(owed.until)) is not None:
                ends.append(lapse)
            await self.first_of(taken, *ends)

    def _lapse_of(self, until: float | None) -> asyncio.Task[None] | None:
        """The sleep that ends when the turn that sent the lines is over, none when the turn's own bound comes first."""
        if until is None or until >= self.ends_at:
            return None
        if self._lapse is not None:
            self._lapse.cancel()
        self._lapse = asyncio.create_task(self._clock.sleep(max(until - self._clock.now(), 0.0)))
        return self._lapse
