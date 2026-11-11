"""The sweep: the task that removes the lines kept longer than the keep period, beside the loops."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from personas.decision import swept_before
from personas.failures import StoreUnavailable
from personas.reporting import Event, StateChange

if TYPE_CHECKING:
    from datetime import datetime

    from personas.clock import Clock
    from personas.config import MemoryTarget
    from personas.reporting import Reporter

SECONDS_PER_MINUTE = 60


class SweptStore(Protocol):
    """What the sweep asks of the store."""

    async def erase_before(self, first_read_before: datetime) -> int:
        """Deletes every line first read before the time and returns how many; ``StoreUnavailable`` when it cannot
        now."""
        ...


async def sweep_lines(store: SweptStore, memory: MemoryTarget, clock: Clock, report: Reporter) -> None:
    """Sweeps once and then after every ``sweep_minutes``; it ends when nothing is ever swept or after the first
    sweep when the minutes are infinite. A store that cannot be reached is reported once per outage."""
    failing = False
    while (cutoff := swept_before(clock.wall(), memory.lines_keep_days)) is not None:
        try:
            removed = await store.erase_before(cutoff)
        except StoreUnavailable as unavailable:
            if not failing:
                report(StateChange(Event.STORE_FAILED, None, None, unavailable.code))
            failing = True
        else:
            if failing:
                report(StateChange(Event.STORE_RESUMED, None, None))
            failing = False
            if removed:
                report(StateChange(Event.SWEPT, None, None, lines=removed))
        if math.isinf(memory.sweep_minutes):
            return
        await clock.sleep(memory.sweep_minutes * SECONDS_PER_MINUTE)
