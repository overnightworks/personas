"""The sweep: the task that removes the lines and the record's entries kept longer than the keep period, beside the
loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from personas.failures import StoreUnavailable
from personas.recall import SECONDS_PER_MINUTE, swept_before
from personas.reporting import Event, StateChange, StoreOutage

if TYPE_CHECKING:
    from datetime import datetime

    from personas.clock import Clock
    from personas.config import MemoryTarget
    from personas.reporting import Reporter


@dataclass(frozen=True)
class Swept:
    """What one sweep removed: lines, and entries of the record."""

    lines: int
    entries: int


class SweptStore(Protocol):
    """What the sweep asks of the store."""

    async def erase_before(self, first_read_before: datetime) -> Swept:
        """Deletes every line first read before the time and every entry of the record that entered before it and that
        no history holds, and returns how many; ``StoreUnavailable`` when it cannot now."""
        ...


async def sweep_lines(store: SweptStore, memory: MemoryTarget, clock: Clock, report: Reporter) -> None:
    """Sweeps once and then after every ``sweep_minutes``; it ends when nothing is ever swept or after the first
    sweep when the minutes are infinite. A store that cannot be reached is reported once per outage."""
    outage = StoreOutage(lambda event, code: report(StateChange(event, None, None, code)))
    while (cutoff := swept_before(clock.wall(), memory.lines_keep_days)) is not None:
        try:
            removed = await store.erase_before(cutoff)
        except StoreUnavailable as unavailable:
            outage.failed(unavailable.code)
        else:
            outage.answered()
            if removed.lines or removed.entries:
                report(StateChange(Event.SWEPT, None, None, lines=removed.lines, entries=removed.entries))
        if math.isinf(memory.sweep_minutes):
            return
        await clock.sleep(memory.sweep_minutes * SECONDS_PER_MINUTE)
