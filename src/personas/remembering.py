"""What a loop keeps of the lines it reads and reads back of its agent's day, and the one report of a store's outage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from personas.decision import earlier_today, today_since
from personas.failures import StoreUnavailable
from personas.reporting import StoreOutage

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from personas.clock import Clock
    from personas.decision import EarlierToday, KeptLine, TodayBounds, Visit, WindowRead
    from personas.reporting import Event


class LineStore(Protocol):
    """Keeps the lines of the windows the loops read; it owns the time one write may take."""

    async def keep(self, read: WindowRead) -> None:
        """Keeps every line of the window once; raises ``StoreUnavailable`` when it cannot now."""
        ...

    async def kept_since(self, client_id: str, since: datetime, other_than: Visit) -> tuple[KeptLine, ...]:
        """The agent's lines first read at or after ``since``, but those of one visit; ``StoreUnavailable`` when it
        cannot now."""
        ...


class Remembering:
    """One loop's way to the store: it keeps each window it reads and reads the agent's day back.

    Without a store there is nothing kept and no day read. A store that fails is reported once per outage, and the
    loop goes on without memory meanwhile; the report goes through ``report_state``, which names the loop's agent
    and app.
    """

    def __init__(
        self,
        store: LineStore | None,
        today: TodayBounds | None,
        client_id: str,
        clock: Clock,
        report_state: Callable[[Event, str | None], None],
    ) -> None:
        self._store = store
        self._today = today
        self._client_id = client_id
        self._clock = clock
        self._outage = StoreOutage(report_state)

    async def keep(self, read: WindowRead) -> None:
        """Hands a window to the store, if there is one; its failure is reported once and never reaches the app."""
        store = self._store
        if store is None or not read.window.entries:
            return
        try:
            await store.keep(read)
        except StoreUnavailable as unavailable:
            self._outage.failed(unavailable.code)
        else:
            self._outage.answered()

    async def read_day(self, visit: Visit) -> EarlierToday | None:
        """What the store kept of the agent's other visits within the day, or nothing without a store or a day, or when
        the store cannot be read now; that failure is reported once, as a write's is."""
        store, bounds = self._store, self._today
        if store is None or bounds is None:
            return None
        now = self._clock.wall()
        try:
            kept = await store.kept_since(self._client_id, today_since(now, bounds), visit)
        except StoreUnavailable as unavailable:
            self._outage.failed(unavailable.code)
            return None
        self._outage.answered()
        return earlier_today(kept, now, bounds)
