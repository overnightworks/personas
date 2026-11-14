"""What a loop keeps of the lines it reads, and the one report of a store's outage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from personas.failures import StoreUnavailable
from personas.reporting import StoreOutage

if TYPE_CHECKING:
    from collections.abc import Callable

    from personas.entries import WindowRead
    from personas.reporting import Event


class LineStore(Protocol):
    """Keeps the lines of the windows the loops read; it owns the time one write may take."""

    async def keep(self, read: WindowRead) -> None:
        """Keeps every line of the window once; raises ``StoreUnavailable`` when it cannot now."""
        ...


class Remembering:
    """One loop's way to the store: it keeps each window it reads.

    Without a store there is nothing kept. A store that fails is reported once per outage, and the loop goes on without
    memory meanwhile; the report goes through ``report_state``, which names the loop's agent and app.
    """

    def __init__(self, store: LineStore | None, report_state: Callable[[Event, str | None], None]) -> None:
        self._store = store
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
