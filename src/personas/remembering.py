"""What a loop keeps of the lines it reads, and the one report of a store's outage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from personas.failures import StoreUnavailable
from personas.reporting import StoreOutage

if TYPE_CHECKING:
    from collections.abc import Callable

    from personas.entries import ToldEntry, WindowRead
    from personas.recall import EntrySearch, LookupBounds
    from personas.reporting import Event


class LineStore(Protocol):
    """Keeps the lines of the windows the loops read, searches them and the record; it owns the time one write or
    read may take."""

    async def keep(self, read: WindowRead) -> None:
        """Keeps every line of the window once; raises ``StoreUnavailable`` when it cannot now."""
        ...

    async def search(self, search: EntrySearch) -> tuple[ToldEntry, ...]:
        """The entries the search asks for, newest first; raises ``StoreUnavailable`` when it cannot answer now."""
        ...


class Remembering:
    """One loop's way to the store: it keeps each window it reads and searches what is kept for a turn's lookup,
    within the ``lookup`` bounds, which are set exactly when the configuration has a memory.

    Without a store there is nothing kept, and no turn looks. A store that fails a write or a search is reported once
    per outage, shared by both, and the loop goes on without memory meanwhile; the report goes through
    ``report_state``, which names the loop's agent and app.
    """

    def __init__(
        self,
        store: LineStore | None,
        report_state: Callable[[Event, str | None], None],
        lookup: LookupBounds | None,
    ) -> None:
        self._store = store
        self.lookup = lookup
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

    async def search(self, search: EntrySearch) -> tuple[ToldEntry, ...] | None:
        """The entries a turn's lookup asked for, or ``None`` after the report of a store that could not answer.

        A defect of the store leaves as itself and ends the runner.
        """
        store = self._store
        assert store is not None  # noqa: S101 - only a configuration with a memory hands out the lookup and builds the store
        try:
            found = await store.search(search)
        except StoreUnavailable as unavailable:
            self._outage.failed(unavailable.code)
            return None
        self._outage.answered()
        return found
