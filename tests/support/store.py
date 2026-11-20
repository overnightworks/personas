"""A store of lines the tests of the loop script: what it keeps, finds and fails with."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tests.support.hopin import LIMIT_SECONDS

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from personas.entries import KeptLine, WindowRead
    from personas.failures import StoreUnavailable
    from personas.recall import LineSearch


class ScriptedStore:
    """Keeps what it is handed, or fails with the failure it is told to, or waits for a release first; it answers a
    search with the lines it is told to give, or fails with the failure it is told to."""

    def __init__(self) -> None:
        self.kept: list[WindowRead] = []
        self.failure: StoreUnavailable | None = None
        self.searches: list[LineSearch] = []
        self.found: tuple[KeptLine, ...] = ()
        self.search_failure: Exception | None = None
        self.on_search_failure: Callable[[], None] = lambda: None
        self.waiting = asyncio.Event()
        self.release: asyncio.Event | None = None
        self.attempts = 0
        self._attempted = asyncio.Condition()

    async def until_waiting(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.waiting.wait()

    async def until_attempts(self, count: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS), self._attempted:
            await self._attempted.wait_for(lambda: self.attempts >= count)

    async def keep(self, read: WindowRead) -> None:
        async with self._attempted:
            self.attempts += 1
            self._attempted.notify_all()
        if self.release is not None:
            self.waiting.set()
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        self.kept.append(read)

    async def search(self, search: LineSearch) -> tuple[KeptLine, ...]:
        self.searches.append(search)
        if self.search_failure is not None:
            self.on_search_failure()
            raise self.search_failure
        return self.found

    async def erase_before(self, first_read_before: datetime) -> int:
        raise AssertionError("a test about the loop swept")

    def seqs_kept(self) -> list[int]:
        return [line.seq for read in self.kept for line in read.window.entries]
