"""A place store held in memory, for the one pod a test runs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.mcp_server.place_store import ListeningTurn, PlaceSignal, Resubscribed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection
    from datetime import timedelta

    from agents.turn import Principal


class MemoryPlaceStore:
    """Delivers each signal to the one subscription and records which stream listens to which URI."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[PlaceSignal | ListeningTurn | Resubscribed] = asyncio.Queue()
        self._streams: dict[str, tuple[Principal, frozenset[str]]] = {}

    async def publish(self, signal: PlaceSignal) -> None:
        self._events.put_nowait(signal)

    async def received(self) -> AsyncIterator[PlaceSignal | ListeningTurn | Resubscribed]:
        yield Resubscribed()
        while True:
            yield await self._events.get()

    async def acknowledged(
        self, stream: str, principal: Principal, uris: Collection[str], holds_for: timedelta
    ) -> frozenset[str]:
        del holds_for
        if stream in self._streams:
            return frozenset()
        turned = frozenset(each for each in uris if not self._listens(principal, each))
        self._streams[stream] = (principal, frozenset(uris))
        for each in turned:
            self._events.put_nowait(ListeningTurn(principal, each, listens=True))
        return turned

    async def ended(self, stream: str, holds_for: timedelta) -> None:
        del holds_for
        recorded = self._streams.pop(stream, None)
        if recorded is None:
            return
        principal, uris = recorded
        for each in uris:
            if not self._listens(principal, each):
                self._events.put_nowait(ListeningTurn(principal, each, listens=False))

    async def listens_to(self, principal: Principal, uri: str) -> bool:
        return self._listens(principal, uri)

    def _listens(self, principal: Principal, uri: str) -> bool:
        return any(who == principal and uri in uris for who, uris in self._streams.values())

    async def listeners(self, uri: str) -> frozenset[Principal]:
        return frozenset(who for who, uris in self._streams.values() if uri in uris)

    async def lapse_due(self, claimer: str) -> None:
        del claimer
