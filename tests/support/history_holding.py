"""Agent histories a loop test holds a write of."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from personas.entries import HistoryDelta, KeptLine, KeptTurns
from tests.support.call import ADDRESSED
from tests.support.ownership import KeptHistories


@dataclass
class HistoriesHoldingTheAddressedLine(KeptHistories):
    """Histories whose first write that holds the addressed line, and the claim of its turn with it or after it, waits
    until ``release`` is set: a write the store is slow to take."""

    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def write(self, client_id: str, generation: int, delta: HistoryDelta | None, turns: KeptTurns) -> None:
        if not self.reached.is_set() and delta is not None and ADDRESSED in _texts_of(delta):
            self.reached.set()
            await self.release.wait()
        await super().write(client_id, generation, delta, turns)


def _texts_of(delta: HistoryDelta) -> set[str]:
    return {each.text for each in delta.entries if isinstance(each, KeptLine)}
