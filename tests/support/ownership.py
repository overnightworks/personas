"""The ownership of a runner that no other pod competes with, and agent histories kept in memory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from itertools import takewhile
from typing import TYPE_CHECKING

from personas.entries import Entry, HistoryDelta, KeptHistory, KeptTurns
from personas.failures import StaleGenerationError
from personas.ownership import taken_over

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import NoReturn

    from personas.ownership import ServeAgent

NOTHING_KEPT = KeptHistory(0, (), 0, ())


@dataclass
class KeptHistories:
    """Each agent's history as a test sets it, his generation renewed at each takeover, every delta written, in order,
    and the turns written last; a write under a generation not current is refused."""

    kept: dict[str, KeptHistory] = field(default_factory=dict[str, KeptHistory])
    generations: dict[str, int] = field(default_factory=dict[str, int])
    written: list[tuple[str, HistoryDelta]] = field(default_factory=list[tuple[str, HistoryDelta]])
    turns: dict[str, KeptTurns] = field(default_factory=dict[str, KeptTurns])

    def entries(self) -> list[Entry]:
        """Every entry written, in order."""
        return [entry for _, delta in self.written for entry in delta.entries]

    async def take_over(self, client_id: str) -> int:
        self.generations[client_id] = self.generations.get(client_id, 0) + 1
        return self.generations[client_id]

    async def load(self, client_id: str) -> KeptHistory:
        return self.kept.get(client_id, NOTHING_KEPT)

    async def write(self, client_id: str, generation: int, delta: HistoryDelta | None, turns: KeptTurns) -> None:
        if generation != self.generations.get(client_id):
            raise StaleGenerationError
        if delta is not None:
            self.written.append((client_id, delta))
            self.kept[client_id] = _kept_after(self.kept.get(client_id, NOTHING_KEPT), delta)
        self.turns[client_id] = turns


def _kept_after(kept: KeptHistory, delta: HistoryDelta) -> KeptHistory:
    """What the store keeps of a history once it took ``delta``: the entries and turns it held that the delta does not
    drop, then the delta's."""
    entries = (*kept.entries[delta.left_out - kept.first : delta.first - kept.first], *delta.entries)
    told = (*kept.told[: delta.first_told - kept.first_told], *delta.told)
    dropped = sum(1 for _ in takewhile(lambda turn: turn.through <= delta.left_out, told))
    return KeptHistory(delta.left_out, entries, kept.first_told + dropped, told[dropped:])


class AllAgentsOwned:
    """Serves every agent without a lock and without a grace, his history kept in ``histories``."""

    def __init__(self, histories: KeptHistories | None = None) -> None:
        self.histories = histories or KeptHistories()

    async def serve(
        self, client_ids: Sequence[str], serve_agent: ServeAgent, contending: Callable[[], None]
    ) -> NoReturn:
        contending()
        async with asyncio.TaskGroup() as group:
            for each in client_ids:
                group.create_task(serve_agent(await taken_over(self.histories, each)))
        raise AssertionError("an agent ended")  # pragma: no cover
