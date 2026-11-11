"""One agent's mind: the history of the lines he took in at any of his apps, which his loops feed and his turns read."""

from __future__ import annotations

from typing import TYPE_CHECKING

from personas.decision import History, history_after

if TYPE_CHECKING:
    from collections.abc import Iterable

    from personas.decision import KeptLine


class Mind:
    """Holds one agent's history in memory for the life of the runner; the rule that bounds it is ``decision``'s."""

    def __init__(self, size_bytes: int) -> None:
        self._size_bytes = size_bytes
        self._history = History((), older_dropped=False)

    @property
    def history(self) -> History:
        return self._history

    def take(self, lines: Iterable[KeptLine]) -> None:
        """Appends the lines a read took, in order."""
        self._history = history_after(self._history, lines, self._size_bytes)
