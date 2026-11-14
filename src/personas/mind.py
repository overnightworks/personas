"""One agent's mind: the history of the lines he took in, the views he read and the calls relayed for him at any of his
apps, the apps where he holds an active place, and the one turn he takes at a time, with the places that have one due.
His loops and turns feed it and every turn reads it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from personas.decision import (
    DEFAULT_BUDGET_PAUSE_CAP_SECONDS,
    Turns,
    backoff_delay,
    budget_pause,
    claimed,
    due_again,
    held_apps,
    history_after,
    released,
    with_due,
    without_due,
)
from personas.entries import History

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from personas.decision import AppPlace, Occasion
    from personas.entries import Entry


class Mind:
    """Holds one agent's history and turns in memory for the life of the runner; the rules that bound the history and
    move the turns are ``decision``'s. Each method is synchronous, so a loop changes it between its awaits."""

    def __init__(self, size_bytes: int, budget_pause_cap_seconds: float = DEFAULT_BUDGET_PAUSE_CAP_SECONDS) -> None:
        self._size_bytes = size_bytes
        self._budget_pause_cap_seconds = budget_pause_cap_seconds
        self._unanswered_turns = 0
        self._history = History((), left_out=0, heard=0)
        self._turns = Turns()
        self._placed = frozenset[str]()
        self._wake_events: list[asyncio.Event] = []

    @property
    def history(self) -> History:
        return self._history

    def take(self, entries: Iterable[Entry]) -> None:
        """Appends the lines, spoken sentences, views and acts taken, in order; taking nothing leaves the history as
        it is."""
        taken = tuple(entries)
        if not taken:
            return
        self._history = history_after(self._history, taken, self._size_bytes)

    def unanswered(self, *, budget_spent: bool) -> float:
        """A turn of his ended without an answer, refused for a spent budget or not; the seconds his turns wait now."""
        self._unanswered_turns += 1
        if budget_spent:
            return budget_pause(self._unanswered_turns, self._budget_pause_cap_seconds)
        return backoff_delay(self._unanswered_turns)

    def answered(self) -> None:
        """A turn of his ended with an answer: the wait starts over."""
        self._unanswered_turns = 0

    def stand_at(self, app: str, *, holds: bool) -> None:
        """Records whether he holds an active place at ``app``, as its loop last read his standing."""
        self._placed = self._placed | {app} if holds else self._placed - {app}

    def held_apps(self, own: str) -> frozenset[str]:
        """The apps, other than ``own``, where he holds an active place."""
        return held_apps(own, self._placed)

    def wake_event(self) -> asyncio.Event:
        """A new event a loop waits on; the end of a turn, and a place dropped while no turn runs, set it."""
        event = asyncio.Event()
        self._wake_events.append(event)
        return event

    def due(self, at: AppPlace, occasion: Occasion) -> None:
        self._turns = with_due(self._turns, at, occasion)

    def due_again(self, at: AppPlace, occasion: Occasion) -> None:
        """The turn at ``at`` ended without an answer: the place is due for ``occasion`` again unless something newer
        made it due meanwhile. It waits for the running turn to end."""
        self._turns = due_again(self._turns, at, occasion)

    def undue(self, leaving: Callable[[AppPlace], bool]) -> None:
        """Drops the due places for which ``leaving`` holds; with no turn running a younger place may then go first."""
        remaining = without_due(self._turns, leaving)
        dropped = len(remaining.due) < len(self._turns.due)
        self._turns = remaining
        if dropped and not remaining.running:
            self._wake()

    def claim(self, app: str) -> tuple[AppPlace, Occasion] | None:
        """Takes the one turn for the place that has waited longest when it is at ``app``."""
        taken = claimed(self._turns, app)
        if taken is None:
            return None
        self._turns, at, occasion = taken
        return at, occasion

    def release(self) -> None:
        """The running turn ended: every loop of his may claim now."""
        self._turns = released(self._turns)
        self._wake()

    def _wake(self) -> None:
        for event in self._wake_events:
            event.set()
