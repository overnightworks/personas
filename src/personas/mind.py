"""One agent's mind: the history of the lines he took in, the views he read and the calls relayed for him at any of his
apps, the apps where he holds an active place, and the one turn he takes at a time, with the places that have one due.
His loops and turns feed it and every turn reads it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from personas.decision import Turns, claimed, held_apps, history_after, released, with_due, without_due
from personas.entries import History

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from personas.decision import AppPlace, Occasion
    from personas.entries import Entry


class Mind:
    """Holds one agent's history and turns in memory for the life of the runner; the rules that bound the history and
    move the turns are ``decision``'s. Each method is synchronous, so a loop changes it between its awaits."""

    def __init__(self, size_bytes: int) -> None:
        self._size_bytes = size_bytes
        self._history = History((), left_out=0, heard=0)
        self._turns = Turns()
        self._placed = frozenset[str]()
        self._wake_events: list[asyncio.Event] = []
        self._next_take = asyncio.Event()

    @property
    def history(self) -> History:
        return self._history

    def take(self, entries: Iterable[Entry]) -> None:
        """Appends the lines, spoken sentences, views and acts taken, in order, and sets the event ``next_take`` gave
        before; taking nothing leaves the history and that event as they are."""
        taken = tuple(entries)
        if not taken:
            return
        this_take, self._next_take = self._next_take, asyncio.Event()
        self._history = history_after(self._history, taken, self._size_bytes)
        this_take.set()

    def next_take(self) -> asyncio.Event:
        """The event the next take sets. A waiter reads it in the same synchronous step as the history it judges, so a
        take made after that step sets it."""
        return self._next_take

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
