"""One agent's mind: the present his requests hold, which is the history of the lines he took in, the views he read and
the calls relayed for him at any of his apps with the turns that were told of it, the apps where he holds an active
place, and the one turn he takes at a time, with the places that have one due. His loops and turns feed it and every
turn reads it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from personas.decision import (
    DEFAULT_BUDGET_PAUSE_CAP_SECONDS,
    EMPTY_PRESENT,
    Turns,
    answered_into,
    backoff_delay,
    budget_pause,
    claimed,
    due_again,
    held_apps,
    released,
    taken_into,
    with_due,
    without_due,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from personas.decision import AppPlace, Occasion, Present, TurnTold
    from personas.entries import Entry, History


class Mind:
    """Holds one agent's present and turns in memory for the life of the runner; the rules that bound the history and
    move the turns are ``decision``'s. Each method is synchronous, so a loop changes it between its awaits."""

    def __init__(
        self,
        history_bytes: int,
        history_keep_bytes: int,
        budget_pause_cap_seconds: float = DEFAULT_BUDGET_PAUSE_CAP_SECONDS,
    ) -> None:
        self._history_bytes = history_bytes
        self._history_keep_bytes = history_keep_bytes
        self._budget_pause_cap_seconds = budget_pause_cap_seconds
        self._unanswered_turns = 0
        self._present = EMPTY_PRESENT
        self._turns = Turns()
        self._placed = frozenset[str]()
        self._wake_events: list[asyncio.Event] = []
        self._next_take = asyncio.Event()

    @property
    def present(self) -> Present:
        return self._present

    @property
    def history(self) -> History:
        return self._present.history

    def take(self, entries: Iterable[Entry]) -> None:
        """Appends the lines, spoken sentences, views and acts taken, in order, and sets the event ``next_take`` gave
        before; taking nothing leaves the history and that event as they are."""
        taken = tuple(entries)
        if not taken:
            return
        previous, self._next_take = self._next_take, asyncio.Event()
        self._present = taken_into(self._present, taken, self._history_bytes, self._history_keep_bytes)
        previous.set()

    def next_take(self) -> asyncio.Event:
        """The event the next take sets. A waiter reads it in the same synchronous step as the history it judges, so a
        take made after that step sets it."""
        return self._next_take

    def unanswered(self, *, budget_spent: bool) -> float:
        """A turn of his ended without an answer, refused for a spent budget or not; the seconds his turns wait now."""
        self._unanswered_turns += 1
        if budget_spent:
            return budget_pause(self._unanswered_turns, self._budget_pause_cap_seconds)
        return backoff_delay(self._unanswered_turns)

    def answered(self, turn: TurnTold) -> None:
        """A turn of his ended with an answer: the wait starts over and its message joins the present."""
        self._unanswered_turns = 0
        self._present = answered_into(self._present, turn)

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
