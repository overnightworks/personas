"""One agent's mind: the present his requests hold, which is the history of the lines he took in, the views he read and
the calls relayed for him at any of his apps with the turns that were told of it, the apps where he holds an active
place, and the one turn he takes at a time, with the places that have one due. His loops and turns feed it and every
turn reads it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from personas.decision import (
    EMPTY_PRESENT,
    Turns,
    answered_into,
    backoff_delay,
    budget_pause,
    claimed,
    due_again,
    held_apps,
    pause_left,
    paused_for,
    refusal_retry_at,
    released,
    taken_into,
    with_due,
    without_due,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from personas.decision import AppPlace, Due, Present
    from personas.entries import Entry, History, Occasion, TurnTold


class Mind:
    """Holds one agent's present and turns in memory while this pod serves him, from the present his history resumed
    as; the rules that bound the history and move the turns are ``decision``'s. Each method is synchronous, so a loop
    changes it between its awaits."""

    def __init__(
        self,
        history_bytes: int,
        history_keep_bytes: int,
        budget_pause_cap_seconds: float,
        present: Present = EMPTY_PRESENT,
    ) -> None:
        self._history_bytes = history_bytes
        self._history_keep_bytes = history_keep_bytes
        self._budget_pause_cap_seconds = budget_pause_cap_seconds
        self._unanswered_turns = 0
        self._present = present
        self._present_changed = asyncio.Event()
        self._turns = Turns()
        self._placed = frozenset[str]()
        self._not_granted: dict[str, float] = {}
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
        self._present_changed.set()

    def next_take(self) -> asyncio.Event:
        """The event the next take sets. A waiter reads it in the same synchronous step as the history it judges, so a
        take made after that step sets it."""
        return self._next_take

    def present_changed(self) -> asyncio.Event:
        """The event a take and an answered turn set, which stays set until its waiter clears it."""
        return self._present_changed

    def unanswered(self, *, budget_spent: bool, now: float) -> float:
        """A turn of his ended without an answer, refused for a spent budget or not: no turn of his starts for the
        seconds this returns, counted from ``now``, whatever becomes of the turn's own task or place meanwhile."""
        self._unanswered_turns += 1
        seconds = (
            budget_pause(self._unanswered_turns, self._budget_pause_cap_seconds)
            if budget_spent
            else backoff_delay(self._unanswered_turns)
        )
        self._turns = paused_for(self._turns, now, seconds)
        return seconds

    def answered(self, turn: TurnTold) -> None:
        """A turn of his ended with an answer: the wait starts over and its message joins the present."""
        self._unanswered_turns = 0
        self._present = answered_into(self._present, turn)
        self._present_changed.set()

    def stand_at(self, app: str, *, holds: bool) -> None:
        """Records whether he holds an active place at ``app``, as its loop last read his standing."""
        self._placed = self._placed | {app} if holds else self._placed - {app}

    def held_apps(self, own: str) -> frozenset[str]:
        """The apps, other than ``own``, where he holds an active place."""
        return held_apps(own, self._placed)

    def holds_back(self, app: str, now: float) -> bool:
        """Whether the realm refused him ``app`` and its retry is not yet due at ``now``."""
        retry_at = self._not_granted.get(app)
        return retry_at is not None and now < retry_at

    def refused(self, app: str, now: float) -> bool:
        """The realm or ``app`` did not grant him ``app`` at ``now``: it is asked again when the retry is due. Whether
        this is the first refusal since the last grant."""
        first = app not in self._not_granted
        self._not_granted[app] = refusal_retry_at(now)
        return first

    def granted(self, app: str) -> bool:
        """``app`` answered him. Whether it was refused before."""
        return self._not_granted.pop(app, None) is not None

    def wake_event(self) -> asyncio.Event:
        """A new event a loop waits on; the end of a turn or of a pause sets it, and so does a place dropped while no
        turn runs."""
        event = asyncio.Event()
        self._wake_events.append(event)
        return event

    def due(self, at: AppPlace, occasion: Occasion, since: float) -> None:
        """The place is due for ``occasion`` from ``since``, a ``time.perf_counter`` value."""
        self._turns = with_due(self._turns, at, occasion, since)

    def due_again(self, at: AppPlace, occasion: Occasion, since: float) -> None:
        """The turn at ``at`` ended without an answer: the place is due for ``occasion`` again from ``since``, the end
        of the pause the turn left, unless something newer made it due meanwhile. It waits for the running turn to
        end."""
        self._turns = due_again(self._turns, at, occasion, since)

    def undue(self, leaving: Callable[[AppPlace], bool]) -> None:
        """Drops the due places for which ``leaving`` holds; with no turn running a younger place may then go first."""
        remaining = without_due(self._turns, leaving)
        dropped = len(remaining.due) < len(self._turns.due)
        self._turns = remaining
        if dropped and not remaining.running:
            self.wake()

    def claim(self, app: str, now: float) -> tuple[AppPlace, Due] | None:
        """Takes the one turn for the place that has waited longest when it is at ``app`` and no pause holds it."""
        taken = claimed(self._turns, app, now)
        if taken is None:
            return None
        self._turns, at, due = taken
        return at, due

    def pause_left(self, app: str, now: float) -> float:
        """The seconds a pause still holds the turn of the place that has waited longest, when it is at ``app``."""
        return pause_left(self._turns, app, now)

    def release(self) -> None:
        """The running turn ended: every loop of his may claim now."""
        self._turns = released(self._turns)
        self.wake()

    def wake(self) -> None:
        """Wakes every loop of his: the end of a pause is as much a reason to claim as the end of a turn."""
        for event in self._wake_events:
            event.set()
