"""The store writer of one agent's mind, written after each change of it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NoReturn

from personas.decision import backoff_delay
from personas.due import kept_turns, pause_deadline
from personas.failures import STORE_TIMEOUT_CODE, StaleGenerationError, StoreDefect, StoreUnavailable
from personas.present import history_delta
from personas.reporting import StateChange, StoreOutage

if TYPE_CHECKING:
    from personas.entries import KeptTurns
    from personas.loop import Boundaries
    from personas.mind import Mind
    from personas.ownership import Owned


class WriteThrough:
    """Writes what his present gained and how his turns changed, after each change of his mind. A write the store
    could not take is reported once per outage and retried after a backoff, with whatever changed meanwhile; a write
    the fence refused raises ``StaleGenerationError``, which stops him on this pod."""

    def __init__(self, mind: Mind, owned: Owned, boundaries: Boundaries) -> None:
        self._mind = mind
        self._owned = owned
        self._boundaries = boundaries
        self._outage = StoreOutage(
            lambda event, code: boundaries.report(StateChange(event, owned.client_id, None, code))
        )
        self._written = mind.present
        self._written_turns: KeptTurns = owned.history.turns
        self._paused = (mind.turn_state().turns.not_before, owned.history.turns.paused_until)
        self._stop_deadline: float | None = None

    async def run(self) -> NoReturn:
        failures = 0
        changed = self._mind.changed()
        while True:
            await changed.wait()
            changed.clear()
            try:
                await self.flush()
            except StoreUnavailable as unavailable:
                self._outage.failed(unavailable.code)
                failures += 1
                changed.set()
                await self._boundaries.clock.sleep(backoff_delay(failures))
                continue
            self._outage.answered()
            failures = 0

    async def flush(self) -> None:
        """Writes whatever changed since the last write that was taken; nothing when nothing did. Either way the mind
        learns that what changed up to now is kept."""
        changes = self._mind.changes()
        present = self._mind.present
        state = self._mind.turn_state()
        not_before, paused_until = self._paused
        if state.turns.not_before != not_before:
            clock = self._boundaries.clock
            paused_until = pause_deadline(state.turns.not_before, clock.now(), clock.wall())
        delta = history_delta(self._written, present)
        turns = kept_turns(state, paused_until)
        if delta is not None or turns != self._written_turns:
            await self._owned.write(delta, turns)
            self._written, self._written_turns = present, turns
            self._paused = (state.turns.not_before, paused_until)
        self._mind.written(changes)

    def stopping(self, margin_seconds: float) -> None:
        """His stop begins now; it has ``margin_seconds`` to end."""
        self._stop_deadline = self._boundaries.clock.now() + margin_seconds

    def seconds_left_to_stop(self) -> float:
        assert self._stop_deadline is not None  # noqa: S101 - a stop begins before the agent's task is cancelled
        return max(0.0, self._stop_deadline - self._boundaries.clock.now())

    async def flush_on_stop(self) -> None:
        """Writes what the stop cut off, within what is left of its margin; a write the time or the store did not take
        is reported and the stop goes on, and one the fence refused is left, since another pod serves him now."""
        try:
            async with asyncio.timeout(self.seconds_left_to_stop()):
                await self.flush()
        except TimeoutError:
            self._outage.failed(STORE_TIMEOUT_CODE)
        except StoreUnavailable as unavailable:
            self._outage.failed(unavailable.code)
        except StoreDefect as defect:
            self._outage.failed(str(defect))
        except StaleGenerationError:
            return
