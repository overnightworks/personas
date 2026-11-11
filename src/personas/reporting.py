"""State changes of a loop, reported once each; the one place that decides what a log line holds."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

_LOG: Final = logging.getLogger("personas")


class Event(StrEnum):
    STARTED = "started"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    BACKOFF = "backoff"
    JOINED = "joined"
    REFUSED = "refused"
    TURN_ENDED = "turn_ended"
    SPEAK_REFUSED = "speak_refused"
    TOOLS_ONLY_APP_FAILED = "tools_only_app_failed"
    STORE_FAILED = "store_failed"
    STORE_RESUMED = "store_resumed"
    SWEPT = "swept"


@dataclass(frozen=True)
class StateChange:
    """One change of an agent's loop for one app, with the HTTP status, refusal, failure or turn code if any.

    The sweep belongs to no agent and no app, and says how many lines it removed.
    """

    event: Event
    client_id: str | None
    app: str | None
    code: str | None = None
    lines: int | None = None


type Reporter = Callable[[StateChange], None]


def log_state_change(change: StateChange) -> None:
    """Writes the fields below and nothing else: no secret, token or tool content reaches a line."""
    removed = "" if change.lines is None else f" lines={change.lines}"
    _LOG.info(
        "%s client_id=%s app=%s code=%s%s",
        change.event,
        change.client_id or "-",
        change.app or "-",
        change.code or "-",
        removed,
    )


class StoreOutage:
    """Whether a store is failing: an outage is reported once, and its end once.

    ``report`` adds the client id and app of whoever uses the outage, or none.
    """

    def __init__(self, report: Callable[[Event, str | None], None]) -> None:
        self._report = report
        self._failing = False

    def failed(self, code: str) -> None:
        if not self._failing:
            self._failing = True
            self._report(Event.STORE_FAILED, code)

    def answered(self) -> None:
        if self._failing:
            self._failing = False
            self._report(Event.STORE_RESUMED, None)
