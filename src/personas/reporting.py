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
    CAPPED = "capped"
    TOOLS_ONLY_APP_FAILED = "tools_only_app_failed"


@dataclass(frozen=True)
class StateChange:
    """One change of an agent's loop for one app, with the HTTP status, refusal, failure or turn code if any."""

    event: Event
    client_id: str
    app: str
    code: str | None = None


type Reporter = Callable[[StateChange], None]


def log_state_change(change: StateChange) -> None:
    """Writes the fields below and nothing else: no secret, token or tool content reaches a line."""
    _LOG.info("%s client_id=%s app=%s code=%s", change.event, change.client_id, change.app, change.code or "-")
