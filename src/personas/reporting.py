"""State changes of a loop, reported once each; the one place that decides what a log line holds."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from agents.turn import TurnUsage

    from personas.entries import InputSizes

_LOG: Final = logging.getLogger("personas")


class Event(StrEnum):
    STARTED = "started"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    BACKOFF = "backoff"
    JOINED = "joined"
    REFUSED = "refused"
    INVITATION_LEFT_OUT = "invitation_left_out"
    TURN_ENDED = "turn_ended"
    SPEAK_REFUSED = "speak_refused"
    TOOLS_ONLY_APP_FAILED = "tools_only_app_failed"
    HELD_APP_FAILED = "held_app_failed"
    STORE_FAILED = "store_failed"
    STORE_RESUMED = "store_resumed"
    SWEPT = "swept"


@dataclass(frozen=True)
class StateChange:
    """One change of an agent's loop for one app, with the HTTP status, refusal, failure or turn code if any.

    The sweep belongs to no agent and no app, and says how many lines it removed. A turn that sent a model request
    says the bytes of its input; one that ended with a final event says the tokens it used, when the library had
    them. An invitation the loop leaves out says its place's kind and the reason as the code.
    """

    event: Event
    client_id: str | None
    app: str | None
    code: str | None = None
    lines: int | None = None
    usage: TurnUsage | None = None
    sizes: InputSizes | None = None
    place_kind: str | None = None


type Reporter = Callable[[StateChange], None]


def log_state_change(change: StateChange) -> None:
    """Writes the fields below and nothing else: no secret, token or tool content reaches a line."""
    removed = "" if change.lines is None else f" lines={change.lines}"
    sized = "" if change.sizes is None else _sizes_of(change.sizes)
    used = "" if change.usage is None else _tokens_used(change.usage)
    kind = "" if change.place_kind is None else f" place_kind={change.place_kind}"
    _LOG.info(
        "%s client_id=%s app=%s code=%s%s%s%s%s",
        change.event,
        change.client_id or "-",
        change.app or "-",
        change.code or "-",
        removed,
        sized,
        used,
        kind,
    )


def _sizes_of(sizes: InputSizes) -> str:
    return f" tools_bytes={sizes.tools_bytes} told_bytes={sizes.told_bytes} results_bytes={sizes.results_bytes}"


def _tokens_used(usage: TurnUsage) -> str:
    return (
        f" input={usage.input_tokens} cache_read={usage.cached_input_tokens}"
        f" cache_write={usage.cache_write_tokens} output={usage.output_tokens}"
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
