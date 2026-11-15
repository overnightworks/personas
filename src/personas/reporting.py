"""State changes of a loop, reported once each; the one place that decides what a log line holds."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from logline import Outcome

from personas.decision import failure_reason, turn_outcome
from personas.failures import StatusFailure

if TYPE_CHECKING:
    from agents.turn import TurnUsage

    from personas.entries import InputSizes
    from personas.failures import CallFailure

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
    PAUSED = "paused"
    SPEAK_REFUSED = "speak_refused"
    TOOLS_ONLY_APP_FAILED = "tools_only_app_failed"
    HELD_APP_FAILED = "held_app_failed"
    STORE_FAILED = "store_failed"
    STORE_RESUMED = "store_resumed"
    SWEPT = "swept"


@dataclass(frozen=True)
class StateChange:
    """One change of an agent's loop for one app, with the refusal, failure or turn code if any.

    The sweep belongs to no agent and no app, and says how many lines it removed. A turn that sent a model request
    says the bytes of its input; one that ended with a final event says the tokens it used, when the library had
    them. The start says the model the agent's turns run on, when it has a persona. An invitation the loop leaves out
    says its place's kind and the reason as the code. A pause of his turns says its reason as the code and its wait in
    seconds. A call that failed says its failure: the line holds its HTTP status, the MCP method that failed and a
    closed reason.
    """

    event: Event
    client_id: str | None
    app: str | None
    code: str | None = None
    lines: int | None = None
    usage: TurnUsage | None = None
    sizes: InputSizes | None = None
    place_kind: str | None = None
    seconds: float | None = None
    model: str | None = None
    failure: CallFailure | None = None


type Reporter = Callable[[StateChange], None]


def log_state_change(change: StateChange) -> None:
    """Writes the fields of ``_fields_of`` and nothing else: no secret, token or tool content reaches a line."""
    fields = _fields_of(change)
    match change.event:
        case Event.STARTED:
            _LOG.info("loop.started", extra=fields)
        case Event.SUSPENDED:
            _LOG.info("loop.suspended", extra=fields)
        case Event.RESUMED:
            _LOG.info("loop.resumed", extra=fields)
        case Event.BACKOFF:
            _LOG.info("loop.backed_off", extra=fields)
        case Event.JOINED:
            _LOG.info("invitation.accepted", extra=fields)
        case Event.REFUSED:
            _LOG.info("invitation.refused", extra=fields)
        case Event.INVITATION_LEFT_OUT:
            _LOG.info("invitation.left_out", extra=fields)
        case Event.TURN_ENDED:
            if fields["outcome"] is Outcome.FAILED:
                _LOG.error("turn.ended", extra=fields)
            else:
                _LOG.info("turn.ended", extra=fields)
        case Event.PAUSED:
            _LOG.info("turns.paused", extra=fields)
        case Event.SPEAK_REFUSED:
            _LOG.info("speak.refused", extra=fields)
        case Event.TOOLS_ONLY_APP_FAILED:
            _LOG.warning("tools_only_app.failed", extra=fields)
        case Event.HELD_APP_FAILED:
            _LOG.warning("held_app.failed", extra=fields)
        case Event.STORE_FAILED:
            _LOG.warning("store.failed", extra=fields)
        case Event.STORE_RESUMED:
            _LOG.info("store.resumed", extra=fields)
        case Event.SWEPT:
            _LOG.info("lines.swept", extra=fields)


def _fields_of(change: StateChange) -> dict[str, str | int | float | Outcome]:
    fields: dict[str, str | int | float | Outcome] = {}
    if change.client_id is not None:
        fields["agent_id"] = change.client_id
    if change.app is not None:
        fields["app_id"] = urlsplit(change.app).hostname or ""
    if change.event is Event.TURN_ENDED:
        fields["outcome"] = turn_outcome(change.code)
    if change.failure is not None:
        fields["reason"] = failure_reason(change.failure)
        if isinstance(change.failure, StatusFailure):
            fields["status"] = change.failure.status
        if change.failure.method is not None:
            fields["mcp_method"] = change.failure.method
    elif change.code is not None:
        fields["reason"] = change.code
    if change.lines is not None:
        fields["line_count"] = change.lines
    if change.place_kind is not None:
        fields["place_kind"] = change.place_kind
    if change.seconds is not None:
        fields["pause_ms"] = round(change.seconds * 1000, 3)
    if change.model is not None:
        fields["model"] = change.model
    if change.sizes is not None:
        fields |= {
            "tools_bytes": change.sizes.tools_bytes,
            "told_bytes": change.sizes.told_bytes,
            "results_bytes": change.sizes.results_bytes,
            "images_bytes": change.sizes.images_bytes,
        }
    if change.usage is not None:
        fields |= {
            "input_tokens": change.usage.input_tokens,
            "cached_input_tokens": change.usage.cached_input_tokens,
            "cache_write_tokens": change.usage.cache_write_tokens,
            "output_tokens": change.usage.output_tokens,
            "model_request_count": change.usage.requests,
        }
    return fields


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
