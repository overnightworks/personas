"""State changes of a loop, reported once each; the one place that decides what a log line holds."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from logline import Outcome, StepTimer

from personas.decision import failure_reason
from personas.failures import StatusFailure

if TYPE_CHECKING:
    from agents.turn import TurnUsage

    from personas.decision import TurnEnding
    from personas.entries import InputSizes
    from personas.failures import CallFailure

_LOG: Final = logging.getLogger("personas")


class Event(Enum):
    STARTED = auto()
    SUSPENDED = auto()
    RESUMED = auto()
    BACKOFF = auto()
    JOINED = auto()
    REFUSED = auto()
    INVITATION_LEFT_OUT = auto()
    PAUSED = auto()
    SPEAK_REFUSED = auto()
    TOOLS_ONLY_APP_FAILED = auto()
    HELD_APP_FAILED = auto()
    STORE_FAILED = auto()
    STORE_RESUMED = auto()
    SWEPT = auto()


@dataclass(frozen=True)
class StateChange:
    """One change of an agent's loop for one app, with the refusal or failure code if any.

    The sweep belongs to no agent and no app, and says how many lines it removed. The start says the model the agent's
    turns run on, when it has a persona. An invitation the loop leaves out says its place's kind and the reason as the
    code. A pause of his turns says its reason as the code and its wait in seconds. A call that failed
    says its failure: the line holds its HTTP status, the MCP method that failed and a closed reason.
    """

    event: Event
    client_id: str | None
    app: str | None
    code: str | None = None
    lines: int | None = None
    place_kind: str | None = None
    seconds: float | None = None
    model: str | None = None
    failure: CallFailure | None = None


type Reporter = Callable[[StateChange], None]


_LINE_OF: Final = {
    Event.STARTED: (logging.INFO, "loop.started"),
    Event.SUSPENDED: (logging.INFO, "loop.suspended"),
    Event.RESUMED: (logging.INFO, "loop.resumed"),
    Event.BACKOFF: (logging.INFO, "loop.backed_off"),
    Event.JOINED: (logging.INFO, "invitation.accepted"),
    Event.REFUSED: (logging.INFO, "invitation.refused"),
    Event.INVITATION_LEFT_OUT: (logging.INFO, "invitation.left_out"),
    Event.PAUSED: (logging.INFO, "turns.paused"),
    Event.SPEAK_REFUSED: (logging.INFO, "speak.refused"),
    Event.TOOLS_ONLY_APP_FAILED: (logging.WARNING, "tools_only_app.failed"),
    Event.HELD_APP_FAILED: (logging.WARNING, "held_app.failed"),
    Event.STORE_FAILED: (logging.WARNING, "store.failed"),
    Event.STORE_RESUMED: (logging.INFO, "store.resumed"),
    Event.SWEPT: (logging.INFO, "lines.swept"),
}


def log_state_change(change: StateChange) -> None:
    """Writes the fields of ``_fields_of`` and nothing else: no secret, token or tool content reaches a line."""
    fields = _fields_of(change)
    level, line = _LINE_OF[change.event]
    _LOG.log(level, line, extra=fields)


def _fields_of(change: StateChange) -> dict[str, str | int | float]:
    fields: dict[str, str | int | float] = {}
    if change.client_id is not None:
        fields["agent_id"] = change.client_id
    if change.app is not None:
        fields["app_id"] = urlsplit(change.app).hostname or ""
    fields |= _failure_fields(change)
    if change.lines is not None:
        fields["line_count"] = change.lines
    if change.place_kind is not None:
        fields["place_kind"] = change.place_kind
    if change.seconds is not None:
        fields["pause_ms"] = round(change.seconds * 1000, 3)
    if change.model is not None:
        fields["model"] = change.model
    return fields


def _failure_fields(change: StateChange) -> dict[str, str | int]:
    failure = change.failure
    if failure is None:
        return {} if change.code is None else {"reason": change.code}
    fields: dict[str, str | int] = {"reason": failure_reason(failure)}
    if isinstance(failure, StatusFailure):
        fields["status"] = failure.status
    if failure.method is not None:
        fields["mcp_method"] = failure.method
    return fields


@dataclass
class TurnTally:
    """What a turn counts while it runs, and its one ``turn.ended`` line: the step timer started with the turn, and the
    counts, sizes and usage the line carries beside the timer's duration, outcome and reason, never a name or content.

    A sentence counts when it is sent to ``speak``, refused and cut ones included, and the first one marks the
    timer's first output. A silent round counts when its marker is read, a tool call when the model makes it.
    ``stop_code`` is the code of a stop the turn reported itself; ``answered`` says its final event came.
    """

    timer: StepTimer
    sentence_count: int = 0
    silent_count: int = 0
    tool_call_count: int = 0
    usage: TurnUsage | None = None
    sizes: InputSizes | None = None
    stop_code: str | None = None
    answered: bool = False

    def sentence_sent(self) -> None:
        self.sentence_count += 1
        self.timer.first_output()

    def end(self, ending: TurnEnding) -> None:
        """Writes the line once, at the level its outcome gives; a failed turn is an error."""
        self.timer.add(
            sentence_count=self.sentence_count, silent_count=self.silent_count, tool_call_count=self.tool_call_count
        )
        if self.sizes is not None:
            self.timer.add(
                tools_bytes=self.sizes.tools_bytes,
                told_bytes=self.sizes.told_bytes,
                results_bytes=self.sizes.results_bytes,
                images_bytes=self.sizes.images_bytes,
            )
        if self.usage is not None:
            self.timer.add(
                input_tokens=self.usage.input_tokens,
                cached_input_tokens=self.usage.cached_input_tokens,
                cache_write_tokens=self.usage.cache_write_tokens,
                output_tokens=self.usage.output_tokens,
                model_request_count=self.usage.requests,
            )
        match ending.outcome:
            case Outcome.OK:
                self.timer.end()
            case Outcome.REFUSED:
                self.timer.refuse(_reason_of(ending))
                self.timer.end()
            case Outcome.FAILED:
                self.timer.fail(_reason_of(ending))
                self.timer.end()
            case _:  # Outcome.CANCELLED, the last outcome
                self.timer.end_cancelled()


def _reason_of(ending: TurnEnding) -> str:
    assert ending.reason is not None  # noqa: S101 - a refused or failed ending always names its reason
    return ending.reason


def start_turn_tally(client_id: str, app: str) -> TurnTally:
    """The tally of a turn that starts now; its line carries the agent and the app's host."""
    return TurnTally(StepTimer("turn.ended", _LOG, agent_id=client_id, app_id=urlsplit(app).hostname or ""))


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
