"""State changes of a loop, reported once each; the one place that decides what a log line holds."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlsplit

from agents.member_outcomes import StatusFailure, UnusableAnswer
from anyio import BrokenResourceError, ClosedResourceError
from logline import Outcome, StepTimer
from typing_extensions import TypedDict

from personas.decision import error_reason, failure_reason

if TYPE_CHECKING:
    from collections.abc import Generator

    from agents.member_outcomes import ArgumentLeftOut, CallFailure
    from agents.turn import TurnUsage

    from personas.config import AppTarget
    from personas.decision import TurnEnding
    from personas.entries import InputSizes

LOG: Final = logging.getLogger("personas")
CUT_FOR_SPOKEN: Final = "spoken"
"""The value of the ``cut`` field of a ``turn.ended`` line of a written turn that ended for a spoken place."""

# The mcp client logs "Error parsing SSE message" at ERROR when its stream closes under a request, which a recovered
# blip does too; the library's own comment names that case as no parse failure. The loop reports the failure itself
# (`loop.backed_off`), so the closed-stream line drops to INFO and any other error of that logger stays at ERROR.
MCP_CLIENT_LOGGER: Final = "mcp.client.streamable_http"


def demote_a_closed_stream(record: logging.LogRecord) -> bool:
    """A logging filter: a record of a closed or broken stream is written at INFO; it always lets the record pass."""
    if record.exc_info and isinstance(record.exc_info[1], ClosedResourceError | BrokenResourceError):
        record.levelno, record.levelname = logging.INFO, logging.getLevelName(logging.INFO)
    return True


class Event(Enum):
    STARTED = auto()
    SUSPENDED = auto()
    RESUMED = auto()
    BACKOFF = auto()
    JOINED = auto()
    REFUSED = auto()
    INVITATION_LEFT_OUT = auto()
    INVITATION_DECLINED = auto()
    DECLINE_REFUSED = auto()
    PAUSED = auto()
    SPEAK_REFUSED = auto()
    SPEECH_DROPPED = auto()
    HOLD_DECIDED = auto()
    TURN_STATE_FAILED = auto()
    TURN_STATE_LOST = auto()
    TOOLS_ONLY_APP_FAILED = auto()
    HELD_APP_FAILED = auto()
    SKILL_REFUSED = auto()
    TOOLS_LEFT_OUT = auto()
    STORE_FAILED = auto()
    STORE_RESUMED = auto()
    SWEPT = auto()
    TAKEN = auto()
    LOST = auto()
    STOP_OVERRAN = auto()
    STOP_CUT = auto()


@dataclass(frozen=True)
class StateChange:
    """One change of an agent's loop for one app, with the refusal or failure code if any.

    The sweep belongs to no agent and no app, and says how many lines it removed. A stop that its deadline cut belongs
    to the runner alone. The lock of an agent taken, lost or slow to stop belongs to no app. The start says the model
    the agent's turns run on, when it has a persona. An invitation the loop leaves out says its place's kind and the
    reason as the code. A pause of his turns says its reason as the code and its wait in seconds. Tools left out for an
    earlier app's of the same name say how many. Sentences dropped unsent say how many and why, as the code. Lines
    that never came say how many. A hold decided says its id. A call that failed says its
    failure: the line holds its HTTP status, the MCP method that failed and a closed reason.
    """

    event: Event
    client_id: str | None
    app: str | None
    code: str | None = None
    lines: int | None = None
    place_kind: str | None = None
    seconds: float | None = None
    model: str | None = None
    tool_count: int | None = None
    failure: CallFailure | None = None
    sentences: int | None = None
    hold_id: str | None = None


type Reporter = Callable[[StateChange], None]


def failure_event(app: AppTarget) -> Event:
    """The event of a call to ``app`` that failed: the held app's or a tools-only app's."""
    return Event.HELD_APP_FAILED if app.listens else Event.TOOLS_ONLY_APP_FAILED


class LoopReport(Protocol):
    """The loop's report of one change of the agent at an app, which the turns call."""

    def __call__(
        self,
        event: Event,
        code: str | None = None,
        app: AppTarget | None = None,
        failure: CallFailure | None = None,
        *,
        hold_id: str | None = None,
    ) -> None: ...


_WRITE_OF: Final[dict[Event, Callable[[dict[str, str | int | float]], None]]] = {
    Event.STARTED: lambda fields: LOG.info("loop.started", extra=fields),
    Event.SUSPENDED: lambda fields: LOG.info("loop.suspended", extra=fields),
    Event.RESUMED: lambda fields: LOG.info("loop.resumed", extra=fields),
    Event.BACKOFF: lambda fields: LOG.info("loop.backed_off", extra=fields),
    Event.JOINED: lambda fields: LOG.info("invitation.accepted", extra=fields),
    Event.REFUSED: lambda fields: LOG.info("invitation.refused", extra=fields),
    Event.INVITATION_LEFT_OUT: lambda fields: LOG.info("invitation.left_out", extra=fields),
    Event.INVITATION_DECLINED: lambda fields: LOG.info("invitation.declined", extra=fields),
    Event.DECLINE_REFUSED: lambda fields: LOG.info("invitation.decline_refused", extra=fields),
    Event.PAUSED: lambda fields: LOG.info("turns.paused", extra=fields),
    Event.SPEAK_REFUSED: lambda fields: LOG.info("speak.refused", extra=fields),
    Event.SPEECH_DROPPED: lambda fields: LOG.info("speech.dropped", extra=fields),
    Event.HOLD_DECIDED: lambda fields: LOG.info("hold.decided", extra=fields),
    Event.TURN_STATE_FAILED: lambda fields: LOG.warning("turn_state.failed", extra=fields),
    Event.TURN_STATE_LOST: lambda fields: LOG.info("turn_state.lost", extra=fields),
    Event.TOOLS_ONLY_APP_FAILED: lambda fields: LOG.warning("tools_only_app.failed", extra=fields),
    Event.HELD_APP_FAILED: lambda fields: LOG.warning("held_app.failed", extra=fields),
    Event.SKILL_REFUSED: lambda fields: LOG.warning("skill.refused", extra=fields),
    Event.TOOLS_LEFT_OUT: lambda fields: LOG.info("tools.left_out", extra=fields),
    Event.STORE_FAILED: lambda fields: LOG.warning("store.failed", extra=fields),
    Event.STORE_RESUMED: lambda fields: LOG.info("store.resumed", extra=fields),
    Event.SWEPT: lambda fields: LOG.info("lines.swept", extra=fields),
    Event.TAKEN: lambda fields: LOG.info("agent.taken", extra=fields),
    Event.LOST: lambda fields: LOG.warning("agent.lost", extra=fields),
    Event.STOP_OVERRAN: lambda fields: LOG.warning("agent.stop_overran", extra=fields),
    Event.STOP_CUT: lambda fields: LOG.warning("runner.stop_cut", extra=fields),
}


def log_state_change(change: StateChange) -> None:
    """Writes the fields of ``_fields_of`` and nothing else: no secret, token or tool content reaches a line."""
    fields = _fields_of(change)
    _WRITE_OF[change.event](fields)


def _fields_of(change: StateChange) -> dict[str, str | int | float]:
    fields: dict[str, str | int | float] = {}
    if change.client_id is not None:
        fields["agent_id"] = change.client_id
    if change.app is not None:
        fields["app_id"] = app_host(change.app)
    fields |= _failure_fields(change)
    if change.lines is not None:
        fields["line_count"] = change.lines
    if change.place_kind is not None:
        fields["place_kind"] = change.place_kind
    if change.seconds is not None:
        fields["pause_ms"] = round(change.seconds * 1000, 3)
    if change.tool_count is not None:
        fields["tool_count"] = change.tool_count
    if change.sentences is not None:
        fields["sentence_count"] = change.sentences
    if change.model is not None:
        fields["model"] = change.model
    if change.hold_id is not None:
        fields["hold_id"] = change.hold_id
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

    A sentence counts when the turn's reply releases it, whether it is ever sent or not, and the first one marks the
    timer's first output. A silent round counts when its marker is read, a tool call when the model makes it.
    ``stop_code`` is the code of a stop the turn reported itself; ``answered`` says its final event came;
    ``cut_for_spoken`` says the loop ended it at a safe point for a spoken place.
    """

    timer: StepTimer
    sentence_count: int = 0
    silent_count: int = 0
    tool_call_count: int = 0
    usage: TurnUsage | None = None
    sizes: InputSizes | None = None
    stop_code: str | None = None
    answered: bool = False
    cut_for_spoken: bool = False

    def sentence_released(self) -> None:
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
                if self.cut_for_spoken:
                    self.timer.add(cut=CUT_FOR_SPOKEN)
                self.timer.end_cancelled()


def _reason_of(ending: TurnEnding) -> str:
    assert ending.reason is not None  # noqa: S101 - a refused or failed ending always names its reason
    return ending.reason


def start_turn_tally(client_id: str, app: str) -> TurnTally:
    """The tally of a turn that starts now; its line carries the agent and the app's host."""
    return TurnTally(StepTimer("turn.ended", LOG, **agent_app_fields(client_id, app)))


class DueOccasion(StrEnum):
    LINE = "line"
    VIEW = "view"
    HOLD = "hold"


def report_fell_due(agent_id: str, app: str, occasion: DueOccasion, *, spoken: bool, read_ms: int) -> None:
    """Writes ``turn.fell_due``: a pass made a place due that was not due. The line carries ids, the occasion's kind,
    whether the turn would be spoken and the milliseconds the pass had read the standing, never a line's text, kind or
    a name. The caller's current trace is the line's."""
    LOG.info(
        "turn.fell_due",
        extra={
            **agent_app_fields(agent_id, app),
            "occasion": occasion.value,
            "spoken": spoken,
            "read_ms": read_ms,
        },
    )


def report_argument_left_out(agent_id: str, app: str, left_out: ArgumentLeftOut) -> None:
    """Writes ``argument.left_out``: the runner sent a call without an optional argument its app's tool does not
    declare. The line carries the agent, the app's host, the tool's name and the argument's name, never a value.
    The caller's current trace is the line's."""
    LOG.info(
        "argument.left_out",
        extra={
            **agent_app_fields(agent_id, app),
            "tool": left_out.tool,
            "argument": left_out.argument,
        },
    )


def app_host(app_url: str) -> str:
    """The host of an app's address: the only part of it a line names, never a path or a query."""
    return urlsplit(app_url).hostname or ""


class AgentAppFields(TypedDict, closed=True):
    agent_id: str
    app_id: str


def agent_app_fields(agent_id: str, app_url: str) -> AgentAppFields:
    """The fields that name an agent and its app's host, the same on every line that names both."""
    return AgentAppFields(agent_id=agent_id, app_id=app_host(app_url))


def began_ago(seconds: float) -> float | None:
    """The start of a step that began ``seconds`` ago, as the caller's clock measured it, for ``StepTimer``'s
    ``started_at``; none when it just began."""
    # logline counts a step's duration on the time.perf_counter scale, so an elapsed time enters it only here.
    return time.perf_counter() - seconds if seconds else None


@contextmanager
def stepping(timer: StepTimer, *, turn_goes_on: bool = False) -> Generator[StepTimer]:
    """Ends the timer's one line when the block does: an error ends it failed with its closed reason, a cancellation
    cancelled, and the timer's own ``end`` inside the block writes it earlier. A failure the unit survives
    (``turn_goes_on``) is written as a handled one. An unusable answer that names the entry of a listing which broke a
    rule adds that index as ``entry``, never the entry."""
    with timer:
        if turn_goes_on:
            timer.handled()
        try:
            yield timer
        except Exception as error:
            timer.fail(error_reason(error))
            if isinstance(error, UnusableAnswer) and error.entry is not None:
                timer.add(entry=error.entry)
            raise


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
