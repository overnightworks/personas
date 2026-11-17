"""What the runner decides, with no process, network or clock: standings in, actions out.

The loop in ``loop`` performs these decisions; nothing here reads a clock or a socket, so every
rule is called with the values it judges. What his next request holds is ``present``'s, and which place has a turn
due is ``due``'s.

Within the pure chain it imports ``entries`` and ``recall`` and is imported by ``present`` and ``due``: ``entries``
<- ``recall`` <- ``decision`` <- ``present`` <- ``due``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

from agents.backend import TurnStopped
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    CONTINUE_SPEAKING_TOOL,
    SPEAK_TOOL,
    STOP_SPEAKING_TOOL,
    AcceptInvitationRefusal,
    PartialSpeech,
    PlaceEntry,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SpeakAnswer,
    SpeakRefusal,
    Standing,
    StandingState,
    VoiceState,
)
from agents.turn import FailureCode, RejectionCode
from logline import Outcome

from personas.entries import (
    ActOutcome,
    Answered,
    Lines,
    Occasion,
    Reading,
    SpeechOutcome,
    Unanswered,
)
from personas.failures import CallFailure, StatusFailure, TokenStatusFailure, UnusableAnswer, leaves_of
from personas.recall import RECALL_TOOL

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from agents.member import FigureRef, Invitation, VoiceName
    from agents.tools import ToolDeclaration, ToolOutcome
    from agents.turn import AgentRole

    from personas.secret import Secret

CALL_PLACE_KIND: Final = "call"
RESERVED_TOOLS: Final = frozenset(
    {ACCEPT_INVITATION_TOOL, SPEAK_TOOL, RECALL_TOOL, CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL}
)
"""The names the runner calls or answers itself in every turn: the accept of an invitation, the voice's ``speak``, the
lookup's ``recall`` and the two tools that decide a hold."""
RENEW_BEFORE_EXPIRY_SECONDS: Final = 60.0
"""A token is renewed once this little of its lifetime remains, so no call carries an expired one."""
NETWORK_FAILURE_CODE: Final = "network"
NO_TURNS_CODE: Final = "no_turns"
"""The reason an invitation is left out: the loop has no way to take turns at the place."""
BACKOFF_BASE_SECONDS: Final = 1.0
BACKOFF_CAP_SECONDS: Final = 30.0
DEFAULT_BUDGET_PAUSE_CAP_SECONDS: Final = 300.0
"""The longest wait after a turn the gateway refused for a spent budget, unless the file sets another."""
SUSPENDED_RETRY_SECONDS: Final = 30.0
"""How often a suspended loop, and a tools-only app the realm refused, ask the realm and then the app again."""
TURN_ROLE: Final[AgentRole] = "user"
"""The app role a turn acts under: an agent holds no admin right."""
SENTENCES_AHEAD: Final = 1
"""How many sentences he sends ahead of the one playing: the wire answers a sentence at its turn, in the order sent."""
TURN_CANCELLED_CODE: Final = "cancelled"
"""The code of an act whose call was cut while it was in flight, by a cancellation of its turn or by the turn's
deadline."""
DEFECT_REASON: Final = "defect"
"""The reason of a turn that an error ended which is no failure of a call."""
NO_FINAL_EVENT_REASON: Final = "no_final_event"
"""The reason of a turn whose events ended without a final event and without an error."""


class UnrecognisedRefusal(StrEnum):
    """A refusal outside the library's record: an app's own code, or an error text that is no such JSON."""

    UNRECOGNISED = "unrecognised"


type Refusal = AcceptInvitationRefusal | UnrecognisedRefusal


class Reaction(StrEnum):
    RENEW = "renew"
    BACKOFF = "backoff"


@dataclass(frozen=True)
class Suspend:
    """The loop suspends and logs this code."""

    code: str


@dataclass(frozen=True)
class Accept:
    place: PlaceRef
    invitation_id: str
    epoch: int
    figure: FigureRef | None = field(default=None, repr=False)
    voices: tuple[VoiceName, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class Token:
    value: Secret = field(repr=False)
    expires_at: float


@dataclass(frozen=True)
class Memory:
    """What a loop remembers: invitations refused as stale, the last other refusal of each, and the invitations it
    reported as left out."""

    stale: frozenset[str] = frozenset()
    refused: Mapping[str, Refusal] = field(default_factory=dict[str, Refusal])
    left_out: frozenset[str] = frozenset()


def _is_call(standing: Standing) -> bool:
    """A call place serves no view, so it is accepted whether the loop takes turns or not."""
    return standing.place.kind == CALL_PLACE_KIND and not standing.serves_view


def _has_window(standing: Standing) -> bool:
    """A place is read through its window when it is a call or serves a view, whose window holds what happened there."""
    return standing.place.kind == CALL_PLACE_KIND or standing.serves_view


def _accepted(standing: Standing, *, takes_turns: bool) -> bool:
    """A call place that serves no view is accepted whether the loop takes turns or not; any other place, and a call
    place that serves a view, only where it does, since an app marks an invited standing as serving no view before it
    is active."""
    if _is_call(standing):
        return True
    return takes_turns and (standing.serves_view or standing.place.kind != CALL_PLACE_KIND)


def _invited(standings: tuple[Standing, ...]) -> tuple[tuple[Standing, Invitation], ...]:
    return tuple(
        (standing, standing.invitation)
        for standing in standings
        if standing.state is StandingState.INVITED and standing.invitation is not None
    )


def accepts_for(
    standings: tuple[Standing, ...],
    memory: Memory,
    *,
    takes_turns: bool,
    figure: FigureRef | None = None,
    voices: tuple[VoiceName, ...] = (),
) -> tuple[Accept, ...]:
    """One accept for each invited call standing whose invitation was not refused as stale, and, where the loop can
    take turns, for each other invited place, whether or not it serves a view yet: a seated agent that cannot act
    leaves the place waiting. Every accept brings the agent's figure and, in his order, the names of his voices, if he
    has them."""
    return tuple(
        Accept(standing.place, invitation.id, invitation.epoch, figure, voices)
        for standing, invitation in _invited(standings)
        if _accepted(standing, takes_turns=takes_turns) and invitation.id not in memory.stale
    )


def left_out_for(
    standings: tuple[Standing, ...], memory: Memory, *, takes_turns: bool
) -> tuple[Memory, tuple[Standing, ...]]:
    """The invited standings ``accepts_for`` leaves out that were not reported yet, and the memory that holds them
    reported: an invitation is news once."""
    news = tuple(
        (standing, invitation)
        for standing, invitation in _invited(standings)
        if not _accepted(standing, takes_turns=takes_turns) and invitation.id not in memory.left_out
    )
    reported = memory.left_out | {invitation.id for _, invitation in news}
    return replace(memory, left_out=reported), tuple(standing for standing, _ in news)


def forget_gone(memory: Memory, standings: tuple[Standing, ...]) -> Memory:
    """Drop the last refusal and the report of an invitation the standings no longer hold; a stale one stays refused."""
    present = {standing.invitation.id for standing in standings if standing.invitation is not None}
    return Memory(
        stale=memory.stale,
        refused={invitation: code for invitation, code in memory.refused.items() if invitation in present},
        left_out=memory.left_out & present,
    )


def after_refusal(memory: Memory, accept: Accept, refusal: Refusal) -> tuple[Memory, bool]:
    """The memory after a refused accept and whether this refusal is news worth reporting."""
    if refusal is AcceptInvitationRefusal.STALE_INVITATION:
        return replace(memory, stale=memory.stale | {accept.invitation_id}), True
    is_news = memory.refused.get(accept.invitation_id) is not refusal
    return replace(memory, refused={**memory.refused, accept.invitation_id: refusal}), is_news


def renewal_due(token: Token | None, now: float) -> bool:
    return token is None or now >= token.expires_at - RENEW_BEFORE_EXPIRY_SECONDS


def _doubling_wait(count: int, cap_seconds: float) -> float:
    """The wait after the ``count``-th event in a row (from one): the backoff base, doubling up to ``cap_seconds``."""
    doublings = min(count - 1, math.ceil(math.log2(cap_seconds / BACKOFF_BASE_SECONDS)))
    return min(BACKOFF_BASE_SECONDS * 2**doublings, cap_seconds)


def backoff_delay(failures: int) -> float:
    """The wait after the ``failures``-th failure in a row (from one): doubling up to a cap."""
    return _doubling_wait(failures, BACKOFF_CAP_SECONDS)


def budget_pause(refusals: int, cap_seconds: float) -> float:
    """The wait before his next turn after the ``refusals``-th turn in a row the gateway refused for a spent budget
    (from one): doubling from the backoff base up to ``cap_seconds``."""
    return _doubling_wait(refusals, cap_seconds)


def status_code(status: int) -> str:
    return f"http_{status}"


def jsonrpc_code(code: int) -> str:
    return f"jsonrpc_{code}"


def failure_code(failure: CallFailure) -> str:
    """The code a failure logs: the HTTP status of a server error, the code of an unusable answer, else a fixed word
    for the network."""
    if isinstance(failure, StatusFailure):
        return status_code(failure.status)
    if isinstance(failure, UnusableAnswer):
        return failure.code
    return NETWORK_FAILURE_CODE


_JSONRPC_ERROR_REASONS: Final = {
    -32700: "parse_error",
    -32600: "invalid_request",
    -32601: "method_not_found",
    -32602: "invalid_params",
    -32603: "internal_error",
}
_MEDIA_TYPE_REASONS: Final = {"application/json": "json", "text/html": "html", "text/plain": "plain_text"}
_JSONRPC_CODE: Final = re.compile(r"jsonrpc_(-?\d+)")
NO_ANSWER_REASON: Final = "no_answer"
"""The reason of an error status that carried neither a JSON-RPC error nor a media type."""
OTHER_JSONRPC_REASON: Final = "other_jsonrpc_error"
OTHER_MEDIA_TYPE_REASON: Final = "other_media_type"
"""The one word of every media type no reason names: an app's content type is no closed set."""
UNEXPECTED_STATUS_REASON: Final = "unexpected_status"
"""The reason of a redirect, which no call expects."""


def _jsonrpc_reason(code: str) -> str | None:
    matched = _JSONRPC_CODE.fullmatch(code)
    return None if matched is None else _JSONRPC_ERROR_REASONS.get(int(matched[1]), OTHER_JSONRPC_REASON)


def _answer_reason(answer: str | None) -> str:
    """The reason an error status's answer gives: the JSON-RPC error by name, a known media type by name, else a
    fixed word; an app's own value never reaches the line."""
    if answer is None:
        return NO_ANSWER_REASON
    return _jsonrpc_reason(answer) or _MEDIA_TYPE_REASONS.get(answer, OTHER_MEDIA_TYPE_REASON)


def failure_reason(failure: CallFailure) -> str:
    """The closed reason a failure logs; its status and failed method are logged beside it."""
    if isinstance(failure, StatusFailure):
        return _answer_reason(failure.answer)
    if isinstance(failure, UnusableAnswer):
        if failure.code.startswith("http_"):
            return UNEXPECTED_STATUS_REASON
        return _jsonrpc_reason(failure.code) or failure.code
    return NETWORK_FAILURE_CODE


def error_reason(error: BaseException) -> str:
    """The closed reason an error that ended a step or turn logs: the reason of the failure that decides the loop's
    reaction, or a defect."""
    failure = decisive_failure(tuple(leaves_of(error)), already_renewed=False)
    return DEFECT_REASON if failure is None else failure_reason(failure)


@dataclass(frozen=True)
class TurnEnding:
    """How a turn ended: its outcome, and the closed reason of one that did not end ``ok`` or cancelled."""

    outcome: Outcome
    reason: str | None = None


def turn_ending(*, cancelled: bool, error: BaseException | None, stop_code: str | None, answered: bool) -> TurnEnding:
    """The ending of a turn from what its task ended with and what the turn recorded.

    A turn that got its final event is ok, whatever its teardown then met. Otherwise a cancellation, whoever asked for
    it and even before the task's first step, is cancelled. A stop the turn reported carries its code: a refusal of
    the gateway is refused, any other stop failed. An error that reached the task is
    failed with the reason of the failure that decides the loop's reaction, or a defect. A turn whose events ended
    without a final event and without an error is failed for it.
    """
    if answered:
        return TurnEnding(Outcome.OK)
    if cancelled:
        return TurnEnding(Outcome.CANCELLED)
    if stop_code is not None:
        refused = stop_code in {each.value for each in RejectionCode}
        return TurnEnding(Outcome.REFUSED if refused else Outcome.FAILED, stop_code)
    if error is not None:
        return TurnEnding(Outcome.FAILED, error_reason(error))
    return TurnEnding(Outcome.FAILED, NO_FINAL_EVENT_REASON)


def reaction_to(failure: CallFailure, *, already_renewed: bool) -> Reaction | Suspend:
    """What the loop does after a failure: a refusal suspends with its code, the rest backs off.

    A server error and a failure of the network back off. An app's first 401 renews the token once.
    """
    match failure:
        case UnusableAnswer(code=code):
            return Suspend(code)
        case StatusFailure(status=status):
            if status >= 500:
                return Reaction.BACKOFF
            if status == 401 and not already_renewed and not isinstance(failure, TokenStatusFailure):
                return Reaction.RENEW
            return Suspend(status_code(status))
        case _:
            return Reaction.BACKOFF


def refusal_retry_at(now: float) -> float:
    """When an app the realm refused an agent is asked again, from the ``now`` of the refusal."""
    return now + SUSPENDED_RETRY_SECONDS


def is_not_granted(failure: CallFailure) -> bool:
    """Whether a failure says the realm does not grant the agent an app: the realm refuses its token request, or the
    app itself answers 401 or 403."""
    match failure:
        case TokenStatusFailure(status=status):
            return status < HTTPStatus.INTERNAL_SERVER_ERROR
        case StatusFailure(status=status):
            return status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
        case _:
            return False


def is_voice_app(has_voice: bool, declared: Iterable[ToolDeclaration]) -> bool:
    """Whether the turns at the call places of an app are spoken: its agent has a voice and the app declares
    ``speak``."""
    return has_voice and any(each.name == SPEAK_TOOL for each in declared)


def speaks_aloud(voice_app: bool, occasion: Occasion, checkpoint: Checkpoint | None) -> bool:
    """Whether a turn is spoken: it is for lines, which a call place gives, in the app of a voice, unless the last
    window of the place states that his voice is unavailable there. A checkpoint that is no reading states nothing."""
    return voice_app and isinstance(occasion, Lines) and not _voice_is_unavailable(checkpoint)


def _voice_is_unavailable(checkpoint: Checkpoint | None) -> bool:
    return isinstance(checkpoint, Reading) and checkpoint.voice_state is VoiceState.UNAVAILABLE


def remaining_seconds(now: float, started: float, max_duration_seconds: float) -> float:
    """What is left of the longest duration of a turn that started at ``started``, at ``now`` on one monotonic scale;
    never below zero."""
    return max(0.0, max_duration_seconds - (now - started))


def past_deadline(now: float, started: float, max_duration_seconds: float) -> bool:
    """Whether a turn that started at ``started`` has used its longest duration at ``now``."""
    return remaining_seconds(now, started, max_duration_seconds) <= 0.0


def is_defect(leaf: BaseException) -> bool:
    """Whether a leaf is a defect: any leaf that is no failure of a call."""
    return not isinstance(leaf, CallFailure)


def decisive_failure(leaves: Sequence[BaseException], *, already_renewed: bool) -> CallFailure | None:
    """The one failure that decides the reaction to several: a refusal before a renewal before a backoff.

    With a defect among the leaves no failure decides. Of equal ranks the first wins.
    """
    if any(is_defect(leaf) for leaf in leaves):
        return None
    failures = [leaf for leaf in leaves if isinstance(leaf, CallFailure)]
    return min(failures, key=lambda failure: _rank(reaction_to(failure, already_renewed=already_renewed)))


def _rank(reaction: Reaction | Suspend) -> int:
    match reaction:
        case Suspend():
            return 0
        case Reaction.RENEW:
            return 1
        case _:
            return 2


@dataclass(frozen=True)
class Restarted:
    """This process did not join the place: what its window holds was said before and is never taken."""


@dataclass(frozen=True)
class Joined:
    """The agent's own accept just made it a member: its window holds every line said since."""


@dataclass(frozen=True)
class Paused:
    """The sponsor is away: lines said meanwhile are never taken."""


type Checkpoint = Restarted | Joined | Paused | Reading


def sync_checkpoints(
    checkpoints: Mapping[PlaceRef, Checkpoint], standings: tuple[Standing, ...], before: tuple[Standing, ...]
) -> dict[PlaceRef, Checkpoint]:
    """The checkpoints after a standing read, given the standing read before it: an active place keeps its join, since
    the window of an accepted place is not read yet, whether or not it serves a view; a window place paused is paused,
    one active keeps its entry or starts as restarted, one that serves a view again after an active read without one
    starts as restarted, and a place that is gone, invited again or no window place any more has none."""
    viewless = frozenset(
        standing.place for standing in before if standing.state is StandingState.ACTIVE and not standing.serves_view
    )
    synced: dict[PlaceRef, Checkpoint] = {}
    for standing in standings:
        held = checkpoints.get(standing.place)
        if standing.state is StandingState.ACTIVE and isinstance(held, Joined):
            synced[standing.place] = held
        elif not _has_window(standing):
            continue
        elif standing.state is StandingState.PAUSED:
            synced[standing.place] = Paused()
        elif standing.state is StandingState.ACTIVE:
            returning = standing.serves_view and standing.place in viewless
            synced[standing.place] = Restarted() if returning else (held or Restarted())
    return synced


def sets_where_reading_starts(checkpoint: Checkpoint) -> bool:
    """A read of a restarted or paused place only sets its checkpoint: nothing in it is taken, so it starts no turn."""
    return isinstance(checkpoint, Restarted | Paused)


def active_window_places(standings: tuple[Standing, ...]) -> tuple[PlaceRef, ...]:
    return tuple(
        standing.place for standing in standings if standing.state is StandingState.ACTIVE and _has_window(standing)
    )


def active_view_places(standings: tuple[Standing, ...]) -> tuple[PlaceRef, ...]:
    return tuple(
        standing.place for standing in standings if standing.state is StandingState.ACTIVE and standing.serves_view
    )


def sync_turned_revisions(turned: Mapping[PlaceRef, int], standings: tuple[Standing, ...]) -> dict[PlaceRef, int]:
    """The revisions after a standing read: a place that is no active view place any more is forgotten."""
    active = set(active_view_places(standings))
    return {place: revision for place, revision in turned.items() if place in active}


def view_turn_due(turned_revision: int | None, view: PlaceView) -> bool:
    """A turn is due while the view waits for the agent at a revision no turn has been started for yet.

    The turn's own outcome does not count, except that a turn ended without an answer makes its place due again
    (``due_again``): an app that wants him to act again moves the revision.
    """
    return view.awaits_you and turned_revision != view.revision


def holds_a_place(standings: tuple[Standing, ...]) -> bool:
    """Whether any of his standings at an app is active."""
    return any(standing.state is StandingState.ACTIVE for standing in standings)


def lends_tools(binds_tools: bool, standings: tuple[Standing, ...]) -> bool:
    """Whether the tools of an app are held by other apps' turns: it binds tools and he holds a place there."""
    return binds_tools and holds_a_place(standings)


def held_apps(own: str, placed: frozenset[str]) -> frozenset[str]:
    """The apps a turn at ``own`` also holds the tools of: every other app where he holds an active place."""
    return placed - {own}


def turn_invalid(place: PlaceRef, standings: tuple[Standing, ...]) -> bool:
    """A turn ends with its place: it is invalid once the place is gone, invited again or paused."""
    return not any(standing.place == place and standing.state is StandingState.ACTIVE for standing in standings)


def _unanswered(ended: BaseException, *, cancelling: bool) -> Unanswered | None:
    """The outcome of a request that ended without an answer. A cancellation can come out of a request as a failure,
    so any ending while cancelling is a request cut in flight. Without a cancel a failure is told by its code, and any
    other error is a defect, which records nothing."""
    if cancelling:
        return Unanswered(TURN_CANCELLED_CODE)
    if isinstance(ended, CallFailure):
        return Unanswered(failure_code(ended))
    return None


def act_outcome(ended: ToolOutcome | BaseException, *, cancelling: bool) -> ActOutcome | None:
    """The outcome of a relayed call from how its request ended and whether the turn's task is being cancelled.

    An answer is an answer even while a cancel is pending.
    """
    if not isinstance(ended, BaseException):
        return Answered(ended.content, ended.is_error, image_shown=ended.image is not None)
    return _unanswered(ended, cancelling=cancelling)


def speech_outcome(ended: SpeakAnswer | SpeakRefusal | BaseException, *, cancelling: bool) -> SpeechOutcome | None:
    """The outcome of a sentence sent to ``speak`` from how its request ended and whether the turn's task is being
    cancelled. The app's answer, a refusal included, is an answer even while a cancel is pending; a request given up
    at the turn's deadline is unanswered with that code."""
    if isinstance(ended, SpeakAnswer | SpeakRefusal):
        return ended
    if not cancelling and isinstance(ended, TurnStopped) and ended.code is FailureCode.LIMIT_EXCEEDED:
        return Unanswered(ended.code.value)
    return _unanswered(ended, cancelling=cancelling)


def may_send_next(owed: int) -> bool:
    """Whether his next sentence may go to ``speak`` while ``owed`` own spoken lines, one per sentence answered as
    spoken, have not entered: the wire holds a sentence back until the one before it ends, so only the sentence playing
    may still owe its line."""
    return owed <= SENTENCES_AHEAD


@dataclass(frozen=True)
class Perception:
    """What reading a window from a checkpoint yields: the checkpoint after it, the lines taken and whether older ones
    were never seen."""

    checkpoint: Reading
    new: tuple[PlaceEntry, ...]
    missed: bool


def perceive(window: PlaceWindow, checkpoint: Checkpoint) -> Perception:
    """Reads a window from a checkpoint: a restart or a pause takes nothing, a join and a new run take every line."""
    if sets_where_reading_starts(checkpoint):
        highest = window.entries[-1].seq if window.entries else window.starts_at - 1
        return Perception(Reading(window.generation, highest, window.voice_state), (), missed=False)
    match checkpoint:
        case Reading(generation=window.generation, last_seq=last_seq):
            return _take_after(window, last_seq)
        case _:
            return _take_after(window, 0)


def _take_after(window: PlaceWindow, last_seq: int) -> Perception:
    new = tuple(line for line in window.entries if line.seq > last_seq)
    return Perception(
        Reading(window.generation, new[-1].seq if new else last_seq, window.voice_state),
        new,
        missed=window.starts_at > last_seq + 1,
    )


def waking_line(new: Sequence[PlaceEntry], *, serves_view: bool) -> PlaceEntry | None:
    """The newest of the lines taken at a place that makes a turn due, none when none does: at a place with a view none
    does, since its view alone calls him; at any other place a line that is neither his own nor an act does."""
    if serves_view:
        return None
    return next((line for line in reversed(new) if not (line.own or line.act)), None)


def offered_declarations(
    declared: Sequence[ToolDeclaration], taken: frozenset[str]
) -> tuple[tuple[ToolDeclaration, ...], int]:
    """The declarations an app offers to a turn in its listing's order, and how many names it offers that an earlier
    app already holds. The runner's reserved names are never offered and never counted."""
    offered = tuple(each for each in declared if each.name not in RESERVED_TOOLS)
    kept = tuple(each for each in offered if each.name not in taken)
    return kept, len(offered) - len(kept)


def turn_call_bound(bounds: Iterable[int]) -> int | None:
    """The calls one turn may make: the sum of the bounds of the apps whose tools it holds, plus one, and none when it
    holds no tool. The library counts every call of a model reply, so the call a turn holds back counts beside the
    calls it makes."""
    held = tuple(bounds)
    return sum(held) + 1 if held else None


def call_within_bound(made: int, bound: int) -> bool:
    """Whether an app whose bound is ``bound`` takes one more call after ``made``."""
    return made < bound


@dataclass(frozen=True)
class AppPlace:
    """One of an agent's places, whichever of his apps it is in."""

    app: str
    place: PlaceRef


@dataclass(frozen=True)
class Held:
    """A hold as the window states it: its id and what people said over him, which stays what the last read that had
    a record said once the record is gone."""

    id: str
    speech: tuple[PartialSpeech, ...]
    record_gone: bool


class HoldOutcome(StrEnum):
    """How a hold ended on the runner's side, as the one line per hold id reports it."""

    CONTINUED = "continued"
    STOPPED = "stopped"
    UNDECIDED = "undecided"
    NO_TURN = "no_turn"


def hold_stated(window: PlaceWindow, seen: Held | None) -> Held | None:
    """The hold the window states, or ``None`` when he is not held; a hold whose record has gone keeps the words an
    earlier read of the same hold had."""
    hold = window.interrupted
    if hold is None:
        return None
    talked_over = window.talked_over
    kept = seen.speech if seen is not None and seen.id == hold.id else ()
    return Held(hold.id, kept if talked_over is None else talked_over.speech, record_gone=talked_over is None)


def hold_tellable(held: Held) -> bool:
    """Whether a hold may be told: a person has said words over him, or the record is gone and no more will come."""
    return held.record_gone or any(each.text for each in held.speech)


def untold_words(said: Iterable[PartialSpeech], told: Iterable[PartialSpeech]) -> tuple[PartialSpeech, ...]:
    """What people have said over him that he was not told: each partial with words that differs from the one of the
    same name that was told."""
    known = {(each.name, each.text) for each in told}
    return tuple(each for each in said if each.text and (each.name, each.text) not in known)


def round_seconds(remaining: float) -> int:
    """The whole seconds a decision round may last, at least one: the library takes only a positive whole bound."""
    return max(1, math.ceil(remaining))
