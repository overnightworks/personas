"""What the runner decides, with no process, network or clock: standings in, actions out.

The loop in ``loop`` performs these decisions; nothing here reads a clock or a socket, so every
rule is called with the values it judges.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agents.member import (
    SPEAK_TOOL,
    AcceptInvitationRefusal,
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
from agents.turn import MAX_EVENT_BYTES

from personas.entries import (
    ActDone,
    ActOutcome,
    Answered,
    Entry,
    History,
    KeptLine,
    SaidAloud,
    SpeechOutcome,
    Unanswered,
    ViewSeen,
    Visit,
    entry_size,
    newest_that_fit,
    told_bytes,
    told_entry,
    told_place,
)
from personas.failures import CallFailure, StatusFailure, TokenStatusFailure, UnusableAnswer
from personas.recall import EarlierToday, told_earlier_today

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from datetime import datetime

    from agents.member import FigureRef, Invitation, VoiceName
    from agents.tools import ToolDeclaration, ToolOutcome
    from agents.turn import AgentRole

    from personas.secret import Secret

CALL_PLACE_KIND: Final = "call"
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
"""How often a suspended loop asks the realm and then the app again."""
TURN_ROLE: Final[AgentRole] = "user"
"""The app role a turn acts under: an agent holds no admin right."""
TURN_CANCELLED_CODE: Final = "cancelled"
"""The code of a turn the loop cancelled because its place left the standing, and of an act whose call was cut while it
was in flight, by that or by the turn's deadline."""
SPOKEN_WAIT_BOUND_SECONDS: Final = 30.0
"""The longest the voice waits after a sentence, whatever length the app says it plays."""
EARLIER_TODAY_KEY: Final = "earlier_today"


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
    voice: VoiceName | None = field(default=None, repr=False)


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
    voice: VoiceName | None = None,
) -> tuple[Accept, ...]:
    """One accept for each invited call standing whose invitation was not refused as stale, and, where the loop can
    take turns, for each other invited place, whether or not it serves a view yet: a seated agent that cannot act
    leaves the place waiting. Every accept brings the agent's figure and the name of his voice, if he has them."""
    return tuple(
        Accept(standing.place, invitation.id, invitation.epoch, figure, voice)
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


def spoken_wait_seconds(duration_ms: int) -> float:
    """How long the voice waits after a sentence the app says plays for ``duration_ms``."""
    return min(duration_ms / 1000, SPOKEN_WAIT_BOUND_SECONDS)


def past_deadline(now: float, started: float, max_duration_seconds: float) -> bool:
    """Whether a turn that started at ``started`` has used its longest duration at ``now``, on one monotonic scale."""
    return now - started >= max_duration_seconds


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


@dataclass(frozen=True)
class Reading:
    """What the last read of a window stated: its generation, the last seq taken and, when the app stated one, the
    state of his voice at the place."""

    generation: str
    last_seq: int
    voice_state: VoiceState | None = None


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
    cancelled. The app's answer, a refusal included, is an answer even while a cancel is pending."""
    if isinstance(ended, SpeakAnswer | SpeakRefusal):
        return ended
    return _unanswered(ended, cancelling=cancelling)


def _revision_key(view: ViewSeen) -> tuple[str, PlaceRef, int]:
    return view.app, view.place, view.view.revision


def _holds_revision(entries: Iterable[Entry], view: ViewSeen) -> bool:
    return any(isinstance(entry, ViewSeen) and _revision_key(entry) == _revision_key(view) for entry in entries)


def _not_his_own(entry: Entry) -> bool:
    return not isinstance(entry, ActDone | SaidAloud) and not (isinstance(entry, KeptLine) and entry.own)


def history_after(history: History, new: Iterable[Entry], size_bytes: int) -> History:
    """The history with ``new`` appended, a view revision it already holds adding nothing and an act always added; past
    ``size_bytes`` the oldest entries are left out, but the newest always stays."""
    entries = history.entries
    heard = history.heard
    for entry in new:
        if not (isinstance(entry, ViewSeen) and _holds_revision(entries, entry)):
            entries = (*entries, entry)
            heard += _not_his_own(entry)
    leaving = max(0, len(entries) - max(1, newest_that_fit(map(entry_size, reversed(entries)), size_bytes)))
    return History(entries[leaving:], history.left_out + leaving, heard)


@dataclass(frozen=True)
class Arrived:
    """The entries that entered a history since an earlier value of it, and whether any of them left again by the
    bound."""

    entries: tuple[Entry, ...] = field(repr=False)
    older_dropped: bool


def arrived_since(current: History, admitted: History) -> Arrived | None:
    """What entered ``current`` since ``admitted``, an earlier value of the same history, when any of it is not his own;
    otherwise ``None``. His own lines, his acts and his spoken sentences alone arrive as nothing."""
    if current.heard == admitted.heard:
        return None
    return Arrived(current.entries[max(0, admitted.taken - current.left_out) :], current.left_out > admitted.taken)


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


def wakes_him(new: Iterable[PlaceEntry], *, serves_view: bool) -> bool:
    """Whether the lines taken at a place make a turn due: at a place with a view none does, since its view alone calls
    him; at any other place a line that is neither his own nor an act does."""
    return not serves_view and any(not (line.own or line.act) for line in new)


def bound_declarations(declared: Sequence[ToolDeclaration], names: Sequence[str]) -> tuple[ToolDeclaration, ...]:
    """The declarations of the bound tools the app lists, in binding order; a bound tool it does not list is left out
    and a listed tool that is not bound never enters."""
    by_name = {declaration.name: declaration for declaration in declared}
    return tuple(by_name[name] for name in names if name in by_name)


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
class Lines:
    """Why a written turn runs: the lines of one visit's window taken since his last turn there, which the history
    holds, and whether older ones were never seen."""

    generation: str
    seqs: frozenset[int]
    missed: bool


@dataclass(frozen=True)
class Waiting:
    """Why a view turn runs: the place's view waits for the agent at this revision, which the history holds."""

    revision: int


type Occasion = Lines | Waiting


@dataclass(frozen=True)
class AppPlace:
    """One of an agent's places, whichever of his apps it is in."""

    app: str
    place: PlaceRef


@dataclass(frozen=True)
class Turns:
    """Whether a turn of one agent runs, and which places have a turn due, in the order they became due, with what the
    turn is for."""

    running: bool = False
    due: Mapping[AppPlace, Occasion] = field(default_factory=dict[AppPlace, Occasion])


def _joined(held: Occasion | None, occasion: Occasion) -> Occasion:
    if isinstance(held, Lines) and isinstance(occasion, Lines) and held.generation == occasion.generation:
        return Lines(held.generation, held.seqs | occasion.seqs, held.missed or occasion.missed)
    return occasion


def with_due(turns: Turns, at: AppPlace, occasion: Occasion) -> Turns:
    """A turn due at ``at`` for ``occasion``: a place already due keeps its position, lines of one generation join, and
    lines of another generation or a newer view revision replace what was due."""
    return Turns(turns.running, {**turns.due, at: _joined(turns.due.get(at), occasion)})


def due_again(turns: Turns, at: AppPlace, occasion: Occasion) -> Turns:
    """A turn due again at ``at`` for ``occasion``, which a turn ended without an answer: a place due meanwhile keeps
    what is newer, except that lines of the same generation join; any other place goes to the back of the order."""
    held = turns.due.get(at)
    if held is None or (
        isinstance(held, Lines) and isinstance(occasion, Lines) and held.generation == occasion.generation
    ):
        return with_due(turns, at, occasion)
    return turns


def without_due(turns: Turns, leaving: Callable[[AppPlace], bool]) -> Turns:
    """The turns without the due places for which ``leaving`` holds."""
    return Turns(turns.running, {at: occasion for at, occasion in turns.due.items() if not leaving(at)})


def claimed(turns: Turns, app: str) -> tuple[Turns, AppPlace, Occasion] | None:
    """The turns after a turn starts at the place that has waited longest, with that place and what the turn is for;
    none while a turn runs or when that place is not in ``app``."""
    if turns.running or not turns.due:
        return None
    at, occasion = next(iter(turns.due.items()))
    if at.app != app:
        return None
    return Turns(True, without_due(turns, lambda each: each == at).due), at, occasion


def released(turns: Turns) -> Turns:
    """The turns once the running turn ended; what is due stays."""
    return Turns(False, turns.due)


def left_standing(app: str, standings: tuple[Standing, ...]) -> Callable[[AppPlace], bool]:
    """Whether a due place is one of ``app`` that is no active window place any more; a view place is one."""
    active = {AppPlace(app, place) for place in active_window_places(standings)}
    return lambda at: at.app == app and at not in active


class WakeReason(StrEnum):
    LINES = "lines"
    VIEW = "view"


def _is_new(entry: Entry, app: str, place: PlaceRef, occasion: Occasion) -> bool:
    """Whether the turn is for this entry: a line taken at its place since his last turn there, or the revision a view
    turn waits at."""
    if isinstance(occasion, Waiting):
        return isinstance(entry, ViewSeen) and _revision_key(entry) == (app, place, occasion.revision)
    return (
        isinstance(entry, KeptLine)
        and entry.visit == Visit(app, place, occasion.generation)
        and entry.seq in occasion.seqs
    )


def held_told(arrived: Arrived, at: datetime) -> str:
    """The result of a call that was not sent: that it was not, and the entries that arrived, each told as new except
    his own speech. When their text would pass ``MAX_EVENT_BYTES`` the oldest are left out, and so is an entry that
    alone does not fit."""
    told = [told_entry(entry, at, new=True) for entry in arrived.entries]
    empty = _held_json([], older_dropped=False)
    separator = len(", ")
    sizes = reversed([told_bytes(each) + separator for each in told])
    fitting = newest_that_fit(sizes, MAX_EVENT_BYTES - len(empty.encode()) + separator)
    kept = told[len(told) - fitting :]
    return _held_json(kept, older_dropped=arrived.older_dropped or fitting < len(told))


def _held_json(entries: list[dict[str, object]], *, older_dropped: bool) -> str:
    return json.dumps(
        {"sent": False, "arrived": {"older_dropped": older_dropped, "entries": entries}}, ensure_ascii=False
    )


def _told_entries(history: History, app: str, place: PlaceRef, occasion: Occasion, at: datetime) -> dict[str, object]:
    entries = [told_entry(entry, at, new=_is_new(entry, app, place, occasion)) for entry in history.entries]
    return {"older_dropped": history.older_dropped, "entries": entries}


def _told_wake(app: str, place: PlaceRef, occasion: Occasion, *, aloud: bool) -> dict[str, object]:
    if isinstance(occasion, Lines):
        woke: dict[str, object] = {
            "why": WakeReason.LINES,
            "app": app,
            "place": told_place(place),
            "missed": occasion.missed,
        }
        return woke | {"aloud": True} if aloud else woke
    return {"why": WakeReason.VIEW, "app": app, "place": told_place(place)}


def turn_told(
    you: str,
    app: str,
    place: PlaceRef,
    occasion: Occasion,
    history: History,
    earlier: EarlierToday | None,
    at: datetime,
    *,
    aloud: bool,
) -> str:
    """The one user message of every turn, the facts in the order things happened: his name, with a memory what he kept
    of the day that his history does not hold, his history as it stood when the turn was admitted at ``at`` with the
    lines or the view revision the turn is for marked new, and what woke him; a turn for lines is told when its answer
    is heard aloud.

    It holds no instruction; those live in the persona.
    """
    told: dict[str, object] = {"you": you}
    if earlier is not None:
        told[EARLIER_TODAY_KEY] = told_earlier_today(earlier)
    told["history"] = _told_entries(history, app, place, occasion, at)
    told["woke"] = _told_wake(app, place, occasion, aloud=aloud)
    return json.dumps(told, ensure_ascii=False)
