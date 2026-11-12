"""What the runner decides, with no process, network or clock: standings in, actions out.

The loop in ``loop`` performs these decisions; nothing here reads a clock or a socket, so every
rule is called with the values it judges.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agents.member import (
    SPEAK_TOOL,
    AcceptInvitationRefusal,
    PlaceEntry,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    Standing,
    StandingState,
)

from personas.failures import CallFailure, StatusFailure, TokenStatusFailure, UnusableAnswer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from agents.tools import ToolDeclaration, ToolOutcome
    from agents.turn import AgentRole, JsonObject

    from personas.secret import Secret

CALL_PLACE_KIND: Final = "call"
SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_HOUR: Final = 3600
SECONDS_PER_DAY: Final = 24 * SECONDS_PER_HOUR
RENEW_BEFORE_EXPIRY_SECONDS: Final = 60.0
"""A token is renewed once this little of its lifetime remains, so no call carries an expired one."""
NETWORK_FAILURE_CODE: Final = "network"
BACKOFF_BASE_SECONDS: Final = 1.0
BACKOFF_CAP_SECONDS: Final = 30.0
_DOUBLINGS_TO_CAP: Final = math.ceil(math.log2(BACKOFF_CAP_SECONDS / BACKOFF_BASE_SECONDS))
SUSPENDED_RETRY_SECONDS: Final = 30.0
"""How often a suspended loop asks the realm and then the app again."""
TURN_ROLE: Final[AgentRole] = "user"
"""The app role a turn acts under: an agent holds no admin right."""
TOOL_NOT_DECLARED_CODE: Final = "tool_not_declared"
"""The code of a turn that ends because the app no longer declares a tool the configuration binds."""
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


@dataclass(frozen=True)
class Token:
    value: Secret = field(repr=False)
    expires_at: float


@dataclass(frozen=True)
class Memory:
    """What a loop remembers: invitations refused as stale, and the last other refusal of each."""

    stale: frozenset[str] = frozenset()
    refused: Mapping[str, Refusal] = field(default_factory=dict[str, Refusal])


def _is_call(standing: Standing) -> bool:
    """A call place is read through its window; a place that serves a view is read through that, whatever its kind."""
    return standing.place.kind == CALL_PLACE_KIND and not standing.serves_view


def accepts_for(standings: tuple[Standing, ...], memory: Memory, *, takes_turns: bool) -> tuple[Accept, ...]:
    """One accept for each invited call standing whose invitation was not refused as stale, and, where the loop can
    take turns, for each invited place that serves a view: a seated agent that cannot act leaves the place waiting."""
    return tuple(
        Accept(standing.place, standing.invitation.id, standing.invitation.epoch)
        for standing in standings
        if standing.state is StandingState.INVITED
        and standing.invitation is not None
        and (_is_call(standing) or (standing.serves_view and takes_turns))
        and standing.invitation.id not in memory.stale
    )


def forget_gone(memory: Memory, standings: tuple[Standing, ...]) -> Memory:
    """Drop the last refusal of an invitation the standings no longer hold; a stale one stays refused."""
    present = {standing.invitation.id for standing in standings if standing.invitation is not None}
    return Memory(
        stale=memory.stale,
        refused={invitation: code for invitation, code in memory.refused.items() if invitation in present},
    )


def after_refusal(memory: Memory, accept: Accept, refusal: Refusal) -> tuple[Memory, bool]:
    """The memory after a refused accept and whether this refusal is news worth reporting."""
    if refusal is AcceptInvitationRefusal.STALE_INVITATION:
        return Memory(memory.stale | {accept.invitation_id}, memory.refused), True
    is_news = memory.refused.get(accept.invitation_id) is not refusal
    return Memory(memory.stale, {**memory.refused, accept.invitation_id: refusal}), is_news


def renewal_due(token: Token | None, now: float) -> bool:
    return token is None or now >= token.expires_at - RENEW_BEFORE_EXPIRY_SECONDS


def backoff_delay(failures: int) -> float:
    """The wait after the ``failures``-th failure in a row (from one): doubling up to a cap."""
    doublings = min(failures - 1, _DOUBLINGS_TO_CAP)
    return min(BACKOFF_BASE_SECONDS * 2**doublings, BACKOFF_CAP_SECONDS)


def status_code(status: int) -> str:
    return f"http_{status}"


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


def is_voice_app(has_voice_section: bool, declared: Iterable[ToolDeclaration]) -> bool:
    """Whether the places of an app run a voice loop: its agent's persona has a voice section and the app declares
    ``speak``. Every other app keeps the per-line turn."""
    return has_voice_section and any(each.name == SPEAK_TOOL for each in declared)


def spoken_wait_seconds(duration_ms: int) -> float:
    """How long the voice waits after a sentence the app says plays for ``duration_ms``."""
    return min(duration_ms / 1000, SPOKEN_WAIT_BOUND_SECONDS)


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
    """How far one run of a window was read: its generation and the last seq taken."""

    generation: str
    last_seq: int


type Checkpoint = Restarted | Joined | Paused | Reading


def sync_checkpoints(
    checkpoints: Mapping[PlaceRef, Checkpoint], standings: tuple[Standing, ...]
) -> dict[PlaceRef, Checkpoint]:
    """The checkpoints after a standing read: a call place paused is paused, one active keeps its entry or starts as
    restarted, and a place that is gone or invited again has none."""
    synced: dict[PlaceRef, Checkpoint] = {}
    for standing in standings:
        if not _is_call(standing):
            continue
        if standing.state is StandingState.PAUSED:
            synced[standing.place] = Paused()
        elif standing.state is StandingState.ACTIVE:
            synced[standing.place] = checkpoints.get(standing.place, Restarted())
    return synced


def sets_where_reading_starts(checkpoint: Checkpoint) -> bool:
    """A read of a restarted or paused place only sets its checkpoint: nothing in it is taken, so it starts no turn."""
    return isinstance(checkpoint, Restarted | Paused)


def active_call_places(standings: tuple[Standing, ...]) -> tuple[PlaceRef, ...]:
    return tuple(
        standing.place for standing in standings if standing.state is StandingState.ACTIVE and _is_call(standing)
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

    The turn's own outcome does not count: an app that wants him to act again moves the revision.
    """
    return view.awaits_you and turned_revision != view.revision


def turn_invalid(place: PlaceRef, standings: tuple[Standing, ...]) -> bool:
    """A turn ends with its place: it is invalid once the place is gone, invited again or paused."""
    return not any(standing.place == place and standing.state is StandingState.ACTIVE for standing in standings)


@dataclass(frozen=True)
class WindowRead:
    """A window as one agent read it from one app's place, and the wall time of that read."""

    client_id: str
    app: str
    place: PlaceRef
    window: PlaceWindow
    read_at: datetime


@dataclass(frozen=True)
class Visit:
    """One stay of an agent in a place: the app, the place and the window's generation, which a new join renews."""

    app: str
    place: PlaceRef
    generation: str


@dataclass(frozen=True)
class KeptLine:
    """A line the memory kept as a visit's window gave it, with the wall time it was first read."""

    visit: Visit
    seq: int
    first_read_at: datetime
    name: str = field(repr=False)
    kind: str
    text: str = field(repr=False)
    own: bool


@dataclass(frozen=True)
class ViewSeen:
    """A revision of a place's view as an agent read it, with the wall time of that read."""

    app: str
    place: PlaceRef
    view: PlaceView = field(repr=False)
    read_at: datetime


@dataclass(frozen=True)
class Answered:
    """The app answered a call; an error result is an answer too."""

    text: str = field(repr=False)
    is_error: bool


@dataclass(frozen=True)
class Unanswered:
    """The app gave no answer, so whether the call took effect is unknown: ``code`` is the failure's, or the code of a
    call cut in flight."""

    code: str


type ActOutcome = Answered | Unanswered


@dataclass(frozen=True)
class ActDone:
    """A call relayed for an agent to the app at ``app``, with the wall time its relay ended."""

    app: str
    tool: str
    arguments: JsonObject = field(repr=False)
    outcome: ActOutcome
    at: datetime


type Entry = KeptLine | ViewSeen | ActDone


def act_outcome(ended: ToolOutcome | BaseException, *, cancelling: bool) -> ActOutcome | None:
    """The outcome of a relayed call from how its request ended and whether the turn's task is being cancelled.

    An answer is an answer even while a cancel is pending. A cancellation can come out of a request as a failure, so
    any other ending while cancelling is a call cut in flight. Without a cancel a failure is told by its code, and any
    other error is a defect, which records nothing.
    """
    if not isinstance(ended, BaseException):
        return Answered(ended.content, ended.is_error)
    if cancelling:
        return Unanswered(TURN_CANCELLED_CODE)
    if isinstance(ended, CallFailure):
        return Unanswered(failure_code(ended))
    return None


@dataclass(frozen=True)
class TodayBounds:
    """What a turn is told of the day: the lines first read within ``hours``, at most ``size_bytes`` of them as told."""

    hours: float
    size_bytes: int


@dataclass(frozen=True)
class EarlierVisit:
    """An earlier visit as a turn is told it: how many minutes ago its first line was read, and its lines. It is a
    visit whose lines the history does not hold."""

    minutes_ago: int
    lines: tuple[KeptLine, ...]


@dataclass(frozen=True)
class EarlierToday:
    """The kept lines of the day that the history does not hold, as earlier visits in the order they happened, and
    whether older lines were left out."""

    visits: tuple[EarlierVisit, ...]
    older_dropped: bool


def _before(now: datetime, seconds: float) -> datetime | None:
    """The time ``seconds`` before ``now``, or nothing when that lies before the calendar's first moment."""
    earliest = datetime.min.replace(tzinfo=now.tzinfo)
    if seconds >= (now - earliest).total_seconds():
        return None
    return now - timedelta(seconds=seconds)


def today_since(now: datetime, bounds: TodayBounds) -> datetime:
    """The earliest first read a turn at ``now`` is told of; a day longer than the calendar reaches any line."""
    return _before(now, bounds.hours * SECONDS_PER_HOUR) or datetime.min.replace(tzinfo=now.tzinfo)


def swept_before(now: datetime, lines_keep_days: float) -> datetime | None:
    """The time before which a line first read is swept at ``now``, or nothing when the period reaches any line."""
    return _before(now, lines_keep_days * SECONDS_PER_DAY)


def _own_time(entry: Entry) -> datetime:
    if isinstance(entry, ActDone):
        return entry.at
    if isinstance(entry, ViewSeen):
        return entry.read_at
    return entry.first_read_at


def _told_bytes(told: dict[str, object]) -> int:
    return len(json.dumps(told, ensure_ascii=False).encode())


def _size(entry: Entry) -> int:
    """A view counts the bytes of its state's JSON; a line and an act count the bytes of the JSON they are told as, with
    their age as 0 and no mark as new, so a size does not change while the entry stays in the history."""
    if isinstance(entry, ViewSeen):
        return len(json.dumps(entry.view.state, ensure_ascii=False).encode())
    return _told_bytes(_told_entry(entry, _own_time(entry), new=False))


def _newest_that_fit(sizes_newest_first: Iterable[int], size_bytes: int) -> int:
    """How many entries, whose sizes are given newest first, fit ``size_bytes``."""
    return sum(1 for total in itertools.accumulate(sizes_newest_first) if total <= size_bytes)


def _remaining(kept: Iterable[KeptLine], leaving_out: History | Visit, now: datetime) -> Iterable[KeptLine]:
    """The kept lines a turn is still to be told: those the history holds are left out by visit and seq, and those first
    read after ``now``, which no history of that turn holds yet; a voice loop leaves out the lines of its own visit."""
    if isinstance(leaving_out, Visit):
        return (line for line in kept if line.visit != leaving_out)
    held = {(entry.visit, entry.seq) for entry in leaving_out.entries if isinstance(entry, KeptLine)}
    return (line for line in kept if (line.visit, line.seq) not in held and line.first_read_at <= now)


def earlier_today(
    kept: Iterable[KeptLine], now: datetime, bounds: TodayBounds, leaving_out: History | Visit
) -> EarlierToday:
    """The lines of the day a turn is told besides its history: grouped by visit, visits in the order of their first
    line and lines in seq order; past ``size_bytes`` of the lines as told the oldest lines are left out, and what is
    left out beforehand (see ``_remaining``) counts nothing toward that bound."""
    by_visit: dict[Visit, list[KeptLine]] = {}
    for line in _remaining(kept, leaving_out, now):
        by_visit.setdefault(line.visit, []).append(line)
    groups = sorted((sorted(lines, key=lambda line: line.seq) for lines in by_visit.values()), key=_first_read)
    newest_first = (line for lines in reversed(groups) for line in reversed(lines))
    fitting = _newest_that_fit((_told_bytes(_told_line(line)) for line in newest_first), bounds.size_bytes)
    left_out = sum(len(lines) for lines in groups) - fitting
    older_dropped = left_out > 0
    visits: list[EarlierVisit] = []
    for lines in groups:
        told, left_out = lines[left_out:], max(0, left_out - len(lines))
        if told:
            visits.append(EarlierVisit(int((now - _first_read(lines)).total_seconds() // 60), tuple(told)))
    return EarlierToday(tuple(visits), older_dropped)


def _first_read(lines: Sequence[KeptLine]) -> datetime:
    return min(line.first_read_at for line in lines)


@dataclass(frozen=True)
class History:
    """Every line one agent took in, every view revision he read and every call relayed for him at any of his places
    since the runner started, in the order they came, and whether older entries were left out to keep it within its
    bound."""

    entries: tuple[Entry, ...] = field(repr=False)
    older_dropped: bool


def taken_lines(visit: Visit, new: Iterable[PlaceEntry], read_at: datetime) -> tuple[KeptLine, ...]:
    """The lines a read of the visit's window took, as the history keeps them."""
    return tuple(KeptLine(visit, line.seq, read_at, line.name, line.kind, line.text, line.own) for line in new)


def _revision_key(view: ViewSeen) -> tuple[str, PlaceRef, int]:
    return view.app, view.place, view.view.revision


def _holds_revision(entries: Iterable[Entry], view: ViewSeen) -> bool:
    return any(isinstance(entry, ViewSeen) and _revision_key(entry) == _revision_key(view) for entry in entries)


def history_after(history: History, new: Iterable[Entry], size_bytes: int) -> History:
    """The history with ``new`` appended, a view revision it already holds adding nothing and an act always added; past
    ``size_bytes`` the oldest entries are left out, but the newest always stays."""
    entries = history.entries
    for entry in new:
        if not (isinstance(entry, ViewSeen) and _holds_revision(entries, entry)):
            entries = (*entries, entry)
    left_out = max(0, len(entries) - max(1, _newest_that_fit(map(_size, reversed(entries)), size_bytes)))
    return History(entries[left_out:], history.older_dropped or left_out > 0)


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
        return Perception(Reading(window.generation, highest), (), missed=False)
    match checkpoint:
        case Reading(generation=window.generation, last_seq=last_seq):
            return _take_after(window, last_seq)
        case _:
            return _take_after(window, 0)


def _take_after(window: PlaceWindow, last_seq: int) -> Perception:
    new = tuple(line for line in window.entries if line.seq > last_seq)
    return Perception(
        Reading(window.generation, new[-1].seq if new else last_seq), new, missed=window.starts_at > last_seq + 1
    )


def addressed(line: PlaceEntry, name: str) -> bool:
    """Whether the line names the agent as a whole word, whatever the case; a line of his own never does."""
    return not line.own and re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", line.text.casefold()) is not None


def bound_declarations(declared: Sequence[ToolDeclaration], names: Sequence[str]) -> tuple[ToolDeclaration, ...] | None:
    """The declarations of the bound tools in binding order, or ``None`` when the app declares one no more."""
    by_name = {declaration.name: declaration for declaration in declared}
    if not all(name in by_name for name in names):
        return None
    return tuple(by_name[name] for name in names)


def turn_call_bound(bounds: Iterable[int]) -> int:
    """The calls one turn may make: the sum of the bounds of the apps whose tools it holds."""
    return sum(bounds)


def call_within_bound(made: int, bound: int) -> bool:
    """Whether an app whose bound is ``bound`` takes one more call after ``made``."""
    return made < bound


@dataclass(frozen=True)
class Lines:
    """Why a written turn runs: the lines of one visit's window that a read took, which the history holds, and whether
    older ones were never seen."""

    generation: str
    seqs: frozenset[int]
    missed: bool


@dataclass(frozen=True)
class Waiting:
    """Why a view turn runs: the place's view waits for the agent at this revision, which the history holds."""

    revision: int


type Occasion = Lines | Waiting


class WakeReason(StrEnum):
    LINES = "lines"
    VIEW = "view"


def told_earlier_today(earlier: EarlierToday) -> dict[str, object]:
    visits = [
        {"minutes_ago": visit.minutes_ago, "lines": [_told_line(each) for each in visit.lines]}
        for visit in earlier.visits
    ]
    return {"older_dropped": earlier.older_dropped, "visits": visits}


def _told_line(line: KeptLine) -> dict[str, object]:
    return {"name": line.name, "kind": line.kind, "text": line.text, "own": line.own}


def _told_place(place: PlaceRef) -> dict[str, str]:
    return {"kind": place.kind, "id": place.id}


def _is_new(entry: Entry, app: str, place: PlaceRef, occasion: Occasion) -> bool:
    """Whether the turn is for this entry: a line the read that admitted a written turn took, or the revision a view
    turn waits at."""
    if isinstance(occasion, Waiting):
        return isinstance(entry, ViewSeen) and _revision_key(entry) == (app, place, occasion.revision)
    return (
        isinstance(entry, KeptLine)
        and entry.visit == Visit(app, place, occasion.generation)
        and entry.seq in occasion.seqs
    )


def _told_outcome(outcome: ActOutcome) -> dict[str, object]:
    if isinstance(outcome, Answered):
        return {"answered": True, "is_error": outcome.is_error, "text": outcome.text}
    return {"answered": False, "code": outcome.code}


def _told_entry(entry: Entry, at: datetime, *, new: bool) -> dict[str, object]:
    if isinstance(entry, ActDone):
        return {
            "entry": "act",
            "app": entry.app,
            "seconds_ago": int((at - entry.at).total_seconds()),
            "tool": entry.tool,
            "arguments": entry.arguments,
            "outcome": _told_outcome(entry.outcome),
            "new": False,
        }
    if isinstance(entry, ViewSeen):
        return {
            "entry": "view",
            "app": entry.app,
            "place": _told_place(entry.place),
            "seconds_ago": int((at - entry.read_at).total_seconds()),
            "revision": entry.view.revision,
            "awaits_you": entry.view.awaits_you,
            "state": entry.view.state,
            "new": new,
        }
    return (
        {
            "entry": "line",
            "app": entry.visit.app,
            "place": _told_place(entry.visit.place),
            "seconds_ago": int((at - entry.first_read_at).total_seconds()),
            "seq": entry.seq,
        }
        | _told_line(entry)
        | {"new": new}
    )


def _told_entries(history: History, app: str, place: PlaceRef, occasion: Occasion, at: datetime) -> dict[str, object]:
    entries = [_told_entry(entry, at, new=_is_new(entry, app, place, occasion)) for entry in history.entries]
    return {"older_dropped": history.older_dropped, "entries": entries}


def _told_wake(app: str, place: PlaceRef, occasion: Occasion) -> dict[str, object]:
    if isinstance(occasion, Lines):
        return {"why": WakeReason.LINES, "app": app, "place": _told_place(place), "missed": occasion.missed}
    return {"why": WakeReason.VIEW, "app": app, "place": _told_place(place)}


def turn_told(
    you: str,
    app: str,
    place: PlaceRef,
    occasion: Occasion,
    history: History,
    earlier: EarlierToday | None,
    at: datetime,
) -> str:
    """The one user message of every turn, the facts in the order things happened: his name, with a memory what he kept
    of the day that his history does not hold, his history as it stood when the turn was admitted at ``at`` with the
    lines or the view revision the turn is for marked new, and what woke him.

    It holds no instruction; those live in the persona.
    """
    told: dict[str, object] = {"you": you}
    if earlier is not None:
        told[EARLIER_TODAY_KEY] = told_earlier_today(earlier)
    told["history"] = _told_entries(history, app, place, occasion, at)
    told["woke"] = _told_wake(app, place, occasion)
    return json.dumps(told, ensure_ascii=False)


def voice_context(earlier: EarlierToday | None) -> str:
    """What a voice loop is told of the day, once, when it starts: the facts a turn is told under the same key; none
    without a memory or when the store cannot be read."""
    if earlier is None:
        return ""
    return json.dumps({EARLIER_TODAY_KEY: told_earlier_today(earlier)}, ensure_ascii=False)
