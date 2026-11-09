"""What the runner decides, with no process, network or clock: standings in, actions out.

The loop in ``loop`` performs these decisions; nothing here reads a clock or a socket, so every
rule is called with the values it judges.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agent_runtime_client.member import PlaceEntry, PlaceRef, PlaceWindow, Standing, StandingState

from agents.failures import CallFailure, StatusFailure, TokenStatusFailure, UnusableAnswer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agent_runtime_client.tools import ToolDeclaration
    from agent_runtime_client.turn import AgentRole

    from agents.secret import Secret

CALL_PLACE_KIND: Final = "call"
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


class Refusal(StrEnum):
    """The refusals of ``accept_invitation`` (hopin's contract) and one for any other tool error."""

    STALE_INVITATION = "stale_invitation"
    SPONSOR_AWAY = "sponsor_away"
    TWO_SEATS = "two_seats"
    BUSY = "busy"
    UNRECOGNISED = "unrecognised"


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


def accepts_for(standings: tuple[Standing, ...], memory: Memory) -> tuple[Accept, ...]:
    """One accept for each invited call standing whose invitation was not refused as stale."""
    return tuple(
        Accept(standing.place, standing.invitation.id, standing.invitation.epoch)
        for standing in standings
        if standing.state is StandingState.INVITED
        and standing.invitation is not None
        and standing.place.kind == CALL_PLACE_KIND
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
    if refusal is Refusal.STALE_INVITATION:
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


def backoff_reason(failure: CallFailure) -> str:
    """The code a backoff logs: the HTTP status of a server error, else a fixed word for the network."""
    if isinstance(failure, StatusFailure):
        return status_code(failure.status)
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


def decisive_failure(leaves: Sequence[BaseException], *, already_renewed: bool) -> CallFailure | None:
    """The one failure that decides the reaction to several: a refusal before a renewal before a backoff.

    Any leaf that is no failure of a call is a defect, and then no failure decides. Of equal ranks the first wins.
    """
    failures = [leaf for leaf in leaves if isinstance(leaf, CallFailure)]
    if len(failures) < len(leaves):
        return None
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
        if standing.place.kind != CALL_PLACE_KIND:
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
        standing.place
        for standing in standings
        if standing.state is StandingState.ACTIVE and standing.place.kind == CALL_PLACE_KIND
    )


@dataclass(frozen=True)
class TurnCap:
    """At most ``max_turns`` turns start in any ``period_seconds``."""

    max_turns: int
    period_seconds: float


def reserve_turn(started: tuple[float, ...], now: float, cap: TurnCap) -> tuple[float, ...] | None:
    """The start times after a turn starting at ``now``, or None when the cap holds it back.

    ``started`` holds the start times of earlier turns; those at or before ``now - period_seconds`` no longer count.
    """
    inside = tuple(each for each in started if each > now - cap.period_seconds)
    if len(inside) >= cap.max_turns:
        return None
    return (*inside, now)


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


@dataclass(frozen=True)
class _TurnLine:
    seq: int
    kind: str
    name: str
    text: str
    own: bool
    new: bool


@dataclass(frozen=True)
class _TurnRoom:
    kind: str
    id: str
    you: str
    missed: bool
    lines: tuple[_TurnLine, ...]


def turn_input(place: PlaceRef, name: str, window: PlaceWindow, new: Sequence[PlaceEntry], *, missed: bool) -> str:
    """The facts of a turn: the room, the agent's name, every line of the window, which are new and which are his own,
    and whether older lines were never seen.

    It holds no instruction; those live in the persona.
    """
    new_seqs = {line.seq for line in new}
    lines = tuple(
        _TurnLine(line.seq, line.kind, line.name, line.text, line.own, line.seq in new_seqs) for line in window.entries
    )
    return json.dumps(asdict(_TurnRoom(place.kind, place.id, name, missed, lines)), ensure_ascii=False)
