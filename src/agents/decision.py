"""What the runner decides, with no process, network or clock: standings in, actions out.

The loop in ``loop`` performs these decisions; nothing here reads a clock or a socket, so every
rule is called with the values it judges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agent_runtime_client.member import PlaceRef, Standing, StandingState

from agents.failures import CallFailure, StatusFailure, TokenStatusFailure, UnusableAnswer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    return min(failures, key=lambda failure: _rank(reaction_to(failure, already_renewed=already_renewed)), default=None)


def _rank(reaction: Reaction | Suspend) -> int:
    match reaction:
        case Suspend():
            return 0
        case Reaction.RENEW:
            return 1
        case _:
            return 2
