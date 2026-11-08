"""What the runner decides, with no process, network or clock: standings in, actions out.

The loop in ``loop`` performs these decisions; nothing here reads a clock or a socket, so every
rule is called with the values it judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agent_runtime_client.wait import StandingState, WaitAnswer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agents.secret import Secret

CALL_PLACE_KIND: Final = "call"
RENEW_BEFORE_EXPIRY_SECONDS: Final = 60.0
"""A token is renewed once this little of its lifetime remains, so no call carries an expired one."""
BACKOFF_BASE_SECONDS: Final = 1.0
BACKOFF_CAP_SECONDS: Final = 30.0
SUSPENDED_RETRY_SECONDS: Final = 30.0
"""How often a suspended loop asks the realm and then the app again."""
WAIT_ERROR_CODE: Final = "wait_error"


class Refusal(StrEnum):
    """The refusals of ``accept_invitation`` (hopin's contract) and one for any other tool error."""

    STALE_INVITATION = "stale_invitation"
    SPONSOR_AWAY = "sponsor_away"
    TWO_SEATS = "two_seats"
    BUSY = "busy"
    UNRECOGNISED = "unrecognised"


class CallSource(StrEnum):
    """Whom a failed HTTP call went to: the realm's token endpoint or the app."""

    REALM = "realm"
    APP = "app"


class Reaction(StrEnum):
    RENEW = "renew"
    SUSPEND = "suspend"
    BACKOFF = "backoff"


@dataclass(frozen=True)
class Accept:
    place_id: str
    invitation_id: str
    epoch: int


@dataclass(frozen=True)
class Token:
    value: Secret = field(repr=False)
    expires_at: float


@dataclass(frozen=True)
class Memory:
    """What a loop remembers between answers: invitations refused as stale, and the last other refusal."""

    stale: frozenset[str] = frozenset()
    refused: Mapping[str, Refusal] = field(default_factory=dict[str, Refusal])


def accepts_for(answer: WaitAnswer, memory: Memory) -> tuple[Accept, ...]:
    """One accept for each invited call standing whose invitation was not refused as stale."""
    return tuple(
        Accept(standing.place.id, standing.invitation.id, standing.invitation.epoch)
        for standing in answer.standings
        if standing.state is StandingState.INVITED
        and standing.invitation is not None
        and standing.place.kind == CALL_PLACE_KIND
        and standing.invitation.id not in memory.stale
    )


def forget_gone(memory: Memory, answer: WaitAnswer) -> Memory:
    """Drop what is remembered of an invitation the answer no longer holds; it cannot come back."""
    present = {standing.invitation.id for standing in answer.standings if standing.invitation is not None}
    return Memory(
        stale=memory.stale & present,
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
    return min(BACKOFF_BASE_SECONDS * 2 ** (failures - 1), BACKOFF_CAP_SECONDS)


def status_code(status: int) -> str:
    return f"http_{status}"


def reaction_to_status(status: int, source: CallSource, *, already_renewed: bool) -> Reaction:
    """A server error backs off; a refusal suspends; an app's first 401 renews the token once."""
    if status >= 500:
        return Reaction.BACKOFF
    if source is CallSource.APP and status == 401 and not already_renewed:
        return Reaction.RENEW
    return Reaction.SUSPEND
