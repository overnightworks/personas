"""What an app scripts for a member: his subject, his standings (invited to a place, active in it or paused there) and
the windows of lines it shows him."""

from __future__ import annotations

from datetime import UTC, datetime

from agents.member import Invitation, ParentPlace, PlaceEntry, PlaceRef, PlaceWindow, Standing, StandingState

AGENT_SUBJECT = "agent-subject"
GENERATION = "generation-1"


def invited(
    place_id: str,
    invitation_id: str,
    epoch: int = 0,
    kind: str = "call",
    *,
    serves_view: bool = False,
    parent: ParentPlace | None = None,
) -> Standing:
    invitation = Invitation(id=invitation_id, epoch=epoch, expires_at=datetime(2099, 1, 1, tzinfo=UTC))
    return Standing(
        place=PlaceRef(kind=kind, id=place_id),
        state=StandingState.INVITED,
        invitation=invitation,
        serves_view=serves_view,
        parent=parent,
    )


def active(place_id: str, kind: str = "call", *, serves_view: bool = False) -> Standing:
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.ACTIVE, serves_view=serves_view)


def paused(place_id: str, kind: str = "call", *, serves_view: bool = False) -> Standing:
    """The sponsor is away: the member keeps the place and its window, and nobody reads it meanwhile."""
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.PAUSED, serves_view=serves_view)


def line(
    seq: int, text: str, *, own: bool = False, spoken: bool = False, act: bool = False, name: str = "Ada"
) -> PlaceEntry:
    return PlaceEntry(seq=seq, kind="message", name=name, text=text, own=own, spoken=spoken, cut=False, act=act)


def window_of(*lines: PlaceEntry, generation: str = GENERATION) -> PlaceWindow:
    return PlaceWindow(generation=generation, starts_at=lines[0].seq if lines else 1, entries=lines)
