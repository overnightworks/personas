"""The standings an app scripts for a member: invited to a place, active in it or paused there."""

from __future__ import annotations

from datetime import UTC, datetime

from agents.member import Invitation, ParentPlace, PlaceRef, Standing, StandingState


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
