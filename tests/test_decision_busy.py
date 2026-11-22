"""One call at a time: a call invitation is declined busy while the agent holds an active place."""

from __future__ import annotations

import pytest
from agents.member import DeclineInvitationReason, PlaceRef

from personas.decision import Decline, Memory, accepts_for, declines_for, forget_gone
from tests.support.standings import active, invited, paused


@pytest.mark.parametrize(
    ("standings", "accepted", "declined"),
    [
        ((invited("room-7", "inv-3"),), ["inv-3"], []),
        ((active("room-1"), invited("room-7", "inv-3")), [], ["inv-3"]),
        ((paused("room-1"), invited("room-7", "inv-3")), ["inv-3"], []),
        ((active("room-1"), invited("table-1", "inv-4", kind="table", serves_view=True)), ["inv-4"], []),
        ((active("room-7"), invited("room-7", "inv-3")), ["inv-3"], []),
    ],
    ids=["no active place", "active call place", "paused place", "table reached from the call", "its own place"],
)
def test_a_call_invitation_while_another_place_is_active_is_declined_busy_and_not_accepted(
    standings, accepted, declined
):
    """TURNS-1"""
    _, declines = declines_for(standings, Memory())

    assert [accept.invitation_id for accept in accepts_for(standings, Memory(), takes_turns=True)] == accepted
    assert [decline.invitation_id for decline in declines] == declined
    assert all(decline.reason is DeclineInvitationReason.BUSY for decline in declines)


def test_only_the_first_of_two_call_invitations_is_accepted_and_the_other_declined_busy():
    """TURNS-1"""
    standings = (invited("room-7", "inv-3"), invited("room-8", "inv-4"))

    _, declines = declines_for(standings, Memory())

    assert [accept.invitation_id for accept in accepts_for(standings, Memory(), takes_turns=True)] == ["inv-3"]
    assert [decline.invitation_id for decline in declines] == ["inv-4"]


def test_a_call_invitation_while_an_active_place_is_held_in_another_app_is_declined_busy_and_not_accepted():
    """TURNS-1"""
    standings = (invited("room-7", "inv-3"),)

    _, declines = declines_for(standings, Memory(), held_elsewhere=True)

    assert accepts_for(standings, Memory(), takes_turns=True, held_elsewhere=True) == ()
    assert [decline.invitation_id for decline in declines] == ["inv-3"]


def test_a_decline_is_made_once_and_forgotten_with_its_invitation():
    standings = (active("room-1"), invited("room-7", "inv-3", epoch=2))

    memory, first = declines_for(standings, Memory())
    memory, again = declines_for(standings, memory)

    assert first == (Decline(PlaceRef(kind="call", id="room-7"), "inv-3", 2),)
    assert again == ()
    assert forget_gone(memory, (active("room-1"),)).declined == frozenset()
