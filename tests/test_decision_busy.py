"""One call at a time: a call invitation is declined busy while the agent holds an active place."""

from __future__ import annotations

import pytest
from agents.member import DeclineInvitationReason, DeclineInvitationRefusal, PlaceRef
from agents.member_outcomes import Delivery

from personas.decision import Decline, Memory, accepts_for, after_decline, declines_for, forget_gone
from tests.support.standings import active, invited, paused


@pytest.mark.parametrize(
    ("standings", "held_elsewhere", "accepted", "declined"),
    [
        ((invited("room-7", "inv-3"),), False, ["inv-3"], []),
        ((active("room-1"), invited("room-7", "inv-3")), False, [], ["inv-3"]),
        ((paused("room-1"), invited("room-7", "inv-3")), False, ["inv-3"], []),
        ((active("room-1"), invited("table-1", "inv-4", kind="table", serves_view=True)), False, ["inv-4"], []),
        ((active("room-7"), invited("room-7", "inv-3")), False, ["inv-3"], []),
        ((active("table-1", "table"), invited("room-7", "inv-3")), False, [], ["inv-3"]),
        ((invited("room-7", "inv-3"),), True, [], ["inv-3"]),
        ((invited("table-1", "inv-4", kind="table", serves_view=True),), True, ["inv-4"], []),
    ],
    ids=[
        "no active place",
        "active call place",
        "paused place",
        "table reached from the call",
        "its own place",
        "active place of another kind",
        "place held in another app",
        "table invited while a place is held in another app",
    ],
)
def test_a_call_invitation_while_another_place_is_active_is_declined_busy_and_not_accepted(
    standings, held_elsewhere, accepted, declined
):
    """TURNS-1"""
    declines = declines_for(standings, Memory(), held_elsewhere=held_elsewhere)

    assert [
        accept.invitation_id
        for accept in accepts_for(standings, Memory(), takes_turns=True, held_elsewhere=held_elsewhere)
    ] == accepted
    assert [decline.invitation_id for decline in declines] == declined
    assert all(decline.reason is DeclineInvitationReason.BUSY for decline in declines)


def test_only_the_first_of_two_call_invitations_is_accepted_and_the_other_declined_busy():
    """TURNS-1"""
    standings = (invited("room-7", "inv-3"), invited("room-8", "inv-4"))

    declines = declines_for(standings, Memory())

    assert [accept.invitation_id for accept in accepts_for(standings, Memory(), takes_turns=True)] == ["inv-3"]
    assert [decline.invitation_id for decline in declines] == ["inv-4"]


def test_a_decline_is_made_once_and_forgotten_with_its_invitation():
    standings = (active("room-1"), invited("room-7", "inv-3", epoch=2))

    first = declines_for(standings, Memory())
    memory = after_decline(Memory(), first[0], Delivery.SENT)
    again = declines_for(standings, memory)

    assert first == (Decline(PlaceRef(kind="call", id="room-7"), "inv-3", 2),)
    assert again == ()
    assert forget_gone(memory, (active("room-1"),)).declined == frozenset()


@pytest.mark.parametrize(
    "answer",
    [Delivery.SENT, Delivery.NOT_SERVED, DeclineInvitationRefusal.STALE_INVITATION],
    ids=lambda each: str(each),
)
def test_a_decline_is_not_made_again_whether_it_was_sent_left_to_run_out_or_refused(answer):
    """TURNS-1"""
    standings = (active("room-1"), invited("room-7", "inv-3"))
    (decline,) = declines_for(standings, Memory())

    memory = after_decline(Memory(), decline, answer)

    assert declines_for(standings, memory) == ()


def test_a_refused_decline_is_not_remembered_as_declined():
    """TURNS-1"""
    decline = Decline(PlaceRef(kind="call", id="room-7"), "inv-3", 2)

    memory = after_decline(Memory(), decline, DeclineInvitationRefusal.STALE_INVITATION)

    assert memory.declined == frozenset()
