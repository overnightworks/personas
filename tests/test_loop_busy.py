"""One call at a time: a call invitation is declined busy while the agent holds an active place."""

from __future__ import annotations

import pytest
from agents.member import DeclineInvitationReason, PlaceRef

from personas.decision import Decline
from personas.reporting import Event
from tests.support.call import PLACE_ID
from tests.support.fakes import read_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, invited

pytestmark = pytest.mark.anyio

OTHER_CALL_ID = "room-at-the-other-app"


async def test_a_call_invitation_at_one_app_is_declined_busy_while_a_call_is_active_at_another(runner, other):
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read(1)
    other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-other"))
    await runner.reports.until(Event.INVITATION_DECLINED)

    assert [(each.place.id, each.reason) for each in other.declined] == [(OTHER_CALL_ID, DeclineInvitationReason.BUSY)]
    assert other.accepted == []


async def test_a_call_invitation_while_a_place_is_active_is_declined_once_and_not_accepted(scenario):
    """TURNS-1"""
    standings = (active("room-1"), invited("room-7", "inv-3", epoch=2))
    run = scenario([read_of(*standings), read_of(*standings)])

    await run.play()

    assert run.session.declined == [Decline(PlaceRef(kind="call", id="room-7"), "inv-3", 2)]
    assert run.session.accepted == []
    assert [event for event, _ in run.reports.reported()].count(Event.INVITATION_DECLINED) == 1
