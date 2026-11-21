"""The whole runner, one call at a time: busy is decided across all the apps an agent listens to."""

from __future__ import annotations

import pytest
from agents.member import DeclineInvitationReason

from personas.reporting import Event
from tests.support.call import PLACE_ID
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
