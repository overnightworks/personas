"""One call at a time: a call invitation is declined busy while the agent holds an active place."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from agents.member import (
    DECLINE_INVITATION_TOOL,
    DeclineInvitationReason,
    DeclineInvitationRefusal,
    DeclineInvitationRefused,
    PlaceRef,
)
from agents.member_outcomes import UnrecognisedRefusal
from agents.tools import ToolOutcome

from personas.decision import Decline
from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import PLACE_ID
from tests.support.fakes import read_of
from tests.support.hopin import PLAY_TOOL, FakeApp
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, invited

pytestmark = pytest.mark.anyio

OTHER_CALL_ID = "room-at-the-other-app"


async def test_a_call_invitation_at_one_app_is_declined_busy_while_a_call_is_active_at_another(runner, other):
    """TURNS-1"""
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read(1)
    await settled(runner.app)
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


async def test_a_decline_that_failed_transiently_is_sent_again_after_the_backoff(runner, other):
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read(1)
    await settled(runner.app)
    other.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    other.refused_tool = "decline_invitation"
    other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-other"))
    await runner.reports.until(Event.BACKOFF)
    other.refuse_with = None
    await runner.reports.until(Event.RESUMED)
    await runner.reports.until(Event.INVITATION_DECLINED)

    assert [each.place.id for each in other.declined] == [OTHER_CALL_ID]


async def test_call_invitations_at_two_apps_in_the_same_instant_leave_the_agent_in_one_call(runner, other):
    """TURNS-1"""
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-here"))
    other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-there"))
    runner.start()
    await runner.reports.until(Event.INVITATION_DECLINED)

    assert len(runner.app.accepted) + len(other.accepted) == 1
    assert len(runner.app.declined) + len(other.declined) == 1


async def test_a_call_active_at_an_app_that_backs_off_does_not_hold_an_invitation_elsewhere_busy(runner, other):
    """TURNS-1"""
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read(1)
    await settled(runner.app)
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.reports.until(Event.BACKOFF)
    other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-other"))
    await runner.reports.until(Event.JOINED)

    assert [each.place.id for each in other.accepted] == [OTHER_CALL_ID]
    assert other.declined == []


async def test_no_decline_is_sent_to_an_app_that_lists_no_tool_to_decline_with(runner):
    """TURNS-1"""
    other = FakeApp(line_tool=PLAY_TOOL, announces_tools=True, withheld_tools={DECLINE_INVITATION_TOOL})
    async with other.serving() as url:
        runner.add_listening_app(other, url, "mcp-other")
        runner.app.stand(SUBJECT, active(PLACE_ID))
        runner.start()
        await runner.app.until_standing_read(1)
        await settled(runner.app)
        other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-other"))
        await settled(other)

    assert DECLINE_INVITATION_TOOL not in other.calls
    assert other.accepted == []
    events = [event for event, _ in runner.reports.reported()]
    assert Event.INVITATION_DECLINED not in events
    assert Event.BACKOFF not in events
    assert [each.invitation.id for each in other.standings[SUBJECT] if each.invitation] == ["inv-other"]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        pytest.param(
            DeclineInvitationRefused(refused=DeclineInvitationRefusal.STALE_INVITATION).model_dump_json(),
            DeclineInvitationRefusal.STALE_INVITATION.value,
            id="a refusal of the library",
        ),
        pytest.param("invalid arguments", UnrecognisedRefusal.UNRECOGNISED.value, id="an error that is no refusal"),
    ],
)
async def test_a_decline_the_app_refuses_is_logged_as_refused_and_not_sent_again(runner, other, answer, reason):
    """TURNS-1"""
    other.decline_outcomes = [ToolOutcome(answer, is_error=True)]
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read(1)
    await settled(runner.app)
    other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-other"))
    await runner.reports.until(Event.DECLINE_REFUSED)
    await settled(other)

    events = [event for event, _ in runner.reports.reported()]
    assert (Event.DECLINE_REFUSED, reason) in runner.reports.reported()
    assert Event.INVITATION_DECLINED not in events
    assert Event.BACKOFF not in events
    assert len(other.declined) == 1


async def test_after_the_call_ends_a_call_invitation_at_another_app_is_accepted(runner, other):
    """TURNS-1"""
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_standing_read(1)
    await settled(runner.app)
    runner.app.stand(SUBJECT)
    await runner.app.until_standing_read(2)
    await settled(runner.app)
    other.stand(SUBJECT, invited(OTHER_CALL_ID, "inv-other"))
    await runner.reports.until(Event.JOINED)

    assert [each.place.id for each in other.accepted] == [OTHER_CALL_ID]
    assert other.declined == []
