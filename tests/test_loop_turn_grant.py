"""The whole runner, a tools-only app that is not granted to an agent: it is left out of his turns until it answers."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion

from personas.decision import SUSPENDED_RETRY_SECONDS, status_code
from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.hopin import LIST_TOOLS
from tests.support.pods import SECOND_AGENT, SECOND_SUBJECT
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG, SCOPE
from tests.support.standings import AGENT_SUBJECT, active

pytestmark = pytest.mark.anyio

SONG_TOOLS = {CREATE_SONG, GENERATE_SONG}


def refused_by_the_realm(runner, songmaker):
    runner.realm.refused_scopes.add(SCOPE)


def refused_by_the_app(runner, songmaker):
    songmaker.refuse_with = HTTPStatus.FORBIDDEN
    songmaker.refused_method = LIST_TOOLS


def granted_by_the_realm(runner, songmaker):
    runner.realm.refused_scopes.discard(SCOPE)


def granted_by_the_app(runner, songmaker):
    songmaker.refuse_with = None


REFUSALS = pytest.mark.parametrize(
    ("refuse", "grant", "code"),
    [
        (refused_by_the_realm, granted_by_the_realm, status_code(HTTPStatus.UNAUTHORIZED)),
        (refused_by_the_app, granted_by_the_app, status_code(HTTPStatus.FORBIDDEN)),
    ],
    ids=["the realm refuses the token", "the app answers 403"],
)


def songmaker_url(runner) -> str:
    (url,) = (each.mcp_url for each in runner.config.apps if not each.listens)
    return url


def song_tools_offered(runner, turn: int) -> set[str]:
    return set(runner.gateway.offered_tools(turn)) & SONG_TOOLS


async def a_turn(runner, number: int) -> None:
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(number)


@REFUSALS
async def test_a_tools_only_app_that_is_not_granted_is_left_out_of_every_turn_and_logged_once(
    runner, songmaker, refuse, grant, code
):
    """TURNS-42."""
    refuse(runner, songmaker)
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)

    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    await a_turn(runner, 2)

    assert [song_tools_offered(runner, turn) for turn in (1, 2)] == [set(), set()]
    assert songmaker.called == []
    assert runner.clock.slept == []
    assert [(report.event, report.code) for report in runner.reports.reported(apps=True, started=False)] == [
        (Event.SUSPENDED, code)
    ]
    assert runner.reports.reported(apps=True, started=False)[0].app == songmaker_url(runner)


@REFUSALS
async def test_a_tools_only_app_comes_back_to_a_turn_once_the_retry_is_due_and_the_grant_answers(
    runner, songmaker, refuse, grant, code
):
    refuse(runner, songmaker)
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    grant(runner, songmaker)

    await a_turn(runner, 2)
    runner.clock.time += SUSPENDED_RETRY_SECONDS
    await a_turn(runner, 3)
    await a_turn(runner, 4)

    assert [song_tools_offered(runner, turn) for turn in (2, 3, 4)] == [set(), SONG_TOOLS, SONG_TOOLS]
    assert runner.reports.reported(started=False) == [(Event.SUSPENDED, code), (Event.RESUMED, None)]


async def test_a_still_refused_tools_only_app_is_asked_again_at_the_retry_and_logged_once(runner, songmaker):
    """TURNS-42."""
    refused_by_the_realm(runner, songmaker)
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    asked = runner.realm.refusals.qsize()

    await a_turn(runner, 2)
    assert runner.realm.refusals.qsize() == asked
    runner.clock.time += SUSPENDED_RETRY_SECONDS
    await a_turn(runner, 3)

    assert runner.realm.refusals.qsize() == asked + 1
    assert runner.reports.reported(started=False) == [(Event.SUSPENDED, status_code(HTTPStatus.UNAUTHORIZED))]


async def test_an_agent_the_realm_grants_keeps_the_tools_only_app_while_another_is_refused(runner, songmaker):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.realm.refused_to.add((runner.config.agents[0].client_id, SCOPE))
    runner.gateway = Gateway(completion("done"))
    runner.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)

    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(2)

    assert sorted(sorted(song_tools_offered(runner, turn)) for turn in (1, 2)) == [[], sorted(SONG_TOOLS)]
    assert [
        (report.event, report.client_id) for report in runner.reports.changes if report.event is Event.SUSPENDED
    ] == [(Event.SUSPENDED, runner.config.agents[0].client_id)]
