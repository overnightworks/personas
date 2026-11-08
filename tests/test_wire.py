"""The SDK client against agent_runtime_client's own server, in process: the real wire."""

from __future__ import annotations

import asyncio
import secrets
from http import HTTPStatus

import httpx2
import pytest
from agent_runtime_client.tools import ToolOutcome

from agents.app_session import open_mcp_session
from agents.decision import Accept, Refusal
from agents.failures import StatusFailure, Unreachable
from agents.secret import Secret
from tests.support.fakes import app_target
from tests.support.hopin import FakeApp, active, invited

pytestmark = pytest.mark.anyio

SUBJECT = "agent-subject"
WAIT_LIMIT_SECONDS = 10


@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture
def bearer(app):
    token = Secret(secrets.token_urlsafe(24))
    app.let_in(token.reveal(), SUBJECT)
    return token


async def test_initialize_and_a_first_wait_answer_at_once_with_every_standing_and_a_cursor(app, bearer):
    app.stand(SUBJECT, invited("room-7", "inv-3", epoch=2), active("room-8"))

    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        answer = await session.wait(None)

    assert answer.cursor
    assert [(each.place.id, each.state.value) for each in answer.standings] == [
        ("room-7", "invited"),
        ("room-8", "active"),
    ]


async def test_a_wait_with_the_current_cursor_is_held_until_a_change_and_then_answers_the_new_standing(app, bearer):
    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        first = await session.wait(None)
        held = asyncio.create_task(session.wait(first.cursor))
        await asyncio.wait_for(app.changes.held.wait(), WAIT_LIMIT_SECONDS)
        assert not held.done()
        app.stand(SUBJECT, invited("room-7", "inv-3"))
        answer = await asyncio.wait_for(held, WAIT_LIMIT_SECONDS)

    assert answer.cursor != first.cursor
    assert [each.place.id for each in answer.standings] == ["room-7"]


async def test_a_cursor_that_is_none_of_the_apps_is_answered_at_once(app, bearer):
    app.stand(SUBJECT, active("room-8"))

    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        answer = await asyncio.wait_for(session.wait("no-such-cursor"), WAIT_LIMIT_SECONDS)

    assert [each.place.id for each in answer.standings] == ["room-8"]


async def test_an_accept_sends_the_place_invitation_and_epoch_and_a_success_is_no_refusal(app, bearer):
    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        refusal = await session.accept(Accept("room-7", "inv-3", 2))

    assert refusal is None
    assert app.accepted == [{"room": "room-7", "invitation": "inv-3", "epoch": 2}]


@pytest.mark.parametrize("refusal", [Refusal.STALE_INVITATION, Refusal.SPONSOR_AWAY, Refusal.TWO_SEATS, Refusal.BUSY])
async def test_an_accept_refused_by_the_app_is_told_by_its_code(app, bearer, refusal):
    app.accept_outcomes.append(ToolOutcome(refusal.value, is_error=True))

    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        assert await session.accept(Accept("room-7", "inv-3", 0)) is refusal


async def test_an_accept_error_of_no_known_code_is_unrecognised(app, bearer):
    app.accept_outcomes.append(ToolOutcome("something else", is_error=True))

    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        assert await session.accept(Accept("room-7", "inv-3", 0)) is Refusal.UNRECOGNISED


async def test_a_renewed_token_is_the_bearer_of_the_next_call(app, bearer):
    renewed = Secret(secrets.token_urlsafe(24))
    app.let_in(renewed.reveal(), SUBJECT)

    async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
        await session.wait(None)
        del app.bearers[bearer.reveal()]
        session.use_token(renewed)
        await session.wait(None)


@pytest.mark.parametrize("status", [HTTPStatus.FORBIDDEN, HTTPStatus.BAD_REQUEST, HTTPStatus.SERVICE_UNAVAILABLE])
async def test_an_http_status_the_app_answers_reaches_the_loop_as_that_status(app, bearer, status):
    app.refuse_with = status

    with pytest.raises(StatusFailure) as failure:
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()):
            pass

    assert failure.value.status == status


async def test_a_bearer_the_app_does_not_know_is_a_401(app):
    with pytest.raises(StatusFailure) as failure:
        async with open_mcp_session(app_target(), Secret("unknown"), transport=app.asgi()):
            pass

    assert failure.value.status == HTTPStatus.UNAUTHORIZED


async def test_a_status_after_the_session_started_reaches_the_loop_as_that_status(app, bearer):
    async def waits_until_the_app_refuses():
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
            await session.wait(None)
            app.refuse_with = HTTPStatus.FORBIDDEN
            await session.wait(None)

    with pytest.raises(StatusFailure) as failure:
        await waits_until_the_app_refuses()

    assert failure.value.status == HTTPStatus.FORBIDDEN


async def test_an_app_that_cannot_be_reached_is_unreachable(bearer):
    def refuse_connection(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    with pytest.raises(Unreachable):
        async with open_mcp_session(app_target(), bearer, transport=httpx2.MockTransport(refuse_connection)):
            pass
