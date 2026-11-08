"""The SDK client against agent_runtime_client's own server, in process: the real wire."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from http import HTTPStatus

import httpx2
import pytest
from agent_runtime_client.tools import ToolOutcome
from mcp_types import INVALID_PARAMS

from agents.app_session import INVALID_ANSWER_CODE, open_mcp_session
from agents.decision import Accept, Refusal, status_code
from agents.failures import CallFailure, StatusFailure, Unreachable, UnusableAnswer, WaitToolError
from agents.main import configure_logging
from agents.secret import Secret
from tests.support.fakes import app_target
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import FakeApp, active, invited

pytestmark = pytest.mark.anyio

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

    opening = open_mcp_session(app_target(), bearer, transport=app.asgi())

    with pytest.raises(StatusFailure) as failure:
        async with opening:
            pass

    assert failure.value.status == status


async def test_a_bearer_the_app_does_not_know_is_a_401(app):
    opening = open_mcp_session(app_target(), Secret(secrets.token_urlsafe(24)), transport=app.asgi())

    with pytest.raises(StatusFailure) as failure:
        async with opening:
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

    opening = open_mcp_session(app_target(), bearer, transport=httpx2.MockTransport(refuse_connection))

    with pytest.raises(Unreachable):
        async with opening:
            pass


class BodyError(Exception):
    pass


async def test_an_error_raised_in_the_session_body_reaches_the_caller_unchanged(app, bearer):
    raised = BodyError("the caller's own code failed")

    async def fails_in_the_body():
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
            await session.wait(None)
            raise raised

    with pytest.raises(BodyError) as caught:
        await fails_in_the_body()

    assert caught.value is raised


async def test_a_wait_tool_that_answers_an_error_is_a_wait_tool_error(app, bearer):
    app.standings_fail = True

    async def waits_for_a_failing_tool():
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
            await session.wait(None)

    with pytest.raises(WaitToolError):
        await waits_for_a_failing_tool()


async def test_a_wait_answer_that_is_no_standing_is_an_unusable_answer(app, bearer):
    app.wait_answers_with = "no standings here"

    async def waits_for_a_wrong_answer():
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
            await session.wait(None)

    with pytest.raises(UnusableAnswer) as failure:
        await waits_for_a_wrong_answer()

    assert failure.value.code == INVALID_ANSWER_CODE


async def test_an_app_without_the_accept_tool_answers_a_protocol_error_that_is_an_unusable_answer(app, bearer):
    app.offers_accept = False

    async def accepts_where_it_is_not_offered():
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
            await session.accept(Accept("room-7", "inv-3", 0))

    with pytest.raises(UnusableAnswer) as failure:
        await accepts_where_it_is_not_offered()

    assert failure.value.code == f"jsonrpc_{INVALID_PARAMS}"


async def test_a_redirect_the_app_answers_is_an_unusable_answer_named_by_its_status(app, bearer):
    app.refuse_with = HTTPStatus.TEMPORARY_REDIRECT

    async def waits_behind_a_redirect():
        async with open_mcp_session(app_target(), bearer, transport=app.asgi()) as session:
            await session.wait(None)

    with pytest.raises(UnusableAnswer) as failure:
        await waits_behind_a_redirect()

    assert failure.value.code == status_code(HTTPStatus.TEMPORARY_REDIRECT)


class SeveredTransport(httpx2.AsyncBaseTransport):
    """Serves through the app until severed, then fails every request as a lost connection does."""

    def __init__(self, app: FakeApp) -> None:
        self.inner = app.asgi()
        self.severed = False
        self.failure: Exception = httpx2.ConnectError("connection lost")

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if self.severed:
            raise self.failure
        return await self.inner.handle_async_request(request)


async def test_a_connection_lost_after_the_session_started_is_unreachable(app, bearer):
    transport = SeveredTransport(app)

    async def waits_after_the_connection_is_lost():
        async with open_mcp_session(app_target(), bearer, transport=transport) as session:
            await session.wait(None)
            transport.severed = True
            await session.wait(None)

    with pytest.raises(Unreachable):
        await waits_after_the_connection_is_lost()


class AnswersToolCallsWith(httpx2.AsyncBaseTransport):
    """Serves through the app, except that every tool call is answered with the given server-sent events."""

    def __init__(self, app: FakeApp, *messages: dict[str, object]) -> None:
        self.inner = app.asgi()
        self.messages = messages

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        call = json.loads(request.content) if request.content else {}
        if call.get("method") != "tools/call":
            return await self.inner.handle_async_request(request)
        replies = [
            {"jsonrpc": "2.0", **message, **({"id": call["id"]} if "result" in message else {})}
            for message in self.messages
        ]
        events = "".join(f"event: message\ndata: {json.dumps(reply)}\n\n" for reply in replies)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=events.encode())


@pytest.mark.parametrize("tool_call", ["wait", "accept"])
async def test_a_tool_result_the_sdk_cannot_validate_is_an_unusable_answer(app, bearer, tool_call):
    transport = AnswersToolCallsWith(app, {"result": {"content": 7}})

    async def calls_the_tool():
        async with open_mcp_session(app_target(), bearer, transport=transport) as session:
            if tool_call == "wait":
                await session.wait(None)
            else:
                await session.accept(Accept("room-7", "inv-3", 0))

    with pytest.raises(UnusableAnswer) as failure:
        await calls_the_tool()

    assert failure.value.code == INVALID_ANSWER_CODE


async def test_a_malformed_notification_of_the_app_logs_none_of_its_content(app, bearer, library_loggers, caplog):
    configure_logging()
    caplog.set_level(logging.DEBUG)
    leaked = "content-of-a-malformed-notification"
    transport = AnswersToolCallsWith(
        app,
        {"method": "notifications/progress", "params": {"progressToken": 1, "progress": leaked}},
        {"result": {"content": [], "isError": False}},
    )

    async def waits():
        async with open_mcp_session(app_target(), bearer, transport=transport) as session:
            await session.wait(None)

    with pytest.raises(UnusableAnswer):
        await waits()

    assert leaked not in caplog.text


async def test_an_error_that_is_no_failure_of_the_app_is_not_turned_into_one(app, bearer):
    transport = SeveredTransport(app)
    transport.failure = BodyError("a programming error")

    async def waits_after_the_transport_breaks():
        async with open_mcp_session(app_target(), bearer, transport=transport) as session:
            await session.wait(None)
            transport.severed = True
            await session.wait(None)

    with pytest.raises(BaseException) as caught:  # noqa: PT011 - the SDK may wrap it in a group
        await waits_after_the_transport_breaks()

    assert not isinstance(caught.value, CallFailure)
