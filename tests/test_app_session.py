"""The SDK client against agent_runtime_client's own server on a loopback socket: the real wire."""

from __future__ import annotations

import asyncio
import logging
import secrets
import subprocess
from contextlib import asynccontextmanager
from http import HTTPStatus

import pytest
from agent_runtime_client.member import ACCEPT_INVITATION_TOOL, STANDING_URI, AcceptInvitationArguments, PlaceRef
from agent_runtime_client.tools import ToolOutcome
from mcp_types import INVALID_PARAMS, ListToolsResult, Tool
from pydantic import ValidationError

from agents.app_session import open_mcp_session
from agents.config import AppTarget
from agents.decision import Accept, Refusal, status_code
from agents.failures import INVALID_ANSWER_CODE, LISTEN_NOT_HONORED_CODE, StatusFailure, Unreachable, UnusableAnswer
from agents.main import configure_logging
from agents.secret import Secret
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, LIMIT_SECONDS, FakeApp, active, invited
from tests.support.loopback import serving

pytestmark = pytest.mark.anyio

SHORT_STREAM_SECONDS = 0.3
POLLING_NAMES = ["agent_runtime_client.wait", "WaitAnswer", "WAIT_TOOL", "WaitToolError", "_cursor"]
SEARCHED = ["src", "tests", "README.md", "docs"]
THIS_FILE = ":(exclude)tests/test_app_session.py"
KEEP_ALIVE_INTERVAL_SECONDS = 0.01
PLACE = PlaceRef(kind="call", id="room-7")
PLACE_CONTENT = ("room-7", "inv-3")


@pytest.fixture
async def app():
    fake = FakeApp()
    async with fake.serving() as url:
        fake.url = url
        yield fake


@pytest.fixture
def bearer(app):
    token = Secret(secrets.token_urlsafe(24))
    app.let_in(token.reveal(), SUBJECT)
    return token


@pytest.fixture(autouse=True)
def nothing_secret_is_logged(app, bearer, library_loggers, caplog):
    """At a test's end no log line holds a token or the content of a place."""
    configure_logging()
    caplog.set_level(logging.DEBUG)
    yield
    logged = "\n".join(record.getMessage() for record in caplog.get_records("call"))
    for held in (bearer.reveal(), *PLACE_CONTENT):
        assert held not in logged


def opened(app, bearer, **keywords):
    return open_mcp_session(AppTarget(app.url, "scope"), bearer, **keywords)


async def opens(app, bearer):
    async with opened(app, bearer):
        pass


async def listens(app, bearer, **keywords):
    async with opened(app, bearer, **keywords) as session, session.changes():
        pass


async def listens_and_reads_on_every_wake(app, bearer, **keywords):
    async with opened(app, bearer, **keywords) as session, session.changes() as changes:
        async for _ in changes:
            await session.read_standing()


async def accepts_a_place(app, bearer):
    async with opened(app, bearer) as session:
        await session.accept(Accept(PLACE, "inv-3", 0))


async def test_a_stream_is_open_once_acknowledged_and_a_change_is_one_wake(app, bearer):
    async with opened(app, bearer) as session, session.changes() as changes:
        app.stand(SUBJECT, invited("room-7", "inv-3"))
        async with asyncio.timeout(LIMIT_SECONDS):
            await anext(aiter(changes))
        standing = await session.read_standing()

    assert [each.place.id for each in standing.standings] == ["room-7"]


async def test_a_read_answers_every_standing_of_the_member(app, bearer):
    app.stand(SUBJECT, invited("room-7", "inv-3", epoch=2), active("room-8"))

    async with opened(app, bearer) as session:
        standing = await session.read_standing()

    assert [(each.place.id, each.state.value) for each in standing.standings] == [
        ("room-7", "invited"),
        ("room-8", "active"),
    ]


async def test_an_app_that_does_not_list_the_standing_does_not_honor_the_stream(app, bearer):
    app.lists_standing = False

    with pytest.raises(UnusableAnswer) as failure:
        await listens(app, bearer)

    assert failure.value.code == LISTEN_NOT_HONORED_CODE


async def test_a_stream_the_app_completes_without_acknowledging_is_not_honored(app, bearer):
    async def completes_at_once(reply):
        await reply.open_stream()
        await reply.complete()

    app.script("subscriptions/listen", completes_at_once)

    with pytest.raises(UnusableAnswer) as failure:
        await listens(app, bearer)

    assert failure.value.code == LISTEN_NOT_HONORED_CODE


async def test_a_stream_the_app_completes_ends_the_iteration_and_a_new_one_opens_on_the_same_session(app, bearer):
    async def acknowledges_and_completes(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        await reply.complete()

    app.script("subscriptions/listen", acknowledges_and_completes)

    async with opened(app, bearer) as session:
        async with session.changes() as changes:
            assert [wake async for wake in changes] == []
        async with session.changes():
            pass


async def test_an_accept_sends_the_place_invitation_and_epoch_and_a_success_is_no_refusal(app, bearer):
    place = PlaceRef(kind="call", id="room-7")

    async with opened(app, bearer) as session:
        refusal = await session.accept(Accept(place, "inv-3", 2))

    assert refusal is None
    assert app.accepted == [AcceptInvitationArguments(place=place, invitation="inv-3", epoch=2)]


@pytest.mark.parametrize("refusal", [Refusal.STALE_INVITATION, Refusal.SPONSOR_AWAY, Refusal.TWO_SEATS, Refusal.BUSY])
async def test_an_accept_refused_by_the_app_is_told_by_its_code(app, bearer, refusal):
    app.accept_outcomes.append(ToolOutcome(refusal.value, is_error=True))

    async with opened(app, bearer) as session:
        assert await session.accept(Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0)) is refusal


async def test_an_accept_error_of_no_known_code_is_unrecognised(app, bearer):
    app.accept_outcomes.append(ToolOutcome("something else", is_error=True))

    async with opened(app, bearer) as session:
        assert await session.accept(Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0)) is Refusal.UNRECOGNISED


async def test_a_renewed_token_is_the_bearer_of_the_next_call(app, bearer):
    renewed = Secret(secrets.token_urlsafe(24))
    app.let_in(renewed.reveal(), SUBJECT)

    async with opened(app, bearer) as session:
        await session.read_standing()
        del app.bearers[bearer.reveal()]
        session.use_token(renewed)
        await session.read_standing()


async def test_a_bearer_the_app_does_not_know_is_a_401(app):
    with pytest.raises(StatusFailure) as failure:
        await opens(app, Secret(secrets.token_urlsafe(24)))

    assert failure.value.status == HTTPStatus.UNAUTHORIZED


async def test_a_redirect_the_app_answers_is_an_unusable_answer_named_by_its_status(app, bearer):
    async def redirects(reply):
        await reply.status(HTTPStatus.TEMPORARY_REDIRECT)

    app.script(CALL, redirects)

    with pytest.raises(UnusableAnswer) as failure:
        await accepts_a_place(app, bearer)

    assert failure.value.code == status_code(HTTPStatus.TEMPORARY_REDIRECT)


async def opens_at(mcp_url, bearer):
    async with open_mcp_session(AppTarget(mcp_url, "scope"), bearer):
        pass


async def test_an_app_that_cannot_be_reached_is_unreachable(app, bearer):
    @asynccontextmanager
    async def a_closed_port():
        async with serving(lambda *_: None) as origin:
            yield origin

    async with a_closed_port() as origin:
        pass

    with pytest.raises(Unreachable):
        await opens_at(f"{origin}/mcp", bearer)


class BodyError(Exception):
    pass


async def raises_in_the_stream_body(app, bearer, raised):
    async with opened(app, bearer) as session, session.changes():
        raise raised


@pytest.mark.parametrize(
    "raised",
    [
        BodyError("the caller's own code failed"),
        OSError("a local file failed"),
        TimeoutError("the caller's own wait timed out"),
        ValidationError.from_exception_data("a local record", []),
        Unreachable("raised by the caller"),
    ],
    ids=["its own error", "os error", "timeout", "validation error", "call failure"],
)
async def test_an_error_raised_in_the_stream_body_reaches_the_caller_unchanged(app, bearer, raised):
    with pytest.raises(type(raised)) as caught:
        await raises_in_the_stream_body(app, bearer, raised)

    assert caught.value is raised


async def test_a_stream_that_outlasts_its_deadline_by_sending_on_is_unreachable(app, bearer):
    async def sends_on_and_on(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        while True:
            await reply.pulse(STANDING_URI)
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)

    app.script("subscriptions/listen", sends_on_and_on)

    with pytest.raises(Unreachable):
        await listens_and_reads_on_every_wake(app, bearer, stream_seconds=SHORT_STREAM_SECONDS)


async def test_traffic_with_no_acknowledgement_until_past_the_deadline_is_unreachable(app, bearer):
    async def talks_without_acknowledging(reply):
        await reply.open_stream()
        while True:
            await reply.keep_alive()
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)

    app.script("subscriptions/listen", talks_without_acknowledging)

    with pytest.raises(Unreachable):
        await listens(app, bearer, stream_seconds=SHORT_STREAM_SECONDS)


async def test_an_accept_without_the_tool_is_a_protocol_error_that_is_an_unusable_answer(app, bearer):
    async def knows_no_such_tool(reply):
        await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)

    app.script(CALL, knows_no_such_tool)

    with pytest.raises(UnusableAnswer) as failure:
        await accepts_a_place(app, bearer)

    assert failure.value.code == f"jsonrpc_{INVALID_PARAMS}"


@pytest.mark.parametrize(
    "answer",
    [
        {"content": 7},
        {"resultType": "complete", "content": [], "structuredContent": {"standings": 7}},
        {"resultType": "complete", "content": []},
        {"resultType": "input_required", "requestState": "x"},
    ],
    ids=[
        "malformed result",
        "structured content against the output schema",
        "structured content missing",
        "input required",
    ],
)
async def test_an_accept_result_the_sdk_cannot_validate_is_an_unusable_answer(app, bearer, answer):
    schema = {"type": "object", "properties": {"standings": {"type": "array"}}, "required": ["standings"]}

    async def lists_a_tool_with_an_output_schema(reply):
        tool = Tool(name=ACCEPT_INVITATION_TOOL, input_schema={"type": "object"}, output_schema=schema)
        await reply.result(ListToolsResult(tools=[tool]))

    async def answers_wrongly(reply):
        await reply.raw_result(answer)

    app.script("tools/list", lists_a_tool_with_an_output_schema)
    app.script(CALL, answers_wrongly)

    with pytest.raises(UnusableAnswer) as failure:
        await accepts_a_place(app, bearer)

    assert failure.value.code == INVALID_ANSWER_CODE


async def test_a_malformed_notification_of_the_app_logs_none_of_its_content(app, bearer, caplog):
    leaked = "content-of-a-malformed-notification"

    async def notifies_wrongly_then_answers(reply):
        await reply.open_stream()
        await reply.event(
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 1, "progress": leaked}}
        )
        await reply.event({"jsonrpc": "2.0", "id": reply.rpc.id, "result": {"content": [], "isError": False}})
        await reply.end()

    app.script(CALL, notifies_wrongly_then_answers)

    with pytest.raises(UnusableAnswer):
        await accepts_a_place(app, bearer)

    assert leaked not in caplog.text


@pytest.mark.parametrize("name", POLLING_NAMES)
async def test_nothing_in_the_tree_names_the_polling_path(name):
    search = await asyncio.create_subprocess_exec(
        "git", "grep", "-n", "-F", name, "--", *SEARCHED, THIS_FILE, stdout=subprocess.PIPE
    )
    stdout, _ = await search.communicate()

    assert stdout == b""
