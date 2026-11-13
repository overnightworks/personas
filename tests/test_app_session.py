"""The SDK client against the library's own server on a loopback socket: the real wire."""

from __future__ import annotations

import asyncio
import logging
import secrets
import subprocess
from contextlib import asynccontextmanager
from http import HTTPStatus

import anyio
import pytest
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    SPEAK_TOOL,
    STANDING_URI,
    AcceptInvitationArguments,
    AcceptInvitationRefusal,
    AcceptInvitationRefused,
    FigureRef,
    PlaceRef,
    SpeakRefusal,
    SpeakRefused,
    Spoken,
)
from agents.tools import ToolEffect, ToolOutcome
from mcp_types import (
    INVALID_PARAMS,
    BlobResourceContents,
    ListToolsResult,
    ReadResourceResult,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)
from pydantic import ValidationError

from personas import app_session
from personas.app_session import open_mcp_session
from personas.config import AppTarget
from personas.decision import Accept, UnrecognisedRefusal, jsonrpc_code, status_code
from personas.failures import (
    INVALID_ANSWER_CODE,
    LISTEN_NOT_HONORED_CODE,
    StatusFailure,
    Unreachable,
    UnusableAnswer,
    leaves_of,
)
from personas.main import configure_logging
from personas.secret import Secret
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import (
    CALL,
    LIMIT_SECONDS,
    LIST_TOOLS,
    LISTEN,
    READ,
    SAY_TOOL,
    SPOKEN_DURATION_MS,
    FakeApp,
    active,
    formatted,
    invited,
)
from tests.support.loopback import serving
from tests.support.transport import (
    transport_closing_with_late_answers,
    transport_closing_with_leaves,
    transport_whose_close_loses_a_child,
)

pytestmark = pytest.mark.anyio

SHORT_STREAM_SECONDS = 0.3
POLLING_NAMES = ["agents.wait", "WaitAnswer", "WAIT_TOOL", "WaitToolError", "_cursor"]
SEARCHED = ["src", "tests", "README.md", "docs"]
THIS_FILE = ":(exclude)tests/test_app_session.py"
PLACE = PlaceRef(kind="call", id="room-7")


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
    """At a test's end no log line, traceback included, holds a bearer the app saw or content it served or took."""
    configure_logging()
    caplog.set_level(logging.DEBUG)
    yield
    logged = formatted(caplog.get_records("call"))
    assert bearer.reveal() in app.kept_out_of_logs()
    assert not [held for held in app.kept_out_of_logs() if held in logged]


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


async def test_an_accept_sends_the_figure_it_carries_and_without_one_no_figure_key(app, bearer):
    place = PlaceRef(kind="call", id="room-7")
    figure = FigureRef(url="https://figures.test/figure.glb", sha256=secrets.token_hex(32))

    async with opened(app, bearer) as session:
        await session.accept(Accept(place, "inv-3", 2, figure))
        await session.accept(Accept(place, "inv-4", 2))

    [(_, with_figure), (_, without)] = app.accepted_on_the_wire
    assert with_figure["figure"] == {"url": figure.url, "sha256": figure.sha256}
    assert "figure" not in without


@pytest.mark.parametrize("refusal", list(AcceptInvitationRefusal))
async def test_an_accept_refused_with_a_code_of_the_record_is_told_by_that_code(app, bearer, refusal):
    app.accept_outcomes.append(ToolOutcome(AcceptInvitationRefused(refused=refusal).model_dump_json(), is_error=True))

    async with opened(app, bearer) as session:
        assert await session.accept(Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0)) is refusal


@pytest.mark.parametrize("text", ['{"refused": "own_account_only"}', "something else", "busy"])
async def test_an_accept_error_outside_the_record_is_unrecognised(app, bearer, text):
    app.accept_outcomes.append(ToolOutcome(text, is_error=True))

    async with opened(app, bearer) as session:
        assert (
            await session.accept(Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0))
            is UnrecognisedRefusal.UNRECOGNISED
        )


async def test_a_speak_sends_the_place_and_the_sentence_and_gives_how_long_the_app_says_it_plays(app, bearer):
    sentence = secrets.token_hex(4)

    async with opened(app, bearer) as session:
        spoken = await session.speak(PLACE, sentence)

    assert spoken == Spoken(duration_ms=SPOKEN_DURATION_MS)
    assert app.spoken == [(SUBJECT, PLACE.id, sentence)]


@pytest.mark.parametrize("refusal", list(SpeakRefusal))
async def test_a_speak_refused_with_a_code_of_the_record_is_told_by_that_code(app, bearer, refusal):
    app.speak_outcomes.append(ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True))

    async with opened(app, bearer) as session:
        assert await session.speak(PLACE, "A sentence.") is refusal


@pytest.mark.parametrize(
    "outcome",
    [
        ToolOutcome("not json", is_error=False),
        ToolOutcome('{"duration_ms": 0}', is_error=False),
        ToolOutcome(SpeakRefused(refused=SpeakRefusal.TOO_LONG).model_dump_json(), is_error=False),
        ToolOutcome('{"refused": "other"}', is_error=True),
        ToolOutcome(Spoken(duration_ms=1).model_dump_json(), is_error=True),
    ],
    ids=["no json", "no duration", "a refusal that succeeded", "an unknown refusal", "a duration that failed"],
)
async def test_a_speak_answer_that_is_neither_a_spoken_nor_a_refusal_of_the_record_is_an_unusable_answer(
    app, bearer, outcome
):
    app.speak_outcomes.append(outcome)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await session.speak(PLACE, "A sentence.")

    assert failure.value.code == INVALID_ANSWER_CODE


async def test_a_speak_the_app_refuses_is_a_failure(app, bearer):
    app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    app.refused_method = CALL

    async with opened(app, bearer) as session:
        with pytest.raises(StatusFailure) as failure:
            await session.speak(PLACE, "A sentence.")

    assert failure.value.status == HTTPStatus.SERVICE_UNAVAILABLE


async def test_a_renewed_token_is_the_bearer_of_the_next_call(app, bearer):
    renewed = Secret(secrets.token_urlsafe(24))
    app.let_in(renewed.reveal(), SUBJECT)

    async with opened(app, bearer) as session:
        await session.read_standing()
        del app.bearers[bearer.reveal()]
        session.use_token(renewed)
        await session.read_standing()


async def test_a_bearer_the_app_does_not_know_is_a_401(app):
    unknown = Secret(secrets.token_urlsafe(24))

    with pytest.raises(StatusFailure) as failure:
        await opens(app, unknown)

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


async def raises_after_a_refused_call(app, bearer, raised):
    async with opened(app, bearer) as session:
        del app.bearers[bearer.reveal()]
        with pytest.raises(StatusFailure):
            await session.read_standing()
        raise raised


@pytest.mark.parametrize(
    "own", [BodyError("the caller's own code failed"), OSError("a local file failed")], ids=["defect", "os error"]
)
async def test_a_leaf_the_body_raises_after_the_app_answered_4xx_reaches_the_caller_unchanged(app, bearer, own):
    group = ExceptionGroup("own", [own])

    with pytest.raises(type(own)) as caught:
        await raises_after_a_refused_call(app, bearer, group)

    assert caught.value is own


@pytest.mark.parametrize(
    "own",
    [
        OSError("a local file failed"),
        TimeoutError("the caller's own wait timed out"),
        ValidationError.from_exception_data("a local record", []),
    ],
    ids=["os error", "timeout", "validation error"],
)
async def test_every_leaf_of_nested_groups_the_body_raises_reaches_the_caller(app, bearer, own):
    failure, defect = Unreachable("raised by the caller"), BodyError("the caller's own code failed")
    nested = ExceptionGroup("outer", [failure, ExceptionGroup("inner", [own, defect])])

    with pytest.raises(ExceptionGroup) as caught:
        await raises_in_the_stream_body(app, bearer, nested)

    assert list(leaves_of(caught.value)) == [failure, own, defect]


@pytest.mark.parametrize("status", [HTTPStatus.FORBIDDEN, HTTPStatus.SERVICE_UNAVAILABLE])
async def test_an_error_answer_whose_body_does_not_decode_is_told_by_its_status(app, bearer, status):
    async def answers_undecodably(reply):
        await reply.undecodable(status)

    app.script(CALL, answers_undecodably)

    with pytest.raises(StatusFailure) as failure:
        await accepts_a_place(app, bearer)

    assert failure.value.status == status


async def test_an_ok_answer_whose_body_does_not_decode_is_unreachable(app, bearer):
    async def answers_undecodably(reply):
        await reply.undecodable(HTTPStatus.OK)

    app.script(CALL, answers_undecodably)

    with pytest.raises(Unreachable):
        await accepts_a_place(app, bearer)


async def test_a_stream_that_outlasts_its_deadline_by_sending_on_is_unreachable(app, bearer):
    async def sends_the_next_once_the_last_was_read(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        reads = 0
        while True:
            await reply.pulse(STANDING_URI)
            reads += 1
            await app.log.until(READ, reads)

    app.script(LISTEN, sends_the_next_once_the_last_was_read)

    with pytest.raises((Unreachable, ExceptionGroup)) as caught:
        await listens_and_reads_on_every_wake(app, bearer, stream_seconds=SHORT_STREAM_SECONDS)

    assert all(isinstance(leaf, Unreachable) for leaf in leaves_of(caught.value))
    assert app.log.count(READ) > 1


@pytest.fixture(params=[anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"])
def late_answer(request, monkeypatch):
    def fails_as_it_closes(*, echoes=1):
        failing = transport_closing_with_late_answers(echo=request.param, echoes=echoes)
        monkeypatch.setattr(app_session, "streamable_http_client", failing)
        return request.param

    return fails_as_it_closes


async def test_an_answer_that_lands_after_the_session_closed_is_unreachable_beside_the_failure_that_closed_it(
    app, bearer, late_answer
):
    late_answer()
    deadline = Unreachable("the stream outlasted its deadline")

    with pytest.raises(ExceptionGroup) as caught:
        await raises_in_the_stream_body(app, bearer, deadline)

    assert [type(leaf) for leaf in leaves_of(caught.value)] == [Unreachable, Unreachable]


async def test_an_answer_refused_before_the_session_closed_does_not_outrank_the_failure_that_closed_it(
    app, bearer, late_answer
):
    late_answer()
    deadline = Unreachable("the stream outlasted its deadline")

    with pytest.raises(ExceptionGroup) as caught:
        await raises_after_a_refused_call(app, bearer, deadline)

    assert [type(leaf) for leaf in leaves_of(caught.value)] == [Unreachable, Unreachable]


async def test_an_answer_landing_after_a_session_closed_by_an_unrequested_cancellation_stays_the_error_it_is(
    app, bearer, late_answer
):
    echo = late_answer()
    cancellation = asyncio.CancelledError()

    with pytest.raises(echo):
        await raises_in_the_stream_body(app, bearer, cancellation)


async def test_answers_all_landing_after_a_session_closed_by_an_unrequested_cancellation_stay_the_errors_they_are(
    app, bearer, late_answer
):
    echo = late_answer(echoes=2)
    cancellation = asyncio.CancelledError()

    with pytest.raises(ExceptionGroup) as caught:
        await raises_in_the_stream_body(app, bearer, cancellation)

    assert [type(leaf) for leaf in leaves_of(caught.value)] == [echo, echo]


async def waits_in_the_stream_body(app, bearer, inside):
    async with opened(app, bearer) as session, session.changes():
        inside.set()
        await asyncio.Event().wait()


async def cancelled_in_the_stream_body(app, bearer):
    """Runs a session whose body waits and cancels its task there; what the task ended with."""
    inside = asyncio.Event()
    task = asyncio.create_task(waits_in_the_stream_body(app, bearer, inside))
    async with asyncio.timeout(LIMIT_SECONDS):
        await inside.wait()
    task.cancel()
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_late_answers_alone_after_a_session_body_was_left_by_a_cancellation_end_as_the_cancellation(
    app, bearer, late_answer
):
    late_answer(echoes=2)

    ended = await cancelled_in_the_stream_body(app, bearer)

    assert isinstance(ended, asyncio.CancelledError)


async def test_a_defect_beside_late_answers_after_a_cancellation_stays_with_them_as_the_errors_they_are(
    app, bearer, monkeypatch
):
    defect = ValueError("a defect")
    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_closing_with_leaves(defect, anyio.BrokenResourceError())
    )

    ended = await cancelled_in_the_stream_body(app, bearer)

    assert isinstance(ended, ExceptionGroup)
    assert [type(leaf) for leaf in leaves_of(ended)] == [ValueError, Unreachable]


@pytest.mark.parametrize(
    "raised",
    [BodyError("the caller's own code failed"), StatusFailure(HTTPStatus.FORBIDDEN)],
    ids=["a defect", "a refusal"],
)
async def test_what_the_body_raises_stays_among_the_leaves_when_the_close_then_loses_a_child(
    app, bearer, monkeypatch, raised
):
    monkeypatch.setattr(app_session, "streamable_http_client", transport_whose_close_loses_a_child(OSError))

    with pytest.raises(ExceptionGroup) as caught:
        await raises_in_the_stream_body(app, bearer, raised)

    assert [leaf for leaf in leaves_of(caught.value) if leaf is raised] == [raised]
    assert sorted(type(leaf).__name__ for leaf in leaves_of(caught.value)) == sorted(
        [type(raised).__name__, "Unreachable"]
    )


async def test_traffic_with_no_acknowledgement_until_past_the_deadline_is_unreachable(app, bearer):
    talked = []

    async def talks_without_acknowledging(reply):
        talked.append(reply)
        await reply.open_stream()
        await reply.keep_alives_until_closed()

    app.script(LISTEN, talks_without_acknowledging)

    with pytest.raises(Unreachable):
        await listens(app, bearer, stream_seconds=SHORT_STREAM_SECONDS)

    assert talked[0].keep_alives > 1


async def test_an_accept_without_the_tool_is_a_protocol_error_that_is_an_unusable_answer(app, bearer):
    async def knows_no_such_tool(reply):
        await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)

    app.script(CALL, knows_no_such_tool)

    with pytest.raises(UnusableAnswer) as failure:
        await accepts_a_place(app, bearer)

    assert failure.value.code == jsonrpc_code(INVALID_PARAMS)


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


async def test_a_window_holds_the_lines_of_the_place_and_marks_the_members_own(app, bearer):
    heard, own = secrets.token_hex(4), secrets.token_hex(4)
    app.stand(SUBJECT, active("room-7"))
    app.speak("room-7", "visitor", "Ada", heard)
    app.speak("room-7", SUBJECT, "Mozart", own)

    async with opened(app, bearer) as session:
        window = await session.read_place(PLACE)

    assert window is not None
    assert [(each.seq, each.text, each.own) for each in window.entries] == [(1, heard, False), (2, own, True)]


@pytest.mark.parametrize("arrangement", ["the app lost the place", "not in the place", "answered inside a 200"])
async def test_a_window_the_app_shows_the_member_no_more_is_none(app, bearer, arrangement):
    async def knows_no_such_resource(reply):
        await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)

    app.stand(SUBJECT, active("room-7"))
    if arrangement == "the app lost the place":
        app.vanished.add("room-7")
    elif arrangement == "not in the place":
        app.stand(SUBJECT)
    else:
        app.script(READ, knows_no_such_resource)

    async with opened(app, bearer) as session:
        window = await session.read_place(PLACE)

    assert window is None


async def test_a_window_read_the_app_refuses_is_a_failure_and_no_missing_window(app, bearer):
    app.stand(SUBJECT, active("room-7"))
    app.refused_method = READ
    app.refuse_with = HTTPStatus.FORBIDDEN

    async with opened(app, bearer) as session:
        with pytest.raises(StatusFailure):
            await session.read_place(PLACE)


@pytest.mark.parametrize(
    "contents",
    [
        TextResourceContents(uri="member://standing/places/call/room-7", text="{}", mime_type="application/json"),
        BlobResourceContents(uri="member://standing/places/call/room-7", blob="e30=", mime_type="application/json"),
    ],
    ids=["not a window", "not text"],
)
async def test_a_window_that_is_no_window_is_an_unusable_answer(app, bearer, contents):
    async def answers_wrongly(reply):
        await reply.result(ReadResourceResult(contents=[contents], ttl_ms=0, cache_scope="private"))

    app.stand(SUBJECT, active("room-7"))
    app.script(READ, answers_wrongly)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await session.read_place(PLACE)

    assert failure.value.code == INVALID_ANSWER_CODE


TABLE = PlaceRef(kind="table", id="table-1")


async def test_a_view_holds_the_revision_the_wait_and_the_state_as_the_app_sent_them(app, bearer):
    state = {"hand": secrets.token_hex(4), "stack": [1, {"seen": True}]}
    app.stand(SUBJECT, active(TABLE.id, TABLE.kind, serves_view=True))
    app.show_view(TABLE.id, 7, state)

    async with opened(app, bearer) as session:
        view = await session.read_view(TABLE)

    assert view is not None
    assert (view.revision, view.awaits_you, view.state) == (7, True, state)


@pytest.mark.parametrize("arrangement", ["the app lost the place", "not in the place", "answered inside a 200"])
async def test_a_view_the_app_shows_the_member_no_more_is_none(app, bearer, arrangement):
    async def knows_no_such_resource(reply):
        await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)

    app.stand(SUBJECT, active(TABLE.id, TABLE.kind, serves_view=True))
    app.show_view(TABLE.id, 1, {})
    if arrangement == "the app lost the place":
        app.vanished.add(TABLE.id)
    elif arrangement == "not in the place":
        app.stand(SUBJECT)
    else:
        app.script(READ, knows_no_such_resource)

    async with opened(app, bearer) as session:
        view = await session.read_view(TABLE)

    assert view is None


async def test_a_view_that_is_no_view_is_an_unusable_answer(app, bearer):
    async def answers_wrongly(reply):
        contents = TextResourceContents(uri="member://standing", text="{}", mime_type="application/json")
        await reply.result(ReadResourceResult(contents=[contents], ttl_ms=0, cache_scope="private"))

    app.stand(SUBJECT, active(TABLE.id, TABLE.kind, serves_view=True))
    app.script(READ, answers_wrongly)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await session.read_view(TABLE)

    assert failure.value.code == INVALID_ANSWER_CODE


async def test_the_declared_tools_are_the_ones_the_app_lists(app, bearer):
    async with opened(app, bearer) as session:
        declared = await session.declared_tools()

    assert {each.name for each in declared} == {ACCEPT_INVITATION_TOOL, SAY_TOOL, SPEAK_TOOL}


async def test_a_tool_marked_read_only_is_declared_read_only(app, bearer):
    async def lists_a_read_only_tool(reply):
        tool = Tool(
            name="peek",
            description="Looks.",
            input_schema={"type": "object"},
            annotations=ToolAnnotations(read_only_hint=True),
        )
        await reply.result(ListToolsResult(tools=[tool]))

    app.script(LIST_TOOLS, lists_a_read_only_tool)

    async with opened(app, bearer) as session:
        declared = await session.declared_tools()

    assert [(each.name, each.effect) for each in declared] == [("peek", ToolEffect.READ_ONLY)]


async def test_a_declaration_that_fails_validation_is_an_unusable_answer(app, bearer):
    async def lists_a_tool_without_a_description(reply):
        await reply.result(ListToolsResult(tools=[Tool(name="peek", input_schema={"type": "object"})]))

    app.script(LIST_TOOLS, lists_a_tool_without_a_description)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await session.declared_tools()

    assert failure.value.code == INVALID_ANSWER_CODE


@pytest.mark.parametrize("outcome", [ToolOutcome("said", is_error=False), ToolOutcome("gone", is_error=True)])
async def test_an_invoked_tool_gives_the_outcome_the_app_answered_an_error_one_too(app, bearer, outcome):
    app.say_outcomes.append(outcome)
    app.content.add(outcome.content)

    async with opened(app, bearer) as session:
        invoked = await session.invoke(SAY_TOOL, {"place": "room-7", "text": secrets.token_hex(4)})

    assert invoked == outcome


async def test_an_invoked_tool_the_app_refuses_is_a_failure(app, bearer):
    app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    app.refused_method = CALL
    arguments = {"place": "room-7", "text": secrets.token_hex(4)}

    async with opened(app, bearer) as session:
        with pytest.raises(StatusFailure) as failure:
            await session.invoke(SAY_TOOL, arguments)

    assert failure.value.status == HTTPStatus.SERVICE_UNAVAILABLE


@pytest.mark.parametrize("name", POLLING_NAMES)
async def test_nothing_in_the_tree_names_the_polling_path(name):
    search = await asyncio.create_subprocess_exec(
        "git", "grep", "-n", "-F", name, "--", *SEARCHED, THIS_FILE, stdout=subprocess.PIPE
    )
    stdout, _ = await search.communicate()

    assert stdout == b""
