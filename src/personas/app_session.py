"""The app boundary: one MCP session of the official SDK against an app's ``/mcp``."""

from __future__ import annotations

import asyncio
import contextvars
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Final, Protocol, cast, override

import httpx2
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    CONTINUE_SPEAKING_TOOL,
    LISTEN_BOUND_SECONDS,
    SET_TURN_STATE_TOOL,
    SPEAK_TOOL,
    STANDING_URI,
    STOP_SPEAKING_TOOL,
    AcceptInvitationArguments,
    AcceptInvitationRefused,
    ContinueSpeakingRefusal,
    ContinueSpeakingRefused,
    MemberResourceRecord,
    MemberStanding,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SetTurnStateArguments,
    SetTurnStateRefusal,
    SetTurnStateRefused,
    SpeakAnswer,
    SpeakRefusal,
    SpeakRefused,
    StopSpeakingRefusal,
    StopSpeakingRefused,
    member_resource_from_json,
    place_view_uri,
    place_window_uri,
)
from agents.tools import JsonObject, ToolDeclaration, ToolEffect, ToolImage, ToolOutcome
from agents.turn import TOOL_IMAGE_MEDIA_TYPE
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import Subscription, SubscriptionLost, ToolsListChanged, listen
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CONNECTION_CLOSED,
    INVALID_PARAMS,
    REQUEST_TIMEOUT,
    UNSUPPORTED_PROTOCOL_VERSION,
    CallToolResult,
    ImageContent,
    InputRequiredResult,
    JSONRPCError,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
    jsonrpc_message_adapter,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

from personas.config import AppTarget
from personas.decision import Accept, Refusal, UnrecognisedRefusal, jsonrpc_code, status_code
from personas.failures import (
    INVALID_ANSWER_CODE,
    LATE_ANSWER_ERRORS,
    LISTEN_NOT_HONORED_CODE,
    UNSUPPORTED_VERSION_CODE,
    CallFailure,
    StatusFailure,
    Unreachable,
    UnusableAnswer,
    leaves_of,
    task_is_being_cancelled,
)
from personas.reporting import carry_trace, report_argument_left_out
from personas.secret import Secret

CONNECT_SECONDS: Final = 30.0
READ_SECONDS: Final = 2 * LISTEN_BOUND_SECONDS
"""Twice the bound an app puts on a stream: the runner cuts a stream that outlasts it."""
_SPEAK_ANSWER: Final = TypeAdapter(SpeakAnswer)
_MEDIA_TYPE: Final = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*")
"""A media type as a log line may carry it: a type and a subtype of token characters, no parameters."""
_TRANSPORT_ERROR_CODES: Final = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT})
"""What the SDK answers for a connection that broke or a request that timed out."""
_ACCEPT_OPTIONAL: Final = frozenset({"figure", "voices"})
_SPEAK_OPTIONAL: Final = frozenset({"style", "interrupted"})
_TURN_STATE_OPTIONAL: Final = frozenset({"thinking"})
"""The arguments the runner builds itself that a call may leave out; every other one of the three tools is
required."""


class AppSession(Protocol):
    def use_token(self, token: Secret) -> None:
        """Every later call carries this bearer."""
        ...

    def changes(self) -> AbstractAsyncContextManager[AsyncIterator[object]]:
        """A stream of wakes; entering returns once the app acknowledged it with the standing honored.

        Iterating yields once per wake and ends when the app ends the stream. The stream, its
        acknowledgement and everything the body does share one deadline. Raises ``StatusFailure``,
        ``Unreachable`` or ``UnusableAnswer``.
        """
        ...

    async def read_standing(self) -> MemberStanding:
        """Raises ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer``."""
        ...

    async def accept(self, accept: Accept) -> Refusal | None:
        """``None`` when joined, else the refusal; raises ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer``."""
        ...

    async def read_place(self, place: PlaceRef) -> PlaceWindow | None:
        """The place's window, or ``None`` when the app shows it to this member no more.

        Raises ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer``.
        """
        ...

    async def read_view(self, place: PlaceRef) -> PlaceView | None:
        """The place's view, or ``None`` when the app shows it to this member no more.

        Raises ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer``.
        """
        ...

    async def speak(
        self, place: PlaceRef, sentence: str, style: str | None = None, interrupted: str | None = None
    ) -> SpeakAnswer | SpeakRefusal:
        """That the sentence starts to play, that the app wrote it at the place instead, or why it does not speak it;
        ``style`` is how it sounds, passed on unchanged; ``interrupted`` names the hold the sentence answers instead of
        the held rest; raises ``StatusFailure``,
        ``Unreachable`` or ``UnusableAnswer``, the last for an answer that is none of these."""
        ...

    async def continue_speaking(self, place: PlaceRef, interrupted: str) -> ContinueSpeakingRefusal | None:
        """``None`` when the held sentence goes on, else the refusal; raises as ``speak`` does."""
        ...

    async def stop_speaking(self, place: PlaceRef, interrupted: str) -> StopSpeakingRefusal | None:
        """``None`` when the hold ended with nothing more said, else the refusal; raises as ``speak`` does."""
        ...

    async def set_turn_state(self, statement: SetTurnStateArguments) -> SetTurnStateRefusal | None:
        """``None`` when the app took what his turns at the place do, else the refusal; raises as ``speak`` does."""
        ...

    async def declared_tools(self) -> tuple[ToolDeclaration, ...]:
        """The first page of the app's tools; raises ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer``."""
        ...

    def kept_tools(self) -> tuple[ToolDeclaration, ...] | None:
        """The list ``declared_tools`` kept, or ``None`` when there is none.

        A list is kept while a stream that honoured the app's tools notice is open and no notice, acknowledgement or
        stream end came after the listing started.
        """
        ...

    def tools_notices(self) -> int:
        """How many tools notices this session received."""
        ...

    async def invoke(self, name: str, arguments: JsonObject) -> ToolOutcome:
        """The tool's outcome, an error one included; raises ``StatusFailure``, ``Unreachable``, ``UnusableAnswer``."""
        ...


type SessionOpener = Callable[[AppTarget, Secret, str], AbstractAsyncContextManager[AppSession]]


@dataclass(frozen=True)
class _HttpAnswer:
    """What the app's last HTTP answer was: its status and, for an error status, the failed method and the answer."""

    status: int
    method: str | None = None
    answer: str | None = None


@dataclass
class _CallAnswer:
    """The last HTTP answer to one call, which only its own requests write, so no call beside it on the session can."""

    answer: _HttpAnswer | None = None


_CALL_ANSWER: Final = contextvars.ContextVar[_CallAnswer | None]("call_answer", default=None)
"""The answer of the call whose request this is; the SDK sends each request in the context of the call that made it."""


class _BearerAuth(httpx2.Auth):
    """Adds the current bearer to each request and keeps the last HTTP answer the app gave, to the session and to the
    call the request belongs to.

    The SDK turns an HTTP error into a bare protocol error, so this is where the status is seen.
    """

    def __init__(self, token: Secret) -> None:
        self.token = token
        self.last_answer: _HttpAnswer | None = None

    @override
    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        request.headers["Authorization"] = f"Bearer {self.token.reveal()}"
        response = yield request
        if response.status_code < HTTPStatus.BAD_REQUEST:
            self.last_answer = _HttpAnswer(response.status_code)
        else:
            self.last_answer = _HttpAnswer(response.status_code, _method_of(request), await _answer_code_of(response))
        if (call := _CALL_ANSWER.get()) is not None:
            call.answer = self.last_answer


def _method_of(request: httpx2.Request) -> str | None:
    """The MCP method the request carried; none for a request that carries no JSON-RPC message, as a reconnect does."""
    try:
        message = jsonrpc_message_adapter.validate_json(request.content, by_name=False)
    except ValidationError:
        return None
    return getattr(message, "method", None)


async def _answer_code_of(response: httpx2.Response) -> str | None:
    """The code of the JSON-RPC error an error status carries, else the media type of the answer, never its content.

    The SDK takes an error body for a JSON-RPC error when the answer says it is JSON and the body parses as one.
    """
    media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type == "application/json":
        try:
            await response.aread()
            message = jsonrpc_message_adapter.validate_json(response.content, by_name=False)
        except httpx2.DecodingError, ValidationError:
            pass
        else:
            if isinstance(message, JSONRPCError):
                return jsonrpc_code(message.error.code)
    return media_type if _MEDIA_TYPE.fullmatch(media_type) else None


class _InvalidToolResult(Exception):  # noqa: N818
    """The app's result does not meet the output schema it advertised; carries none of its content."""


class _UnknownResource(Exception):  # noqa: N818
    """The app shows this member no such resource."""


_CALL_ERRORS: Final = (
    MCPError,
    SubscriptionLost,
    ValidationError,
    _InvalidToolResult,
    httpx2.HTTPError,
    OSError,
)
"""The errors of the SDK and its transport that tell a call failed; any other error is a defect."""


class _ValidatingSession(ClientSession):
    """Names the SDK's validation failure, which it raises as a bare ``RuntimeError``."""

    @override
    async def validate_tool_result(self, name: str, result: CallToolResult) -> None:
        try:
            await super().validate_tool_result(name, result)
        except RuntimeError:
            raise _InvalidToolResult from None


@dataclass
class _McpAppSession:
    session: ClientSession
    auth: _BearerAuth
    stream_seconds: float
    _honoured: bool = False
    """Whether the open stream was acknowledged with the tools notice honoured."""
    _list_counter: int = 0
    """Moves at each acknowledgement, each tools notice and each stream end; a listing that started before it moved is
    not kept."""
    _notices: int = 0
    _kept: tuple[ToolDeclaration, ...] | None = None
    _declared_arguments: dict[str, frozenset[str]] | None = None
    """The argument names each tool of the app's latest list declares; a list moves it with the tools notice."""
    _left_out: set[tuple[str, str]] = field(default_factory=set[tuple[str, str]])
    """The tool and argument of each left-out argument already reported in this session."""
    app_url: str = ""
    agent_id: str = ""

    def use_token(self, token: Secret) -> None:
        self.auth.token = token

    @asynccontextmanager
    async def changes(self) -> AsyncGenerator[AsyncIterator[object]]:
        self.auth.last_answer = None
        deadline = asyncio.timeout(self.stream_seconds)
        raised_in_body: Exception | None = None
        try:
            async with (
                deadline,
                listen(self.session, resource_subscriptions=[STANDING_URI], tools_list_changed=True) as subscription,
            ):
                if STANDING_URI not in (subscription.honored.resource_subscriptions or ()):
                    raise UnusableAnswer(LISTEN_NOT_HONORED_CODE)
                self._move_list_counter(honoured=bool(subscription.honored.tools_list_changed))
                try:
                    yield _wakes_of(subscription, self.auth, self._tools_notice)
                except Exception as error:
                    raised_in_body = error
                    raise
        except CallFailure:
            raise
        except _CALL_ERRORS as error:
            if error is raised_in_body:
                raise
            if isinstance(error, TimeoutError) and deadline.expired():
                raise Unreachable("the stream outlasted its deadline") from None
            raise _failure_of(error, self.auth.last_answer) from None
        finally:
            self._move_list_counter(honoured=False)

    def _move_list_counter(self, *, honoured: bool) -> None:
        self._honoured = honoured
        self._list_counter += 1
        self._kept = None
        self._declared_arguments = None

    def _tools_notice(self) -> None:
        self._notices += 1
        self._move_list_counter(honoured=self._honoured)

    def kept_tools(self) -> tuple[ToolDeclaration, ...] | None:
        return self._kept

    def tools_notices(self) -> int:
        return self._notices

    async def read_standing(self) -> MemberStanding:
        return _record_of(MemberStanding, await self._asked(self._read(STANDING_URI)))

    async def read_place(self, place: PlaceRef) -> PlaceWindow | None:
        return await self._read_record(PlaceWindow, place_window_uri(place))

    async def read_view(self, place: PlaceRef) -> PlaceView | None:
        return await self._read_record(PlaceView, place_view_uri(place))

    async def _read_record[Record: MemberResourceRecord](self, record: type[Record], uri: str) -> Record | None:
        """The record at the resource, or ``None`` when the app shows this member no such resource."""
        read = self._read(uri, unknown_is_absent=True)
        try:
            return _record_of(record, await self._asked(read))
        except _UnknownResource:
            return None

    async def declared_tools(self) -> tuple[ToolDeclaration, ...]:
        started = self._list_counter
        listing = await self._asked(self.session.list_tools())
        try:
            declared = tuple(_declaration_of(tool) for tool in listing.tools)
        except ValidationError:
            raise UnusableAnswer(INVALID_ANSWER_CODE) from None
        if self._list_counter == started:
            self._declared_arguments = {tool.name: _argument_names_of(tool.parameters) for tool in declared}
            if self._honoured:
                self._kept = declared
        return declared

    async def accept(self, accept: Accept) -> Refusal | None:
        arguments = AcceptInvitationArguments(
            place=accept.place,
            invitation=accept.invitation_id,
            epoch=accept.epoch,
            figure=accept.figure,
            voices=accept.voices or None,
        ).model_dump(mode="json")
        result = await self._called_declared(ACCEPT_INVITATION_TOOL, arguments, optional=_ACCEPT_OPTIONAL, lists=True)
        return _refusal_of(_text_of(result)) if result.is_error else None

    async def speak(
        self, place: PlaceRef, sentence: str, style: str | None = None, interrupted: str | None = None
    ) -> SpeakAnswer | SpeakRefusal:
        # The arguments are the wire shape of ``SpeakArguments``, built unvalidated: a refusal of a sentence is
        # the app's, and a validation error would carry the sentence.
        arguments: JsonObject = {"place": place.model_dump(mode="json"), "text": sentence}
        if style is not None:
            arguments["style"] = style
        if interrupted is not None:
            arguments["interrupted"] = interrupted
        result = await self._called_declared(SPEAK_TOOL, arguments, optional=_SPEAK_OPTIONAL)
        return _answer_of(result)

    async def continue_speaking(self, place: PlaceRef, interrupted: str) -> ContinueSpeakingRefusal | None:
        result = await self._called(CONTINUE_SPEAKING_TOOL, _naming_hold(place, interrupted))
        return _refusal_in(ContinueSpeakingRefused, result).refused if result.is_error else None

    async def stop_speaking(self, place: PlaceRef, interrupted: str) -> StopSpeakingRefusal | None:
        result = await self._called(STOP_SPEAKING_TOOL, _naming_hold(place, interrupted))
        return _refusal_in(StopSpeakingRefused, result).refused if result.is_error else None

    async def set_turn_state(self, statement: SetTurnStateArguments) -> SetTurnStateRefusal | None:
        result = await self._called_declared(
            SET_TURN_STATE_TOOL, statement.model_dump(mode="json"), optional=_TURN_STATE_OPTIONAL
        )
        return _refusal_in(SetTurnStateRefused, result).refused if result.is_error else None

    async def invoke(self, name: str, arguments: JsonObject) -> ToolOutcome:
        result = await self._called(name, arguments)
        return ToolOutcome(_text_of(result), result.is_error, _picture_of(result))

    async def _called(self, name: str, arguments: JsonObject) -> CallToolResult:
        result = await self._asked(self.session.call_tool(name, arguments, allow_input_required=True))
        if not isinstance(result, CallToolResult):
            raise UnusableAnswer(INVALID_ANSWER_CODE)
        return result

    async def _called_declared(
        self, name: str, arguments: JsonObject, *, optional: frozenset[str], lists: bool = False
    ) -> CallToolResult:
        """``_called`` without the ``optional`` arguments its tool's declared schema does not name.

        The schema is the app's latest list as ``declared_tools`` read it. A call that carries an optional argument
        and has none yet asks for the list only when it ``lists``; a turn's own statements do not, since the turn lists
        beside them. An argument left out is reported once per session.
        """
        if optional.isdisjoint(arguments):
            return await self._called(name, arguments)
        if self._declared_arguments is None and lists:
            await self.declared_tools()
        names = (self._declared_arguments or {}).get(name)
        if names is None:
            return await self._called(name, arguments)
        sent, left_out = _split_undeclared(arguments, names, optional)
        for argument in left_out:
            if (name, argument) not in self._left_out:
                self._left_out.add((name, argument))
                report_argument_left_out(self.agent_id, self.app_url, name, argument)
        return await self._called(name, sent)

    async def _read(self, uri: str, *, unknown_is_absent: bool = False) -> ReadResourceResult:
        """The resource; an app that shows this member no such one answers an invalid-params error."""
        try:
            result = await self.session.read_resource(uri, allow_input_required=True)
        except MCPError as error:
            if unknown_is_absent and error.code == INVALID_PARAMS:
                raise _UnknownResource from None
            raise
        if isinstance(result, InputRequiredResult):
            raise UnusableAnswer(INVALID_ANSWER_CODE)
        return result

    async def _asked[Answer](self, request: Awaitable[Answer]) -> Answer:
        """The call's answer; a failure is told by the call's own last HTTP answer, never by one to a call beside it."""
        self.auth.last_answer = None
        call = _CallAnswer()
        reset = _CALL_ANSWER.set(call)
        try:
            return await request
        except _CALL_ERRORS as error:
            raise _failure_of(error, call.answer) from None
        finally:
            _CALL_ANSWER.reset(reset)


async def _wakes_of(
    subscription: Subscription, auth: _BearerAuth, on_tools_notice: Callable[[], None]
) -> AsyncIterator[object]:
    try:
        async for wake in subscription:
            if isinstance(wake, ToolsListChanged):
                on_tools_notice()
            else:
                yield wake
    except SubscriptionLost as lost:
        raise _failure_of(lost, auth.last_answer) from None


def _record_of[Record: MemberResourceRecord](record: type[Record], result: ReadResourceResult) -> Record:
    text = "".join(block.text for block in result.contents if isinstance(block, TextResourceContents))
    try:
        return member_resource_from_json(record, text)
    except ValidationError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None


def _argument_names_of(schema: JsonObject) -> frozenset[str]:
    properties = schema.get("properties")
    return frozenset(properties) if isinstance(properties, dict) else frozenset()


def _split_undeclared(
    arguments: JsonObject, declared: frozenset[str], optional: frozenset[str]
) -> tuple[JsonObject, tuple[str, ...]]:
    """The arguments to send and the names left out: an ``optional`` argument the tool's ``declared`` names do not
    include. Every other argument stays, so a required one is never dropped."""
    left_out = tuple(name for name in arguments if name in optional and name not in declared)
    return {name: value for name, value in arguments.items() if name not in left_out}, left_out


def _declaration_of(tool: Tool) -> ToolDeclaration:
    read_only = tool.annotations is not None and bool(tool.annotations.read_only_hint)
    return ToolDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.input_schema,
        effect=ToolEffect.READ_ONLY if read_only else ToolEffect.SIDE_EFFECTING,
    )


def _text_of(result: CallToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


class _UnshownToolImage(ToolImage):
    """A tool image whose representation names no byte of the picture."""

    @override
    def __repr_args__(self) -> Iterable[tuple[str | None, object]]:
        return ()


def _picture_of(result: CallToolResult) -> ToolImage | None:
    """The one picture of the result, as the wire carries it; none when the result holds no image block.

    An image block is the wire's picture, so one that breaks the wire, a second one included, is an unusable answer.
    Audio and resource blocks are not read.
    """
    blocks = [block for block in result.content if isinstance(block, ImageContent)]
    if not blocks:
        return None
    if len(blocks) > 1 or blocks[0].mime_type != TOOL_IMAGE_MEDIA_TYPE:
        raise UnusableAnswer(INVALID_ANSWER_CODE)
    try:
        return _UnshownToolImage.model_validate({"content": blocks[0].data})
    except ValidationError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None


def _refusal_of(text: str) -> Refusal:
    try:
        return AcceptInvitationRefused.model_validate_json(text).refused
    except ValidationError:
        return UnrecognisedRefusal.UNRECOGNISED


def _answer_of(result: CallToolResult) -> SpeakAnswer | SpeakRefusal:
    text = _text_of(result)
    try:
        return SpeakRefused.model_validate_json(text).refused if result.is_error else _SPEAK_ANSWER.validate_json(text)
    except ValidationError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None


def _naming_hold(place: PlaceRef, interrupted: str) -> JsonObject:
    """The wire's arguments of ``continue_speaking`` and ``stop_speaking``, built unvalidated as ``speak``'s are."""
    return {"place": place.model_dump(mode="json"), "interrupted": interrupted}


def _refusal_in[Refused: BaseModel](refused: type[Refused], result: CallToolResult) -> Refused:
    try:
        return refused.model_validate_json(_text_of(result))
    except ValidationError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None


def _failure_of(error: BaseException, answered: _HttpAnswer | None) -> CallFailure:
    """A call that gave no usable answer: told by the last HTTP answer it was given, else by the error.

    A protocol error or a result the SDK cannot validate is an unusable answer; only a broken or
    silent connection is unreachable. The error is one of ``_CALL_ERRORS``: the status is read only for
    those, never for a late answer's error, which no call raised.
    """
    if (
        isinstance(error, MCPError)
        and error.code == UNSUPPORTED_PROTOCOL_VERSION
        and answered is not None
        and answered.status >= HTTPStatus.BAD_REQUEST
    ):
        code = jsonrpc_code(error.code)
        return UnusableAnswer(code, answered.method, code)
    if answered is not None and answered.status >= HTTPStatus.BAD_REQUEST:
        return StatusFailure(answered.status, answered.method, answered.answer)
    if answered is not None and answered.status >= HTTPStatus.MULTIPLE_CHOICES:
        return UnusableAnswer(status_code(answered.status))
    if isinstance(error, MCPError):
        if error.code in _TRANSPORT_ERROR_CODES:
            return Unreachable("the app did not answer")
        return UnusableAnswer(jsonrpc_code(error.code))
    if isinstance(error, SubscriptionLost):
        return Unreachable("the stream was lost")
    if isinstance(error, ValidationError | _InvalidToolResult):
        return UnusableAnswer(INVALID_ANSWER_CODE)
    return Unreachable("the app did not answer")


def _settled(leaf: BaseException, from_body: set[int], auth: _BearerAuth, *, beside_no_echo: bool) -> BaseException:
    """A failure or a leaf the caller's body raised stays as it was; an error of a call becomes its failure.

    The transport's task finds its read stream closed when an answer lands after the session ended: beside
    a leaf that is no such error that is a lost answer, unreachable whatever status it carried. Alone it is
    how a cancelled session unwinds, and it stays as it was so that no cancellation turns into a failure
    the loop backs off from; the session itself turns it into the cancellation where its task was cancelled.
    """
    if isinstance(leaf, CallFailure) or id(leaf) in from_body:
        return leaf
    if isinstance(leaf, _CALL_ERRORS):
        return _failure_of(leaf, auth.last_answer)
    if beside_no_echo and isinstance(leaf, LATE_ANSWER_ERRORS):
        return Unreachable("an answer landed after the session closed")
    return leaf


@asynccontextmanager
async def open_mcp_session(
    app: AppTarget, token: Secret, agent_id: str, *, stream_seconds: float = READ_SECONDS
) -> AsyncGenerator[AppSession]:
    """A session at the app's revision; the failures a call raises are the ones ``AppSession`` names.

    A bare ``ClientSession`` discovers the revision: the SDK's own client would fall back to
    ``initialize`` on any error, so a 401 on the probe would enter a path this runner does not have.
    Several leaves, failures or defects, leave as one ``ExceptionGroup`` holding each of them. A session whose
    task was being cancelled while it discovered the revision or when the body left, and that closes with late
    answers alone, raises the cancellation: the SDK's task groups lower the task's cancellation count and turn it
    into those answers.
    """
    auth = _BearerAuth(token)
    timeout = httpx2.Timeout(CONNECT_SECONDS, read=stream_seconds)
    raised_in_body: Exception | None = None
    cancelled_in_body = False
    try:
        async with (
            httpx2.AsyncClient(auth=auth, timeout=timeout, event_hooks={"request": [carry_trace]}) as client,
            streamable_http_client(app.mcp_url, http_client=client) as (read_stream, write_stream),
            _ValidatingSession(read_stream, write_stream) as session,
        ):
            try:
                await _discover(session, auth)
            except asyncio.CancelledError:
                cancelled_in_body = task_is_being_cancelled()
                raise
            try:
                yield _McpAppSession(session, auth, stream_seconds, app_url=app.mcp_url, agent_id=agent_id)
            except Exception as error:
                raised_in_body = error
                raise
            except asyncio.CancelledError:
                cancelled_in_body = task_is_being_cancelled()
                raise
    except Exception as error:
        # The SDK's task groups wrap what is raised inside them, so every leaf is looked at: a failure
        # of the call and what the caller's own code raised stay as they were, an error of the
        # connection becomes its failure and a defect stays a defect. The loop picks what decides.
        from_body = list(leaves_of(raised_in_body)) if raised_in_body else []
        found = list(leaves_of(error))
        # A close that failed or was cut can replace what the body raised; the body's leaves stay, ahead of what the
        # close met, so that of equal ranks a call's own failure tells how the session failed, not an answer it cut.
        found = [*from_body, *(leaf for leaf in found if all(leaf is not own for own in from_body))]
        from_body_ids = {id(own) for own in from_body}
        if cancelled_in_body and all(isinstance(leaf, LATE_ANSWER_ERRORS) for leaf in found):
            raise asyncio.CancelledError from None
        beside_no_echo = any(not isinstance(leaf, LATE_ANSWER_ERRORS) for leaf in found)
        leaves = [_settled(leaf, from_body_ids, auth, beside_no_echo=beside_no_echo) for leaf in found]
        if len(leaves) == 1:
            raise leaves[0] from None
        raise ExceptionGroup("the session failed", cast("list[Exception]", leaves)) from None


async def _discover(session: ClientSession, auth: _BearerAuth) -> None:
    auth.last_answer = None
    try:
        await session.discover()
    except RuntimeError:
        raise UnusableAnswer(UNSUPPORTED_VERSION_CODE) from None
    except _CALL_ERRORS as error:
        raise _failure_of(error, auth.last_answer) from None
