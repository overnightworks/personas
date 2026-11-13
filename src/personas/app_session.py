"""The app boundary: one MCP session of the official SDK against an app's ``/mcp``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Final, Protocol, cast, override

import httpx2
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    LISTEN_BOUND_SECONDS,
    SPEAK_TOOL,
    STANDING_URI,
    AcceptInvitationArguments,
    AcceptInvitationRefused,
    MemberStanding,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SpeakRefusal,
    SpeakRefused,
    Spoken,
    place_view_uri,
    place_window_uri,
)
from agents.tools import JsonObject, ToolDeclaration, ToolEffect, ToolOutcome
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import Subscription, SubscriptionLost, listen
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CONNECTION_CLOSED,
    INVALID_PARAMS,
    REQUEST_TIMEOUT,
    UNSUPPORTED_PROTOCOL_VERSION,
    CallToolResult,
    InputRequiredResult,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import BaseModel, ValidationError

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
from personas.secret import Secret

CONNECT_SECONDS: Final = 30.0
READ_SECONDS: Final = 2 * LISTEN_BOUND_SECONDS
"""Twice the bound an app puts on a stream: the runner cuts a stream that outlasts it."""
_TRANSPORT_ERROR_CODES: Final = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT})
"""What the SDK answers for a connection that broke or a request that timed out."""


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

    async def speak(self, place: PlaceRef, sentence: str) -> Spoken | SpeakRefusal:
        """How long the app says the sentence plays, or why it does not speak it; raises ``StatusFailure``,
        ``Unreachable`` or ``UnusableAnswer``, the last for an answer that is neither."""
        ...

    async def declared_tools(self) -> tuple[ToolDeclaration, ...]:
        """The first page of the app's tools; raises ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer``."""
        ...

    async def invoke(self, name: str, arguments: JsonObject) -> ToolOutcome:
        """The tool's outcome, an error one included; raises ``StatusFailure``, ``Unreachable``, ``UnusableAnswer``."""
        ...


type SessionOpener = Callable[[AppTarget, Secret], AbstractAsyncContextManager[AppSession]]


class _BearerAuth(httpx2.Auth):
    """Adds the current bearer to each request and keeps the last HTTP status the app answered.

    The SDK turns an HTTP error into a bare protocol error, so this is where the status is seen.
    """

    def __init__(self, token: Secret) -> None:
        self.token = token
        self.last_status: int | None = None

    @override
    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response]:
        request.headers["Authorization"] = f"Bearer {self.token.reveal()}"
        response = yield request
        self.last_status = response.status_code


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

    def use_token(self, token: Secret) -> None:
        self.auth.token = token

    @asynccontextmanager
    async def changes(self) -> AsyncGenerator[AsyncIterator[object]]:
        self.auth.last_status = None
        deadline = asyncio.timeout(self.stream_seconds)
        raised_in_body: Exception | None = None
        try:
            async with deadline, listen(self.session, resource_subscriptions=[STANDING_URI]) as subscription:
                if STANDING_URI not in (subscription.honored.resource_subscriptions or ()):
                    raise UnusableAnswer(LISTEN_NOT_HONORED_CODE)
                try:
                    yield _wakes_of(subscription, self.auth)
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
            raise _failure_of(error, self.auth) from None

    async def read_standing(self) -> MemberStanding:
        return _record_of(MemberStanding, await self._asked(self._read(STANDING_URI)))

    async def read_place(self, place: PlaceRef) -> PlaceWindow | None:
        return await self._read_record(PlaceWindow, place_window_uri(place))

    async def read_view(self, place: PlaceRef) -> PlaceView | None:
        return await self._read_record(PlaceView, place_view_uri(place))

    async def _read_record[Record: BaseModel](self, record: type[Record], uri: str) -> Record | None:
        """The record at the resource, or ``None`` when the app shows this member no such resource."""
        read = self._read(uri, unknown_is_absent=True)
        try:
            return _record_of(record, await self._asked(read))
        except _UnknownResource:
            return None

    async def declared_tools(self) -> tuple[ToolDeclaration, ...]:
        listing = await self._asked(self.session.list_tools())
        try:
            return tuple(_declaration_of(tool) for tool in listing.tools)
        except ValidationError:
            raise UnusableAnswer(INVALID_ANSWER_CODE) from None

    async def accept(self, accept: Accept) -> Refusal | None:
        arguments = AcceptInvitationArguments(
            place=accept.place,
            invitation=accept.invitation_id,
            epoch=accept.epoch,
            figure=accept.figure,
            voice=accept.voice,
        ).model_dump(mode="json")
        result = await self._called(ACCEPT_INVITATION_TOOL, arguments)
        return _refusal_of(_text_of(result)) if result.is_error else None

    async def speak(self, place: PlaceRef, sentence: str) -> Spoken | SpeakRefusal:
        # The arguments are the wire shape of ``SpeakArguments``, built unvalidated: a refusal of a sentence is
        # the app's, and a validation error would carry the sentence.
        result = await self._called(SPEAK_TOOL, {"place": place.model_dump(mode="json"), "text": sentence})
        return _spoken_of(result)

    async def invoke(self, name: str, arguments: JsonObject) -> ToolOutcome:
        result = await self._called(name, arguments)
        return ToolOutcome(_text_of(result), result.is_error)

    async def _called(self, name: str, arguments: JsonObject) -> CallToolResult:
        result = await self._asked(self.session.call_tool(name, arguments, allow_input_required=True))
        if not isinstance(result, CallToolResult):
            raise UnusableAnswer(INVALID_ANSWER_CODE)
        return result

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
        self.auth.last_status = None
        try:
            return await request
        except _CALL_ERRORS as error:
            raise _failure_of(error, self.auth) from None


async def _wakes_of(subscription: Subscription, auth: _BearerAuth) -> AsyncIterator[object]:
    try:
        async for wake in subscription:
            yield wake
    except SubscriptionLost as lost:
        raise _failure_of(lost, auth) from None


def _record_of[Record: BaseModel](record: type[Record], result: ReadResourceResult) -> Record:
    text = "".join(block.text for block in result.contents if isinstance(block, TextResourceContents))
    try:
        return record.model_validate_json(text)
    except ValidationError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None


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


def _refusal_of(text: str) -> Refusal:
    try:
        return AcceptInvitationRefused.model_validate_json(text).refused
    except ValidationError:
        return UnrecognisedRefusal.UNRECOGNISED


def _spoken_of(result: CallToolResult) -> Spoken | SpeakRefusal:
    text = _text_of(result)
    try:
        return SpeakRefused.model_validate_json(text).refused if result.is_error else Spoken.model_validate_json(text)
    except ValidationError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None


def _failure_of(error: BaseException, auth: _BearerAuth) -> CallFailure:
    """A call that gave no usable answer: told by the last HTTP status the app sent, else by the error.

    A protocol error or a result the SDK cannot validate is an unusable answer; only a broken or
    silent connection is unreachable. The error is one of ``_CALL_ERRORS``: the status is read only for
    those, never for a late answer's error, which no call raised.
    """
    if isinstance(error, MCPError) and error.code == UNSUPPORTED_PROTOCOL_VERSION:
        return UnusableAnswer(jsonrpc_code(error.code))
    status = auth.last_status
    if status is not None and status >= 400:
        return StatusFailure(status)
    if status is not None and status >= 300:
        return UnusableAnswer(status_code(status))
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
        return _failure_of(leaf, auth)
    if beside_no_echo and isinstance(leaf, LATE_ANSWER_ERRORS):
        return Unreachable("an answer landed after the session closed")
    return leaf


@asynccontextmanager
async def open_mcp_session(
    app: AppTarget, token: Secret, *, stream_seconds: float = READ_SECONDS
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
            httpx2.AsyncClient(auth=auth, timeout=timeout) as client,
            streamable_http_client(app.mcp_url, http_client=client) as (read_stream, write_stream),
            _ValidatingSession(read_stream, write_stream) as session,
        ):
            try:
                await _discover(session, auth)
            except asyncio.CancelledError:
                cancelled_in_body = task_is_being_cancelled()
                raise
            try:
                yield _McpAppSession(session, auth, stream_seconds)
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
        # A close that failed or was cut can replace what the body raised; the body's leaves stay.
        found = [*(own for own in from_body if all(own is not leaf for leaf in found)), *found]
        from_body_ids = {id(own) for own in from_body}
        if cancelled_in_body and all(isinstance(leaf, LATE_ANSWER_ERRORS) for leaf in found):
            raise asyncio.CancelledError from None
        beside_no_echo = any(not isinstance(leaf, LATE_ANSWER_ERRORS) for leaf in found)
        leaves = [_settled(leaf, from_body_ids, auth, beside_no_echo=beside_no_echo) for leaf in found]
        if len(leaves) == 1:
            raise leaves[0] from None
        raise ExceptionGroup("the session failed", cast("list[Exception]", leaves)) from None


async def _discover(session: ClientSession, auth: _BearerAuth) -> None:
    auth.last_status = None
    try:
        await session.discover()
    except RuntimeError:
        raise UnusableAnswer(UNSUPPORTED_VERSION_CODE) from None
    except _CALL_ERRORS as error:
        raise _failure_of(error, auth) from None
