"""The app boundary: one MCP session of the official SDK against an app's ``/mcp``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Final, Protocol, cast, override

import httpx2
from agent_runtime_client.member import (
    ACCEPT_INVITATION_TOOL,
    LISTEN_BOUND_SECONDS,
    STANDING_URI,
    AcceptInvitationArguments,
    MemberStanding,
)
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import Subscription, SubscriptionLost, listen
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CONNECTION_CLOSED,
    REQUEST_TIMEOUT,
    UNSUPPORTED_PROTOCOL_VERSION,
    CallToolResult,
    InputRequiredResult,
    TextContent,
    TextResourceContents,
)
from pydantic import ValidationError

from agents.config import AppTarget
from agents.decision import Accept, Refusal, status_code
from agents.failures import (
    INVALID_ANSWER_CODE,
    LISTEN_NOT_HONORED_CODE,
    UNSUPPORTED_VERSION_CODE,
    CallFailure,
    StatusFailure,
    Unreachable,
    UnusableAnswer,
)
from agents.secret import Secret

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
        except Exception as error:
            if error is raised_in_body:
                raise
            if isinstance(error, TimeoutError) and deadline.expired():
                raise Unreachable("the stream outlasted its deadline") from None
            raise _failure_or_raise(error, self.auth) from None

    async def read_standing(self) -> MemberStanding:
        result = await self._asked(self.session.read_resource(STANDING_URI, allow_input_required=True))
        if isinstance(result, InputRequiredResult):
            raise UnusableAnswer(INVALID_ANSWER_CODE)
        text = "".join(block.text for block in result.contents if isinstance(block, TextResourceContents))
        try:
            return MemberStanding.model_validate_json(text)
        except ValidationError:
            raise UnusableAnswer(INVALID_ANSWER_CODE) from None

    async def accept(self, accept: Accept) -> Refusal | None:
        arguments = AcceptInvitationArguments(
            place=accept.place, invitation=accept.invitation_id, epoch=accept.epoch
        ).model_dump(mode="json")
        result = await self._asked(self.session.call_tool(ACCEPT_INVITATION_TOOL, arguments, allow_input_required=True))
        if isinstance(result, InputRequiredResult):
            raise UnusableAnswer(INVALID_ANSWER_CODE)
        return _refusal_of(_text_of(result)) if result.is_error else None

    async def _asked[Answer](self, request: Awaitable[Answer]) -> Answer:
        self.auth.last_status = None
        try:
            return await request
        except Exception as error:
            raise _failure_or_raise(error, self.auth) from None


async def _wakes_of(subscription: Subscription, auth: _BearerAuth) -> AsyncIterator[object]:
    try:
        async for wake in subscription:
            yield wake
    except SubscriptionLost as lost:
        raise _failure_or_raise(lost, auth) from None


def _text_of(result: CallToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


def _refusal_of(text: str) -> Refusal:
    return next((code for code in Refusal if code is not Refusal.UNRECOGNISED and code in text), Refusal.UNRECOGNISED)


def _failure_of(error: BaseException, auth: _BearerAuth) -> CallFailure | None:
    """A call that gave no usable answer: told by the last HTTP status the app sent, else by the error.

    A protocol error or a result the SDK cannot validate is an unusable answer; only a broken or
    silent connection is unreachable. Any other error is not the app's and has no failure.
    """
    if isinstance(error, MCPError) and error.code == UNSUPPORTED_PROTOCOL_VERSION:
        return UnusableAnswer(f"jsonrpc_{error.code}")
    status = auth.last_status
    if status is not None and status >= 400:
        return StatusFailure(status)
    if status is not None and status >= 300:
        return UnusableAnswer(status_code(status))
    if isinstance(error, MCPError):
        if error.code in _TRANSPORT_ERROR_CODES:
            return Unreachable("the app did not answer")
        return UnusableAnswer(f"jsonrpc_{error.code}")
    if isinstance(error, SubscriptionLost):
        return Unreachable("the stream was lost")
    if isinstance(error, ValidationError | _InvalidToolResult):
        return UnusableAnswer(INVALID_ANSWER_CODE)
    if isinstance(error, httpx2.TransportError | OSError):
        return Unreachable("the app did not answer")
    return None


def _failure_or_raise(error: Exception, auth: _BearerAuth) -> CallFailure:
    failure = _failure_of(error, auth)
    if failure is None:
        raise error
    return failure


def _leaves(error: BaseException) -> Iterator[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        for inner in cast("BaseExceptionGroup[BaseException]", error).exceptions:
            yield from _leaves(inner)
    else:
        yield error


@asynccontextmanager
async def open_mcp_session(
    app: AppTarget, token: Secret, *, stream_seconds: float = READ_SECONDS
) -> AsyncGenerator[AppSession]:
    """A session at the app's revision; the failures a call raises are the ones ``AppSession`` names.

    A bare ``ClientSession`` discovers the revision: the SDK's own client would fall back to
    ``initialize`` on any error, so a 401 on the probe would enter a path this runner does not have.
    """
    auth = _BearerAuth(token)
    timeout = httpx2.Timeout(CONNECT_SECONDS, read=stream_seconds)
    raised_in_body: Exception | None = None
    try:
        async with (
            httpx2.AsyncClient(auth=auth, timeout=timeout) as client,
            streamable_http_client(app.mcp_url, http_client=client) as (read_stream, write_stream),
            _ValidatingSession(read_stream, write_stream) as session,
        ):
            await _discover(session, auth)
            try:
                yield _McpAppSession(session, auth, stream_seconds)
            except Exception as error:
                raised_in_body = error
                raise
    except Exception as error:
        # The SDK's task groups wrap what is raised inside them, so a failure of the call is looked
        # for in the group. What the caller's own code raised stays as it was; any other error is the
        # connection failing.
        leaves = list(_leaves(error))
        failure = next((leaf for leaf in leaves if isinstance(leaf, CallFailure)), None)
        if failure is not None:
            raise failure from None
        if raised_in_body is not None:
            raise raised_in_body from None
        for leaf in leaves:
            failure = _failure_of(leaf, auth)
            if failure is not None:
                raise failure from None
        raise


async def _discover(session: ClientSession, auth: _BearerAuth) -> None:
    auth.last_status = None
    try:
        await session.discover()
    except RuntimeError:
        raise UnusableAnswer(UNSUPPORTED_VERSION_CODE) from None
    except Exception as error:
        raise _failure_or_raise(error, auth) from None
