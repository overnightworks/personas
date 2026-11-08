"""The app boundary: one MCP session of the official SDK against an app's ``/mcp``."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Final, Protocol, cast, override

import httpx2
from agent_runtime_client.wait import STANDINGS_READ_SECONDS, WAIT_BOUND_SECONDS, WaitAnswer
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT, CallToolResult, TextContent
from pydantic import ValidationError

from agents.config import AppTarget
from agents.decision import Accept, Refusal, status_code
from agents.failures import CallFailure, StatusFailure, Unreachable, UnusableAnswer, WaitToolError
from agents.secret import Secret

CONNECT_SECONDS: Final = 30.0
READ_SECONDS: Final = 2 * (WAIT_BOUND_SECONDS + STANDINGS_READ_SECONDS)
"""Well above the longest a held ``wait`` takes, so the client never cuts one short."""
WAIT_TOOL: Final = "wait"
ACCEPT_TOOL: Final = "accept_invitation"
INVALID_ANSWER_CODE: Final = "invalid_answer"
_TRANSPORT_ERROR_CODES: Final = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT})
"""What the SDK answers for a connection that broke or a request that timed out."""


class AppSession(Protocol):
    def use_token(self, token: Secret) -> None:
        """Every later call carries this bearer."""
        ...

    async def wait(self, after: str | None) -> WaitAnswer:
        """Raises ``StatusFailure``, ``Unreachable``, ``UnusableAnswer`` or ``WaitToolError``."""
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


@dataclass
class _McpAppSession:
    session: ClientSession
    auth: _BearerAuth

    def use_token(self, token: Secret) -> None:
        self.auth.token = token

    async def wait(self, after: str | None) -> WaitAnswer:
        result = await self._call(WAIT_TOOL, {} if after is None else {"after": after})
        if result.is_error:
            raise WaitToolError("the wait tool answered an error")
        try:
            return WaitAnswer.model_validate_json(_text_of(result))
        except ValidationError:
            raise UnusableAnswer(INVALID_ANSWER_CODE) from None

    async def accept(self, accept: Accept) -> Refusal | None:
        arguments = {"room": accept.place_id, "invitation": accept.invitation_id, "epoch": accept.epoch}
        result = await self._call(ACCEPT_TOOL, arguments)
        return _refusal_of(_text_of(result)) if result.is_error else None

    async def _call(self, name: str, arguments: dict[str, str | int]) -> CallToolResult:
        self.auth.last_status = None
        try:
            return await self.session.call_tool(name, arguments)
        except Exception as error:
            raise _failure_or_raise(error, self.auth) from None


def _text_of(result: CallToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


def _refusal_of(text: str) -> Refusal:
    return next((code for code in Refusal if code is not Refusal.UNRECOGNISED and code in text), Refusal.UNRECOGNISED)


def _failure_of(error: BaseException, auth: _BearerAuth) -> CallFailure | None:
    """A call that gave no usable answer: told by the last HTTP status the app sent, else by the error.

    A protocol error or a result the SDK cannot validate is an unusable answer; only a broken or
    silent connection is unreachable. Any other error is not the app's and has no failure.
    """
    status = auth.last_status
    if status is not None and status >= 400:
        return StatusFailure(status)
    if status is not None and status >= 300:
        return UnusableAnswer(status_code(status))
    if isinstance(error, MCPError):
        if error.code in _TRANSPORT_ERROR_CODES:
            return Unreachable("the app did not answer")
        return UnusableAnswer(f"jsonrpc_{error.code}")
    if isinstance(error, ValidationError):
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
    app: AppTarget, token: Secret, *, transport: httpx2.AsyncBaseTransport | None = None
) -> AsyncGenerator[AppSession]:
    """An initialised session; the failures a call raises are the ones ``AppSession`` names.

    ``transport`` serves an app in the same process; without it the network is used.
    """
    auth = _BearerAuth(token)
    timeout = httpx2.Timeout(CONNECT_SECONDS, read=READ_SECONDS)
    raised_in_body: Exception | None = None
    try:
        async with (
            httpx2.AsyncClient(auth=auth, transport=transport, timeout=timeout) as client,
            streamable_http_client(app.mcp_url, http_client=client) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await _initialise(session, auth)
            try:
                yield _McpAppSession(session, auth)
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


async def _initialise(session: ClientSession, auth: _BearerAuth) -> None:
    try:
        await session.initialize()
    except Exception as error:
        raise _failure_or_raise(error, auth) from None
