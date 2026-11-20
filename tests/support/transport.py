"""Stand-ins for the SDK's transport: late answers, chosen leaves, a waiting close, a failing child, a stray cancel, and
sessions whose requests raise a defect."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import anyio

from personas import app_session
from personas.app_session import open_mcp_session

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from contextlib import AbstractAsyncContextManager

    from personas.app_session import AppSession, SessionOpener
    from personas.config import AppTarget
    from personas.secret import Secret


def transport_closing_with_late_answers(*, echo: type[Exception], echoes: int = 1, only_close: int | None = None):
    """A stand-in for ``streamable_http_client`` whose close raises ``echoes`` late-answer errors.

    The echoes leave beside the error that ended the session, or alone when the session was cancelled.
    ``only_close`` counts closes from one and limits the failure to that one; every close fails without it.
    """
    real = app_session.streamable_http_client
    closes = 0

    @asynccontextmanager
    async def failing(url, *, http_client):
        nonlocal closes
        async with real(url, http_client=http_client) as streams:
            try:
                yield streams
            except Exception as error:
                closes += 1
                if only_close not in (None, closes):
                    raise
                raise ExceptionGroup("the transport", [error, *(echo() for _ in range(echoes))]) from None
            except asyncio.CancelledError:
                closes += 1
                if only_close not in (None, closes):
                    raise
                raise ExceptionGroup("the transport", [echo() for _ in range(echoes)]) from None

    return failing


def transport_closing_with_leaves(*leaves: Exception, only_close: int | None = None):
    """A stand-in for ``streamable_http_client`` whose close raises exactly these leaves when its body was cancelled.

    ``only_close`` counts the closes of a cancelled body from one and limits the failure to that one.
    """
    real = app_session.streamable_http_client
    closes = 0

    @asynccontextmanager
    async def failing(url, *, http_client):
        nonlocal closes
        async with real(url, http_client=http_client) as streams:
            try:
                yield streams
            except asyncio.CancelledError:
                closes += 1
                if only_close not in (None, closes):
                    raise
                raise ExceptionGroup("the transport", list(leaves)) from None

    return failing


class ClosingHold:
    """What a test sees of, and does to, one transport that waits while it closes."""

    def __init__(self) -> None:
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _held(self) -> None:
        """Tells ``closing`` and waits for ``release``; a cancel while it waits tells ``cancelled``."""
        self.closing.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def transport(self, *, nth: int = 1, of_url: str | None = None):
        """A stand-in for ``streamable_http_client`` whose ``nth`` close waits for ``release``.

        ``closing`` is set when that close starts to wait, and ``cancelled`` when the task was cancelled while it
        waited. With ``of_url`` only the closes of that URL are counted.
        """
        real = app_session.streamable_http_client
        closes = 0

        @asynccontextmanager
        async def waiting(url, *, http_client):
            nonlocal closes
            counted = of_url in (None, str(url))
            async with real(url, http_client=http_client) as streams:
                try:
                    yield streams
                finally:
                    closes += counted
                    if counted and closes == nth:
                        await self._held()

        return waiting

    def opener_holding_cancelled(self, request: str) -> SessionOpener:
        """A session opener of real sessions whose first ``request`` waits, telling ``closing``, and when cancelled
        tells ``cancelled`` and unwinds only once ``release`` is set."""

        async def around(real: Callable[[], Awaitable[object]]) -> object:
            if self.closing.is_set():
                return await real()
            self.closing.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
                raise
            return await real()

        return _opener_around(request, around, of_url=None)

    def opener(self, *, nth: int = 1, of_url: str | None = None) -> SessionOpener:
        """A session opener whose ``nth`` session leaves only once ``release`` is set, the real session at that app.

        ``closing`` is set when that session starts to leave and ``cancelled`` when the task was cancelled while it
        waited. With ``of_url`` only the sessions at that URL are counted.
        """
        leaves = 0

        @asynccontextmanager
        async def waiting(app: AppTarget, token: Secret) -> AsyncGenerator[AppSession]:
            nonlocal leaves
            try:
                async with open_mcp_session(app, token) as session:
                    yield session
            finally:
                if of_url in (None, str(app.mcp_url)):
                    leaves += 1
                    if leaves == nth:
                        await self._held()

        return waiting


type _Around = Callable[[Callable[[], Awaitable[object]]], Awaitable[object]]


class _AroundSession:
    """The real session at one app, whose named requests run inside their ``around``, which gets the real request to
    call."""

    def __init__(self, session: AppSession, arounds: dict[str, _Around]) -> None:
        self._session = session
        self._arounds = arounds

    def __getattr__(self, name: str) -> object:
        real = getattr(self._session, name)
        around = self._arounds.get(name)
        if around is None:
            return real

        async def wrapped(*arguments: object) -> object:
            return await around(lambda: real(*arguments))

        return wrapped


def _opener_around(request: str, around: _Around, *, of_url: str | None) -> SessionOpener:
    return _opener_arounds({request: around}, of_url=of_url)


def _opener_arounds(arounds: dict[str, _Around], *, of_url: str | None) -> SessionOpener:
    @asynccontextmanager
    async def opened(app: AppTarget, token: Secret) -> AsyncGenerator[AppSession]:
        async with open_mcp_session(app, token) as session:
            if of_url in (None, str(app.mcp_url)):
                yield cast("AppSession", _AroundSession(session, arounds))
            else:
                yield session

    return opened


def opener_whose_request_raises(
    request: str,
    defect: Exception,
    *,
    of_url: str | None = None,
    once: bool = False,
    after: int = 0,
    arrived: asyncio.Event | None = None,
    when_cancelled: bool = False,
) -> SessionOpener:
    """A session opener of real sessions whose ``request``, an ``AppSession`` method, raises ``defect``.

    With ``of_url`` only the sessions at that URL raise, and with ``once`` only the first such request does. The
    first ``after`` requests are real. With ``when_cancelled`` the request waits, tells ``arrived`` and raises
    ``defect`` in place of the cancellation that reaches it.
    """
    raised = False
    made = 0

    async def around(real: Callable[[], Awaitable[object]]) -> object:
        nonlocal raised, made
        made += 1
        if made <= after or (once and raised):
            return await real()
        raised = True
        if not when_cancelled:
            raise defect
        if arrived is not None:
            arrived.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise defect from None
        return None

    return _opener_around(request, around, of_url=of_url)


def opener_whose_requests_raise_together(defects: dict[str, tuple[Exception, int]]) -> SessionOpener:
    """A session opener of real sessions whose named requests each raise their defect once all of them are in flight.

    The requests raise in the order of ``defects``, each after as many turns of the loop as its number says since the
    one before it raised, so that a scenario can let one failure leave its task group while the other is still on its
    way, however late each call arrives.
    """
    in_flight = {request: asyncio.Event() for request in defects}
    raised = {request: asyncio.Event() for request in defects}
    before = dict(zip(defects, [None, *defects], strict=False))

    def raising(request: str, defect: Exception, loop_turns: int) -> _Around:
        async def around(_real: Callable[[], Awaitable[object]]) -> object:
            in_flight[request].set()
            await asyncio.gather(*(each.wait() for each in in_flight.values()))
            if (earlier := before[request]) is not None:
                await raised[earlier].wait()
            for _ in range(loop_turns):
                await asyncio.sleep(0)
            raised[request].set()
            raise defect

        return around

    return _opener_arounds({request: raising(request, *each) for request, each in defects.items()}, of_url=None)


def transport_whose_session_loses_a_child(
    failing: asyncio.Event,
    *,
    nth: int = 2,
    of_url: str | None = None,
    error: Callable[[], Exception] = anyio.BrokenResourceError,
):
    """A stand-in for ``streamable_http_client`` whose ``nth`` open runs a child that fails once ``failing`` is set.

    The child fails with ``error``, a late-answer error as the SDK's request task does unless a scenario chooses
    another, so anyio cancels the session's own task with no one else having done so. The first open is the
    lifecycle's, the second a turn's. With ``of_url`` only the opens of that URL are counted, so a scenario names
    the app's own open and not its place among the others.
    """
    real = app_session.streamable_http_client
    opens = 0

    @asynccontextmanager
    async def losing(url, *, http_client):
        nonlocal opens
        counted = of_url in (None, str(url))
        opens += counted
        is_the_one = counted and opens == nth
        async with real(url, http_client=http_client) as streams, anyio.create_task_group() as group:
            if is_the_one:

                async def child() -> None:
                    await failing.wait()
                    raise error()

                group.start_soon(child)
            yield streams

    return losing


def transport_whose_close_loses_a_child(error: Callable[[], Exception], *, nth: int = 1, of_url: str | None = None):
    """A stand-in for ``streamable_http_client`` whose ``nth`` close awaits inside an anyio group, as the SDK's
    close does, while a child of that group fails with ``error``.

    With ``of_url`` only the closes of that URL are counted.
    """
    real = app_session.streamable_http_client
    opens = 0

    @asynccontextmanager
    async def losing(url, *, http_client):
        nonlocal opens
        counted = of_url in (None, str(url))
        opens += counted
        is_the_one = counted and opens == nth
        async with real(url, http_client=http_client) as streams, anyio.create_task_group() as group:
            try:
                yield streams
            finally:
                if is_the_one:

                    async def child() -> None:
                        await asyncio.sleep(0)
                        raise error()

                    group.start_soon(child)
                    await asyncio.Event().wait()

    return losing


def transport_by_url(stand_ins: dict[str, Callable[..., AbstractAsyncContextManager[Any]]]):
    """A stand-in for ``streamable_http_client`` that gives each URL its own stand-in, every other URL the real one."""
    real = app_session.streamable_http_client

    def pick(url, *, http_client):
        return stand_ins.get(str(url), real)(url, http_client=http_client)

    return pick


def transport_cancelled_once():
    """A stand-in for ``streamable_http_client`` whose first open is cancelled by nobody."""
    real = app_session.streamable_http_client
    opens = 0

    @asynccontextmanager
    async def cancelling(url, *, http_client):
        nonlocal opens
        opens += 1
        if opens == 1:
            raise asyncio.CancelledError
        async with real(url, http_client=http_client) as streams:
            yield streams

    return cancelling
