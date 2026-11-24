"""Session openers of real sessions that fail, wait or are lost at the points a scenario names.

Each wraps the real ``open_app_session`` (or the ``inner`` opener it is given) and raises at the same point what
agents' fold of a failed close gives, or runs an anyio group so that the real client meets a real cancellation.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final, cast

import anyio
from agents.mcp_client.exchange import BearerAuth
from agents.mcp_client.fold import folded
from agents.member_outcomes import Unreachable
from pydantic import SecretStr

from personas.app_session import open_app_session
from personas.failures import leaves_of, task_is_being_cancelled

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from agents.mcp_client import MemberSession

    from personas.app_session import SessionOpener
    from personas.config import AppTarget
    from personas.secret import Secret


_NO_ANSWER_YET: Final = BearerAuth(lambda: SecretStr(""))
"""The auth of a session that was given no HTTP answer, which ``folded`` reads the last answer of."""


def _of_app(of_url: str | None, app: AppTarget) -> bool:
    return of_url in (None, str(app.mcp_url))


def _the_nth_session(nth: int, of_url: str | None) -> Callable[[AppTarget], bool]:
    """A test that, called once per session, counts those at ``of_url`` (all without it) and says whether this is the
    ``nth``."""
    counted = 0

    def is_nth(app: AppTarget) -> bool:
        nonlocal counted
        if not _of_app(of_url, app):
            return False
        counted += 1
        return counted == nth

    return is_nth


def _folded_close(
    leaves: list[Exception], raised_in_body: Exception | None, *, cancelled_in_body: bool = False
) -> Exception | asyncio.CancelledError:
    """What the client raises of a session that ended with these leaves."""
    return folded(
        ExceptionGroup("the close", leaves), raised_in_body, _NO_ANSWER_YET, cancelled_in_body=cancelled_in_body
    )


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

    def opener(
        self, *, nth: int = 1, of_url: str | None = None, inner: SessionOpener = open_app_session
    ) -> SessionOpener:
        """A session opener whose ``nth`` session leaves only once ``release`` is set, the real session at that app.

        ``closing`` is set when that session starts to leave and ``cancelled`` when the task was cancelled while it
        waited. With ``of_url`` only the sessions at that URL are counted; ``inner`` opens the session that leaves.
        """
        is_nth = _the_nth_session(nth, of_url)

        @asynccontextmanager
        async def waiting(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
            try:
                async with inner(app, bearer, agent_id) as session:
                    yield session
            finally:
                if is_nth(app):
                    await self._held()

        return waiting


type _Around = Callable[[Callable[[], Awaitable[object]]], Awaitable[object]]


class _AroundSession:
    """The real session at one app, whose named requests run inside their ``around``, which gets the real request to
    call."""

    def __init__(self, session: MemberSession, arounds: dict[str, _Around]) -> None:
        self._session = session
        self._arounds = arounds

    def __getattr__(self, name: str) -> object:
        real = getattr(self._session, name)
        around = self._arounds.get(name)
        if around is None:
            return real

        async def wrapped(*arguments: object, **named: object) -> object:
            return await around(lambda: real(*arguments, **named))

        return wrapped


def _opener_around(request: str, around: _Around, *, of_url: str | None) -> SessionOpener:
    return _opener_arounds({request: around}, of_url=of_url)


def _opener_arounds(arounds: dict[str, _Around], *, of_url: str | None) -> SessionOpener:
    @asynccontextmanager
    async def opened(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
        async with open_app_session(app, bearer, agent_id) as session:
            if _of_app(of_url, app):
                yield cast("MemberSession", _AroundSession(session, arounds))
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
    """A session opener of real sessions whose ``request``, a ``MemberSession`` method, raises ``defect``.

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


def opener_whose_close_adds_a_late_answer(
    *, only_close: int | None = None, inner: SessionOpener = open_app_session
) -> SessionOpener:
    """A session opener whose session, once it failed, leaves with a late answer beside the failure.

    A session that was cancelled leaves as the cancellation, since late answers alone do not replace it.
    ``only_close`` counts the sessions that left with an error or a cancellation from one and limits the late answer
    to that one; every failed session gets it without.
    """
    return _opener_whose_close_adds(
        anyio.BrokenResourceError, replaces_cancellation=False, only_close=only_close, inner=inner
    )


def opener_whose_close_adds_a_defect(
    defect: Callable[[], Exception], *, only_close: int | None = None, inner: SessionOpener = open_app_session
) -> SessionOpener:
    """A session opener whose session leaves with the defect ``defect`` makes beside the failure that ended it.

    A session that was cancelled leaves with the defect alone, which replaces the cancellation. ``only_close``
    counts the sessions that left with an error or a cancellation from one and limits the defect to that one.
    """
    return _opener_whose_close_adds(defect, replaces_cancellation=True, only_close=only_close, inner=inner)


def _opener_whose_close_adds(
    added: Callable[[], Exception], *, replaces_cancellation: bool, only_close: int | None, inner: SessionOpener
) -> SessionOpener:
    closes = 0

    @asynccontextmanager
    async def adding(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
        nonlocal closes
        try:
            async with inner(app, bearer, agent_id) as session:
                yield session
        except Exception as error:
            closes += 1
            if only_close not in (None, closes):
                raise
            raise _folded_close([*leaves_of(error), added()], error) from None
        except asyncio.CancelledError:
            closes += 1
            if not replaces_cancellation or only_close not in (None, closes):
                raise
            raise _folded_close([added()], None) from None

    return adding


def opener_whose_cancelled_close_fails(*, only_close: int | None = None) -> SessionOpener:
    """A session opener whose session leaves as ``Unreachable`` where its body was cancelled and its close failed.

    ``only_close`` counts the cancelled sessions from one and limits the failure to that one.
    """
    closes = 0

    @asynccontextmanager
    async def failing(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
        nonlocal closes
        try:
            async with open_app_session(app, bearer, agent_id) as session:
                yield session
        except asyncio.CancelledError:
            closes += 1
            if only_close not in (None, closes):
                raise
            raise Unreachable("the close failed") from None

    return failing


def opener_whose_close_loses_a_child(
    *, nth: int = 1, of_url: str | None = None, inner: SessionOpener = open_app_session
) -> SessionOpener:
    """A session opener whose ``nth`` session leaves as ``Unreachable`` beside what its body raised, as the client's
    fold gives a close that loses a child of its task group.

    With ``of_url`` only the sessions at that URL are counted.
    """
    is_nth = _the_nth_session(nth, of_url)

    @asynccontextmanager
    async def losing(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
        if not is_nth(app):
            async with inner(app, bearer, agent_id) as session:
                yield session
            return
        try:
            async with inner(app, bearer, agent_id) as session:
                yield session
        except Exception as error:
            raise ExceptionGroup(
                "the session failed", [*leaves_of(error), Unreachable("the close lost a child")]
            ) from None
        raise Unreachable("the close lost a child")

    return losing


def opener_whose_session_loses_a_child(
    failing: asyncio.Event,
    *,
    nth: int = 2,
    of_url: str | None = None,
    error: Callable[[], Exception] = anyio.BrokenResourceError,
    inner: SessionOpener = open_app_session,
) -> SessionOpener:
    """A session opener whose ``nth`` session runs beside a child that fails with ``error`` once ``failing`` is set.

    The child fails as the SDK's request task does, so anyio cancels the session's own task with no one else having
    done so, and the real client meets that cancellation; the session then leaves as the fold gives it. With ``of_url``
    only the sessions at that URL are counted, so a scenario names the app's own open and not its place among the
    others.
    """
    is_nth = _the_nth_session(nth, of_url)

    @asynccontextmanager
    async def losing(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
        if not is_nth(app):
            async with inner(app, bearer, agent_id) as session:
                yield session
            return

        async def child() -> None:
            await failing.wait()
            raise error()

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(child)
                async with inner(app, bearer, agent_id) as session:
                    yield session
        except ExceptionGroup as lost:
            # The scenario's own group cancelling the body is the one choice kept; the client's fold gives the rest.
            if not task_is_being_cancelled():
                raise asyncio.CancelledError from None
            raise _folded_close(list(leaves_of(lost)), None, cancelled_in_body=True) from None

    return losing


def opener_cancelled_once() -> SessionOpener:
    """A session opener whose first open is cancelled by nobody."""
    opens = 0

    @asynccontextmanager
    async def cancelling(app: AppTarget, bearer: Callable[[], Secret], agent_id: str) -> AsyncGenerator[MemberSession]:
        nonlocal opens
        opens += 1
        if opens == 1:
            raise asyncio.CancelledError
        async with open_app_session(app, bearer, agent_id) as session:
            yield session

    return cancelling
