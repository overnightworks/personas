"""Stand-ins for the SDK's transport that ``test_app_session`` drives the runner's own client with: late answers,
chosen leaves and a close that loses a child."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio

from personas import app_session

if TYPE_CHECKING:
    from collections.abc import Callable


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
