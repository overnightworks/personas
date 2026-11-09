"""Stand-ins for the SDK's transport: late answers, chosen leaves, a waiting close, a failing child, a stray cancel."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import anyio

from agents import app_session


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


def transport_closing_with_leaves(*leaves: Exception):
    """A stand-in for ``streamable_http_client`` whose close raises exactly these leaves when its body was cancelled."""
    real = app_session.streamable_http_client

    @asynccontextmanager
    async def failing(url, *, http_client):
        async with real(url, http_client=http_client) as streams:
            try:
                yield streams
            except asyncio.CancelledError:
                raise ExceptionGroup("the transport", list(leaves)) from None

    return failing


class ClosingHold:
    """What a test sees of, and does to, one transport that waits while it closes."""

    def __init__(self) -> None:
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    def transport(self):
        """A stand-in for ``streamable_http_client`` whose first close waits for ``release``.

        ``closing`` is set when that close starts to wait, and ``cancelled`` when the task was cancelled while it
        waited.
        """
        real = app_session.streamable_http_client
        closes = 0

        @asynccontextmanager
        async def waiting(url, *, http_client):
            nonlocal closes
            async with real(url, http_client=http_client) as streams:
                try:
                    yield streams
                finally:
                    closes += 1
                    if closes == 1:
                        self.closing.set()
                        try:
                            await self.release.wait()
                        except asyncio.CancelledError:
                            self.cancelled.set()
                            raise

        return waiting


def transport_whose_second_session_loses_a_child(failing: asyncio.Event):
    """A stand-in for ``streamable_http_client`` whose second open runs a child that fails once ``failing`` is set.

    The child fails with a late-answer error, as the SDK's request task does, so anyio cancels the session's own
    task with no one else having done so. The first open is the lifecycle's, the second a turn's.
    """
    real = app_session.streamable_http_client
    opens = 0

    @asynccontextmanager
    async def losing(url, *, http_client):
        nonlocal opens
        opens += 1
        is_the_second = opens == 2
        async with real(url, http_client=http_client) as streams, anyio.create_task_group() as group:
            if is_the_second:

                async def child() -> None:
                    await failing.wait()
                    raise anyio.BrokenResourceError

                group.start_soon(child)
            yield streams

    return losing


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
