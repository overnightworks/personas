"""The SDK's transport whose request task finds its read stream closed when an answer lands late."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

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

    def transport(self, *, only_close: int = 1):
        """A stand-in for ``streamable_http_client`` whose ``only_close``-th close waits for ``release``.

        Closes count from one. ``closing`` is set when that close starts to wait, and ``cancelled`` when the task
        was cancelled while it waited.
        """
        real = app_session.streamable_http_client
        closes = 0
        hold = self

        @asynccontextmanager
        async def waiting(url, *, http_client):
            nonlocal closes
            async with real(url, http_client=http_client) as streams:
                try:
                    yield streams
                finally:
                    closes += 1
                    if closes == only_close:
                        hold.closing.set()
                        try:
                            await hold.release.wait()
                        except asyncio.CancelledError:
                            hold.cancelled.set()
                            raise

        return waiting


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
