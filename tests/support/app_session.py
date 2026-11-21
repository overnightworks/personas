"""The arrangement the app session's tests share: a served fake app, its bearer and the session opened on them."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from agents.member import PlaceRef

from personas.app_session import open_mcp_session
from personas.config import AppTarget
from personas.secret import Secret
from tests.support.hopin import FakeApp
from tests.support.standings import AGENT_SUBJECT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def served(fake: FakeApp) -> AsyncIterator[FakeApp]:
    """``fake`` served, its ``url`` set, until the end."""
    async with fake.serving() as url:
        fake.url = url
        yield fake


@pytest.fixture
def place() -> PlaceRef:
    return PlaceRef(kind="call", id="room-7")


@pytest.fixture
async def app() -> AsyncIterator[FakeApp]:
    async with served(FakeApp()) as fake:
        yield fake


@pytest.fixture
def bearer(app: FakeApp) -> Secret:
    token = Secret(secrets.token_urlsafe(24))
    app.let_in(token.reveal(), AGENT_SUBJECT)
    return token


def opened(app: FakeApp, bearer: Secret, **keywords: Any):
    return open_mcp_session(AppTarget(app.url, "scope"), bearer, "agent-one", **keywords)
