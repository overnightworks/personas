"""The realm's token endpoint, as far as the runner meets it: client credentials, granted or refused."""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

import httpx2

if TYPE_CHECKING:
    from agents.config import AgentCredentials
    from tests.support.hopin import FakeApp

TOKEN_ENDPOINT = "https://realm.test/protocol/openid-connect/token"
LIFETIME_SECONDS = 300


class FakeRealm:
    def __init__(self, app: FakeApp, agent: AgentCredentials, subject: str, scope: str) -> None:
        self._app = app
        self._agent = agent
        self._subject = subject
        self._scope = scope
        self.issued: list[str] = []
        self.refusing = False
        self.reaches_app = True
        self.refusals: asyncio.Queue[None] = asyncio.Queue()

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._answer)

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        form = {name: values[0] for name, values in parse_qs(request.content.decode()).items()}
        asked_as_agent = (
            form.get("grant_type") == "client_credentials"
            and form.get("client_id") == self._agent.client_id
            and form.get("client_secret") == self._agent.client_secret.reveal()
            and form.get("scope") == self._scope
        )
        if self.refusing or not asked_as_agent:
            self.refusals.put_nowait(None)
            return httpx2.Response(401, json={"error": "invalid_client"})
        token = secrets.token_urlsafe(24)
        self.issued.append(token)
        if self.reaches_app:
            self._app.let_in(token, self._subject)
        body = {"access_token": token, "expires_in": LIFETIME_SECONDS, "token_type": "Bearer"}
        return httpx2.Response(200, content=json.dumps(body))
