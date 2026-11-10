"""The realm's token endpoint, as far as the runner meets it: client credentials, granted or refused."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

import httpx2

from tests.support.hopin import ISSUER

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from personas.config import AgentCredentials
    from tests.support.hopin import FakeApp

TOKEN_ENDPOINT = "https://realm.test/protocol/openid-connect/token"
LIFETIME_SECONDS = 300
API_GROUP = "agents-api"
AGENT_NAME = "Mozart"


def _segment(document: Mapping[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(document).encode()).rstrip(b"=").decode()


def jwt_of(claims: Mapping[str, object]) -> str:
    """A token shaped like the realm's: a header, these claims and a signature nobody can check."""
    return ".".join([_segment({"alg": "RS256", "typ": "JWT"}), _segment(claims), secrets.token_urlsafe(24)])


class FakeRealm:
    def __init__(
        self,
        app: FakeApp,
        agent: AgentCredentials,
        subject: str,
        scope: str,
        *,
        groups: Sequence[str] = (API_GROUP,),
        agent_name: str = AGENT_NAME,
    ) -> None:
        self._apps = {scope: app}
        self._clients = {agent.client_id: (agent, subject)}
        self._names: dict[str, str] = {}
        self.groups = list(groups)
        self.agent_name = agent_name
        self.issued: list[str] = []
        self.issued_for: dict[str, list[str]] = {scope: []}
        self.refusing = False
        self.refused_scopes: set[str] = set()
        self.reaches_app = True
        self.refusals: asyncio.Queue[None] = asyncio.Queue()

    def add_app(self, scope: str, app: FakeApp) -> None:
        """Lets the agents ask for a token for one more scope, which lets them into this app alone."""
        self._apps[scope] = app
        self.issued_for[scope] = []

    def add_client(self, agent: AgentCredentials, subject: str, name: str | None = None) -> None:
        """Lets one more agent ask for tokens, each with a subject of its own and, if given, a name of its own."""
        self._clients[agent.client_id] = (agent, subject)
        if name is not None:
            self._names[agent.client_id] = name

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._answer)

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        form = {name: values[0] for name, values in parse_qs(request.content.decode()).items()}
        agent, subject = self._clients.get(form.get("client_id", ""), (None, ""))
        asked_as_agent = (
            agent is not None
            and form.get("grant_type") == "client_credentials"
            and form.get("client_secret") == agent.client_secret.reveal()
            and form.get("scope") in self._apps
        )
        if self.refusing or not asked_as_agent or form["scope"] in self.refused_scopes:
            self.refusals.put_nowait(None)
            return httpx2.Response(401, json={"error": "invalid_client"})
        claims = {
            "iss": ISSUER,
            "sub": subject,
            "groups": self.groups,
            "agent_name": self._names.get(form["client_id"], self.agent_name),
        }
        token = jwt_of(claims)
        self.issued.append(token)
        self.issued_for[form["scope"]].append(token)
        if self.reaches_app:
            self._apps[form["scope"]].let_in(token, subject)
        body = {"access_token": token, "expires_in": LIFETIME_SECONDS, "token_type": "Bearer"}
        return httpx2.Response(200, content=json.dumps(body))
