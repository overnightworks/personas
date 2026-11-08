"""The token boundary: client credentials against the realm's token endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Protocol

import httpx2

from agents.failures import TokenStatusFailure, Unreachable
from agents.secret import Secret

if TYPE_CHECKING:
    from agents.config import AgentCredentials

TOKEN_TIMEOUT_SECONDS: Final = 30.0


@dataclass(frozen=True)
class GrantedToken:
    value: Secret
    lifetime_seconds: float


class TokenSource(Protocol):
    async def fetch(self, agent: AgentCredentials, scope: str) -> GrantedToken:
        """Raises ``TokenStatusFailure`` when the realm refuses and ``Unreachable`` when it does not answer."""
        ...


class HttpTokenSource:
    def __init__(self, endpoint: str, client: httpx2.AsyncClient) -> None:
        self._endpoint = endpoint
        self._client = client

    async def fetch(self, agent: AgentCredentials, scope: str) -> GrantedToken:
        form = {
            "grant_type": "client_credentials",
            "client_id": agent.client_id,
            "client_secret": agent.client_secret.reveal(),
            "scope": scope,
        }
        try:
            response = await self._client.post(self._endpoint, data=form, timeout=TOKEN_TIMEOUT_SECONDS)
        except httpx2.HTTPError:
            raise Unreachable("the token endpoint did not answer") from None
        if response.status_code != HTTPStatus.OK:
            raise TokenStatusFailure(response.status_code)
        return _granted(response.content)


def _granted(body: bytes) -> GrantedToken:
    try:
        grant = json.loads(body)
        token = grant["access_token"]
        lifetime = grant["expires_in"]
    except ValueError, KeyError, TypeError:
        raise Unreachable("the token endpoint answered something else than a token") from None
    if not isinstance(token, str) or not token or not isinstance(lifetime, int | float) or lifetime <= 0:
        raise Unreachable("the token endpoint answered something else than a token")
    return GrantedToken(Secret(token), float(lifetime))
