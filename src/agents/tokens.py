"""The token boundary: client credentials against the realm's token endpoint."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Protocol

import httpx2
from agent_runtime_client.turn import NonEmptyUnicodeText, Principal
from pydantic import BaseModel, ConfigDict

from agents.failures import INVALID_ANSWER_CODE, TokenStatusFailure, Unreachable, UnusableAnswer
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
        """Raises ``TokenStatusFailure`` when the realm refuses, ``UnusableAnswer`` when it answers no token
        and ``Unreachable`` when it does not answer."""
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
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None
    if not isinstance(token, str) or not token or not isinstance(lifetime, int | float) or lifetime <= 0:
        raise UnusableAnswer(INVALID_ANSWER_CODE)
    return GrantedToken(Secret(token), float(lifetime))


@dataclass(frozen=True)
class TokenFacts:
    """What the realm signed into a granted token about the agent itself."""

    principal: Principal
    groups: frozenset[str]
    name: str


class _Claims(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    iss: NonEmptyUnicodeText
    sub: NonEmptyUnicodeText
    agent_name: NonEmptyUnicodeText
    groups: list[str] = []


def facts_of(token: Secret) -> TokenFacts:
    """The facts in the payload of a granted token; its signature is the app's to check, never read here.

    Raises ``UnusableAnswer`` when the token is no JWT with these claims.
    """
    try:
        _, payload, _ = token.reveal().split(".")
        claims = _Claims.model_validate_json(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:
        raise UnusableAnswer(INVALID_ANSWER_CODE) from None
    return TokenFacts(Principal(iss=claims.iss, sub=claims.sub), frozenset(claims.groups), claims.agent_name)
