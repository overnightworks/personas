"""The token endpoint adapter: what the realm answers becomes a grant or a failure the loop understands."""

from __future__ import annotations

import json
import secrets

import httpx2
import pytest

from agents.failures import INVALID_ANSWER_CODE, TokenStatusFailure, Unreachable, UnusableAnswer
from agents.tokens import HttpTokenSource
from tests.support.fakes import credentials

pytestmark = pytest.mark.anyio

ENDPOINT = "https://realm.test/token"


def source_answering(handler):
    return HttpTokenSource(ENDPOINT, httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


async def test_a_grant_carries_the_token_and_its_lifetime():
    token = secrets.token_urlsafe(24)
    source = source_answering(
        lambda request: httpx2.Response(200, content=json.dumps({"access_token": token, "expires_in": 300}))
    )

    granted = await source.fetch(credentials(), "mcp-app")

    assert (granted.value.reveal(), granted.lifetime_seconds) == (token, 300.0)


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_a_refusal_of_the_realm_is_its_status(status):
    source = source_answering(lambda request: httpx2.Response(status, json={"error": "invalid_client"}))
    agent = credentials()

    with pytest.raises(TokenStatusFailure) as failure:
        await source.fetch(agent, "mcp-app")

    assert failure.value.status == status


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"expires_in": 300}',
        b'{"access_token": "%s", "expires_in": 0}' % secrets.token_urlsafe(8).encode(),
        b'{"access_token": 7, "expires_in": 5}',
    ],
    ids=["not json", "not an object", "no token", "no lifetime left", "token not text"],
)
async def test_an_answer_that_is_no_token_is_an_unusable_answer(body):
    source = source_answering(lambda request: httpx2.Response(200, content=body))
    agent = credentials()

    with pytest.raises(UnusableAnswer) as failure:
        await source.fetch(agent, "mcp-app")

    assert failure.value.code == INVALID_ANSWER_CODE


async def test_a_realm_that_cannot_be_reached_is_unreachable():
    def refuse_connection(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    source = source_answering(refuse_connection)
    agent = credentials()

    with pytest.raises(Unreachable):
        await source.fetch(agent, "mcp-app")


async def test_the_request_asks_for_client_credentials_with_the_apps_scope():
    seen: list[dict[str, str]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(dict(pair.split("=") for pair in request.content.decode().split("&")))
        return httpx2.Response(401)

    agent = credentials("agent-bach")
    source = source_answering(record)
    with pytest.raises(TokenStatusFailure):
        await source.fetch(agent, "mcp-app")

    assert seen == [
        {
            "grant_type": "client_credentials",
            "client_id": "agent-bach",
            "client_secret": agent.client_secret.reveal(),
            "scope": "mcp-app",
        }
    ]
