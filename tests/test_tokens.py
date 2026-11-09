"""The token endpoint adapter: what the realm answers becomes a grant or a failure the loop understands."""

from __future__ import annotations

import base64
import json
import secrets

import httpx2
import pytest

from agents.failures import INVALID_ANSWER_CODE, TokenStatusFailure, Unreachable, UnusableAnswer
from agents.secret import Secret
from agents.tokens import HttpTokenSource, facts_of
from tests.support.fakes import credentials
from tests.support.hopin import ISSUER
from tests.support.realm import jwt_of

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


def _claims(**changes: object) -> dict[str, object]:
    return {
        "iss": ISSUER,
        "sub": secrets.token_hex(8),
        "groups": [secrets.token_hex(4), secrets.token_hex(4)],
        "agent_name": secrets.token_hex(4),
    } | changes


def test_the_facts_of_a_token_are_its_principal_its_groups_and_the_agents_name():
    claims = _claims()

    facts = facts_of(Secret(jwt_of(claims)))

    assert (facts.principal.iss, facts.principal.sub, facts.name) == (
        claims["iss"],
        claims["sub"],
        claims["agent_name"],
    )
    assert facts.groups == frozenset(claims["groups"])


def test_a_token_that_names_no_groups_has_none():
    claims = {name: value for name, value in _claims().items() if name != "groups"}

    assert facts_of(Secret(jwt_of(claims))).groups == frozenset()


def _segments(payload: bytes) -> str:
    return f"header.{base64.urlsafe_b64encode(payload).decode()}.signature"


@pytest.mark.parametrize(
    "token",
    [
        "no-dots-at-all",
        "header.payload",
        "header.payload.signature.extra",
        "header.%%%not-base64%%%.signature",
        _segments(b"not json"),
        _segments(b'["iss"]'),
        _segments(b"\xff\xfe"),
        jwt_of(_claims(iss="")),
        jwt_of(_claims(sub=7)),
        jwt_of(_claims(agent_name=None)),
        jwt_of(_claims(groups="a-group")),
        jwt_of(_claims(groups=[7])),
        jwt_of({name: value for name, value in _claims().items() if name != "sub"}),
        jwt_of({name: value for name, value in _claims().items() if name != "agent_name"}),
    ],
    ids=[
        "no segments",
        "two segments",
        "four segments",
        "payload not base64",
        "payload not JSON",
        "payload not an object",
        "payload not text",
        "empty issuer",
        "subject not text",
        "name null",
        "groups not a list",
        "group not text",
        "no subject",
        "no name",
    ],
)
def test_a_token_that_is_no_jwt_with_these_claims_is_an_unusable_answer(token):
    with pytest.raises(UnusableAnswer) as failure:
        facts_of(Secret(token))

    assert failure.value.code == INVALID_ANSWER_CODE
