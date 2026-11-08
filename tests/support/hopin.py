"""An app that serves agent-runtime's own MCP front, with the gate in front of it a host has.

The front, the ``wait`` tool and the cursor rules are agent_runtime_client's; only the accept tool
is scripted, as the app owns it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus

import httpx2
from agent_runtime_client.mcp_server import (
    InMemoryMemberChanges,
    McpPersonFront,
    PersonRequest,
    PersonTool,
    wait_tool,
)
from agent_runtime_client.mcp_wire import MCP_ENDPOINT_PATH
from agent_runtime_client.tools import ToolDeclaration, ToolEffect, ToolOutcome
from agent_runtime_client.turn import JsonObject, Principal
from agent_runtime_client.wait import Invitation, PlaceRef, Standing, StandingState

from agents.app_session import ACCEPT_TOOL

ISSUER = "https://realm.test/realms/test"
APP_URL = f"http://app.test{MCP_ENDPOINT_PATH}"
_SUBJECT_KEY = "test.subject"

_ACCEPT_DECLARATION = ToolDeclaration(
    name=ACCEPT_TOOL,
    description="Joins the place an invitation is for.",
    parameters={
        "type": "object",
        "properties": {"room": {"type": "string"}, "invitation": {"type": "string"}, "epoch": {"type": "integer"}},
        "required": ["room", "invitation", "epoch"],
    },
    effect=ToolEffect.SIDE_EFFECTING,
)


def invited(place_id: str, invitation_id: str, epoch: int = 0, kind: str = "call") -> Standing:
    invitation = Invitation(id=invitation_id, epoch=epoch, expires_at=datetime(2099, 1, 1, tzinfo=UTC))
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.INVITED, invitation=invitation)


def active(place_id: str) -> Standing:
    return Standing(place=PlaceRef(kind="call", id=place_id), state=StandingState.ACTIVE)


class RecordingChanges(InMemoryMemberChanges):
    """The front's own change signal, which also records every wait that reaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.cursors_waited_on: list[str | None] = []
        self.in_flight = 0
        self.most_in_flight = 0
        self.held = asyncio.Event()
        self.started: asyncio.Queue[str | None] = asyncio.Queue()

    async def wait(self, recipient: Principal, after: str | None, seconds: float) -> str:
        self.cursors_waited_on.append(after)
        self.started.put_nowait(after)
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        if after is not None:
            self.held.set()
        try:
            return await super().wait(recipient, after, seconds)
        finally:
            self.in_flight -= 1


@dataclass
class FakeApp:
    """What a test scripts and reads of the app: who stands where, who is let in and what was accepted."""

    bearers: dict[str, str] = field(default_factory=dict[str, str])
    standings: dict[str, list[Standing]] = field(default_factory=dict[str, list[Standing]])
    refuse_with: HTTPStatus | None = None
    accept_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    accepted: list[JsonObject] = field(default_factory=list[JsonObject])
    requests_seen: int = 0
    refusals: asyncio.Queue[HTTPStatus] = field(default_factory=asyncio.Queue[HTTPStatus])
    changes: RecordingChanges = field(default_factory=RecordingChanges)
    accepted_event: asyncio.Event = field(default_factory=asyncio.Event)

    def let_in(self, bearer: str, subject: str) -> None:
        self.bearers[bearer] = subject

    def stand(self, subject: str, *standings: Standing) -> None:
        """Sets the subject's standings and signals the change, as an app does once the state is visible."""
        self.standings[subject] = list(standings)
        self.changes.changed(Principal(iss=ISSUER, sub=subject))

    async def _standings(self, principal: Principal) -> list[Standing]:
        return self.standings.get(principal.sub, [])

    async def _accept(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        self.accepted.append(arguments)
        self.accepted_event.set()
        if self.accept_outcomes:
            return self.accept_outcomes.pop(0)
        return ToolOutcome(json.dumps({"room": arguments["room"], "name": "Mozart"}), is_error=False)

    def asgi(self) -> httpx2.ASGITransport:
        front = McpPersonFront(
            [wait_tool(self.changes, self._standings), PersonTool(_ACCEPT_DECLARATION, self._accept)],
            resolve_person=lambda scope: Principal(iss=ISSUER, sub=scope[_SUBJECT_KEY]),
        )

        async def gate(scope, receive, send) -> None:
            self.requests_seen += 1
            headers = {name.decode(): value.decode() for name, value in scope["headers"]}
            bearer = headers.get("authorization", "").removeprefix("Bearer ")
            if bearer not in self.bearers:
                self.refusals.put_nowait(HTTPStatus.UNAUTHORIZED)
                await _answer(send, HTTPStatus.UNAUTHORIZED)
            elif self.refuse_with is not None:
                self.refusals.put_nowait(self.refuse_with)
                await _answer(send, self.refuse_with)
            else:
                scope[_SUBJECT_KEY] = self.bearers[bearer]
                await front(scope, receive, send)

        return httpx2.ASGITransport(app=gate)


async def _answer(send, status: HTTPStatus) -> None:
    await send({"type": "http.response.start", "status": status.value, "headers": []})
    await send({"type": "http.response.body", "body": b""})
