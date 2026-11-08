"""An app that serves agent-runtime's own MCP front on a loopback socket, behind the gate a host has.

The front, its resources and its change signal are agent_runtime_client's; the app's standings and
the accept tool are scripted. A test may script the answer to a request method to misbehave once.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from agent_runtime_client.mcp_server import McpPersonFront, PersonRequest, PersonTool, ResourceChanges
from agent_runtime_client.mcp_wire import MCP_ENDPOINT_PATH
from agent_runtime_client.member import (
    ACCEPT_INVITATION_TOOL,
    STANDING_URI,
    AcceptInvitationArguments,
    Invitation,
    MemberStanding,
    PlaceRef,
    PlaceWindow,
    Standing,
    StandingState,
    place_window_uri,
)
from agent_runtime_client.tools import ToolDeclaration, ToolEffect, ToolOutcome
from agent_runtime_client.turn import JsonObject, Principal

from tests.support.loopback import serving
from tests.support.wire import ConnectionCut, Handler, Reply, Rpc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

AGENT_SUBJECT = "agent-subject"
ISSUER = "https://realm.test/realms/test"
LIMIT_SECONDS = 10
_SUBJECT_KEY = "test.subject"

DISCOVER = "server/discover"
LISTEN = "subscriptions/listen"
READ = "resources/read"
CALL = "tools/call"
INITIALIZE = "initialize"
ACKNOWLEDGED = "acknowledged"

_ACCEPT_DECLARATION = ToolDeclaration(
    name=ACCEPT_INVITATION_TOOL,
    description="Joins the place an invitation is for.",
    parameters=AcceptInvitationArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)


def invited(place_id: str, invitation_id: str, epoch: int = 0, kind: str = "call") -> Standing:
    invitation = Invitation(id=invitation_id, epoch=epoch, expires_at=datetime(2099, 1, 1, tzinfo=UTC))
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.INVITED, invitation=invitation)


def active(place_id: str) -> Standing:
    return Standing(place=PlaceRef(kind="call", id=place_id), state=StandingState.ACTIVE)


class RequestLog:
    """What the app saw, in order: each request's method, and the acknowledgement of each stream."""

    def __init__(self) -> None:
        self.entries: list[str] = []
        self._added = asyncio.Event()

    def add(self, entry: str) -> None:
        self.entries.append(entry)
        self._added.set()

    def count(self, entry: str) -> int:
        return self.entries.count(entry)

    def reads_follow_acknowledgements(self) -> bool:
        """Every read came after the acknowledgement of the stream that was open when it arrived."""
        acknowledged = False
        for entry in self.entries:
            if entry == LISTEN:
                acknowledged = False
            elif entry == ACKNOWLEDGED:
                acknowledged = True
            elif entry == READ and not acknowledged:
                return False
        return True

    async def until(self, entry: str, count: int = 1) -> None:
        """Returns once the entry was seen this many times; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while self.count(entry) < count:
                self._added.clear()
                await self._added.wait()


@dataclass
class FakeApp:
    """What a test scripts and reads of the app: who stands where, who is let in and what was accepted."""

    bearers: dict[str, str] = field(default_factory=dict[str, str])
    standings: dict[str, list[Standing]] = field(default_factory=dict[str, list[Standing]])
    refuse_with: HTTPStatus | None = None
    refused_method: str | None = None
    lists_standing: bool = True
    pulses_on_read: bool = False
    accept_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    accepted: list[AcceptInvitationArguments] = field(default_factory=list[AcceptInvitationArguments])
    log: RequestLog = field(default_factory=RequestLog)
    refusals: asyncio.Queue[HTTPStatus] = field(default_factory=asyncio.Queue[HTTPStatus])
    changes: ResourceChanges = field(default_factory=ResourceChanges)
    scripted: defaultdict[str, list[Handler]] = field(default_factory=lambda: defaultdict(list[Handler]))
    held_read: asyncio.Event | None = None
    read_held: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self._front = McpPersonFront(
            [PersonTool(_ACCEPT_DECLARATION, self._accept)],
            resolve_person=lambda scope: Principal(iss=ISSUER, sub=scope[_SUBJECT_KEY]),
            resources=self,
            changes=self.changes,
        )

    def let_in(self, bearer: str, subject: str) -> None:
        self.bearers[bearer] = subject

    def stand(self, subject: str, *standings: Standing) -> None:
        """Sets the subject's standings and signals the change, as an app does once the state is visible."""
        self.standings[subject] = list(standings)
        self.changes.changed(Principal(iss=ISSUER, sub=subject), STANDING_URI)

    def hold_next_read(self) -> asyncio.Event:
        """The next read takes its snapshot, tells ``read_held`` and answers once the returned event is set."""
        self.held_read = asyncio.Event()
        return self.held_read

    def script(self, method: str, *handlers: Handler) -> None:
        """Each of the next requests of this method is answered by one handler, in order."""
        self.scripted[method].extend(handlers)

    async def current(self, principal: Principal) -> list[str]:
        if not self.lists_standing:
            return []
        return [STANDING_URI, *(place_window_uri(each.place) for each in self.standings.get(principal.sub, []))]

    async def read(self, principal: Principal, uri: str) -> MemberStanding | PlaceWindow | None:
        if uri != STANDING_URI:
            return None
        snapshot = MemberStanding(standings=tuple(self.standings.get(principal.sub, [])))
        if self.pulses_on_read:
            self.changes.changed(principal, STANDING_URI)
        held, self.held_read = self.held_read, None
        if held is not None:
            self.read_held.set()
            await held.wait()
        return snapshot

    async def _accept(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        self.accepted.append(AcceptInvitationArguments.model_validate(arguments))
        if self.accept_outcomes:
            return self.accept_outcomes.pop(0)
        return ToolOutcome(json.dumps({"name": "Mozart"}), is_error=False)

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[str]:
        """The app's MCP URL, served until the end."""
        try:
            async with self._front.running(), serving(self._gate) as origin:
                yield f"{origin}{MCP_ENDPOINT_PATH}"
        except ExceptionGroup as group:
            # The front's task group wraps what its host raises through it; the host's caller sees the bare error.
            if len(group.exceptions) != 1:
                raise
            raise group.exceptions[0] from None

    async def _gate(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        body = await _body_of(receive)
        document = json.loads(body) if body else {}
        method = document.get("method", "")
        self.log.add(method)
        headers = {name.decode(): value.decode() for name, value in scope["headers"]}
        bearer = headers.get("authorization", "").removeprefix("Bearer ")
        noting = self._noting_acknowledgements(send)
        reply = Reply(Rpc(document.get("id", 0), method, document.get("params") or {}), noting)
        if bearer not in self.bearers:
            self.refusals.put_nowait(HTTPStatus.UNAUTHORIZED)
            await reply.status(HTTPStatus.UNAUTHORIZED)
        elif self.refuse_with is not None and self.refused_method in (None, method):
            self.refusals.put_nowait(self.refuse_with)
            await reply.status(self.refuse_with)
        elif self.scripted[method]:
            try:
                await self.scripted[method].pop(0)(reply)
            except ConnectionCut:
                self.log.add(f"{method} cut")
                raise
        else:
            scope[_SUBJECT_KEY] = self.bearers[bearer]
            await self._front(scope, _replayed(body, receive), noting)

    def _noting_acknowledgements(self, send: Callable[..., Any]) -> Callable[..., Any]:
        async def noting(message: dict[str, Any]) -> None:
            if b"subscriptions/acknowledged" in message.get("body", b""):
                self.log.add(ACKNOWLEDGED)
            await send(message)

        return noting


async def _body_of(receive: Callable[..., Any]) -> bytes:
    body = b""
    more = True
    while more:
        message = await receive()
        body += message.get("body", b"")
        more = message.get("more_body", False)
    return body


def _replayed(body: bytes, receive: Callable[..., Any]) -> Callable[..., Any]:
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def replay() -> dict[str, Any]:
        return pending.pop() if pending else await receive()

    return replay
