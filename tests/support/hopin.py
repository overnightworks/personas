"""An app that serves agent-runtime's own MCP front on a loopback socket, behind the gate a host has.

The front, its resources and its change signal are agent_runtime_client's; the app's standings and
the accept tool are scripted. A test may script the answer to a request method to misbehave once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
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
    PlaceEntry,
    PlaceRef,
    PlaceWindow,
    Standing,
    StandingState,
    place_ref_from_window_uri,
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
LIST_TOOLS = "tools/list"
INITIALIZE = "initialize"
ACKNOWLEDGED = "acknowledged"

SAY_TOOL = "say"
GENERATION = "generation-1"
WINDOW_LINES = 100
"""How many of a place's newest lines its window holds, as hopin's chat window does."""

_SAY_DECLARATION = ToolDeclaration(
    name=SAY_TOOL,
    description="Says a line in a call.",
    parameters={
        "type": "object",
        "properties": {"place": {"type": "string"}, "text": {"type": "string"}},
        "required": ["place", "text"],
    },
    effect=ToolEffect.SIDE_EFFECTING,
)

_ACCEPT_DECLARATION = ToolDeclaration(
    name=ACCEPT_INVITATION_TOOL,
    description="Joins the place an invitation is for.",
    parameters=AcceptInvitationArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)


def invited(place_id: str, invitation_id: str, epoch: int = 0, kind: str = "call") -> Standing:
    invitation = Invitation(id=invitation_id, epoch=epoch, expires_at=datetime(2099, 1, 1, tzinfo=UTC))
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.INVITED, invitation=invitation)


@dataclass(frozen=True)
class Line:
    """A line of a place, as the app keeps it: who said it and what name it shows."""

    seq: int
    author: str
    name: str
    text: str


def active(place_id: str) -> Standing:
    return Standing(place=PlaceRef(kind="call", id=place_id), state=StandingState.ACTIVE)


def paused(place_id: str) -> Standing:
    """The sponsor is away: the member keeps the place and its window, and nobody reads it meanwhile."""
    return Standing(place=PlaceRef(kind="call", id=place_id), state=StandingState.PAUSED)


ACCEPTED_NAME = "Mozart"


def formatted(records: list[logging.LogRecord]) -> str:
    """The records as a handler writes them, with the traceback of each."""
    formatter = logging.Formatter()
    return "\n".join(formatter.format(record) for record in records)


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
    lines: dict[str, list[Line]] = field(default_factory=dict[str, list[Line]])
    generations: dict[str, str] = field(default_factory=dict[str, str])
    reads: dict[str, int] = field(default_factory=dict[str, int])
    read_added: asyncio.Event = field(default_factory=asyncio.Event)
    vanished: set[str] = field(default_factory=set[str])
    say_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    said: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    said_added: asyncio.Event = field(default_factory=asyncio.Event)
    held_window_read: asyncio.Event | None = None
    window_read_held: asyncio.Event = field(default_factory=asyncio.Event)
    log: RequestLog = field(default_factory=RequestLog)
    refusals: asyncio.Queue[HTTPStatus] = field(default_factory=asyncio.Queue[HTTPStatus])
    changes: ResourceChanges = field(default_factory=ResourceChanges)
    scripted: defaultdict[str, list[Handler]] = field(default_factory=lambda: defaultdict(list[Handler]))
    held_read: asyncio.Event | None = None
    read_held: asyncio.Event = field(default_factory=asyncio.Event)
    presented: set[str] = field(default_factory=set[str])
    content: set[str] = field(default_factory=lambda: {ACCEPTED_NAME})

    def __post_init__(self) -> None:
        self._front = McpPersonFront(
            self._tools(),
            resolve_person=lambda scope: Principal(iss=ISSUER, sub=scope[_SUBJECT_KEY]),
            resources=self,
            changes=self.changes,
        )

    def _tools(self) -> list[PersonTool]:
        return [PersonTool(_ACCEPT_DECLARATION, self._accept), PersonTool(_SAY_DECLARATION, self._say)]

    def let_in(self, bearer: str, subject: str) -> None:
        self.bearers[bearer] = subject
        self.presented.add(bearer)

    def stand(self, subject: str, *standings: Standing) -> None:
        """Sets the subject's standings and signals the change, as an app does once the state is visible."""
        self.standings[subject] = list(standings)
        for each in standings:
            self.content.update([each.place.id, *([each.invitation.id] if each.invitation else [])])
        self.changes.changed(Principal(iss=ISSUER, sub=subject), STANDING_URI)

    def speak(self, place_id: str, author: str, name: str, text: str) -> int:
        """Adds a line to a place and signals its window to every member standing there, once it is visible."""
        place_lines = self.lines.setdefault(place_id, [])
        seq = place_lines[-1].seq + 1 if place_lines else 1
        place_lines.append(Line(seq, author, name, text))
        del place_lines[:-WINDOW_LINES]
        self.content.add(text)
        window = place_window_uri(PlaceRef(kind="call", id=place_id))
        for subject, standings in self.standings.items():
            if any(standing.place.id == place_id for standing in standings):
                self.changes.changed(Principal(iss=ISSUER, sub=subject), window)
        return seq

    def join(self, place_id: str) -> None:
        """A join gives the place a new, existing window: a fresh generation, numbered from seq 1, with no old line."""
        self.generations[place_id] = f"generation-{secrets.token_hex(4)}"
        self.lines[place_id] = []
        self.vanished.discard(place_id)

    def hold_next_window_read(self) -> asyncio.Event:
        """The next window read takes its snapshot, tells ``window_read_held`` and answers once the event is set."""
        self.held_window_read = asyncio.Event()
        self.window_read_held.clear()
        return self.held_window_read

    def kept_out_of_logs(self) -> set[str]:
        """Every bearer this app let in or was shown, and every piece of content it served or took."""
        return {*self.presented, *self.content} - {""}

    def hold_next_read(self) -> asyncio.Event:
        """The next read takes its snapshot, tells ``read_held`` and answers once the returned event is set."""
        self.held_read = asyncio.Event()
        self.read_held.clear()
        return self.held_read

    def script(self, method: str, *handlers: Handler) -> None:
        """Each of the next requests of this method is answered by one handler, in order."""
        self.scripted[method].extend(handlers)

    async def current(self, principal: Principal) -> list[str]:
        if not self.lists_standing:
            return []
        return [STANDING_URI, *(place_window_uri(each.place) for each in self.standings.get(principal.sub, []))]

    async def read(self, principal: Principal, uri: str) -> MemberStanding | PlaceWindow | None:
        self.reads[uri] = self.reads.get(uri, 0) + 1
        self.read_added.set()
        place = place_ref_from_window_uri(uri)
        if place is not None:
            return await self._window(principal, place)
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

    async def _window(self, principal: Principal, place: PlaceRef) -> PlaceWindow | None:
        window = self._snapshot(principal, place)
        held, self.held_window_read = self.held_window_read, None
        if held is not None:
            self.window_read_held.set()
            await held.wait()
        return window

    def _snapshot(self, principal: Principal, place: PlaceRef) -> PlaceWindow | None:
        if place.id in self.vanished or not any(each.place == place for each in self.standings.get(principal.sub, [])):
            return None
        entries = tuple(
            PlaceEntry(seq=line.seq, kind="message", name=line.name, text=line.text, own=line.author == principal.sub)
            for line in self.lines.get(place.id, [])
        )
        return PlaceWindow(
            generation=self.generations.get(place.id, GENERATION),
            starts_at=entries[0].seq if entries else 1,
            entries=entries,
        )

    async def _say(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        if self.say_outcomes:
            return self.say_outcomes.pop(0)
        place_id, text = str(arguments["place"]), str(arguments["text"])
        self.said.append((request.principal.sub, place_id, text))
        self.said_added.set()
        self.speak(place_id, request.principal.sub, ACCEPTED_NAME, text)
        return ToolOutcome("said", is_error=False)

    async def until_window_read_held(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.window_read_held.wait()

    async def until_read(self, uri: str, count: int = 1) -> None:
        """Returns once the resource was read this many times, answered or not; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while self.reads.get(uri, 0) < count:
                self.read_added.clear()
                await self.read_added.wait()

    async def until_window_read(self, place_id: str, count: int = 1) -> None:
        await self.until_read(place_window_uri(PlaceRef(kind="call", id=place_id)), count)

    async def until_standing_read(self, count: int = 1) -> None:
        await self.until_read(STANDING_URI, count)

    def window_reads(self, place_id: str) -> int:
        return self.reads.get(place_window_uri(PlaceRef(kind="call", id=place_id)), 0)

    async def until_read_held(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.read_held.wait()

    async def said_count(self, count: int) -> None:
        """Returns once this many lines were said through the tool; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.said) < count:
                self.said_added.clear()
                await self.said_added.wait()

    async def _accept(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        taken = AcceptInvitationArguments.model_validate(arguments)
        self.accepted.append(taken)
        if self.accept_outcomes:
            return self.accept_outcomes.pop(0)
        self.join(taken.place.id)
        return ToolOutcome(json.dumps({"name": ACCEPTED_NAME}), is_error=False)

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
        if method == CALL:
            self._take_in(document.get("params") or {})
        headers = {name.decode(): value.decode() for name, value in scope["headers"]}
        bearer = headers.get("authorization", "").removeprefix("Bearer ")
        self.presented.add(bearer)
        noting = self._noting_acknowledgements(send)
        reply = Reply(Rpc(document.get("id", 0), method, document.get("params") or {}), noting, receive)
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

    def _take_in(self, call: dict[str, Any]) -> None:
        """Content a request carries counts as held before any scripted answer, refusal or status can ignore it."""
        arguments = call.get("arguments") or {}
        place = arguments.get("place") or {}
        place_id = place.get("id") if isinstance(place, dict) else place
        taken = (place_id, arguments.get("invitation"), arguments.get("text"))
        self.content.update(value for value in taken if isinstance(value, str))

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
