"""An app that serves agents' own MCP front on a loopback socket, behind the gate a host has.

The front, its resources and its change signal are the library's; the app's standings and
the accept tool are scripted. A test may script the answer to a request method to misbehave once.
Its places live in ``hopin_places`` and its voice in ``hopin_voice``.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from agents.mcp_server import McpPersonFront, PersonRequest, PersonTool
from agents.mcp_wire import MCP_ENDPOINT_PATH
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    SET_TURN_STATE_TOOL,
    SPEAK_TOOL,
    STANDING_URI,
    AcceptInvitationArguments,
    Interrupted,
    MemberStanding,
    PlaceEntry,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SetTurnStateArguments,
    SetTurnStateRefusal,
    SetTurnStateRefused,
    Standing,
    StandingState,
    TalkedOver,
    place_ref_from_view_uri,
    place_ref_from_window_uri,
    place_view_uri,
    place_window_uri,
)
from agents.tools import ToolDeclaration, ToolEffect, ToolOutcome
from agents.turn import JsonObject, Principal

from tests.support.hopin_places import ACCEPTED_NAME, ISSUER
from tests.support.hopin_voice import CONTINUE_DECLARATION, SPEAK_DECLARATION, STOP_DECLARATION, AppVoice
from tests.support.loopback import serving
from tests.support.pictures import base64_of
from tests.support.standings import GENERATION
from tests.support.wire import ConnectionCut, Handler, Reply, Rpc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


LIMIT_SECONDS = 10
_SUBJECT_KEY = "test.subject"

DISCOVER = "server/discover"
LISTEN = "subscriptions/listen"
READ = "resources/read"
CALL = "tools/call"
LIST_TOOLS = "tools/list"
NO_METHOD = ""
"""The key a request that carries no JSON-RPC message, a stream reconnect, is scripted and logged under."""
INITIALIZE = "initialize"
ACKNOWLEDGED = "acknowledged"

SAY_TOOL = "say"
PLAY_TOOL = "play"

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


def a_traceparent() -> str:
    """A W3C ``traceparent`` of a new, sampled trace, as an app stores the trace a line entered or a view moved in."""
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def trace_id_of(traceparent: str) -> str:
    return traceparent.split("-")[1]


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


_SET_TURN_STATE_DECLARATION = ToolDeclaration(
    name=SET_TURN_STATE_TOOL,
    description="States what the member's turns at a place do.",
    parameters=SetTurnStateArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)


@dataclass
class FakeApp(AppVoice):
    """What a test scripts and reads of the app: who stands where, who is let in and what was accepted."""

    bearers: dict[str, str] = field(default_factory=dict[str, str])
    refuse_with: HTTPStatus | None = None
    refused_method: str | None = None
    lists_standing: bool = True
    pulses_on_read: bool = False
    accept_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    accepted: list[AcceptInvitationArguments] = field(default_factory=list[AcceptInvitationArguments])
    accepted_on_the_wire: list[tuple[str, JsonObject]] = field(default_factory=list[tuple[str, JsonObject]])
    reads: dict[str, int] = field(default_factory=dict[str, int])
    read_order: list[str] = field(default_factory=list[str])
    """The uri of each resource read, in the order the reads arrived."""
    read_added: asyncio.Event = field(default_factory=asyncio.Event)
    without_window: set[str] = field(default_factory=set[str])
    """Places whose window is listed and answers a read as unknown, as an app that serves none yet does."""
    failing_windows: set[str] = field(default_factory=set[str])
    """Places whose window read the app answers with a server error."""
    say_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    speak_bearers: list[str] = field(default_factory=list[str])
    traceparents: list[tuple[str, str | None]] = field(default_factory=list[tuple[str, str | None]])
    """The method and the ``traceparent`` header of each request, in order; none when the request carried none."""
    refused_tool: str | None = None
    calls: list[str] = field(default_factory=list[str])
    """The tools called, in the order the calls arrived."""
    call_added: asyncio.Event = field(default_factory=asyncio.Event)
    in_flight: int = 0
    most_in_flight: int = 0
    """The most calls that were being answered at once."""
    said: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    said_added: asyncio.Event = field(default_factory=asyncio.Event)
    held_window_read: asyncio.Event | None = None
    window_read_held: asyncio.Event = field(default_factory=asyncio.Event)
    log: RequestLog = field(default_factory=RequestLog)
    refusals: asyncio.Queue[HTTPStatus] = field(default_factory=asyncio.Queue[HTTPStatus])
    scripted: defaultdict[str, list[Handler]] = field(default_factory=lambda: defaultdict(list[Handler]))
    held_read: asyncio.Event | None = None
    read_held: asyncio.Event = field(default_factory=asyncio.Event)
    presented: set[str] = field(default_factory=set[str])
    says_move_the_view: bool = False
    line_tool: str = SAY_TOOL
    """The name of the tool that takes a line or a move at a place of this app; ``say`` as hopin declares it."""
    turn_states: list[tuple[str, SetTurnStateArguments]] = field(
        default_factory=list[tuple[str, SetTurnStateArguments]]
    )
    """Each statement the app took, with its member, in order; a refused one is not among them. A statement is
    no call: it is not counted among ``calls``, and is logged and scripted under its tool's name."""
    turn_state_taken: asyncio.Condition = field(default_factory=asyncio.Condition)
    turn_state_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    """The answer to each next statement as it arrives, in order, before the app judges it."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._front = McpPersonFront(
            self._tools(),
            resolve_person=lambda scope: Principal(iss=ISSUER, sub=scope[_SUBJECT_KEY]),
            resources=self,
            changes=self.changes,
        )

    def _tools(self) -> list[PersonTool]:
        tools = [
            PersonTool(_ACCEPT_DECLARATION, self._accept),
            PersonTool(self._line_declaration(), self._say),
            PersonTool(SPEAK_DECLARATION, self._speak),
            PersonTool(_SET_TURN_STATE_DECLARATION, self._set_turn_state),
        ]
        if self.declares_hold_tools:
            tools += [
                PersonTool(CONTINUE_DECLARATION, self._continue_speaking),
                PersonTool(STOP_DECLARATION, self._stop_speaking),
            ]
        return tools

    def _line_declaration(self) -> ToolDeclaration:
        return _SAY_DECLARATION.model_copy(update={"name": self.line_tool})

    def let_in(self, bearer: str, subject: str) -> None:
        self.bearers[bearer] = subject
        self.presented.add(bearer)

    def hold_next_window_read(self) -> asyncio.Event:
        """The next window read takes its snapshot, tells ``window_read_held`` and answers once the event is set."""
        self.held_window_read = asyncio.Event()
        self.window_read_held.clear()
        return self.held_window_read

    def served(self, outcome: ToolOutcome) -> ToolOutcome:
        """The outcome, with the picture it carries, in its base64 and its bytes-literal spelling, counted among the
        content kept out of logs."""
        if outcome.image is not None:
            self.content.update((base64_of(outcome.image), repr(outcome.image.content)))
        return outcome

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
        held = self.standings.get(principal.sub, [])
        return [
            STANDING_URI,
            *(uri for each in held for uri in self._uris_of(each)),
        ]

    @staticmethod
    def _uris_of(standing: Standing) -> list[str]:
        """Every place lists its window, and the view beside it where it serves one."""
        window = place_window_uri(standing.place)
        return [place_view_uri(standing.place), window] if standing.serves_view else [window]

    async def read(self, principal: Principal, uri: str) -> MemberStanding | PlaceWindow | PlaceView | None:
        self.reads[uri] = self.reads.get(uri, 0) + 1
        self.read_added.set()
        self.read_order.append(uri)
        viewed = place_ref_from_view_uri(uri)
        if viewed is not None:
            return self._view(principal, viewed)
        place = place_ref_from_window_uri(uri)
        if place is not None:
            return await self._window(principal, place)
        if uri != STANDING_URI:
            return None
        snapshot = MemberStanding(standings=tuple(self.standings.get(principal.sub, [])))
        if self.pulses_on_read:
            await self.changes.changed(principal, STANDING_URI)
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

    def _view(self, principal: Principal, place: PlaceRef) -> PlaceView | None:
        serving = any(each.place == place and each.serves_view for each in self.standings.get(principal.sub, []))
        return self.views.get(place.id) if serving and place.id not in self.vanished else None

    def _snapshot(self, principal: Principal, place: PlaceRef) -> PlaceWindow | None:
        if place.id in self.vanished | self.without_window or not any(
            each.place == place for each in self.standings.get(principal.sub, [])
        ):
            return None
        entries = tuple(
            PlaceEntry(
                seq=line.seq,
                kind="message",
                name=line.name,
                text=line.text,
                own=line.author == principal.sub,
                spoken=line.spoken,
                cut=line.cut,
                act=line.act,
                traceparent=line.traceparent,
            )
            for line in self.lines.get(place.id, [])
        )
        hold_id, said = self.holds.get(place.id, (None, self.talked_over.get(place.id)))
        return PlaceWindow(
            generation=self.generations.get(place.id, GENERATION),
            starts_at=entries[0].seq if entries else 1,
            entries=entries,
            talked_over=None if not said else TalkedOver(id=f"talk-{hold_id}", playing=True, speech=said),
            interrupted=None if hold_id is None else Interrupted(id=hold_id),
            voice_state=self.voice_states.get(place.id),
        )

    async def _say(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        if self.say_outcomes:
            return self.served(self.say_outcomes.pop(0))
        place_id, text = str(arguments["place"]), str(arguments["text"])
        self.said.append((request.principal.sub, place_id, text))
        self.said_added.set()
        self.speak(place_id, request.principal.sub, ACCEPTED_NAME, text)
        if self.says_move_the_view and place_id in self.views:
            moved = self.views[place_id]
            self.show_view(place_id, moved.revision + 1, {"hand": secrets.token_hex(8)}, awaits_you=False)
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

    async def until_window_read(self, place_id: str, count: int = 1, kind: str = "call") -> None:
        await self.until_read(place_window_uri(PlaceRef(kind=kind, id=place_id)), count)

    async def until_standing_read(self, count: int = 1) -> None:
        await self.until_read(STANDING_URI, count)

    async def until_view_read(self, place: PlaceRef, count: int = 1) -> None:
        await self.until_read(place_view_uri(place), count)

    def view_reads(self, place: PlaceRef) -> int:
        return self.reads.get(place_view_uri(place), 0)

    def window_reads(self, place_id: str, kind: str = "call") -> int:
        return self.reads.get(place_window_uri(PlaceRef(kind=kind, id=place_id)), 0)

    async def until_read_held(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.read_held.wait()

    async def said_count(self, count: int) -> None:
        """Returns once this many lines were said through the tool; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.said) < count:
                self.said_added.clear()
                await self.said_added.wait()

    async def until_called(self, count: int) -> None:
        """Returns once this many calls of tools arrived; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.calls) < count:
                self.call_added.clear()
                await self.call_added.wait()

    async def spoken_count(self, count: int) -> None:
        """Returns once this many sentences were spoken through the tool; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.spoken) < count:
                self.spoken_added.clear()
                await self.spoken_added.wait()

    async def until_turn_states(self, count: int) -> None:
        """Returns once the app took this many statements; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS), self.turn_state_taken:
            await self.turn_state_taken.wait_for(lambda: len(self.turn_states) >= count)

    async def _set_turn_state(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Takes the statement of a member who holds a seat, active or paused, at its place."""
        if self.turn_state_outcomes:
            return self.turn_state_outcomes.pop(0)
        statement = SetTurnStateArguments.model_validate(arguments)
        subject = request.principal.sub
        if not any(
            each.place == statement.place and each.state is not StandingState.INVITED
            for each in self.standings.get(subject, [])
        ):
            return ToolOutcome(
                SetTurnStateRefused(refused=SetTurnStateRefusal.NOT_HELD).model_dump_json(), is_error=True
            )
        async with self.turn_state_taken:
            self.turn_states.append((subject, statement))
            self.turn_state_taken.notify_all()
        return ToolOutcome("{}", is_error=False)

    async def _accept(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        taken = AcceptInvitationArguments.model_validate_json(json.dumps(arguments))
        self.accepted.append(taken)
        self.accepted_on_the_wire.append((request.principal.sub, arguments))
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

    @contextmanager
    def _counting_calls(self, document: dict[str, Any]) -> Iterator[None]:
        """A call of a tool counts from its arrival to its answer."""
        if document.get("method") != CALL or _is_statement(document):
            yield
            return
        self.calls.append((document.get("params") or {}).get("name", ""))
        self.call_added.set()
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            yield
        finally:
            self.in_flight -= 1

    async def _gate(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        body = await _body_of(receive)
        document = json.loads(body) if body else {}
        with self._counting_calls(document):
            await self._answer(scope, receive, send, body, document)

    async def _answer(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
        body: bytes,
        document: dict[str, Any],
    ) -> None:
        method = document.get("method", NO_METHOD)
        key = _key_of(document)
        self.log.add(key)
        if method == CALL:
            self._take_in(document.get("params") or {})
        headers = {name.decode(): value.decode() for name, value in scope["headers"]}
        bearer = headers.get("authorization", "").removeprefix("Bearer ")
        self.presented.add(bearer)
        self.traceparents.append((method, headers.get("traceparent")))
        if method == CALL and (document.get("params") or {}).get("name") == SPEAK_TOOL:
            self.speak_bearers.append(bearer)
        noting = self._noting_acknowledgements(send)
        reply = Reply(Rpc(document.get("id", 0), method, document.get("params") or {}), noting, receive)
        if bearer not in self.bearers:
            self.refusals.put_nowait(HTTPStatus.UNAUTHORIZED)
            await reply.status(HTTPStatus.UNAUTHORIZED)
        elif (
            self.refuse_with is not None
            and self.refused_method in (None, method)
            and self.refused_tool in (None, (document.get("params") or {}).get("name"))
        ):
            self.refusals.put_nowait(self.refuse_with)
            await reply.status(self.refuse_with)
        elif method == READ and self._fails_the_window_read(document):
            await reply.status(HTTPStatus.SERVICE_UNAVAILABLE)
        elif self.scripted[key]:
            try:
                await self.scripted[key].pop(0)(reply)
            except ConnectionCut:
                self.log.add(f"{method} cut")
                raise
        else:
            scope[_SUBJECT_KEY] = self.bearers[bearer]
            await self._front(scope, _replayed(body, receive), noting)

    def _fails_the_window_read(self, document: dict[str, Any]) -> bool:
        place = place_ref_from_window_uri((document.get("params") or {}).get("uri", ""))
        return place is not None and place.id in self.failing_windows

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


@dataclass
class QuietApp(FakeApp):
    """An app that lets its members say lines in a call but declares no ``speak``."""

    def _tools(self) -> list[PersonTool]:
        return [PersonTool(_ACCEPT_DECLARATION, self._accept), PersonTool(self._line_declaration(), self._say)]


def _is_statement(document: dict[str, Any]) -> bool:
    return document.get("method") == CALL and (document.get("params") or {}).get("name") == SET_TURN_STATE_TOOL


def _key_of(document: dict[str, Any]) -> str:
    """What a request is logged and scripted under: its method, and a statement its tool's name."""
    return SET_TURN_STATE_TOOL if _is_statement(document) else document.get("method", NO_METHOD)


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
