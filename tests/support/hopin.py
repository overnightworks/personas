"""An app that serves agents' own MCP front on a loopback socket, behind the gate a host has.

The front, its resources and its change signal are the library's; the app's standings and
the accept tool are scripted. A test may script the answer to a request method to misbehave once.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from agents.mcp_server import McpPersonFront, PersonRequest, PersonTool, ResourceChanges
from agents.mcp_wire import MCP_ENDPOINT_PATH
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    SPEAK_TOOL,
    STANDING_URI,
    AcceptInvitationArguments,
    Invitation,
    MemberStanding,
    ParentPlace,
    PlaceEntry,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SpeakArguments,
    Spoken,
    Standing,
    StandingState,
    VoiceState,
    Written,
    place_ref_from_view_uri,
    place_ref_from_window_uri,
    place_view_uri,
    place_window_uri,
)
from agents.tools import ToolDeclaration, ToolEffect, ToolImage, ToolOutcome
from agents.turn import JsonObject, Principal

from tests.support.loopback import serving
from tests.support.wire import ConnectionCut, Handler, Reply, Rpc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

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
PLAY_TOOL = "play"
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

_SPEAK_DECLARATION = ToolDeclaration(
    name=SPEAK_TOOL,
    description="Speaks a sentence in a call.",
    parameters=SpeakArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)

SPOKEN_DURATION_MS = 1500
"""How long a sentence plays unless a test scripts the answer."""

_ACCEPT_DECLARATION = ToolDeclaration(
    name=ACCEPT_INVITATION_TOOL,
    description="Joins the place an invitation is for.",
    parameters=AcceptInvitationArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)


def a_picture(size: int = 64) -> ToolImage:
    """A still picture as the library's wire takes it: random bytes behind the JPEG start marker."""
    return ToolImage(content=b"\xff\xd8\xff" + secrets.token_bytes(size))


def base64_of(picture: ToolImage) -> str:
    """The picture as an image block of a tool result and a data URL of a model request write it."""
    return base64.b64encode(picture.content).decode("ascii")


def invited(
    place_id: str,
    invitation_id: str,
    epoch: int = 0,
    kind: str = "call",
    *,
    serves_view: bool = False,
    parent: ParentPlace | None = None,
) -> Standing:
    invitation = Invitation(id=invitation_id, epoch=epoch, expires_at=datetime(2099, 1, 1, tzinfo=UTC))
    return Standing(
        place=PlaceRef(kind=kind, id=place_id),
        state=StandingState.INVITED,
        invitation=invitation,
        serves_view=serves_view,
        parent=parent,
    )


@dataclass(frozen=True)
class Line:
    """A line of a place, as the app keeps it: who said it and what name it shows."""

    seq: int
    author: str
    name: str
    text: str
    spoken: bool = False
    cut: bool = False
    act: bool = False


def active(place_id: str, kind: str = "call", *, serves_view: bool = False) -> Standing:
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.ACTIVE, serves_view=serves_view)


def paused(place_id: str, kind: str = "call", *, serves_view: bool = False) -> Standing:
    """The sponsor is away: the member keeps the place and its window, and nobody reads it meanwhile."""
    return Standing(place=PlaceRef(kind=kind, id=place_id), state=StandingState.PAUSED, serves_view=serves_view)


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
    accepted_on_the_wire: list[tuple[str, JsonObject]] = field(default_factory=list[tuple[str, JsonObject]])
    lines: dict[str, list[Line]] = field(default_factory=dict[str, list[Line]])
    generations: dict[str, str] = field(default_factory=dict[str, str])
    reads: dict[str, int] = field(default_factory=dict[str, int])
    read_order: list[str] = field(default_factory=list[str])
    """The uri of each resource read, in the order the reads arrived."""
    read_added: asyncio.Event = field(default_factory=asyncio.Event)
    vanished: set[str] = field(default_factory=set[str])
    without_window: set[str] = field(default_factory=set[str])
    """Places whose window is listed and answers a read as unknown, as an app that serves none yet does."""
    failing_windows: set[str] = field(default_factory=set[str])
    """Places whose window read the app answers with a server error."""
    say_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    speak_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    spoken: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    spoken_added: asyncio.Event = field(default_factory=asyncio.Event)
    speak_bearers: list[str] = field(default_factory=list[str])
    speak_duration_ms: int = SPOKEN_DURATION_MS
    written_speaks: int = 0
    """How many of the next ``speak`` calls the app answers as written to the place instead of voiced."""
    voice_states: dict[str, VoiceState] = field(default_factory=dict[str, VoiceState])
    held_speak: asyncio.Event | None = None
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
    changes: ResourceChanges = field(default_factory=ResourceChanges)
    scripted: defaultdict[str, list[Handler]] = field(default_factory=lambda: defaultdict(list[Handler]))
    held_read: asyncio.Event | None = None
    read_held: asyncio.Event = field(default_factory=asyncio.Event)
    views: dict[str, PlaceView] = field(default_factory=dict[str, PlaceView])
    presented: set[str] = field(default_factory=set[str])
    content: set[str] = field(default_factory=lambda: {ACCEPTED_NAME})
    says_move_the_view: bool = False
    line_tool: str = SAY_TOOL
    """The name of the tool that takes a line or a move at a place of this app; ``say`` as hopin declares it."""

    def __post_init__(self) -> None:
        self._front = McpPersonFront(
            self._tools(),
            resolve_person=lambda scope: Principal(iss=ISSUER, sub=scope[_SUBJECT_KEY]),
            resources=self,
            changes=self.changes,
        )

    def _tools(self) -> list[PersonTool]:
        return [
            PersonTool(_ACCEPT_DECLARATION, self._accept),
            PersonTool(self._line_declaration(), self._say),
            PersonTool(_SPEAK_DECLARATION, self._speak),
        ]

    def _line_declaration(self) -> ToolDeclaration:
        return _SAY_DECLARATION.model_copy(update={"name": self.line_tool})

    def let_in(self, bearer: str, subject: str) -> None:
        self.bearers[bearer] = subject
        self.presented.add(bearer)

    def stand(self, subject: str, *standings: Standing) -> None:
        """Sets the subject's standings and signals the change, as an app does once the state is visible."""
        self.standings[subject] = list(standings)
        for each in standings:
            self.content.update([each.place.id, *([each.invitation.id] if each.invitation else [])])
        self.changes.changed(Principal(iss=ISSUER, sub=subject), STANDING_URI)

    def speak(
        self,
        place_id: str,
        author: str,
        name: str,
        text: str,
        *,
        spoken: bool = False,
        cut: bool = False,
        act: bool = False,
    ) -> int:
        """Adds a line to a place and signals its window to every member standing there, once it is visible."""
        place_lines = self.lines.setdefault(place_id, [])
        seq = place_lines[-1].seq + 1 if place_lines else 1
        place_lines.append(Line(seq, author, name, text, spoken, cut, act))
        del place_lines[:-WINDOW_LINES]
        self.content.add(text)
        self._signal_window(place_id)
        return seq

    def _signal_window(self, place_id: str) -> None:
        for subject, standings in self.standings.items():
            for each in standings:
                if each.place.id == place_id:
                    self.changes.changed(Principal(iss=ISSUER, sub=subject), place_window_uri(each.place))

    def state_voice(self, place_id: str, state: VoiceState | None) -> None:
        """Sets what every window of the place states of his voice, none when ``state`` is ``None``, and signals it."""
        if state is None:
            self.voice_states.pop(place_id, None)
        else:
            self.voice_states[place_id] = state
        self._signal_window(place_id)

    def show_view(self, place_id: str, revision: int, state: dict[str, Any], *, awaits_you: bool = True) -> None:
        """Sets a place's view for every member standing there and signals it, as an app does once it is visible."""
        self.views[place_id] = PlaceView(revision=revision, awaits_you=awaits_you, state=dict(state))
        self.content.update(value for value in state.values() if isinstance(value, str))
        for subject, standings in self.standings.items():
            for each in standings:
                if each.place.id == place_id and each.serves_view:
                    self.changes.changed(Principal(iss=ISSUER, sub=subject), place_view_uri(each.place))

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

    def served(self, outcome: ToolOutcome) -> ToolOutcome:
        """The outcome, with the picture it carries counted among the content kept out of logs."""
        if outcome.image is not None:
            self.content.add(base64_of(outcome.image))
        return outcome

    def hold_next_speak(self) -> asyncio.Event:
        """The next ``speak`` that arrives is answered once the event is set."""
        self.held_speak = asyncio.Event()
        return self.held_speak

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
            )
            for line in self.lines.get(place.id, [])
        )
        return PlaceWindow(
            generation=self.generations.get(place.id, GENERATION),
            starts_at=entries[0].seq if entries else 1,
            entries=entries,
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

    async def _speak(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Takes a sentence, appends it as the member's own line and answers its length or that it was written."""
        taken = SpeakArguments.model_validate(arguments)
        self.content.add(taken.text)
        if (held := self.held_speak) is not None:
            self.held_speak = None
            await held.wait()
        if self.speak_outcomes:
            return self.speak_outcomes.pop(0)
        self.spoken.append((request.principal.sub, taken.place.id, taken.text))
        self.spoken_added.set()
        if self.written_speaks:
            self.written_speaks -= 1
            self.speak(taken.place.id, request.principal.sub, ACCEPTED_NAME, taken.text)
            return ToolOutcome(Written(written=True).model_dump_json(), is_error=False)
        self.speak(taken.place.id, request.principal.sub, ACCEPTED_NAME, taken.text, spoken=True)
        return ToolOutcome(Spoken(duration_ms=self.speak_duration_ms).model_dump_json(), is_error=False)

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

    async def _accept(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        taken = AcceptInvitationArguments.model_validate(arguments)
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
        if document.get("method") != CALL:
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
        method = document.get("method", "")
        self.log.add(method)
        if method == CALL:
            self._take_in(document.get("params") or {})
        headers = {name.decode(): value.decode() for name, value in scope["headers"]}
        bearer = headers.get("authorization", "").removeprefix("Bearer ")
        self.presented.add(bearer)
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
        elif self.scripted[method]:
            try:
                await self.scripted[method].pop(0)(reply)
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
