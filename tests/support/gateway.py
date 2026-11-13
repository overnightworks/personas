"""The model gateway as the library's scripted fake, with a way to hold a request and to read what a turn was asked."""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx2
from agents.backend.fake_gateway import FakeGateway, Reply, completion, tool_call, tool_round

from tests.support.call import PLACE_ID
from tests.support.history import TABLE
from tests.support.hopin import LIMIT_SECONDS, PLAY_TOOL, SAY_TOOL

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.tools import JsonObject

TURN_OPENING_MESSAGES = 2
"""A turn's first round holds the persona and the message told; every later round adds the model's calls and results."""
BUDGET_SPENT = Reply(
    status=HTTPStatus.UNPROCESSABLE_ENTITY,
    body=json.dumps({"error": {"type": "budget_exceeded", "code": "422"}}).encode(),
)
"""The gateway's refusal once the agent's budget is spent."""


def says(place_id: str, *texts: str, tool: str = SAY_TOOL) -> list[Reply]:
    """A model that answers each turn with one call of the tool with the next text and then a closing word."""
    replies: list[Reply] = []
    for text in texts:
        replies.append(tool_round(tool_call(f"call-{len(replies)}", tool, {"place": place_id, "text": text})))
        replies.append(completion("done"))
    return replies


def speaks_and_says(place_id: str, spoken: str, written: str, *, after: str = "") -> list[Reply]:
    """A model that writes the spoken text and calls ``say`` with the written one in the same round, and closes with
    the text after, when there is one."""
    call = tool_call("call-0", SAY_TOOL, {"place": place_id, "text": written})
    return [tool_round(call, content=spoken), completion(*([after] if after else []))]


def say(text: str, place_id: str = PLACE_ID, tool: str = SAY_TOOL):
    """A call of the line tool that says the text, in the call place unless another is named."""
    return tool_call(secrets.token_hex(4), tool, {"place": place_id, "text": text})


def play(text: str):
    """A call of the line tool of the table, at his table."""
    return say(text, TABLE.id, PLAY_TOOL)


def says_after_a_hold(place_id: str, text: str, *, tool: str = SAY_TOOL) -> list[Reply]:
    """A model that answers a turn with one call of the tool with the text, calls it again when the first was not sent,
    and then closes."""
    arguments = {"place": place_id, "text": text}
    return [
        tool_round(tool_call("call-0", tool, arguments)),
        tool_round(tool_call("call-1", tool, arguments)),
        completion("done"),
    ]


class PausedReply:
    """A model reply written in pieces: the first piece arrives at once, each later one when the test releases it."""

    def __init__(self, *pieces: str) -> None:
        events = [each + b"\n\n" for each in completion(*pieces).body.split(b"\n\n") if each]
        self._pieces = events[: len(pieces)]
        self._ending = b"".join(events[len(pieces) :])
        self._gates = [asyncio.Event() for _ in pieces]
        self._gates[0].set()
        self._cut = False
        self.finished = False
        """Whether the reply was written to its end."""

    def release(self) -> None:
        """Lets the next piece arrive."""
        next(gate for gate in self._gates if not gate.is_set()).set()

    def cut(self) -> None:
        """Ends the reply with a piece the gateway cannot read, once the pieces released so far are written."""
        self._cut = True
        for gate in self._gates:
            gate.set()

    async def stream(self) -> AsyncIterator[bytes]:
        for gate, piece in zip(self._gates, self._pieces, strict=True):
            await gate.wait()
            if self._cut:
                yield b"data: {\n\n"
                return
            yield piece
        self.finished = True
        yield self._ending


class _Written(httpx2.AsyncByteStream):
    def __init__(self, reply: PausedReply) -> None:
        self._reply = reply

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._reply.stream()


@dataclass(frozen=True)
class TurnOpening:
    """What a turn's first request told the gateway: the persona, the message it was told and whom it is for."""

    system: str
    room: JsonObject
    user: str

    @property
    def missed(self) -> bool:
        """Whether the turn was told that lines older than the ones it saw were missed."""
        return bool(self.room["woke"]["missed"])

    @property
    def aloud(self) -> bool:
        """Whether the turn was told that its answer is heard aloud."""
        return bool(self.room["woke"].get("aloud", False))

    def told(self, *kinds: str) -> list[JsonObject]:
        """The entries of the history the turn was told, those of these kinds, in the order told."""
        return [each for each in self.room["history"]["entries"] if each["entry"] in kinds]

    @property
    def new_seqs(self) -> list[int]:
        """The seqs of the lines the turn is for; an entry of another kind marked new fails, never filtered out."""
        new = [each for each in self.room["history"]["entries"] if each["new"]]
        assert all(each["entry"] == "line" for each in new), "a view entry is marked new in a turn told as lines"
        return [line["seq"] for line in new]


class Gateway(FakeGateway):
    """Answers like the library's fake; a test may hold the next request until it releases it."""

    def __init__(self, *replies: Reply) -> None:
        super().__init__(*replies)
        self._holds: dict[int, asyncio.Event] = {}
        self._arrivals = 0
        self._arrived = asyncio.Event()
        self._answered = asyncio.Event()
        self.cancelling_at_arrival: list[int] = []
        """The cancel requests pending on the task that made each request, in arrival order: a turn's own task."""
        self.users_at_arrival: list[str] = []
        """The ``user`` of each request in arrival order, a held request included."""
        self._scripts: dict[str, list[Reply]] = {}
        self._pauses: dict[int, PausedReply] = {}

    @property
    def arrivals(self) -> int:
        """How many requests arrived, a held one included."""
        return self._arrivals

    def script_for(self, user: str, *replies: Reply) -> None:
        """The requests of this ``user`` take their replies from these, each the next one and the last repeating,
        whatever the other users are answered."""
        self._scripts[user] = list(replies)

    def pause_request(self, number: int, *pieces: str) -> PausedReply:
        """The request that arrives as this number (from one) is answered with a reply of these pieces, which takes none
        of the scripted replies."""
        self._pauses[number] = PausedReply(*pieces)
        return self._pauses[number]

    def hold_request(self, number: int) -> asyncio.Event:
        """The request that arrives as this number (from one) waits, once it arrived, until the event is set."""
        self._holds[number] = asyncio.Event()
        return self._holds[number]

    async def handle_async_request(self, request):
        self._arrivals += 1
        number = self._arrivals
        task = asyncio.current_task()
        self.cancelling_at_arrival.append(task.cancelling() if task is not None else 0)
        await request.aread()
        self.users_at_arrival.append(json.loads(request.content)["user"])
        held = self._holds.pop(self._arrivals, None)
        self._arrived.set()
        if held is not None:
            await held.wait()
        await request.aread()
        paused = self._pauses.pop(number, None)
        script = self._scripts.get(json.loads(request.content)["user"])
        if paused is not None:
            self.requests.append(request)
            response = httpx2.Response(
                HTTPStatus.OK, headers={"content-type": "text/event-stream"}, stream=_Written(paused)
            )
        elif script is None:
            response = await super().handle_async_request(request)
        else:
            self.requests.append(request)
            reply = script.pop(0) if len(script) > 1 else script[0]
            response = httpx2.Response(reply.status, headers=dict(reply.headers), content=reply.body)
        self._answered.set()
        return response

    async def until_requested(self, count: int) -> None:
        """Returns once this many requests arrived; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while self._arrivals < count:
                self._arrived.clear()
                await self._arrived.wait()

    async def until_answered(self, count: int) -> None:
        """Returns once this many requests were answered; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.requests) < count:
                self._answered.clear()
                await self._answered.wait()

    async def until_turns(self, count: int, why: str | None = None) -> None:
        """Returns once this many turns (of this kind, when told: ``lines`` or ``view``) have been answered their first
        round; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.turns(why)) < count:
                self._answered.clear()
                await self._answered.wait()

    def turns(self, why: str | None = None) -> list[TurnOpening]:
        """The turns' first requests in arrival order, all or those whose message says they woke for this reason; a
        voice decision's request is no turn."""
        return [opening for opening in self.openings() if why is None or opening.room.get("woke", {}).get("why") == why]

    async def until_written_turn(self, place_id: str, seq: int, user: str | None = None) -> TurnOpening:
        """The first written turn (of this ``user``, when told) at this place whose message marks this line new, once
        it was answered its first round; fails after the limit. Lines taken while another turn runs join one turn, so
        how many turns precede it is not told."""

        def for_the_line() -> TurnOpening | None:
            return next(
                (
                    each
                    for each in self.turns("lines")
                    if each.room["woke"]["place"]["id"] == place_id
                    and seq in each.new_seqs
                    and user in (None, each.user)
                ),
                None,
            )

        async with asyncio.timeout(LIMIT_SECONDS):
            while (found := for_the_line()) is None:
                self._answered.clear()
                await self._answered.wait()
        return found

    def openings(self) -> list[TurnOpening]:
        bodies = [json.loads(request.content) for request in self.requests]
        return [
            TurnOpening(body["messages"][0]["content"], json.loads(body["messages"][1]["content"]), body["user"])
            for body in bodies
            if len(body["messages"]) == TURN_OPENING_MESSAGES
        ]

    def system_of(self, request_number: int) -> str:
        """The system text the request that arrived as this number (from one) carried."""
        return json.loads(self.requests[request_number - 1].content)["messages"][0]["content"]

    def told_in(self, request_number: int) -> str:
        """Everything but the system text that the request that arrived as this number (from one) carried."""
        messages = json.loads(self.requests[request_number - 1].content)["messages"]
        return "\n".join(message["content"] for message in messages[1:])

    def lines_told(self, request_number: int) -> list[JsonObject]:
        """The lines of the last user message of the request that arrived as this number (from one), as written."""
        messages = json.loads(self.requests[request_number - 1].content)["messages"]
        return [json.loads(each) for each in messages[-1]["content"].splitlines()]

    def heard_in(self, request_number: int) -> list[JsonObject]:
        """Every line the request that arrived as this number (from one) told, as written, in order."""
        messages = json.loads(self.requests[request_number - 1].content)["messages"]
        return [
            json.loads(each)
            for message in messages
            if message["role"] == "user"
            for each in message["content"].splitlines()
        ]

    def user_of(self, request_number: int) -> str:
        """The ``user`` the request that arrived as this number (from one) carried."""
        return json.loads(self.requests[request_number - 1].content)["user"]

    def tool_results(self, request_number: int) -> list[str]:
        """The tool messages the request that arrived as this number (from one) carried, in order."""
        messages = json.loads(self.requests[request_number - 1].content)["messages"]
        return [message["content"] for message in messages if message["role"] == "tool"]

    def held_result_texts(self, request_number: int) -> list[str]:
        """The tool results of the request that arrived as this number (from one) that are a held call's, unparsed."""
        return [each for each in self.tool_results(request_number) if each.startswith('{"sent": false')]

    def held_results(self, request_number: int) -> list[dict]:
        """The tool results of the request that arrived as this number (from one) that are a held call's, as the model
        read them."""
        return [json.loads(each) for each in self.held_result_texts(request_number)]

    def offered_tools(self, request_number: int = 1) -> list[str]:
        """The names of the tools the request that arrived as this number (from one) offered, in the order offered."""
        return [tool["function"]["name"] for tool in json.loads(self.requests[request_number - 1].content)["tools"]]

    def round_sizes(self) -> list[int]:
        """The number of messages of each request in arrival order: a turn's rounds grow, a new turn starts at two."""
        return [len(json.loads(request.content)["messages"]) for request in self.requests]

    def users(self) -> set[str]:
        """The ``user`` every request carried."""
        return {json.loads(request.content)["user"] for request in self.requests}


def words(count: int) -> list[str]:
    return [secrets.token_hex(4) for _ in range(count)]
