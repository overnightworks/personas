"""The model gateway as the library's scripted fake, with a way to hold a request and to read what a turn was asked."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

from agents.backend.fake_gateway import FakeGateway, Reply, completion, tool_call, tool_round

from tests.support.hopin import LIMIT_SECONDS, SAY_TOOL

if TYPE_CHECKING:
    from agents.tools import JsonObject

TURN_OPENING_MESSAGES = 2
"""A turn's first round holds the persona and the message told; every later round adds the model's calls and results."""
BUDGET_SPENT = Reply(
    status=HTTPStatus.UNPROCESSABLE_ENTITY,
    body=json.dumps({"error": {"type": "budget_exceeded", "code": "422"}}).encode(),
)
"""The gateway's refusal once the agent's budget is spent."""


def says(place_id: str, *texts: str) -> list[Reply]:
    """A model that answers each turn with one ``say`` of the next text and then a closing word."""
    replies: list[Reply] = []
    for text in texts:
        replies.append(tool_round(tool_call(f"call-{len(replies)}", SAY_TOOL, {"place": place_id, "text": text})))
        replies.append(completion("done"))
    return replies


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

    def hold_request(self, number: int) -> asyncio.Event:
        """The request that arrives as this number (from one) waits, once it arrived, until the event is set."""
        self._holds[number] = asyncio.Event()
        return self._holds[number]

    async def handle_async_request(self, request):
        self._arrivals += 1
        task = asyncio.current_task()
        self.cancelling_at_arrival.append(task.cancelling() if task is not None else 0)
        held = self._holds.pop(self._arrivals, None)
        self._arrived.set()
        if held is not None:
            await held.wait()
        response = await super().handle_async_request(request)
        self._answered.set()
        return response

    async def until_requested(self, count: int) -> None:
        """Returns once this many requests arrived; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while self._arrivals < count:
                self._arrived.clear()
                await self._arrived.wait()

    async def until_turns(self, count: int) -> None:
        """Returns once this many turns have been answered their first round; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.openings()) < count:
                self._answered.clear()
                await self._answered.wait()

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

    def offered_tools(self) -> list[str]:
        """The names of the tools the first request offered."""
        return [tool["function"]["name"] for tool in json.loads(self.requests[0].content)["tools"]]

    def round_sizes(self) -> list[int]:
        """The number of messages of each request in arrival order: a turn's rounds grow, a new turn starts at two."""
        return [len(json.loads(request.content)["messages"]) for request in self.requests]

    def users(self) -> set[str]:
        """The ``user`` every request carried."""
        return {json.loads(request.content)["user"] for request in self.requests}
