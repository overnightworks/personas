"""The model gateway as the library's scripted fake, with a way to hold a request and to read what a turn was asked."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_runtime_client.backend.fake_gateway import FakeGateway, Reply, completion, tool_call, tool_round

from tests.support.hopin import LIMIT_SECONDS, SAY_TOOL

if TYPE_CHECKING:
    from agent_runtime_client.tools import JsonObject

TURN_OPENING_MESSAGES = 2
"""A turn's first round holds the persona and the room; every later round adds the model's calls and their results."""


def says(place_id: str, *texts: str) -> list[Reply]:
    """A model that answers each turn with one ``say`` of the next text and then a closing word."""
    replies: list[Reply] = []
    for text in texts:
        replies.append(tool_round(tool_call(f"call-{len(replies)}", SAY_TOOL, {"place": place_id, "text": text})))
        replies.append(completion("done"))
    return replies


@dataclass(frozen=True)
class TurnOpening:
    """What a turn's first request told the gateway: the persona, the room's facts and whom it is for."""

    system: str
    room: JsonObject
    user: str

    @property
    def new_seqs(self) -> list[int]:
        return [line["seq"] for line in self.room["lines"] if line["new"]]


class Gateway(FakeGateway):
    """Answers like the library's fake; a test may hold the next request until it releases it."""

    def __init__(self, *replies: Reply) -> None:
        super().__init__(*replies)
        self._holds: dict[int, asyncio.Event] = {}
        self._arrivals = 0
        self._arrived = asyncio.Event()

    def hold_request(self, number: int) -> asyncio.Event:
        """The request that arrives as this number (from one) waits, once it arrived, until the event is set."""
        self._holds[number] = asyncio.Event()
        return self._holds[number]

    async def handle_async_request(self, request):
        self._arrivals += 1
        held = self._holds.pop(self._arrivals, None)
        self._arrived.set()
        if held is not None:
            await held.wait()
        return await super().handle_async_request(request)

    async def until_requested(self, count: int) -> None:
        """Returns once this many requests arrived; fails after the limit."""
        async with asyncio.timeout(LIMIT_SECONDS):
            while self._arrivals < count:
                self._arrived.clear()
                await self._arrived.wait()

    def openings(self) -> list[TurnOpening]:
        bodies = [json.loads(request.content) for request in self.requests]
        return [
            TurnOpening(body["messages"][0]["content"], json.loads(body["messages"][1]["content"]), body["user"])
            for body in bodies
            if len(body["messages"]) == TURN_OPENING_MESSAGES
        ]

    def round_sizes(self) -> list[int]:
        """The number of messages of each request in arrival order: a turn's rounds grow, a new turn starts at two."""
        return [len(json.loads(request.content)["messages"]) for request in self.requests]

    def users(self) -> set[str]:
        """The ``user`` every request carried."""
        return {json.loads(request.content)["user"] for request in self.requests}
