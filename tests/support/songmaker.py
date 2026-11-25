"""An app that serves tools only, as songmaker's MCP front does: the gate and the scripting are the hopin fake's."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from agents.backend.fake_gateway import tool_call
from agents.mcp_server import PersonRequest, PersonTool
from agents.tools import ToolDeclaration, ToolEffect, ToolOutcome
from mcp_types import ListToolsResult, Tool

from tests.support.hopin import FakeApp

if TYPE_CHECKING:
    from agents.turn import JsonObject

CREATE_SONG = "create_song"
GENERATE_SONG = "generate_song"
SCOPE = "mcp-songmaker"
OWN_DESCRIPTION = "Its own."
CALLS = 2
"""The call bound a scenario gives the app."""
_TOOLS = (CREATE_SONG, GENERATE_SONG)


def a_song(call_id: str, tool: str = CREATE_SONG):
    """A call of one of the app's tools."""
    return tool_call(call_id, tool, {"title": "t"})


def lists_only(*names: str):
    """A listing of the app that holds only these tools, each described as the app's own."""

    async def lists(reply) -> None:
        tools = [Tool(name=each, description=OWN_DESCRIPTION, input_schema={"type": "object"}) for each in names]
        await reply.result(ListToolsResult(tools=tools))

    return lists


def songmaker_url(runner) -> str:
    (url,) = (each.mcp_url for each in runner.config.apps if not each.listens)
    return url


def _declaration(name: str) -> ToolDeclaration:
    return ToolDeclaration(
        name=name,
        description=f"{name} in songmaker.",
        parameters={"type": "object", "properties": {"title": {"type": "string"}}},
        effect=ToolEffect.SIDE_EFFECTING,
    )


@dataclass
class ToolsOnlyApp(FakeApp):
    """What a test reads of the app: which tool ran for whom, and what each tool answers unless it is scripted."""

    results: dict[str, str] = field(default_factory=lambda: {name: secrets.token_hex(8) for name in _TOOLS})
    called: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])

    def _tools(self) -> list[PersonTool]:
        return [PersonTool(_declaration(name), partial(self._run, name)) for name in _TOOLS]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.content.update(self.results.values())

    async def _run(self, name: str, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        self.called.append((name, request.principal.sub))
        if self.outcomes:
            return self.served(self.outcomes.pop(0))
        return ToolOutcome(self.results[name], is_error=False)
