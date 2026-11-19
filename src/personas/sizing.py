"""What a turn's input weighs, for the turn's end line."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from agents.tools import HandTool, ToolOutcome
from agents.turn import TurnRole

from personas.entries import InputSizes, message_bytes, picture_bytes, told_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.tools import ToolDeclaration
    from agents.turn import JsonObject, Principal

_MODEL_FACING: Final = {"name", "description", "parameters"}
_RESULT_ROLE: Final = "tool"


class Sizing:
    """What the turn's input weighs: the request's tools and told messages once it is built, and every tool result
    relayed back to the model, held and refused calls included. Only bytes are kept."""

    def __init__(self) -> None:
        self._request: tuple[int, int] | None = None
        self._results = 0
        self._images = 0

    def request_built(self, declarations: Iterable[ToolDeclaration], told: Iterable[str]) -> None:
        tools = sum(told_bytes(each.model_dump(mode="json", include=_MODEL_FACING)) for each in declarations)
        self._request = (tools, sum(message_bytes(TurnRole.USER.value, each) for each in told))

    def measuring(self, tool: HandTool) -> HandTool:
        """The tool with a handler that counts the result it returns."""

        async def handler(principal: Principal, arguments: JsonObject) -> ToolOutcome:
            outcome = await tool.handler(principal, arguments)
            self._results += message_bytes(_RESULT_ROLE, outcome.content)
            self._images += picture_bytes(outcome.image)
            return outcome

        return HandTool(tool.declaration, handler)

    def sizes(self) -> InputSizes | None:
        """The weights, or none while the request is not built."""
        if self._request is None:
            return None
        return InputSizes(self._request[0], self._request[1], self._results, self._images)
