"""What a turn holds of the apps' tools, and what a call of it passes: its app's bound, and the hold-back of a call
for news that arrived since the turn's admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from personas.decision import call_within_bound
from personas.present import arrived_since

if TYPE_CHECKING:
    from agents.member import PartialSpeech
    from agents.tools import HandTool, ToolDeclaration

    from personas.entries import History
    from personas.holds import HoldInbox
    from personas.present import Arrived


@dataclass
class Calls:
    """How many calls an app took in this turn, against its own bound."""

    bound: int
    made: int = 0

    def take(self) -> bool:
        if not call_within_bound(self.made, self.bound):
            return False
        self.made += 1
        return True

    def close(self) -> None:
        """The app takes no more calls in this turn."""
        self.made = self.bound


@dataclass
class HoldBack:
    """Whether a call of the turn is held back: once, when an entry not his own entered his history since the turn's
    admission, and again when people said words over him that no round told him."""

    admitted: History = field(repr=False)
    inbox: HoldInbox | None
    spent: bool = False

    def holding(self, current: History) -> Arrived | None:
        """What arrived for the first call that finds news; every later call of the turn is let through."""
        if self.spent:
            return None
        arrived = arrived_since(current, self.admitted)
        self.spent = arrived is not None
        return arrived

    def said_over_him(self) -> tuple[PartialSpeech, ...]:
        return () if self.inbox is None else self.inbox.untold()


@dataclass(frozen=True)
class HeldTools:
    """What a turn holds: the hand tools of every app that answered, and the calls the turn may make in all."""

    tools: tuple[HandTool, ...]
    max_tool_calls: int | None
    inbox: HoldInbox | None = None

    @property
    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(each.declaration for each in self.tools)
