"""What a turn holds of the apps' tools, and what a call of it passes: its app's bound and the hold-back of a call for
news that arrived since the turn's admission."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from personas.decision import call_within_bound
from personas.holds import deciding_first
from personas.present import arrived_since

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from agents.member import PartialSpeech
    from agents.tools import HandTool, ToolDeclaration

    from personas.entries import History
    from personas.holds import HoldInbox, TurnHold
    from personas.present import Arrived


@dataclass
class Calls:
    """How many calls an app took in this turn, against its own bound."""

    bound: int
    made: int = 0
    closed: bool = False

    def take(self) -> bool:
        if not call_within_bound(self.made, self.bound):
            return False
        self.made += 1
        return True

    def close(self) -> None:
        """The app takes no more calls in this turn."""
        self.made = self.bound
        self.closed = True


@dataclass
class HoldBack:
    """Whether a call of the turn is held back: once, when an entry not his own entered his history since the turn's
    admission, and again when people said words over him that no round told him."""

    admitted: History = field(repr=False)
    inbox: HoldInbox | None
    told: TurnHold | None = None
    spent: bool = False

    @property
    def deciding_first(self) -> bool:
        return deciding_first(self.told)

    def holding(self, current: History) -> Arrived | None:
        """What arrived for the first call that finds news; every later call of the turn is let through."""
        if self.spent:
            return None
        arrived = arrived_since(current, self.admitted)
        self.spent = arrived is not None
        return arrived

    def sending(self) -> AbstractContextManager[None]:
        """The call of the turn is on its way to its app until its answer."""
        return nullcontext() if self.inbox is None else self.inbox.sending()

    def said_over_him(self) -> tuple[PartialSpeech, ...]:
        return () if self.inbox is None else self.inbox.untold()


@dataclass(frozen=True)
class HeldTools:
    """What a turn holds: the hand tools of every app that answered, and the calls the turn may make in all."""

    tools: tuple[HandTool, ...]
    max_tool_calls: int | None
    inbox: HoldInbox | None = None
    told: TurnHold | None = None

    @property
    def deciding_first(self) -> bool:
        return deciding_first(self.told)

    @property
    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(each.declaration for each in self.tools)
