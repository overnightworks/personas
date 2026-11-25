"""What a turn holds of the apps' tools, and what a call of it passes: its app's bound and the hold-back of a call for
news that arrived since the turn's admission."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from agents.tools import ToolOutcome

from personas.decision import call_within_bound
from personas.holds import DECIDING_FIRST_TEXT, deciding_first
from personas.present import NO_SUCH_SKILL_TEXT, arrived_since, listed_skill_named, tools_to_load

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from contextlib import AbstractContextManager

    from agents.member import PartialSpeech
    from agents.tools import HandTool, ToolDeclaration
    from agents.turn import JsonObject, Principal

    from personas.entries import History
    from personas.holds import HoldInbox, TurnHold
    from personas.present import AppOffer, Arrived, Deferred, ListedSkill

LOADS_SPENT_TEXT: Final = "no more tools can be loaded in this turn"
NOTHING_WAITS_TEXT: Final = "no tool of this skill waits to be loaded"


@dataclass
class Calls:
    """How many calls an app took in this turn, against its own bound."""

    bound: int
    made: int = 0
    closed: bool = False

    def take(self) -> bool:
        if self.closed or not call_within_bound(self.made, self.bound):
            return False
        self.made += 1
        return True

    def close(self) -> None:
        """The app takes no more calls in this turn."""
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
class TurnOffer:
    """What the listings of a turn's apps give it: the skills they list, the tools they offer and the tools of those
    that are sent deferred."""

    skills: Mapping[str, ListedSkill]
    offers: tuple[AppOffer, ...]
    deferred: Deferred


@dataclass
class ToolLoads:
    """The loads of a turn: the skills it lists, the tools each app sends deferred and the ones already loaded, whether
    a round he was told or an earlier load of the turn loaded them, against the bound of its loads. ``enter`` keeps a
    load in his history, with the skill's app and name."""

    skills: Mapping[str, ListedSkill]
    deferred: Deferred
    loaded: set[str]
    calls: Calls
    hold: HoldBack
    enter: Callable[[str, str, tuple[str, ...]], None]

    async def load(self, _principal: Principal, arguments: JsonObject) -> ToolOutcome:  # NOSONAR(S7503) async handler
        """Loads the tools of a skill this turn listed that its app sends deferred and that nothing loaded yet, and
        enters the load into his history. It sends no request and is never held back by what arrived, only by a hold he
        has not decided. A name no line holds and a call past the bound answer an error; a skill with nothing waiting
        answers so, and enters nothing."""
        if self.hold.deciding_first:
            return ToolOutcome(DECIDING_FIRST_TEXT, is_error=True)
        if not self.calls.take():
            return ToolOutcome(LOADS_SPENT_TEXT, is_error=True)
        listed = listed_skill_named(self.skills, arguments)
        if listed is None:
            return ToolOutcome(NO_SUCH_SKILL_TEXT, is_error=True)
        tools = tools_to_load(listed.skill, self.deferred.get(listed.app, ()), self.loaded)
        if not tools:
            return ToolOutcome(NOTHING_WAITS_TEXT, is_error=False)
        self.loaded.update(tools)
        self.enter(listed.app, listed.skill.name, tools)
        return ToolOutcome("", is_error=False, loads=tools)


@dataclass(frozen=True)
class HeldTools:
    """What a turn holds: the hand tools of every app that answered, the calls the turn may make in all and the tools
    each app sends deferred."""

    tools: tuple[HandTool, ...]
    max_tool_calls: int | None
    inbox: HoldInbox | None = None
    told: TurnHold | None = None
    deferred: Deferred = field(default_factory=dict[str, tuple[str, ...]])

    @property
    def deferred_names(self) -> frozenset[str]:
        return frozenset(name for names in self.deferred.values() for name in names)

    @property
    def deciding_first(self) -> bool:
        return deciding_first(self.told)

    @property
    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(each.declaration for each in self.tools)
