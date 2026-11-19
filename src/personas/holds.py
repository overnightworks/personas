"""What a spoken turn hears of a hold of his while it speaks, and the two tools a round decides the hold with."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL
from agents.tools import ToolDeclaration, ToolEffect

from personas.speech import Held, HoldOutcome, hold_stated, hold_tellable, untold_words

if TYPE_CHECKING:
    from agents.member import PartialSpeech, PlaceWindow
    from agents.turn_events import ToolCallEvent

DECISION_CALLS: Final = 1
DECISION_DECLARATIONS: Final = tuple(
    ToolDeclaration(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string", "description": "The id of the hold you were told."}},
            "required": ["id"],
            "additionalProperties": False,
        },
        effect=ToolEffect.SIDE_EFFECTING,
    )
    for name, description in (
        (
            CONTINUE_SPEAKING_TOOL,
            "Lets the sentence you were held in go on to its end, and your next sentences follow.",
        ),
        (STOP_SPEAKING_TOOL, "Drops the rest of the sentence you were held in and everything you had not yet said."),
    )
)


class TurnHold:
    """The hold a turn was told at its start and has not decided yet: until he decides, the turn's sentences wait
    unsent and its every other call is answered that he decides first."""

    def __init__(self, held: Held) -> None:
        self.held = held
        self.outcome: HoldOutcome | None = None
        self.released = asyncio.Event()
        """Set when the turn's sentences may go out: after the decision, or when the speech is silenced."""

    @property
    def pending(self) -> bool:
        return self.outcome is None

    def decided(self) -> None:
        self.released.set()


def deciding_first(told: TurnHold | None) -> bool:
    """Whether the turn was told a hold that he has not decided."""
    return told is not None and told.pending


async def close_when_over(inbox: HoldInbox, *speech: asyncio.Task[object]) -> None:
    await asyncio.wait(speech)
    inbox.close()


class HoldInbox:
    """What the window says of him while a spoken turn runs: the hold that stands, the call that waits, and what people
    say over him, with what a round told him of it."""

    def __init__(self, standing: Held | None = None, *, told: bool = False) -> None:
        self.hold = standing
        self.waiting: ToolCallEvent | None = None
        self.told: tuple[PartialSpeech, ...] | None = standing.speech if told and standing is not None else None
        self._said: tuple[PartialSpeech, ...] = () if standing is None else standing.speech
        self._taken: str | None = standing.id if told and standing is not None else None
        self._closed = False
        self._changed = asyncio.Event()

    def hear(self, window: PlaceWindow) -> None:
        self.hold = hold_stated(window, self.hold)
        self._said = () if window.talked_over is None else window.talked_over.speech
        self._changed.set()

    def untold(self) -> tuple[PartialSpeech, ...]:
        """What people said over him since a round told him of it, once; nothing before a round has."""
        if self.told is None:
            return ()
        words = untold_words(self._said, self.told)
        if words:
            self.told = self._said
        return words

    def close(self) -> None:
        self._closed = True
        self._changed.set()

    @property
    def undecided(self) -> Held | None:
        return self.hold if self.hold is not None and self.hold.id != self._taken else None

    async def next_hold(self) -> Held | None:
        while not self._closed:
            hold = self.undecided
            if hold is not None and hold_tellable(hold):
                self._taken = hold.id
                return hold
            self._changed.clear()
            await self._changed.wait()
        return None
