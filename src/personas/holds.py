"""What a spoken turn hears of a hold of his while it speaks, and the two tools a round decides the hold with."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL
from agents.tools import ToolDeclaration, ToolEffect

from personas.speech import Held, HoldOutcome, hold_stated, hold_tellable, untold_words

if TYPE_CHECKING:
    from collections.abc import Generator

    from agents.member import PartialSpeech, PlaceWindow
    from agents.turn_events import ToolCallEvent

    from personas.turn_bound import TurnBound

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
    """The hold a turn was told at its start and has not decided yet: until he decides, its every other call is
    answered that he decides first, and the sentences wait unsent in the mind's record, where the hold stands."""

    def __init__(self, held: Held) -> None:
        self.held = held
        self.outcome: HoldOutcome | None = None

    @property
    def pending(self) -> bool:
        return self.outcome is None

    def dropped(self) -> bool:
        """The turn's speech was cut before he decided this hold: it counts as undecided, once. Whether it was still
        pending."""
        if not self.pending:
            return False
        self.outcome = HoldOutcome.UNDECIDED
        return True


def deciding_first(told: TurnHold | None) -> bool:
    """Whether the turn was told a hold that he has not decided."""
    return told is not None and told.pending


async def close_when_over(inbox: HoldInbox, *speech: asyncio.Task[object]) -> None:
    await asyncio.wait(speech)
    inbox.close()


class HoldInbox:
    """What the window says of him while a spoken turn runs: the hold that stands, the call that waits for it or is on
    its way to its app, and what people say over him, with what a round told him of it."""

    def __init__(self, standing: Held | None = None, *, told: bool = False) -> None:
        self.hold = standing
        self.waiting: ToolCallEvent | None = None
        self.told: tuple[PartialSpeech, ...] | None = standing.speech if told and standing is not None else None
        self._said: tuple[PartialSpeech, ...] = () if standing is None else standing.speech
        self._taken: str | None = standing.id if told and standing is not None else None
        self._decided: str | None = None
        self._calls_sent = 0
        self._calls_answered = asyncio.Event()
        self._calls_answered.set()
        self._closed = False
        self._changed = asyncio.Event()
        self._freed = asyncio.Event()
        """Beside ``_changed`` because each of the two waits clears the event it waits on, which would swallow the
        other's wake-up."""

    def _wake(self) -> None:
        self._changed.set()
        self._freed.set()

    def hear(self, window: PlaceWindow) -> None:
        stated = hold_stated(window, self.hold)
        self.hold = None if stated is not None and stated.id == self._decided else stated
        self._said = () if window.talked_over is None else window.talked_over.speech
        self._wake()

    def decided(self, hold_id: str) -> None:
        """The hold ``hold_id`` was decided: it stands no more, whatever a window read before the decision says."""
        self._decided = hold_id
        if self.hold is not None and self.hold.id == hold_id:
            self.hold = None
        self._wake()

    async def until_none_stands(self, bound: TurnBound) -> None:
        """Returns once no hold stands; the turn's deadline ends the turn first when it passes before."""
        while self.hold is not None and not self._closed:
            self._freed.clear()
            await bound.first_of(self._freed, bound.deadline())
            bound.stop_if_deadline_done()

    @contextmanager
    def sending(self) -> Generator[None]:
        """A call of the turn is on its way to its app, until its answer has entered his history."""
        self._calls_sent += 1
        self._calls_answered.clear()
        try:
            yield
        finally:
            self._calls_sent -= 1
            if not self._calls_sent:
                self._calls_answered.set()

    async def calls_answered(self) -> None:
        """Returns once no call of the turn is on its way to its app."""
        await self._calls_answered.wait()

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
        self._wake()

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
