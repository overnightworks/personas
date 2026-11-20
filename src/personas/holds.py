"""What a spoken turn hears of a hold of his while it speaks, and the two tools a round decides the hold with."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL
from agents.tools import ToolDeclaration, ToolEffect

from personas.speech import Held, HoldKey, HoldOutcome, hold_stated, hold_tellable, untold_words

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from agents.member import PartialSpeech, PlaceWindow
    from agents.tools import ToolOutcome
    from agents.turn_events import ToolCallEvent

    from personas.turn_bound import TurnBound

DECISION_CALLS: Final = 1
LINE_ARGUMENT: Final = "line"
APP_FAILED_TEXT: Final = "the app did not answer"
CONTINUED_TEXT: Final = '{"continued": true}'
GONE_TEXT: Final = '{"gone": true}'
_ANSWER_INSTEAD: Final = (
    " Or reply with text instead, which answers the hold: the text is what you say now, and everything you had not "
    "yet said is dropped."
)
_HOLD_ID: Final = {"id": {"type": "string", "description": "The id of the hold you were told."}}
_SHORT_LINE: Final = {
    LINE_ARGUMENT: {
        "type": "string",
        "minLength": 1,
        "description": (
            "One short line to say at once. The sentence you were held in is cut, not said again, and your next "
            "sentences follow the line."
        ),
    }
}
DECISION_DECLARATIONS: Final = tuple(
    ToolDeclaration(
        name=name,
        description=description + _ANSWER_INSTEAD,
        parameters={
            "type": "object",
            "properties": {**_HOLD_ID, **properties},
            "required": ["id"],
            "additionalProperties": False,
        },
        effect=ToolEffect.SIDE_EFFECTING,
    )
    for name, description, properties in (
        (
            CONTINUE_SPEAKING_TOOL,
            "Lets the sentence you were held in go on to its end, and your next sentences follow. With `line`, that "
            "one short line is said at once instead, the sentence you were held in is cut, and your next sentences "
            "follow the line.",
            _SHORT_LINE,
        ),
        (
            STOP_SPEAKING_TOOL,
            "Drops the rest of the sentence you were held in and everything you had not yet said.",
            {},
        ),
    )
)


class TurnHold:
    """The hold a turn was told at its start and has not decided yet: until he decides, its every other call is
    answered that he decides first, and the sentences wait unsent in the mind's record, where the hold stands."""

    def __init__(self, held: Held) -> None:
        self.held = held
        self.outcome: HoldOutcome | None = None
        self.say_line: Callable[[object], Awaitable[ToolOutcome]] | None = None
        """His short line before he goes on, set once the turn's speech exists."""

    @property
    def pending(self) -> bool:
        return self.outcome is None

    @property
    def silenced(self) -> bool:
        """His answer instead of this hold failed: the rest of its reply is dropped."""
        return self.outcome in {HoldOutcome.UNDECIDED, HoldOutcome.GONE}

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
        self._taken: HoldKey | None = standing.key if told and standing is not None else None
        self._decided: HoldKey | None = None
        self._answering = False
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
        self.hold = None if stated is not None and stated.key == self._decided else stated
        self._said = () if window.talked_over is None else window.talked_over.speech
        self._wake()

    def answering(self) -> None:
        """His answer instead of a hold is on its way: no round decides a hold from now on; one that stands is
        left for his next turn."""
        self._answering = True
        self._wake()

    def decided(self, key: HoldKey) -> None:
        """The hold ``key`` was decided: it stands no more, whatever a window read before the decision says."""
        self._decided = key
        if self.hold is not None and self.hold.key == key:
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
        return self.hold if self.hold is not None and self.hold.key != self._taken else None

    async def next_hold(self) -> Held | None:
        while not self._closed and not self._answering:
            hold = self.undecided
            if hold is not None and hold_tellable(hold):
                self._taken = hold.key
                return hold
            self._changed.clear()
            await self._changed.wait()
        return None
