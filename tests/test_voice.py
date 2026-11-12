from __future__ import annotations

import asyncio
import secrets

import pytest
from agents.backend import TurnStopped
from agents.member import PlaceRef, Spoken
from agents.turn import FailureCode
from agents.turn_events import AssistantTextEvent, FinalEvent

from personas.voice import Speaking
from tests.support.fakes import FakeClock

pytestmark = pytest.mark.anyio

PLACE = PlaceRef(kind="call", id="room-7")
DEADLINE_SECONDS = 30
SENTENCE_MS = 20_000


async def renewed() -> None:
    return None


def test_a_speaking_shows_no_sentence_in_its_repr():
    marker = secrets.token_hex(8)
    speaking = Speaking(
        object(),
        "https://app.test/mcp",
        PLACE,
        FakeClock(),
        30,
        renewed,
        lambda event, code: None,
        lambda entries: None,
    )

    speaking.hear(AssistantTextEvent(text=f"{marker}. And one more sentence."))

    assert marker not in repr(speaking)


class SpeakingScenario:
    """A speaking whose sentences are answered as played for ``SENTENCE_MS``, and whose token renewal takes
    ``renewal_seconds`` and ends once ``renewal_gate`` is set."""

    def __init__(self, text: str, renewal_seconds: float = 0) -> None:
        self.clock = FakeClock()
        self.sent: list[str] = []
        self.entered: list[object] = []
        self.renewal_seconds = renewal_seconds
        self.renewal_gate = asyncio.Event()
        self.speaking = Speaking(
            self,
            "https://app.test/mcp",
            PLACE,
            self.clock,
            DEADLINE_SECONDS,
            self._renew,
            lambda event, code: None,
            self.entered.extend,
        )
        self.speaking.hear(AssistantTextEvent(text=text))
        self.speaking.hear(FinalEvent(text=text))
        self.speaking.end()

    async def _renew(self) -> None:
        await self.renewal_gate.wait()
        self.clock.time += self.renewal_seconds

    async def speak(self, place: PlaceRef, sentence: str) -> Spoken:
        self.sent.append(sentence)
        return Spoken(duration_ms=SENTENCE_MS)


async def test_a_final_sentence_whose_wait_crosses_the_deadline_ends_the_turn_limit_exceeded_after_both_are_sent():
    scenario = SpeakingScenario("One. Two.")
    scenario.renewal_gate.set()

    with pytest.raises(TurnStopped) as stopped:
        await scenario.speaking.play()

    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == ["One.", "Two."]
    assert len(scenario.entered) == 2
