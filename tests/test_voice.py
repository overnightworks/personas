from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime

import anyio
import pytest
from agents.backend import TurnStopped
from agents.member import PlaceRef, SpeakAnswer, Spoken, Written
from agents.turn import FailureCode
from agents.turn_events import AssistantTextEvent, FinalEvent

from personas.entries import KeptLine, SaidAloud, Visit
from personas.mind import Mind
from personas.voice import Speaking
from tests.support.fakes import FakeClock
from tests.support.voice import HeldSleeps

pytestmark = pytest.mark.anyio

APP = "https://app.test/mcp"
PLACE = PlaceRef(kind="call", id="room-7")
DEADLINE_SECONDS = 30
READ_AT = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


async def renewed() -> None:
    return None


def a_line(*, own: bool, spoken: bool, cut: bool = False, place: PlaceRef = PLACE) -> KeptLine:
    return KeptLine(Visit(APP, place, "g1"), 1, READ_AT, "Mozart", "message", "x", own, spoken, cut, act=False)


def test_a_speaking_shows_no_sentence_in_its_repr():
    marker = secrets.token_hex(8)
    speaking = Speaking(object(), APP, PLACE, FakeClock(), 30, renewed, lambda event, code: None, Mind(1_000))

    speaking.hear(AssistantTextEvent(text=f"{marker}. And one more sentence."))

    assert marker not in repr(speaking)


class SpeakingScenario:
    """A speaking whose sentences are answered as started, and whose token renewal takes ``renewal_seconds`` and ends
    once ``renewal_gate`` is set. Each ``speak`` takes ``speak_seconds``, then, unless ``line_follows`` is off, his own
    spoken line for the sentence enters the mind's history; ``renewal_started`` is set when a renewal begins."""

    def __init__(
        self,
        text: str,
        *,
        renewal_seconds: float = 0,
        answer: SpeakAnswer | None = None,
        speak_seconds: float = 0,
        line_follows: bool = True,
        cut: bool = False,
    ) -> None:
        self.answer = Spoken(spoken=True) if answer is None else answer
        self.clock = FakeClock()
        self.mind = Mind(10_000)
        self.sent: list[str] = []
        self.reported: list[tuple[object, str | None]] = []
        self.renewal_seconds = renewal_seconds
        self.renewal_gate = asyncio.Event()
        self.renewal_started = asyncio.Event()
        self._speak_seconds = speak_seconds
        self._line_follows = line_follows
        self._cut = cut
        self.speaking = Speaking(
            self,
            APP,
            PLACE,
            self.clock,
            DEADLINE_SECONDS,
            self._renew,
            lambda event, code: self.reported.append((event, code)),
            self.mind,
        )
        self.speaking.hear(AssistantTextEvent(text=text))
        self.speaking.hear(FinalEvent(text=text))
        self.speaking.end()

    @property
    def entered(self) -> list[SaidAloud]:
        return [entry for entry in self.mind.history.entries if isinstance(entry, SaidAloud)]

    def his_spoken_line_enters(self, *, cut: bool = False) -> None:
        self.mind.take((a_line(own=True, spoken=True, cut=cut),))

    async def _renew(self) -> None:
        self.renewal_started.set()
        await self.renewal_gate.wait()
        self.clock.time += self.renewal_seconds

    async def speak(self, place: PlaceRef, sentence: str) -> SpeakAnswer:
        self.sent.append(sentence)
        self.clock.time += self._speak_seconds
        if self._line_follows:
            self.his_spoken_line_enters(cut=self._cut)
        return self.answer


@pytest.mark.parametrize("cut", [False, True], ids=["whole", "cut"])
async def test_each_sentence_is_sent_once_his_own_line_for_the_one_before_has_entered_whole_or_cut(cut):
    scenario = SpeakingScenario("One. Two. Three.", cut=cut)
    scenario.renewal_gate.set()

    await scenario.speaking.play()

    assert scenario.sent == ["One.", "Two.", "Three."]
    assert len(scenario.entered) == 3
    assert scenario.reported == []


async def test_a_sentence_waits_for_his_line_and_the_next_is_sent_only_after_it_entered():
    scenario = SpeakingScenario("One. Two.", line_follows=False)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await sleeps.until_began(1)
    await anyio.wait_all_tasks_blocked()

    assert scenario.sent == ["One."]

    scenario.his_spoken_line_enters()
    await sleeps.until_cancelled()
    await anyio.wait_all_tasks_blocked()
    scenario.his_spoken_line_enters()
    await playing

    assert scenario.sent == ["One.", "Two."]


async def test_the_wait_for_his_line_ends_the_sentence_without_waiting_out_the_deadline_and_reports_nothing():
    scenario = SpeakingScenario("One.", line_follows=False, speak_seconds=10)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await sleeps.until_began(1)

    scenario.his_spoken_line_enters()
    await playing

    assert sleeps.cancelled.is_set()
    assert sleeps.began == [DEADLINE_SECONDS - 10]
    assert scenario.reported == []


async def test_what_enters_that_is_not_his_own_spoken_line_at_the_place_does_not_end_the_wait_or_renew_its_bound():
    scenario = SpeakingScenario("One.", line_follows=False, speak_seconds=10)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await sleeps.until_began(1)

    for other in (
        a_line(own=False, spoken=True),
        a_line(own=True, spoken=False),
        a_line(own=True, spoken=True, place=PlaceRef(kind="call", id="room-9")),
    ):
        scenario.mind.take((other,))
        await anyio.wait_all_tasks_blocked()
    assert playing.done() is False
    assert sleeps.began == [DEADLINE_SECONDS - 10]

    scenario.his_spoken_line_enters()
    await playing


async def test_a_line_missing_at_the_deadline_ends_the_turn_limit_exceeded_with_nothing_more_sent():
    scenario = SpeakingScenario("One. Two.", line_follows=False)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await sleeps.until_began(1)

    sleeps.release.set()

    with pytest.raises(TurnStopped) as stopped:
        await playing
    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == ["One."]
    assert len(scenario.entered) == 1


async def test_a_deadline_that_passes_in_the_step_his_line_enters_in_ends_the_turn():
    scenario = SpeakingScenario("One. Two.", line_follows=False)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await sleeps.until_began(1)

    sleeps.release.set()
    scenario.his_spoken_line_enters()

    with pytest.raises(TurnStopped) as stopped:
        await playing
    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == ["One."]


async def test_a_sentence_answered_after_the_deadline_ends_the_turn_at_once_without_a_wait_though_his_line_entered():
    scenario = SpeakingScenario("One. Two.", speak_seconds=DEADLINE_SECONDS + 1)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)

    with pytest.raises(TurnStopped) as stopped:
        await scenario.speaking.play()

    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == ["One."]
    assert len(scenario.entered) == 1
    assert sleeps.began == []


async def test_a_sentence_whose_token_renewal_ends_past_the_deadline_is_not_sent_and_ends_the_turn_limit_exceeded():
    scenario = SpeakingScenario("One.", renewal_seconds=DEADLINE_SECONDS + 1)
    scenario.renewal_gate.set()

    with pytest.raises(TurnStopped) as stopped:
        await scenario.speaking.play()

    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == []
    assert scenario.entered == []


async def test_a_sentence_whose_token_renewal_ends_after_the_events_failed_is_not_sent_and_is_not_entered():
    scenario = SpeakingScenario("One.")
    playing = asyncio.create_task(scenario.speaking.play())
    await scenario.renewal_started.wait()
    settling = asyncio.create_task(scenario.speaking.settle())
    await anyio.wait_all_tasks_blocked()

    scenario.renewal_gate.set()
    await asyncio.gather(playing, settling)

    assert scenario.sent == []
    assert scenario.entered == []


async def test_sentences_the_app_wrote_instead_of_voicing_are_each_sent_and_entered_with_no_wait_between_them():
    scenario = SpeakingScenario("One. Two. Three.", answer=Written(written=True), line_follows=False)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)

    await scenario.speaking.play()

    assert scenario.sent == ["One.", "Two.", "Three."]
    assert len(scenario.entered) == 3
    assert sleeps.began == []
