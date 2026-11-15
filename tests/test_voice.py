from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime

import anyio
import pytest
from agents.backend import TurnStopped
from agents.member import PlaceRef, SpeakAnswer, SpeakRefusal, Spoken, Written
from agents.turn import FailureCode
from agents.turn_events import AssistantTextEvent, FinalEvent

from personas.decision import DEFAULT_BUDGET_PAUSE_CAP_SECONDS, TURN_CANCELLED_CODE
from personas.entries import KeptLine, SaidAloud, Unanswered, Visit
from personas.mind import Mind
from personas.reporting import Event, start_turn_tally
from personas.voice import Speaking
from tests.support.fakes import FakeClock
from tests.support.voice import HeldSleeps

pytestmark = pytest.mark.anyio

APP = "https://app.test/mcp"
AGENT = "mozart"
PLACE = PlaceRef(kind="call", id="room-7")
DEADLINE_SECONDS = 30
READ_AT = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


async def renewed() -> None:
    return None


def a_line(*, own: bool, spoken: bool, place: PlaceRef = PLACE) -> KeptLine:
    return KeptLine(Visit(APP, place, "g1"), 1, READ_AT, "Mozart", "message", "x", own, spoken, False, act=False)


def test_a_speaking_shows_no_sentence_in_its_repr():
    marker = secrets.token_hex(8)
    speaking = Speaking(
        object(),
        APP,
        PLACE,
        FakeClock(),
        30,
        renewed,
        lambda event, code: None,
        Mind(1_000, 1_000, DEFAULT_BUDGET_PAUSE_CAP_SECONDS),
        start_turn_tally(AGENT, APP),
    )

    speaking.hear(AssistantTextEvent(text=f"{marker}. And one more sentence."))

    assert marker not in repr(speaking)


class SpeakingScenario:
    """A speaking whose sentences are answered as started, and whose token renewal takes ``renewal_seconds`` and ends
    once ``renewal_gate`` is set. Each ``speak`` takes ``speak_seconds``, then, unless ``line_follows`` is off, his own
    spoken line for the sentence enters the mind's history; ``renewal_started`` is set when a renewal begins. The second
    sentence is answered ``second_answer`` when given, and a ``speak`` never answers while ``answers_lost`` is set.
    Unless ``ends`` is off, the reply has ended."""

    def __init__(
        self,
        text: str,
        *,
        renewal_seconds: float = 0,
        answer: SpeakAnswer | None = None,
        speak_seconds: float = 0,
        line_follows: bool = True,
        second_answer: SpeakAnswer | SpeakRefusal | None = None,
        answers_lost: bool = False,
        ends: bool = True,
    ) -> None:
        self.answer = Spoken(spoken=True) if answer is None else answer
        self._second_answer = second_answer
        self._answers_lost = answers_lost
        self.clock = FakeClock()
        self.mind = Mind(10_000, 10_000, DEFAULT_BUDGET_PAUSE_CAP_SECONDS)
        self.tally = start_turn_tally(AGENT, APP)
        self.sent: list[str] = []
        self.reported: list[tuple[object, str | None]] = []
        self.renewal_seconds = renewal_seconds
        self.renewal_gate = asyncio.Event()
        self.renewal_started = asyncio.Event()
        self._speak_seconds = speak_seconds
        self._line_follows = line_follows
        self.speaking = Speaking(
            self,
            APP,
            PLACE,
            self.clock,
            DEADLINE_SECONDS,
            self._renew,
            lambda event, code: self.reported.append((event, code)),
            self.mind,
            self.tally,
        )
        self.speaking.hear(AssistantTextEvent(text=text))
        self.speaking.hear(FinalEvent(text=text))
        if ends:
            self.speaking.end()

    @property
    def entered(self) -> list[SaidAloud]:
        return [entry for entry in self.mind.history.entries if isinstance(entry, SaidAloud)]

    def his_spoken_line_enters(self) -> None:
        self.mind.take((a_line(own=True, spoken=True),))

    async def _renew(self) -> None:
        self.renewal_started.set()
        await self.renewal_gate.wait()
        self.clock.time += self.renewal_seconds

    async def speak(self, place: PlaceRef, sentence: str) -> SpeakAnswer | SpeakRefusal:
        self.sent.append(sentence)
        if self._answers_lost:
            await asyncio.Event().wait()
        self.clock.time += self._speak_seconds
        if len(self.sent) == 2 and self._second_answer is not None:
            return self._second_answer
        if self._line_follows:
            self.his_spoken_line_enters()
        return self.answer


async def test_the_wait_for_his_line_ends_the_sentence_without_waiting_out_the_deadline_and_reports_nothing():
    scenario = SpeakingScenario("One.", line_follows=False, speak_seconds=10)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()

    scenario.his_spoken_line_enters()
    await playing

    assert sleeps.cancelled.is_set()
    assert sleeps.began == [DEADLINE_SECONDS]
    assert scenario.reported == []


async def test_what_enters_that_is_not_his_own_spoken_line_at_the_place_does_not_end_the_wait_or_renew_its_bound():
    scenario = SpeakingScenario("One.", line_follows=False, speak_seconds=10)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()

    for other in (
        a_line(own=False, spoken=True),
        a_line(own=True, spoken=False),
        a_line(own=True, spoken=True, place=PlaceRef(kind="call", id="room-9")),
    ):
        scenario.mind.take((other,))
        await anyio.wait_all_tasks_blocked()
    assert playing.done() is False
    assert sleeps.began == [DEADLINE_SECONDS]

    scenario.his_spoken_line_enters()
    await playing


async def test_a_deadline_that_passes_in_the_step_his_line_enters_in_ends_the_turn():
    scenario = SpeakingScenario("One. Two.", line_follows=False)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()

    sleeps.release.set()
    scenario.his_spoken_line_enters()
    scenario.his_spoken_line_enters()

    with pytest.raises(TurnStopped) as stopped:
        await playing
    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == ["One.", "Two."]


async def test_a_sentence_answered_after_the_deadline_ends_the_turn_at_once_without_a_wait_though_his_line_entered():
    scenario = SpeakingScenario("One. Two.", speak_seconds=DEADLINE_SECONDS + 1)
    scenario.renewal_gate.set()
    HeldSleeps(scenario)

    with pytest.raises(TurnStopped) as stopped:
        await scenario.speaking.play()

    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.sent == ["One."]
    assert len(scenario.entered) == 1


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
    HeldSleeps(scenario)

    await scenario.speaking.play()

    assert scenario.sent == ["One.", "Two.", "Three."]
    assert len(scenario.entered) == 3


async def test_a_refusal_ends_the_speech_but_the_end_of_play_still_awaits_the_line_of_the_sentence_playing():
    scenario = SpeakingScenario("One. Two.", line_follows=False, second_answer=SpeakRefusal.STOPPED)
    scenario.renewal_gate.set()
    HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()

    assert scenario.sent == ["One.", "Two."]
    assert scenario.reported == [(Event.SPEAK_REFUSED, SpeakRefusal.STOPPED.value)]
    assert not playing.done()
    scenario.his_spoken_line_enters()
    await playing


async def test_a_refusal_ends_the_speech_but_a_call_still_waits_for_the_line_of_the_sentence_playing():
    scenario = SpeakingScenario("One. Two.", line_follows=False, second_answer=SpeakRefusal.STOPPED, ends=False)
    scenario.renewal_gate.set()
    HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    waiting = asyncio.create_task(scenario.speaking.spoken())
    await anyio.wait_all_tasks_blocked()

    assert scenario.sent == ["One.", "Two."]
    assert not waiting.done()
    scenario.his_spoken_line_enters()
    await waiting
    scenario.speaking.end()
    await playing


async def test_every_sentence_sent_is_counted_the_refused_one_included_and_a_sentence_after_the_refusal_is_not():
    scenario = SpeakingScenario("One. Two. Three.", line_follows=False, second_answer=SpeakRefusal.STOPPED, ends=False)
    scenario.renewal_gate.set()
    HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()
    scenario.his_spoken_line_enters()
    scenario.speaking.end()
    await playing

    assert (scenario.sent, scenario.tally.sentence_count) == (["One.", "Two."], 2)


async def test_a_round_that_opens_with_the_silent_marker_is_counted_and_sends_no_sentence():
    scenario = SpeakingScenario("[silent]")
    scenario.renewal_gate.set()
    await scenario.speaking.play()

    assert (scenario.tally.silent_count, scenario.tally.sentence_count, scenario.sent) == (1, 0, [])


async def test_a_request_not_answered_by_the_deadline_enters_once_as_unanswered_and_ends_the_turn_limit_exceeded():
    scenario = SpeakingScenario("One.", answers_lost=True)
    scenario.renewal_gate.set()
    sleeps = HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()

    sleeps.release.set()

    with pytest.raises(TurnStopped) as stopped:
        await playing
    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert [entry.outcome for entry in scenario.entered] == [Unanswered(FailureCode.LIMIT_EXCEEDED.value)]


async def test_a_request_open_when_the_turn_is_cancelled_enters_once_as_unanswered_and_the_cancel_goes_on():
    scenario = SpeakingScenario("One.", answers_lost=True)
    scenario.renewal_gate.set()
    HeldSleeps(scenario)
    playing = asyncio.create_task(scenario.speaking.play())
    await anyio.wait_all_tasks_blocked()

    playing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await playing
    assert [entry.outcome for entry in scenario.entered] == [Unanswered(TURN_CANCELLED_CODE)]
