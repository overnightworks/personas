from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime

import anyio
import pytest
from agents.member import PlaceRef, SpeakAnswer, SpeakRefusal, Spoken, Written
from agents.turn import FailureCode
from agents.turn_events import AssistantTextEvent, FinalEvent

from personas.config import AppTarget
from personas.decision import DEFAULT_BUDGET_PAUSE_CAP_SECONDS, TURN_CANCELLED_CODE, AppPlace
from personas.entries import KeptLine, Reading, SaidAloud, Unanswered, Visit
from personas.holds import HoldInbox
from personas.mind import Mind
from personas.reporting import Event, start_turn_tally
from personas.speech import DropReason, Held, HoldOutcome
from personas.turn_bound import TurnBound
from personas.voice import Answering, Sender, Speaking, Voicing
from tests.support.fakes import FakeClock

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

APP = "https://app.test/mcp"
AGENT = "mozart"
PLACE = PlaceRef(kind="call", id="room-7")
AT = AppPlace(APP, PLACE)
BOUND_SECONDS = 30
READ_AT = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


def a_line(*, own: bool, spoken: bool, place: PlaceRef = PLACE) -> KeptLine:
    return KeptLine(Visit(APP, place, "g1"), 1, READ_AT, "Mozart", "message", "x", own, spoken, False, act=False)


class GatedClock(FakeClock):
    """A fake clock whose sleeps wait until ``sleeps_end`` is set, then take their time off the clock."""

    def __init__(self) -> None:
        super().__init__()
        self.sleeps_end = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        await self.sleeps_end.wait()
        self.time += seconds


class Scenario:
    """A turn's speech and the sender of its place over one mind: the turn releases the text it is given, the sender
    sends it to a ``speak`` that takes ``speak_seconds`` and answers as started unless ``answers_lost`` is set, then,
    unless ``line_follows`` is off, his own spoken line for the sentence enters the history. The second sentence is
    answered ``second_answer`` when given. The sender's token renewal ends once ``renewal_gate`` is set."""

    def __init__(
        self,
        text: str,
        *,
        answer: SpeakAnswer | None = None,
        speak_seconds: float = 0,
        line_follows: bool = True,
        second_answer: SpeakAnswer | SpeakRefusal | None = None,
        answers_lost: bool = False,
        renewal_seconds: float = 0,
    ) -> None:
        self.answer = Spoken(spoken=True) if answer is None else answer
        self._second_answer = second_answer
        self._answers_lost = answers_lost
        self._speak_seconds = speak_seconds
        self._line_follows = line_follows
        self._renewal_seconds = renewal_seconds
        self.clock = GatedClock()
        self.mind = Mind(10_000, 10_000, DEFAULT_BUDGET_PAUSE_CAP_SECONDS)
        self.tally = start_turn_tally(AGENT, APP)
        self.sent: list[str] = []
        self.reported: list[tuple[Event, str | None]] = []
        self.dropped: list[tuple[DropReason, int]] = []
        self.renewal_gate = asyncio.Event()
        self.renewal_started = asyncio.Event()
        self.bound = TurnBound(self.clock, BOUND_SECONDS)
        self.speaking = Speaking(
            APP,
            PLACE,
            self.bound,
            self.mind,
            self.tally,
            lambda reason, count: self.report_dropped(reason, count),
            told=None,
            replacing=False,
        )
        hear_whole(self.speaking, text)
        self.sender = Sender(
            self,
            APP,
            PLACE,
            self.clock,
            self._renew,
            lambda event, code: self.reported.append((event, code)),
            lambda reason, count: self.report_dropped(reason, count),
            self.mind,
        )

    def report_dropped(self, reason: DropReason, count: int) -> None:
        if count:
            self.dropped.append((reason, count))

    @property
    def entered(self) -> list[SaidAloud]:
        return [entry for entry in self.mind.history.entries if isinstance(entry, SaidAloud)]

    def his_spoken_line_enters(self) -> None:
        self.mind.take((a_line(own=True, spoken=True),))

    async def _renew(self) -> None:
        self.renewal_started.set()
        await self.renewal_gate.wait()
        self.clock.time += self._renewal_seconds

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

    async def answered_instead(self) -> HoldOutcome:
        """His reply answers a hold instead; the sentence goes to the app at once, as the answer of the hold."""
        voicing = Voicing(
            _Sessions(self),
            AppTarget(APP, "scope"),
            lambda *_arguments: None,
            self.clock,
            self.mind,
            lambda reason, count: self.report_dropped(reason, count),
        )
        answering = Answering(
            voicing,
            PLACE,
            None,
            self.tally,
            self.bound,
            HoldInbox(),
            None,
            lambda *_ended: None,
            _undecided,
        )
        return await answering.instead(Held("hold-1", "g1", (), False), ("One.",))

    async def sending(self) -> asyncio.Task[None]:
        """Starts the sender and returns once it has nothing left to do but wait."""
        self.renewal_gate.set()
        sending = asyncio.create_task(self.sender.run())
        await anyio.wait_all_tasks_blocked()
        return sending


class _Sessions:
    """Asks the scenario's ``speak`` as the one session there is."""

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario

    async def ask_elsewhere(self, _app, ask, _record):
        return await ask(self)

    async def speak(self, place, text, _style, *, interrupted):
        return await self._scenario.speak(place, text)


async def _undecided(_hold):
    return None


def hear_whole(speaking: Speaking, text: str) -> None:
    speaking.hear(AssistantTextEvent(text=text))
    speaking.hear(FinalEvent(text=text))


async def ended(sending: asyncio.Task[None]) -> None:
    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending


def test_a_speaking_shows_no_sentence_in_its_repr():
    marker = secrets.token_hex(8)
    scenario = Scenario(f"{marker}. And one more sentence.")

    assert marker not in repr(scenario.speaking)


def test_the_sentences_a_round_completes_are_released_and_counted_and_sent_by_nobody():
    scenario = Scenario("One. Two. Three.")

    assert (scenario.tally.sentence_count, scenario.sent) == (3, [])
    assert scenario.mind.speech_unsaid(AT) == ("One.", "Two.", "Three.")


def test_a_round_that_opens_with_the_silent_marker_is_counted_and_releases_no_sentence():
    """TURNS-26, LOG-29."""
    scenario = Scenario("[silent]")

    assert (scenario.tally.silent_count, scenario.tally.sentence_count, scenario.mind.speech_unsaid(AT)) == (1, 0, ())


def test_a_stopped_turn_releases_no_sentence_from_then_on():
    scenario = Scenario("")

    scenario.speaking.cut()
    hear_whole(scenario.speaking, "One.")

    assert (scenario.tally.sentence_count, scenario.mind.speech_unsaid(AT)) == (0, ())


@pytest.mark.parametrize(("replacing", "unsent"), [(True, ("New.",)), (False, ("Old.", "New."))])
def test_the_first_sentence_of_a_turn_told_no_hold_replaces_the_unsent_rest_of_the_turns_before(replacing, unsent):
    """TURNS-33."""
    scenario = Scenario("Old.")
    later = Speaking(
        APP,
        PLACE,
        scenario.bound,
        scenario.mind,
        scenario.tally,
        scenario.report_dropped,
        told=None,
        replacing=replacing,
    )

    hear_whole(later, "New.")

    assert scenario.mind.speech_unsaid(AT) == unsent
    assert scenario.dropped == ([(DropReason.REPLACED, 1)] if replacing else [])


async def test_the_sender_sends_each_sentence_after_the_answer_of_the_one_before_and_enters_it_answered():
    """TURNS-25."""
    scenario = Scenario("One. Two.")

    sending = await scenario.sending()

    assert scenario.sent == ["One.", "Two."]
    assert [entry.outcome for entry in scenario.entered] == [scenario.answer, scenario.answer]
    assert scenario.reported == []
    await ended(sending)


VISITS = pytest.mark.parametrize(
    ("checkpoint", "generation"), [(Reading("generation-5", 1), "generation-5"), (None, None)], ids=["read", "unread"]
)


@VISITS
async def test_a_sentence_is_entered_with_the_visit_of_the_places_last_read(checkpoint, generation):
    scenario = Scenario("One.")
    scenario.mind.move_checkpoint(AT, checkpoint)

    sending = await scenario.sending()

    assert [entry.generation for entry in scenario.entered] == [generation]
    await ended(sending)


@VISITS
async def test_a_sentence_answering_a_hold_is_entered_with_the_visit_of_the_places_last_read(checkpoint, generation):
    scenario = Scenario("One.", line_follows=False)
    scenario.mind.move_checkpoint(AT, checkpoint)

    await scenario.answered_instead()

    assert [entry.generation for entry in scenario.entered] == [generation]


async def test_the_sentence_last_answered_plays_until_his_line_enters_and_is_reported_by_nothing():
    """TURNS-25."""
    scenario = Scenario("One.", line_follows=False)

    sending = await scenario.sending()

    assert scenario.reported == []
    assert scenario.mind.speech_saying(AT) == "One."
    await ended(sending)


async def test_sentences_the_app_wrote_instead_of_voicing_are_each_sent_and_entered_with_no_wait_between_them():
    scenario = Scenario("One. Two. Three.", answer=Written(written=True), line_follows=False)

    sending = await scenario.sending()

    assert scenario.sent == ["One.", "Two.", "Three."]
    assert len(scenario.entered) == 3
    await ended(sending)


async def test_a_sentence_refused_is_reported_and_enters_and_the_unsent_after_it_are_never_sent():
    scenario = Scenario("One. Two. Three.", line_follows=False, second_answer=SpeakRefusal.STOPPED)

    sending = await scenario.sending()
    scenario.his_spoken_line_enters()
    await anyio.wait_all_tasks_blocked()

    assert scenario.sent == ["One.", "Two."]
    assert scenario.reported == [(Event.SPEAK_REFUSED, SpeakRefusal.STOPPED.value)]
    assert scenario.mind.speech_unsaid(AT) == ()
    await ended(sending)


async def test_a_sentence_unsent_when_its_bound_is_reached_is_dropped_and_not_sent():
    """TURNS-28."""
    scenario = Scenario("One. Two.", speak_seconds=BOUND_SECONDS + 1)

    sending = await scenario.sending()

    assert scenario.sent == ["One."]
    assert scenario.dropped == [(DropReason.LIMIT_EXCEEDED, 1)]
    await ended(sending)


async def test_a_sentence_whose_token_renewal_ends_past_its_bound_is_dropped_and_not_sent():
    """TURNS-28."""
    scenario = Scenario("One.", renewal_seconds=BOUND_SECONDS + 1)

    sending = await scenario.sending()

    assert (scenario.sent, scenario.entered) == ([], [])
    assert scenario.dropped == [(DropReason.LIMIT_EXCEEDED, 1)]
    await ended(sending)


async def test_a_request_not_answered_by_its_bound_enters_once_as_unanswered_and_the_sender_goes_on():
    """TURNS-28."""
    scenario = Scenario("One.", answers_lost=True)

    sending = await scenario.sending()
    scenario.clock.sleeps_end.set()
    await asyncio.sleep(0)
    await anyio.wait_all_tasks_blocked()

    assert [entry.outcome for entry in scenario.entered] == [Unanswered(FailureCode.LIMIT_EXCEEDED.value)]
    assert scenario.dropped == [(DropReason.LIMIT_EXCEEDED, 1)]
    assert not sending.done()
    await ended(sending)


@pytest.mark.parametrize(("stopping", "reason"), [(False, DropReason.FAILED), (True, DropReason.STOPPED)])
async def test_a_request_open_when_the_sender_is_cancelled_enters_once_as_unanswered_and_the_cancel_goes_on(
    stopping, reason
):
    scenario = Scenario("One. Two.", answers_lost=True)
    if stopping:
        scenario.mind.stopping()

    sending = await scenario.sending()
    await ended(sending)

    assert [entry.outcome for entry in scenario.entered] == [Unanswered(TURN_CANCELLED_CODE)]
    assert scenario.dropped == [(reason, 1)]
