from __future__ import annotations

import pytest
from agents.member import Interrupted, PartialSpeech, PlaceWindow, TalkedOver

from personas.speech import (
    Held,
    SpeechRecord,
    Telling,
    hold_decided,
    hold_marked_told,
    hold_read,
    hold_stated,
    hold_tellable,
    hold_turn_ended,
    hold_untold,
    may_send_next,
    speech_answered,
    speech_ended,
    speech_line_entered,
    speech_sent,
    untold_words,
)


def _spoken(record: SpeechRecord) -> SpeechRecord:
    return speech_answered(record, "One.", spoken=True)


def _written(record: SpeechRecord) -> SpeechRecord:
    return speech_answered(record, "One.", spoken=False)


@pytest.mark.parametrize(
    ("steps", "owed", "saying"),
    [
        pytest.param((), 0, None, id="nothing sent"),
        pytest.param((speech_line_entered,), 0, None, id="a line with no request and nothing owed is ignored"),
        pytest.param((speech_sent, _spoken), 1, "One.", id="a sentence answered as spoken owes its line"),
        pytest.param((speech_sent, _spoken, speech_line_entered), 0, "One.", id="the line brings the debt in"),
        pytest.param(
            (speech_sent, speech_line_entered, _spoken), 0, "One.", id="a line before its answer leaves nothing owed"
        ),
        pytest.param(
            (speech_sent, speech_line_entered, _written), 0, None, id="a line before a non-spoken answer is no debt"
        ),
        pytest.param((speech_sent, _written), 0, None, id="a sentence answered otherwise owes nothing"),
        pytest.param(
            (speech_sent, _spoken, speech_line_entered, speech_line_entered),
            0,
            "One.",
            id="a line owed to none is ignored",
        ),
        pytest.param((speech_sent, _spoken, speech_sent, _spoken), 2, "One.", id="two sentences owe two lines"),
    ],
)
def test_the_speech_record_counts_a_request_from_its_send_and_credits_his_line_only_while_one_is_open_or_owed(
    steps, owed, saying
):
    record = SpeechRecord()
    for step in steps:
        record = step(record)

    assert (record.owed, record.saying) == (owed, saying)


@pytest.mark.parametrize(
    ("owed", "may_send"),
    [(0, True), (1, True), (2, False), (3, False)],
    ids=["nothing owed", "the one playing is owed", "one playing and one before it not yet entered", "more"],
)
def test_the_next_sentence_goes_out_while_at_most_the_one_playing_is_owed_a_line(owed, may_send):
    assert may_send_next(owed) is may_send


def _window(*, interrupted: str | None, speech: tuple[PartialSpeech, ...] | None) -> PlaceWindow:
    talked_over = None if speech is None else TalkedOver(id="talk", playing=True, speech=speech)
    return PlaceWindow(
        generation="g",
        starts_at=1,
        entries=(),
        talked_over=talked_over,
        interrupted=None if interrupted is None else Interrupted(id=interrupted),
    )


ADA_SAYS = (PartialSpeech(name="Ada", text="wait"),)
ADA_STARTED = (PartialSpeech(name="Ada", text=""),)


@pytest.mark.parametrize(
    ("window", "seen", "stated"),
    [
        (_window(interrupted=None, speech=ADA_SAYS), None, None),
        (_window(interrupted="h1", speech=ADA_SAYS), None, Held("h1", ADA_SAYS, record_gone=False)),
        (_window(interrupted="h1", speech=None), Held("h1", ADA_SAYS, record_gone=False), Held("h1", ADA_SAYS, True)),
        (_window(interrupted="h2", speech=None), Held("h1", ADA_SAYS, record_gone=False), Held("h2", (), True)),
    ],
    ids=["not held", "held", "record gone keeps the words of the same hold", "record gone of a new hold has none"],
)
def test_a_window_states_the_hold_with_what_was_said_over_him(window, seen, stated):
    assert hold_stated(window, seen) == stated


@pytest.mark.parametrize(
    ("held", "tellable"),
    [
        (Held("h", ADA_STARTED, record_gone=False), False),
        (Held("h", ADA_SAYS, record_gone=False), True),
        (Held("h", ADA_STARTED, record_gone=True), True),
        (Held("h", (), record_gone=True), True),
    ],
    ids=["a person started and said no words", "words", "record gone", "record gone, no one said anything"],
)
def test_a_hold_may_be_told_once_someone_has_said_words_or_the_record_is_gone(held, tellable):
    assert hold_tellable(held) is tellable


@pytest.mark.parametrize(
    ("said", "told", "untold"),
    [
        (ADA_SAYS, ADA_SAYS, ()),
        (ADA_SAYS, (), ADA_SAYS),
        (ADA_STARTED, (), ()),
        (
            (PartialSpeech(name="Ada", text="wait a moment"),),
            ADA_SAYS,
            (PartialSpeech(name="Ada", text="wait a moment"),),
        ),
        ((PartialSpeech(name="Ben", text="no"), *ADA_SAYS), ADA_SAYS, (PartialSpeech(name="Ben", text="no"),)),
    ],
    ids=["told", "not told", "no words yet", "more words", "another person"],
)
def test_the_words_said_over_him_that_were_not_told_are_each_partial_with_words_that_differs_from_the_told(
    said, told, untold
):
    assert untold_words(said, told) == untold


def _held(telling: Telling = Telling.NOT_TOLD) -> Held:
    return Held("h1", ADA_SAYS, record_gone=False, telling=telling)


def _record_with(held: Held | None, *, decided: str | None = None) -> SpeechRecord:
    return SpeechRecord(saying="One.", hold=held, decided=decided)


def test_the_hold_a_window_states_stands_in_the_record_unless_it_is_the_one_decided_last():
    stating = _window(interrupted="h1", speech=ADA_SAYS)

    assert hold_read(SpeechRecord(), stating).hold == _held()
    assert hold_read(_record_with(None, decided="h1"), stating).hold is None
    assert hold_read(_record_with(_held()), _window(interrupted=None, speech=None)).hold is None


def test_a_read_of_the_same_hold_keeps_what_a_turn_was_told_of_it():
    told = hold_marked_told(_record_with(_held()))

    assert hold_read(told, _window(interrupted="h1", speech=ADA_SAYS)).hold == _held(Telling.TELLING)


def test_only_a_hold_that_may_be_told_and_that_no_turn_was_told_is_to_be_told():
    assert hold_untold(_record_with(_held())) == _held()
    assert hold_untold(_record_with(_held(Telling.TELLING))) is None
    assert hold_untold(_record_with(Held("h1", ADA_STARTED, record_gone=False))) is None
    assert hold_untold(SpeechRecord()) is None


def test_a_hold_the_turn_that_was_told_it_did_not_decide_stays_undecided_and_a_decided_one_stands_no_more():
    told = hold_marked_told(_record_with(_held()))

    assert hold_turn_ended(told).hold == _held(Telling.UNDECIDED)
    assert hold_turn_ended(_record_with(_held())).hold == _held()
    assert hold_turn_ended(SpeechRecord()) == SpeechRecord()
    assert hold_decided(told, "h1") == SpeechRecord(saying="One.", hold=None, decided="h1")


def test_the_sentence_he_was_in_outlives_the_turn_that_sent_only_while_a_hold_stands():
    ended = speech_ended(SpeechRecord(in_flight=1, credit=2, saying="One.", hold=_held()))

    assert ended == SpeechRecord(saying="One.", hold=_held())
    assert speech_ended(SpeechRecord(in_flight=1, credit=2, saying="One.")) == SpeechRecord()
