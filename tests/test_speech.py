from __future__ import annotations

from dataclasses import replace

import pytest
from agents.member import Interrupted, PartialSpeech, PlaceWindow, SpeakRefusal, Spoken, TalkedOver, Written

from personas.speech import (
    Answered,
    Held,
    HoldKey,
    Request,
    Sentence,
    SpeechRecord,
    Telling,
    answered_as,
    hold_decided,
    hold_forgotten,
    hold_marked_told,
    hold_read,
    hold_stated,
    hold_tellable,
    hold_turn_ended,
    hold_untold,
    sentence_taken,
    sentences_released,
    speech_answered,
    speech_continued,
    speech_ended,
    speech_line_entered,
    speech_started,
    speech_stopped,
    speech_unsaid,
    unsent_dropped,
    untold_words,
)

ONE, TWO, THREE = Sentence("One.", 100.0), Sentence("Two.", 100.0), Sentence("Three.", 100.0)


@pytest.mark.parametrize(
    ("ended", "answered"),
    [
        (Spoken(spoken=True), Answered.SPOKEN),
        (Written(written=True), Answered.WRITTEN),
        (SpeakRefusal.HELD, Answered.HELD),
        (SpeakRefusal.STOPPED, Answered.STOPPED),
        (SpeakRefusal.NOT_INTERRUPTED, Answered.REFUSED),
        (ValueError("a failure"), Answered.UNANSWERED),
        (None, Answered.UNANSWERED),
    ],
)
def test_a_request_is_answered_as_spoken_written_held_refused_or_not_at_all(ended, answered):
    assert answered_as(ended) is answered


@pytest.mark.parametrize(
    ("replacing", "unsent"),
    [(False, (ONE, TWO, THREE)), (True, (TWO, THREE))],
    ids=["a turn told a hold adds", "a turn told none replaces the rest"],
)
def test_the_sentences_a_turn_releases_are_added_to_the_unsent_ones_or_replace_them(replacing, unsent):
    record = SpeechRecord(in_flight=Request(Sentence("Sent.", 100.0)), saying="Plays.", unsent=(ONE,))

    released, dropped = sentences_released(record, ("Two.", "Three."), 100.0, replacing=replacing)

    assert released.unsent == unsent
    assert dropped == (1 if replacing else 0)
    assert (released.in_flight, released.saying) == (record.in_flight, record.saying)


@pytest.mark.parametrize(
    ("record", "now", "taken", "expired"),
    [
        pytest.param(SpeechRecord(unsent=(ONE, TWO)), 0.0, ONE, 0, id="the first unsent one goes out"),
        pytest.param(SpeechRecord(unsent=(ONE, TWO)), 100.0, None, 2, id="at its bound a sentence is dropped"),
        pytest.param(SpeechRecord(unsent=(ONE,), in_flight=Request(TWO)), 0.0, None, 0, id="one request is open"),
        pytest.param(SpeechRecord(unsent=(ONE,), held_unread=True), 0.0, None, 0, id="a held answer waits for a read"),
        pytest.param(SpeechRecord(unsent=(ONE,), hold=Held("h", "g", (), True)), 0.0, None, 0, id="a hold stands"),
        pytest.param(
            SpeechRecord(unsent=(ONE,), hold=Held("h", "g", (), True)), 100.0, None, 0, id="a hold keeps them"
        ),
        pytest.param(SpeechRecord(unsent=(ONE,), saying="Plays."), 0.0, ONE, 0, id="the one that plays does not wait"),
        pytest.param(SpeechRecord(), 0.0, None, 0, id="nothing unsent"),
    ],
)
def test_the_next_sentence_goes_out_unless_a_request_is_open_or_a_hold_stands_and_those_past_their_bound_drop(
    record, now, taken, expired
):
    after, sentence, dropped = sentence_taken(record, now)

    assert (sentence, dropped) == (taken, expired)
    assert after.in_flight == (record.in_flight if taken is None else Request(taken))


@pytest.mark.parametrize(
    ("answered", "saying", "unsent", "held_unread"),
    [
        pytest.param(Answered.SPOKEN, "One.", (TWO,), False, id="spoken is the one that plays"),
        pytest.param(Answered.WRITTEN, None, (TWO,), False, id="written plays nothing"),
        pytest.param(Answered.UNANSWERED, None, (TWO,), False, id="unanswered plays nothing"),
        pytest.param(Answered.HELD, None, (ONE, TWO), True, id="held goes back first among the unsent"),
        pytest.param(Answered.REFUSED, None, (), False, id="any other refusal drops the unsent ones"),
        pytest.param(Answered.STOPPED, None, (), False, id="a stopped he did not cause drops the unsent ones"),
    ],
)
def test_the_answer_to_the_open_request_decides_what_plays_and_what_stays_unsent(answered, saying, unsent, held_unread):
    record = SpeechRecord(in_flight=Request(ONE), unsent=(TWO,))

    after = speech_answered(record, answered)

    assert (after.in_flight, after.saying, after.unsent, after.held_unread) == (None, saying, unsent, held_unread)


def test_a_stopped_the_app_answers_to_a_request_his_own_stop_or_answer_cut_leaves_the_unsent_sentences():
    cut, _ = speech_stopped(SpeechRecord(in_flight=Request(ONE)))
    answered = speech_answered(replace(cut, unsent=(TWO,)), Answered.STOPPED)

    assert (answered.in_flight, answered.unsent) == (None, (TWO,))


def test_the_sentence_that_answered_his_hold_is_the_one_that_plays():
    assert speech_started(SpeechRecord(saying="Old."), "New.").saying == "New."


def test_a_sentence_answered_as_spoken_whose_line_entered_already_does_not_play():
    record = SpeechRecord(in_flight=Request(ONE))

    after = speech_answered(speech_line_entered(record), Answered.SPOKEN)

    assert after.saying is None


def test_a_sentence_answered_as_spoken_plays_until_his_own_line_for_it_enters():
    playing = speech_answered(SpeechRecord(in_flight=Request(ONE)), Answered.SPOKEN)

    assert playing.saying == "One."
    assert speech_line_entered(playing).saying is None
    assert speech_line_entered(SpeechRecord()) == SpeechRecord()


@pytest.mark.parametrize(
    ("record", "unsaid"),
    [
        (SpeechRecord(), ()),
        (SpeechRecord(in_flight=Request(ONE), unsent=(TWO, THREE)), ("One.", "Two.", "Three.")),
        (SpeechRecord(saying="Plays.", unsent=(TWO,)), ("Two.",)),
    ],
    ids=["nothing", "the one in flight and the unsent ones", "the one that plays is said"],
)
def test_the_sentences_not_said_yet_are_the_one_in_flight_and_the_unsent_ones_in_order(record, unsaid):
    assert speech_unsaid(record) == unsaid


def test_his_stop_drops_the_unsent_sentences_and_marks_the_request_in_flight_cut_which_runs_to_its_answer():
    stopped, dropped = speech_stopped(SpeechRecord(in_flight=Request(ONE), unsent=(TWO, THREE)))

    assert (stopped.unsent, dropped) == ((), 2)
    assert stopped.in_flight == Request(ONE, cut=True)
    assert stopped.in_flight.cut_by_stop(SpeakRefusal.STOPPED)
    assert not stopped.in_flight.cut_by_stop(SpeakRefusal.HELD)
    assert not Request(ONE).cut_by_stop(SpeakRefusal.STOPPED)


def test_his_continue_gives_the_unsent_sentences_the_bound_of_the_turn_that_decided():
    continued = speech_continued(SpeechRecord(unsent=(ONE, TWO)), 500.0)

    assert continued.unsent == (Sentence("One.", 500.0), Sentence("Two.", 500.0))


def test_unsent_sentences_drop_and_are_counted_whatever_else_the_record_holds():
    record = SpeechRecord(in_flight=Request(ONE), saying="Plays.", unsent=(TWO, THREE))

    dropped, count = unsent_dropped(record)

    assert (dropped, count) == (SpeechRecord(in_flight=Request(ONE), saying="Plays."), 2)


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
        (_window(interrupted="h1", speech=ADA_SAYS), None, Held("h1", "g", ADA_SAYS, record_gone=False)),
        (
            _window(interrupted="h1", speech=None),
            Held("h1", "g", ADA_SAYS, record_gone=False),
            Held("h1", "g", ADA_SAYS, True),
        ),
        (
            _window(interrupted="h2", speech=None),
            Held("h1", "g", ADA_SAYS, record_gone=False),
            Held("h2", "g", (), True),
        ),
    ],
    ids=["not held", "held", "record gone keeps the words of the same hold", "record gone of a new hold has none"],
)
def test_a_window_states_the_hold_with_what_was_said_over_him(window, seen, stated):
    assert hold_stated(window, seen) == stated


@pytest.mark.parametrize(
    ("held", "tellable"),
    [
        (Held("h", "g", ADA_STARTED, record_gone=False), False),
        (Held("h", "g", ADA_SAYS, record_gone=False), True),
        (Held("h", "g", ADA_STARTED, record_gone=True), True),
        (Held("h", "g", (), record_gone=True), True),
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
    return Held("h1", "g", ADA_SAYS, record_gone=False, telling=telling)


def _record_with(held: Held | None, *, decided: HoldKey | None = None) -> SpeechRecord:
    return SpeechRecord(saying="One.", hold=held, decided=decided)


def test_the_hold_a_window_states_stands_in_the_record_unless_it_is_the_one_decided_last():
    stating = _window(interrupted="h1", speech=ADA_SAYS)

    assert hold_read(SpeechRecord(), stating).hold == _held()
    assert hold_read(_record_with(None, decided=HoldKey("g", "h1")), stating).hold is None
    assert hold_read(_record_with(_held()), _window(interrupted=None, speech=None)).hold is None


def test_a_read_of_the_same_hold_keeps_what_a_turn_was_told_of_it():
    told = hold_marked_told(_record_with(_held()))

    assert hold_read(told, _window(interrupted="h1", speech=ADA_SAYS)).hold == _held(Telling.TELLING)


def test_only_a_hold_that_may_be_told_and_that_no_turn_was_told_is_to_be_told():
    assert hold_untold(_record_with(_held())) == _held()
    assert hold_untold(_record_with(_held(Telling.TELLING))) is None
    assert hold_untold(_record_with(Held("h1", "g", ADA_STARTED, record_gone=False))) is None
    assert hold_untold(SpeechRecord()) is None


def test_a_hold_id_decided_in_an_earlier_window_generation_is_a_new_hold():
    decided = _record_with(None, decided=HoldKey("g", "h1"))
    later = PlaceWindow(generation="g2", starts_at=1, entries=(), interrupted=Interrupted(id="h1"))

    assert hold_read(decided, later).hold == Held("h1", "g2", (), record_gone=True)


def test_a_place_that_left_forgets_the_hold_it_decided():
    decided = _record_with(None, decided=HoldKey("g", "h1"))

    assert hold_forgotten(decided) == SpeechRecord(saying="One.")


def test_a_hold_the_turn_that_was_told_it_did_not_decide_stays_undecided_and_a_decided_one_stands_no_more():
    told = hold_marked_told(_record_with(_held()))

    assert hold_turn_ended(told).hold == _held(Telling.UNDECIDED)
    assert hold_turn_ended(_record_with(_held())).hold == _held()
    assert hold_turn_ended(SpeechRecord()) == SpeechRecord()
    assert hold_decided(told, HoldKey("g", "h1")) == SpeechRecord(saying="One.", hold=None, decided=HoldKey("g", "h1"))


def test_the_end_of_the_sender_drops_the_unsent_sentences_and_keeps_the_sentence_that_plays_and_the_hold():
    ended, dropped = speech_ended(
        SpeechRecord(in_flight=Request(ONE), unsent=(TWO,), saying="One.", hold=_held(), held_unread=True)
    )

    assert (ended, dropped) == (SpeechRecord(saying="One.", hold=_held()), 1)
