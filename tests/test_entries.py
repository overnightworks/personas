"""What the history keeps of an agent: taken lines and the entries that show nothing of their content in a repr."""

from __future__ import annotations

import json
import secrets
from datetime import timedelta, timezone

import pytest
from agents.member import (
    PlaceEntry,
    PlaceView,
    SpeakRefusal,
    Written,
)

from personas.entries import (
    Answered,
    KeptLine,
    Lines,
    Room,
    TurnTold,
    Unanswered,
    ViewSeen,
    Waiting,
    entry_from_stored,
    entry_size,
    is_told,
    stored_entry,
    stored_turn,
    taken_lines,
    told_entry,
    told_line,
    told_outcome,
    turn_from_stored,
)
from tests.support.history import (
    APP,
    LATER,
    MORNING,
    NOON,
    TABLE,
    a_sentence,
    a_skill_read,
    a_tools_load,
    an_act,
    history_holding,
    said,
)


def test_taken_lines_are_the_visits_lines_as_kept_with_the_time_of_the_read():
    entries = (
        PlaceEntry(seq=4, kind="message", name="Ada", text="four", own=False),
        PlaceEntry(seq=5, kind="message", name="Mozart", text="five", own=True, spoken=True),
    )

    assert taken_lines(MORNING, entries, LATER) == (
        KeptLine(MORNING, 4, LATER, "Ada", "message", "four", own=False, spoken=False, cut=False, act=False),
        KeptLine(MORNING, 5, LATER, "Mozart", "message", "five", own=True, spoken=True, cut=False, act=False),
    )


def test_a_line_that_is_an_act_is_kept_as_one_and_told_with_the_mark_while_any_other_line_is_told_without_the_key():
    """TURNS-11."""
    acting = PlaceEntry(seq=1, kind="message", name="Ada", text="took a card", own=False, act=True)
    talking = PlaceEntry(seq=2, kind="message", name="Ada", text="good luck", own=False)

    kept_act, kept_talk = taken_lines(MORNING, (acting, talking), LATER)

    assert kept_act.act is True
    assert told_line(kept_act)["act"] is True
    assert "act" not in told_line(kept_talk)
    assert told_entry(kept_act)["act"] is True


def test_a_history_its_lines_and_its_views_show_no_name_no_text_and_no_state_in_their_repr():
    marker = secrets.token_hex(8)
    line = said(1, marker, name=marker)
    view = ViewSeen(APP, TABLE, PlaceView(revision=1, awaits_you=True, state={"hand": marker}), LATER)

    assert marker not in repr(history_holding((line, view)))
    assert marker not in repr(line)
    assert marker not in repr(view)


@pytest.mark.parametrize(
    ("answered", "told"),
    [
        (Answered("seen", is_error=False), {"answered": True, "is_error": False, "text": "seen"}),
        (
            Answered("seen", is_error=False, image_shown=True),
            {"answered": True, "is_error": False, "text": "seen", "image": True},
        ),
    ],
    ids=["no picture, no key", "a picture, told as image"],
)
def test_an_answer_is_told_with_the_image_mark_only_when_a_picture_was_shown(answered, told):
    """TURNS-13."""
    assert told_outcome(answered) == told


def test_an_act_shows_no_arguments_and_no_answer_in_its_repr_nor_in_the_history_that_holds_it():
    marker = secrets.token_hex(8)
    act = an_act(arguments={"text": marker}, outcome=Answered(marker, is_error=False))

    assert marker not in repr(act)
    assert marker not in repr(act.outcome)
    assert marker not in repr(history_holding((act,)))


def test_a_spoken_sentence_shows_no_text_in_its_repr_nor_in_the_history_that_holds_it():
    marker = secrets.token_hex(8)
    sentence = a_sentence(text=marker)

    assert marker not in repr(sentence)
    assert marker not in repr(history_holding((sentence,)))


@pytest.mark.parametrize(
    "entry",
    [
        KeptLine(MORNING, 2**70, LATER, "Ada", "message", "a\x00b \ud800", own=False, spoken=True, cut=True, act=True),
        said(1, "plain"),
        ViewSeen(
            APP,
            TABLE,
            PlaceView(revision=3, awaits_you=False, state={"z": 1.5, "a": [1, None, "x"]}, traceparent=None),
            LATER.astimezone(timezone(timedelta(hours=2))),
        ),
        an_act(arguments={"z": 0.1, "a": {"b": [True]}}, outcome=Answered("seen", is_error=True, image_shown=True)),
        an_act(outcome=Unanswered("cancelled")),
        ViewSeen(APP, TABLE, PlaceView(revision=3, awaits_you=True, state={}), LATER, generation="morning"),
        an_act(app="https://other.test/mcp", room=Room(APP, MORNING.place, "morning")),
        an_act(room=Room(APP, MORNING.place, None)),
        a_sentence(),
        a_sentence(generation="morning"),
        a_sentence(outcome=Written(written=True)),
        a_sentence(outcome=SpeakRefusal.HELD),
        a_sentence(outcome=Unanswered("network")),
        a_skill_read(),
        a_tools_load(),
    ],
    ids=[
        "line with every mark",
        "plain line",
        "view",
        "act answered",
        "act unanswered",
        "view with a visit",
        "act to another app with its room and visit",
        "act with a room and no visit",
        "sentence spoken",
        "sentence with a visit",
        "sentence written",
        "sentence refused",
        "sentence unanswered",
        "skill read",
        "tools loaded",
    ],
)
def test_every_entry_kind_comes_back_from_its_stored_json_as_it_went_in(entry):
    stored = json.dumps(stored_entry(entry))

    back = entry_from_stored(json.loads(stored))

    assert back == entry
    assert json.dumps(stored_entry(back)) == stored
    if is_told(entry):
        assert json.dumps(told_entry(back)) == json.dumps(told_entry(entry))


@pytest.mark.parametrize("kind", ["view", "act", "sentence"])
def test_an_entry_stored_without_its_room_or_visit_loads_with_none_and_is_told_and_sized_as_one_with_them(kind):
    kept = {
        "view": ViewSeen(APP, TABLE, PlaceView(revision=1, awaits_you=True, state={}), LATER, generation="morning"),
        "act": an_act(room=Room(APP, MORNING.place, "morning")),
        "sentence": a_sentence(generation="morning"),
    }[kind]
    stored = stored_entry(kept)
    for key in ("room", "generation"):
        stored.pop(key, None)

    loaded = entry_from_stored(stored)

    assert getattr(loaded, "room", None) is None
    assert getattr(loaded, "generation", None) is None
    if is_told(kept):
        assert json.dumps(told_entry(kept)) == json.dumps(told_entry(loaded))
    assert entry_size(kept) == entry_size(loaded)


def test_a_skill_read_shows_no_body_in_its_repr_nor_in_the_history_that_holds_it():
    marker = secrets.token_hex(8)
    skill = a_skill_read(body=marker)

    assert marker not in repr(skill)
    assert marker not in repr(history_holding((skill,)))


@pytest.mark.parametrize(
    "turn",
    [
        TurnTold(LATER, APP, NOON.place, Lines("noon", frozenset({1, 2**65}), missed=True, traceparent="tp"), True, 7),
        TurnTold(LATER, APP, TABLE, Waiting(4), aloud=False, through=0),
    ],
    ids=["for lines", "at a view"],
)
def test_an_answered_turn_comes_back_from_its_stored_json_as_it_went_in(turn):
    stored = json.dumps(stored_turn(turn))

    back = turn_from_stored(json.loads(stored))

    assert back == turn


@pytest.mark.parametrize(
    "stored",
    [stored_entry(said(1, "plain")) | {"entry": "dream"}, None, ["plain"], "plain"],
    ids=["unknown kind", "null", "array", "string"],
)
def test_a_stored_value_that_names_no_kind_of_entry_is_refused(stored):
    with pytest.raises(ValueError, match="no kind of entry"):
        entry_from_stored(stored)
