"""What the history keeps of an agent: taken lines and the entries that show nothing of their content in a repr."""

from __future__ import annotations

import secrets

from agents.member import (
    PlaceEntry,
    PlaceView,
)

from personas.entries import Answered, KeptLine, ViewSeen, taken_lines
from tests.support.history import (
    APP,
    LATER,
    MORNING,
    TABLE,
    a_sentence,
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
        KeptLine(MORNING, 4, LATER, "Ada", "message", "four", own=False, spoken=False),
        KeptLine(MORNING, 5, LATER, "Mozart", "message", "five", own=True, spoken=True),
    )


def test_a_history_its_lines_and_its_views_show_no_name_no_text_and_no_state_in_their_repr():
    marker = secrets.token_hex(8)
    line = said(1, marker, name=marker)
    view = ViewSeen(APP, TABLE, PlaceView(revision=1, awaits_you=True, state={"hand": marker}), LATER)

    assert marker not in repr(history_holding((line, view)))
    assert marker not in repr(line)
    assert marker not in repr(view)


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
