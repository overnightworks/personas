"""What an agent's history gained since its last write, and the present a history kept in the store resumes as."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from personas.entries import HistoryDelta, KeptHistory, Lines, TurnTold, entry_size
from personas.present import (
    EMPTY_PRESENT,
    Present,
    answered_into,
    arrived_since,
    history_delta,
    present_of,
    taken_into,
    turn_messages,
)
from tests.support.history import APP, LATER, MORNING, ROOMY_BYTES, a_skill_read, a_tools_load, an_act, said

if TYPE_CHECKING:
    from personas.entries import Entry


def answered_through(present: Present, *seqs: int) -> Present:
    """The present once a turn for these lines, told the whole history, was answered."""
    lines = Lines(MORNING.generation, frozenset(seqs), missed=False)
    turn = TurnTold(LATER, APP, MORNING.place, lines, aloud=False, through=present.history.taken)
    return answered_into(present, turn)


def took(present: Present, *entries: Entry, bound: int = ROOMY_BYTES) -> Present:
    return taken_into(present, entries, bound, bound)


TWO_LINES_BYTES = 2 * entry_size(said(1, "one"))
"""A bound that holds two of the lines below, each of the same size."""


def after_a_cut() -> tuple[Present, Present]:
    """Three lines, the first two answered one turn each, in a history that holds two: the first line and its turn
    have left."""
    first = answered_through(took(EMPTY_PRESENT, said(1, "one")), 1)
    second = answered_through(took(first, said(2, "two"), bound=TWO_LINES_BYTES), 2)
    return first, took(second, said(3, "six"), bound=TWO_LINES_BYTES)


def test_the_delta_since_a_write_holds_the_entries_and_answered_turns_added_after_it_at_their_ordinals():
    written = answered_through(took(EMPTY_PRESENT, said(1, "one")), 1)
    current = answered_through(took(written, said(2, "two"), an_act()), 2)

    assert history_delta(written, current) == HistoryDelta(1, (said(2, "two"), an_act()), 1, current.told[1:], 0)


def test_what_left_by_the_bound_before_a_write_is_not_written_and_the_delta_says_where_the_history_starts():
    written, current = after_a_cut()

    assert history_delta(written, current) == HistoryDelta(1, (said(2, "two"), said(3, "six")), 1, current.told, 1)
    assert history_delta(EMPTY_PRESENT, current) == history_delta(written, current)


def test_a_present_unchanged_since_its_write_has_no_delta():
    _, current = after_a_cut()

    assert history_delta(current, current) is None


def test_a_delta_that_only_drops_entries_says_so():
    written = took(EMPTY_PRESENT, said(1, "one"), said(2, "two"))
    current = took(written, bound=TWO_LINES_BYTES)
    shrunk = took(written, said(3, "six"), bound=TWO_LINES_BYTES)

    assert history_delta(written, current) is None
    assert history_delta(took(written, said(3, "six")), shrunk) == HistoryDelta(3, (), 0, (), 1)


def test_a_present_resumed_from_its_kept_history_tells_the_next_turn_the_same_messages_byte_for_byte():
    _, current = after_a_cut()
    kept = KeptHistory(1, current.history.entries, 1, current.told)
    next_turn = TurnTold(LATER, APP, MORNING.place, Lines(MORNING.generation, frozenset({3}), missed=False), False, 3)

    resumed = present_of(kept)

    assert turn_messages("Mozart", resumed, next_turn) == turn_messages("Mozart", current, next_turn)


def test_the_first_delta_after_a_resume_continues_the_ordinals_of_the_kept_history():
    _, current = after_a_cut()
    resumed = present_of(KeptHistory(1, current.history.entries, 1, current.told))

    grown = answered_through(took(resumed, said(4, "four")), 4)

    assert history_delta(resumed, grown) == HistoryDelta(3, (said(4, "four"),), 2, grown.told[-1:], 1)


def test_a_turn_is_told_a_history_holding_a_skill_and_a_load_without_them():
    held = took(EMPTY_PRESENT, said(1, "one"), a_skill_read(), a_tools_load(), said(2, "two"))
    turn = TurnTold(LATER, APP, MORNING.place, Lines(MORNING.generation, frozenset({1, 2}), missed=False), False, 4)

    *_, message = (json.loads(each) for each in turn_messages("Mozart", held, turn))

    assert [entry["entry"] for entry in message["entries"]] == ["line", "line"]


def test_a_skill_and_a_load_taken_since_the_admission_hold_back_no_call():
    admitted = took(EMPTY_PRESENT, said(1, "one")).history

    taken = took(EMPTY_PRESENT, said(1, "one"), a_skill_read(), a_tools_load()).history

    assert arrived_since(taken, admitted) is None


def test_the_cut_counts_a_skill_by_its_body_and_tool_names():
    skill = a_skill_read(body="x" * 100, tools=("say", "play"))
    bound = entry_size(skill) + entry_size(said(2, "two"))

    present = took(took(EMPTY_PRESENT, said(1, "one"), skill), said(2, "two"), bound=bound)

    assert entry_size(skill) == 100 + len("say") + len("play")
    assert present.history.entries == (skill, said(2, "two"))
