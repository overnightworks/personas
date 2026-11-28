"""What an agent's history gained since its last write, and the present a history kept in the store resumes as."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agents.turn import ToolRound

from personas.entries import HistoryDelta, KeptHistory, Lines, TurnTold, entry_size
from personas.present import (
    EMPTY_PRESENT,
    EMPTY_SKILL_TEXT,
    NOT_OFFERED_TEXT,
    AppOffer,
    Arrived,
    Present,
    answered_into,
    arrived_since,
    blocks_due,
    held_told,
    history_delta,
    present_of,
    taken_into,
    turn_messages,
)
from tests.support.history import APP, LATER, MORNING, ROOMY_BYTES, a_skill_of, a_skill_read, a_tools_load, an_act, said

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

    assert turn_messages("Mozart", resumed, next_turn, {}) == turn_messages("Mozart", current, next_turn, {})


def test_a_present_resumed_from_a_kept_block_tells_its_rounds_byte_for_byte_and_finds_none_due():
    block = (a_skill_read(call_id="personas-skill-0"), a_tools_load(call_id="personas-skill-1"))
    current = answered_through(took(took(EMPTY_PRESENT, said(1, "one")), *block), 1)
    kept = KeptHistory(0, current.history.entries, 0, current.told)
    next_turn = TurnTold(LATER, APP, MORNING.place, Lines(MORNING.generation, frozenset({2}), missed=False), False, 3)
    deferred = {APP: ("say", "play")}

    resumed = present_of(kept)

    assert turn_messages("Mozart", resumed, next_turn, deferred) == turn_messages(
        "Mozart", current, next_turn, deferred
    )
    offers = (AppOffer(APP, (), (a_skill_of("greeting", "say", "play"),)),)
    assert blocks_due(offers, (APP,), deferred, resumed.history) == ()


def test_the_first_delta_after_a_resume_continues_the_ordinals_of_the_kept_history():
    _, current = after_a_cut()
    resumed = present_of(KeptHistory(1, current.history.entries, 1, current.told))

    grown = answered_through(took(resumed, said(4, "four")), 4)

    assert history_delta(resumed, grown) == HistoryDelta(3, (said(4, "four"),), 2, grown.told[-1:], 1)


def seen_in(messages) -> list[object]:
    """What a turn's messages hold in order: the entries a message tells, or the id, call and result of a round."""
    return [
        ("round", each.calls[0].call_id, each.calls[0].arguments, each.calls[0].result, len(each.calls))
        if isinstance(each, ToolRound)
        else ("message", [entry["seq"] for entry in json.loads(each.text).get("entries", [])])
        for each in messages
    ]


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param(
            (said(1, "one"), a_skill_read(), said(2, "two")),
            [
                ("message", []),
                ("message", [1]),
                ("round", "call-body", '{"skill": "greeting"}', "Greet people by name.", 1),
                ("message", [2]),
            ],
            id="one read between two lines",
        ),
        pytest.param(
            (
                said(1, "one"),
                a_skill_read(),
                said(2, "two"),
                a_skill_read(name="other", body="", call_id="call-other"),
                said(3, "six"),
            ),
            [
                ("message", []),
                ("message", [1]),
                ("round", "call-body", '{"skill": "greeting"}', "Greet people by name.", 1),
                ("message", [2]),
                ("round", "call-other", '{"skill": "other"}', EMPTY_SKILL_TEXT, 1),
                ("message", [3]),
            ],
            id="two reads, the second with no text",
        ),
        pytest.param(
            (a_skill_read(), a_tools_load(), said(1, "one")),
            [
                ("message", []),
                ("round", "call-body", '{"skill": "greeting"}', "Greet people by name.", 1),
                ("round", "call-load", '{"skill": "greeting"}', "", 1),
                ("message", [1]),
            ],
            id="a read and a load before the first line, each its round",
        ),
    ],
)
def test_a_turn_message_splits_around_each_skill_and_load_entrys_round(entries, expected):
    held = took(EMPTY_PRESENT, *entries)
    turn = TurnTold(
        LATER, APP, MORNING.place, Lines(MORNING.generation, frozenset({1, 2}), missed=False), False, len(entries)
    )

    messages = turn_messages("Mozart", held, turn, {APP: ("say", "play")})

    assert seen_in(messages) == expected
    assert "woke" in json.loads(messages[-1].text)
    assert all("woke" not in json.loads(each.text) for each in messages[1:-1] if not isinstance(each, ToolRound))


@pytest.mark.parametrize(
    ("deferred", "loads", "result"),
    [
        pytest.param({APP: ("play", "say")}, ("say", "play"), "", id="all kept"),
        pytest.param({APP: ("say", "wave")}, ("say",), "", id="one gone"),
        pytest.param({APP: ("wave",)}, (), NOT_OFFERED_TEXT, id="all gone"),
        pytest.param({"https://elsewhere.test": ("say", "play")}, (), NOT_OFFERED_TEXT, id="its app not listed"),
    ],
)
def test_a_loads_round_references_only_the_tools_its_app_still_sends_deferred(deferred, loads, result):
    held = took(EMPTY_PRESENT, a_tools_load(tools=("say", "play")), said(1, "one"))
    turn = TurnTold(LATER, APP, MORNING.place, Lines(MORNING.generation, frozenset({1}), missed=False), False, 2)

    (round_,) = (each for each in turn_messages("Mozart", held, turn, deferred) if isinstance(each, ToolRound))

    (call,) = round_.calls
    assert (call.loads, call.result, call.is_error) == (loads, result, False)


def test_a_skill_and_a_load_taken_since_the_admission_hold_back_no_call():
    admitted = took(EMPTY_PRESENT, said(1, "one")).history

    taken = took(EMPTY_PRESENT, said(1, "one"), a_skill_read(), a_tools_load()).history

    assert arrived_since(taken, admitted) is None


@pytest.mark.parametrize(
    ("kept", "size"),
    [
        pytest.param(a_skill_read(body="x" * 100), 100, id="a skill"),
        pytest.param(a_tools_load(tools=("say", "play")), len("say") + len("play"), id="a load"),
    ],
)
def test_the_cut_counts_a_skill_by_its_body_and_a_load_by_its_tool_names(kept, size):
    bound = entry_size(kept) + entry_size(said(2, "two"))

    present = took(took(EMPTY_PRESENT, said(1, "one"), kept), said(2, "two"), bound=bound)

    assert entry_size(kept) == size
    assert present.history.entries == (kept, said(2, "two"))


def test_a_held_call_is_told_no_skill_and_no_load_that_arrived_beside_a_line():
    arrived = Arrived((a_skill_read(), a_tools_load(), said(1, "one")), older_dropped=False)

    told = json.loads(held_told(arrived))

    assert [each["entry"] for each in told["arrived"]["entries"]] == ["line"]
