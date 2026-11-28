"""What his next request holds: the history kept, what arrived since a mark, and the messages a turn is told."""

from __future__ import annotations

import json
import secrets
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceView, SpeakRefusal, Spoken, Written
from agents.turn import MAX_EVENT_BYTES

from personas.decision import TURN_CANCELLED_CODE
from personas.entries import Answered, Entry, History, KeptLine, Lines, TurnTold, Unanswered, ViewSeen, Waiting
from personas.present import (
    EMPTY_PRESENT,
    Arrived,
    Present,
    answered_into,
    arrived_since,
    held_told,
    taken_into,
    told_header,
    told_turn,
    turn_messages,
)
from tests.support.history import (
    APP,
    EMPTY_HISTORY,
    LATER,
    MORNING,
    NOON,
    ONCE_DROPPED,
    OTHER_TABLE,
    ROOMY_BYTES,
    TABLE,
    a_sentence,
    a_view,
    an_act,
    history_holding,
    message_texts,
    said,
    told_line_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def history_after(history: History, new: Iterable[Entry], size_bytes: int) -> History:
    """The history once ``new`` joined it, in a step that keeps what ``size_bytes`` holds, with no message but the
    newest."""
    return taken_into(Present(history), new, size_bytes, size_bytes).history


def told_a_written_turn(
    history: History = EMPTY_HISTORY, *, missed: bool = False, aloud: bool = False, seqs: tuple[int, ...] = ()
) -> str:
    """The message of a turn at ``LATER`` in the place of the morning visit that tells the whole history."""
    turn = TurnTold(
        LATER,
        APP,
        MORNING.place,
        Lines(MORNING.generation, frozenset(seqs), missed),
        aloud=aloud,
        through=history.taken,
    )
    return told_turn(turn, history.entries)


@pytest.mark.parametrize("left_out", [1, 0])
def test_the_header_tells_his_name_and_whether_older_entries_left_the_history(left_out):
    """TURNS-9."""
    history = History((), left_out=left_out, heard=0)

    assert json.loads(told_header("Mozart", history)) == {"you": "Mozart", "older_dropped": left_out > 0}


def a_lines_turn(*seqs: int, through: int) -> TurnTold:
    lines = Lines(MORNING.generation, frozenset(seqs), missed=False)
    return TurnTold(LATER, APP, MORNING.place, lines, aloud=False, through=through)


def present_with_a_turn_told(line_seq: int) -> Present:
    """The present after one line was taken and the turn at the table for it was answered."""
    present = taken_into(EMPTY_PRESENT, (said(line_seq, "one"),), ROOMY_BYTES, ROOMY_BYTES)
    return answered_into(present, TurnTold(LATER, APP, TABLE, Waiting(1), aloud=False, through=present.history.taken))


def test_the_messages_before_the_last_do_not_change_with_the_clock_a_turn_is_composed_at():
    """TURNS-15."""
    present = present_with_a_turn_told(1)
    next_turn = a_lines_turn(1, through=1)

    early = message_texts(turn_messages("Mozart", present, next_turn, {}))
    late = message_texts(turn_messages("Mozart", present, replace(next_turn, at=LATER + timedelta(hours=3)), {}))

    assert early[:-1] == late[:-1]
    assert early[-1] != late[-1]


def test_a_line_another_places_turn_told_first_is_named_by_the_woke_seqs_of_the_turn_it_is_for():
    present = present_with_a_turn_told(1)

    told = turn_messages("Mozart", present, a_lines_turn(1, through=1), {})
    *_, own = (json.loads(each) for each in message_texts(told))

    assert (own["entries"], own["woke"]["seqs"]) == ([], [1])


def test_a_turns_woke_seqs_leave_out_a_line_of_its_place_the_history_dropped_though_another_place_holds_its_seq():
    present = taken_into(EMPTY_PRESENT, (said(4, "elsewhere", visit=NOON),), ROOMY_BYTES, ROOMY_BYTES)

    told = turn_messages("Mozart", present, a_lines_turn(4, through=1), {})
    *_, own = (json.loads(each) for each in message_texts(told))

    assert own["woke"]["seqs"] == []


def test_a_view_another_places_turn_told_first_is_named_by_the_woke_revision_of_the_turn_it_is_for():
    view = ViewSeen(APP, TABLE, PlaceView(revision=2, awaits_you=True, state={"hand": []}), LATER)
    present = answered_into(taken_into(EMPTY_PRESENT, (view,), ROOMY_BYTES, ROOMY_BYTES), a_lines_turn(through=1))
    for_the_view = TurnTold(LATER, APP, TABLE, Waiting(2), aloud=False, through=1)

    *_, own = (json.loads(each) for each in message_texts(turn_messages("Mozart", present, for_the_view, {})))

    assert (own["entries"], own["woke"]["revision"]) == ([], 2)


@pytest.fixture
def told_one_bytes() -> int:
    """The bytes the line ``said(1, "one")`` is told as in the history."""
    return told_line_bytes(APP, "room-7", "one", name="Bo")


def test_a_view_turn_message_tells_its_admission_what_woke_him_and_every_entry_with_the_time_it_happened_at():
    """TURNS-10."""
    view = PlaceView(revision=3, awaits_you=True, state={"hand": ["x", {"seen": True}], "umläut": "ä"})
    history = history_holding(
        (
            said(1, "Spiel den Buben", name="Ada", visit=NOON, spoken=True),
            ViewSeen(APP, TABLE, view, LATER - timedelta(seconds=7, microseconds=400)),
        )
    )
    turn = TurnTold(LATER + timedelta(microseconds=900), APP, TABLE, Waiting(3), aloud=False, through=history.taken)

    told = told_turn(turn, history.entries)

    assert json.loads(told) == {
        "at": "2026-11-10T18:00:00Z",
        "woke": {"why": "view", "app": APP, "place": {"kind": "table", "id": "table-1"}, "revision": 3},
        "entries": [
            {
                "entry": "line",
                "app": NOON.app,
                "place": {"kind": "call", "id": "room-8"},
                "at": "2026-11-10T18:00:00Z",
                "seq": 1,
                "name": "Ada",
                "kind": "message",
                "text": "Spiel den Buben",
                "own": False,
                "spoken": True,
            },
            {
                "entry": "view",
                "app": APP,
                "place": {"kind": "table", "id": "table-1"},
                "at": "2026-11-10T17:59:52Z",
                "revision": 3,
                "awaits_you": True,
                "state": view.state,
            },
        ],
    }
    assert "ä" in told
    assert told.count(json.dumps(view.state, ensure_ascii=False)) == 1


def test_a_time_is_told_in_utc_whatever_the_zone_it_is_held_in():
    """TURNS-10."""
    zone = timezone(timedelta(hours=2))
    turn = TurnTold(datetime(2026, 11, 10, 20, 0, 5, tzinfo=zone), APP, TABLE, Waiting(3), aloud=False, through=0)

    assert json.loads(told_turn(turn, ()))["at"] == "2026-11-10T18:00:05Z"


@pytest.mark.parametrize("missed", [False, True])
def test_a_written_turn_is_told_in_which_place_it_woke_which_lines_it_is_for_and_whether_older_ones_were_not_seen(
    missed,
):
    """TURNS-10."""
    woke = json.loads(told_a_written_turn(missed=missed, seqs=(9, 3)))["woke"]

    assert woke == {
        "why": "lines",
        "app": APP,
        "place": {"kind": "call", "id": "room-7"},
        "missed": missed,
        "seqs": [3, 9],
    }


def test_a_turn_that_is_spoken_is_told_so_in_what_woke_him_and_any_other_turn_is_told_no_such_key():
    """TURNS-10."""
    assert json.loads(told_a_written_turn(aloud=True))["woke"]["aloud"] is True
    assert "aloud" not in json.loads(told_a_written_turn())["woke"]


def test_a_turn_told_an_empty_history_is_told_no_entries():
    assert json.loads(told_a_written_turn())["entries"] == []


def test_the_history_is_told_in_the_order_it_holds_its_entries():
    history = history_holding((said(2, "later", visit=NOON), said(1, "earlier")))

    told = json.loads(told_a_written_turn(history))

    assert [entry["text"] for entry in told["entries"]] == ["later", "earlier"]


def test_new_lines_join_the_history_after_the_lines_it_holds():
    history = history_after(history_holding((said(1, "one"),)), (said(2, "two"), said(3, "three")), 10_000)

    assert [line.text for line in history.entries] == ["one", "two", "three"]
    assert history.older_dropped is False


@pytest.mark.parametrize(
    ("bound_over_the_two_lines", "kept_seqs", "older_dropped"),
    [
        pytest.param(1, (1, 2), False, id="the bound lies above the lines"),
        pytest.param(0, (1, 2), False, id="the bound is the lines' size"),
        pytest.param(-1, (2,), True, id="the bound lies one byte below the lines"),
    ],
)
def test_the_history_keeps_the_newest_lines_that_fit_the_bound_counted_in_utf8_bytes_of_their_told_json(
    text_and_its_told_json, bound_over_the_two_lines, kept_seqs, older_dropped
):
    text, told_text = text_and_its_told_json
    two_lines = 2 * told_line_bytes(APP, "room-7", told_text, name="Bo")

    history = history_after(EMPTY_HISTORY, (said(1, text), said(2, text)), two_lines + bound_over_the_two_lines)

    assert [line.seq for line in history.entries] == list(kept_seqs)
    assert history.older_dropped is older_dropped


@pytest.mark.parametrize(
    ("history", "expected_dropped"),
    [
        pytest.param(history_holding((said(1, "one"),)), True, id="an earlier entry leaves"),
        pytest.param(EMPTY_HISTORY, False, id="no earlier entry"),
    ],
)
def test_the_newest_entry_stays_even_when_it_alone_is_past_the_bound_and_the_history_says_when_an_older_one_left(
    history, expected_dropped, told_one_bytes
):
    kept_history = history_after(history, (said(2, "x" * 50),), told_one_bytes - 1)

    assert [line.text for line in kept_history.entries] == ["x" * 50]
    assert kept_history.older_dropped is expected_dropped


def test_a_view_the_history_holds_by_app_place_and_revision_adds_nothing_whatever_its_state():
    """TURNS-12."""
    held = ViewSeen(APP, TABLE, a_view(3), LATER)
    again = ViewSeen(APP, TABLE, a_view(3), LATER + timedelta(seconds=5))

    history = history_after(history_holding((held,)), (again,), 10_000)

    assert history.entries == (held,)


def test_a_view_that_differs_in_app_place_or_revision_from_every_held_one_is_appended_in_the_order_read():
    views = [
        ViewSeen(APP, TABLE, a_view(3), LATER),
        ViewSeen(APP, TABLE, a_view(4), LATER),
        ViewSeen(APP, OTHER_TABLE, a_view(3), LATER),
        ViewSeen("https://other.test/mcp", TABLE, a_view(3), LATER),
    ]

    history = history_after(EMPTY_HISTORY, views, 10_000)

    assert history.entries == tuple(views)


def test_a_revision_whose_entry_left_by_the_bound_is_appended_again_when_read_again():
    state_bytes = len('{"hand": "00000000"}')
    first = ViewSeen(APP, TABLE, a_view(1), LATER)
    second = ViewSeen(APP, TABLE, a_view(2), LATER)
    again = ViewSeen(APP, TABLE, a_view(1), LATER + timedelta(seconds=5))

    history = history_after(EMPTY_HISTORY, (first, second), state_bytes)
    history = history_after(history, (again,), state_bytes)

    assert history.entries == (again,)
    assert history.older_dropped is True


@pytest.mark.parametrize(
    ("bound_over_the_two", "kept_kinds", "older_dropped"),
    [
        pytest.param(1, ["line", "view"], False, id="the bound lies above a line and a state"),
        pytest.param(0, ["line", "view"], False, id="the bound is a line's and a state's size"),
        pytest.param(-1, ["view"], True, id="the bound lies one byte below them"),
    ],
)
def test_the_bound_counts_a_line_by_its_told_json_and_a_view_by_the_utf8_json_bytes_of_its_state_alone(
    bound_over_the_two, kept_kinds, older_dropped, told_one_bytes
):
    view = ViewSeen(APP, TABLE, PlaceView(revision=1, awaits_you=True, state={"hand": "ä♠"}), LATER)
    state_bytes = len('{"hand": "ä♠"}'.encode())
    size = told_one_bytes + state_bytes + bound_over_the_two

    history = history_after(EMPTY_HISTORY, (said(1, "one"), view), size)

    assert ["line" if isinstance(each, KeptLine) else "view" for each in history.entries] == kept_kinds
    assert history.older_dropped is older_dropped


@pytest.mark.parametrize(
    ("act", "told_act"),
    [
        pytest.param(
            an_act(),
            '{"entry": "act", "app": "https://app.test/mcp", "at": "2026-11-10T17:59:51Z", "tool": "say",'
            ' "arguments": {"text": "hi"}, "outcome": {"answered": true, "is_error": false, "text": "posted"}}',
            id="an answer",
        ),
        pytest.param(
            an_act(outcome=Unanswered("network")),
            '{"entry": "act", "app": "https://app.test/mcp", "at": "2026-11-10T17:59:51Z", "tool": "say",'
            ' "arguments": {"text": "hi"}, "outcome": {"answered": false, "code": "network"}}',
            id="no answer",
        ),
        pytest.param(
            an_act(arguments={"text": 'ä♠"\n'}),
            '{"entry": "act", "app": "https://app.test/mcp", "at": "2026-11-10T17:59:51Z", "tool": "say",'
            ' "arguments": {"text": "ä♠\\"\\n"},'
            ' "outcome": {"answered": true, "is_error": false, "text": "posted"}}',
            id="arguments of two-, three- and escaped characters",
        ),
    ],
)
@pytest.mark.parametrize(
    ("bound_over_the_two", "kept_kinds", "older_dropped"),
    [
        pytest.param(1, ["line", "act"], False, id="the bound lies above a line and an act"),
        pytest.param(0, ["line", "act"], False, id="the bound is a line's and an act's size"),
        pytest.param(-1, ["act"], True, id="the bound lies one byte below them"),
    ],
)
def test_the_bound_counts_an_act_by_the_utf8_bytes_of_its_told_json_(
    act, told_act, bound_over_the_two, kept_kinds, older_dropped, told_one_bytes
):
    size = told_one_bytes + len(told_act.encode()) + bound_over_the_two

    history = history_after(EMPTY_HISTORY, (said(1, "one"), act), size)

    assert ["line" if isinstance(each, KeptLine) else "act" for each in history.entries] == kept_kinds
    assert history.older_dropped is older_dropped


def test_an_act_is_always_appended_and_alone_past_the_bound_it_stays_as_the_newest_entry():
    first, second = an_act(), an_act(tool="play")

    history = history_after(history_holding((said(1, "one"),)), (first, second), 1)

    assert history.entries == (second,)
    assert history.older_dropped is True


def test_an_act_the_history_already_holds_is_appended_again():
    act = an_act()

    assert history_after(history_holding((act,)), (act,), 10_000).entries == (act, act)


def test_an_answered_act_is_told_with_its_app_tool_arguments_answer_and_time():
    """TURNS-13."""
    act = an_act(arguments={"place": "room-7", "text": "ä"}, outcome=Answered("gone", is_error=True))

    told_entries = json.loads(told_a_written_turn(history_holding((act,))))["entries"]

    assert told_entries == [
        {
            "entry": "act",
            "app": APP,
            "at": "2026-11-10T17:59:51Z",
            "tool": "say",
            "arguments": {"place": "room-7", "text": "ä"},
            "outcome": {"answered": True, "is_error": True, "text": "gone"},
        }
    ]


def test_an_unanswered_act_is_told_with_the_code_and_no_text():
    """TURNS-13."""
    act = an_act(outcome=Unanswered(TURN_CANCELLED_CODE))

    told_entry = json.loads(told_a_written_turn(history_holding((act,))))["entries"][0]

    assert told_entry["outcome"] == {"answered": False, "code": TURN_CANCELLED_CODE}


@pytest.mark.parametrize(
    ("outcome", "told_outcome"),
    [
        pytest.param(Spoken(spoken=True), {"answered": True}, id="spoken"),
        pytest.param(Written(written=True), {"answered": True, "written": True}, id="written to the place instead"),
        pytest.param(SpeakRefusal.TOO_LONG, {"answered": True, "refused": "too_long"}, id="refused"),
        pytest.param(Unanswered("network"), {"answered": False, "code": "network"}, id="no answer"),
    ],
)
def test_a_spoken_sentence_is_told_as_a_speech_entry_with_its_text_time_and_how_the_app_took_it(outcome, told_outcome):
    """TURNS-14."""
    sentence = a_sentence(text="Grüß dich.", outcome=outcome)

    told_entries = json.loads(told_a_written_turn(history_holding((sentence,))))["entries"]

    assert told_entries == [
        {
            "entry": "speech",
            "app": APP,
            "place": {"kind": "call", "id": "room-7"},
            "at": "2026-11-10T17:59:51Z",
            "text": "Grüß dich.",
            "outcome": told_outcome,
        }
    ]


@pytest.mark.parametrize(
    ("bound_over_the_two", "kept_kinds", "older_dropped"),
    [
        pytest.param(1, ["line", "speech"], False, id="the bound lies above a line and a sentence"),
        pytest.param(0, ["line", "speech"], False, id="the bound is a line's and a sentence's size"),
        pytest.param(-1, ["speech"], True, id="the bound lies one byte below them"),
    ],
)
def test_the_bound_counts_a_spoken_sentence_by_the_utf8_bytes_of_its_told_json_(
    bound_over_the_two, kept_kinds, older_dropped, told_one_bytes
):
    sentence = a_sentence(text="ä♠")
    told_sentence = json.dumps(
        {
            "entry": "speech",
            "app": APP,
            "place": {"kind": "call", "id": "room-7"},
            "at": "2026-11-10T17:59:51Z",
            "text": "ä♠",
            "outcome": {"answered": True},
        },
        ensure_ascii=False,
    )

    size = told_one_bytes + len(told_sentence.encode()) + bound_over_the_two
    history = history_after(EMPTY_HISTORY, (said(1, "one"), sentence), size)

    assert ["line" if isinstance(each, KeptLine) else "speech" for each in history.entries] == kept_kinds
    assert history.older_dropped is older_dropped


def test_the_history_stays_marked_once_it_dropped_older_lines_though_the_next_lines_fit():
    assert history_after(ONCE_DROPPED, (said(1, "one"),), 10_000).older_dropped is True


def an_own_line(seq: int, text: str = "mine") -> KeptLine:
    return KeptLine(MORNING, seq, LATER, "Mozart", "message", text, own=True, spoken=False, cut=False, act=False)


def a_view_seen(revision: int, *, size: int | None = None) -> ViewSeen:
    state = {"hand": "x" * size} if size is not None else {"hand": secrets.token_hex(4)}
    return ViewSeen(APP, TABLE, PlaceView(revision=revision, awaits_you=False, state=state), LATER)


def taking(history: History, *entries, size_bytes: int = ROOMY_BYTES) -> History:
    return history_after(history, entries, size_bytes)


@pytest.mark.parametrize(
    ("entries", "heard"),
    [
        pytest.param((said(1, "a"), said(2, "b")), 2, id="lines of others count as heard"),
        pytest.param((a_view_seen(1),), 1, id="a view revision counts as heard"),
        pytest.param(
            (an_own_line(1), an_act(), a_sentence()),
            0,
            id="his own line, his act and his spoken sentence are not heard",
        ),
    ],
)
def test_the_history_counts_the_entries_not_his_own_it_took_in(entries, heard):
    history = taking(EMPTY_HISTORY, *entries)

    assert (history.heard, history.taken) == (heard, len(entries))


def test_a_revision_the_history_already_holds_is_not_heard_again():
    """TURNS-35."""
    held = a_view_seen(3)

    history = taking(taking(EMPTY_HISTORY, held), a_view_seen(3))

    assert history.heard == 1


def test_the_history_counts_what_the_bound_removed_and_every_entry_ever_taken(told_one_bytes):
    history = taking(EMPTY_HISTORY, said(1, "one"), said(2, "one"), said(3, "one"), size_bytes=told_one_bytes + 5)

    assert (history.left_out, history.heard, history.taken) == (2, 3, 3)
    assert history.older_dropped is True
    assert len(history.entries) == 1


@pytest.mark.parametrize(
    ("held", "taken"),
    [
        pytest.param((said(1, "one"),), (), id="nothing was taken"),
        pytest.param(
            (said(1, "one"),),
            (an_own_line(2), an_act(), a_sentence()),
            id="only his own line, his act and his spoken sentence were taken",
        ),
        pytest.param((a_view_seen(3),), (a_view_seen(3),), id="a revision the history already held was taken again"),
    ],
)
def test_nothing_arrived_when_nothing_not_his_own_and_new_was_taken_since_the_admission(held, taken):
    admitted = taking(EMPTY_HISTORY, *held)

    assert arrived_since(taking(admitted, *taken), admitted) is None


def test_the_entries_taken_since_the_admission_arrived_in_the_order_taken_his_own_among_them():
    admitted = taking(EMPTY_HISTORY, said(1, "before"))
    act, own, theirs, view = an_act(), an_own_line(2), said(3, "after"), a_view_seen(4)

    arrived = arrived_since(taking(admitted, act, own, theirs, view), admitted)

    assert arrived == Arrived((act, own, theirs, view), older_dropped=False)


def test_a_revision_that_left_by_the_bound_and_is_read_again_arrives(told_one_bytes):
    revision = a_view_seen(3)
    admitted = taking(EMPTY_HISTORY, revision)
    pushed_out = taking(admitted, said(1, "x" * 50), size_bytes=told_one_bytes)
    assert revision not in pushed_out.entries

    arrived = arrived_since(taking(pushed_out, revision), admitted)

    assert arrived is not None
    assert revision in arrived.entries


def test_arrivals_the_bound_removed_before_the_hold_are_left_out_and_older_dropped_is_said(told_one_bytes):
    admitted = taking(EMPTY_HISTORY, *(said(each, "one") for each in range(1, 5)))
    during = tuple(said(each, "one") for each in range(5, 8))

    arrived = arrived_since(taking(admitted, *during, size_bytes=2 * told_one_bytes + 5), admitted)

    assert arrived == Arrived(during[1:], older_dropped=True)


@pytest.mark.parametrize(
    ("held", "between", "during", "older_dropped"),
    [
        pytest.param(
            (said(1, "a"), said(2, "b")),
            (),
            (a_view_seen(1, size=200),),
            False,
            id="the newest arrival alone past the bound is all that arrived",
        ),
        pytest.param(
            (said(1, "a"), said(2, "b")),
            (said(3, "c"),),
            (a_view_seen(1, size=200),),
            True,
            id="an arrival the bound removed leaves the newest one alone",
        ),
        pytest.param(
            (said(1, "a"),),
            (said(2, "c"),),
            (an_act(arguments={"text": "x" * 200}),),
            True,
            id="an arrival not his own left with only his own act left",
        ),
    ],
)
def test_what_arrived_when_the_bound_leaves_only_the_newest_entry_taken_during_the_turn(
    held, between, during, older_dropped
):
    admitted = taking(EMPTY_HISTORY, *held)

    after = taking(taking(admitted, *between), *during, size_bytes=50)

    assert arrived_since(after, admitted) == Arrived(during, older_dropped=older_dropped)


def told_held(arrived: Arrived) -> dict:
    return json.loads(held_told(arrived))


def test_a_held_call_is_told_it_was_not_sent_and_every_arrived_entry_his_own_act_too():
    """TURNS-20."""
    act = an_act(at=LATER - timedelta(seconds=4))
    line = said(2, "Spiel den Buben", name="Ada")

    told_text = told_held(Arrived((act, line), older_dropped=True))

    assert told_text["sent"] is False
    assert told_text["arrived"]["older_dropped"] is True
    assert [each["entry"] for each in told_text["arrived"]["entries"]] == ["act", "line"]
    assert told_text["arrived"]["entries"][0]["at"] == "2026-11-10T17:59:56Z"
    assert told_text["arrived"]["entries"][1]["text"] == "Spiel den Buben"


def test_a_held_text_whose_arrivals_pass_the_event_bound_tells_the_newest_that_fit_and_says_older_dropped():
    third = MAX_EVENT_BYTES // 3 + 1
    views = tuple(a_view_seen(each, size=third) for each in (1, 2, 3))

    text = held_told(Arrived(views, older_dropped=False))

    told_text = json.loads(text)
    assert len(text.encode()) <= MAX_EVENT_BYTES
    assert [each["revision"] for each in told_text["arrived"]["entries"]] == [2, 3]
    assert told_text["arrived"]["older_dropped"] is True


def test_an_arrival_that_alone_passes_the_event_bound_is_left_out_and_older_dropped_is_said():
    text = held_told(Arrived((a_view_seen(1, size=MAX_EVENT_BYTES),), older_dropped=False))

    assert json.loads(text) == {"sent": False, "arrived": {"older_dropped": True, "entries": []}}


def test_the_repr_of_an_arrival_shows_no_entry():
    marker = secrets.token_hex(8)
    arrived = Arrived((said(1, marker, name=marker), a_view_seen(1)), older_dropped=False)

    assert marker not in repr(arrived)
