"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import asyncio
import itertools
import json
import secrets
from dataclasses import replace
from datetime import timedelta

import pytest
from agents.member import (
    LEASE_SECONDS,
    LISTEN_BOUND_SECONDS,
    SPEAK_TOOL,
    AcceptInvitationRefusal,
    FigureRef,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SpeakRefusal,
    Spoken,
    Standing,
    StandingState,
    VoiceState,
    Written,
)
from agents.tools import ToolDeclaration, ToolOutcome
from agents.turn import MAX_EVENT_BYTES
from mcp_types import INVALID_PARAMS

from personas.app_session import READ_SECONDS
from personas.decision import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    NETWORK_FAILURE_CODE,
    RENEW_BEFORE_EXPIRY_SECONDS,
    SPOKEN_WAIT_BOUND_SECONDS,
    TURN_CANCELLED_CODE,
    Accept,
    AppPlace,
    Arrived,
    Joined,
    Lines,
    Memory,
    Paused,
    Perception,
    Reaction,
    Reading,
    Restarted,
    Suspend,
    Token,
    Turns,
    UnrecognisedRefusal,
    Waiting,
    accepts_for,
    act_outcome,
    active_view_places,
    active_window_places,
    after_refusal,
    arrived_since,
    backoff_delay,
    bound_declarations,
    budget_pause,
    call_within_bound,
    claimed,
    due_again,
    decisive_failure,
    failure_code,
    forget_gone,
    held_apps,
    held_told,
    history_after,
    holds_a_place,
    is_voice_app,
    jsonrpc_code,
    left_out_for,
    left_standing,
    lends_tools,
    past_deadline,
    perceive,
    reaction_to,
    released,
    renewal_due,
    sets_where_reading_starts,
    speaks_aloud,
    speech_outcome,
    spoken_wait_seconds,
    sync_checkpoints,
    sync_turned_revisions,
    turn_call_bound,
    turn_invalid,
    turn_told,
    view_turn_due,
    wakes_him,
    with_due,
    without_due,
)
from personas.entries import Answered, History, KeptLine, Unanswered, ViewSeen
from personas.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer
from personas.recall import EarlierToday, earlier_today
from personas.secret import Secret
from tests.support.fakes import line, window_of
from tests.support.history import (
    APP,
    EMPTY_HISTORY,
    LATER,
    MORNING,
    NOON,
    ROOMY,
    TABLE,
    a_sentence,
    a_view,
    an_act,
    history_holding,
    kept,
    said,
    told_line_bytes,
)
from tests.support.hopin import GENERATION, a_picture, active, invited, paused

TRANSIENT_REFUSALS = [
    AcceptInvitationRefusal.SPONSOR_AWAY,
    AcceptInvitationRefusal.TWO_SEATS,
    AcceptInvitationRefusal.BUSY,
    UnrecognisedRefusal.UNRECOGNISED,
]


def test_an_invited_call_standing_is_accepted_with_its_place_invitation_and_epoch():
    held = (invited("room-7", "inv-3", epoch=4),)

    assert accepts_for(held, Memory(), takes_turns=False) == (Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 4),)


def test_every_accept_brings_the_figure_the_agent_has():
    figure = FigureRef(url="https://figures.test/figure.glb", sha256=secrets.token_hex(32))
    held = (invited("room-7", "inv-3"), invited("table-1", "inv-4", kind="table", serves_view=True))

    accepts = accepts_for(held, Memory(), takes_turns=True, figure=figure)

    assert [accept.figure for accept in accepts] == [figure, figure]


@pytest.mark.parametrize(
    "standing",
    [
        active("room-7"),
        Standing(place=PlaceRef(kind="call", id="room-7"), state=StandingState.PAUSED),
        invited("table-1", "inv-3", kind="table"),
    ],
    ids=["active", "paused", "another kind of place"],
)
def test_a_standing_that_is_no_call_invitation_is_not_accepted(standing):
    assert accepts_for((standing,), Memory(), takes_turns=False) == ()


def test_a_stale_invitation_is_never_accepted_again():
    first = invited("room-7", "inv-3")
    memory, _ = after_refusal(
        Memory(), accepts_for((first,), Memory(), takes_turns=False)[0], AcceptInvitationRefusal.STALE_INVITATION
    )

    assert accepts_for((first,), memory, takes_turns=False) == ()
    assert accepts_for((invited("room-7", "inv-4"),), memory, takes_turns=False) == (
        Accept(PlaceRef(kind="call", id="room-7"), "inv-4", 0),
    )


@pytest.mark.parametrize("refusal", TRANSIENT_REFUSALS)
def test_any_other_refusal_is_accepted_again_at_the_next_read(refusal):
    standing = invited("room-7", "inv-3")
    attempt = accepts_for((standing,), Memory(), takes_turns=False)[0]
    memory, _ = after_refusal(Memory(), attempt, refusal)

    assert accepts_for((standing,), memory, takes_turns=False) == (attempt,)


def test_a_repeated_refusal_is_news_once_and_a_changed_one_is_news_again():
    attempt = Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0)
    memory, first = after_refusal(Memory(), attempt, AcceptInvitationRefusal.BUSY)
    memory, repeated = after_refusal(memory, attempt, AcceptInvitationRefusal.BUSY)
    _, changed = after_refusal(memory, attempt, AcceptInvitationRefusal.SPONSOR_AWAY)

    assert (first, repeated, changed) == (True, False, True)


def test_what_is_remembered_of_an_invitation_that_left_the_standing_is_dropped():
    attempt = Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0)
    memory, _ = after_refusal(Memory(), attempt, AcceptInvitationRefusal.STALE_INVITATION)
    memory, _ = after_refusal(
        memory, Accept(PlaceRef(kind="call", id="room-8"), "inv-4", 0), AcceptInvitationRefusal.BUSY
    )

    forgotten = forget_gone(memory, (invited("room-8", "inv-4"),))

    assert forgotten.refused == {"inv-4": AcceptInvitationRefusal.BUSY}


def test_a_stale_invitation_stays_refused_when_a_read_leaves_it_out_and_a_later_one_holds_it_again():
    standing = invited("room-7", "inv-3")
    memory, _ = after_refusal(
        Memory(), Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0), AcceptInvitationRefusal.STALE_INVITATION
    )

    memory = forget_gone(memory, ())

    assert accepts_for((standing,), memory, takes_turns=False) == ()


@pytest.mark.parametrize(
    ("seconds_left", "due"),
    [(RENEW_BEFORE_EXPIRY_SECONDS + 1, False), (RENEW_BEFORE_EXPIRY_SECONDS, True), (0, True), (-5, True)],
)
def test_a_token_is_due_for_renewal_before_it_expires(seconds_left, due):
    now = 500.0
    token = Token(Secret(secrets.token_hex(8)), expires_at=now + seconds_left)

    assert renewal_due(token, now) is due


def test_no_token_is_due_for_renewal():
    assert renewal_due(None, 0.0)


def test_the_backoff_grows_never_beyond_its_cap_and_is_never_zero():
    delays = [backoff_delay(failures) for failures in [*range(1, 40), 1025, 10**6]]

    assert delays == sorted(delays)
    assert all(0 < delay <= BACKOFF_CAP_SECONDS for delay in delays)
    assert delays[-1] == BACKOFF_CAP_SECONDS


@pytest.mark.parametrize(
    ("refusals", "cap_seconds", "wait"),
    [
        (1, 300.0, BACKOFF_BASE_SECONDS),
        (2, 300.0, 2 * BACKOFF_BASE_SECONDS),
        (4, 300.0, 8 * BACKOFF_BASE_SECONDS),
        (9, 300.0, 256 * BACKOFF_BASE_SECONDS),
        (10, 300.0, 300.0),
        (10**6, 300.0, 300.0),
        (3, 2.5, 2.5),
        (1, BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS),
        (7, BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS),
    ],
    ids=["first", "second", "fourth", "last doubling", "capped", "far past the cap", "odd cap", "cap at base", "base"],
)
def test_the_pause_for_a_spent_budget_doubles_from_the_backoff_base_up_to_its_cap(refusals, cap_seconds, wait):
    assert budget_pause(refusals, cap_seconds) == wait


@pytest.mark.parametrize(
    ("failure", "already_renewed", "reaction"),
    [
        (TokenStatusFailure(401), False, Suspend("http_401")),
        (TokenStatusFailure(400), False, Suspend("http_400")),
        (TokenStatusFailure(503), False, Reaction.BACKOFF),
        (StatusFailure(401), False, Reaction.RENEW),
        (StatusFailure(401), True, Suspend("http_401")),
        (StatusFailure(403), False, Suspend("http_403")),
        (StatusFailure(400), True, Suspend("http_400")),
        (StatusFailure(500), False, Reaction.BACKOFF),
        (StatusFailure(502), True, Reaction.BACKOFF),
        (Unreachable("down"), False, Reaction.BACKOFF),
        (UnusableAnswer(jsonrpc_code(INVALID_PARAMS)), False, Suspend(jsonrpc_code(INVALID_PARAMS))),
        (UnusableAnswer("http_307"), True, Suspend("http_307")),
    ],
)
def test_each_failure_makes_the_loop_renew_suspend_or_back_off(failure, already_renewed, reaction):
    assert reaction_to(failure, already_renewed=already_renewed) == reaction


def test_a_json_rpc_code_is_logged_as_the_word_and_the_signed_number():
    assert jsonrpc_code(INVALID_PARAMS) == "jsonrpc_-32602"


def test_a_secret_shows_nothing_of_itself():
    value = secrets.token_hex(8)
    secret = Secret(value)

    assert value not in repr(secret) + str(secret) + repr(Token(secret, 1.0))


def test_a_token_is_renewed_before_a_stream_the_app_bounds_could_end_on_an_expired_one():
    assert RENEW_BEFORE_EXPIRY_SECONDS > LISTEN_BOUND_SECONDS


def test_the_runner_cuts_a_stream_before_the_lease_of_its_member_would_have_ended():
    assert READ_SECONDS < LEASE_SECONDS


REFUSAL = UnusableAnswer(jsonrpc_code(INVALID_PARAMS))
RENEWABLE = StatusFailure(401)
SERVER_ERROR = StatusFailure(503)
NETWORK = Unreachable("down")
# Lower is decided first: a refusal, then a renewal, then a backoff, which a server error and the network share.
STRENGTH = {id(REFUSAL): 0, id(RENEWABLE): 1, id(SERVER_ERROR): 2, id(NETWORK): 2}
STRENGTH_AFTER_RENEWAL = {**STRENGTH, id(RENEWABLE): 0}
KINDS = [REFUSAL, RENEWABLE, SERVER_ERROR, NETWORK]
ORDERS = [order for size in (2, 3) for order in itertools.permutations(KINDS, size)]


@pytest.mark.parametrize("already_renewed", [False, True], ids=["first 401", "renewed 401"])
@pytest.mark.parametrize("order", ORDERS, ids=lambda order: "-".join(str(failure) for failure in order))
def test_the_failure_that_decides_is_the_refusal_before_the_renewal_before_the_backoff_and_else_the_first(
    order, already_renewed
):
    strength = STRENGTH_AFTER_RENEWAL if already_renewed else STRENGTH
    expected = min(order, key=lambda failure: strength[id(failure)])

    assert decisive_failure(order, already_renewed=already_renewed) is expected


@pytest.mark.parametrize(
    "leaves",
    [
        (ValueError("a defect"),),
        (Unreachable("down"), ValueError("a defect")),
        (ValueError("a defect"), StatusFailure(403)),
    ],
    ids=["only a defect", "failure then defect", "defect then refusal"],
)
def test_no_failure_decides_when_any_leaf_is_no_call_failure(leaves):
    assert decisive_failure(leaves, already_renewed=False) is None


PLACE = PlaceRef(kind="call", id="room-7")


def test_only_the_active_places_that_are_a_call_or_serve_a_view_are_read_through_a_window():
    paused = Standing(place=PlaceRef(kind="call", id="room-8"), state=StandingState.PAUSED)
    other = Standing(place=PlaceRef(kind="table", id="table-1"), state=StandingState.ACTIVE)
    table = PlaceRef(kind="table", id="table-2")

    standings = (
        active("room-7"),
        paused,
        other,
        invited("room-9", "inv-3"),
        active("table-2", "table", serves_view=True),
    )

    assert active_window_places(standings) == (PLACE, table)


@pytest.mark.parametrize(
    ("standings", "invalid"),
    [
        ((active("room-7"),), False),
        ((active("room-8"),), True),
        ((), True),
        ((paused("room-7"), active("room-8")), True),
        ((invited("room-7", "inv-3"),), True),
        ((active("room-8"), active("room-7")), False),
    ],
    ids=["active", "gone", "no standing", "paused", "invited again", "active beside another place"],
)
def test_a_turn_is_invalid_once_its_place_is_not_active_in_the_standing(standings, invalid):
    assert turn_invalid(PLACE, standings) is invalid


A_TABLE = PlaceRef(kind="table", id="table-1")
OTHER_GENERATION = "generation-2"
OTHER_PLACE = PlaceRef(kind="call", id="room-8")


@pytest.mark.parametrize(
    ("checkpoints", "standings", "expected"),
    [
        ({}, (active("room-7"),), {PLACE: Restarted()}),
        ({PLACE: Reading(GENERATION, 4)}, (active("room-7"),), {PLACE: Reading(GENERATION, 4)}),
        ({PLACE: Joined()}, (active("room-7"),), {PLACE: Joined()}),
        ({PLACE: Paused()}, (active("room-7"),), {PLACE: Paused()}),
        ({}, (paused("room-7"),), {PLACE: Paused()}),
        ({PLACE: Reading(GENERATION, 4)}, (paused("room-7"),), {PLACE: Paused()}),
        ({PLACE: Joined()}, (paused("room-7"),), {PLACE: Paused()}),
        ({PLACE: Reading(GENERATION, 4)}, (invited("room-7", "inv-3"),), {}),
        ({PLACE: Joined()}, (invited("room-7", "inv-3"),), {}),
        ({PLACE: Reading(GENERATION, 4)}, (), {}),
        ({OTHER_PLACE: Reading(GENERATION, 2)}, (active("room-7"),), {PLACE: Restarted()}),
        ({}, (active("room-7", serves_view=True),), {PLACE: Restarted()}),
        ({}, (active("table-1", "table", serves_view=True),), {A_TABLE: Restarted()}),
        (
            {A_TABLE: Reading(GENERATION, 4)},
            (active("table-1", "table", serves_view=True),),
            {A_TABLE: Reading(GENERATION, 4)},
        ),
        ({A_TABLE: Joined()}, (active("table-1", "table", serves_view=True),), {A_TABLE: Joined()}),
        ({}, (paused("table-1", "table", serves_view=True),), {A_TABLE: Paused()}),
        ({A_TABLE: Reading(GENERATION, 4)}, (active("table-1", "table"),), {}),
        ({A_TABLE: Reading(GENERATION, 4)}, (invited("table-1", "inv-3", kind="table", serves_view=True),), {}),
    ],
    ids=[
        "an active place this process did not join starts as restarted",
        "an active place keeps its reading",
        "an active place keeps its join",
        "an active place keeps its pause",
        "a paused place with no entry is paused",
        "a paused place that was read is paused",
        "a paused place that was joined is paused",
        "a place invited again drops its reading",
        "a place invited again drops its join",
        "a place gone from the standing drops its entry",
        "an entry of a place gone is dropped beside a new one",
        "a call place that serves a view is read through its window",
        "an active place that serves a view starts as restarted",
        "an active place that serves a view keeps its reading",
        "an active place that serves a view keeps its join",
        "a paused place that serves a view is paused",
        "a place that stops serving a view drops its reading",
        "a place that serves a view and is invited again drops its reading",
    ],
)
def test_the_standing_decides_which_checkpoints_exist(checkpoints, standings, expected):
    assert sync_checkpoints(checkpoints, standings, ()) == expected


@pytest.mark.parametrize(
    ("kind", "place"), [("call", PLACE), ("table", A_TABLE)], ids=["a call place", "a table place"]
)
def test_a_place_that_serves_a_view_again_after_an_active_read_without_one_starts_as_restarted(kind, place):
    before = (Standing(place=place, state=StandingState.ACTIVE),)
    now = (Standing(place=place, state=StandingState.ACTIVE, serves_view=True),)

    assert sync_checkpoints({place: Reading(GENERATION, 4)}, now, before) == {place: Restarted()}


@pytest.mark.parametrize(
    ("place", "now", "before"),
    [
        (A_TABLE, (active("table-1", "table"),), (invited("table-1", "inv-3", kind="table"),)),
        (
            A_TABLE,
            (active("table-1", "table", serves_view=True),),
            (active("table-1", "table"),),
        ),
        (
            PLACE,
            (active("room-7", serves_view=True),),
            (active("room-7"),),
        ),
    ],
    ids=[
        "an active place of another kind without a view",
        "a place of another kind that serves its view after an active read without one",
        "a call that serves its view after an active read without one",
    ],
)
def test_the_join_of_an_accepted_place_lasts_until_its_window_is_read_whatever_the_views_it_showed(place, now, before):
    assert sync_checkpoints({place: Joined()}, now, before) == {place: Joined()}


def test_a_join_ends_with_a_pause_of_a_place_of_another_kind_that_serves_no_view():
    assert sync_checkpoints({A_TABLE: Joined()}, (paused("table-1", "table"),), ()) == {}


def test_a_place_of_another_kind_that_serves_no_view_gets_no_checkpoint():
    table = Standing(place=PlaceRef(kind="table", id="table-1"), state=StandingState.ACTIVE)

    assert sync_checkpoints({}, (table,), ()) == {}


@pytest.mark.parametrize(
    ("checkpoint", "sets_start"),
    [(Restarted(), True), (Paused(), True), (Joined(), False), (Reading(GENERATION, 2), False)],
    ids=["restarted", "paused", "joined", "reading"],
)
def test_only_a_restarted_or_paused_checkpoint_waits_for_a_read_to_set_where_reading_starts(checkpoint, sets_start):
    assert sets_where_reading_starts(checkpoint) is sets_start


@pytest.mark.parametrize(
    "checkpoint",
    [Restarted(), Paused(), Joined(), Reading(GENERATION, 1), Reading(GENERATION, 1, VoiceState.UNAVAILABLE)],
    ids=["restarted", "paused", "joined", "reading", "reading that stated unavailable"],
)
@pytest.mark.parametrize("stated", [VoiceState.UNAVAILABLE, VoiceState.AVAILABLE, None])
def test_a_read_keeps_the_voice_state_its_window_states_even_when_it_takes_no_line(checkpoint, stated):
    window = PlaceWindow(generation=GENERATION, starts_at=1, entries=(line(1, "a"),), voice_state=stated)

    assert perceive(window, checkpoint).checkpoint.voice_state is stated


@pytest.mark.parametrize(
    ("checkpoint", "window", "expected"),
    [
        (Restarted(), window_of(line(1, "a"), line(2, "b"), line(3, "c")), (Reading(GENERATION, 3), [], False)),
        (Paused(), window_of(line(1, "a"), line(2, "b"), line(3, "c")), (Reading(GENERATION, 3), [], False)),
        (Restarted(), PlaceWindow(generation=GENERATION, starts_at=5, entries=()), (Reading(GENERATION, 4), [], False)),
        (Restarted(), window_of(line(4, "a"), line(5, "b")), (Reading(GENERATION, 5), [], False)),
        (Joined(), window_of(line(1, "a"), line(2, "b"), line(3, "c")), (Reading(GENERATION, 3), [1, 2, 3], False)),
        (Joined(), PlaceWindow(generation=GENERATION, starts_at=1, entries=()), (Reading(GENERATION, 0), [], False)),
        (Joined(), window_of(line(4, "a"), line(5, "b")), (Reading(GENERATION, 5), [4, 5], True)),
        (
            Reading(GENERATION, 2),
            window_of(line(1, "a"), line(2, "b"), line(3, "c")),
            (Reading(GENERATION, 3), [3], False),
        ),
        (
            Reading(GENERATION, 3),
            window_of(line(1, "a"), line(2, "b"), line(3, "c")),
            (Reading(GENERATION, 3), [], False),
        ),
        (Reading(GENERATION, 1), window_of(line(5, "a"), line(6, "b")), (Reading(GENERATION, 6), [5, 6], True)),
        (Reading(GENERATION, 4), window_of(line(5, "a"), line(6, "b")), (Reading(GENERATION, 6), [5, 6], False)),
        (
            Reading(OTHER_GENERATION, 3),
            window_of(line(1, "a"), line(2, "b"), line(3, "c")),
            (Reading(GENERATION, 3), [1, 2, 3], False),
        ),
        (Reading(OTHER_GENERATION, 3), window_of(line(4, "a"), line(5, "b")), (Reading(GENERATION, 5), [4, 5], True)),
    ],
    ids=[
        "restarted: reads to the highest line and takes none",
        "paused: reads to the highest line and takes none",
        "restarted on an empty window: reads to the line before the window",
        "restarted on a window that starts late: reads to its highest line",
        "joined: every line is new",
        "joined on an empty window: nothing is new",
        "joined on a window that starts after seq 1: lines were missed",
        "reading: the lines after the last one are new",
        "reading: nothing after the last one",
        "reading: the window moved past the last one",
        "reading: the window starts right after the last one",
        "another generation: reading starts over at seq 1",
        "another generation on a window that starts after seq 1: lines were missed",
    ],
)
def test_a_window_is_perceived_from_its_checkpoint(checkpoint, window, expected):
    reading, new, missed = expected
    perception = perceive(window, checkpoint)

    assert perception == Perception(reading, perception.new, missed)
    assert [each.seq for each in perception.new] == new


@pytest.mark.parametrize(
    ("taken", "serves_view", "expected"),
    [
        (({},), False, True),
        (({"own": True},), False, False),
        (({"own": True}, {}), False, True),
        ((), False, False),
        (({"act": True},), False, False),
        (({"act": True}, {}), False, True),
        (({"own": True, "act": True},), False, False),
        (({},), True, False),
        (({"act": True},), True, False),
        (({"own": True}, {}, {"act": True}), True, False),
    ],
    ids=[
        "someone else's line",
        "his own line",
        "one of them someone else's",
        "no line",
        "an act line",
        "an act line beside a line said",
        "his own act line",
        "a line at a place with a view",
        "an act line at a place with a view",
        "lines of every kind at a place with a view",
    ],
)
def test_a_line_wakes_him_unless_it_is_his_own_or_an_act_or_the_place_serves_a_view(taken, serves_view, expected):
    lines = [line(seq, "Hello everyone", **marks) for seq, marks in enumerate(taken, 1)]

    assert wakes_him(lines, serves_view=serves_view) is expected


CALL_AT_A = AppPlace("https://a.test/mcp", PlaceRef(kind="call", id="room-1"))
CALL_AT_B = AppPlace("https://b.test/mcp", PlaceRef(kind="call", id="room-2"))
TABLE_AT_A = AppPlace("https://a.test/mcp", PlaceRef(kind="table", id="table-1"))


def _due_in_order(*due: tuple[AppPlace, Lines | Waiting]) -> Turns:
    turns = Turns()
    for at, occasion in due:
        turns = with_due(turns, at, occasion)
    return turns


def test_lines_due_at_a_place_in_one_generation_join_and_keep_the_place_where_it_was_in_the_order():
    turns = _due_in_order(
        (CALL_AT_A, Lines("g1", frozenset({1}), missed=False)), (CALL_AT_B, Lines("g1", frozenset({4}), missed=False))
    )

    turns = with_due(turns, CALL_AT_A, Lines("g1", frozenset({2, 3}), missed=True))

    assert turns.due == {
        CALL_AT_A: Lines("g1", frozenset({1, 2, 3}), missed=True),
        CALL_AT_B: Lines("g1", frozenset({4}), missed=False),
    }
    assert list(turns.due) == [CALL_AT_A, CALL_AT_B]


def test_lines_of_another_generation_and_a_newer_revision_replace_what_was_due_at_the_place_in_its_position():
    turns = _due_in_order(
        (CALL_AT_A, Lines("g1", frozenset({1}), missed=True)), (TABLE_AT_A, Waiting(1)), (CALL_AT_B, Waiting(5))
    )

    turns = with_due(turns, CALL_AT_A, Lines("g2", frozenset({1}), missed=False))
    turns = with_due(turns, TABLE_AT_A, Waiting(2))

    assert turns.due == {
        CALL_AT_A: Lines("g2", frozenset({1}), missed=False),
        TABLE_AT_A: Waiting(2),
        CALL_AT_B: Waiting(5),
    }
    assert list(turns.due) == [CALL_AT_A, TABLE_AT_A, CALL_AT_B]


def test_a_turn_without_an_answer_makes_its_place_due_again_with_what_it_was_for_at_the_back_of_the_order():
    turns = _due_in_order((CALL_AT_B, Waiting(1)))

    for occasion in (Waiting(2), Lines("g1", frozenset({3}), missed=True)):
        again = due_again(turns, CALL_AT_A, occasion)

        assert again.due == {CALL_AT_B: Waiting(1), CALL_AT_A: occasion}
        assert list(again.due) == [CALL_AT_B, CALL_AT_A]


def test_a_place_that_became_due_meanwhile_keeps_its_newer_occasion_and_only_lines_of_its_generation_join_the_retry():
    turns = _due_in_order(
        (CALL_AT_A, Lines("g2", frozenset({5}), missed=False)),
        (TABLE_AT_A, Waiting(3)),
        (CALL_AT_B, Lines("g1", frozenset({7}), missed=False)),
    )

    turns = due_again(turns, CALL_AT_A, Lines("g1", frozenset({1}), missed=False))
    turns = due_again(turns, TABLE_AT_A, Waiting(2))
    turns = due_again(turns, CALL_AT_B, Lines("g1", frozenset({4}), missed=True))

    assert turns.due == {
        CALL_AT_A: Lines("g2", frozenset({5}), missed=False),
        TABLE_AT_A: Waiting(3),
        CALL_AT_B: Lines("g1", frozenset({4, 7}), missed=True),
    }
    assert list(turns.due) == [CALL_AT_A, TABLE_AT_A, CALL_AT_B]


def test_the_turn_goes_to_the_place_that_waited_longest_and_only_in_its_app():
    turns = _due_in_order((CALL_AT_B, Waiting(1)), (CALL_AT_A, Waiting(2)))

    assert claimed(turns, CALL_AT_A.app) is None
    after, at, occasion = claimed(turns, CALL_AT_B.app) or pytest.fail("the oldest place is at its app")
    assert (after.running, at, occasion) == (True, CALL_AT_B, Waiting(1))
    assert after.due == {CALL_AT_A: Waiting(2)}


def test_no_turn_starts_while_one_runs_or_when_nothing_is_due():
    assert claimed(replace(_due_in_order((CALL_AT_A, Waiting(1))), running=True), CALL_AT_A.app) is None
    assert claimed(Turns(), CALL_AT_A.app) is None


def test_a_released_turn_leaves_what_is_due_and_lets_the_next_one_start():
    running = replace(_due_in_order((CALL_AT_A, Waiting(1))), running=True)

    after = released(running)

    assert after.running is False
    assert after.due == running.due
    assert claimed(after, CALL_AT_A.app) is not None


def test_the_places_that_leave_are_no_longer_due_and_the_rest_keep_their_order():
    turns = _due_in_order((CALL_AT_A, Waiting(1)), (CALL_AT_B, Waiting(1)), (TABLE_AT_A, Waiting(1)))

    after = without_due(turns, lambda at: at.app == CALL_AT_A.app)

    assert list(after.due) == [CALL_AT_B]


def test_a_due_place_of_an_app_leaves_with_its_call_or_table_and_stays_while_it_is_active():
    standings = (active("room-1"), active("table-1", "table", serves_view=True), paused("room-9"))
    app = CALL_AT_A.app
    due = [CALL_AT_A, TABLE_AT_A, AppPlace(app, PlaceRef(kind="call", id="room-9")), CALL_AT_B]

    leaving = left_standing(app, standings)

    assert [leaving(at) for at in due] == [False, False, True, False]


@pytest.mark.parametrize(
    ("standings", "holds"),
    [
        ((), False),
        ((paused("room-1"), invited("room-2", "invitation-1")), False),
        ((paused("room-1"), active("room-2")), True),
        ((active("table-1", "table", serves_view=True),), True),
    ],
    ids=["none", "paused and invited only", "an active call", "an active table"],
)
def test_he_holds_a_place_at_an_app_while_any_of_his_standings_there_is_active(standings, holds):
    assert holds_a_place(standings) is holds


@pytest.mark.parametrize(
    ("binds_tools", "standings", "lends"),
    [
        (True, (active("room-1"),), True),
        (False, (active("room-1"),), False),
        (True, (paused("room-1"),), False),
        (False, (), False),
    ],
    ids=["binds tools and holds a place", "binds no tools", "holds no place", "neither"],
)
def test_an_app_lends_its_tools_only_while_it_binds_tools_and_he_holds_a_place_there(binds_tools, standings, lends):
    assert lends_tools(binds_tools, standings) is lends


def test_the_apps_a_turn_holds_are_those_he_holds_a_place_at_other_than_its_own():
    placed = frozenset({"https://a.test/mcp", "https://b.test/mcp", "https://c.test/mcp"})

    assert held_apps("https://a.test/mcp", placed) == {"https://b.test/mcp", "https://c.test/mcp"}
    assert held_apps("https://d.test/mcp", placed) == placed
    assert held_apps("https://a.test/mcp", frozenset({"https://a.test/mcp"})) == frozenset()


def _declaration(name: str) -> ToolDeclaration:
    return ToolDeclaration(name=name, description=secrets.token_hex(4), parameters={"type": "object"})


def test_the_bound_tools_are_declared_in_the_order_they_are_bound():
    declared = [_declaration("a"), _declaration("b"), _declaration("c")]

    assert bound_declarations(declared, ["c", "a"]) == (declared[2], declared[0])


@pytest.mark.parametrize(
    ("listed", "bound", "expected"),
    [
        (["a", "b"], ["a", "b"], ["a", "b"]),
        (["c", "a", "b"], ["b", "a"], ["b", "a"]),
        (["a", "c"], ["a", "b", "c"], ["a", "c"]),
        ([], ["a", "b"], []),
        (["a", "extra"], ["a"], ["a"]),
    ],
    ids=["all listed", "binding order", "a subset", "none listed", "an unbound tool stays out"],
)
def test_a_turn_binds_the_bound_tools_the_app_lists_in_binding_order(listed, bound, expected):
    declared = {name: _declaration(name) for name in {*listed, *bound}}

    assert bound_declarations([declared[name] for name in listed], bound) == tuple(declared[name] for name in expected)


ONCE_DROPPED = History((), left_out=1, heard=0)


def told_a_written_turn(
    earlier: EarlierToday | None, history: History = EMPTY_HISTORY, *, missed: bool = False, aloud: bool = False
) -> str:
    """The message of a turn at ``LATER`` in the place of the morning visit, for lines that are none yet."""
    return turn_told(
        "Mozart",
        APP,
        MORNING.place,
        Lines(MORNING.generation, frozenset(), missed),
        history,
        earlier,
        LATER,
        aloud=aloud,
    )


def test_a_turn_is_told_each_earlier_visit_how_long_ago_it_was_its_lines_their_speakers_and_his_own():
    mine, theirs = secrets.token_hex(4), secrets.token_hex(4)
    earlier = earlier_today(
        [
            kept(NOON, 1, theirs, hours_before=4),
            kept(NOON, 2, mine, hours_before=4, own=True, spoken=True, name="Mozart"),
        ],
        LATER,
        ROOMY,
        EMPTY_HISTORY,
    )

    facts = json.loads(told_a_written_turn(earlier))

    assert facts["earlier_today"] == {
        "older_dropped": False,
        "visits": [
            {
                "minutes_ago": 240,
                "lines": [
                    {"name": "Ada", "kind": "message", "text": theirs, "own": False, "spoken": False},
                    {"name": "Mozart", "kind": "message", "text": mine, "own": True, "spoken": True},
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("earlier", "keys"),
    [
        (None, ["you", "history", "woke"]),
        (EarlierToday((), older_dropped=False), ["you", "earlier_today", "history", "woke"]),
    ],
    ids=["without a memory he is told no day", "with a memory the day comes between his name and his history"],
)
def test_a_turn_is_told_his_name_then_the_day_then_his_history_and_last_what_woke_him(earlier, keys):
    assert list(json.loads(told_a_written_turn(earlier))) == keys


@pytest.mark.parametrize("has_voice", [True, False])
@pytest.mark.parametrize("declared_names", [[SPEAK_TOOL, "say"], ["say"]], ids=["declares speak", "declares none"])
def test_the_places_of_an_app_are_spoken_only_for_an_agent_with_a_voice_and_an_app_that_declares_speak(
    has_voice, declared_names
):
    declared = [_declaration(name) for name in declared_names]

    assert is_voice_app(has_voice, declared) is (has_voice and SPEAK_TOOL in declared_names)


@pytest.mark.parametrize(
    ("duration_ms", "waited"),
    [(1500, 1.5), (1, 0.001), (int(SPOKEN_WAIT_BOUND_SECONDS * 1000), SPOKEN_WAIT_BOUND_SECONDS)],
    ids=["a sentence", "the shortest", "exactly the bound"],
)
def test_the_voice_waits_as_long_as_a_sentence_plays_up_to_the_bound(duration_ms, waited):
    assert spoken_wait_seconds(duration_ms) == waited


def test_the_voice_waits_no_longer_than_the_bound_whatever_the_app_says():
    assert spoken_wait_seconds(int(SPOKEN_WAIT_BOUND_SECONDS * 1000) + 1) == SPOKEN_WAIT_BOUND_SECONDS


def test_a_turns_call_bound_is_the_sum_of_the_bounds_of_the_apps_whose_tools_it_holds_plus_the_call_it_may_hold_back():
    assert turn_call_bound([4, 5, 2]) == 12
    assert turn_call_bound([4]) == 5


def test_a_turn_that_holds_no_tool_has_no_call_bound():
    assert turn_call_bound([]) is None


@pytest.mark.parametrize(("made", "within"), [(0, True), (2, True), (3, False), (4, False)])
def test_an_app_takes_calls_until_its_own_bound_is_reached(made, within):
    assert call_within_bound(made, 3) is within


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (StatusFailure(503), "http_503"),
        (UnusableAnswer("invalid_answer"), "invalid_answer"),
        (Unreachable("cut"), NETWORK_FAILURE_CODE),
    ],
    ids=["a status", "an unusable answer", "the network"],
)
def test_a_failure_is_logged_with_its_status_its_answer_code_or_the_word_for_the_network(failure, code):
    assert failure_code(failure) == code


OTHER_TABLE = PlaceRef(kind="table", id="table-2")


def serving(place: PlaceRef = TABLE, *, state: StandingState = StandingState.ACTIVE) -> Standing:
    if state is StandingState.INVITED:
        return invited(place.id, "inv-3", kind=place.kind, serves_view=True)
    return Standing(place=place, state=state, serves_view=True)


@pytest.mark.parametrize("serves_view", [False, True], ids=["not yet serving a view", "serving a view"])
def test_an_invited_place_that_is_no_call_is_accepted_wherever_the_loop_takes_turns_whether_or_not_it_serves_a_view(
    serves_view,
):
    standing = invited("table-1", "inv-3", kind="table", serves_view=serves_view)

    assert accepts_for((standing,), Memory(), takes_turns=True) == (Accept(TABLE, "inv-3", 0),)
    assert accepts_for((standing,), Memory(), takes_turns=False) == ()


def test_an_invited_call_that_serves_a_view_is_accepted_only_where_the_loop_takes_turns():
    standing = invited("room-7", "inv-3", serves_view=True)

    assert accepts_for((standing,), Memory(), takes_turns=True) == (Accept(PLACE, "inv-3", 0),)
    assert accepts_for((standing,), Memory(), takes_turns=False) == ()


@pytest.mark.parametrize("kind", ["table", "call"])
def test_an_invitation_the_loop_leaves_out_is_news_once_with_its_standing_and_news_again_in_a_new_invitation(kind):
    standing = invited("place-1", "inv-3", kind=kind, serves_view=True)
    memory, first = left_out_for((standing,), Memory(), takes_turns=False)
    memory, repeated = left_out_for((standing,), memory, takes_turns=False)
    _, renewed = left_out_for((invited("place-1", "inv-4", kind=kind, serves_view=True),), memory, takes_turns=False)

    assert (first, repeated, len(renewed)) == ((standing,), (), 1)


@pytest.mark.parametrize(
    ("standing", "takes_turns"),
    [
        (invited("room-7", "inv-3"), False),
        (invited("table-1", "inv-3", kind="table"), True),
        (active("table-1", "table"), False),
    ],
    ids=["a call", "a table the loop takes", "a standing that is no invitation"],
)
def test_an_invitation_the_loop_accepts_or_holds_is_no_news_of_being_left_out(standing, takes_turns):
    assert left_out_for((standing,), Memory(), takes_turns=takes_turns)[1] == ()


def test_an_invitation_left_out_is_reported_again_after_the_standing_stopped_holding_it():
    standing = invited("table-1", "inv-3", kind="table")
    memory, _ = left_out_for((standing,), Memory(), takes_turns=False)

    memory = forget_gone(memory, ())

    assert left_out_for((standing,), memory, takes_turns=False)[1] == (standing,)


def test_an_invited_view_place_whose_invitation_was_refused_as_stale_is_not_accepted_again():
    standing = serving(state=StandingState.INVITED)
    memory, _ = after_refusal(
        Memory(), accepts_for((standing,), Memory(), takes_turns=True)[0], AcceptInvitationRefusal.STALE_INVITATION
    )

    assert accepts_for((standing,), memory, takes_turns=True) == ()


def test_a_call_place_that_serves_a_view_is_a_view_place_and_its_window_is_kept_and_read():
    standing = Standing(place=PLACE, state=StandingState.ACTIVE, serves_view=True)

    assert active_window_places((standing,)) == (PLACE,)
    assert sync_checkpoints({}, (standing,), ()) == {PLACE: Restarted()}
    assert active_view_places((standing,)) == (PLACE,)
    assert accepts_for((invited("room-7", "inv-3", serves_view=True),), Memory(), takes_turns=False) == ()


def test_only_the_active_places_that_serve_a_view_are_view_places_in_the_order_of_the_standing():
    standings = (
        serving(OTHER_TABLE),
        serving(TABLE, state=StandingState.PAUSED),
        active("room-7"),
        active("table-3", "table"),
        serving(PlaceRef(kind="table", id="table-4")),
        serving(state=StandingState.INVITED),
    )

    assert active_view_places(standings) == (OTHER_TABLE, PlaceRef(kind="table", id="table-4"))


def test_the_standing_decides_which_turned_revisions_are_kept():
    turned = {TABLE: 4, OTHER_TABLE: 1}

    assert sync_turned_revisions(turned, (serving(TABLE), serving(OTHER_TABLE, state=StandingState.PAUSED))) == {
        TABLE: 4
    }
    assert sync_turned_revisions(turned, ()) == {}


@pytest.mark.parametrize(
    ("turned_revision", "view", "due"),
    [
        (None, a_view(1), True),
        (None, a_view(1, awaits_you=False), False),
        (1, a_view(1), False),
        (1, a_view(2), True),
        (1, a_view(2, awaits_you=False), False),
    ],
    ids=[
        "a revision that waits gets its first turn",
        "a revision that does not wait gets none",
        "a revision a turn was started for gets no second turn",
        "a new revision that waits gets its turn",
        "a new revision that does not wait gets none",
    ],
)
def test_a_view_is_due_a_turn_while_it_waits_at_a_revision_no_turn_was_started_for(turned_revision, view, due):
    assert view_turn_due(turned_revision, view) is due


@pytest.fixture
def told_one_bytes() -> int:
    """The bytes the line ``said(1, "one")`` is told as in the history."""
    return told_line_bytes(APP, "room-7", "one", name="Bo")


def test_a_view_turn_is_told_its_revision_as_the_one_entry_marked_new_and_woke_names_only_why_app_and_place():
    view = PlaceView(revision=3, awaits_you=True, state={"hand": ["x", {"seen": True}], "umläut": "ä"})
    history = history_after(
        ONCE_DROPPED,
        (
            said(1, "Spiel den Buben", name="Ada", visit=NOON, spoken=True),
            ViewSeen(APP, TABLE, view, LATER - timedelta(seconds=7)),
        ),
        ROOMY.size_bytes,
    )

    told = turn_told("Mozart", APP, TABLE, Waiting(3), history, None, LATER, aloud=False)

    assert json.loads(told) == {
        "you": "Mozart",
        "history": {
            "older_dropped": True,
            "entries": [
                {
                    "entry": "line",
                    "app": NOON.app,
                    "place": {"kind": "call", "id": "room-8"},
                    "seconds_ago": 0,
                    "seq": 1,
                    "name": "Ada",
                    "kind": "message",
                    "text": "Spiel den Buben",
                    "own": False,
                    "spoken": True,
                    "new": False,
                },
                {
                    "entry": "view",
                    "app": APP,
                    "place": {"kind": "table", "id": "table-1"},
                    "seconds_ago": 7,
                    "revision": 3,
                    "awaits_you": True,
                    "state": view.state,
                    "new": True,
                },
            ],
        },
        "woke": {"why": "view", "app": APP, "place": {"kind": "table", "id": "table-1"}},
    }
    assert "ä" in told
    assert told.count(json.dumps(view.state, ensure_ascii=False)) == 1


@pytest.mark.parametrize(
    ("app", "place", "revision"),
    [(APP, TABLE, 4), (APP, OTHER_TABLE, 3), ("https://other.test/mcp", TABLE, 3)],
    ids=["another revision", "another place", "another app"],
)
def test_a_view_turn_marks_new_only_the_entry_of_its_app_place_and_revision(app, place, revision):
    held = ViewSeen(app, place, a_view(revision), LATER)
    admitted = ViewSeen(APP, TABLE, a_view(3), LATER)

    told = json.loads(
        turn_told("Mozart", APP, TABLE, Waiting(3), history_holding((held, admitted)), None, LATER, aloud=False)
    )

    assert [each["new"] for each in told["history"]["entries"]] == [False, True]


def test_a_written_turn_marks_no_view_new():
    history = history_holding((ViewSeen(APP, MORNING.place, a_view(1), LATER), said(1, "hello")))

    told = json.loads(told_a_written_turn(None, history))

    assert [(each["entry"], each["new"]) for each in told["history"]["entries"]] == [("view", False), ("line", False)]


@pytest.mark.parametrize("missed", [False, True])
def test_a_written_turn_is_told_in_which_place_it_woke_and_whether_older_lines_were_not_seen(missed):
    woke = json.loads(told_a_written_turn(None, missed=missed))["woke"]

    assert woke == {"why": "lines", "app": APP, "place": {"kind": "call", "id": "room-7"}, "missed": missed}


def test_a_turn_that_is_spoken_is_told_so_in_what_woke_him_and_any_other_turn_is_told_no_such_key():
    spoken = json.loads(told_a_written_turn(None, aloud=True))["woke"]

    assert spoken == {
        "why": "lines",
        "app": APP,
        "place": {"kind": "call", "id": "room-7"},
        "missed": False,
        "aloud": True,
    }
    assert "aloud" not in json.loads(told_a_written_turn(None))["woke"]


LINES = Lines(MORNING.generation, frozenset({1}), missed=False)


@pytest.mark.parametrize(
    ("voice_app", "occasion", "checkpoint", "aloud"),
    [
        pytest.param(True, LINES, None, True, id="lines at a voice app"),
        pytest.param(False, LINES, None, False, id="lines elsewhere"),
        pytest.param(True, Waiting(3), None, False, id="a view revision at a voice app"),
        pytest.param(False, Waiting(3), None, False, id="a view revision elsewhere"),
        pytest.param(True, LINES, Reading(GENERATION, 1, VoiceState.AVAILABLE), True, id="a voice stated available"),
        pytest.param(True, LINES, Reading(GENERATION, 1), True, id="a voice stated nothing"),
        pytest.param(
            True, LINES, Reading(GENERATION, 1, VoiceState.UNAVAILABLE), False, id="a voice stated unavailable"
        ),
        pytest.param(False, LINES, Reading(GENERATION, 1, VoiceState.AVAILABLE), False, id="available, no voice app"),
        pytest.param(True, LINES, Restarted(), True, id="a place restarted states nothing"),
        pytest.param(True, LINES, Paused(), True, id="a place paused states nothing"),
        pytest.param(True, LINES, Joined(), True, id="a place just joined states nothing"),
    ],
)
def test_a_turn_for_lines_at_a_voice_app_is_spoken_unless_the_last_window_states_his_voice_unavailable(
    voice_app, occasion, checkpoint, aloud
):
    assert speaks_aloud(voice_app, occasion, checkpoint) is aloud


def test_a_turn_told_an_empty_history_is_told_no_entries_and_none_dropped():
    assert json.loads(told_a_written_turn(None))["history"] == {"older_dropped": False, "entries": []}


def test_a_history_line_is_new_only_when_the_turn_is_for_that_line_of_that_visit():
    evening = replace(MORNING, generation="evening")
    history = history_holding(
        (
            KeptLine(
                NOON,
                1,
                LATER - timedelta(seconds=300),
                "Bo",
                "message",
                "elsewhere",
                own=False,
                spoken=False,
                cut=False,
                act=False,
            ),
            KeptLine(
                evening,
                1,
                LATER - timedelta(seconds=200),
                "Bo",
                "message",
                "an earlier run",
                own=False,
                spoken=False,
                cut=False,
                act=False,
            ),
            KeptLine(
                MORNING,
                1,
                LATER - timedelta(seconds=90),
                "Ada",
                "message",
                "woke him",
                own=False,
                spoken=False,
                cut=False,
                act=False,
            ),
            KeptLine(
                MORNING,
                2,
                LATER - timedelta(seconds=1),
                "Mozart",
                "message",
                "his own",
                own=True,
                spoken=False,
                cut=False,
                act=False,
            ),
        )
    )
    occasion = Lines(MORNING.generation, frozenset({1}), missed=False)

    told = json.loads(turn_told("Mozart", APP, MORNING.place, occasion, history, None, LATER, aloud=False))

    assert [
        (each["place"]["id"], each["seq"], each["text"], each["own"], each["new"])
        for each in told["history"]["entries"]
    ] == [
        ("room-8", 1, "elsewhere", False, False),
        ("room-7", 1, "an earlier run", False, False),
        ("room-7", 1, "woke him", False, True),
        ("room-7", 2, "his own", True, False),
    ]


def test_the_history_is_told_in_the_order_it_holds_its_entries():
    history = history_holding((said(2, "later", visit=NOON), said(1, "earlier")))

    told = json.loads(told_a_written_turn(None, history))

    assert [entry["text"] for entry in told["history"]["entries"]] == ["later", "earlier"]


def test_an_accept_shows_neither_its_figure_nor_its_voice_in_its_repr():
    voice = f"Voice_{secrets.token_hex(6)}"
    figure = FigureRef(url=f"https://figures.test/{secrets.token_hex(6)}.glb", sha256=secrets.token_hex(32))

    shown = repr(Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 4, figure, voice))

    assert voice not in shown
    assert figure.url not in shown
    assert figure.sha256 not in shown


def test_the_places_and_lines_a_turn_is_due_for_show_no_name_no_text_and_no_state_in_their_repr():
    marker = secrets.token_hex(8)
    taken = said(1, marker, name=marker)
    at = AppPlace(taken.visit.app, taken.visit.place)
    lines = Lines(taken.visit.generation, frozenset({taken.seq}), missed=False)
    turns = with_due(with_due(Turns(), at, lines), AppPlace(APP, TABLE), Waiting(revision=1))

    assert marker not in repr(at)
    assert marker not in repr(lines)
    assert marker not in repr(turns)


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
            '{"entry": "act", "app": "https://app.test/mcp", "seconds_ago": 0, "tool": "say",'
            ' "arguments": {"text": "hi"}, "outcome": {"answered": true, "is_error": false, "text": "posted"},'
            ' "new": false}',
            id="an answer",
        ),
        pytest.param(
            an_act(outcome=Unanswered("network")),
            '{"entry": "act", "app": "https://app.test/mcp", "seconds_ago": 0, "tool": "say",'
            ' "arguments": {"text": "hi"}, "outcome": {"answered": false, "code": "network"}, "new": false}',
            id="no answer",
        ),
        pytest.param(
            an_act(arguments={"text": 'ä♠"\n'}),
            '{"entry": "act", "app": "https://app.test/mcp", "seconds_ago": 0, "tool": "say",'
            ' "arguments": {"text": "ä♠\\"\\n"},'
            ' "outcome": {"answered": true, "is_error": false, "text": "posted"}, "new": false}',
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
def test_the_bound_counts_an_act_by_the_utf8_bytes_of_its_told_json_with_its_age_as_zero(
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


def test_an_answered_act_is_told_with_its_app_tool_arguments_answer_and_age_and_is_never_new():
    act = an_act(arguments={"place": "room-7", "text": "ä"}, outcome=Answered("gone", is_error=True))

    told_entries = json.loads(told_a_written_turn(None, history_holding((act,))))["history"]["entries"]

    assert told_entries == [
        {
            "entry": "act",
            "app": APP,
            "seconds_ago": 9,
            "tool": "say",
            "arguments": {"place": "room-7", "text": "ä"},
            "outcome": {"answered": True, "is_error": True, "text": "gone"},
            "new": False,
        }
    ]


def test_an_unanswered_act_is_told_with_the_code_and_no_text():
    act = an_act(outcome=Unanswered(TURN_CANCELLED_CODE))

    told_entry = json.loads(told_a_written_turn(None, history_holding((act,))))["history"]["entries"][0]

    assert told_entry["outcome"] == {"answered": False, "code": TURN_CANCELLED_CODE}


def test_a_view_turn_marks_no_act_new():
    history = history_holding((an_act(), ViewSeen(APP, TABLE, a_view(3), LATER)))

    told_entries = json.loads(turn_told("Mozart", APP, TABLE, Waiting(3), history, None, LATER, aloud=False))[
        "history"
    ]["entries"]

    assert [each["new"] for each in told_entries] == [False, True]


@pytest.mark.parametrize(
    ("ended", "cancelling", "expected"),
    [
        pytest.param(
            ToolOutcome("posted", is_error=False),
            True,
            Answered("posted", is_error=False),
            id="an answer while cancelled",
        ),
        pytest.param(ToolOutcome("gone", is_error=True), False, Answered("gone", is_error=True), id="an error answer"),
        pytest.param(
            ToolOutcome("seen", is_error=False, image=a_picture()),
            False,
            Answered("seen", is_error=False, image_shown=True),
            id="an answer with a picture",
        ),
        pytest.param(StatusFailure(502), True, Unanswered(TURN_CANCELLED_CODE), id="a failure while cancelled"),
        pytest.param(StatusFailure(502), False, Unanswered("http_502"), id="a failure"),
        pytest.param(Unreachable("cut"), False, Unanswered(NETWORK_FAILURE_CODE), id="an unreachable app"),
        pytest.param(asyncio.CancelledError(), True, Unanswered(TURN_CANCELLED_CODE), id="a cancellation"),
        pytest.param(RuntimeError("defect"), True, Unanswered(TURN_CANCELLED_CODE), id="another error while cancelled"),
        pytest.param(RuntimeError("defect"), False, None, id="another error"),
    ],
)
def test_how_a_request_ended_decides_the_outcome_of_the_act_it_made(ended, cancelling, expected):
    assert act_outcome(ended, cancelling=cancelling) == expected


@pytest.mark.parametrize(
    ("outcome", "told_outcome"),
    [
        pytest.param(Spoken(duration_ms=1500), {"answered": True}, id="spoken"),
        pytest.param(Written(written=True), {"answered": True, "written": True}, id="written to the place instead"),
        pytest.param(SpeakRefusal.TOO_LONG, {"answered": True, "refused": "too_long"}, id="refused"),
        pytest.param(Unanswered("network"), {"answered": False, "code": "network"}, id="no answer"),
    ],
)
def test_a_spoken_sentence_is_told_as_a_speech_entry_with_its_text_age_and_how_the_app_took_it(outcome, told_outcome):
    sentence = a_sentence(text="Grüß dich.", outcome=outcome)

    told_entries = json.loads(told_a_written_turn(None, history_holding((sentence,))))["history"]["entries"]

    assert told_entries == [
        {
            "entry": "speech",
            "app": APP,
            "place": {"kind": "call", "id": "room-7"},
            "seconds_ago": 9,
            "text": "Grüß dich.",
            "outcome": told_outcome,
            "new": False,
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
def test_the_bound_counts_a_spoken_sentence_by_the_utf8_bytes_of_its_told_json_with_its_age_as_zero(
    bound_over_the_two, kept_kinds, older_dropped, told_one_bytes
):
    sentence = a_sentence(text="ä♠")
    told_sentence = json.dumps(
        {
            "entry": "speech",
            "app": APP,
            "place": {"kind": "call", "id": "room-7"},
            "seconds_ago": 0,
            "text": "ä♠",
            "outcome": {"answered": True},
            "new": False,
        },
        ensure_ascii=False,
    )

    size = told_one_bytes + len(told_sentence.encode()) + bound_over_the_two
    history = history_after(EMPTY_HISTORY, (said(1, "one"), sentence), size)

    assert ["line" if isinstance(each, KeptLine) else "speech" for each in history.entries] == kept_kinds
    assert history.older_dropped is older_dropped


@pytest.mark.parametrize(
    ("ended", "cancelling", "expected"),
    [
        pytest.param(Spoken(duration_ms=900), True, Spoken(duration_ms=900), id="a spoken sentence while cancelled"),
        pytest.param(Written(written=True), True, Written(written=True), id="a written sentence while cancelled"),
        pytest.param(SpeakRefusal.NOT_ACTIVE, False, SpeakRefusal.NOT_ACTIVE, id="a refusal"),
        pytest.param(StatusFailure(502), True, Unanswered(TURN_CANCELLED_CODE), id="a failure while cancelled"),
        pytest.param(StatusFailure(502), False, Unanswered("http_502"), id="a failure"),
        pytest.param(asyncio.CancelledError(), True, Unanswered(TURN_CANCELLED_CODE), id="a cancellation"),
        pytest.param(RuntimeError("defect"), True, Unanswered(TURN_CANCELLED_CODE), id="another error while cancelled"),
        pytest.param(RuntimeError("defect"), False, None, id="another error"),
    ],
)
def test_how_a_speak_request_ended_decides_the_outcome_of_the_sentence_it_sent(ended, cancelling, expected):
    assert speech_outcome(ended, cancelling=cancelling) == expected


@pytest.mark.parametrize(
    ("started", "now", "past"),
    [(1000.0, 1000.0, False), (1000.0, 1029.9, False), (1000.0, 1030.0, True), (1000.0, 1100.0, True)],
    ids=["at the start", "just before the bound", "at the bound", "long past it"],
)
def test_a_turn_is_past_its_deadline_once_its_longest_duration_has_gone_by(started, now, past):
    assert past_deadline(now, started, 30) is past


def test_the_history_stays_marked_once_it_dropped_older_lines_though_the_next_lines_fit():
    assert history_after(ONCE_DROPPED, (said(1, "one"),), 10_000).older_dropped is True


def an_own_line(seq: int, text: str = "mine") -> KeptLine:
    return KeptLine(MORNING, seq, LATER, "Mozart", "message", text, own=True, spoken=False, cut=False, act=False)


def a_view_seen(revision: int, *, size: int | None = None) -> ViewSeen:
    state = {"hand": "x" * size} if size is not None else {"hand": secrets.token_hex(4)}
    return ViewSeen(APP, TABLE, PlaceView(revision=revision, awaits_you=False, state=state), LATER)


def taking(history: History, *entries, size_bytes: int = ROOMY.size_bytes) -> History:
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
    return json.loads(held_told(arrived, LATER))


def test_a_held_call_is_told_it_was_not_sent_and_every_arrived_entry_as_new_his_own_act_too():
    act = an_act(at=LATER - timedelta(seconds=4))
    line = said(2, "Spiel den Buben", name="Ada")

    told_text = told_held(Arrived((act, line), older_dropped=True))

    assert told_text["sent"] is False
    assert told_text["arrived"]["older_dropped"] is True
    assert [(each["entry"], each["new"]) for each in told_text["arrived"]["entries"]] == [("act", True), ("line", True)]
    assert told_text["arrived"]["entries"][0]["seconds_ago"] == 4
    assert told_text["arrived"]["entries"][1]["text"] == "Spiel den Buben"


def test_a_held_text_whose_arrivals_pass_the_event_bound_tells_the_newest_that_fit_and_says_older_dropped():
    third = MAX_EVENT_BYTES // 3 + 1
    views = tuple(a_view_seen(each, size=third) for each in (1, 2, 3))

    text = held_told(Arrived(views, older_dropped=False), LATER)

    told_text = json.loads(text)
    assert len(text.encode()) <= MAX_EVENT_BYTES
    assert [each["revision"] for each in told_text["arrived"]["entries"]] == [2, 3]
    assert told_text["arrived"]["older_dropped"] is True


def test_an_arrival_that_alone_passes_the_event_bound_is_left_out_and_older_dropped_is_said():
    text = held_told(Arrived((a_view_seen(1, size=MAX_EVENT_BYTES),), older_dropped=False), LATER)

    assert json.loads(text) == {"sent": False, "arrived": {"older_dropped": True, "entries": []}}


def test_the_repr_of_an_arrival_shows_no_entry():
    marker = secrets.token_hex(8)
    arrived = Arrived((said(1, marker, name=marker), a_view_seen(1)), older_dropped=False)

    assert marker not in repr(arrived)
