"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import asyncio
import itertools
import secrets

import pytest
from agents.backend import TurnStopKind, TurnStopped
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    CONTINUE_SPEAKING_TOOL,
    LISTEN_BOUND_SECONDS,
    SPEAK_TOOL,
    STOP_SPEAKING_TOOL,
    AcceptInvitationRefusal,
    FigureRef,
    Interrupted,
    PartialSpeech,
    PlaceRef,
    PlaceWindow,
    SpeakRefusal,
    Spoken,
    Standing,
    StandingState,
    TalkedOver,
    VoiceState,
    Written,
)
from agents.tools import ToolDeclaration, ToolOutcome
from agents.turn import FailureCode, RejectionCode
from logline import Outcome
from mcp_types import INVALID_PARAMS

from personas.decision import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    DEFECT_REASON,
    NETWORK_FAILURE_CODE,
    NO_FINAL_EVENT_REASON,
    RENEW_BEFORE_EXPIRY_SECONDS,
    TURN_CANCELLED_CODE,
    Accept,
    Held,
    Joined,
    Memory,
    Paused,
    Perception,
    Reaction,
    Restarted,
    Suspend,
    Token,
    TurnEnding,
    UnrecognisedRefusal,
    accepts_for,
    act_outcome,
    active_view_places,
    active_window_places,
    after_refusal,
    backoff_delay,
    budget_pause,
    call_within_bound,
    decisive_failure,
    failure_code,
    failure_reason,
    forget_gone,
    held_apps,
    hold_stated,
    hold_tellable,
    holds_a_place,
    is_not_granted,
    is_voice_app,
    jsonrpc_code,
    left_out_for,
    lends_tools,
    may_send_next,
    offered_declarations,
    past_deadline,
    perceive,
    reaction_to,
    remaining_seconds,
    renewal_due,
    sets_where_reading_starts,
    speaks_aloud,
    speech_outcome,
    sync_checkpoints,
    sync_turned_revisions,
    turn_call_bound,
    turn_ending,
    turn_invalid,
    untold_words,
    view_turn_due,
    waking_line,
)
from personas.entries import Answered, Lines, Reading, Unanswered, Waiting
from personas.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer
from personas.recall import RECALL_TOOL
from personas.secret import Secret
from tests.support.history import MORNING, OTHER_TABLE, TABLE, a_view
from tests.support.pictures import a_picture
from tests.support.standings import GENERATION, active, invited, line, paused, window_of

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


@pytest.mark.parametrize(
    ("failure", "not_granted"),
    [
        (TokenStatusFailure(400), True),
        (TokenStatusFailure(401), True),
        (TokenStatusFailure(403), True),
        (TokenStatusFailure(500), False),
        (StatusFailure(401), True),
        (StatusFailure(403), True),
        (StatusFailure(500), False),
        (UnusableAnswer("http_307"), False),
        (Unreachable("down"), False),
    ],
)
def test_only_a_realm_refusal_or_an_app_401_or_403_says_the_agent_is_not_granted(failure, not_granted):
    assert is_not_granted(failure) is not_granted


def test_a_json_rpc_code_is_logged_as_the_word_and_the_signed_number():
    assert jsonrpc_code(INVALID_PARAMS) == "jsonrpc_-32602"


def test_a_secret_shows_nothing_of_itself():
    value = secrets.token_hex(8)
    secret = Secret(value)

    assert value not in repr(secret) + str(secret) + repr(Token(secret, 1.0))


def test_a_token_is_renewed_before_a_stream_the_app_bounds_could_end_on_an_expired_one():
    assert RENEW_BEFORE_EXPIRY_SECONDS > LISTEN_BOUND_SECONDS


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
        (({},), False, 1),
        (({"own": True},), False, None),
        (({"own": True}, {}), False, 2),
        ((), False, None),
        (({"act": True},), False, None),
        (({"act": True}, {}), False, 2),
        (({"own": True, "act": True},), False, None),
        (({}, {}, {"own": True}, {"act": True}), False, 2),
        (({},), True, None),
        (({"act": True},), True, None),
        (({"own": True}, {}, {"act": True}), True, None),
    ],
    ids=[
        "someone else's line",
        "his own line",
        "one of them someone else's",
        "no line",
        "an act line",
        "an act line beside a line said",
        "his own act line",
        "lines said, then his own and an act",
        "a line at a place with a view",
        "an act line at a place with a view",
        "lines of every kind at a place with a view",
    ],
)
def test_the_newest_line_that_is_neither_his_own_nor_an_act_wakes_him_unless_the_place_serves_a_view(
    taken, serves_view, expected
):
    lines = [line(seq, "Hello everyone", **marks) for seq, marks in enumerate(taken, 1)]

    waking = waking_line(lines, serves_view=serves_view)

    assert (None if waking is None else waking.seq) == expected


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


@pytest.mark.parametrize(
    ("listed", "taken", "kept", "left_out"),
    [
        (["a", "b"], [], ["a", "b"], 0),
        (["c", "a", "b"], [], ["c", "a", "b"], 0),
        (["a", "b", "c"], ["b"], ["a", "c"], 1),
        (["a", "b"], ["a", "b"], [], 2),
        ([], ["a"], [], 0),
        (
            ["a", SPEAK_TOOL, RECALL_TOOL, ACCEPT_INVITATION_TOOL, CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL],
            [],
            ["a"],
            0,
        ),
        (["a", SPEAK_TOOL], [SPEAK_TOOL], ["a"], 0),
    ],
    ids=[
        "all",
        "listing order",
        "one held by an earlier app",
        "all held",
        "none listed",
        "reserved names",
        "reserved and held",
    ],
)
def test_a_turn_declares_what_an_app_lists_but_the_reserved_names_and_those_an_earlier_app_holds(
    listed, taken, kept, left_out
):
    declared = {name: _declaration(name) for name in listed}

    offered = offered_declarations(list(declared.values()), frozenset(taken))

    assert offered == (tuple(declared[name] for name in kept), left_out)


@pytest.mark.parametrize("has_voice", [True, False])
@pytest.mark.parametrize("declared_names", [[SPEAK_TOOL, "say"], ["say"]], ids=["declares speak", "declares none"])
def test_the_places_of_an_app_are_spoken_only_for_an_agent_with_a_voice_and_an_app_that_declares_speak(
    has_voice, declared_names
):
    declared = [_declaration(name) for name in declared_names]

    assert is_voice_app(has_voice, declared) is (has_voice and SPEAK_TOOL in declared_names)


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


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (StatusFailure(503), "no_answer"),
        (StatusFailure(400, "tools/list", "jsonrpc_-32602"), "invalid_params"),
        (StatusFailure(400, "tools/list", "jsonrpc_-32000"), "other_jsonrpc_error"),
        (StatusFailure(400, "tools/list", "text/html"), "html"),
        (StatusFailure(400, "tools/list", "application/x-secret-front"), "other_media_type"),
        (UnusableAnswer("invalid_answer"), "invalid_answer"),
        (UnusableAnswer("jsonrpc_-32601"), "method_not_found"),
        (UnusableAnswer("http_302"), "unexpected_status"),
        (Unreachable("cut"), NETWORK_FAILURE_CODE),
    ],
    ids=[
        "a status with no answer",
        "a standard jsonrpc error by name",
        "another jsonrpc error",
        "a known media type by name",
        "an unknown media type",
        "an unusable answer",
        "an unusable jsonrpc error",
        "a redirect",
        "the network",
    ],
)
def test_a_failure_logs_a_closed_reason_that_never_holds_a_value_the_app_chose(failure, reason):
    assert failure_reason(failure) == reason


@pytest.mark.parametrize(
    ("ended", "ending"),
    [
        pytest.param({"answered": True}, TurnEnding(Outcome.OK), id="an answer"),
        pytest.param({}, TurnEnding(Outcome.FAILED, NO_FINAL_EVENT_REASON), id="events that end without a final event"),
        pytest.param(
            {"error": Unreachable("down")}, TurnEnding(Outcome.FAILED, NETWORK_FAILURE_CODE), id="a failed call"
        ),
        pytest.param(
            {"error": BaseExceptionGroup("session", [Unreachable("down"), StatusFailure(502)])},
            TurnEnding(Outcome.FAILED, NETWORK_FAILURE_CODE),
            id="failed calls in a group",
        ),
        pytest.param({"error": RuntimeError("bug")}, TurnEnding(Outcome.FAILED, DEFECT_REASON), id="a defect"),
        pytest.param(
            {"error": ExceptionGroup("session", [Unreachable("down"), RuntimeError("bug")])},
            TurnEnding(Outcome.FAILED, DEFECT_REASON),
            id="a defect beside a failed call",
        ),
        pytest.param(
            {"stop_code": RejectionCode.BUDGET_SPENT.value},
            TurnEnding(Outcome.REFUSED, RejectionCode.BUDGET_SPENT.value),
            id="a stop the gateway refused",
        ),
        pytest.param(
            {"stop_code": FailureCode.LIMIT_EXCEEDED.value},
            TurnEnding(Outcome.FAILED, FailureCode.LIMIT_EXCEEDED.value),
            id="a stop that failed",
        ),
        pytest.param(
            {"stop_code": RejectionCode.BUSY.value, "error": Unreachable("late answer")},
            TurnEnding(Outcome.REFUSED, RejectionCode.BUSY.value),
            id="a reported stop beside the failure that left with it",
        ),
        pytest.param({"cancelled": True}, TurnEnding(Outcome.CANCELLED), id="a cancellation"),
        pytest.param(
            {"cancelled": True, "answered": True},
            TurnEnding(Outcome.OK),
            id="a cancellation in the teardown of an answered turn",
        ),
        pytest.param(
            {"error": RuntimeError("bug"), "answered": True},
            TurnEnding(Outcome.OK),
            id="an error in the teardown of an answered turn",
        ),
    ],
)
def test_a_turn_ends_with_the_outcome_and_reason_its_ending_gives(ended, ending):
    given = {"cancelled": False, "error": None, "stop_code": None, "answered": False} | ended

    assert turn_ending(**given) == ending


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


def test_an_accept_shows_neither_its_figure_nor_its_voice_in_its_repr():
    voice = f"Voice_{secrets.token_hex(6)}"
    figure = FigureRef(url=f"https://figures.test/{secrets.token_hex(6)}.glb", sha256=secrets.token_hex(32))

    shown = repr(Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 4, figure, (voice,)))

    assert voice not in shown
    assert figure.url not in shown
    assert figure.sha256 not in shown


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
    ("ended", "cancelling", "expected"),
    [
        pytest.param(Spoken(spoken=True), True, Spoken(spoken=True), id="a spoken sentence while cancelled"),
        pytest.param(Written(written=True), True, Written(written=True), id="a written sentence while cancelled"),
        pytest.param(SpeakRefusal.NOT_ACTIVE, False, SpeakRefusal.NOT_ACTIVE, id="a refusal"),
        pytest.param(StatusFailure(502), True, Unanswered(TURN_CANCELLED_CODE), id="a failure while cancelled"),
        pytest.param(StatusFailure(502), False, Unanswered("http_502"), id="a failure"),
        pytest.param(asyncio.CancelledError(), True, Unanswered(TURN_CANCELLED_CODE), id="a cancellation"),
        pytest.param(RuntimeError("defect"), True, Unanswered(TURN_CANCELLED_CODE), id="another error while cancelled"),
        pytest.param(RuntimeError("defect"), False, None, id="another error"),
        pytest.param(
            TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED),
            False,
            Unanswered(FailureCode.LIMIT_EXCEEDED.value),
            id="a request given up at the turn's deadline",
        ),
        pytest.param(
            TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED),
            True,
            Unanswered(TURN_CANCELLED_CODE),
            id="a request given up at the deadline while cancelled",
        ),
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


@pytest.mark.parametrize(
    ("now", "remaining"),
    [(1000.0, 30.0), (1012.5, 17.5), (1030.0, 0.0), (1100.0, 0.0)],
    ids=["at the start", "part way", "at the deadline", "long past it"],
)
def test_what_is_left_of_a_turns_longest_duration_is_never_below_zero(now, remaining):
    assert remaining_seconds(now, 1000.0, 30) == remaining


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
