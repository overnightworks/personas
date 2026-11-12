"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import secrets
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from agents.member import (
    LEASE_SECONDS,
    LISTEN_BOUND_SECONDS,
    SPEAK_TOOL,
    AcceptInvitationRefusal,
    PlaceEntry,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    Standing,
    StandingState,
)
from agents.tools import ToolDeclaration, ToolOutcome

from personas.app_session import READ_SECONDS
from personas.decision import (
    BACKOFF_CAP_SECONDS,
    NETWORK_FAILURE_CODE,
    RENEW_BEFORE_EXPIRY_SECONDS,
    SPOKEN_WAIT_BOUND_SECONDS,
    TURN_CANCELLED_CODE,
    Accept,
    ActDone,
    Answered,
    EarlierToday,
    History,
    Joined,
    KeptLine,
    Lines,
    Memory,
    Paused,
    Perception,
    Reaction,
    Reading,
    Restarted,
    Suspend,
    TodayBounds,
    Token,
    Unanswered,
    UnrecognisedRefusal,
    ViewSeen,
    Visit,
    Waiting,
    accepts_for,
    act_outcome,
    active_call_places,
    active_view_places,
    addressed,
    after_refusal,
    backoff_delay,
    bound_declarations,
    call_within_bound,
    decisive_failure,
    earlier_today,
    failure_code,
    forget_gone,
    history_after,
    is_voice_app,
    perceive,
    reaction_to,
    renewal_due,
    sets_where_reading_starts,
    spoken_wait_seconds,
    swept_before,
    sync_checkpoints,
    sync_turned_revisions,
    taken_lines,
    today_since,
    turn_call_bound,
    turn_invalid,
    turn_told,
    view_turn_due,
    voice_context,
)
from personas.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer
from personas.secret import Secret
from tests.support.fakes import line, window_of
from tests.support.history import told_day_line_bytes, told_line_bytes
from tests.support.hopin import GENERATION, active, invited, paused

TRANSIENT_REFUSALS = [
    AcceptInvitationRefusal.SPONSOR_AWAY,
    AcceptInvitationRefusal.TWO_SEATS,
    AcceptInvitationRefusal.BUSY,
    UnrecognisedRefusal.UNRECOGNISED,
]


def test_an_invited_call_standing_is_accepted_with_its_place_invitation_and_epoch():
    held = (invited("room-7", "inv-3", epoch=4),)

    assert accepts_for(held, Memory(), takes_turns=False) == (Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 4),)


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
        (UnusableAnswer("jsonrpc_-32602"), False, Suspend("jsonrpc_-32602")),
        (UnusableAnswer("http_307"), True, Suspend("http_307")),
    ],
)
def test_each_failure_makes_the_loop_renew_suspend_or_back_off(failure, already_renewed, reaction):
    assert reaction_to(failure, already_renewed=already_renewed) == reaction


def test_a_secret_shows_nothing_of_itself():
    value = secrets.token_hex(8)
    secret = Secret(value)

    assert value not in repr(secret) + str(secret) + repr(Token(secret, 1.0))


def test_a_token_is_renewed_before_a_stream_the_app_bounds_could_end_on_an_expired_one():
    assert RENEW_BEFORE_EXPIRY_SECONDS > LISTEN_BOUND_SECONDS


def test_the_runner_cuts_a_stream_before_the_lease_of_its_member_would_have_ended():
    assert READ_SECONDS < LEASE_SECONDS


REFUSAL = UnusableAnswer("jsonrpc_-32602")
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


def test_only_the_active_call_places_are_read():
    paused = Standing(place=PlaceRef(kind="call", id="room-8"), state=StandingState.PAUSED)
    other = Standing(place=PlaceRef(kind="table", id="table-1"), state=StandingState.ACTIVE)

    assert active_call_places((active("room-7"), paused, other, invited("room-9", "inv-3"))) == (PLACE,)


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
    ],
)
def test_the_standing_decides_which_checkpoints_exist(checkpoints, standings, expected):
    assert sync_checkpoints(checkpoints, standings) == expected


def test_a_place_of_another_kind_gets_no_checkpoint():
    table = Standing(place=PlaceRef(kind="table", id="table-1"), state=StandingState.ACTIVE)

    assert sync_checkpoints({}, (table,)) == {}


@pytest.mark.parametrize(
    ("checkpoint", "sets_start"),
    [(Restarted(), True), (Paused(), True), (Joined(), False), (Reading(GENERATION, 2), False)],
    ids=["restarted", "paused", "joined", "reading"],
)
def test_only_a_restarted_or_paused_checkpoint_waits_for_a_read_to_set_where_reading_starts(checkpoint, sets_start):
    assert sets_where_reading_starts(checkpoint) is sets_start


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
    ("text", "name", "expected"),
    [
        ("Mozart, are you there?", "Mozart", True),
        ("hey MOZART", "Mozart", True),
        ("what do you think, mozart?!", "Mozart", True),
        ("@Mozart", "Mozart", True),
        ("Mozart's view", "Mozart", True),
        ("Mozartian music", "Mozart", False),
        ("Amozart", "Mozart", False),
        ("Wolfgang", "Mozart", False),
        ("Mozart", "Mo.art", False),
        ("a Mo.art here", "Mo.art", True),
        ("STRASSE", "Straße", True),
    ],
    ids=[
        "named",
        "other case",
        "lower case",
        "mention",
        "possessive",
        "longer word",
        "inside a word",
        "not named",
        "the name is no pattern",
        "a name with a dot",
        "case folding",
    ],
)
def test_a_line_is_addressed_when_it_names_the_agent_as_a_whole_word_whatever_the_case(text, name, expected):
    assert addressed(line(1, text), name) is expected


def test_a_line_of_his_own_is_never_addressed_to_him():
    assert not addressed(line(1, "Mozart answers", own=True), "Mozart")


def _declaration(name: str) -> ToolDeclaration:
    return ToolDeclaration(name=name, description=secrets.token_hex(4), parameters={"type": "object"})


def test_the_bound_tools_are_declared_in_the_order_they_are_bound():
    declared = [_declaration("a"), _declaration("b"), _declaration("c")]

    assert bound_declarations(declared, ["c", "a"]) == (declared[2], declared[0])


def test_a_bound_tool_the_app_declares_no_more_leaves_no_declarations():
    assert bound_declarations([_declaration("a")], ["a", "gone"]) is None


LATER = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)
MORNING = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-7"), "morning")
NOON = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-8"), "noon")
ROOMY = TodayBounds(hours=24.0, size_bytes=10_000)
EMPTY_HISTORY = History((), older_dropped=False)


def kept(visit, seq, text, *, hours_before, own=False, name="Ada"):
    """A kept line of a visit, first read this many hours before ``LATER``."""
    return KeptLine(visit, seq, LATER - timedelta(hours=hours_before), name, "message", text, own)


def told(earlier):
    return [[(each.seq, each.text) for each in visit.lines] for visit in earlier.visits]


def test_two_earlier_visits_are_told_in_the_order_they_happened_each_as_its_own_group_in_seq_order():
    lines = [
        kept(NOON, 2, "noon two", hours_before=6),
        kept(MORNING, 1, "morning one", hours_before=9),
        kept(NOON, 1, "noon one", hours_before=6),
        kept(MORNING, 2, "morning two", hours_before=9),
    ]

    earlier = earlier_today(lines, LATER, ROOMY, EMPTY_HISTORY)

    assert told(earlier) == [[(1, "morning one"), (2, "morning two")], [(1, "noon one"), (2, "noon two")]]
    assert [visit.minutes_ago for visit in earlier.visits] == [9 * 60, 6 * 60]
    assert earlier.older_dropped is False


@pytest.fixture(
    params=[
        pytest.param(("one", "one"), id="plain text"),
        pytest.param(('ä♠"\n', r"ä♠\"\n"), id="two-, three- and escaped characters"),
    ]
)
def text_and_its_told_json(request):
    """A text, and that text as the JSON a turn is told it in, between its quotes."""
    return request.param


@pytest.mark.parametrize(
    ("bound_over_the_two_lines", "told_seqs", "older_dropped"),
    [
        pytest.param(1, (1, 2), False, id="the bound lies above the lines"),
        pytest.param(0, (1, 2), False, id="the bound is the lines' size"),
        pytest.param(-1, (2,), True, id="the bound lies one byte below the lines"),
    ],
)
def test_the_day_keeps_the_newest_lines_that_fit_its_bound_counted_in_utf8_bytes_of_their_told_json(
    text_and_its_told_json, bound_over_the_two_lines, told_seqs, older_dropped
):
    text, told_text = text_and_its_told_json
    lines = [kept(MORNING, seq, text, hours_before=1, name="Bo") for seq in (1, 2)]
    bound = TodayBounds(hours=24.0, size_bytes=2 * told_day_line_bytes(told_text, name="Bo") + bound_over_the_two_lines)

    earlier = earlier_today(lines, LATER, bound, EMPTY_HISTORY)

    assert told(earlier) == [[(seq, text) for seq in told_seqs]]
    assert earlier.older_dropped is older_dropped


def test_an_earlier_visit_whose_lines_all_lie_past_the_size_bound_leaves_the_day_whole():
    lines = [
        kept(MORNING, 1, "oldest", hours_before=9, name="Bo"),
        kept(NOON, 1, "middle", hours_before=6, name="Bo"),
        kept(NOON, 2, "newest", hours_before=6, name="Bo"),
    ]
    two_lines = TodayBounds(hours=24.0, size_bytes=2 * told_day_line_bytes("middle", name="Bo"))

    earlier = earlier_today(lines, LATER, two_lines, EMPTY_HISTORY)

    assert told(earlier) == [[(1, "middle"), (2, "newest")]]
    assert [visit.minutes_ago for visit in earlier.visits] == [6 * 60]
    assert earlier.older_dropped is True


def test_no_line_of_the_day_is_told_and_the_day_says_so_when_not_even_the_newest_fits_its_bound():
    lines = [kept(NOON, 1, "middle", hours_before=6, name="Bo")]
    too_small = TodayBounds(hours=24.0, size_bytes=told_day_line_bytes("middle", name="Bo") - 1)

    earlier = earlier_today(lines, LATER, too_small, EMPTY_HISTORY)

    assert earlier == EarlierToday((), older_dropped=True)


def test_the_day_reaches_back_the_configured_hours_and_an_unbounded_day_reaches_back_to_any_line():
    assert today_since(LATER, TodayBounds(hours=24.0, size_bytes=1)) == LATER - timedelta(hours=24)
    assert today_since(LATER, TodayBounds(hours=math.inf, size_bytes=1)) <= datetime(1, 1, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    ("keep_days", "oldest_kept"),
    [
        (30.0, datetime(2026, 10, 11, 18, 0, tzinfo=UTC)),
        (0.5, datetime(2026, 11, 10, 6, 0, tzinfo=UTC)),
        (math.inf, None),
    ],
    ids=["a month", "half a day", "every line"],
)
def test_the_oldest_first_read_that_stays_is_the_keep_period_before_now_and_an_infinite_period_sweeps_nothing(
    keep_days, oldest_kept
):
    assert swept_before(LATER, keep_days) == oldest_kept


def test_a_keep_period_that_reaches_before_the_calendar_sweeps_nothing():
    assert swept_before(LATER, 1e12) is None


def told_a_written_turn(earlier: EarlierToday | None, history: History = EMPTY_HISTORY, *, missed: bool = False) -> str:
    """The message of a written turn at ``LATER`` in the place of the morning visit, for lines that are none yet."""
    return turn_told(
        "Mozart", APP, MORNING.place, Lines(MORNING.generation, frozenset(), missed), history, earlier, LATER
    )


def test_a_turn_is_told_each_earlier_visit_how_long_ago_it_was_its_lines_their_speakers_and_his_own():
    mine, theirs = secrets.token_hex(4), secrets.token_hex(4)
    earlier = earlier_today(
        [kept(NOON, 1, theirs, hours_before=4), kept(NOON, 2, mine, hours_before=4, own=True, name="Mozart")],
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
                    {"name": "Ada", "kind": "message", "text": theirs, "own": False},
                    {"name": "Mozart", "kind": "message", "text": mine, "own": True},
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


def test_a_voice_context_is_the_day_a_turn_is_told_under_the_same_key():
    earlier = earlier_today([kept(NOON, 1, "Good morning", hours_before=4)], LATER, ROOMY, EMPTY_HISTORY)

    told_to_a_turn = json.loads(told_a_written_turn(earlier))

    assert json.loads(voice_context(earlier)) == {"earlier_today": told_to_a_turn["earlier_today"]}


def test_a_voice_loop_is_told_nothing_of_the_day_without_a_memory():
    assert voice_context(None) == ""


@pytest.mark.parametrize("has_voice_section", [True, False])
@pytest.mark.parametrize("declared_names", [[SPEAK_TOOL, "say"], ["say"]], ids=["declares speak", "declares none"])
def test_the_places_of_an_app_run_a_voice_loop_only_for_a_persona_with_a_voice_section_and_an_app_that_declares_speak(
    has_voice_section, declared_names
):
    declared = [_declaration(name) for name in declared_names]

    assert is_voice_app(has_voice_section, declared) is (has_voice_section and SPEAK_TOOL in declared_names)


@pytest.mark.parametrize(
    ("duration_ms", "waited"),
    [(1500, 1.5), (1, 0.001), (int(SPOKEN_WAIT_BOUND_SECONDS * 1000), SPOKEN_WAIT_BOUND_SECONDS)],
    ids=["a sentence", "the shortest", "exactly the bound"],
)
def test_the_voice_waits_as_long_as_a_sentence_plays_up_to_the_bound(duration_ms, waited):
    assert spoken_wait_seconds(duration_ms) == waited


def test_the_voice_waits_no_longer_than_the_bound_whatever_the_app_says():
    assert spoken_wait_seconds(int(SPOKEN_WAIT_BOUND_SECONDS * 1000) + 1) == SPOKEN_WAIT_BOUND_SECONDS


def test_a_turns_call_bound_is_the_sum_of_the_bounds_of_the_apps_whose_tools_it_holds():
    assert turn_call_bound([4, 5, 2]) == 11
    assert turn_call_bound([4]) == 4


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


TABLE = PlaceRef(kind="table", id="table-1")
OTHER_TABLE = PlaceRef(kind="table", id="table-2")


def serving(place: PlaceRef = TABLE, *, state: StandingState = StandingState.ACTIVE) -> Standing:
    if state is StandingState.INVITED:
        return invited(place.id, "inv-3", kind=place.kind, serves_view=True)
    return Standing(place=place, state=state, serves_view=True)


def a_view(revision: int, *, awaits_you: bool = True) -> PlaceView:
    return PlaceView(revision=revision, awaits_you=awaits_you, state={"hand": secrets.token_hex(4)})


@pytest.mark.parametrize(
    ("takes_turns", "accepted"),
    [(True, True), (False, False)],
    ids=["the loop can take turns", "the loop cannot take turns"],
)
def test_an_invited_place_that_serves_a_view_is_accepted_only_where_the_loop_can_take_turns(takes_turns, accepted):
    standing = serving(state=StandingState.INVITED)

    expected = (Accept(TABLE, "inv-3", 0),) if accepted else ()
    assert accepts_for((standing,), Memory(), takes_turns=takes_turns) == expected


def test_an_invited_view_place_whose_invitation_was_refused_as_stale_is_not_accepted_again():
    standing = serving(state=StandingState.INVITED)
    memory, _ = after_refusal(
        Memory(), accepts_for((standing,), Memory(), takes_turns=True)[0], AcceptInvitationRefusal.STALE_INVITATION
    )

    assert accepts_for((standing,), memory, takes_turns=True) == ()


def test_a_call_place_that_serves_a_view_is_a_view_place_and_no_window_is_kept_or_read_for_it():
    standing = Standing(place=PLACE, state=StandingState.ACTIVE, serves_view=True)

    assert active_call_places((standing,)) == ()
    assert sync_checkpoints({}, (standing,)) == {}
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


APP = "https://app.test/mcp"


def said(seq: int, text: str, *, name: str = "Bo", visit: Visit = MORNING) -> KeptLine:
    return KeptLine(visit, seq, LATER, name, "message", text, own=False)


@pytest.fixture
def told_one_bytes() -> int:
    """The bytes the line ``said(1, "one")`` is told as in the history."""
    return told_line_bytes(APP, "room-7", "one", name="Bo")


def test_a_view_turn_is_told_its_revision_as_the_one_entry_marked_new_and_woke_names_only_why_app_and_place():
    view = PlaceView(revision=3, awaits_you=True, state={"hand": ["x", {"seen": True}], "umläut": "ä"})
    history = History(
        (said(1, "Spiel den Buben", name="Ada", visit=NOON), ViewSeen(APP, TABLE, view, LATER - timedelta(seconds=7))),
        older_dropped=True,
    )

    told = turn_told("Mozart", APP, TABLE, Waiting(3), history, None, LATER)

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
        turn_told("Mozart", APP, TABLE, Waiting(3), History((held, admitted), older_dropped=False), None, LATER)
    )

    assert [each["new"] for each in told["history"]["entries"]] == [False, True]


def test_a_written_turn_marks_no_view_new():
    history = History((ViewSeen(APP, MORNING.place, a_view(1), LATER), said(1, "hello")), older_dropped=False)

    told = json.loads(told_a_written_turn(None, history))

    assert [(each["entry"], each["new"]) for each in told["history"]["entries"]] == [("view", False), ("line", False)]


@pytest.mark.parametrize("missed", [False, True])
def test_a_written_turn_is_told_in_which_place_it_woke_and_whether_older_lines_were_not_seen(missed):
    woke = json.loads(told_a_written_turn(None, missed=missed))["woke"]

    assert woke == {"why": "lines", "app": APP, "place": {"kind": "call", "id": "room-7"}, "missed": missed}


def test_a_turn_told_an_empty_history_is_told_no_entries_and_none_dropped():
    assert json.loads(told_a_written_turn(None))["history"] == {"older_dropped": False, "entries": []}


def test_a_history_line_is_new_only_when_the_turn_is_for_that_line_of_that_visit():
    evening = replace(MORNING, generation="evening")
    history = History(
        (
            KeptLine(NOON, 1, LATER - timedelta(seconds=300), "Bo", "message", "elsewhere", own=False),
            KeptLine(evening, 1, LATER - timedelta(seconds=200), "Bo", "message", "an earlier run", own=False),
            KeptLine(MORNING, 1, LATER - timedelta(seconds=90), "Ada", "message", "woke him", own=False),
            KeptLine(MORNING, 2, LATER - timedelta(seconds=1), "Mozart", "message", "his own", own=True),
        ),
        older_dropped=False,
    )
    occasion = Lines(MORNING.generation, frozenset({1}), missed=False)

    told = json.loads(turn_told("Mozart", APP, MORNING.place, occasion, history, None, LATER))

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
    history = History((said(2, "later", visit=NOON), said(1, "earlier")), older_dropped=False)

    told = json.loads(told_a_written_turn(None, history))

    assert [entry["text"] for entry in told["history"]["entries"]] == ["later", "earlier"]


def test_taken_lines_are_the_visits_lines_as_kept_with_the_time_of_the_read():
    entries = (
        PlaceEntry(seq=4, kind="message", name="Ada", text="four", own=False),
        PlaceEntry(seq=5, kind="message", name="Mozart", text="five", own=True),
    )

    assert taken_lines(MORNING, entries, LATER) == (
        KeptLine(MORNING, 4, LATER, "Ada", "message", "four", own=False),
        KeptLine(MORNING, 5, LATER, "Mozart", "message", "five", own=True),
    )


def test_a_history_its_lines_and_its_views_show_no_name_no_text_and_no_state_in_their_repr():
    marker = secrets.token_hex(8)
    line = said(1, marker, name=marker)
    view = ViewSeen(APP, TABLE, PlaceView(revision=1, awaits_you=True, state={"hand": marker}), LATER)

    assert marker not in repr(History((line, view), older_dropped=False))
    assert marker not in repr(line)
    assert marker not in repr(view)


def test_new_lines_join_the_history_after_the_lines_it_holds():
    history = history_after(History((said(1, "one"),), older_dropped=False), (said(2, "two"), said(3, "three")), 10_000)

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
        pytest.param(History((said(1, "one"),), older_dropped=False), True, id="an earlier entry leaves"),
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

    history = history_after(History((held,), older_dropped=False), (again,), 10_000)

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


def an_act(*, arguments=None, outcome=None, tool="say", app=APP, at=LATER - timedelta(seconds=9)) -> ActDone:
    return ActDone(
        app, tool, {"text": "hi"} if arguments is None else arguments, outcome or Answered("posted", is_error=False), at
    )


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

    history = history_after(History((said(1, "one"),), older_dropped=False), (first, second), 1)

    assert history.entries == (second,)
    assert history.older_dropped is True


def test_an_act_the_history_already_holds_is_appended_again():
    act = an_act()

    assert history_after(History((act,), older_dropped=False), (act,), 10_000).entries == (act, act)


def test_an_answered_act_is_told_with_its_app_tool_arguments_answer_and_age_and_is_never_new():
    act = an_act(arguments={"place": "room-7", "text": "ä"}, outcome=Answered("gone", is_error=True))

    told_entries = json.loads(told_a_written_turn(None, History((act,), older_dropped=False)))["history"]["entries"]

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

    told_entry = json.loads(told_a_written_turn(None, History((act,), older_dropped=False)))["history"]["entries"][0]

    assert told_entry["outcome"] == {"answered": False, "code": TURN_CANCELLED_CODE}


def test_a_view_turn_marks_no_act_new():
    history = History((an_act(), ViewSeen(APP, TABLE, a_view(3), LATER)), older_dropped=False)

    told_entries = json.loads(turn_told("Mozart", APP, TABLE, Waiting(3), history, None, LATER))["history"]["entries"]

    assert [each["new"] for each in told_entries] == [False, True]


def test_the_day_is_the_same_whether_or_not_the_history_holds_acts_and_holds_no_act():
    kept_lines = [kept(MORNING, 1, "held", hours_before=5), kept(NOON, 1, "not held", hours_before=4)]

    without = earlier_today(kept_lines, LATER, ROOMY, History((said(1, "held"),), older_dropped=False))
    with_acts = earlier_today(kept_lines, LATER, ROOMY, History((said(1, "held"), an_act()), older_dropped=False))

    assert with_acts == without


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


def test_an_act_shows_no_arguments_and_no_answer_in_its_repr_nor_in_the_history_that_holds_it():
    marker = secrets.token_hex(8)
    act = an_act(arguments={"text": marker}, outcome=Answered(marker, is_error=False))

    assert marker not in repr(act)
    assert marker not in repr(act.outcome)
    assert marker not in repr(History((act,), older_dropped=False))


def test_the_history_stays_marked_once_it_dropped_older_lines_though_the_next_lines_fit():
    marked = History((), older_dropped=True)

    assert history_after(marked, (said(1, "one"),), 10_000).older_dropped is True


def test_the_day_leaves_out_the_lines_the_history_holds_by_visit_and_seq_and_those_first_read_after_the_turn():
    kept_lines = [
        kept(MORNING, 1, "held", hours_before=5),
        kept(MORNING, 2, "not held", hours_before=5),
        kept(NOON, 1, "same seq of another visit", hours_before=4),
        kept(NOON, 2, "read as the turn was admitted", hours_before=0),
        kept(NOON, 3, "read after the turn was admitted", hours_before=-1),
    ]

    earlier = earlier_today(kept_lines, LATER, ROOMY, History((said(1, "held"),), older_dropped=False))

    assert told(earlier) == [
        [(2, "not held")],
        [(1, "same seq of another visit"), (2, "read as the turn was admitted")],
    ]


def test_the_day_is_the_same_whether_or_not_the_history_holds_views_and_holds_no_view():
    kept_lines = [kept(MORNING, 1, "held", hours_before=5), kept(NOON, 1, "not held", hours_before=4)]
    views = (ViewSeen(APP, TABLE, a_view(1), LATER),)

    without = earlier_today(kept_lines, LATER, ROOMY, History((said(1, "held"),), older_dropped=False))
    with_views = earlier_today(kept_lines, LATER, ROOMY, History((said(1, "held"), *views), older_dropped=False))

    assert with_views == without
    assert told(with_views) == [[(1, "not held")]]


@pytest.mark.parametrize(
    "leaving_out",
    [History((said(1, "held"), said(2, "held")), older_dropped=False), MORNING],
    ids=["the lines a turn's history holds", "the visit a voice loop is in"],
)
def test_what_is_left_out_of_the_day_counts_nothing_toward_its_size_bound(leaving_out):
    two_lines = TodayBounds(hours=24.0, size_bytes=2 * told_day_line_bytes("middle", name="Bo"))
    kept_lines = [
        kept(MORNING, 1, "held" * 20, hours_before=5, name="Bo"),
        kept(MORNING, 2, "held" * 20, hours_before=5, name="Bo"),
        kept(NOON, 1, "middle", hours_before=4, name="Bo"),
        kept(NOON, 2, "second", hours_before=4, name="Bo"),
    ]

    earlier = earlier_today(kept_lines, LATER, two_lines, leaving_out)

    assert told(earlier) == [[(1, "middle"), (2, "second")]]
    assert earlier.older_dropped is False
