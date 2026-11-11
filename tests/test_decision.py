"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import itertools
import json
import math
import secrets
from datetime import UTC, datetime, timedelta

import pytest
from agents.member import (
    LEASE_SECONDS,
    LISTEN_BOUND_SECONDS,
    SPEAK_TOOL,
    AcceptInvitationRefusal,
    PlaceRef,
    PlaceWindow,
    Standing,
    StandingState,
)
from agents.tools import ToolDeclaration

from personas.app_session import READ_SECONDS
from personas.decision import (
    BACKOFF_CAP_SECONDS,
    NETWORK_FAILURE_CODE,
    RENEW_BEFORE_EXPIRY_SECONDS,
    SPOKEN_WAIT_BOUND_SECONDS,
    Accept,
    Joined,
    KeptLine,
    Memory,
    Paused,
    Perception,
    Reaction,
    Reading,
    Restarted,
    Suspend,
    TodayBounds,
    Token,
    TurnCap,
    UnrecognisedRefusal,
    Visit,
    accepts_for,
    active_call_places,
    addressed,
    after_refusal,
    backoff_delay,
    bound_declarations,
    call_within_bound,
    decisive_failure,
    earlier_today,
    failure_code,
    forget_gone,
    is_voice_app,
    perceive,
    reaction_to,
    renewal_due,
    reserve_turn,
    sets_where_reading_starts,
    spoken_wait_seconds,
    swept_before,
    sync_checkpoints,
    today_since,
    turn_call_bound,
    turn_input,
    turn_invalid,
    voice_context,
)
from personas.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer
from personas.secret import Secret
from tests.support.fakes import line, window_of
from tests.support.hopin import GENERATION, active, invited, paused

TRANSIENT_REFUSALS = [
    AcceptInvitationRefusal.SPONSOR_AWAY,
    AcceptInvitationRefusal.TWO_SEATS,
    AcceptInvitationRefusal.BUSY,
    UnrecognisedRefusal.UNRECOGNISED,
]


def test_an_invited_call_standing_is_accepted_with_its_place_invitation_and_epoch():
    held = (invited("room-7", "inv-3", epoch=4),)

    assert accepts_for(held, Memory()) == (Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 4),)


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
    assert accepts_for((standing,), Memory()) == ()


def test_a_stale_invitation_is_never_accepted_again():
    first = invited("room-7", "inv-3")
    memory, _ = after_refusal(Memory(), accepts_for((first,), Memory())[0], AcceptInvitationRefusal.STALE_INVITATION)

    assert accepts_for((first,), memory) == ()
    assert accepts_for((invited("room-7", "inv-4"),), memory) == (
        Accept(PlaceRef(kind="call", id="room-7"), "inv-4", 0),
    )


@pytest.mark.parametrize("refusal", TRANSIENT_REFUSALS)
def test_any_other_refusal_is_accepted_again_at_the_next_read(refusal):
    standing = invited("room-7", "inv-3")
    attempt = accepts_for((standing,), Memory())[0]
    memory, _ = after_refusal(Memory(), attempt, refusal)

    assert accepts_for((standing,), memory) == (attempt,)


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

    assert accepts_for((standing,), memory) == ()


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


def test_a_turn_is_told_facts_only_the_room_the_lines_which_are_new_and_which_are_his_own():
    own, theirs, new = secrets.token_hex(4), secrets.token_hex(4), secrets.token_hex(4)
    window = window_of(line(4, theirs, name="Ada"), line(5, own, own=True, name="Mozart"), line(6, new, name="Ada"))

    facts = json.loads(turn_input(PLACE, "Mozart", window, window.entries[2:], missed=False))

    assert (facts["kind"], facts["id"], facts["you"]) == (PLACE.kind, PLACE.id, "Mozart")
    assert [(each["seq"], each["text"], each["own"], each["new"]) for each in facts["lines"]] == [
        (4, theirs, False, False),
        (5, own, True, False),
        (6, new, False, True),
    ]


@pytest.mark.parametrize("missed", [False, True])
def test_a_turn_is_told_whether_older_lines_were_not_seen(missed):
    window = window_of(line(4, "a"))

    assert json.loads(turn_input(PLACE, "Mozart", window, window.entries, missed=missed))["missed"] is missed


LATER = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)
MORNING = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-7"), "morning")
NOON = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-8"), "noon")
ROOMY = TodayBounds(hours=24.0, size_bytes=10_000)


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

    earlier = earlier_today(lines, LATER, ROOMY)

    assert told(earlier) == [[(1, "morning one"), (2, "morning two")], [(1, "noon one"), (2, "noon two")]]
    assert [visit.minutes_ago for visit in earlier.visits] == [9 * 60, 6 * 60]
    assert earlier.older_dropped is False


def test_lines_beyond_the_size_bound_leave_the_oldest_out_keep_the_newest_and_say_so():
    lines = [
        kept(MORNING, 1, "oldest" * 20, hours_before=9, name="Bo"),
        kept(NOON, 1, "middle", hours_before=6, name="Bo"),
        kept(NOON, 2, "newest", hours_before=6, name="Bo"),
    ]
    two_lines = TodayBounds(hours=24.0, size_bytes=2 * len("Bo" + "middle"))

    earlier = earlier_today(lines, LATER, two_lines)

    assert told(earlier) == [[(1, "middle"), (2, "newest")]]
    assert earlier.older_dropped is True


def test_lines_exactly_at_the_size_bound_are_all_told():
    lines = [kept(MORNING, 1, "one", hours_before=1, name="Bo"), kept(MORNING, 2, "two", hours_before=1, name="Bo")]

    earlier = earlier_today(lines, LATER, TodayBounds(hours=24.0, size_bytes=2 * len("Bo" + "one")))

    assert told(earlier) == [[(1, "one"), (2, "two")]]
    assert earlier.older_dropped is False


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


def test_a_turn_is_told_each_earlier_visit_how_long_ago_it_was_its_lines_their_speakers_and_his_own():
    mine, theirs = secrets.token_hex(4), secrets.token_hex(4)
    window = window_of(line(1, "now"))
    earlier = earlier_today(
        [kept(MORNING, 1, theirs, hours_before=4), kept(MORNING, 2, mine, hours_before=4, own=True, name="Mozart")],
        LATER,
        ROOMY,
    )

    facts = json.loads(turn_input(PLACE, "Mozart", window, window.entries, missed=False, earlier=earlier))

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
    assert [each["text"] for each in facts["lines"]] == ["now"]


def test_a_turn_without_a_memory_is_told_no_earlier_visits():
    window = window_of(line(1, "now"))

    assert "earlier_today" not in json.loads(turn_input(PLACE, "Mozart", window, window.entries, missed=False))


def test_a_voice_context_is_the_day_a_turn_is_told_under_the_same_key():
    earlier = earlier_today([kept(MORNING, 1, "Good morning", hours_before=4)], LATER, ROOMY)
    window = window_of(line(1, "now"))

    told_to_a_turn = json.loads(turn_input(PLACE, "Mozart", window, window.entries, missed=False, earlier=earlier))

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


CAP = TurnCap(max_turns=3, period_seconds=100.0)
NOW = 5000.0


def test_a_turn_with_no_turns_started_is_reserved_at_now():
    assert reserve_turn((), NOW, CAP) == (NOW,)


def test_without_a_cap_every_turn_may_start_and_no_start_time_is_kept():
    started = (NOW - 50, NOW - 10)

    assert reserve_turn(started, NOW, None) == started


def test_turns_below_the_cap_keep_their_times_and_add_now():
    started = (NOW - 50, NOW - 10)

    assert reserve_turn(started, NOW, CAP) == (*started, NOW)


def test_a_turn_at_the_cap_inside_the_period_is_held_back():
    assert reserve_turn((NOW - 90, NOW - 50, NOW - 10), NOW, CAP) is None


def test_a_start_exactly_one_period_old_no_longer_counts():
    assert reserve_turn((NOW - CAP.period_seconds, NOW - 50, NOW - 10), NOW, CAP) == (NOW - 50, NOW - 10, NOW)


def test_a_start_just_inside_the_period_still_counts():
    assert reserve_turn((NOW - CAP.period_seconds + 1, NOW - 50, NOW - 10), NOW, CAP) is None


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
