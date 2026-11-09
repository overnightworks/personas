"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import itertools
import json
import secrets

import pytest
from agent_runtime_client.member import LEASE_SECONDS, LISTEN_BOUND_SECONDS, PlaceRef, Standing, StandingState
from agent_runtime_client.tools import ToolDeclaration

from agents.app_session import READ_SECONDS
from agents.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    Accept,
    Memory,
    Reaction,
    Refusal,
    Seen,
    Suspend,
    Token,
    accepts_for,
    active_call_places,
    addressed,
    after_refusal,
    backoff_delay,
    bound_declarations,
    decisive_failure,
    forget_gone,
    reaction_to,
    renewal_due,
    turn_input,
    unread,
)
from agents.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer
from agents.secret import Secret
from tests.support.fakes import line, window_of
from tests.support.hopin import active, invited

TRANSIENT_REFUSALS = [Refusal.SPONSOR_AWAY, Refusal.TWO_SEATS, Refusal.BUSY, Refusal.UNRECOGNISED]


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
    memory, _ = after_refusal(Memory(), accepts_for((first,), Memory())[0], Refusal.STALE_INVITATION)

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
    memory, first = after_refusal(Memory(), attempt, Refusal.BUSY)
    memory, repeated = after_refusal(memory, attempt, Refusal.BUSY)
    _, changed = after_refusal(memory, attempt, Refusal.SPONSOR_AWAY)

    assert (first, repeated, changed) == (True, False, True)


def test_what_is_remembered_of_an_invitation_that_left_the_standing_is_dropped():
    attempt = Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0)
    memory, _ = after_refusal(Memory(), attempt, Refusal.STALE_INVITATION)
    memory, _ = after_refusal(memory, Accept(PlaceRef(kind="call", id="room-8"), "inv-4", 0), Refusal.BUSY)

    forgotten = forget_gone(memory, (invited("room-8", "inv-4"),))

    assert forgotten.refused == {"inv-4": Refusal.BUSY}


def test_a_stale_invitation_stays_refused_when_a_read_leaves_it_out_and_a_later_one_holds_it_again():
    standing = invited("room-7", "inv-3")
    memory, _ = after_refusal(
        Memory(), Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 0), Refusal.STALE_INVITATION
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
    ("seen", "expected"),
    [
        (None, [1, 2, 3]),
        (Seen("generation-1", 0), [1, 2, 3]),
        (Seen("generation-1", 2), [3]),
        (Seen("generation-1", 3), []),
        (Seen("generation-2", 3), [1, 2, 3]),
    ],
    ids=["nothing seen", "seen up to nothing", "seen up to the second", "seen all", "another generation"],
)
def test_the_unread_lines_are_the_ones_after_what_was_seen_and_reading_otherwise_starts_at_seq_1(seen, expected):
    window = window_of(line(1, "a"), line(2, "b"), line(3, "c"))

    assert [each.seq for each in unread(window, seen)] == expected


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

    facts = json.loads(turn_input(PLACE, "Mozart", window, window.entries[2:]))

    assert (facts["kind"], facts["id"], facts["you"]) == (PLACE.kind, PLACE.id, "Mozart")
    assert [(each["seq"], each["text"], each["own"], each["new"]) for each in facts["lines"]] == [
        (4, theirs, False, False),
        (5, own, True, False),
        (6, new, False, True),
    ]
