"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import secrets

import pytest
from agent_runtime_client.member import LEASE_SECONDS, LISTEN_BOUND_SECONDS, PlaceRef, Standing, StandingState

from agents.app_session import READ_SECONDS
from agents.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    Accept,
    Memory,
    Reaction,
    Refusal,
    Suspend,
    Token,
    accepts_for,
    after_refusal,
    backoff_delay,
    forget_gone,
    reaction_to,
    renewal_due,
)
from agents.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer
from agents.secret import Secret
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
