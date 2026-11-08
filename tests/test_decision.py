"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import secrets

import pytest
from agent_runtime_client.wait import PlaceRef, Standing, StandingState

from agents.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    WAIT_ERROR_CODE,
    Accept,
    Memory,
    Reaction,
    Refusal,
    Token,
    accepts_for,
    after_refusal,
    backoff_delay,
    forget_gone,
    reaction_to,
    renewal_due,
)
from agents.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer, WaitToolError
from agents.secret import Secret
from tests.support.fakes import answer
from tests.support.hopin import active, invited

TRANSIENT_REFUSALS = [Refusal.SPONSOR_AWAY, Refusal.TWO_SEATS, Refusal.BUSY, Refusal.UNRECOGNISED]


def test_an_invited_call_standing_is_accepted_with_its_place_invitation_and_epoch():
    held = answer("c1", invited("room-7", "inv-3", epoch=4))

    assert accepts_for(held, Memory()) == (Accept("room-7", "inv-3", 4),)


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
    assert accepts_for(answer("c1", standing), Memory()) == ()


def test_a_stale_invitation_is_never_accepted_again():
    first = invited("room-7", "inv-3")
    memory, _ = after_refusal(Memory(), accepts_for(answer("c1", first), Memory())[0], Refusal.STALE_INVITATION)

    assert accepts_for(answer("c2", first), memory) == ()
    assert accepts_for(answer("c2", invited("room-7", "inv-4")), memory) == (Accept("room-7", "inv-4", 0),)


@pytest.mark.parametrize("refusal", TRANSIENT_REFUSALS)
def test_any_other_refusal_is_accepted_again_at_the_next_answer(refusal):
    standing = invited("room-7", "inv-3")
    attempt = accepts_for(answer("c1", standing), Memory())[0]
    memory, _ = after_refusal(Memory(), attempt, refusal)

    assert accepts_for(answer("c2", standing), memory) == (attempt,)


def test_a_repeated_refusal_is_news_once_and_a_changed_one_is_news_again():
    attempt = Accept("room-7", "inv-3", 0)
    memory, first = after_refusal(Memory(), attempt, Refusal.BUSY)
    memory, repeated = after_refusal(memory, attempt, Refusal.BUSY)
    _, changed = after_refusal(memory, attempt, Refusal.SPONSOR_AWAY)

    assert (first, repeated, changed) == (True, False, True)


def test_what_is_remembered_of_an_invitation_that_left_the_answer_is_dropped():
    attempt = Accept("room-7", "inv-3", 0)
    memory, _ = after_refusal(Memory(), attempt, Refusal.STALE_INVITATION)
    memory, _ = after_refusal(memory, Accept("room-8", "inv-4", 0), Refusal.BUSY)

    forgotten = forget_gone(memory, answer("c2", invited("room-8", "inv-4")))

    assert forgotten.refused == {"inv-4": Refusal.BUSY}


def test_a_stale_invitation_stays_refused_when_an_answer_leaves_it_out_and_a_later_one_holds_it_again():
    standing = invited("room-7", "inv-3")
    memory, _ = after_refusal(Memory(), Accept("room-7", "inv-3", 0), Refusal.STALE_INVITATION)

    memory = forget_gone(memory, answer("c2"))

    assert accepts_for(answer("c3", standing), memory) == ()


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
    ("failure", "already_renewed", "reaction", "code"),
    [
        (TokenStatusFailure(401), False, Reaction.SUSPEND, "http_401"),
        (TokenStatusFailure(400), False, Reaction.SUSPEND, "http_400"),
        (TokenStatusFailure(503), False, Reaction.BACKOFF, None),
        (StatusFailure(401), False, Reaction.RENEW, None),
        (StatusFailure(401), True, Reaction.SUSPEND, "http_401"),
        (StatusFailure(403), False, Reaction.SUSPEND, "http_403"),
        (StatusFailure(400), True, Reaction.SUSPEND, "http_400"),
        (StatusFailure(500), False, Reaction.BACKOFF, None),
        (StatusFailure(502), True, Reaction.BACKOFF, None),
        (Unreachable("down"), False, Reaction.BACKOFF, None),
        (WaitToolError("failed"), False, Reaction.SUSPEND, WAIT_ERROR_CODE),
        (UnusableAnswer("jsonrpc_-32602"), False, Reaction.SUSPEND, "jsonrpc_-32602"),
        (UnusableAnswer("http_307"), True, Reaction.SUSPEND, "http_307"),
    ],
)
def test_each_failure_makes_the_loop_renew_suspend_or_back_off(failure, already_renewed, reaction, code):
    assert reaction_to(failure, already_renewed=already_renewed) == (reaction, code)


def test_a_secret_shows_nothing_of_itself():
    value = secrets.token_hex(8)
    secret = Secret(value)

    assert value not in repr(secret) + str(secret) + repr(Token(secret, 1.0))
