"""The pure rules: standings in, actions out; expiry against a given clock; reactions to failures."""

from __future__ import annotations

import pytest
from agent_runtime_client.wait import PlaceRef, Standing, StandingState

from agents.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    Accept,
    CallSource,
    Memory,
    Reaction,
    Refusal,
    Token,
    accepts_for,
    after_refusal,
    backoff_delay,
    forget_gone,
    reaction_to_status,
    renewal_due,
)
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

    assert forgotten.stale == frozenset()
    assert forgotten.refused == {"inv-4": Refusal.BUSY}


@pytest.mark.parametrize(
    ("seconds_left", "due"),
    [(RENEW_BEFORE_EXPIRY_SECONDS + 1, False), (RENEW_BEFORE_EXPIRY_SECONDS, True), (0, True), (-5, True)],
)
def test_a_token_is_due_for_renewal_before_it_expires(seconds_left, due):
    now = 500.0
    token = Token(Secret("value"), expires_at=now + seconds_left)

    assert renewal_due(token, now) is due


def test_no_token_is_due_for_renewal():
    assert renewal_due(None, 0.0)


def test_the_backoff_grows_never_beyond_its_cap_and_is_never_zero():
    delays = [backoff_delay(failures) for failures in range(1, 40)]

    assert delays == sorted(delays)
    assert all(0 < delay <= BACKOFF_CAP_SECONDS for delay in delays)
    assert delays[-1] == BACKOFF_CAP_SECONDS


@pytest.mark.parametrize(
    ("status", "source", "already_renewed", "reaction"),
    [
        (401, CallSource.REALM, False, Reaction.SUSPEND),
        (400, CallSource.REALM, False, Reaction.SUSPEND),
        (503, CallSource.REALM, False, Reaction.BACKOFF),
        (401, CallSource.APP, False, Reaction.RENEW),
        (401, CallSource.APP, True, Reaction.SUSPEND),
        (403, CallSource.APP, False, Reaction.SUSPEND),
        (400, CallSource.APP, True, Reaction.SUSPEND),
        (500, CallSource.APP, False, Reaction.BACKOFF),
        (502, CallSource.APP, True, Reaction.BACKOFF),
    ],
)
def test_each_status_makes_the_loop_renew_suspend_or_back_off(status, source, already_renewed, reaction):
    assert reaction_to_status(status, source, already_renewed=already_renewed) is reaction


def test_a_secret_shows_nothing_of_itself():
    secret = Secret("s3cret-value")

    assert "s3cret-value" not in repr(secret) + str(secret) + repr(Token(secret, 1.0))
