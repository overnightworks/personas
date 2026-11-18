"""The bounds the session puts on the app's stream."""

from agents.member import LEASE_SECONDS

from personas.app_session import READ_SECONDS


def test_the_runner_cuts_a_stream_before_the_lease_of_its_member_would_have_ended():
    assert READ_SECONDS < LEASE_SECONDS
