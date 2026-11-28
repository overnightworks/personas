"""A session reads its bearer from the loop's cached token, which the realm granted, across renewals and suspensions."""

from __future__ import annotations

import pytest
from agents.member_outcomes import StatusFailure

from tests.support.fakes import read_of

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_no_session_reads_a_bearer_the_realm_did_not_grant_across_a_renewal_and_a_suspension(scenario):
    run = scenario([StatusFailure(401), StatusFailure(401), StatusFailure(403), read_of(), read_of()])

    await run.play()

    read = [
        *run.session.opened_with,
        *(bearer for bearer, _ in run.session.entered_with_time),
        *(bearer for bearer, _ in run.session.bearers_with_time),
    ]
    assert run.session.opened >= 3
    assert set(read) <= set(run.tokens.expires_at)
