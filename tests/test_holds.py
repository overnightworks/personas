from __future__ import annotations

import anyio
import pytest
from agents.member import Interrupted, PlaceWindow

from personas.holds import HoldInbox
from personas.speech import Held, Telling
from tests.support.standings import GENERATION

pytestmark = pytest.mark.anyio

HOLD_ID = "hold-1"


def _window_stating(hold_id: str) -> PlaceWindow:
    return PlaceWindow(generation=GENERATION, starts_at=1, entries=(), interrupted=Interrupted(id=hold_id))


def _standing(hold_id: str) -> Held:
    return Held(hold_id, (), record_gone=False, telling=Telling.NOT_TOLD)


def test_a_window_read_before_the_decision_that_states_the_hold_does_not_bring_it_back():
    inbox = HoldInbox(_standing(HOLD_ID))
    inbox.decided(HOLD_ID)

    inbox.hear(_window_stating(HOLD_ID))

    assert inbox.hold is None


def test_a_window_that_states_a_newer_hold_after_the_decision_holds_him_again():
    inbox = HoldInbox(_standing(HOLD_ID))
    inbox.decided(HOLD_ID)

    inbox.hear(_window_stating("hold-2"))

    assert inbox.hold is not None
    assert inbox.hold.id == "hold-2"


async def test_calls_answered_waits_while_one_of_two_calls_is_still_on_its_way():
    inbox = HoldInbox(_standing(HOLD_ID))

    with inbox.sending():
        with inbox.sending():
            pass
        with anyio.move_on_after(0) as still_waiting:
            await inbox.calls_answered()
        assert still_waiting.cancelled_caught

    with anyio.fail_after(1):
        await inbox.calls_answered()
