from __future__ import annotations

import secrets

from agents.member import PlaceRef
from agents.turn_events import AssistantTextEvent

from personas.voice import Speaking
from tests.support.fakes import FakeClock

PLACE = PlaceRef(kind="call", id="room-7")


async def renewed() -> None:
    return None


def test_a_speaking_shows_no_sentence_in_its_repr():
    marker = secrets.token_hex(8)
    speaking = Speaking(
        object(),
        "https://app.test/mcp",
        PLACE,
        FakeClock(),
        30,
        renewed,
        lambda event, code: None,
        lambda entries: None,
    )

    speaking.hear(AssistantTextEvent(text=f"{marker}. And one more sentence."))

    assert marker not in repr(speaking)
