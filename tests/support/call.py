"""The call place the stack tests play in, and the lines said in it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.member import STANDING_URI, PlaceRef, place_window_uri

from tests.support.standings import AGENT_SUBJECT, active

if TYPE_CHECKING:
    import asyncio

PLACE_ID = "room-7"
CALL_PLACE = PlaceRef(kind="call", id=PLACE_ID)
PERSON = "visitor-subject"
ADDRESSED = "Mozart, are you there?"
ANSWER = "Here I am."


def say_in_the_call(runner, text, *, author=PERSON, name="Ada", spoken=False, cut=False, act=False, traceparent=None):
    return runner.app.speak(PLACE_ID, author, name, text, spoken=spoken, cut=cut, act=act, traceparent=traceparent)


async def restarted_in_the_call(runner, place_id=PLACE_ID):
    """The runner starts with the place active and has read its window once: whatever it holds was said before."""
    runner.app.stand(AGENT_SUBJECT, active(place_id))
    runner.start()
    await runner.app.until_window_read(place_id)


def pulses_the_window_once(line_said: asyncio.Event, *, ended: asyncio.Event | None = None):
    """A listen that pulses the call's window once the line is said, then ends: at once, or when `ended` is set."""

    async def answered(reply) -> None:
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        await line_said.wait()
        await reply.pulse(place_window_uri(CALL_PLACE))
        if ended is not None:
            await ended.wait()
        await reply.complete()

    return answered
