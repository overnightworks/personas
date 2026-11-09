"""The call place the stack tests play in, and the lines said in it."""

from __future__ import annotations

from agent_runtime_client.member import PlaceRef

from tests.support.hopin import AGENT_SUBJECT, active

PLACE_ID = "room-7"
CALL_PLACE = PlaceRef(kind="call", id=PLACE_ID)
PERSON = "visitor-subject"
ADDRESSED = "Mozart, are you there?"
ANSWER = "Here I am."


def say_in_the_call(runner, text, *, author=PERSON, name="Ada"):
    return runner.app.speak(PLACE_ID, author, name, text)


async def restarted_in_the_call(runner, place_id=PLACE_ID):
    """The runner starts with the place active and has read its window once: whatever it holds was said before."""
    runner.app.stand(AGENT_SUBJECT, active(place_id))
    runner.start()
    await runner.app.until_window_read(place_id)
