"""The call place the stack tests play in, and the lines said in it."""

from __future__ import annotations

from agent_runtime_client.member import PlaceRef

PLACE_ID = "room-7"
CALL_PLACE = PlaceRef(kind="call", id=PLACE_ID)
PERSON = "visitor-subject"
ADDRESSED = "Mozart, are you there?"
ANSWER = "Here I am."


def say_in_the_call(runner, text, *, author=PERSON, name="Ada"):
    return runner.app.speak(PLACE_ID, author, name, text)
