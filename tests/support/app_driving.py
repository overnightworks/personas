"""What a loop test does to a fake app: shows a view, says a line, and waits until the loop has taken it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.member import STANDING_URI

from tests.support.call import PERSON
from tests.support.history import SPEAKER, TABLE, a_state
from tests.support.standings import AGENT_SUBJECT as SUBJECT

if TYPE_CHECKING:
    from tests.support.hopin import FakeApp


def waits(app: FakeApp, revision: int) -> dict[str, str]:
    state = a_state()
    app.show_view(TABLE.id, revision, state)
    return state


async def settled(app: FakeApp, subject: str = SUBJECT) -> None:
    """Every line said so far in a place of the agent was read and taken by the loop of this app.

    A standing the app signals again makes the loop read once more, and its next standing read starts only after the
    reconcile before it ended; so does the one after that.
    """
    for _ in range(2):
        reads = app.reads[STANDING_URI]
        app.stand(subject, *app.standings[subject])
        await app.until_standing_read(reads + 1)


async def said_and_taken(
    app: FakeApp,
    place_id: str,
    text: str,
    *,
    author: str = PERSON,
    name: str = SPEAKER,
    spoken: bool = False,
    act: bool = False,
    continues: bool = False,
    place_kind: str = "call",
) -> None:
    reads = app.window_reads(place_id, place_kind)
    app.speak(place_id, author, name, text, spoken=spoken, act=act, continues=continues)
    await app.until_window_read(place_id, reads + 1, place_kind)
    await settled(app)


async def shows(app, revision: int, state=None, *, awaits_you: bool = False) -> dict[str, str]:
    """The place's view shows this revision and the loop of the app has read it and taken it."""
    state = state or a_state()
    reads = app.view_reads(TABLE)
    app.show_view(TABLE.id, revision, state, awaits_you=awaits_you)
    await app.until_view_read(TABLE, reads + 1)
    await settled(app)
    return state
