"""What the tests of the history share: the places, the lines and the message a turn is told, as the gateway saw it."""

from __future__ import annotations

import secrets

from agents.member import STANDING_URI, PlaceRef

from tests.support.call import PERSON
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import FakeApp, active

TABLE = PlaceRef(kind="table", id="table-1")
OTHER_CALL_ID = "room-9"
SPEAKER = "Ada"
REMARK = "Play the jack."


def a_state() -> dict[str, str]:
    return {"hand": secrets.token_hex(8)}


STATE_BYTES = len('{"hand": "0123456789abcdef"}')
"""The size of every state ``waits`` shows, as a history counts it."""


def told_line_bytes(
    app_url: str,
    place_id: str,
    text: str,
    *,
    seq: int = 1,
    name: str = SPEAKER,
    own: bool = False,
    spoken: bool = False,
) -> int:
    """The bytes a line of a call is told as in the history, its age 0, not new, its text with no escape."""
    told = (
        f'{{"entry": "line", "app": "{app_url}", "place": {{"kind": "call", "id": "{place_id}"}}, "seconds_ago": 0,'
        f' "seq": {seq}, "name": "{name}", "kind": "message", "text": "{text}", "own": {str(own).lower()},'
        f' "spoken": {str(spoken).lower()}, "new": false}}'
    )
    return len(told.encode())


def told_day_line_bytes(text: str, *, name: str = SPEAKER, own: bool = False, spoken: bool = False) -> int:
    """The bytes a line of the day is told as, its text with no escape."""
    told = (
        f'{{"name": "{name}", "kind": "message", "text": "{text}", "own": {str(own).lower()},'
        f' "spoken": {str(spoken).lower()}}}'
    )
    return len(told.encode())


def url_of(runner, number: int) -> str:
    return runner.config.apps[number].mcp_url


def at_table():
    return active(TABLE.id, TABLE.kind, serves_view=True)


def waits(app: FakeApp, revision: int) -> dict[str, str]:
    state = a_state()
    app.show_view(TABLE.id, revision, state)
    return state


def entry(
    app_url: str,
    place_id: str,
    text: str,
    *,
    seq: int = 1,
    seconds_ago: int = 0,
    name: str = SPEAKER,
    own: bool = False,
    spoken: bool = False,
    new: bool = False,
) -> dict[str, object]:
    return {
        "entry": "line",
        "app": app_url,
        "place": {"kind": "call", "id": place_id},
        "seconds_ago": seconds_ago,
        "seq": seq,
        "name": name,
        "kind": "message",
        "text": text,
        "own": own,
        "spoken": spoken,
        "new": new,
    }


def view_entry(
    app_url: str,
    state: dict[str, str],
    *,
    revision: int,
    awaits_you: bool = True,
    seconds_ago: int = 0,
    new: bool = False,
) -> dict[str, object]:
    return {
        "entry": "view",
        "app": app_url,
        "place": {"kind": TABLE.kind, "id": TABLE.id},
        "seconds_ago": seconds_ago,
        "revision": revision,
        "awaits_you": awaits_you,
        "state": state,
        "new": new,
    }


def answered(text: str, *, is_error: bool = False) -> dict[str, object]:
    return {"answered": True, "is_error": is_error, "text": text}


def unanswered(code: str) -> dict[str, object]:
    return {"answered": False, "code": code}


def act_entry(
    app_url: str, tool: str, arguments: dict[str, object], outcome: dict[str, object], *, seconds_ago: int = 0
) -> dict[str, object]:
    return {
        "entry": "act",
        "app": app_url,
        "seconds_ago": seconds_ago,
        "tool": tool,
        "arguments": arguments,
        "outcome": outcome,
        "new": False,
    }


def woke_at_table(app_url: str) -> dict[str, object]:
    return {"why": "view", "app": app_url, "place": {"kind": TABLE.kind, "id": TABLE.id}}


def woke_at_lines(app_url: str, place_id: str, *, missed: bool = False) -> dict[str, object]:
    return {"why": "lines", "app": app_url, "place": {"kind": "call", "id": place_id}, "missed": missed}


def turns_woken_by(runner, why: str):
    return runner.gateway.turns(why)


def view_turns(runner):
    return turns_woken_by(runner, "view")


def written_turns(runner):
    return turns_woken_by(runner, "lines")


def histories(runner) -> list[object]:
    return [opening.room["history"] for opening in view_turns(runner)]


def history_of(entries: list[dict[str, object]], *, older_dropped: bool = False) -> dict[str, object]:
    return {"older_dropped": older_dropped, "entries": entries}


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
    app: FakeApp, place_id: str, text: str, *, author: str = PERSON, name: str = SPEAKER, spoken: bool = False
) -> None:
    reads = app.window_reads(place_id)
    app.speak(place_id, author, name, text, spoken=spoken)
    await app.until_window_read(place_id, reads + 1)
    await settled(app)


async def shows(app, revision: int, state=None, *, awaits_you: bool = False) -> dict[str, str]:
    """The place's view shows this revision and the loop of the app has read it and taken it."""
    state = state or a_state()
    reads = app.view_reads(TABLE)
    app.show_view(TABLE.id, revision, state, awaits_you=awaits_you)
    await app.until_view_read(TABLE, reads + 1)
    await settled(app)
    return state
