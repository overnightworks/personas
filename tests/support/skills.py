"""The skills a fake app serves: its skill files, their listing entries and what it says of itself at discovery."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.backend.fake_gateway import completion, tool_round
from agents.front_matter import parse_front_matter, split_front_matter
from agents.skills import SKILLS_EXTENSION, skill_digest, skill_uri
from mcp_types import ReadResourceResult, TextResourceContents

from tests.support.call import ADDRESSED, PLACE_ID, a_turn, restarted, restarted_in_the_call
from tests.support.gateway import Gateway
from tests.support.history import bounded_history, told_line_bytes
from tests.support.hopin import DISCOVER, READ, SKILLS_GET, SKILLS_LIST, discovering
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG, songmaker_url

if TYPE_CHECKING:
    from tests.support.wire import Handler, Rpc

DESCRIPTION = "Greets."
SONGMAKER = "songmaker"
HOPIN = "hopin"
SONG_BODY = f"Make songs {secrets.token_hex(6)}."
SONG_DESCRIPTION = f"Makes songs {secrets.token_hex(4)}."
HOPIN_DESCRIPTION = f"Plays at the table {secrets.token_hex(4)}."


def a_skill_file(
    name: str, *, tools: str | None = "say play", body: str = "Greet people by name.", description: str = DESCRIPTION
) -> bytes:
    """A skill file naming these tools, or none when ``tools`` is ``None``."""
    metadata = "" if tools is None else f"metadata:\n  tools: {tools}\n"
    return f"---\nname: {name}\ndescription: {description}\n{metadata}---\n{body}\n".encode()


def an_entry(name: str, content: bytes) -> dict:
    """The listing entry that promises this file: its front matter and its digest."""
    uri = skill_uri(name)
    front_matter, _ = split_front_matter(content.decode())
    return {
        "uri": uri,
        "frontmatter": parse_front_matter(front_matter),
        "resources": [{"uri": uri, "digest": skill_digest(content), "size": len(content)}],
    }


@dataclass
class Shelf:
    """What an app serves of its skills: with ``declares`` it declares the skills extension and answers its listing and
    its entries, and serves each file of ``files`` (by skill name) unless its name is in ``tampered``, whose file then
    differs from what its entry promised. Without it the app says only its ``instructions`` at discovery."""

    files: dict[str, bytes] = field(default_factory=dict[str, bytes])
    instructions: str | None = None
    declares: bool = True
    tampered: set[str] = field(default_factory=set[str])

    def answers(self, rpc: Rpc) -> Handler | None:
        """The handler for a request of the shelf's, or ``None`` for any other request."""
        if rpc.method == DISCOVER:
            return discovering(
                instructions=self.instructions, extensions={SKILLS_EXTENSION: {}} if self.declares else None
            )
        if not self.declares:
            return None
        if rpc.method == SKILLS_LIST:
            return self._listing
        if rpc.method == SKILLS_GET:
            return self._entry_at(rpc.params["uri"])
        if rpc.method == READ and (name := self._named_by(rpc.params["uri"])) is not None:
            return self._file_of(name)
        return None

    def _named_by(self, uri: str) -> str | None:
        return next((name for name in self.files if skill_uri(name) == uri), None)

    def _entries(self) -> dict[str, dict]:
        return {name: an_entry(name, content) for name, content in self.files.items()}

    async def _listing(self, reply) -> None:
        await reply.raw_result({"skills": list(self._entries().values())})

    def _entry_at(self, uri: str) -> Handler:
        async def entry(reply) -> None:
            await reply.raw_result({"skill": self._entries()[self._named_by(uri)]})

        return entry

    def _file_of(self, name: str) -> Handler:
        async def file(reply) -> None:
            content = self.files[name] + b"tampered" if name in self.tampered else self.files[name]
            contents = [TextResourceContents(uri=skill_uri(name), text=content.decode())]
            await reply.result(ReadResourceResult(contents=contents))

        return file


def skilled(runner, songmaker) -> None:
    """Both apps declare a skill of their own: songmaker's makes songs and hopin's plays at the table."""
    songmaker.shelf = Shelf(
        {
            SONGMAKER: a_skill_file(
                SONGMAKER, tools=f"{CREATE_SONG} {GENERATE_SONG}", body=SONG_BODY, description=SONG_DESCRIPTION
            )
        }
    )
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN, description=HOPIN_DESCRIPTION)})


async def assert_a_round_is_told_byte_for_byte_after_a_restart(runner, songmaker, call) -> None:
    """A turn whose first reply is ``call``, the turn after it and, once the runner restarted, a third: the second tells
    the round of the call and the third tells it byte for byte again."""
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(call), completion("done"), completion("again"), completion("end"))
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)
    await a_turn(runner, 3)
    await restarted(runner)
    await a_turn(runner, 4)
    told = runner.gateway.tool_rounds(3)
    assert len(told) == 1
    assert runner.gateway.tool_rounds(4) == told


async def turns_past_a_cut_of_a_round(runner, songmaker, call, entry_bytes: int) -> None:
    """A turn whose first reply is ``call``, whose entry weighs ``entry_bytes``, and five more turns over a history that
    holds only two lines and that entry: the request that arrives as 3 tells the round, the ones as 5 and 6 do not."""
    skilled(runner, songmaker)
    line_bytes = told_line_bytes(songmaker_url(runner), PLACE_ID, f"{ADDRESSED} {secrets.token_hex(3)}")
    runner.config = bounded_history(runner.config, 2 * line_bytes + entry_bytes)
    replies = [tool_round(call), *(completion(text) for text in ("done", "again", "later", "last", "end"))]
    runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)
    for number in (1, 3, 4, 5, 6):
        await a_turn(runner, number)
