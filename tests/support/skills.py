"""The skills a fake app serves: its skill files, their listing entries and what it says of itself at discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.skills import SKILLS_EXTENSION, skill_digest, skill_uri
from mcp_types import ReadResourceResult, TextResourceContents

from tests.support.hopin import DISCOVER, READ, SKILLS_GET, SKILLS_LIST, discovering

if TYPE_CHECKING:
    from tests.support.wire import Handler, Rpc

DESCRIPTION = "Greets."


def a_skill_file(name: str, *, tools: str = "say play", body: str = "Greet people by name.") -> bytes:
    return f"---\nname: {name}\ndescription: {DESCRIPTION}\nmetadata:\n  tools: {tools}\n---\n{body}\n".encode()


def an_entry(name: str, content: bytes, *, description: str = DESCRIPTION) -> dict:
    uri = skill_uri(name)
    return {
        "uri": uri,
        "frontmatter": {"name": name, "description": description, "metadata": {"tools": "say play"}},
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
    descriptions: dict[str, str] = field(default_factory=dict[str, str])

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
        return {
            name: an_entry(name, content, description=self.descriptions.get(name, DESCRIPTION))
            for name, content in self.files.items()
        }

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
