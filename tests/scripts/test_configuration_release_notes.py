import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from configuration_release_notes import PLACEHOLDER_SENTENCE, main, notes, schema
from personas.config import configuration_schema
from tests.support.scripts import run_from

PREVIOUS_TAG = "v2026.11.17.2"
VERSION = "2026.11.18.1"


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Backend(Enum):
    LITELLM = "litellm"


class OwnershipBefore(Entry):
    grace_seconds: float = 12.0


class OwnershipAfter(Entry):
    grace_seconds: float = 12.0
    stop_seconds: float = 10.0


class AppBefore(Entry):
    mcp_url: str
    tools: list[str] = []


class AppAfter(Entry):
    mcp_url: str


class Gateway(Entry):
    origin: str
    ca_file: str | None = None


class Database(Entry):
    host: str
    port: int
    backend: Backend
    gateway: Gateway
    write_seconds: float = 2.0
    listens: bool = True
    keep_days: float = math.inf
    names: list[str] = ["first"]
    choice: int | str


class Ownership(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppBefore]


class WithStop(BaseModel):
    ownership: OwnershipAfter = OwnershipAfter()
    apps: list[AppBefore]


class WithoutTools(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppAfter]


class WithDatabase(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppBefore]
    database: Database


class TopLevelBefore(BaseModel):
    ready_file: str | None = None


class TopLevelAfter(BaseModel):
    ready_file: str
    history_bytes: int | None = None


class HistoryBefore(BaseModel):
    history_bytes: int = 49152


class HistoryAfter(BaseModel):
    history_bytes: int = 1000


class HistoryCountedInKibibytes(BaseModel):
    history_bytes: float = 48.0


class HistoryRequired(BaseModel):
    history_bytes: int


class HistoryRequiredAsText(BaseModel):
    history_bytes: str


class IntPort(Entry):
    port: int


class StrPort(Entry):
    port: str


class WithIntPort(BaseModel):
    database: IntPort


class WithStrPort(BaseModel):
    database: StrPort


class LowPort(Entry):
    port: int = Field(ge=1)


class HighPort(Entry):
    port: int = Field(ge=1024)


class WithLowPort(BaseModel):
    database: LowPort


class WithHighPort(BaseModel):
    database: HighPort


class UnboundLookups(BaseModel):
    lookup_calls: int = Field(default=2, gt=0)


class BoundLookups(BaseModel):
    lookup_calls: int = Field(default=2, gt=0, le=4)


class Clip(Entry):
    url: str
    note: str = ""


class TwoGestures(Entry):
    gestures: dict[Literal["wave", "nod"], Clip] = {}


class OneGesture(Entry):
    gestures: dict[Literal["wave"], Clip] = {}


class WithTwoGestures(BaseModel):
    figure: TwoGestures


class WithOneGesture(BaseModel):
    figure: OneGesture


class StringVoices(Entry):
    names: tuple[str, ...] = ()


class IntegerVoices(Entry):
    names: tuple[int, ...] = ()


class WithStringVoices(BaseModel):
    voice: StringVoices = StringVoices()


class WithIntegerVoices(BaseModel):
    voice: IntegerVoices = IntegerVoices()


class AppWithScope(Entry):
    mcp_url: str
    tools: list[str] = []
    scope: str


class WithScope(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppWithScope]


class Memory(Entry):
    keep_days: float = 30.0


class MemoryWithLookup(Entry):
    keep_days: float = 30.0
    lookup_bytes: int


class WithMemory(BaseModel):
    memory: Memory | None = None


class WithLookup(BaseModel):
    memory: MemoryWithLookup | None = None


class AgentWithMemory(Entry):
    client_id: str
    memory: Memory | None = None


class AgentWithLookup(Entry):
    client_id: str
    memory: MemoryWithLookup | None = None


class MemoryWithMode(Entry):
    keep_days: float = 30.0
    mode: str = "x"


class WithModedMemory(BaseModel):
    memory: MemoryWithMode | None = None


class WithAgentMemory(BaseModel):
    agents: list[AgentWithMemory]


class WithAgentLookup(BaseModel):
    agents: list[AgentWithLookup]


class IntPortBehindGateway(Entry):
    port: int
    gateway: Gateway


class WithGateway(BaseModel):
    database: IntPortBehindGateway


class AgentWithScope(Entry):
    client_id: str
    memory: Memory | None = None
    scope: str


class WithOptionalAgents(BaseModel):
    agents: list[AgentWithMemory] = []


class WithOptionalScopedAgents(BaseModel):
    agents: list[AgentWithScope] = []


class AgentWithVoice(Entry):
    client_id: str
    memory: Memory | None = None
    voice: Gateway | None = None


class WithAgentVoice(BaseModel):
    agents: list[AgentWithVoice]


class VoiceBehindGateway(Entry):
    origin: str
    ca_file: str | None = None
    gateway: Gateway


class AgentWithVoiceBehindGateway(Entry):
    client_id: str
    memory: Memory | None = None
    voice: VoiceBehindGateway | None = None


class WithAgentVoiceBehindGateway(BaseModel):
    agents: list[AgentWithVoiceBehindGateway]


class WithNoClips(BaseModel):
    clips: list[Clip] = []


class WithOneClip(BaseModel):
    clips: list[Clip] = [Clip(url="https://clip.test")]


class OwnershipOnly(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()


class TunedOwnershipOnly(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore(grace_seconds=20.0)


class MemoryOnByDefault(BaseModel):
    memory: Memory = Memory()


class RuledHistory(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "refusals": [
                "history_keep_bytes: must be a multiple of 1024",
                "history_keep_bytes: must not be above history_bytes",
            ]
        }
    )

    history_bytes: int = 49152
    history_keep_bytes: int = 32768


class RuledShorterHistory(RuledHistory):
    history_bytes: int = 16384


class DeclaresNoRules(BaseModel):
    model_config = ConfigDict(json_schema_extra={"refusals": []})

    ownership: OwnershipAfter = OwnershipAfter()


class RuledOwnership(Entry):
    model_config = ConfigDict(json_schema_extra={"refusals": ["stop_seconds: must be above grace_seconds"]})

    grace_seconds: float = 12.0
    stop_seconds: float = 10.0


class DeclaresRules(BaseModel):
    model_config = ConfigDict(json_schema_extra={"refusals": ["history_keep_bytes: must not be above history_bytes"]})

    ownership: RuledOwnership = RuledOwnership()


def notes_between(before: type[BaseModel], after: type[BaseModel]) -> str:
    return notes(PREVIOUS_TAG, VERSION, before.model_json_schema(), after.model_json_schema())


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        pytest.param(
            Ownership,
            WithStop,
            "- `[ownership] stop_seconds`: added, optional, default `10.0`.\n\n"
            "It goes under the file's `[ownership]` header; a file without one adds the header first.\n\n"
            "```toml\nstop_seconds = 10.0\n```\n",
            id="an optional field with its default",
        ),
        pytest.param(
            Ownership, WithoutTools, "- `[[apps]] tools`: removed: delete it.\n", id="a field of an array of tables"
        ),
        pytest.param(
            Ownership,
            WithDatabase,
            "- `[database]`: added, required.\n\nIt goes at the end of the file.\n\n"
            "```toml\n[database]\nhost = <string>\nport = <integer>\n"
            'backend = <"litellm">\nwrite_seconds = 2.0\nlistens = true\nkeep_days = inf\nnames = ["first"]\n'
            "choice = <integer or string>\n"
            "\n[database.gateway]\norigin = <string>\n```\n",
            id="a required table with what it requires and the defaults it sets",
        ),
        pytest.param(
            TopLevelBefore,
            TopLevelAfter,
            "- `history_bytes`: added, optional.\n\nIt goes above the file's first table.\n\n"
            "```toml\nhistory_bytes = <integer>\n```\n\n"
            "- `ready_file`: now required.\n\nIt goes above the file's first table. Where the file already sets it,"
            " nothing changes.\n\n"
            "```toml\nready_file = <string>\n```\n",
            id="top-level fields added and made required",
        ),
        pytest.param(
            WithIntPort,
            WithStrPort,
            "- `[database] port`: now takes `string`, was `integer`.\n\n"
            "It goes under the file's `[database]` header. Where the file already sets it, it replaces that.\n\n"
            "```toml\nport = <string>\n```\n",
            id="a field whose type changed",
        ),
        pytest.param(
            WithLowPort,
            WithHighPort,
            "- `[database] port`: now takes `integer (at least 1024)`, was `integer (at least 1)`.\n\n"
            "It goes under the file's `[database]` header. Where the file already sets it, it replaces that.\n\n"
            "```toml\nport = <integer (at least 1024)>\n```\n",
            id="a bound tightened on an unchanged type",
        ),
        pytest.param(
            UnboundLookups,
            BoundLookups,
            "- `lookup_calls`: now takes `integer (above 0, at most 4)`, was `integer (above 0)`.\n\n"
            "It goes above the file's first table. Where the file already sets it, it replaces that.\n\n"
            "```toml\nlookup_calls = 2\n```\n",
            id="a bound gained on an unchanged type",
        ),
        pytest.param(
            WithTwoGestures,
            WithOneGesture,
            '- `[figure] gestures`: now takes `map of "wave" to { url = string, optional note = string }`,'
            ' was `map of "wave" or "nod" to { url = string, optional note = string }`.\n\n'
            "It goes under the file's `[figure]` header. Where the file already sets it, it replaces that.\n\n"
            "```toml\ngestures = {}\n```\n",
            id="a map whose key type changed",
        ),
        pytest.param(
            WithStringVoices,
            WithIntegerVoices,
            "- `[voice] names`: now takes `array of integer`, was `array of string`.\n\n"
            "It goes under the file's `[voice]` header; a file without one adds the header first."
            " Where the file already sets it, it replaces that.\n\n```toml\nnames = []\n```\n",
            id="an array whose item type changed",
        ),
        pytest.param(
            HistoryBefore,
            HistoryAfter,
            "- `history_bytes`: default now `1000`, was `49152`: set it to keep the old one.\n\n"
            "It goes above the file's first table. Where the file already sets it, nothing changes.\n\n"
            "```toml\nhistory_bytes = 49152\n```\n",
            id="a field whose default changed, with the line that keeps the old one",
        ),
        pytest.param(
            RuledHistory,
            RuledShorterHistory,
            "- `history_bytes`: default now `16384`, was `49152`: set it to keep the old one.\n\n"
            "It goes above the file's first table. Where the file already sets it, nothing changes. A file that leaves"
            " it unset must still keep `history_keep_bytes: must not be above history_bytes`.\n\n"
            "```toml\nhistory_bytes = 49152\n```\n",
            id="a changed default with the rule a file that takes it must still keep",
        ),
        pytest.param(
            OwnershipOnly,
            TunedOwnershipOnly,
            "- `[ownership]`: default now `{ grace_seconds = 20.0 }`, was its fields' defaults: set it to keep the"
            " old one.\n\nIt goes at the end of the file. Where the file already sets it, nothing changes.\n\n"
            "```toml\n[ownership]\n```\n",
            id="a table's default changed from its fields' defaults",
        ),
        pytest.param(
            TunedOwnershipOnly,
            OwnershipOnly,
            "- `[ownership]`: default now its fields' defaults, was `{ grace_seconds = 20.0 }`: set it to keep the"
            " old one.\n\nIt goes at the end of the file. Where the file already sets it, nothing changes.\n\n"
            "```toml\n[ownership]\ngrace_seconds = 20.0\n```\n",
            id="a table's default changed to its fields' defaults",
        ),
        pytest.param(
            WithMemory,
            MemoryOnByDefault,
            "- `[memory]`: default now its fields' defaults, was none.\n",
            id="a table set by default that was unset, which no TOML can undo",
        ),
        pytest.param(
            WithOneClip,
            WithNoClips,
            '- `[[clips]]`: default now `[]`, was `[{ url = "https://clip.test", note = "" }]`: set it to keep the old'
            " one.\n\nIt goes at the end of the file. Where the file already sets it, nothing changes.\n\n"
            '```toml\n[[clips]]\nurl = "https://clip.test"\nnote = ""\n```\n',
            id="an array of tables whose default entries changed",
        ),
        pytest.param(
            WithNoClips,
            WithOneClip,
            '- `[[clips]]`: default now `[{ url = "https://clip.test", note = "" }]`, was `[]`.\n',
            id="an array of tables given default entries, which no TOML can undo",
        ),
        pytest.param(
            HistoryBefore,
            HistoryCountedInKibibytes,
            "- `history_bytes`: now takes `number`, was `integer`; default now `48.0`, was `49152`.\n\n"
            "It goes above the file's first table. Where the file already sets it, it replaces that.\n\n"
            "```toml\nhistory_bytes = 48.0\n```\n",
            id="a field whose type and default changed",
        ),
        pytest.param(
            HistoryBefore,
            HistoryRequiredAsText,
            "- `history_bytes`: now takes `string`, was `integer`; now required.\n\n"
            "It goes above the file's first table. Where the file already sets it, it replaces that.\n\n"
            "```toml\nhistory_bytes = <string>\n```\n",
            id="a field whose type changed and that became required",
        ),
        pytest.param(
            HistoryRequiredAsText,
            HistoryBefore,
            "- `history_bytes`: now takes `integer`, was `string`; optional, default `49152`.\n\n"
            "It goes above the file's first table. Where the file already sets it, it replaces that.\n\n"
            "```toml\nhistory_bytes = 49152\n```\n",
            id="a required field whose type changed and that gained a default",
        ),
        pytest.param(
            Ownership,
            WithScope,
            "- `[[apps]] scope`: added, required.\n\nIt goes into every existing `[[apps]]` entry.\n\n"
            "```toml\nscope = <string>\n```\n",
            id="a required field of an array of tables, without the header that starts an entry",
        ),
        pytest.param(
            WithMemory,
            WithLookup,
            "- `[memory] lookup_bytes`: added, required.\n\n"
            "It goes under the file's `[memory]` header. It is required only where `[memory]` is set.\n\n"
            "```toml\nlookup_bytes = <integer>\n```\n",
            id="a required field of an optional table",
        ),
        pytest.param(
            WithMemory,
            WithModedMemory,
            '- `[memory] mode`: added, optional, default `"x"`.\n\n'
            "It goes under the file's `[memory]` header, only where `[memory]` is set.\n\n"
            '```toml\nmode = "x"\n```\n',
            id="an optional field of a nullable table, placed only where the table is set",
        ),
        pytest.param(
            WithAgentMemory,
            WithAgentLookup,
            "- `[agents.memory] lookup_bytes`: added, required.\n\n"
            "It goes into the `[agents.memory]` of each `[[agents]]` entry that has one.\n\n"
            "```toml\nlookup_bytes = <integer>\n```\n",
            id="a required field of an optional table inside an array of tables",
        ),
        pytest.param(
            WithIntPort,
            WithGateway,
            "- `[database.gateway]`: added, required.\n\nIt goes at the end of the file.\n\n"
            "```toml\n[database.gateway]\norigin = <string>\n```\n",
            id="a required table inside a required table, placed by its own header",
        ),
        pytest.param(
            WithOptionalAgents,
            WithOptionalScopedAgents,
            "- `[[agents]] scope`: added, required.\n\nIt goes into every existing `[[agents]]` entry.\n\n"
            "```toml\nscope = <string>\n```\n",
            id="a required field of an optional array of tables, needed only in the entries that exist",
        ),
        pytest.param(
            WithAgentMemory,
            WithAgentVoice,
            "- `[agents.voice]`: added, optional.\n\nIt goes at the end of each `[[agents]]` entry that sets it.\n\n"
            "```toml\n[agents.voice]\norigin = <string>\n```\n",
            id="a table of an array of tables, at the end of an entry",
        ),
        pytest.param(
            DeclaresNoRules,
            DeclaresRules,
            "- A new rule across the fields of the top level: `history_keep_bytes: must not be above history_bytes`."
            " A file that breaks it no longer starts.\n\n"
            "- A new rule across the fields of `[ownership]`: `stop_seconds: must be above grace_seconds`."
            " A file that breaks it no longer starts.\n",
            id="rules across fields added",
        ),
    ],
)
def test_the_notes_name_each_change_a_deployer_must_make_with_its_toml(before, after, expected):
    rendered = notes_between(before, after)

    assert rendered.startswith("## Configuration changes\n")
    assert f"Since {PREVIOUS_TAG}," in rendered
    assert f"/v{VERSION}/docs/configuration.md" in rendered
    assert rendered.endswith(expected)


@pytest.mark.parametrize(
    ("before", "after", "explained"),
    [
        pytest.param(Ownership, WithStop, False, id="only defaults"),
        pytest.param(Ownership, WithoutTools, False, id="only a removal"),
        pytest.param(Ownership, WithDatabase, True, id="a required table's placeholders"),
        pytest.param(TopLevelBefore, TopLevelAfter, True, id="an optional and a required field's placeholders"),
        pytest.param(WithIntPort, WithStrPort, True, id="a changed type's placeholder"),
        pytest.param(HistoryBefore, HistoryAfter, False, id="only a changed default"),
        pytest.param(DeclaresNoRules, DeclaresRules, False, id="only rules"),
    ],
)
def test_the_notes_explain_the_placeholder_only_when_a_block_holds_one(before, after, explained):
    paragraphs = notes_between(before, after).split("\n\n")

    assert (PLACEHOLDER_SENTENCE in paragraphs) is explained


type Paste = Callable[[str, list[str]], str]


def above_the_first_table(text: str, block: list[str]) -> str:
    return "\n".join([*block, text])


def at_the_end(text: str, block: list[str]) -> str:
    return "\n".join([text, *block, ""])


def under(header: str) -> Paste:
    def paste(text: str, block: list[str]) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            lines.append(line)
            if line == header:
                lines.extend(block)
        return "\n".join([*lines, ""])

    return paste


def replacing(key: str, paste: Paste) -> Paste:
    def replace(text: str, block: list[str]) -> str:
        kept = [line for line in text.splitlines() if not line.startswith(f"{key} =")]
        return paste("\n".join(kept), block)

    return replace


def at_the_end_of(header: str) -> Paste:
    """Pastes the block before the first header after each line that is the header."""

    def paste(text: str, block: list[str]) -> str:
        lines: list[str] = []
        inside = False
        for line in text.splitlines():
            if inside and line.startswith("["):
                lines.extend(block)
                inside = False
            inside = inside or line == header
            lines.append(line)
        return "\n".join([*lines, *(block if inside else []), ""])

    return paste


def at_the_end_of_each(entry: str) -> Paste:
    """Pastes the block before the first header after each entry's start that belongs to no table of the entry."""
    inside = f"[{entry.strip('[]')}."

    def paste(text: str, block: list[str]) -> str:
        lines: list[str] = []
        in_entry = False
        for line in text.splitlines():
            starts_another_table = line.startswith("[") and not line.startswith(inside)
            if in_entry and starts_another_table:
                lines.extend(block)
            if starts_another_table:
                in_entry = line == entry
            lines.append(line)
        return "\n".join([*lines, *(block if in_entry else []), ""])

    return paste


TWO_AGENTS = '[[agents]]\nclient_id = "a"\n[agents.memory]\nkeep_days = 7.0\n\n[[agents]]\nclient_id = "b"\n'


@pytest.mark.parametrize(
    ("before", "after", "sentence", "paste", "fill", "file", "parsed"),
    [
        pytest.param(
            HistoryBefore,
            HistoryAfter,
            "It goes above the file's first table.",
            above_the_first_table,
            "",
            "[ownership]\ngrace_seconds = 12.0\n",
            {"history_bytes": 49152, "ownership": {"grace_seconds": 12.0}},
            id="a top-level value above the first table",
        ),
        pytest.param(
            Ownership,
            WithStop,
            "It goes under the file's `[ownership]` header",
            under("[ownership]"),
            "",
            '[ownership]\ngrace_seconds = 12.0\n\n[[apps]]\nmcp_url = "https://a.test"\n',
            {"ownership": {"grace_seconds": 12.0, "stop_seconds": 10.0}, "apps": [{"mcp_url": "https://a.test"}]},
            id="a value of an existing table, under its header",
        ),
        pytest.param(
            WithLowPort,
            WithHighPort,
            "Where the file already sets it, it replaces that.",
            replacing("port", under("[database]")),
            "6543",
            "[database]\nport = 5432\n",
            {"database": {"port": 6543}},
            id="a changed value of an existing table, in place of its line",
        ),
        pytest.param(
            Ownership,
            WithScope,
            "It goes into every existing `[[apps]]` entry.",
            under("[[apps]]"),
            '"mcp-a"',
            '[[apps]]\nmcp_url = "https://a.test"\n\n[[apps]]\nmcp_url = "https://b.test"\n',
            {"apps": [{"mcp_url": f"https://{name}.test", "scope": "mcp-a"} for name in "ab"]},
            id="a value of every entry",
        ),
        pytest.param(
            WithAgentMemory,
            WithAgentLookup,
            "It goes into the `[agents.memory]` of each `[[agents]]` entry that has one.",
            under("[agents.memory]"),
            "8192",
            TWO_AGENTS,
            {"agents": [{"client_id": "a", "memory": {"keep_days": 7.0, "lookup_bytes": 8192}}, {"client_id": "b"}]},
            id="a value of an optional table inside the entries that have it",
        ),
        pytest.param(
            WithIntPort,
            WithGateway,
            "It goes at the end of the file.",
            at_the_end,
            '"https://gateway.test"',
            "[database]\nport = 5432\n",
            {"database": {"port": 5432, "gateway": {"origin": "https://gateway.test"}}},
            id="a table inside an existing table, at the end of the file",
        ),
        pytest.param(
            WithAgentMemory,
            WithAgentVoice,
            "It goes at the end of each `[[agents]]` entry that sets it.",
            at_the_end_of_each("[[agents]]"),
            '"https://voice.test"',
            TWO_AGENTS,
            {
                "agents": [
                    {"client_id": "a", "memory": {"keep_days": 7.0}, "voice": {"origin": "https://voice.test"}},
                    {"client_id": "b", "voice": {"origin": "https://voice.test"}},
                ]
            },
            id="a table of the entries, at each entry's end",
        ),
        pytest.param(
            WithAgentVoice,
            WithAgentVoiceBehindGateway,
            "It goes at the end of the `[agents.voice]` of each `[[agents]]` entry that has one.",
            at_the_end_of("[agents.voice]"),
            '"https://gateway.test"',
            '[[agents]]\nclient_id = "a"\n[agents.voice]\norigin = "https://voice.test"\n\n'
            '[[agents]]\nclient_id = "b"\n',
            {
                "agents": [
                    {
                        "client_id": "a",
                        "voice": {"origin": "https://voice.test", "gateway": {"origin": "https://gateway.test"}},
                    },
                    {"client_id": "b"},
                ]
            },
            id="a table of an optional table inside the entries that have it, at that table's end",
        ),
        pytest.param(
            TunedOwnershipOnly,
            OwnershipOnly,
            "It goes at the end of the file.",
            at_the_end,
            "",
            "history_bytes = 1024\n",
            {"history_bytes": 1024, "ownership": {"grace_seconds": 20.0}},
            id="a table's old default, at the end of a file without it",
        ),
    ],
)
def test_each_block_pasted_where_its_sentence_says_parses_into_its_table(
    before, after, sentence, paste, fill, file, parsed
):
    rendered = notes_between(before, after)
    block = rendered.split("```toml\n")[1].split("\n```")[0]

    pasted = paste(file, re.sub(r"<[^>]*>", fill, block).splitlines())

    assert sentence in rendered
    assert tomllib.loads(pasted) == parsed


def test_the_notes_list_every_rule_when_the_previous_schema_names_none():
    rendered = notes_between(Ownership, DeclaresRules)

    assert f"{PREVIOUS_TAG} names no rule across fields in its schema" in rendered
    assert rendered.endswith(
        "- A rule across the fields of the top level: `history_keep_bytes: must not be above history_bytes`.\n\n"
        "- A rule across the fields of `[ownership]`: `stop_seconds: must be above grace_seconds`.\n"
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(Ownership, Ownership, id="no rules named"),
        pytest.param(DeclaresRules, DeclaresRules, id="rules named"),
        pytest.param(HistoryRequired, HistoryAfter, id="a required field given a default, which every file sets"),
    ],
)
def test_the_notes_say_so_when_nothing_changed(before, after):
    rendered = notes_between(before, after)

    assert rendered.splitlines() == [
        "## Configuration changes",
        "",
        "No field of the runner's configuration file was added, removed, made required or changed in what it takes"
        f" or its default, and no rule across fields was added, since {PREVIOUS_TAG}.",
    ]


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({"properties": ["ready_file"]}, id="properties as a list"),
        pytest.param({"properties": {}, "required": "ready_file"}, id="required as a string"),
        pytest.param({"properties": {}, "refusals": [1]}, id="refusals not sentences"),
    ],
)
def test_a_malformed_schema_fails_loud(malformed):
    with pytest.raises(ValueError, match="belongs"):
        notes(PREVIOUS_TAG, VERSION, malformed, malformed)


def test_the_shipped_configuration_compared_with_itself_has_no_change(monkeypatch, capsys):
    main(["schema"])
    written = capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO(written))

    main(["notes", PREVIOUS_TAG, VERSION])

    assert json.loads(written) == configuration_schema()
    assert "No field" in capsys.readouterr().out


def test_the_schema_of_a_tag_older_than_configuration_schema_is_its_file_models():
    older = ModuleType("personas.config")
    older._FileModel = OwnershipBefore  # noqa: SLF001 - the name an older tag gives its file's model

    assert schema(older) == OwnershipBefore.model_json_schema()


SCRIPT = Path(__file__).parents[2] / "scripts" / "configuration_release_notes.sh"
UNTAGGED = "nightly"
CONFIG_AT_TAG = """from pydantic import BaseModel


class _FileModel(BaseModel):
    ready_file: str
"""
CONFIG_AT_CHECKOUT = """from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(json_schema_extra={"refusals": []})

    ready_file: str
    history_bytes: int = 1000


def configuration_schema():
    return _Model.model_json_schema()
"""
# The tag's tree is installed by uv sync; this stand-in gives it the test's own interpreter instead.
FAKE_UV = """#!/bin/sh
case "$1" in
    sync)
        while [ "$1" != --project ]; do shift; done
        mkdir -p "$2/.venv/bin"
        printf '#!/bin/sh\nexec "%s" "$@"\n' "{python}" > "$2/.venv/bin/python"
        chmod +x "$2/.venv/bin/python" ;;
    run)
        shift 3
        exec "{python}" "$@" ;;
esac
"""


def git(tree: Path, *arguments: str) -> None:
    subprocess.run(  # noqa: S603 - git on a repository this test made
        [
            "git",
            "-C",
            str(tree),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.test",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],  # noqa: S607 - git from PATH, as the script runs it
        check=True,
        capture_output=True,
    )


def commit_config(tree: Path, text: str, tag: str) -> None:
    (tree / "src" / "personas" / "config.py").write_text(text)
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", f"config at {tag}")
    git(tree, "tag", tag)


@pytest.fixture
def released_tree(tmp_path: Path) -> Path:
    """A repository whose config.py changed since its newest release tag, and which carries a newer tag that names
    no release, with the shipped script and renderer."""
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "src" / "personas").mkdir(parents=True)
    (tree / "src" / "personas" / "__init__.py").write_text("")
    for name in (SCRIPT.name, "configuration_release_notes.py"):
        shutil.copy2(SCRIPT.parent / name, tree / "scripts" / name)
    git(tree, "init", "-q")
    commit_config(tree, CONFIG_AT_TAG, PREVIOUS_TAG)
    commit_config(tree, CONFIG_AT_CHECKOUT, UNTAGGED)
    return tree


@pytest.fixture
def fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    uv = bin_directory / "uv"
    uv.write_text(FAKE_UV.format(python=sys.executable))
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_directory}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.usefixtures("fake_uv")
def test_the_script_compares_the_newest_release_tag_before_with_the_checkout_after(released_tree):
    result = run_from(released_tree, SCRIPT, VERSION)

    assert result.returncode == 0, result.stderr
    assert f"Since {PREVIOUS_TAG}," in result.stdout
    assert "- `history_bytes`: added, optional, default `1000`.\n" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param((), id="none"),
        pytest.param(("",), id="empty"),
        pytest.param((f"v{VERSION}",), id="with a leading v"),
        pytest.param(("--help",), id="an option"),
    ],
)
def test_the_script_refuses_anything_but_a_version_with_its_usage(released_tree, arguments):
    result = run_from(released_tree, SCRIPT, *arguments)

    assert result.returncode == 2
    assert "usage" in result.stderr
    assert not result.stdout
