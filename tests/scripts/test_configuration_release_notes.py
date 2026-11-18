import io
import json
import math
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from configuration_release_notes import PLACEHOLDER_SENTENCE, main, notes, schema

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


class IntPort(Entry):
    port: int


class StrPort(Entry):
    port: str


class WithIntPort(BaseModel):
    database: IntPort


class WithStrPort(BaseModel):
    database: StrPort


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
            "```toml\n[ownership]\nstop_seconds = 10.0\n```\n",
            id="an optional field with its default",
        ),
        pytest.param(
            Ownership, WithoutTools, "- `[[apps]] tools`: removed: delete it.\n", id="a field of an array of tables"
        ),
        pytest.param(
            Ownership,
            WithDatabase,
            "- `[database]`: added, required.\n\n```toml\n[database]\nhost = <string>\nport = <integer>\n"
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
            "- `ready_file`: now required.\n\nIt goes above the file's first table.\n\n"
            "```toml\nready_file = <string>\n```\n",
            id="top-level fields added and made required",
        ),
        pytest.param(
            WithIntPort,
            WithStrPort,
            "- `[database] port`: now takes `string`, was `integer`.\n\n```toml\n[database]\nport = <string>\n```\n",
            id="a field whose type changed",
        ),
        pytest.param(
            HistoryBefore,
            HistoryAfter,
            "- `history_bytes`: default now `1000`, was `49152`: set it to keep the old one.\n",
            id="a field whose default changed",
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
            "- `[memory] lookup_bytes`: added, required.\n\nIt is required only where `[memory]` is set.\n\n"
            "```toml\n[memory]\nlookup_bytes = <integer>\n```\n",
            id="a required field of an optional table",
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


def test_the_notes_list_every_rule_when_the_previous_schema_names_none():
    rendered = notes_between(Ownership, DeclaresRules)

    assert f"{PREVIOUS_TAG} names no rule across fields in its schema" in rendered
    assert rendered.endswith(
        "- A rule across the fields of the top level: `history_keep_bytes: must not be above history_bytes`.\n\n"
        "- A rule across the fields of `[ownership]`: `stop_seconds: must be above grace_seconds`.\n"
    )


@pytest.mark.parametrize(
    "unchanged", [pytest.param(Ownership, id="no rules named"), pytest.param(DeclaresRules, id="rules named")]
)
def test_the_notes_say_so_when_nothing_changed(unchanged):
    rendered = notes_between(unchanged, unchanged)

    assert rendered.splitlines() == [
        "## Configuration changes",
        "",
        "No field of the runner's configuration file was added, removed, made required or changed in its type or"
        f" default, and no rule across fields was added, since {PREVIOUS_TAG}.",
    ]


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({"properties": ["ready_file"]}, id="properties as a list"),
        pytest.param({"properties": {}, "required": "ready_file"}, id="required as a string"),
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

    assert json.loads(written) == schema()
    assert "No field" in capsys.readouterr().out


SCRIPT = Path(__file__).parents[2] / "scripts" / "configuration_release_notes.sh"
UNTAGGED = "nightly"
CONFIG_AT_TAG = """from pydantic import BaseModel


class _FileModel(BaseModel):
    ready_file: str
"""
CONFIG_AT_CHECKOUT = """from pydantic import BaseModel, ConfigDict


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


def run_script(tree: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - the shipped script under test, at a path this module computes
        [str(tree / "scripts" / SCRIPT.name), *arguments], capture_output=True, text=True, check=False
    )


@pytest.mark.usefixtures("fake_uv")
def test_the_script_compares_the_newest_release_tag_before_with_the_checkout_after(released_tree):
    result = run_script(released_tree, VERSION)

    assert result.returncode == 0, result.stderr
    assert f"Since {PREVIOUS_TAG}," in result.stdout
    assert "- `history_bytes`: added, optional, default `1000`.\n" in result.stdout


@pytest.mark.parametrize("arguments", [pytest.param((), id="none"), pytest.param(("",), id="empty")])
def test_the_script_refuses_a_missing_version_with_its_usage(released_tree, arguments):
    result = run_script(released_tree, *arguments)

    assert result.returncode == 2
    assert "usage" in result.stderr
    assert not result.stdout
