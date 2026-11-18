import json
import math
from enum import Enum

import pytest
from pydantic import BaseModel, ConfigDict

from configuration_release_notes import main, notes, schema

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
            "- `history_bytes`: added, optional.\n\n```toml\nhistory_bytes = <integer>\n```\n\n"
            "- `ready_file`: now required.\n\n```toml\nready_file = <string>\n```\n",
            id="top-level fields added and made required",
        ),
    ],
)
def test_the_notes_name_each_change_a_deployer_must_make_with_its_toml(before, after, expected):
    rendered = notes_between(before, after)

    assert rendered.startswith("## Configuration changes\n")
    assert f"Since {PREVIOUS_TAG}," in rendered
    assert f"/v{VERSION}/docs/configuration.md" in rendered
    assert rendered.endswith(expected)


def test_the_notes_say_so_when_no_field_changed():
    rendered = notes_between(Ownership, Ownership)

    assert rendered.splitlines() == [
        "## Configuration changes",
        "",
        f"No field of the runner's configuration file was added, removed or made required since {PREVIOUS_TAG}.",
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


def test_the_shipped_configuration_compared_with_itself_has_no_change(tmp_path, capsys):
    main(["schema"])
    written = tmp_path / "schema.json"
    written.write_text(capsys.readouterr().out, encoding="utf-8")

    main(["notes", PREVIOUS_TAG, VERSION, str(written), str(written)])

    assert json.loads(written.read_text(encoding="utf-8")) == schema()
    assert "No field" in capsys.readouterr().out
