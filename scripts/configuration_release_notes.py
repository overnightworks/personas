"""The release notes' configuration lines: python3 scripts/configuration_release_notes.py schema | notes ...

``schema`` writes the JSON schema of the runner's configuration file as the checked-out ``personas.config``
validates it. ``notes <previous-tag> <version> <before.json> <after.json>`` compares two such schemas and writes each
field a deployer must change: added, removed or newly required, with the TOML it needs. The schema is the file's one
owner, so the notes need no entry a change could forget; ``scripts/configuration_release_notes.sh`` runs both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

# The file's model is private to config.py, and every tag the notes compare names it _FileModel.
from personas.config import _FileModel  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

REFERENCE_URL = "https://github.com/overnightworks/personas/blob/v{version}/docs/configuration.md"

type JsonObject = dict[str, object]


@dataclass(frozen=True)
class Step:
    """One key on the way to a field; ``repeated`` marks an array of tables, ``[[name]]``."""

    name: str
    repeated: bool


type FieldPath = tuple[Step, ...]


@dataclass(frozen=True)
class ConfigField:
    path: FieldPath
    required: bool
    default: object | None
    kind: str
    table: bool

    @property
    def label(self) -> str:
        if self.table:
            return table_header(self.path)
        if len(self.path) == 1:
            return self.path[0].name
        return f"{table_header(self.path[:-1])} {self.path[-1].name}"


class Change(Enum):
    ADDED = "added"
    NOW_REQUIRED = "now required"
    REMOVED = "removed"


def table_header(path: FieldPath) -> str:
    dotted = ".".join(step.name for step in path)
    return f"[[{dotted}]]" if path[-1].repeated else f"[{dotted}]"


def as_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"the schema holds {value!r} where an object belongs")
    return cast("JsonObject", value)


def as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"the schema holds {value!r} where a list belongs")
    return cast("list[object]", value)


def fields_of(schema: JsonObject) -> dict[FieldPath, ConfigField]:
    """Every table and field the schema allows, keyed by its path, in the schema's order."""
    definitions = as_object(schema.get("$defs", {}))

    def resolved(node: JsonObject) -> JsonObject:
        reference = node.get("$ref")
        if not isinstance(reference, str):
            return node
        return as_object(definitions[reference.removeprefix("#/$defs/")])

    def described(node: JsonObject) -> str:
        if "enum" in node:
            return " or ".join(json.dumps(choice) for choice in as_list(node["enum"]))
        return str(node.get("format") or node.get("type") or "value")

    def visit(table: JsonObject, prefix: FieldPath) -> Iterator[ConfigField]:
        required = set(as_list(table.get("required", [])))
        for name, value in as_object(table.get("properties", {})).items():
            node = as_object(value)
            choices = [
                resolved(as_object(each)) for each in as_list(node.get("anyOf", [node])) if each != {"type": "null"}
            ]
            only = choices[0] if len(choices) == 1 else {}
            items = resolved(as_object(only.get("items", {})))
            table_schema = only if "properties" in only else items if "properties" in items else None
            path = (*prefix, Step(name, repeated=table_schema is items))
            yield ConfigField(
                path,
                required=name in required,
                default=node.get("default"),
                kind=" or ".join(described(each) for each in choices),
                table=table_schema is not None,
            )
            if table_schema is not None:
                yield from visit(table_schema, path)

    return {each.path: each for each in visit(schema, ())}


def changes(
    before: dict[FieldPath, ConfigField], after: dict[FieldPath, ConfigField]
) -> list[tuple[Change, ConfigField]]:
    """What a deployer must change, a table added or removed whole counted once, not once per field in it."""

    def outermost(path: FieldPath, other: dict[FieldPath, ConfigField]) -> bool:
        return path not in other and (len(path) == 1 or path[:-1] in other)

    found = [(Change.ADDED, now) for path, now in after.items() if outermost(path, before)]
    found.extend(
        (Change.NOW_REQUIRED, now)
        for path, now in after.items()
        if path in before and now.required and not before[path].required
    )
    found.extend((Change.REMOVED, then) for path, then in before.items() if outermost(path, after))
    return found


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return f"[{', '.join(toml_value(each) for each in cast('list[object]', value))}]"
    return json.dumps(value)


def assignment(field: ConfigField) -> str:
    value = f"<{field.kind}>" if field.default is None else toml_value(field.default)
    return f"{field.path[-1].name} = {value}"


def toml_lines(field: ConfigField, every: dict[FieldPath, ConfigField]) -> Iterator[str]:
    """The TOML that sets the field: a value under its table's header, or a table with each field it requires and
    each default it sets, and the tables it requires.

    A value without a default is written as ``<kind>``, which no TOML parser accepts, so a pasted example fails
    at start until the deployer writes the real value.
    """
    if not field.table:
        if len(field.path) > 1:
            yield table_header(field.path[:-1])
        yield assignment(field)
        return
    yield table_header(field.path)
    inside = [each for each in every.values() if each.path[:-1] == field.path]
    yield from (assignment(each) for each in inside if not each.table and (each.required or each.default is not None))
    for each in inside:
        if each.table and each.required:
            yield ""
            yield from toml_lines(each, every)


def summary(change: Change, field: ConfigField) -> str:
    if change is Change.REMOVED:
        return "removed: delete it"
    if change is Change.NOW_REQUIRED:
        return "now required"
    if field.required:
        return "added, required"
    if field.table or field.default is None:
        return "added, optional"
    return f"added, optional, default `{toml_value(field.default)}`"


def notes(previous_tag: str, version: str, before: JsonObject, after: JsonObject) -> str:
    fields_before, fields_after = fields_of(before), fields_of(after)
    found = changes(fields_before, fields_after)
    lines = ["## Configuration changes", ""]
    if not found:
        lines.append(
            f"No field of the runner's configuration file was added, removed or made required since {previous_tag}."
        )
        return "\n".join(lines) + "\n"
    lines.append(
        f"Since {previous_tag}, the runner's configuration file changed as follows; what each field means is in the"
        f" [configuration reference]({REFERENCE_URL.format(version=version)})."
    )
    for change, field in found:
        lines.extend(["", f"- `{field.label}`: {summary(change, field)}."])
        if change is not Change.REMOVED:
            lines.extend(["", "```toml", *toml_lines(field, fields_after), "```"])
    return "\n".join(lines) + "\n"


def schema() -> JsonObject:
    return _FileModel.model_json_schema()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the release notes' configuration lines.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="write the JSON schema of the checked-out configuration file")
    compare = commands.add_parser("notes", help="write the configuration changes between two schemas")
    compare.add_argument("previous_tag", help="the release the schema before is from, as v<version>")
    compare.add_argument("version", help="the release the notes are for, without a v")
    compare.add_argument("before", type=Path, help="the schema at the previous release")
    compare.add_argument("after", type=Path, help="the schema of this release")
    parsed = parser.parse_args(arguments)
    if parsed.command == "schema":
        sys.stdout.write(json.dumps(schema()) + "\n")
        return
    before, after = (as_object(json.loads(path.read_text(encoding="utf-8"))) for path in (parsed.before, parsed.after))
    sys.stdout.write(notes(parsed.previous_tag, parsed.version, before, after))


if __name__ == "__main__":
    main()
