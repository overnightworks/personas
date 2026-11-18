"""The release notes' configuration lines: python3 scripts/configuration_release_notes.py schema | notes ...

``schema`` writes the JSON schema of the runner's configuration file as the checked-out ``personas.config``
validates it. ``notes <previous-tag> <version> < before.json`` compares the previous release's schema, read from
standard input, with the checked-out one and writes each change a deployer must know of: a field added, removed,
newly required or changed in its type or default, with the TOML it needs and where that goes, and each new rule
across fields. The schema is the file's one owner, so the notes need no entry a change could forget;
``scripts/configuration_release_notes.sh`` runs both.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

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
    refusals: tuple[str, ...] = ()

    @property
    def placeholder(self) -> bool:
        """Whether the field's TOML is written as ``<kind>``: it has no default to write."""
        return self.default is None

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
    TYPE_CHANGED = "type changed"
    DEFAULT_CHANGED = "default changed"
    REMOVED = "removed"


@dataclass(frozen=True)
class Rule:
    """A rule across the fields of a table, or of the top level when ``table`` is empty, that config.py refuses a
    file for breaking."""

    table: FieldPath
    sentence: str


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


def refusals_of(table: JsonObject) -> tuple[str, ...]:
    sentences = as_list(table.get("refusals", []))
    if not all(isinstance(each, str) for each in sentences):
        raise ValueError(f"the schema holds {sentences!r} where a list of sentences belongs")
    return tuple(cast("list[str]", sentences))


def described(node: JsonObject) -> str:
    if "enum" in node:
        return " or ".join(json.dumps(choice) for choice in as_list(node["enum"]))
    return str(node.get("format") or node.get("type") or "value")


@dataclass(frozen=True)
class Schema:
    """A configuration file's JSON schema, its ``$ref`` links resolved against its ``$defs``."""

    root: JsonObject

    def resolved(self, node: JsonObject) -> JsonObject:
        reference = node.get("$ref")
        if not isinstance(reference, str):
            return node
        return as_object(as_object(self.root.get("$defs", {}))[reference.removeprefix("#/$defs/")])

    def choices(self, node: JsonObject) -> list[JsonObject]:
        """The values a field allows, its optional ``null`` left out."""
        return [
            self.resolved(as_object(each)) for each in as_list(node.get("anyOf", [node])) if each != {"type": "null"}
        ]

    def table(self, choices: list[JsonObject]) -> tuple[JsonObject | None, bool]:
        """The table a field holds, if it holds one, and whether it is an array of tables."""
        if len(choices) != 1:
            return None, False
        if "properties" in choices[0]:
            return choices[0], False
        items = self.resolved(as_object(choices[0].get("items", {})))
        if "properties" in items:
            return items, True
        return None, False

    def fields(self, table: JsonObject, prefix: FieldPath) -> Iterator[ConfigField]:
        required = set(as_list(table.get("required", [])))
        for name, value in as_object(table.get("properties", {})).items():
            node = as_object(value)
            choices = self.choices(node)
            table_schema, repeated = self.table(choices)
            path = (*prefix, Step(name, repeated=repeated))
            yield ConfigField(
                path,
                required=name in required,
                default=node.get("default"),
                kind=" or ".join(described(each) for each in choices),
                table=table_schema is not None,
                refusals=() if table_schema is None else refusals_of(table_schema),
            )
            if table_schema is not None:
                yield from self.fields(table_schema, path)


def fields_of(schema: JsonObject) -> dict[FieldPath, ConfigField]:
    """Every table and field the schema allows, keyed by its path, in the schema's order."""
    return {each.path: each for each in Schema(schema).fields(schema, ())}


def rules_of(schema: JsonObject) -> list[Rule]:
    """Every rule across fields the schema declares, the top level's first, in the schema's order."""
    tables = (each for each in fields_of(schema).values() if each.table)
    return [Rule((), sentence) for sentence in refusals_of(schema)] + [
        Rule(table.path, sentence) for table in tables for sentence in table.refusals
    ]


def declares_rules(schema: JsonObject) -> bool:
    """Whether the schema names its rules across fields; a tag older than configuration_schema names none."""
    return "refusals" in schema


def kept_change(then: ConfigField, now: ConfigField) -> Change | None:
    """How a field both schemas hold changed, if it changed in a way a deployer must know of."""
    if (then.kind, then.table) != (now.kind, now.table):
        return Change.TYPE_CHANGED
    if now.required and not then.required:
        return Change.NOW_REQUIRED
    if not now.table and not now.required and then.default != now.default:
        return Change.DEFAULT_CHANGED
    return None


def changes(
    before: dict[FieldPath, ConfigField], after: dict[FieldPath, ConfigField]
) -> list[tuple[Change, ConfigField, ConfigField | None]]:
    """What a deployer must know of, each with the field as it is now and as it was, if it was; a table added or
    removed whole is counted once, not once per field in it."""

    def outermost(path: FieldPath, other: dict[FieldPath, ConfigField]) -> bool:
        return path not in other and (len(path) == 1 or path[:-1] in other)

    found: list[tuple[Change, ConfigField, ConfigField | None]] = [
        (Change.ADDED, now, None) for path, now in after.items() if outermost(path, before)
    ]
    kept = [(kept_change(before[path], now), now, before[path]) for path, now in after.items() if path in before]
    found.extend((change, now, then) for change, now, then in kept if change is not None)
    found.extend((Change.REMOVED, then, then) for path, then in before.items() if outermost(path, after))
    order = list(Change)
    return sorted(found, key=lambda each: order.index(each[0]))


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return f"[{', '.join(toml_value(each) for each in cast('list[object]', value))}]"
    return json.dumps(value)


PLACEHOLDER_SENTENCE = (
    "A value written as `<kind>` has no default: a required field's must be filled in, an optional field's may be"
    " left out."
)


def assignment(field: ConfigField) -> str:
    value = f"<{field.kind}>" if field.placeholder else toml_value(field.default)
    return f"{field.path[-1].name} = {value}"


type TomlEntry = str | ConfigField
"""A line of TOML: a header or blank line as it is written, or a field whose ``assignment`` is written."""


def toml_entries(field: ConfigField, every: dict[FieldPath, ConfigField]) -> Iterator[TomlEntry]:
    """The TOML that sets the field: a value under its table's header, or a table with each field it requires and
    each default it sets, and the tables it requires. A value inside an array of tables comes without the
    ``[[name]]`` header, which would start a new entry where it is pasted.

    A value without a default is written as ``<kind>``, which no TOML parser accepts, so a pasted example fails
    at start until the deployer writes the real value.
    """
    if not field.table:
        if len(field.path) > 1 and not field.path[-2].repeated:
            yield table_header(field.path[:-1])
        yield field
        return
    yield table_header(field.path)
    inside = [each for each in every.values() if each.path[:-1] == field.path]
    yield from (each for each in inside if not each.table and (each.required or not each.placeholder))
    for each in inside:
        if each.table and each.required:
            yield ""
            yield from toml_entries(each, every)


def placement(field: ConfigField, every: dict[FieldPath, ConfigField]) -> Iterator[str]:
    """Where the field's TOML goes in a file that already holds the other tables and entries."""
    if len(field.path) == 1 and not field.table:
        yield "It goes above the file's first table."
    ancestors = [every[field.path[:length]] for length in range(len(field.path) - 1, 0, -1)]
    entry = next((each for each in ancestors if each.path[-1].repeated), None)
    if entry is not None:
        where = "every existing" if field.required else "each"
        yield f"It goes into {where} `{entry.label}` entry{'' if field.required else ' that sets it'}."
    optional = next((each for each in ancestors if not each.required), None)
    if field.required and optional is not None:
        yield f"It is required only where `{optional.label}` is set."


def written(value: object | None) -> str:
    return "none" if value is None else f"`{toml_value(value)}`"


def summary(change: Change, field: ConfigField, then: ConfigField | None) -> str:
    if change is Change.REMOVED:
        return "removed: delete it"
    if change is Change.NOW_REQUIRED:
        return "now required"
    if change is Change.TYPE_CHANGED and then is not None:
        return f"now takes `{field.kind}`, was `{then.kind}`"
    if change is Change.DEFAULT_CHANGED and then is not None:
        return f"default now {written(field.default)}, was {written(then.default)}: set it to keep the old one"
    if field.required:
        return "added, required"
    if field.table or field.placeholder:
        return "added, optional"
    return f"added, optional, default `{toml_value(field.default)}`"


def rule_line(rule: Rule, *, new: bool) -> str:
    where = "the top level" if not rule.table else f"`{table_header(rule.table)}`"
    if not new:
        return f"- A rule across the fields of {where}: `{rule.sentence}`."
    return f"- A new rule across the fields of {where}: `{rule.sentence}`. A file that breaks it no longer starts."


def rule_lines(previous_tag: str, before: JsonObject, after: JsonObject) -> list[str]:
    """The rules across fields that the checked-out file adds, or every rule it holds when the previous release's
    schema names none, with a sentence that says so."""
    if declares_rules(before):
        known = set(rules_of(before))
        return [rule_line(rule, new=True) for rule in rules_of(after) if rule not in known]
    rules = rules_of(after)
    if not rules:
        return []
    return [
        f"{previous_tag} names no rule across fields in its schema, so here is each rule this release holds; a file"
        " that breaks one does not start, and some may be older than this release.",
        *(rule_line(rule, new=False) for rule in rules),
    ]


def notes(previous_tag: str, version: str, before: JsonObject, after: JsonObject) -> str:
    fields_after = fields_of(after)
    found = changes(fields_of(before), fields_after)
    rules = rule_lines(previous_tag, before, after)
    lines = ["## Configuration changes", ""]
    if not found and not rules:
        lines.append(
            "No field of the runner's configuration file was added, removed, made required or changed in its type or"
            f" default, and no rule across fields was added, since {previous_tag}."
        )
        return "\n".join(lines) + "\n"
    lines.append(
        f"Since {previous_tag}, the runner's configuration file changed as follows; what each field means is in the"
        f" [configuration reference]({REFERENCE_URL.format(version=version)})."
    )
    blocks = [
        (
            change,
            field,
            then,
            [] if change in {Change.REMOVED, Change.DEFAULT_CHANGED} else list(toml_entries(field, fields_after)),
        )
        for change, field, then in found
    ]
    if any(isinstance(entry, ConfigField) and entry.placeholder for *_, block in blocks for entry in block):
        lines.extend(["", PLACEHOLDER_SENTENCE])
    for change, field, then, block in blocks:
        lines.extend(["", f"- `{field.label}`: {summary(change, field, then)}."])
        if block:
            sentences = " ".join(placement(field, fields_after))
            if sentences:
                lines.extend(["", sentences])
            toml = [entry if isinstance(entry, str) else assignment(entry) for entry in block]
            lines.extend(["", "```toml", *toml, "```"])
    for line in rules:
        lines.extend(["", line])
    return "\n".join(lines) + "\n"


def schema() -> JsonObject:
    configuration = importlib.import_module("personas.config")
    # A tag older than configuration_schema can be the previous release, and it names the file's model _FileModel.
    if not hasattr(configuration, "configuration_schema"):
        return as_object(configuration._FileModel.model_json_schema())  # noqa: SLF001
    accessor = cast("Callable[[], JsonObject]", configuration.configuration_schema)
    return accessor()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the release notes' configuration lines.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="write the JSON schema of the checked-out configuration file")
    compare = commands.add_parser(
        "notes", help="write the configuration changes from the schema on standard input to the checked-out one"
    )
    compare.add_argument("previous_tag", help="the release the schema on standard input is from, as v<version>")
    compare.add_argument("version", help="the release the notes are for, without a v")
    parsed = parser.parse_args(arguments)
    if parsed.command == "schema":
        sys.stdout.write(json.dumps(schema()) + "\n")
        return
    before = as_object(json.loads(sys.stdin.read()))
    sys.stdout.write(notes(parsed.previous_tag, parsed.version, before, schema()))


if __name__ == "__main__":
    main()
