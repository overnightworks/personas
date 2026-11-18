"""The release notes' configuration lines: python3 scripts/configuration_release_notes.py schema | notes ...

``schema`` writes the JSON schema of the runner's configuration file as the checked-out ``personas.config``
validates it. ``notes <previous-tag> <version> < before.json`` compares the previous release's schema, read from
standard input, with the checked-out one and writes each change a deployer must know of: a field added, removed,
newly required, changed in what it takes (its type, its bounds, a map's key or value type) or in its default, a
table's included, with the TOML it needs and where that goes, so that it pastes into the file as it is, and each new
rule across fields. The schema is the file's one owner, so the notes need no entry a change could forget;
``scripts/configuration_release_notes.sh`` runs both.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

REFERENCE_URL = "https://github.com/overnightworks/personas/blob/v{version}/docs/configuration.md"

type JsonObject = dict[str, object]

AT_LEAST = "at least"
AT_MOST = "at most"

BOUNDS: dict[str, tuple[str, tuple[str, str] | None]] = {
    "minimum": (AT_LEAST, None),
    "exclusiveMinimum": ("above", None),
    "maximum": (AT_MOST, None),
    "exclusiveMaximum": ("below", None),
    "multipleOf": ("a multiple of", None),
    "minLength": (AT_LEAST, ("character", "characters")),
    "maxLength": (AT_MOST, ("character", "characters")),
    "pattern": ("matching", None),
    "minItems": (AT_LEAST, ("item", "items")),
    "maxItems": (AT_MOST, ("item", "items")),
    "minProperties": (AT_LEAST, ("entry", "entries")),
    "maxProperties": (AT_MOST, ("entry", "entries")),
}
"""The JSON schema's keywords that bound a value, each with the words before the bound and, for a count, what it
counts."""


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
    shape: str
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


def table_shape(*, repeated: bool) -> str:
    return "array of tables" if repeated else "table"


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


def bound(words: str, value: object, unit: tuple[str, str] | None) -> str:
    if unit is None:
        return f"{words} {json.dumps(value)}"
    return f"{words} {value} {unit[0] if value == 1 else unit[1]}"


def overrides(table: JsonObject, default: object | None) -> JsonObject | None:
    """What a table's default instance sets apart from its fields' own defaults, so that a field's changed default
    is not counted again as its table's; none where the table is unset unless the file sets it."""
    if default is None:
        return None
    own = {name: as_object(node).get("default") for name, node in as_object(table.get("properties", {})).items()}
    return {name: value for name, value in as_object(default).items() if value != own.get(name)}


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

    def kind(self, node: JsonObject) -> str:
        """What a field takes, as a deployer reads it; an array names what each of its items takes."""
        return " or ".join(self.described(each) for each in self.choices(node))

    def described(self, choice: JsonObject) -> str:
        """What one choice takes, with the bounds it sets, so that a bound tightened on an unchanged type reads as
        a change of what the field takes."""
        if "enum" in choice:
            return " or ".join(json.dumps(each) for each in as_list(choice["enum"]))
        if "const" in choice:
            return json.dumps(choice["const"])
        if "properties" in choice:
            return self.bounded(self.inline_table(choice), choice)
        if choice.get("type") == "array" and "items" in choice:
            return self.bounded(f"array of {self.kind(as_object(choice['items']))}", choice)
        values = choice.get("additionalProperties")
        if isinstance(values, dict):
            keys = self.kind(as_object(choice.get("propertyNames", {"type": "string"})))
            return self.bounded(f"map of {keys} to {self.kind(cast('JsonObject', values))}", choice)
        return self.bounded(str(choice.get("format") or choice.get("type") or "value"), choice)

    def inline_table(self, choice: JsonObject) -> str:
        """A table that is a value, such as a map's, written with each field and what it takes."""
        required = set(as_list(choice.get("required", [])))
        fields = (
            f"{'' if name in required else 'optional '}{name} = {self.kind(as_object(node))}"
            for name, node in as_object(choice["properties"]).items()
        )
        return f"{{ {', '.join(fields)} }}"

    def shape(self, node: JsonObject) -> str:
        """What a field takes without its bounds, a map as any map and an inline table as any table: a change that
        keeps the shape may leave the value a file sets fitting."""
        return " or ".join(self.unbounded(each) for each in self.choices(node))

    def unbounded(self, choice: JsonObject) -> str:
        if "enum" in choice or "const" in choice:
            return self.described(choice)
        if "properties" in choice:
            return "table"
        if choice.get("type") == "array" and "items" in choice:
            return f"array of {self.shape(as_object(choice['items']))}"
        if isinstance(choice.get("additionalProperties"), dict):
            return "map"
        return str(choice.get("format") or choice.get("type") or "value")

    @staticmethod
    def bounded(kind: str, choice: JsonObject) -> str:
        bounds = [bound(words, choice[keyword], unit) for keyword, (words, unit) in BOUNDS.items() if keyword in choice]
        return f"{kind} ({', '.join(bounds)})" if bounds else kind

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

    def table_kind(self, choice: JsonObject, *, repeated: bool) -> str:
        """A table field takes a table, whose own fields are compared one by one; an array of tables keeps its
        bounds."""
        return self.bounded("array of tables", choice) if repeated else "table"

    def fields(self, table: JsonObject, prefix: FieldPath) -> Iterator[ConfigField]:
        required = set(as_list(table.get("required", [])))
        for name, value in as_object(table.get("properties", {})).items():
            node = as_object(value)
            choices = self.choices(node)
            table_schema, repeated = self.table(choices)
            path = (*prefix, Step(name, repeated=repeated))
            default = node.get("default")
            yield ConfigField(
                path,
                required=name in required,
                default=default if table_schema is None or repeated else overrides(table_schema, default),
                kind=self.kind(node) if table_schema is None else self.table_kind(choices[0], repeated=repeated),
                shape=self.shape(node) if table_schema is None else table_shape(repeated=repeated),
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
    if not now.required and not then.required and then.default != now.default:
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
    if isinstance(value, dict):
        pairs = (f"{name} = {toml_value(each)}" for name, each in cast("JsonObject", value).items())
        return f"{{ {', '.join(pairs)} }}" if value else "{}"
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
    """The TOML that sets the field: a value alone, or a table with each field it requires and each default it sets,
    and the tables it requires. A value comes without its table's header, which the file already holds or, for an
    array of tables, would start a new entry where it is pasted; ``placement`` says where it goes.

    A value without a default is written as ``<kind>``, which no TOML parser accepts, so a pasted example fails
    at start until the deployer writes the real value.
    """
    if not field.table:
        yield field
        return
    yield table_header(field.path)
    inside = [each for each in every.values() if each.path[:-1] == field.path]
    yield from (each for each in inside if not each.table and (each.required or not each.placeholder))
    for each in inside:
        if each.table and each.required:
            yield ""
            yield from toml_entries(each, every)


def old_default(then: ConfigField) -> Iterator[TomlEntry]:
    """The TOML that keeps a changed default as it was: the field at its old default, the table's header with what
    its old default instance set, or each old default entry; nothing where the old default left it unset or empty,
    which no TOML of a table can write."""
    if then.default is None:
        return
    if not then.table:
        yield then
        return
    entries = as_list(then.default) if then.path[-1].repeated else [then.default]
    for entry in entries:
        yield table_header(then.path)
        yield from (f"{name} = {toml_value(value)}" for name, value in as_object(entry).items() if value is not None)


def toml_block(
    change: Change, field: ConfigField, then: ConfigField | None, every: dict[FieldPath, ConfigField]
) -> list[TomlEntry]:
    """The TOML a deployer pastes for a change: none for a removal."""
    if change is Change.REMOVED:
        return []
    if change is Change.DEFAULT_CHANGED and then is not None:
        return list(old_default(then))
    return list(toml_entries(field, every))


def placement(change: Change, field: ConfigField, every: dict[FieldPath, ConfigField]) -> Iterator[str]:
    """Where the field's TOML goes in a file that already holds the other tables and entries."""
    ancestors = [every[field.path[:length]] for length in range(len(field.path) - 1, 0, -1)]
    entry = next((each for each in ancestors if each.path[-1].repeated), None)
    optional = next((each for each in ancestors if not each.required), None)
    if entry is None:
        yield from placement_outside_entries(field, ancestors, optional)
    else:
        yield from placement_inside_entry(change, field, ancestors[0], entry, optional)


def placement_outside_entries(
    field: ConfigField, ancestors: list[ConfigField], optional: ConfigField | None
) -> Iterator[str]:
    """Where a field outside every array of tables goes: at the top level, or under its table's header.

    A table whose default is null is a switch, so a file never adds its header for a field: the field goes where the
    table is already set. A table with a default instance may be added, since its header changes nothing."""
    if field.table:
        yield "It goes at the end of the file."
    elif not ancestors:
        yield "It goes above the file's first table."
    else:
        header = f"It goes under the file's `{ancestors[0].label}` header"
        if optional is None or field.required:
            yield f"{header}."
        elif optional.default is None:
            yield f"{header}, only where `{optional.label}` is set."
        else:
            yield f"{header}; a file without one adds the header first."
    if field.required and optional is not None and not field.table:
        yield f"It is required only where `{optional.label}` is set."


def placement_inside_entry(
    change: Change, field: ConfigField, parent: ConfigField, entry: ConfigField, optional: ConfigField | None
) -> Iterator[str]:
    """Where a field inside an array of tables goes: into its entries, under its table there if it has one; a
    table at the end of its entry, or of its table there, since its header takes every line below it. An old
    default is kept only in the entries that leave the field unset, since an entry that sets it keeps its own."""
    if optional is not None and len(optional.path) > len(entry.path):
        has = "one" if optional == parent else f"`{optional.label}`"
        into = "at the end of" if field.table else "into"
        yield f"It goes {into} the `{parent.label}` of each `{entry.label}` entry that has {has}."
        return
    where = "every existing" if field.required else "each"
    that = " that sets it"
    if field.required:
        that = ""
    elif change is Change.DEFAULT_CHANGED:
        that = " that leaves it unset"
    if field.table:
        yield f"It goes at the end of {where} `{entry.label}` entry{that}."
        return
    under = "" if parent == entry else f", under `{parent.label}`"
    yield f"It goes into {where} `{entry.label}` entry{that}{under}."


def already_set(change: Change, field: ConfigField, then: ConfigField | None) -> Iterator[str]:
    """What the block does where the file already sets the field, so that no paste declares a key or table twice;
    where only bounds or a map's keys or values changed, a value the file sets may still fit and stays."""
    if change is Change.TYPE_CHANGED and then is not None and then.shape == field.shape:
        yield (
            "Where the file already sets it, a value that still fits what it now takes stays as it is; one that does"
            " not is changed to fit, or replaced by this."
        )
    elif change is Change.TYPE_CHANGED:
        yield "Where the file already sets it, it replaces that."
    elif change in {Change.NOW_REQUIRED, Change.DEFAULT_CHANGED}:
        yield "Where the file already sets it, nothing changes."


def bound_by(change: Change, field: ConfigField, rules: list[Rule]) -> Iterator[str]:
    """Each rule of the field's table that names a field whose default changed: a file that takes the new default
    may break it."""
    if change is not Change.DEFAULT_CHANGED or field.table:
        return
    named = re.compile(rf"\b{re.escape(field.path[-1].name)}\b")
    for rule in rules:
        if rule.table == field.path[:-1] and named.search(rule.sentence):
            yield f"A file that leaves it unset must still keep `{rule.sentence}`."


def written(field: ConfigField) -> str:
    """A field's default as a deployer reads it; a table's names only what it sets apart from its fields'."""
    if field.default is None:
        return "none"
    if field.table and field.default == {}:
        return "its fields' defaults"
    return f"`{toml_value(field.default)}`"


def default_moved(field: ConfigField, then: ConfigField) -> str:
    return f"default now {written(field)}, was {written(then)}"


def requirement_moved(field: ConfigField, then: ConfigField) -> str:
    """What a type change also did to whether the field is required and to its default."""
    if field.required and not then.required:
        return "; now required"
    if then.required and not field.required:
        return "; optional" if field.default is None else f"; optional, default {written(field)}"
    return "" if then.default == field.default else f"; {default_moved(field, then)}"


def summary(change: Change, field: ConfigField, then: ConfigField | None) -> str:
    if change is Change.REMOVED:
        return "removed: delete it"
    if change is Change.NOW_REQUIRED:
        return "now required"
    if change is Change.TYPE_CHANGED and then is not None:
        return f"now takes `{field.kind}`, was `{then.kind}`{requirement_moved(field, then)}"
    if change is Change.DEFAULT_CHANGED and then is not None:
        keep = ": set it to keep the old one" if next(old_default(then), None) is not None else ""
        return f"{default_moved(field, then)}{keep}"
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
    rules_after = rules_of(after)
    found = changes(fields_of(before), fields_after)
    rules = rule_lines(previous_tag, before, after)
    lines = ["## Configuration changes", ""]
    if not found and not rules:
        lines.append(
            "No field of the runner's configuration file was added, removed, made required or changed in what it takes"
            f" or its default, and no rule across fields was added, since {previous_tag}."
        )
        return "\n".join(lines) + "\n"
    lines.append(
        f"Since {previous_tag}, the runner's configuration file changed as follows; what each field means is in the"
        f" [configuration reference]({REFERENCE_URL.format(version=version)})."
    )
    blocks = [(change, field, then, toml_block(change, field, then, fields_after)) for change, field, then in found]
    if any(isinstance(entry, ConfigField) and entry.placeholder for *_, block in blocks for entry in block):
        lines.extend(["", PLACEHOLDER_SENTENCE])
    for change, field, then, block in blocks:
        lines.extend(["", f"- `{field.label}`: {summary(change, field, then)}."])
        if block:
            sentences = " ".join(
                [
                    *placement(change, field, fields_after),
                    *already_set(change, field, then),
                    *bound_by(change, field, rules_after),
                ]
            )
            toml = [entry if isinstance(entry, str) else assignment(entry) for entry in block]
            lines.extend(["", sentences, "", "```toml", *toml, "```"])
    for line in rules:
        lines.extend(["", line])
    return "\n".join(lines) + "\n"


def schema(configuration: ModuleType) -> JsonObject:
    """The JSON schema of the configuration file that ``configuration``, a tree's ``personas.config``, reads."""
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
    checked_out = schema(importlib.import_module("personas.config"))
    if parsed.command == "schema":
        sys.stdout.write(json.dumps(checked_out) + "\n")
        return
    before = as_object(json.loads(sys.stdin.read()))
    sys.stdout.write(notes(parsed.previous_tag, parsed.version, before, checked_out))


if __name__ == "__main__":
    main()
