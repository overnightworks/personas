"""Which of an app's declared arguments a call may leave out, read from the tool's declared schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.tools import JsonObject
    from pydantic import JsonValue


def _property_schemas(schema: JsonValue) -> tuple[JsonObject, ...]:
    """The ``properties`` of every object schema ``schema`` is or reaches through its alternatives and array items.
    An app's declared schema has every reference inlined, so none is resolved."""
    if not isinstance(schema, dict):
        return ()
    properties = schema.get("properties")
    found = [properties] if isinstance(properties, dict) else []
    for keyword in ("anyOf", "oneOf", "allOf"):
        alternatives = schema.get(keyword)
        if isinstance(alternatives, list):
            found.extend(each for alternative in alternatives for each in _property_schemas(alternative))
    found.extend(_property_schemas(schema.get("items")))
    return tuple(found)


def split_undeclared(
    arguments: JsonObject, schema: JsonValue, optional: frozenset[str], prefix: str = ""
) -> tuple[JsonObject, tuple[str, ...]]:
    """The arguments to send and the paths left out: an ``optional`` argument, at any depth, that the tool's declared
    ``schema`` does not name. Every other argument stays, so a required one is never dropped."""
    declared = _property_schemas(schema)
    sent: JsonObject = {}
    left_out: list[str] = []
    for name, value in arguments.items():
        path = f"{prefix}{name}"
        named = next((properties[name] for properties in declared if name in properties), None)
        if named is None and path in optional:
            left_out.append(path)
        elif named is not None and isinstance(value, list):
            entries = [
                split_undeclared(entry, named, optional, f"{path}[].") if isinstance(entry, dict) else (entry, ())
                for entry in value
            ]
            sent[name] = [entry for entry, _ in entries]
            left_out.extend(path for _, paths in entries for path in paths)
        else:
            sent[name] = value
    return sent, tuple(left_out)
