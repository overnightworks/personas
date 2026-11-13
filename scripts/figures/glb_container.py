"""The GLB container: a binary buffer with typed accessors, framed with a JSON document into one file."""

from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

type JsonObject = dict[str, object]

# Coordinates are rounded to this many decimals before they are packed, so the same code writes the same bytes.
ROUNDING = 6
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BINARY_CHUNK = 0x004E4942
ACCESSOR_WIDTHS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class ComponentType(IntEnum):
    FLOAT = 5126
    UNSIGNED_SHORT = 5123
    UNSIGNED_BYTE = 5121


STRUCT_FORMATS = {
    ComponentType.FLOAT: "f",
    ComponentType.UNSIGNED_SHORT: "H",
    ComponentType.UNSIGNED_BYTE: "B",
}


class GlbBuilder:
    """Collects accessors into one buffer, then encodes the file around the caller's document."""

    def __init__(self) -> None:
        self._binary = bytearray()
        self._views: list[JsonObject] = []
        self._accessors: list[JsonObject] = []

    def add_accessor(
        self,
        rows: Sequence[Sequence[float]],
        kind: str,
        component: ComponentType,
        *,
        bounded: bool = False,
        normalized: bool = False,
    ) -> int:
        """Packs the rows as one buffer view and accessor and returns the accessor's index.

        `normalized` marks integers that stand for the range 0 to 1, as colours and skin weights do.
        """
        width = ACCESSOR_WIDTHS[kind]
        values = [
            round(value, ROUNDING) if component is ComponentType.FLOAT else value for row in rows for value in row
        ]
        data = struct.pack("<" + STRUCT_FORMATS[component] * len(values), *values)
        self._binary.extend(bytes(-len(self._binary) % 4))
        self._views.append({"buffer": 0, "byteOffset": len(self._binary), "byteLength": len(data)})
        self._binary.extend(data)
        accessor: JsonObject = {
            "bufferView": len(self._views) - 1,
            "componentType": int(component),
            "count": len(rows),
            "type": kind,
        }
        if normalized:
            accessor["normalized"] = True
        if bounded:
            stored = [struct.unpack("<f", struct.pack("<f", value))[0] for value in values]
            accessor["min"] = [min(stored[column::width]) for column in range(width)]
            accessor["max"] = [max(stored[column::width]) for column in range(width)]
        self._accessors.append(accessor)
        return len(self._accessors) - 1

    def encode(self, document: JsonObject) -> bytes:
        """The GLB file: the document with this builder's buffer, views and accessors, then the buffer."""
        binary = bytes(self._binary) + bytes(-len(self._binary) % 4)
        complete = document | {
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": self._views,
            "accessors": self._accessors,
        }
        text = json.dumps(complete, sort_keys=True, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        total = 12 + 8 + len(text) + 8 + len(binary)
        return (
            struct.pack("<4sII", b"glTF", GLB_VERSION, total)
            + struct.pack("<II", len(text), JSON_CHUNK)
            + text
            + struct.pack("<II", len(binary), BINARY_CHUNK)
            + binary
        )
