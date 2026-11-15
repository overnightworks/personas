"""A persona's room as one static GLB: one node draws one mesh in one unlit material, its light in the vertex colours.

The room follows figure's convention: metres, +Y up, the floor at y = 0, the figure at the origin facing +Z, the
side toward the camera open; its single-sided material leaves every face that turns away from the camera unseen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.glb_container import ComponentType, GlbBuilder, JsonObject

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.room_light import Lit

UNLIT = "KHR_materials_unlit"
# An unsigned short index reaches this many vertices, so a primitive holds no more.
MOST_VERTICES = 65_536


@dataclass(frozen=True)
class Room:
    """A persona's room: the persona's name and what furnishes it, its light already baked in."""

    name: str
    furnish: Callable[[], Sequence[Lit]]


def build_room(room: Room) -> bytes:
    """The GLB file of one persona's room."""
    builder = GlbBuilder()
    primitives = [_primitive(builder, batch) for batch in _batches(room.furnish())]
    document: JsonObject = {
        "asset": {"version": "2.0", "generator": "personas room generator"},
        "extensionsUsed": [UNLIT],
        "scene": 0,
        "scenes": [{"name": room.name, "nodes": [0]}],
        "nodes": [{"name": "room", "mesh": 0}],
        "meshes": [{"name": "room", "primitives": primitives}],
        "materials": [
            {
                "name": "baked",
                "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]},
                "extensions": {UNLIT: {}},
            }
        ],
    }
    return builder.encode(document)


def _batches(surfaces: Sequence[Lit]) -> list[list[Lit]]:
    """The surfaces in order, in as few primitives as the index width allows."""
    batches: list[list[Lit]] = [[]]
    count = 0
    for surface in surfaces:
        size = len(surface.shape.positions)
        if count + size > MOST_VERTICES:
            batches.append([])
            count = 0
        batches[-1].append(surface)
        count += size
    return batches


def _primitive(builder: GlbBuilder, surfaces: Sequence[Lit]) -> JsonObject:
    positions = [position for surface in surfaces for position in surface.shape.positions]
    colours = [colour for surface in surfaces for colour in surface.colours]
    indices: list[tuple[int]] = []
    base = 0
    for surface in surfaces:
        indices += [(base + index,) for index in surface.shape.indices]
        base += len(surface.shape.positions)
    return {
        "attributes": {
            "POSITION": builder.add_accessor(positions, "VEC3", ComponentType.FLOAT, bounded=True),
            "COLOR_0": builder.add_accessor(colours, "VEC3", ComponentType.FLOAT),
        },
        "indices": builder.add_accessor(indices, "SCALAR", ComponentType.UNSIGNED_SHORT),
        "material": 0,
    }
