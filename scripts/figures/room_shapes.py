"""What a room is built of: flat panels, boxes, prisms and turned shapes, painted, and the walls that frame them."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.figure_shapes import Shape, add, cross, normalised, scaled, subtract
from figures.figure_surfaces import loft, oval, seamless, spread

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.figure_parts import Colour, Paint
    from figures.figure_shapes import Vector
    from figures.figure_surfaces import Ring

UP: Vector = (0.0, 1.0, 0.0)
ACROSS: Vector = (1.0, 0.0, 0.0)
AHEAD: Vector = (0.0, 0.0, 1.0)
# The outward direction of each face of a box, as `box` takes the faces it leaves out.
BELOW: Vector = (0.0, -1.0, 0.0)
ABOVE: Vector = UP
BEHIND: Vector = (0.0, 0.0, -1.0)
IN_FRONT: Vector = AHEAD
LEFT: Vector = (-1.0, 0.0, 0.0)
RIGHT: Vector = ACROSS


@dataclass(frozen=True)
class Piece:
    """A surface of the room with the colour of what it is made of at each vertex, before any light falls on it."""

    shape: Shape
    colours: tuple[Colour, ...]
    glows: bool = False


def piece(shape: Shape, paint: Paint, *, glows: bool = False) -> Piece:
    """A surface painted one colour or by place; one that glows shows its colour whatever light falls on it."""
    positions = shape.positions
    colours = tuple(paint(position) for position in positions) if callable(paint) else (paint,) * len(positions)
    return Piece(shape, colours, glows)


def placed(pieces: Sequence[Piece], turn: float, at: Vector) -> list[Piece]:
    """The pieces turned by `turn` radians about the vertical through the origin, then moved to `at`."""
    return [
        Piece(
            Shape(
                tuple(add(turned(position, turn), at) for position in each.shape.positions),
                tuple(turned(normal, turn) for normal in each.shape.normals),
                each.shape.indices,
            ),
            each.colours,
            each.glows,
        )
        for each in pieces
    ]


def turned(vector: Vector, turn: float) -> Vector:
    """The vector turned by `turn` radians about the vertical, from +Z toward +X."""
    x, y, z = vector
    cosine, sine = math.cos(turn), math.sin(turn)
    return (x * cosine + z * sine, y, z * cosine - x * sine)


def rotated(shape: Shape, axis: Vector, angle: float, pivot: Vector) -> Shape:
    """The shape turned by `angle` radians about the line through `pivot` along the unit `axis`."""

    def turn(vector: Vector) -> Vector:
        cosine, sine = math.cos(angle), math.sin(angle)
        along = scaled(axis, (1.0 - cosine) * (axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]))
        return add(add(scaled(vector, cosine), scaled(cross(axis, vector), sine)), along)

    positions = tuple(add(pivot, turn(subtract(position, pivot))) for position in shape.positions)
    return Shape(positions, tuple(turn(normal) for normal in shape.normals), shape.indices)


def panel(
    corner: Vector,
    across: Vector,
    up: Vector,
    across_cuts: Sequence[float],
    up_cuts: Sequence[float],
    hole: Callable[[float, float], bool] | None = None,
) -> Shape:
    """A flat rectangle from `corner` along the unit vectors `across` and `up`, cut at the given distances from it.

    It faces `across` x `up`; a cell whose middle lies in the `hole` is left out.
    """
    normal = normalised(cross(across, up))
    positions: list[Vector] = []
    indices: list[int] = []
    columns = len(across_cuts)
    for along_up in up_cuts:
        positions += [add(corner, add(scaled(across, sideways), scaled(up, along_up))) for sideways in across_cuts]
    for row in range(len(up_cuts) - 1):
        for column in range(columns - 1):
            middle = ((across_cuts[column] + across_cuts[column + 1]) / 2, (up_cuts[row] + up_cuts[row + 1]) / 2)
            if hole is not None and hole(*middle):
                continue
            here, above = row * columns + column, (row + 1) * columns + column
            indices += (here, here + 1, above + 1, here, above + 1, above)
    return Shape(tuple(positions), (normal,) * len(positions), tuple(indices))


def cuts(stops: Sequence[float], cell: float) -> list[float]:
    """The ascending stops with even cuts between each two, no further apart than `cell`."""
    distances = [stops[0]]
    for start, end in itertools.pairwise(stops):
        distances += spread(start, end, max(1, math.ceil((end - start) / cell - 1e-9)))[1:]
    return distances


def box(low: Vector, high: Vector, *, cell: float = math.inf, omit: Sequence[Vector] = ()) -> Shape:
    """A box between two corners with flat faces, each cut into cells no larger than `cell`.

    `omit` names the outward directions of faces that lie against something and are never seen.
    """
    (x0, y0, z0), (x1, y1, z1) = low, high
    width, height, depth = x1 - x0, y1 - y0, z1 - z0
    faces = (
        (IN_FRONT, (x0, y0, z1), ACROSS, UP, width, height),
        (BEHIND, (x1, y0, z0), LEFT, UP, width, height),
        (RIGHT, (x1, y0, z1), BEHIND, UP, depth, height),
        (LEFT, (x0, y0, z0), AHEAD, UP, depth, height),
        (ABOVE, (x0, y1, z1), ACROSS, BEHIND, width, depth),
        (BELOW, (x0, y0, z0), ACROSS, AHEAD, width, depth),
    )
    return joined(
        [
            panel(corner, across, up, cuts((0.0, across_length), cell), cuts((0.0, up_length), cell))
            for outward, corner, across, up, across_length, up_length in faces
            if outward not in omit
        ]
    )


def prism(outline: Sequence[tuple[float, float]], bottom: float, top: float) -> Shape:
    """An upright solid whose plan is the convex `outline` of (x, z) points, from `bottom` to `top`."""
    area = sum(
        x * next_z - next_x * z for (x, z), (next_x, next_z) in zip(outline, [*outline[1:], outline[0]], strict=True)
    )
    ordered = list(outline) if area < 0 else list(reversed(outline))
    return loft([[(x, height, z) for x, z in ordered] for height in (bottom, top)])


def lathe(base: Vector, profile: Sequence[tuple[float, float]], count: int, turn: float = 0.0) -> Shape:
    """A turned solid standing on `base`: a ring of `count` points at each (height, radius) of the profile, upward.

    `turn` turns every ring about the vertical, so four points make a square with its sides along the axes.
    """
    return loft(turned_rings(base, profile, count, turn))


def turned_rings(base: Vector, profile: Sequence[tuple[float, float]], count: int, turn: float = 0.0) -> list[Ring]:
    side, front = (math.cos(turn), 0.0, -math.sin(turn)), (math.sin(turn), 0.0, math.cos(turn))
    return [oval(add(base, (0.0, height, 0.0)), side, front, (radius, radius), count) for height, radius in profile]


def banded(rings: Sequence[Ring], colours: Sequence[Colour]) -> list[Piece]:
    """An open tube through the rings painted in bands, `colours[i]` from ring i to the next, each edge crisp."""
    runs: list[tuple[int, int, Colour]] = []
    for index, colour in enumerate(colours):
        if runs and runs[-1][2] == colour:
            runs[-1] = (runs[-1][0], index + 1, colour)
        else:
            runs.append((index, index + 1, colour))
    shapes = seamless([loft(rings[start : end + 1], capped=False) for start, end, _ in runs])
    return [piece(shape, colour) for shape, (_, _, colour) in zip(shapes, runs, strict=True)]


def joined(shapes: Sequence[Shape]) -> Shape:
    """The shapes as one, each keeping its vertices, normals and triangles."""
    positions: list[Vector] = []
    normals: list[Vector] = []
    indices: list[int] = []
    for shape in shapes:
        base = len(positions)
        positions += shape.positions
        normals += shape.normals
        indices += [base + index for index in shape.indices]
    return Shape(tuple(positions), tuple(normals), tuple(indices))


@dataclass(frozen=True)
class Wall:
    """A wall's own frame: `along` it from `origin` on the floor, up, and out into the room, which it faces."""

    origin: Vector
    along: Vector

    @property
    def out(self) -> Vector:
        return cross(self.along, UP)

    def at(self, along: float, up: float, out: float = 0.0) -> Vector:
        return add(self.origin, add(scaled(self.along, along), add(scaled(UP, up), scaled(self.out, out))))

    def panel(
        self,
        across_cuts: Sequence[float],
        up_cuts: Sequence[float],
        out: float = 0.0,
        hole: Callable[[float, float], bool] | None = None,
    ) -> Shape:
        """A flat sheet on the wall, `out` in front of it, cut at the given places along it and up."""
        start, bottom = across_cuts[0], up_cuts[0]
        return panel(
            self.at(start, bottom, out),
            self.along,
            UP,
            [cut - start for cut in across_cuts],
            [cut - bottom for cut in up_cuts],
            None if hole is None else lambda along, up: hole(along + start, up + bottom),
        )

    def box(self, along: tuple[float, float], up: tuple[float, float], out: tuple[float, float], cell: float) -> Shape:
        """A block standing out of the wall over the given ranges; its face against the wall is left out."""
        corners = (self.at(along[0], up[0], out[0]), self.at(along[1], up[1], out[1]))
        low = (min(corners[0][0], corners[1][0]), up[0], min(corners[0][2], corners[1][2]))
        high = (max(corners[0][0], corners[1][0]), up[1], max(corners[0][2], corners[1][2]))
        return box(low, high, cell=cell, omit=[scaled(self.out, -1.0)])


@dataclass(frozen=True)
class Floor:
    """A floor of square tiles: where it starts, how large it is, how each tile is cut and what colours it."""

    corner: Vector
    width: float
    depth: float
    tile: float
    joint: float
    joint_colour: Colour
    # Where, along a tile from its start, the floor is cut; the last is the tile's own size.
    cuts_in_tile: tuple[float, ...]
    tile_colour: Callable[[int, int], Colour]


def tiled_floor(floor: Floor) -> Piece:
    """The floor from its `corner` across to the right and back, a joint of `joint` round every tile.

    The tiles are counted from the corner along each side, and `tile_colour` takes a tile's column and row.
    """
    columns, rows = round(floor.width / floor.tile), round(floor.depth / floor.tile)

    def tile_cuts(length: float) -> list[float]:
        stops = [0.0]
        for tile in range(round(length / floor.tile)):
            start = tile * floor.tile
            stops += [start + each for each in floor.cuts_in_tile]
        return stops

    def colour(position: Vector) -> Colour:
        across, back = position[0] - floor.corner[0], floor.corner[2] - position[2]
        column, row = min(int(across / floor.tile), columns - 1), min(int(back / floor.tile), rows - 1)
        inside = min(
            across - column * floor.tile,
            (column + 1) * floor.tile - across,
            back - row * floor.tile,
            (row + 1) * floor.tile - back,
        )
        return floor.joint_colour if inside < floor.joint / 2 else floor.tile_colour(column, row)

    return piece(panel(floor.corner, ACROSS, BEHIND, tile_cuts(floor.width), tile_cuts(floor.depth)), colour)


@dataclass(frozen=True)
class Openings:
    """Windows cut through a wall: their sides along it, their sill and head, and the heights where they bend."""

    spans: tuple[tuple[float, float], ...]
    sill: float
    head: float
    bends: tuple[float, ...] = ()


def along_cuts(length: float, openings: Openings | None, cell: float) -> list[float]:
    """The places along a wall of `length` where its sheets are cut: its ends and the sides of its openings."""
    spans = openings.spans if openings else ()
    return cuts(sorted({0.0, length, *(edge for span in spans for edge in span)}), cell)


def pierced_panel(
    wall: Wall,
    length: float,
    up: tuple[float, float],
    openings: Openings | None,
    cell: float,
    paint: Paint,
) -> Piece:
    """The wall from `up[0]` to `up[1]` along its `length`, in cells no larger than `cell`, open at `openings`."""
    heights = {*up, *((openings.sill, openings.head, *openings.bends) if openings else ())}

    def hole(along: float, rise: float) -> bool:
        return (
            openings is not None
            and openings.sill < rise < openings.head
            and any(start < along < end for start, end in openings.spans)
        )

    return piece(wall.panel(along_cuts(length, openings, cell), cuts(sorted(heights), cell), hole=hole), paint)


def leaf(wall: Wall, along: tuple[float, float], head: float, out: float, paint: Paint, cell: float) -> Piece:
    """A door leaf on the wall: a sheet over the range `along` from the floor up to `head`, `out` in front of it."""
    return piece(wall.panel(cuts(along, cell), cuts((0.0, head), cell), out), paint)
