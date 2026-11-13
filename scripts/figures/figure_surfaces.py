"""Smooth surfaces through rows of points: lofts through rings, open patches, and the curves that shape them."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_shapes import Shape, Vector, add, cross, dot, normalised, scaled, subtract

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

type Ring = Sequence[Vector]
type Key = tuple[float, Sequence[float]]
type Face = tuple[Sequence[int], Vector]

WELD_DECIMALS = 6
DEGENERATE_AREA = 1e-18
# Faces that meet at a place within this angle share a normal there; where they meet more sharply the edge stays.
CREASE = math.cos(math.radians(75.0))
ZERO: Vector = (0.0, 0.0, 0.0)


def seamless(shapes: Sequence[Shape]) -> list[Shape]:
    """The same surfaces, shaded as one surface across the places where they touch."""
    return _shaded([(shape.positions, _faces(shape.positions, shape.indices)) for shape in shapes])


def mirrored(shape: Shape) -> Shape:
    """The surface reflected to the other side of the body, still facing outward."""
    positions = tuple((-x, y, z) for x, y, z in shape.positions)
    normals = tuple((-x, y, z) for x, y, z in shape.normals)
    indices = shape.indices
    turned = tuple(indices[start + offset] for start in range(0, len(indices), 3) for offset in (0, 2, 1))
    return Shape(positions, normals, turned)


def loft(rings: Sequence[Ring], *, capped: bool = True) -> Shape:
    """A tube through closed rings of equal length, facing outward when the rings advance along front x side."""
    count = len(rings[0])
    positions = [point for ring in rings for point in ring]
    indices = _strip_indices(len(rings), count, wrap=True)
    if capped:
        for ring_index, facing_start in ((0, True), (len(rings) - 1, False)):
            hub = len(positions)
            positions.append(_centroid(rings[ring_index]))
            indices += _fan_indices(hub, ring_index * count, count, facing_start=facing_start)
    return _smooth(positions, indices)


def patch(rows: Sequence[Sequence[Vector]], *, wrap: bool = False) -> Shape:
    """An open sheet through rows of equal length; `wrap` closes each row into a ring without capping the ends."""
    positions = [point for row in rows for point in row]
    return _smooth(positions, _strip_indices(len(rows), len(rows[0]), wrap=wrap))


def sheet(rows: Sequence[Sequence[Vector]], normal_at: Callable[[Vector], Vector], *, wrap: bool = False) -> Shape:
    """A patch that keeps one vertex per point, in the order of the rows, and takes its normals from a rule by place."""
    positions = tuple(point for row in rows for point in row)
    faces = _faces(positions, _strip_indices(len(rows), len(rows[0]), wrap=wrap))
    indices = tuple(corner for corners, _ in faces for corner in corners)
    return Shape(positions, tuple(normal_at(position) for position in positions), indices)


def sweep(
    path: Sequence[Vector],
    radii: Sequence[Sequence[float]],
    reference: Vector,
    count: int,
    power: float = 2.0,
) -> Shape:
    """A capped tube along a path: an oval of (side, front) radii at each point, its side towards `reference`."""
    rings: list[Ring] = []
    for index, (centre, radius) in enumerate(zip(path, radii, strict=True)):
        along = normalised(subtract(path[min(index + 1, len(path) - 1)], path[max(index - 1, 0)]))
        side = normalised(subtract(reference, scaled(along, dot(reference, along))))
        rings.append(oval(centre, side, cross(side, along), radius, count, power))
    return loft(rings)


def shell(outer: Sequence[Sequence[Vector]], inner: Sequence[Sequence[Vector]]) -> Shape:
    """A sheet with thickness, open along one side: outer rows turn like rings, inner rows line the same places."""
    rings = [[*face, *reversed(lining)] for face, lining in zip(outer, inner, strict=True)]
    hem, top = [*outer[0], *reversed(outer[0])], [*outer[-1], *reversed(outer[-1])]
    return loft([hem, *rings, top], capped=False)


def oval(centre: Vector, side: Vector, front: Vector, radii: Sequence[float], count: int, power: float = 2.0) -> Ring:
    """A ring that turns from `front` to `side`; radii are (side, front) or (side, front, back), power 2 an ellipse."""
    return arc(centre, side, front, radii, [2.0 * math.pi * step / count for step in range(count)], power)


def arc(
    centre: Vector, side: Vector, front: Vector, radii: Sequence[float], angles: Sequence[float], power: float = 2.0
) -> list[Vector]:
    """The points of an oval at the given angles, 0 at `front` and a quarter turn at `side`."""
    half_width, front_depth, back_depth = (*radii, radii[1])[:3]
    points: list[Vector] = []
    for angle in angles:
        across, along = _squared_off(math.sin(angle), power), _squared_off(math.cos(angle), power)
        depth = front_depth if along >= 0.0 else back_depth
        points.append(add(centre, add(scaled(side, across * half_width), scaled(front, along * depth))))
    return points


def spread(start: float, end: float, steps: int) -> list[float]:
    """Evenly spaced values from the start to the end, both included."""
    return [start + (end - start) * step / steps for step in range(steps + 1)]


def rippled(row: Sequence[Vector], centre: Vector, waves: float, amount: float, phase: float = 0.0) -> list[Vector]:
    """The row pushed off its centre in waves: `waves` crests around a ring, each `amount` of the distance high."""
    return [
        add(
            centre,
            scaled(subtract(point, centre), 1.0 + amount * math.cos(2.0 * math.pi * waves * index / len(row) + phase)),
        )
        for index, point in enumerate(row)
    ]


def interpolated(keys: Sequence[Key], at: float) -> tuple[float, ...]:
    """The values of a smooth curve through keyed rows, held constant beyond the first and last key."""
    if at <= keys[0][0]:
        return tuple(keys[0][1])
    if at >= keys[-1][0]:
        return tuple(keys[-1][1])
    upper = next(index for index, key in enumerate(keys) if key[0] >= at)
    lower = upper - 1
    span = keys[upper][0] - keys[lower][0]
    progress = (at - keys[lower][0]) / span
    start_slope, end_slope = _slope(keys, lower), _slope(keys, upper)
    square, cube = progress * progress, progress * progress * progress
    weights = (2 * cube - 3 * square + 1, cube - 2 * square + progress, -2 * cube + 3 * square, cube - square)
    return tuple(
        weights[0] * start + weights[1] * span * start_tangent + weights[2] * end + weights[3] * span * end_tangent
        for start, end, start_tangent, end_tangent in zip(
            keys[lower][1], keys[upper][1], start_slope, end_slope, strict=True
        )
    )


def smoothstep(low: float, high: float, value: float) -> float:
    """0 at or below `low`, 1 at or above `high`, and an eased ramp between."""
    progress = min(1.0, max(0.0, (value - low) / (high - low)))
    return progress * progress * (3.0 - 2.0 * progress)


def bump(distance: float) -> float:
    """A smooth hill: 1 at distance 0, falling to 0 at distance 1 and beyond."""
    if abs(distance) >= 1.0:
        return 0.0
    remaining = 1.0 - distance * distance
    return remaining * remaining


def _smooth(positions: Sequence[Vector], indices: Sequence[int]) -> Shape:
    """The surface shaded smoothly: faces that meet at a place share their area-weighted normal, up to a crease."""
    return _shaded([(positions, _faces(positions, indices))])[0]


def _faces(positions: Sequence[Vector], indices: Sequence[int]) -> list[Face]:
    """Every triangle that has an area: its corners and its normal, as long as twice that area."""
    faces: list[Face] = []
    for start in range(0, len(indices), 3):
        corners = indices[start : start + 3]
        first, second, third = (positions[corner] for corner in corners)
        normal = cross(subtract(second, first), subtract(third, first))
        if dot(normal, normal) >= DEGENERATE_AREA:
            faces.append((corners, normal))
    return faces


def _shaded(surfaces: Sequence[tuple[Sequence[Vector], list[Face]]]) -> list[Shape]:
    """Each surface with its normals, taken from the faces of all of them that meet at each place."""
    meeting: dict[Vector, list[tuple[Vector, Vector]]] = {}
    placed: list[list[Vector]] = []
    for positions, faces in surfaces:
        places = [(round(x, WELD_DECIMALS), round(y, WELD_DECIMALS), round(z, WELD_DECIMALS)) for x, y, z in positions]
        placed.append(places)
        for corners, normal in faces:
            for corner in corners:
                meeting.setdefault(places[corner], []).append((normal, normalised(normal)))
    return [
        _creased(positions, places, faces, meeting) for (positions, faces), places in zip(surfaces, placed, strict=True)
    ]


def _creased(
    positions: Sequence[Vector],
    places: Sequence[Vector],
    faces: Sequence[Face],
    meeting: Mapping[Vector, Sequence[tuple[Vector, Vector]]],
) -> Shape:
    """One vertex per point and normal: a point on a crease becomes a vertex for each side of it."""
    numbers: dict[tuple[int, Vector], int] = {}
    kept: list[Vector] = []
    normals: list[Vector] = []
    indices: list[int] = []
    for corners, normal in faces:
        facing = normalised(normal)
        for corner in corners:
            total = ZERO
            for weighted, direction in meeting[places[corner]]:
                if dot(facing, direction) > CREASE:
                    total = add(total, weighted)
            shared = normalised(total)
            number = numbers.setdefault((corner, shared), len(kept))
            if number == len(kept):
                kept.append(positions[corner])
                normals.append(shared)
            indices.append(number)
    return Shape(tuple(kept), tuple(normals), tuple(indices))


def _slope(keys: Sequence[Key], index: int) -> tuple[float, ...]:
    before, after = keys[max(index - 1, 0)], keys[min(index + 1, len(keys) - 1)]
    return tuple((end - start) / (after[0] - before[0]) for start, end in zip(before[1], after[1], strict=True))


def _squared_off(value: float, power: float) -> float:
    return math.copysign(abs(value) ** (2.0 / power), value)


def _centroid(ring: Ring) -> Vector:
    total = ZERO
    for point in ring:
        total = add(total, point)
    return scaled(total, 1.0 / len(ring))


def _strip_indices(rows: int, columns: int, *, wrap: bool) -> list[int]:
    indices: list[int] = []
    for row in range(rows - 1):
        for column in range(columns if wrap else columns - 1):
            here, beside = row * columns + column, row * columns + (column + 1) % columns
            indices += (here, beside, here + columns, beside, beside + columns, here + columns)
    return indices


def _fan_indices(hub: int, first: int, count: int, *, facing_start: bool) -> list[int]:
    indices: list[int] = []
    for step in range(count):
        here, beside = first + step, first + (step + 1) % count
        indices += (hub, beside, here) if facing_start else (hub, here, beside)
    return indices
