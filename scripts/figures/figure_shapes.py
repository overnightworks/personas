"""The primitive shapes of the figures: ellipsoids and tapered tubes along any axis, every triangle wound outward."""

import math
from dataclasses import dataclass

type Vector = tuple[float, float, float]

RINGS = 12
SEGMENTS = 18
# A tube whose axis lies this close to Z cannot take Z as its forward axis.
ALONG_Z_LIMIT = 0.9


@dataclass(frozen=True)
class Shape:
    """Positions with their unit normals, and triangles that wind counter-clockwise seen from outside."""

    positions: tuple[Vector, ...]
    normals: tuple[Vector, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class TubeFrame:
    """The axes of a tube: along its length, forward across it, and side, with along = forward x side."""

    along: Vector
    forward: Vector
    side: Vector


def add(first: Vector, second: Vector) -> Vector:
    return (first[0] + second[0], first[1] + second[1], first[2] + second[2])


def subtract(first: Vector, second: Vector) -> Vector:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def multiply(first: Vector, second: Vector) -> Vector:
    return (first[0] * second[0], first[1] * second[1], first[2] * second[2])


def scaled(vector: Vector, factor: float) -> Vector:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def dot(first: Vector, second: Vector) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def cross(first: Vector, second: Vector) -> Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def normalised(vector: Vector) -> Vector:
    return scaled(vector, 1.0 / (math.sqrt(dot(vector, vector)) or 1.0))


def ellipsoid(centre: Vector, radii: Vector, rings: int = RINGS, segments: int = SEGMENTS) -> Shape:
    positions: list[Vector] = []
    normals: list[Vector] = []
    for ring in range(rings + 1):
        polar = math.pi * ring / rings
        for segment in range(segments + 1):
            azimuth = 2 * math.pi * segment / segments
            unit = (math.sin(polar) * math.sin(azimuth), math.cos(polar), math.sin(polar) * math.cos(azimuth))
            positions.append(add(centre, multiply(radii, unit)))
            normals.append(normalised((unit[0] / radii[0], unit[1] / radii[1], unit[2] / radii[2])))
    return Shape(tuple(positions), tuple(normals), _grid_indices(rings, segments))


def tapered_tube(start: Vector, end: Vector, start_radius: float, end_radius: float, depth: float = 1.0) -> Shape:
    """A capped tube from start to end in any direction; `depth` flattens its cross-section along the forward axis."""
    frame = _tube_frame(start, end)
    slope = (start_radius - end_radius) / math.dist(start, end)
    positions: list[Vector] = []
    normals: list[Vector] = []
    for centre, radius in ((start, start_radius), (end, end_radius)):
        for segment in range(SEGMENTS + 1):
            position, normal = _ring_point(frame, centre, radius, depth, slope, _ring_angle(segment))
            positions.append(position)
            normals.append(normal)
    indices = list(_grid_indices(1, SEGMENTS))
    for centre, radius, outward in ((start, start_radius, -1.0), (end, end_radius, 1.0)):
        cap_normal = scaled(frame.along, outward)
        hub = len(positions)
        positions.append(centre)
        positions.extend(
            _ring_point(frame, centre, radius, depth, slope, _ring_angle(segment))[0] for segment in range(SEGMENTS + 1)
        )
        normals.extend([cap_normal] * (SEGMENTS + 2))
        indices.extend(_cap_indices(hub, outward < 0))
    return Shape(tuple(positions), tuple(normals), tuple(indices))


def _ring_angle(segment: int) -> float:
    return 2 * math.pi * segment / SEGMENTS


def _tube_frame(start: Vector, end: Vector) -> TubeFrame:
    along = normalised(subtract(end, start))
    reference: Vector = (0.0, 1.0, 0.0) if abs(along[2]) > ALONG_Z_LIMIT else (0.0, 0.0, 1.0)
    forward = normalised(subtract(reference, scaled(along, dot(reference, along))))
    return TubeFrame(along, forward, cross(along, forward))


def _ring_point(
    frame: TubeFrame, centre: Vector, radius: float, depth: float, slope: float, angle: float
) -> tuple[Vector, Vector]:
    cosine, sine = math.cos(angle), math.sin(angle)
    offset = add(scaled(frame.forward, radius * depth * cosine), scaled(frame.side, -radius * sine))
    radial = add(scaled(frame.forward, cosine / depth), scaled(frame.side, -sine))
    normal = normalised(add(normalised(radial), scaled(frame.along, slope)))
    return add(centre, offset), normal


def _grid_indices(rows: int, columns: int) -> tuple[int, ...]:
    indices: list[int] = []
    for row in range(rows):
        for column in range(columns):
            first = row * (columns + 1) + column
            second = first + columns + 1
            indices.extend([first, second, first + 1, first + 1, second, second + 1])
    return tuple(indices)


def _cap_indices(hub: int, facing_start: bool) -> list[int]:
    indices: list[int] = []
    for segment in range(SEGMENTS):
        first = hub + 1 + segment
        indices.extend([hub, first, first + 1] if facing_start else [hub, first + 1, first])
    return indices
