"""What every surface of a figure is made of: vectors, a shape of triangles wound outward, and the ellipsoid."""

import math
from dataclasses import dataclass

type Vector = tuple[float, float, float]


@dataclass(frozen=True)
class Shape:
    """Positions with their unit normals, and triangles that wind counter-clockwise seen from outside."""

    positions: tuple[Vector, ...]
    normals: tuple[Vector, ...]
    indices: tuple[int, ...]


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


def ellipsoid(centre: Vector, radii: Vector, rings: int, segments: int) -> Shape:
    """An ellipsoid of `rings` bands from pole to pole, each of `segments` faces around."""
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


def _grid_indices(rows: int, columns: int) -> tuple[int, ...]:
    indices: list[int] = []
    for row in range(rows):
        for column in range(columns):
            first = row * (columns + 1) + column
            second = first + columns + 1
            indices.extend([first, second, first + 1, first + 1, second, second + 1])
    return tuple(indices)
