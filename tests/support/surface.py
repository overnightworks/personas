"""Judges a triangle mesh the way a renderer that culls back faces sees it."""

from collections.abc import Sequence

type Triple = Sequence[float]

# Twice a triangle's area, squared, below which it has no face to judge.
DEGENERATE = 1e-14


def facing(positions: Sequence[Triple], normals: Sequence[Triple], indices: Sequence[int]) -> list[float]:
    """Per triangle with an area, how far its winding agrees with its vertex normals: positive when it faces them."""
    agreements: list[float] = []
    for start in range(0, len(indices), 3):
        first, second, third = (positions[index] for index in indices[start : start + 3])
        edge_one = [second[axis] - first[axis] for axis in range(3)]
        edge_two = [third[axis] - first[axis] for axis in range(3)]
        winding = (
            edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
            edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
            edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
        )
        if sum(component * component for component in winding) < DEGENERATE:
            continue
        summed = [sum(normals[index][axis] for index in indices[start : start + 3]) for axis in range(3)]
        agreements.append(sum(w * n for w, n in zip(winding, summed, strict=True)))
    return agreements


def seen_from_side(
    positions: Sequence[Triple], normals: Sequence[Triple], indices: Sequence[int], height: float, depth: float
) -> tuple[float, list[float]] | None:
    """Where a ray from far along +x at this height and depth first meets the mesh: how far out, and the vertex normal
    blended across the triangle there; None when it misses."""
    nearest: tuple[float, list[float]] | None = None
    for start in range(0, len(indices), 3):
        corners = indices[start : start + 3]
        first, second, third = (positions[index] for index in corners)
        determinant = (second[1] - first[1]) * (third[2] - first[2]) - (third[1] - first[1]) * (second[2] - first[2])
        if abs(determinant) < DEGENERATE:
            continue
        up, along = height - first[1], depth - first[2]
        towards_second = (up * (third[2] - first[2]) - (third[1] - first[1]) * along) / determinant
        towards_third = ((second[1] - first[1]) * along - up * (second[2] - first[2])) / determinant
        weights = (1.0 - towards_second - towards_third, towards_second, towards_third)
        if min(weights) < 0.0:
            continue
        out = sum(weight * positions[index][0] for weight, index in zip(weights, corners, strict=True))
        if nearest is None or out > nearest[0]:
            blended = [
                sum(weight * normals[index][axis] for weight, index in zip(weights, corners, strict=True))
                for axis in range(3)
            ]
            nearest = (out, blended)
    return nearest
