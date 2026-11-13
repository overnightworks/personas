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
