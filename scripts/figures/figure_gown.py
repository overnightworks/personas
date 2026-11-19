"""A long garment over the trunk, from a hem near the floor up to the neck: what a gown and a long dress share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.figure_body import ACROSS, AHEAD, TRUNK_POWER
from figures.figure_surfaces import interpolated, loft, oval, rippled, smoothstep, spread

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_shapes import Shape, Vector
    from figures.figure_surfaces import Key

GOWN_COLUMNS = 84


@dataclass(frozen=True)
class Pleats:
    """Folds that fall from the garment's hem: how many crests round it, how deep they stand at full, and the heights
    where they start to show and where they are fully deep."""

    count: float
    depth: float
    start: float
    full: float


def gown_ring(course: Sequence[Key], height: float, grown: float, pleats: Pleats) -> list[Vector]:
    """The garment at a height: an oval of the course's (half width, front, back) radii that falls into pleats below
    their start, deepest towards the hem; `grown` moves it outward."""
    centre = (0.0, height, 0.0)
    radii = [radius + grown for radius in interpolated(course, height)]
    ring = oval(centre, ACROSS, AHEAD, radii, GOWN_COLUMNS, TRUNK_POWER)
    return rippled(ring, centre, pleats.count, pleats.depth * smoothstep(pleats.start, pleats.full, height))


def gown(course: Sequence[Key], rings: int, pleats: Pleats) -> Shape:
    """The closed garment from the course's lowest height to its highest."""
    return loft([gown_ring(course, height, 0.0, pleats) for height in spread(course[0][0], course[-1][0], rings)])
