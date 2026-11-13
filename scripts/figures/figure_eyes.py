"""The eyes: a ball on each eye bone, lids that close over it, and a brow above."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.figure_head import BROW_LEVEL, Face, face_point
from figures.figure_parts import Colour, Layer, Material, Part, mixed
from figures.figure_shapes import Shape, Vector, add, normalised, scaled, subtract
from figures.figure_skeleton import BONE_POSITION
from figures.figure_surfaces import interpolated, loft, mirrored, oval, sheet, spread

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

EYE_RADIUS = 0.013
LID_RADIUS = 0.0141
LID_COLUMNS = 24
LID_ROWS = (0.0, 0.1, 0.22, 0.38, 0.56, 0.76, 1.0)
# How far past a lid's edge its rim reaches where it meets the eyeball, as a share of the lid's height, so that the
# rim faces out of the eye instead of lying edge-on to it.
LID_UNDERCUT = 0.03
CORNER = math.radians(62.0)
UPPER_ARCH = math.radians(27.0)
LOWER_ARCH = math.radians(16.0)
CORNER_SLANT = math.radians(3.0)
LID_TOP = math.radians(88.0)
SQUINT = math.radians(9.0)
PUPIL = (14, 11, 11)
SCLERA = (238, 234, 228)
LASH = (46, 30, 24)
# Angle from the line of sight where the pupil, the iris, its dark rim and the white of the eye each end.
EYE_BANDS = ((0.0, 10.0, 3, PUPIL), (10.0, 26.0, 4, None), (26.0, 29.0, 2, LASH), (29.0, 110.0, 9, SCLERA))
# Distance from the midline, then the brow's level there and its half height.
BROW = (
    (0.010, (BROW_LEVEL - 0.002, 0.0018)),
    (0.016, (BROW_LEVEL - 0.0005, 0.0042)),
    (0.030, (BROW_LEVEL + 0.003, 0.0044)),
    (0.044, (BROW_LEVEL + 0.0035, 0.0034)),
    (0.058, (BROW_LEVEL - 0.001, 0.0013)),
)
BROW_RINGS = 14


@dataclass(frozen=True)
class Lid:
    """A lid: the elevation where it leaves the socket, how dark its lash line is and how far up the lid it reaches."""

    layer: Layer
    root: float
    lash: float
    lash_reach: float


LIDS = (Lid(Layer.UPPER_LID, LID_TOP, 1.0, 0.22), Lid(Layer.LOWER_LID, -LID_TOP, 0.35, 0.1))


def eyes(face: Face) -> list[Part]:
    """Both eyes with their lids and brows; the right side mirrors the left."""
    centre = BONE_POSITION["leftEye"]
    lid_colour = mixed(face.skin, face.blush, 0.14)
    parts: list[Part] = []
    for start, end, rings, colour in EYE_BANDS:
        band = _eye_band(centre, math.radians(start), math.radians(end), rings)
        parts += _both(band, (colour or face.iris,) * len(band.positions), Material.EYE, ("leftEye", "rightEye"))
    for lid in LIDS:
        shape, travel, shade = _lid(centre, lid)
        colours = tuple(mixed(lid_colour, LASH, share) for share in shade)
        parts += _both(shape, colours, Material.EYELID, ("head", "head"), lid.layer, travel)
    brow = _brow(face)
    return [*parts, *_both(brow, (face.brow,) * len(brow.positions), Material.BROW, ("head", "head"), Layer.FACE)]


def _both(
    shape: Shape,
    colours: tuple[Colour, ...],
    material: Material,
    bones: tuple[str, str],
    layer: Layer = Layer.BODY,
    travel: Sequence[Vector] = (),
) -> list[Part]:
    """The left part on its bone and its mirror image on the right one, with its lid travel mirrored too."""
    sides = (
        (shape, bones[0], tuple(travel)),
        (mirrored(shape), bones[1], tuple((-x, y, z) for x, y, z in travel)),
    )
    return [
        Part(side, colours, material, (((bone, 1.0),),) * len(colours), layer, moved) for side, bone, moved in sides
    ]


def _eye_band(centre: Vector, start: float, end: float, rings: int) -> Shape:
    rows = [_ball_ring(centre, end + (start - end) * ring / rings, LID_COLUMNS) for ring in range(rings + 1)]
    return sheet(rows, _away_from(centre), wrap=True)


def _ball_ring(centre: Vector, angle: float, count: int) -> list[Vector]:
    ring: list[Vector] = []
    for turn in (2.0 * math.pi * step / count for step in range(count)):
        towards = (math.sin(angle) * math.cos(turn), math.sin(angle) * math.sin(turn), math.cos(angle))
        ring.append(add(centre, scaled(towards, EYE_RADIUS)))
    return ring


def _away_from(centre: Vector) -> Callable[[Vector], Vector]:
    """The normals of the ball a surface lies on, so separate bands and lids shade as one ball."""

    def normal(position: Vector) -> Vector:
        return normalised(subtract(position, centre))

    return normal


def _lid_point(centre: Vector, azimuth: float, elevation: float, radius: float) -> Vector:
    level = math.cos(elevation)
    return add(centre, scaled((math.sin(azimuth) * level, math.sin(elevation), math.cos(azimuth) * level), radius))


def _lid_edges(azimuth: float) -> tuple[float, float]:
    """The elevation of the upper and the lower lid's edge: an almond between the corners, one seam beyond them."""
    seam = CORNER_SLANT * azimuth / CORNER
    reach = min(1.0, abs(azimuth) / CORNER)
    opening = (1.0 - reach * reach) ** 0.75
    return seam + UPPER_ARCH * opening, seam - LOWER_ARCH * opening


def _azimuths() -> list[float]:
    return [math.radians(-90.0 + 180.0 * column / LID_COLUMNS) for column in range(LID_COLUMNS + 1)]


def _lid(centre: Vector, lid: Lid) -> tuple[Shape, list[Vector], list[float]]:
    """A lid, how far each of its vertices travels to close or narrow the eye, and how dark its lash line is there."""
    courses = [(-LID_UNDERCUT, EYE_RADIUS - 0.0004), *((row, LID_RADIUS) for row in LID_ROWS)]
    rows: list[list[Vector]] = []
    travel: list[Vector] = []
    shade: list[float] = []
    for share, radius in courses if lid.layer is Layer.UPPER_LID else reversed(courses):
        row: list[Vector] = []
        for azimuth in _azimuths():
            rest, moved = _lid_way(azimuth, lid)
            opened = _lid_point(centre, azimuth, rest + (lid.root - rest) * share, radius)
            closed = _lid_point(centre, azimuth, moved + (lid.root - moved) * share, radius)
            row.append(opened)
            travel.append(subtract(closed, opened))
            shade.append(lid.lash * max(0.0, 1.0 - share / lid.lash_reach))
        rows.append(row)
    return sheet(rows, _away_from(centre)), travel, shade


def _lid_way(azimuth: float, lid: Lid) -> tuple[float, float]:
    """The elevation of a lid's edge at rest and where it moves to: the upper lid closes, the lower one squints."""
    upper, lower = _lid_edges(azimuth)
    return (upper, lower) if lid.layer is Layer.UPPER_LID else (lower, lower + SQUINT)


def _brow(face: Face) -> Shape:
    """The left brow: a tapering ridge of hair from beside the nose over the eye to the temple."""
    rings: list[Sequence[Vector]] = []
    for x in spread(BROW[0][0], BROW[-1][0], BROW_RINGS):
        level, half_height = interpolated(BROW, x)
        centre = add(face_point(face, x, level), (0.0, 0.0, 0.0006))
        radii = (0.0022 * face.brow_weight, half_height * face.brow_weight)
        rings.append(oval(centre, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), radii, 8))
    return loft(rings)
