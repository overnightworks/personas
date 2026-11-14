"""Mozart's powdered wig: hair brushed back off the forehead, two rolls over each ear and a queue tied with a bow."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from figures.figure_body import ACROSS
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Colour, Material, Part, mixed, part
from figures.figure_shapes import add, ellipsoid, scaled, subtract
from figures.figure_skeleton import EYE_HEIGHT
from figures.figure_surfaces import arc, interpolated, mirrored, rippled, shell, spread, sweep

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_shapes import Vector

POWDER = (238, 236, 232)
SHADE = (186, 186, 194)
RIBBON = (30, 28, 34)
CENTRE: Vector = (0.0, EYE_HEIGHT + 0.012, 0.0)
SQUARENESS = 2.3
# Pitch over the head, from straight ahead through straight up and the back down to the nape, then how far the hair
# stands off the centre there and its half width; its edge stands off the forehead, it swells behind it and gathers
# towards the queue.
PROFILE = (
    (math.radians(25.0), (0.100, 0.080)),
    (math.radians(40.0), (0.109, 0.086)),
    (math.radians(60.0), (0.120, 0.091)),
    (math.radians(90.0), (0.124, 0.092)),
    (math.radians(135.0), (0.126, 0.092)),
    (math.radians(180.0), (0.118, 0.090)),
    (math.radians(210.0), (0.126, 0.088)),
    (math.radians(228.0), (0.144, 0.080)),
    (math.radians(244.0), (0.162, 0.062)),
)
# Angle across the head from its middle, then the pitch at which the hair ends there: at the nape in the middle, behind
# the ear at the sides.
REAR = (
    (math.radians(0.0), (math.radians(244.0),)),
    (math.radians(30.0), (math.radians(242.0),)),
    (math.radians(55.0), (math.radians(232.0),)),
    (math.radians(80.0), (math.radians(214.0),)),
)
# The pitch over which the hair sinks to its lining before it ends.
CLOSING = math.radians(36.0)
ROWS = 32
COLUMNS = 34
REACH = math.radians(80.0)
STRANDS = 15.0
STRAND_DEPTH = 0.012
LINING = 0.8
# Height above the eye line and radius of each roll over the left ear, the upper one first.
ROLLS = ((0.008, 0.017), (-0.029, 0.0165))
ROLL_OFFSET = 0.091
# Where a roll's middle lies behind the eye line, half its length, and the share of its radius that is its hollow.
ROLL_MIDDLE = -0.014
ROLL_HALF_LENGTH = 0.037
ROLL_HOLLOW = 0.45
# How far round its ends a roll is drawn before its flat end face closes it.
ROLL_END = math.radians(80.0)
# The queue from under the gathered hair at the nape, over the collar and down the back, with its radii across and
# in depth; the bow ties it where it leaves the wig.
QUEUE = (
    ((0.0, 1.53, -0.085), (0.026, 0.010)),
    ((0.0, 1.475, -0.082), (0.018, 0.014)),
    ((0.0, 1.44, -0.097), (0.014, 0.013)),
    ((0.0, 1.40, -0.114), (0.015, 0.014)),
    ((0.0, 1.33, -0.125), (0.016, 0.015)),
    ((0.0, 1.27, -0.126), (0.013, 0.012)),
    ((0.0, 1.245, -0.123), (0.004, 0.004)),
)
BOW: Vector = (0.0, 1.478, -0.099)
# Each ribbon end from the knot down the left of the back, lying on the coat.
RIBBON_END = ((0.004, 1.472, -0.103), (0.020, 1.44, -0.087), (0.030, 1.395, -0.106))


class _Place(NamedTuple):
    """A point of the cap: its pitch, its angle across the head and the pitch at which the hair ends there."""

    pitch: float
    across: float
    end: float


def mozart_wig() -> list[Part]:
    places = _cap_places()
    lining = _cap_rows(places, LINING, 0.0)
    cap = shell(_sunk(places, _cap_rows(places, 1.0, STRAND_DEPTH), lining), lining)
    queue = sweep([place for place, _ in QUEUE], [radii for _, radii in QUEUE], ACROSS, 10)
    return [
        part(cap, _powder, Material.HAIR, NECK_BINDER),
        *_rolls(),
        part(queue, _powder, Material.HAIR, NECK_BINDER),
        *_bow(),
    ]


def _powder(position: Vector) -> Colour:
    """White powder, greyer in the furrows between strands."""
    return mixed(POWDER, SHADE, 0.3 - 0.3 * math.cos(2.0 * math.pi * position[0] / 0.011))


def _cap_places() -> list[list[_Place]]:
    """The cap's grid: rows from the hairline back, each column running to the pitch where the hair ends there."""
    rows: list[list[_Place]] = []
    for share in spread(0.0, 1.0, ROWS):
        row: list[_Place] = []
        for across in spread(-REACH, REACH, COLUMNS):
            (end,) = interpolated(REAR, abs(across))
            row.append(_Place(PROFILE[0][0] + (end - PROFILE[0][0]) * share, across, end))
        rows.append(row)
    return rows


def _cap_rows(places: Sequence[Sequence[_Place]], scale: float, strand_depth: float) -> list[list[Vector]]:
    """The cap from ear to ear, from the hairline back to where the hair ends, in strands `strand_depth` deep."""
    rows: list[list[Vector]] = []
    for row_places in places:
        row: list[Vector] = []
        for pitch, across, _ in row_places:
            radius, half_width = interpolated(PROFILE, pitch)
            over = (0.0, math.sin(pitch), math.cos(pitch))
            radii = (half_width * scale, radius * scale)
            row.append(arc(CENTRE, ACROSS, over, radii, [across], SQUARENESS)[0])
        rows.append(rippled(row, CENTRE, STRANDS, strand_depth))
    return rows


def _sunk(
    places: Sequence[Sequence[_Place]], surface: Sequence[Sequence[Vector]], lining: Sequence[Sequence[Vector]]
) -> list[list[Vector]]:
    """The surface drawn down to the lining as the hair ends, so the cap closes in a hairline and has no end face."""
    return [
        [
            add(inside, scaled(subtract(outside, inside), _quarter_circle((end - pitch) / CLOSING)))
            for (pitch, _, end), outside, inside in zip(row_places, outer_row, inner_row, strict=True)
        ]
        for row_places, outer_row, inner_row in zip(places, surface, lining, strict=True)
    ]


def _quarter_circle(share: float) -> float:
    """0 to 1 over a quarter circle: it stays near 1 until close to 0, then falls away steeply."""
    clamped = min(1.0, max(0.0, share))
    return math.sqrt(1.0 - (1.0 - clamped) ** 2)


def _rolls() -> list[Part]:
    return [roll for above, radius in ROLLS for roll in _roll(EYE_HEIGHT + above, radius)]


def _roll(height: float, radius: float) -> list[Part]:
    """A curl over each ear, rolled around a line from behind the ear to in front of it, with rounded, hollow ends."""

    def paint(position: Vector) -> Colour:
        off_axis = math.hypot(abs(position[0]) - ROLL_OFFSET, position[1] - height)
        return SHADE if off_axis < ROLL_HOLLOW * radius else POWDER

    stations = [math.sin(turn) for turn in spread(-ROLL_END, ROLL_END, 12)]
    path = [(ROLL_OFFSET, height, ROLL_MIDDLE + ROLL_HALF_LENGTH * station) for station in stations]
    radii = [(radius * math.sqrt(1.0 - station**4),) * 2 for station in stations]
    left = sweep(path, radii, ACROSS, 14)
    return [part(shape, paint, Material.HAIR, "head") for shape in (left, mirrored(left))]


def _bow() -> list[Part]:
    x, height, depth = BOW
    shapes = [
        ellipsoid((x + 0.024, height, depth), (0.023, 0.014, 0.008), 6, 10),
        ellipsoid((x - 0.024, height, depth), (0.023, 0.014, 0.008), 6, 10),
        ellipsoid((x, height, depth - 0.003), (0.010, 0.012, 0.010), 5, 8),
    ]
    left = sweep(RIBBON_END, [(0.008, 0.003), (0.010, 0.003), (0.012, 0.003)], ACROSS, 8)
    shapes += [left, mirrored(left)]
    return [part(shape, RIBBON, Material.SATIN, NECK_BINDER) for shape in shapes]
