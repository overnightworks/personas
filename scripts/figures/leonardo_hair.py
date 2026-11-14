"""Leonardo's hair: long grey locks that fall to the shoulders, a flowing beard and a moustache that runs into it."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import ACROSS, AHEAD, open_point
from figures.figure_head import NECK_BINDER, face_point, level_height, skin_point
from figures.figure_parts import Colour, Layer, Material, Part, mixed, part
from figures.figure_shapes import add, scaled, subtract
from figures.figure_surfaces import arc, interpolated, patch, shell, smoothstep, spread

if TYPE_CHECKING:
    from figures.figure_head import Face
    from figures.figure_shapes import Vector

SILVER = (230, 226, 216)
ASH = (152, 148, 144)
# Height, then the hair's half width, front depth and back depth there and how far from the middle it parts: it leaves
# the face free and falls on down the back; HAIR_EDGE gives where it ends.
HAIR = (
    (1.335, (0.126, 0.050, 0.124, math.radians(84.0))),
    (1.40, (0.118, 0.060, 0.120, math.radians(80.0))),
    (1.445, (0.106, 0.080, 0.116, math.radians(70.0))),
    (1.50, (0.100, 0.080, 0.113, math.radians(66.0))),
    (1.56, (0.097, 0.084, 0.113, math.radians(67.0))),
    (1.615, (0.093, 0.088, 0.114, math.radians(63.0))),
    (1.66, (0.084, 0.088, 0.108, math.radians(50.0))),
    (1.69, (0.074, 0.080, 0.098, math.radians(34.0))),
    (1.715, (0.060, 0.064, 0.082, math.radians(20.0))),
)
HAIR_ROWS = 36
HAIR_COLUMNS = 96
HAIR_THICKNESS = 0.016
# Share of the way round from the edge beside the face to the middle of the back, then the height the hair reaches
# there: above the shoulder beside the neck, down the back behind it.
HAIR_EDGE = (
    (0.0, (1.490,)),
    (0.08, (1.480,)),
    (0.40, (1.470,)),
    (0.50, (1.462,)),
    (0.58, (1.440,)),
    (0.66, (1.395,)),
    (0.74, (1.345,)),
    (0.82, (1.335,)),
    (1.0, (1.335,)),
)
# The height over which a lock thins to its tip, and how far a furrow between two locks ends above their tips, at most
# twice this.
HAIR_END = 0.05
HAIR_TIPS = 0.008
LOCKS = 13.0
# Height, then the hanging beard's half width, how far forward its middle lies, how far it stands out from there and
# how far round it closes: on the chest it hangs as one closed lock, at the jaw it opens to the reach of the cheeks.
BEARD = (
    (1.262, (0.002, 0.118, 0.003, math.radians(176.0))),
    (1.285, (0.026, 0.104, 0.028, math.radians(176.0))),
    (1.33, (0.044, 0.088, 0.044, math.radians(172.0))),
    (1.39, (0.058, 0.066, 0.056, math.radians(150.0))),
    (1.45, (0.066, 0.040, 0.066, math.radians(100.0))),
    (1.50, (0.070, 0.022, 0.078, math.radians(100.0))),
    (1.53, (0.072, 0.010, 0.088, math.radians(100.0))),
)
# Angle around the head from straight ahead, then the level below which the beard grows there: under the lower lip,
# past the corners of the mouth and up the cheeks to the ears.
BEARD_LINE = (
    (math.radians(0.0), (-0.083,)),
    (math.radians(14.0), (-0.082,)),
    (math.radians(22.0), (-0.074,)),
    (math.radians(29.0), (-0.063,)),
    (math.radians(40.0), (-0.055,)),
    (math.radians(55.0), (-0.046,)),
    (math.radians(72.0), (-0.036,)),
    (math.radians(100.0), (-0.024,)),
)
BEARD_REACH = math.radians(100.0)
BEARD_COLUMNS = 56
BEARD_HANGING_ROWS = 14
BEARD_GROWN_ROWS = 22
# The level where the beard leaves the jaw, and the level below it where it hangs free of the face.
BEARD_JAW = -0.092
BEARD_SPLIT = -0.125
# The beard's edge lies this far under the skin; within a few millimetres it stands off it, then thickens downward.
BEARD_TUCK = 0.0015
BEARD_EDGE = 0.0055
BEARD_BODY = 0.010
BEARD_LOCKS = 16.0
BEARD_LOCK_DEPTH = 0.10
# Distance from the midline, then the levels of the moustache's lower and upper edge there and its thickness.
MOUSTACHE = (
    (0.0, (-0.0615, -0.0545, 0.0040)),
    (0.010, (-0.0645, -0.0540, 0.0050)),
    (0.022, (-0.0680, -0.0560, 0.0050)),
    (0.030, (-0.0760, -0.0600, 0.0045)),
    (0.036, (-0.0840, -0.0680, 0.0030)),
    (0.040, (-0.0880, -0.0780, 0.0012)),
)
MOUSTACHE_COLUMNS = 32
MOUSTACHE_ROWS = 6
MOUSTACHE_TUCK = 0.001


def leonardo_hair(face: Face) -> list[Part]:
    return [_locks(), _beard(face), _moustache(face)]


def _strands(position: Vector) -> Colour:
    """Silver hair with darker strands that wander as they fall."""
    x, height, z = position
    wander = 2.0 * math.sin(30.0 * height)
    return mixed(SILVER, ASH, 0.3 + 0.25 * math.cos(16.0 * math.atan2(x, z) + wander))


def _locks() -> Part:
    """The hair as one sheet with thickness: each column falls from the crown to its own edge, waved into locks that
    deepen towards their ends and thin to a tip there."""
    top = HAIR[-1][0]
    outer: list[list[Vector]] = [[] for _ in range(HAIR_ROWS + 1)]
    inner: list[list[Vector]] = [[] for _ in range(HAIR_ROWS + 1)]
    for column in range(HAIR_COLUMNS + 1):
        edge = _lock_edge(column)
        for row, height in enumerate(spread(edge, top, HAIR_ROWS)):
            lining, surface = _lock_point(column, height, edge)
            inner[row].append(lining)
            outer[row].append(surface)
    return part(shell(outer, inner), _strands, Material.HAIR, NECK_BINDER)


def _lock_turn(column: int, height: float) -> float:
    """The phase of the lock wave at a column and a height; the wave turns as it falls."""
    return 2.0 * math.pi * LOCKS * column / (HAIR_COLUMNS + 1) + 2.4 * math.sin(34.0 * height)


def _lock_edge(column: int) -> float:
    """The height where the lock at a column ends: its keyed height, lifted by up to twice HAIR_TIPS with its wave."""
    across = column / HAIR_COLUMNS
    (keyed,) = interpolated(HAIR_EDGE, 1.0 - abs(2.0 * across - 1.0))
    return keyed + HAIR_TIPS * (1.0 - math.cos(_lock_turn(column, keyed)))


def _lock_point(column: int, height: float, edge: float) -> tuple[Vector, Vector]:
    """The lining and the waved surface of the hair at a column and a height; towards its edge the surface sinks to the
    lining, so the sheet ends in a tip without an end face."""
    across = column / HAIR_COLUMNS
    lining = open_point(HAIR, height, across, -HAIR_THICKNESS)
    x, _, z = open_point(HAIR, height, across)
    depth = 0.015 + 0.025 * smoothstep(1.60, 1.40, height)
    wave = 1.0 + depth * math.cos(_lock_turn(column, height))
    waved = (x * wave, height, z * wave)
    return lining, add(lining, scaled(subtract(waved, lining), smoothstep(edge, edge + HAIR_END, height)))


def _beard(face: Face) -> Part:
    """One sheet from the beard's tip on the chest up to its edge on the face; it drops with the jaw."""
    angles = spread(-BEARD_REACH, BEARD_REACH, BEARD_COLUMNS)
    tip = (BEARD[0][0] - level_height(face, 0.0)) / face.length
    hanging = [
        [_beard_point(face, level, angle) for angle in angles] for level in spread(tip, BEARD_SPLIT, BEARD_HANGING_ROWS)
    ]
    grown = [
        [_beard_point(face, BEARD_SPLIT + (_beard_line(angle) - BEARD_SPLIT) * share, angle) for angle in angles]
        for share in spread(0.0, 1.0, BEARD_GROWN_ROWS)[1:]
    ]
    return part(patch([*hanging, *grown]), _strands, Material.WHISKERS, NECK_BINDER, Layer.BEARD)


def _beard_line(angle: float) -> float:
    return interpolated(BEARD_LINE, abs(angle))[0]


def _beard_point(face: Face, level: float, angle: float) -> Vector:
    """The beard at a level of the face and an angle around the head: down to the jaw it grows from the skin, below it
    hangs, and away from its edge it falls in locks."""
    height = level_height(face, level)
    half_width, middle, depth, closing = interpolated(BEARD, height)
    hanging = arc((0.0, height, middle), ACROSS, AHEAD, (half_width, depth), [angle * closing / BEARD_REACH])[0]
    on_face = smoothstep(BEARD_SPLIT, BEARD_JAW, level)
    x, _, z = add(scaled(hanging, 1.0 - on_face), scaled(_grown(face, level, angle), on_face))
    locks = math.cos(BEARD_LOCKS * angle + 3.0 * math.sin(26.0 * height))
    wave = 1.0 + BEARD_LOCK_DEPTH * smoothstep(0.0, 0.03, _beard_line(angle) - level) * locks
    return (x * wave, height, middle + (z - middle) * wave)


def _grown(face: Face, level: float, angle: float) -> Vector:
    """The beard on the face: it follows the skin down to the jaw and keeps the jaw's outline below it."""
    x, height, z = skin_point(face, max(level, BEARD_JAW), angle)
    below = _beard_line(angle) - level
    out = BEARD_EDGE * min(1.0, below / 0.003) + BEARD_BODY * smoothstep(0.0, 0.04, below) - BEARD_TUCK
    return (x + out * math.sin(angle), height, z + out * math.cos(angle))


def _moustache(face: Face) -> Part:
    """A moustache draped over the upper lip, from under the nose past the corners of the mouth into the beard."""
    reach = MOUSTACHE[-1][0]
    rows = [
        [_moustache_point(face, x, share) for x in spread(-reach, reach, MOUSTACHE_COLUMNS)]
        for share in spread(0.0, 1.0, MOUSTACHE_ROWS)
    ]
    return part(patch(rows), _strands, Material.WHISKERS, "head", Layer.FACE)


def _moustache_point(face: Face, x: float, share: float) -> Vector:
    """The moustache a share of the way up from its lower edge: thickest in between, under the skin at both edges."""
    low, high, thickness = interpolated(MOUSTACHE, abs(x))
    swell = 4.0 * thickness * share * (1.0 - share) - MOUSTACHE_TUCK
    return add(face_point(face, x, low + (high - low) * share), (0.0, 0.0, swell))
