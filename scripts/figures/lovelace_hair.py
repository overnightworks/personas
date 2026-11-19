"""Lovelace's hair: dark brown, parted in the middle and smooth over the crown, a cluster of ringlets at each side
falling to the jaw and a knot at the back of the head."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_hair import hair_cap, knot, ringlet, strands
from figures.figure_parts import Colour, Material, Part, mixed, part
from figures.figure_skeleton import EYE_HEIGHT
from figures.figure_surfaces import mirrored, smoothstep

if TYPE_CHECKING:
    from figures.figure_shapes import Vector

BROWN = (50, 34, 28)
HIGHLIGHT = (84, 58, 44)
SCALP = (112, 84, 70)
# Level above the eye line, then the hair's half width, front depth and back depth there and how far from the middle
# its front edges part: close to the skull, covering the ears and swept down from the middle parting.
HAIR = (
    (-0.082, (0.062, 0.066, 0.080, math.radians(88.0))),
    (-0.062, (0.070, 0.073, 0.094, math.radians(86.0))),
    (-0.045, (0.074, 0.078, 0.100, math.radians(84.0))),
    (-0.015, (0.080, 0.085, 0.107, math.radians(76.0))),
    (0.015, (0.083, 0.090, 0.110, math.radians(66.0))),
    (0.045, (0.082, 0.092, 0.109, math.radians(52.0))),
    (0.070, (0.076, 0.086, 0.102, math.radians(32.0))),
    (0.090, (0.066, 0.073, 0.090, math.radians(10.0))),
    (0.105, (0.052, 0.058, 0.073, math.radians(0.0))),
    (0.120, (0.032, 0.036, 0.045, math.radians(0.0))),
    (0.131, (0.004, 0.004, 0.005, math.radians(0.0))),
)
BODY = 0.010
# The parting runs this wide along the middle of the crown.
PARTING = 0.007
STRANDS = strands(BROWN, HIGHLIGHT, 0.2, 0.15, 30.0, 6.0)
# Where the knot sits (level above the eye line, how far behind the middle of the head) and its radii; the smaller coil
# lies against it (how far to the side, then its level and how far behind the middle).
KNOT = (-0.005, -0.103)
KNOT_RADII = (0.050, 0.038, 0.040)
COIL = (0.0, -0.045, -0.092)
COIL_RADII = (0.036, 0.030, 0.034)
# Each ringlet of the left cluster: how far to the side, how far ahead of the middle of the head, level it hangs from
# above the eye line and level of its tip, how far its tip lies to the side, and its turns.
RINGLETS = (
    (0.090, 0.040, 0.000, -0.088, 0.072, 3.5),
    (0.092, 0.020, 0.004, -0.100, 0.070, 4.0),
    (0.090, 0.000, 0.000, -0.094, 0.074, 3.5),
)
CURL = 0.009
THICKNESS = 0.0075


def lovelace_hair() -> list[Part]:
    return [*hair_cap(HAIR, _parted, BODY, 0.0, 0.0), *_ringlets(), *knot(KNOT, KNOT_RADII, COIL, COIL_RADII, STRANDS)]


def _parted(position: Vector) -> Colour:
    """Brown strands with the scalp showing along the middle of the crown's front."""
    x, height, z = position
    ahead = smoothstep(-0.02, 0.03, z)
    along = smoothstep(EYE_HEIGHT + 0.02, EYE_HEIGHT + 0.07, height) * (
        1.0 - smoothstep(EYE_HEIGHT + 0.12, EYE_HEIGHT + 0.131, height)
    )
    return mixed(STRANDS(position), SCALP, ahead * along * (1.0 - smoothstep(0.0, PARTING, abs(x))))


def _ringlets() -> list[Part]:
    shapes = [
        ringlet(
            (side, EYE_HEIGHT + top, ahead),
            (tip_side, EYE_HEIGHT + low, ahead),
            turns,
            CURL,
            THICKNESS,
        )
        for side, ahead, top, low, tip_side, turns in RINGLETS
    ]
    return [part(shape, STRANDS, Material.HAIR, "head") for left in shapes for shape in (left, mirrored(left))]
