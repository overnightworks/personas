"""Curie's own hair: dark blonde to light brown, a little grey at the temples, pinned up in a loose knot on the crown
with soft volume round the head."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_hair import hair_cap
from figures.figure_parts import Colour, Material, Part, mixed, part
from figures.figure_shapes import ellipsoid
from figures.figure_skeleton import EYE_HEIGHT

if TYPE_CHECKING:
    from figures.figure_shapes import Vector

BLONDE = (150, 124, 90)
GREY = (170, 162, 150)
# Level above the eye line, then the hair's half width, front depth and back depth there and how far from the middle
# its front edges part: swept back off the face, fuller at the sides and the back than the skull beneath.
HAIR = (
    (-0.082, (0.062, 0.066, 0.080, math.radians(88.0))),
    (-0.062, (0.070, 0.073, 0.094, math.radians(86.0))),
    (-0.045, (0.074, 0.078, 0.100, math.radians(84.0))),
    (-0.015, (0.080, 0.085, 0.107, math.radians(78.0))),
    (0.015, (0.083, 0.090, 0.110, math.radians(68.0))),
    (0.045, (0.082, 0.092, 0.109, math.radians(54.0))),
    (0.070, (0.076, 0.086, 0.102, math.radians(34.0))),
    (0.090, (0.066, 0.073, 0.090, math.radians(10.0))),
    (0.105, (0.052, 0.058, 0.073, math.radians(0.0))),
    (0.120, (0.032, 0.036, 0.045, math.radians(0.0))),
    (0.131, (0.004, 0.004, 0.005, math.radians(0.0))),
)
BODY = 0.016
TOUSLE = 0.003
TUFTS = 7.0
# Where the knot sits (level above the eye line, how far behind the middle of the head) and its radii; the smaller coil
# lies against it (how far to the side, then its level and how far behind the middle).
KNOT = (0.108, -0.040)
KNOT_RADII = (0.050, 0.040, 0.048)
COIL = (0.030, 0.092, -0.064)
COIL_RADII = (0.034, 0.030, 0.034)


def curie_hair() -> list[Part]:
    return [*hair_cap(HAIR, _strands, BODY, TOUSLE, TUFTS), *_knot()]


def _knot() -> list[Part]:
    """The loose knot: one rounded coil on the crown with a smaller one tucked against it."""
    level, behind = KNOT
    side, low, back = COIL
    coils = (
        ellipsoid((0.0, EYE_HEIGHT + level, behind), KNOT_RADII, 8, 14),
        ellipsoid((side, EYE_HEIGHT + low, back), COIL_RADII, 7, 12),
    )
    return [part(coil, _strands, Material.HAIR, "head") for coil in coils]


def _strands(position: Vector) -> Colour:
    """Dark blonde hair with greyer strands that wander as they lie."""
    x, height, z = position
    return mixed(BLONDE, GREY, 0.12 + 0.12 * math.cos(26.0 * math.atan2(x, z) + 11.0 * height))
