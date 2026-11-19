"""Humboldt's clothes: a dark dress coat cut away at the front with a high standing collar, a white cravat, a
waistcoat, long trousers and plain shoes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import (
    SEAT,
    arm_binder,
    arm_tube,
    both_sides,
    hands,
    leg_binder,
    leg_tube,
    open_rows,
    shoes,
    trunk,
    trunk_binder,
    waistcoat,
)
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Material, Part, part
from figures.figure_shapes import ellipsoid
from figures.figure_surfaces import interpolated, shell

if TYPE_CHECKING:
    from figures.figure_head import Face
    from figures.figure_shapes import Shape

NAVY = (34, 38, 56)
CREAM = (226, 214, 186)
FAWN = (172, 160, 140)
LINEN = (240, 238, 230)
BRASS = (190, 156, 84)
BLACK = (28, 26, 28)
# Height, then the coat's half width, front depth and back depth there and how far from the middle its front edges
# part: below the waist the fronts are cut away to the sides and the tails hang behind, above it they meet.
COAT = (
    (0.52, (0.180, 0.120, 0.150, math.radians(118.0))),
    (0.70, (0.176, 0.116, 0.146, math.radians(106.0))),
    (0.90, (0.166, 0.112, 0.130, math.radians(86.0))),
    (1.03, (0.140, 0.104, 0.108, math.radians(64.0))),
    (1.08, (0.134, 0.102, 0.104, math.radians(34.0))),
    (1.22, (0.150, 0.112, 0.112, math.radians(15.0))),
    (1.34, (0.160, 0.106, 0.109, math.radians(17.0))),
    (1.40, (0.166, 0.090, 0.100, math.radians(22.0))),
    (1.425, (0.160, 0.074, 0.088, math.radians(26.0))),
    (1.445, (0.112, 0.064, 0.080, math.radians(33.0))),
    (1.462, (0.066, 0.058, 0.076, math.radians(44.0))),
)
COAT_RINGS = 46
COAT_COLUMNS = 36
COAT_THICKNESS = 0.008
# The collar stands round the neck from the coat's neckline up under the chin, open at the front.
COLLAR = (
    (1.44, (0.070, 0.060, 0.074, math.radians(38.0))),
    (1.47, (0.067, 0.057, 0.071, math.radians(34.0))),
    (1.505, (0.064, 0.055, 0.068, math.radians(30.0))),
)
COLLAR_RINGS = 6
COLLAR_COLUMNS = 24
COLLAR_THICKNESS = 0.006
WAISTCOAT_GAP = 0.015
WAISTCOAT_HEIGHTS = (0.84, 0.90, 0.96, *(height for height, _ in COAT[3:]))
WAISTCOAT = tuple(
    (height, tuple(radius - WAISTCOAT_GAP for radius in interpolated(COAT, height)[:3])) for height in WAISTCOAT_HEIGHTS
)
SLEEVE = (
    (0.125, (0.040, 0.040)),
    (0.155, (0.049, 0.048)),
    (0.20, (0.050, 0.049)),
    (0.445, (0.043, 0.042)),
    (0.60, (0.039, 0.038)),
    (0.665, (0.037, 0.036)),
)
# Height, then the trousers' half width, depth and how far forward they stand there; they narrow into the waistcoat.
TROUSERS = (
    (0.06, (0.044, 0.048, 0.0)),
    (0.20, (0.047, 0.052, -0.004)),
    (0.45, (0.055, 0.060, 0.0)),
    (0.70, (0.066, 0.076, 0.0)),
    (0.82, (0.060, 0.076, 0.0)),
    (0.90, (0.050, 0.068, 0.0)),
    (0.97, (0.040, 0.060, 0.0)),
)
# Height, then the cravat's half width, front depth and back depth there, with its knot at the front.
CRAVAT = ((1.425, (0.060, 0.052, 0.065)), (1.46, (0.062, 0.055, 0.067)), (1.50, (0.059, 0.053, 0.065)))
KNOT = (0.0, 1.462, 0.058)


def humboldt_clothes(face: Face) -> list[Part]:
    return [
        *_coat(),
        *_waistcoat(),
        *both_sides(arm_tube(SLEEVE, 28), NAVY, Material.CLOTH, arm_binder),
        part(trunk(CRAVAT, 4), LINEN, Material.CLOTH, NECK_BINDER),
        part(_knot(), LINEN, Material.CLOTH, NECK_BINDER),
        part(trunk(SEAT, 8), FAWN, Material.CLOTH, trunk_binder),
        *both_sides(leg_tube(TROUSERS, 30), FAWN, Material.CLOTH, leg_binder),
        *shoes(BLACK),
        *hands(face.skin),
    ]


def _waistcoat() -> list[Part]:
    return waistcoat(WAISTCOAT, CREAM, BRASS, (0.97, 1.21, 6))


def _coat() -> list[Part]:
    body = shell(open_rows(COAT, COAT_RINGS, COAT_COLUMNS), open_rows(COAT, COAT_RINGS, COAT_COLUMNS, -COAT_THICKNESS))
    collar = shell(
        open_rows(COLLAR, COLLAR_RINGS, COLLAR_COLUMNS),
        open_rows(COLLAR, COLLAR_RINGS, COLLAR_COLUMNS, -COLLAR_THICKNESS),
    )
    return [part(body, NAVY, Material.CLOTH, trunk_binder), part(collar, NAVY, Material.CLOTH, NECK_BINDER)]


def _knot() -> Shape:
    """The cravat's knot at the front of the neck."""
    return ellipsoid(KNOT, (0.016, 0.014, 0.010), 6, 10)
