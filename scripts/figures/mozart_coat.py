"""Mozart's clothes: a braided coat open over a waistcoat, a lace jabot and cuffs, breeches, stockings, buckle shoes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import (
    ACROSS,
    AHEAD,
    DOWN,
    LEG_SPACING,
    TRUNK_POWER,
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
)
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Colour, Material, Part, part
from figures.figure_shapes import ellipsoid
from figures.figure_skeleton import SHOULDER_HEIGHT
from figures.figure_surfaces import arc, interpolated, loft, mirrored, oval, patch, rippled, shell, spread

if TYPE_CHECKING:
    from figures.figure_head import Face
    from figures.figure_shapes import Shape, Vector

CRIMSON = (168, 30, 42)
GILT = (214, 174, 84)
PALE_GOLD = (234, 212, 156)
LACE = (246, 244, 238)
STOCKING = (238, 236, 230)
BLACK = (26, 24, 26)
# Height, then the coat's half width, front depth and back depth there and how far from the middle its front edges part.
COAT = (
    (0.56, (0.200, 0.150, 0.170, math.radians(55.0))),
    (0.75, (0.186, 0.138, 0.152, math.radians(44.0))),
    (0.93, (0.166, 0.122, 0.128, math.radians(31.0))),
    (1.06, (0.134, 0.106, 0.106, math.radians(22.0))),
    (1.22, (0.150, 0.114, 0.112, math.radians(17.0))),
    (1.34, (0.160, 0.106, 0.109, math.radians(19.0))),
    (1.40, (0.166, 0.090, 0.100, math.radians(22.0))),
    (1.425, (0.160, 0.074, 0.088, math.radians(26.0))),
    (1.445, (0.112, 0.064, 0.080, math.radians(33.0))),
    (1.462, (0.066, 0.058, 0.076, math.radians(44.0))),
)
COAT_RINGS = 46
COAT_COLUMNS = 36
COAT_THICKNESS = 0.008
BRAID_WIDTH = 0.016
BRAID_RELIEF = 0.0015
BRAID_WRAP = 0.002
BRAID_TOP = 1.40
WAISTCOAT_GAP = 0.015
# The waistcoat lies inside the coat from its hem on the thighs up to the neck.
WAISTCOAT_HEIGHTS = (0.87, *(height for height, _ in COAT[2:]))
WAISTCOAT = tuple(
    (height, tuple(radius - WAISTCOAT_GAP for radius in interpolated(COAT, height)[:3])) for height in WAISTCOAT_HEIGHTS
)
# The seat of the breeches, which joins the legs under the waistcoat's hem; its front lies back between the thighs
# and it closes where they part.
SEAT = (
    (0.785, (0.026, 0.018, 0.040)),
    (0.81, (0.050, 0.036, 0.066)),
    (0.86, (0.110, 0.056, 0.092)),
    (0.94, (0.146, 0.100, 0.108)),
)
SLEEVE = (
    (0.125, (0.040, 0.040)),
    (0.155, (0.049, 0.048)),
    (0.20, (0.050, 0.049)),
    (0.445, (0.043, 0.042)),
    (0.60, (0.039, 0.038)),
)
CUFF = ((0.548, (0.050, 0.049)), (0.556, (0.057, 0.056)), (0.665, (0.060, 0.059)))
CUFF_MOUTH = ((0.668, (0.052, 0.051)), (0.63, (0.038, 0.037)))
# Distance from the midline, then the wrist lace's radius there and how deep its folds are.
RUFFLE = ((0.63, 0.032, 0.02), (0.672, 0.040, 0.06), (0.708, 0.056, 0.16), (0.704, 0.044, 0.12), (0.672, 0.028, 0.03))
BREECHES = (
    (0.44, (0.050, 0.052, 0.0)),
    (0.50, (0.056, 0.058, 0.004)),
    (0.70, (0.066, 0.076, 0.0)),
    (0.88, (0.070, 0.090, 0.0)),
    (0.97, (0.066, 0.088, 0.0)),
)
KNEE_BAND = 0.458
STOCKINGS = (
    (0.05, (0.034, 0.038, 0.0)),
    (0.10, (0.033, 0.038, -0.002)),
    (0.30, (0.048, 0.054, -0.012)),
    (0.42, (0.044, 0.046, -0.002)),
    (0.47, (0.047, 0.049, 0.002)),
)
# Each tier of the jabot from the lowest up: its lower edge, its upper edge, its half width and its depth.
JABOT = ((1.255, 1.33, 0.052, 0.034), (1.32, 1.39, 0.046, 0.032), (1.38, 1.446, 0.038, 0.028))
JABOT_FOLDS = 9
STOCK = ((1.425, (0.060, 0.052, 0.065)), (1.45, (0.062, 0.054, 0.067)), (1.484, (0.060, 0.053, 0.066)))


def mozart_clothes(face: Face) -> list[Part]:
    return [
        *_coat(),
        *_waistcoat(),
        *_arms(),
        *_jabot(),
        part(trunk(STOCK, 4), LACE, Material.CLOTH, NECK_BINDER),
        *_legs(),
        *hands(face.skin),
    ]


def _coat() -> list[Part]:
    body = shell(open_rows(COAT, COAT_RINGS, COAT_COLUMNS), open_rows(COAT, COAT_RINGS, COAT_COLUMNS, -COAT_THICKNESS))
    braid = patch(
        [
            [
                _coat_front(height, -BRAID_WRAP, -COAT_THICKNESS - BRAID_WRAP),
                _coat_front(height, -BRAID_WRAP),
                _coat_front(height, BRAID_WIDTH),
            ]
            for height in spread(COAT[0][0], BRAID_TOP, COAT_RINGS)
        ]
    )
    hem_top = COAT[0][0] + BRAID_WIDTH
    hem = patch(open_rows((COAT[0], (hem_top, interpolated(COAT, hem_top))), 1, COAT_COLUMNS, BRAID_RELIEF))
    left = [braid, *(_button(_coat_front(height, 0.036)) for height in spread(0.99, 1.31, 6))]
    trim = [hem, *left, *(mirrored(shape) for shape in left)]
    return [
        part(body, CRIMSON, Material.CLOTH, trunk_binder),
        *(part(shape, GILT, Material.GILT, trunk_binder) for shape in trim),
    ]


def _coat_front(height: float, inset: float, grown: float = BRAID_RELIEF) -> Vector:
    """A place on the coat's left front where its trim lies, `inset` back from the edge."""
    half_width, front, back, opening = interpolated(COAT, height)
    radii = (half_width + grown, front + grown, back + grown)
    return arc((0.0, height, 0.0), ACROSS, AHEAD, radii, [opening + inset / half_width], TRUNK_POWER)[0]


def _waistcoat() -> list[Part]:
    buttons = [
        part(_button((0.0, height, interpolated(WAISTCOAT, height)[1])), GILT, Material.GILT, trunk_binder)
        for height in spread(0.922, 1.21, 7)
    ]
    return [part(trunk(WAISTCOAT, 34), PALE_GOLD, Material.SATIN, trunk_binder), *buttons]


def _button(place: Vector) -> Shape:
    return ellipsoid(place, (0.0075, 0.0075, 0.0045), 5, 8)


def _arms() -> list[Part]:
    def cuff_paint(position: Vector) -> Colour:
        return GILT if abs(position[0]) < CUFF[1][0] + 0.014 else CRIMSON

    cuff_buttons = [
        part(_button((side * x, SHOULDER_HEIGHT + 0.058, 0.0)), GILT, Material.GILT, prefix + "LowerArm")
        for x in (0.585, 0.61, 0.635)
        for prefix, side in (("left", 1.0), ("right", -1.0))
    ]
    return [
        *both_sides(arm_tube(SLEEVE, 28), CRIMSON, Material.CLOTH, arm_binder),
        *both_sides(arm_tube(CUFF, 6, CUFF_MOUTH), cuff_paint, Material.CLOTH, arm_binder),
        *both_sides(_ruffle(), LACE, Material.CLOTH, arm_binder),
        *cuff_buttons,
    ]


def _ruffle() -> Shape:
    """The lace at the left wrist: a folded funnel that opens towards the hand and is hollow inside."""
    rings: list[list[Vector]] = []
    for x, radius, folds in RUFFLE:
        centre = (x, SHOULDER_HEIGHT, 0.0)
        rings.append(rippled(oval(centre, DOWN, AHEAD, (radius, radius), 40), centre, 10.0, folds))
    return loft(rings)


def _jabot() -> list[Part]:
    """Tiers of folded lace down the chest, each standing out furthest at its lower edge."""
    tiers: list[Part] = []
    for low, high, half_width, depth in JABOT:
        rings: list[list[Vector]] = []
        for height in spread(low, high, 5):
            share = 1.0 - 0.5 * (height - low) / (high - low)
            centre = (0.0, height, interpolated(WAISTCOAT, height)[1] - 0.012)
            ring = oval(centre, ACROSS, AHEAD, (half_width * share, depth * share), 54)
            rings.append(rippled(ring, centre, JABOT_FOLDS, 0.14 * share))
        tiers.append(part(loft(rings), LACE, Material.CLOTH, trunk_binder))
    return tiers


def _legs() -> list[Part]:
    def breeches_paint(position: Vector) -> Colour:
        return GILT if position[1] < KNEE_BAND else CRIMSON

    buckles = [
        part(
            ellipsoid((side * LEG_SPACING, 0.064, 0.042), (0.021, 0.005, 0.014), 5, 8),
            GILT,
            Material.GILT,
            prefix + "Foot",
        )
        for prefix, side in (("left", 1.0), ("right", -1.0))
    ]
    return [
        part(trunk(SEAT, 8), CRIMSON, Material.CLOTH, trunk_binder),
        *both_sides(leg_tube(BREECHES, 30), breeches_paint, Material.CLOTH, leg_binder),
        *both_sides(leg_tube(STOCKINGS, 22), STOCKING, Material.CLOTH, leg_binder),
        *shoes(BLACK),
        *buckles,
    ]
