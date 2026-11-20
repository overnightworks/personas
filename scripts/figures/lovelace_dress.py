"""Lovelace's dress: an evening gown of about 1840 in plum, a fitted bodice that comes to a point at the waist, a wide
neckline that leaves the neck, the upper chest and the shoulders bare, short sleeves below the shoulder, a full bell
skirt to the floor and a narrow light trim along the neckline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import arm_binder, arm_tube, both_sides, hands, open_point, shoes, trunk, trunk_binder
from figures.figure_gown import Pleats, gown
from figures.figure_parts import Material, Part, part
from figures.figure_surfaces import loft, smoothstep, spread

if TYPE_CHECKING:
    from collections.abc import Callable

    from figures.figure_head import Face
    from figures.figure_shapes import Shape, Vector

PLUM = (84, 32, 62)
TRIM = (232, 220, 206)
LEATHER = (40, 30, 32)
# Height, then the dress's half width, front depth and back depth there: a bell skirt from the hem to a small waist, a
# bodice that fits the trunk and the shoulders and neck it leaves bare above.
DRESS = (
    (0.02, (0.520, 0.500, 0.520)),
    (0.30, (0.460, 0.440, 0.460)),
    (0.60, (0.340, 0.320, 0.340)),
    (0.85, (0.215, 0.185, 0.205)),
    (1.00, (0.135, 0.105, 0.115)),
    (1.10, (0.120, 0.092, 0.098)),
    (1.16, (0.124, 0.096, 0.100)),
    (1.24, (0.136, 0.108, 0.098)),
    (1.33, (0.150, 0.110, 0.100)),
    (1.40, (0.160, 0.090, 0.100)),
    (1.425, (0.156, 0.074, 0.088)),
    (1.445, (0.112, 0.064, 0.080)),
    (1.462, (0.066, 0.058, 0.076)),
    (1.475, (0.050, 0.040, 0.058)),
)
SKIRT = DRESS[:7]
SKIRT_RINGS = 56
PLEATS = Pleats(count=22.0, depth=0.016, start=1.0, full=0.3)
# The bodice and the skin above it follow the dress; the skin lies this far inside it, the bodice this far outside.
SKIN_INSET = 0.003
BODICE_STANDING = 0.005
TRIM_STANDING = 0.0075
# Cloth edges sink this far under what they lie on, so no gap shows, over this height.
SINK = 0.003
EDGE = 0.015
COLUMNS = 84
BODICE_ROWS = 22
TRIM_ROWS = 5
TRIM_WIDTH = 0.016
# The bodice's lower edge at the waist, and how much lower and how far round either side its point reaches.
WAIST = 1.10
POINT_DEPTH = 0.075
POINT_REACH = 0.6
# The neckline's height at the sides, the front and the back: it runs off the shoulders.
NECKLINE_SIDE = 1.335
NECKLINE_FRONT_DROP = 0.050
NECKLINE_BACK_DROP = 0.020
SKIN_TOP = 1.16
ARM = (
    (0.125, (0.044, 0.044)),
    (0.155, (0.046, 0.046)),
    (0.20, (0.043, 0.043)),
    (0.445, (0.034, 0.034)),
    (0.60, (0.029, 0.029)),
    (0.665, (0.027, 0.027)),
)
# The short sleeve, puffed below the shoulder and closing onto the arm at both ends.
SLEEVE = (
    (0.130, (0.041, 0.041)),
    (0.150, (0.056, 0.056)),
    (0.190, (0.060, 0.058)),
    (0.235, (0.056, 0.054)),
    (0.262, (0.047, 0.047)),
    (0.272, (0.039, 0.039)),
)
OPEN_DRESS = tuple((height, (*radii, 0.0)) for height, radii in DRESS)
SKIN_TRUNK = tuple(
    (height, tuple(radius - SKIN_INSET for radius in radii)) for height, radii in DRESS if height >= SKIN_TOP
)


def lovelace_clothes(face: Face) -> list[Part]:
    return [
        part(gown(SKIRT, SKIRT_RINGS, PLEATS), PLUM, Material.CLOTH, trunk_binder),
        part(_cloth(_waist, _neckline, BODICE_ROWS, BODICE_STANDING, EDGE), PLUM, Material.CLOTH, trunk_binder),
        part(
            _cloth(_trim_low, _trim_high, TRIM_ROWS, TRIM_STANDING, TRIM_WIDTH / 3), TRIM, Material.CLOTH, trunk_binder
        ),
        part(trunk(SKIN_TRUNK, 14), face.skin, Material.SKIN, trunk_binder),
        *both_sides(arm_tube(ARM, 20), face.skin, Material.SKIN, arm_binder),
        *both_sides(arm_tube(SLEEVE, 14), PLUM, Material.CLOTH, arm_binder),
        *shoes(LEATHER),
        *hands(face.skin),
    ]


def _front_angle(angle: float) -> float:
    """The angle from straight ahead, turned to run from a half turn to the left to a half turn to the right."""
    return angle if angle <= math.pi else angle - 2.0 * math.pi


def _waist(angle: float) -> float:
    """The bodice's lower edge: level at the sides and back, dropping to a point at the front."""
    reach = max(0.0, 1.0 - abs(_front_angle(angle)) / POINT_REACH)
    return WAIST - POINT_DEPTH * reach**1.2


def _neckline(angle: float) -> float:
    """The neckline's height round the trunk: highest at the shoulders, lower at the back, lowest at the front."""
    ahead = math.cos(angle)
    return NECKLINE_SIDE - NECKLINE_FRONT_DROP * max(0.0, ahead) ** 1.5 - NECKLINE_BACK_DROP * max(0.0, -ahead)


def _trim_low(angle: float) -> float:
    return _neckline(angle) - TRIM_WIDTH


def _trim_high(angle: float) -> float:
    return _neckline(angle) + SINK


def _cloth(
    lower: Callable[[float], float], upper: Callable[[float], float], rows: int, standing: float, edge: float
) -> Shape:
    """Cloth round the trunk between a lower and an upper edge, given by the angle from straight ahead: it stands off
    the dress by `standing`, and both edges sink under what they lie on over a height of `edge`."""
    rings: list[list[Vector]] = []
    for share in spread(0.0, 1.0, rows):
        ring: list[Vector] = []
        for column in range(COLUMNS):
            angle = 2.0 * math.pi * column / COLUMNS
            low, high = lower(angle), upper(angle)
            height = low + (high - low) * share
            reach = smoothstep(0.0, edge, min(height - low, high - height))
            ring.append(open_point(OPEN_DRESS, height, column / COLUMNS, (standing + SINK) * reach - SINK))
        rings.append(ring)
    return loft(rings, capped=False)
