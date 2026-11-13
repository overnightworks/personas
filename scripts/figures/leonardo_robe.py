"""Leonardo's clothes: a long pleated gown with wide sleeves, a belt, a linen collar, a soft cap and plain shoes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import (
    ACROSS,
    AHEAD,
    TRUNK_POWER,
    arm_binder,
    arm_tube,
    band,
    both_sides,
    hands,
    shoes,
    trunk,
    trunk_binder,
)
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Colour, Material, Part, part
from figures.figure_skeleton import EYE_HEIGHT, SHOULDER_HEIGHT
from figures.figure_surfaces import interpolated, loft, oval, patch, rippled, smoothstep, spread

if TYPE_CHECKING:
    from figures.figure_head import Face
    from figures.figure_shapes import Vector

ROSE = (150, 62, 70)
UMBER = (62, 44, 38)
WINE = (96, 38, 46)
OCHRE = (184, 142, 66)
LINEN = (236, 230, 214)
VELVET = (48, 30, 36)
HIDE = (70, 46, 34)
# Height, then the gown's half width, front depth and back depth there.
GOWN = (
    (0.05, (0.186, 0.140, 0.176)),
    (0.30, (0.178, 0.142, 0.168)),
    (0.60, (0.166, 0.138, 0.154)),
    (0.95, (0.150, 0.128, 0.134)),
    (1.07, (0.142, 0.120, 0.124)),
    (1.22, (0.156, 0.124, 0.120)),
    (1.34, (0.160, 0.108, 0.112)),
    (1.40, (0.166, 0.090, 0.100)),
    (1.425, (0.160, 0.074, 0.088)),
    (1.445, (0.112, 0.064, 0.080)),
    (1.462, (0.066, 0.058, 0.076)),
)
GOWN_RINGS = 40
GOWN_COLUMNS = 84
PLEATS = 14.0
PLEAT_DEPTH = 0.05
PLEATS_START = 1.05
PLEATS_FULL = 0.25
TRIM_RELIEF = 0.003
HEM_BAND = 0.035
BELT = (1.03, 1.072)
COLLAR = (1.442, 1.462)
SHIRT = ((1.44, (0.063, 0.055, 0.068)), (1.478, (0.061, 0.053, 0.066)))
SLEEVE = (
    (0.125, (0.044, 0.044)),
    (0.155, (0.055, 0.054)),
    (0.20, (0.057, 0.055)),
    (0.445, (0.055, 0.052)),
    (0.60, (0.062, 0.058)),
    (0.655, (0.070, 0.063)),
)
SLEEVE_MOUTH = ((0.658, (0.062, 0.055)), (0.60, (0.034, 0.033)))
UNDERSLEEVE = ((0.56, (0.031, 0.030)), (0.685, (0.029, 0.028)))
# Level above the eye line, then the cap's half width, front depth and back depth there: a band, then a soft crown.
CAP = (
    (0.038, (0.086, 0.090, 0.110)),
    (0.044, (0.092, 0.097, 0.117)),
    (0.056, (0.093, 0.098, 0.118)),
    (0.066, (0.091, 0.095, 0.116)),
    (0.080, (0.098, 0.102, 0.124)),
    (0.098, (0.100, 0.103, 0.127)),
    (0.114, (0.093, 0.094, 0.118)),
    (0.126, (0.072, 0.070, 0.090)),
    (0.133, (0.036, 0.036, 0.046)),
)
CAP_TILT: Vector = (0.0, 0.2, 1.0)


def leonardo_clothes(face: Face) -> list[Part]:
    return [
        *_gown(),
        part(band(GOWN, *BELT, 0.008), OCHRE, Material.LEATHER, trunk_binder),
        part(band(GOWN, *COLLAR, 0.004), UMBER, Material.CLOTH, trunk_binder),
        part(trunk(SHIRT, 3), LINEN, Material.CLOTH, NECK_BINDER),
        *both_sides(arm_tube(SLEEVE, 28, SLEEVE_MOUTH), _sleeve_paint, Material.CLOTH, arm_binder),
        *both_sides(arm_tube(UNDERSLEEVE, 4), LINEN, Material.CLOTH, arm_binder),
        _cap(),
        *shoes(HIDE),
        *hands(face.skin),
    ]


def _gown() -> list[Part]:
    """The gown, with a dark band around its hem and another down its front."""
    hem, top = GOWN[0][0], GOWN[-1][0]
    body = loft([_gown_ring(height, 0.0) for height in spread(hem, top, GOWN_RINGS)])
    hem_band = patch([_gown_ring(height, TRIM_RELIEF) for height in (hem - 0.002, hem + HEM_BAND)], wrap=True)
    front_band = patch([_front_band_row(height) for height in spread(hem, COLLAR[0], GOWN_RINGS)])
    return [
        part(body, ROSE, Material.CLOTH, trunk_binder),
        *(part(shape, UMBER, Material.CLOTH, trunk_binder) for shape in (hem_band, front_band)),
    ]


def _gown_ring(height: float, grown: float) -> list[Vector]:
    """The gown at a height: an oval that falls into pleats below the belt, deepest towards the hem."""
    centre = (0.0, height, 0.0)
    radii = [radius + grown for radius in interpolated(GOWN, height)]
    ring = oval(centre, ACROSS, AHEAD, radii, GOWN_COLUMNS, TRUNK_POWER)
    return rippled(ring, centre, PLEATS, PLEAT_DEPTH * smoothstep(PLEATS_START, PLEATS_FULL, height))


def _front_band_row(height: float) -> list[Vector]:
    """A strip over the gown's front middle, raised off the cloth and tucked back into it at both edges."""
    raised, tucked = _gown_ring(height, TRIM_RELIEF), _gown_ring(height, -TRIM_RELIEF)
    return [tucked[-2], raised[-1], raised[0], raised[1], tucked[2]]


def _sleeve_paint(position: Vector) -> Colour:
    """Rose outside; the darker lining shows where the sleeve's mouth turns back inside."""
    x, height, z = position
    off_axis = math.hypot(height - SHOULDER_HEIGHT, z)
    return WINE if abs(x) > SLEEVE[-1][0] + 0.001 or off_axis < SLEEVE_MOUTH[-1][1][0] + 0.005 else ROSE


def _cap() -> Part:
    """A soft cap worn on the back of the head, its band lower at the nape than over the forehead."""
    rings = [
        oval((0.0, EYE_HEIGHT + level, 0.0), ACROSS, CAP_TILT, interpolated(CAP, level), 28)
        for level in spread(CAP[0][0], CAP[-1][0], 16)
    ]
    return part(loft(rings), VELVET, Material.CLOTH, "head")
