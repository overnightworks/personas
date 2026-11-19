"""Curie's dress: plain dark, high at the neck, a fitted bodice, long narrow sleeves and a long skirt to just above the
shoes, with a small light edge at the collar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from figures.figure_body import arm_binder, arm_tube, band, both_sides, hands, shoes, trunk, trunk_binder
from figures.figure_gown import Pleats, gown
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Material, Part, part

if TYPE_CHECKING:
    from figures.figure_head import Face

CLOTH = (30, 32, 44)
COLLAR_EDGE = (226, 222, 212)
LEATHER = (36, 30, 30)
# Height, then the dress's half width, front depth and back depth there: a skirt that clears the legs and falls
# straight to the hem, a waist drawn in, a fitted bodice and a high neckline.
DRESS = (
    (0.09, (0.215, 0.165, 0.185)),
    (0.35, (0.190, 0.152, 0.168)),
    (0.65, (0.172, 0.140, 0.150)),
    (0.92, (0.158, 0.118, 0.126)),
    (1.08, (0.122, 0.094, 0.098)),
    (1.13, (0.116, 0.090, 0.092)),
    (1.24, (0.136, 0.108, 0.098)),
    (1.33, (0.150, 0.110, 0.100)),
    (1.40, (0.160, 0.090, 0.100)),
    (1.425, (0.156, 0.074, 0.088)),
    (1.445, (0.112, 0.064, 0.080)),
    (1.462, (0.066, 0.058, 0.076)),
)
DRESS_RINGS = 44
PLEATS = Pleats(count=16.0, depth=0.012, start=1.0, full=0.3)
WAIST = (1.075, 1.135)
# The standing collar rises from the neckline to under the chin; its light edge runs round the top.
NECKBAND = ((1.44, (0.064, 0.056, 0.069)), (1.47, (0.060, 0.052, 0.065)), (1.50, (0.058, 0.050, 0.063)))
COLLAR_EDGE_COURSE = ((1.494, (0.0605, 0.0525, 0.0655)), (1.508, (0.0605, 0.0525, 0.0655)))
SLEEVE = (
    (0.125, (0.047, 0.047)),
    (0.155, (0.052, 0.051)),
    (0.20, (0.050, 0.049)),
    (0.445, (0.041, 0.040)),
    (0.60, (0.034, 0.034)),
    (0.665, (0.033, 0.033)),
)


def curie_clothes(face: Face) -> list[Part]:
    return [
        part(gown(DRESS, DRESS_RINGS, PLEATS), CLOTH, Material.CLOTH, trunk_binder),
        part(band(DRESS, *WAIST, 0.004), CLOTH, Material.CLOTH, trunk_binder),
        part(trunk(NECKBAND, 4), CLOTH, Material.CLOTH, NECK_BINDER),
        part(trunk(COLLAR_EDGE_COURSE, 2), COLLAR_EDGE, Material.CLOTH, NECK_BINDER),
        *both_sides(arm_tube(SLEEVE, 28), CLOTH, Material.CLOTH, arm_binder),
        *shoes(LEATHER),
        *hands(face.skin),
    ]
