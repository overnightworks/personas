"""Hildegard's habit below the head: a black tunic to the floor with long sleeves that widen towards the wrists, a
darker scapular lying over it front and back, the white cuffs of an undertunic and plain shoes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from figures.figure_body import arm_binder, arm_tube, both_sides, hands, shoes, trunk_binder
from figures.figure_gown import GOWN_COLUMNS, Pleats, gown, gown_ring
from figures.figure_parts import Material, Part, part
from figures.figure_surfaces import patch, spread

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_head import Face
    from figures.figure_shapes import Shape, Vector

TUNIC_BLACK = (46, 44, 50)
SCAPULAR_BLACK = (16, 15, 18)
CUFF = (232, 228, 216)
LEATHER = (30, 26, 26)
# Height, then the tunic's half width, front depth and back depth there: a loose fall from sloping shoulders to the hem
# with no waist, closing to the neck.
TUNIC = (
    (0.03, (0.205, 0.168, 0.192)),
    (0.30, (0.188, 0.152, 0.174)),
    (0.65, (0.170, 0.140, 0.154)),
    (0.95, (0.154, 0.128, 0.134)),
    (1.10, (0.150, 0.120, 0.126)),
    (1.22, (0.156, 0.120, 0.120)),
    (1.28, (0.152, 0.114, 0.116)),
    (1.33, (0.140, 0.105, 0.110)),
    (1.37, (0.124, 0.096, 0.104)),
    (1.40, (0.106, 0.086, 0.096)),
    (1.43, (0.082, 0.070, 0.086)),
    (1.455, (0.066, 0.060, 0.078)),
    (1.47, (0.060, 0.056, 0.076)),
)
TUNIC_RINGS = 48
TUNIC_PLEATS = Pleats(count=18.0, depth=0.016, start=0.95, full=0.3)
# The scapular is a panel over the tunic's front and one over its back, each this many columns either side of the
# middle, standing this far off the cloth, from this height to this height.
SCAPULAR_REACH = 8
SCAPULAR_RELIEF = 0.007
SCAPULAR_SPAN = (0.10, 1.42)
SCAPULAR_ROWS = 40
# Distance from the midline, then the sleeve's (height, depth) radii there: it widens from the upper arm to a wide mouth
# at the wrist.
SLEEVE = (
    (0.125, (0.050, 0.050)),
    (0.20, (0.056, 0.055)),
    (0.40, (0.066, 0.062)),
    (0.55, (0.086, 0.074)),
    (0.665, (0.108, 0.088)),
)
SLEEVE_MOUTH = ((0.668, (0.100, 0.082)), (0.60, (0.056, 0.048)))
# The undertunic's cuff, a narrow tube from the forearm to the wrist inside the sleeve's mouth.
CUFF_COURSE = ((0.56, (0.031, 0.030)), (0.690, (0.029, 0.028)))
SLEEVE_RINGS = 28


def hildegard_clothes(face: Face) -> list[Part]:
    return [
        part(gown(TUNIC, TUNIC_RINGS, TUNIC_PLEATS), TUNIC_BLACK, Material.CLOTH, trunk_binder),
        *(part(panel, SCAPULAR_BLACK, Material.CLOTH, trunk_binder) for panel in _scapular()),
        *both_sides(arm_tube(SLEEVE, SLEEVE_RINGS, SLEEVE_MOUTH), TUNIC_BLACK, Material.CLOTH, arm_binder),
        *both_sides(arm_tube(CUFF_COURSE, 4), CUFF, Material.CLOTH, arm_binder),
        *shoes(LEATHER),
        *hands(face.skin),
    ]


def _scapular() -> list[Shape]:
    """The front panel, whose rows run from the body's right to its left, and the back panel, whose rows run the other
    way so that both face outward."""
    heights = spread(SCAPULAR_SPAN[0], SCAPULAR_SPAN[1], SCAPULAR_ROWS)
    rings = [gown_ring(TUNIC, height, SCAPULAR_RELIEF, TUNIC_PLEATS) for height in heights]
    back = GOWN_COLUMNS // 2
    front_columns = range(-SCAPULAR_REACH, SCAPULAR_REACH + 1)
    back_columns = range(back - SCAPULAR_REACH, back + SCAPULAR_REACH + 1)
    return [patch(_rows(rings, columns)) for columns in (front_columns, back_columns)]


def _rows(rings: Sequence[Sequence[Vector]], columns: range) -> list[list[Vector]]:
    return [[ring[column % GOWN_COLUMNS] for column in columns] for ring in rings]
