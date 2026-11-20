"""Hildegard's head covering: a white wimple that wraps the neck and the line of the chin, covers the ears and frames
the face between a band over the forehead and a gorget on the chest, and a black veil over the top and back of the head
that falls over the shoulders and down the upper back. No hair shows. Each is made of parts that are either closed or
parted, never turning from one to the other, so that no edge is a sawtooth and no seam runs down the front."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import trunk
from figures.figure_hair import cap_shell
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Material, part
from figures.figure_skeleton import EYE_HEIGHT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_parts import Colour, Part
    from figures.figure_shapes import Shape
    from figures.figure_surfaces import Key

WIMPLE_WHITE: Colour = (240, 236, 226)
VEIL_BLACK: Colour = (20, 19, 22)
BODY = 0.003
# Rings and rows of the wimple's parts, rows and columns of the veil's sheet and rings of its cap.
COLLAR_RINGS = 24
FRAME_ROWS = 40
BAND_RINGS = 20
VEIL_GRID = (60, 30)
VEIL_CAP_RINGS = 16
OPEN = math.radians


# Level above the eye line, then the wimple's half width, front depth, back depth and how far from the middle its front
# edges part. The wimple is three parts, so that no edge has to turn from closed to parted within a few rows: a
# collar that wraps the neck below the chin, a frame that parts to leave the face free and covers the ears, and a band
# over the forehead that closes over the crown. Each lies over the end of the one below it.
COLLAR = (
    (-0.285, (0.156, 0.122, 0.126)),
    (-0.250, (0.140, 0.112, 0.120)),
    (-0.215, (0.122, 0.102, 0.112)),
    (-0.185, (0.098, 0.086, 0.102)),
    (-0.160, (0.082, 0.076, 0.094)),
    (-0.145, (0.076, 0.072, 0.090)),
    (-0.130, (0.062, 0.062, 0.078)),
    (-0.112, (0.052, 0.058, 0.072)),
)
FRAME = (
    (-0.118, (0.052, 0.056, 0.072, OPEN(62.0))),
    (-0.094, (0.054, 0.082, 0.078, OPEN(62.0))),
    (-0.088, (0.058, 0.084, 0.082, OPEN(60.0))),
    (-0.066, (0.067, 0.087, 0.092, OPEN(57.0))),
    (-0.047, (0.072, 0.088, 0.098, OPEN(56.0))),
    (-0.025, (0.076, 0.088, 0.103, OPEN(55.0))),
    (0.0, (0.078, 0.088, 0.106, OPEN(54.0))),
    (0.020, (0.080, 0.092, 0.107, OPEN(54.0))),
    (0.040, (0.079, 0.091, 0.105, OPEN(54.0))),
    (0.058, (0.079, 0.091, 0.104, OPEN(54.0))),
)
BAND = (
    (0.046, (0.079, 0.091, 0.105)),
    (0.056, (0.079, 0.091, 0.104)),
    (0.078, (0.071, 0.081, 0.094)),
    (0.100, (0.054, 0.060, 0.072)),
    (0.113, (0.034, 0.038, 0.046)),
    (0.119, (0.018, 0.020, 0.024)),
    (0.127, (0.004, 0.004, 0.005)),
)
# How far the collar and the band stand off the skin: the frame's edge lies against the collar's and the band's edge
# lies over the frame, so that no two surfaces share a place.
COLLAR_BODY = 0.0035
BAND_BODY = 0.0075
# The same, for the veil: a sheet that lies over the wimple's frame and back, parts wide at the sides of the head and
# falls behind the shoulders to the upper back, and a cap over the crown that lies over the end of the sheet.
VEIL = (
    (-0.520, (0.148, 0.148, 0.158, OPEN(134.0))),
    (-0.460, (0.150, 0.150, 0.158, OPEN(132.0))),
    (-0.400, (0.152, 0.152, 0.158, OPEN(132.0))),
    (-0.345, (0.152, 0.150, 0.156, OPEN(130.0))),
    (-0.300, (0.146, 0.140, 0.150, OPEN(128.0))),
    (-0.262, (0.134, 0.128, 0.142, OPEN(125.0))),
    (-0.232, (0.122, 0.118, 0.134, OPEN(120.0))),
    (-0.200, (0.110, 0.108, 0.124, OPEN(115.0))),
    (-0.175, (0.100, 0.100, 0.116, OPEN(110.0))),
    (-0.150, (0.088, 0.094, 0.110, OPEN(108.0))),
    (-0.120, (0.072, 0.082, 0.096, OPEN(104.0))),
    (-0.100, (0.078, 0.090, 0.092, OPEN(100.0))),
    (-0.066, (0.082, 0.101, 0.106, OPEN(100.0))),
    (-0.025, (0.090, 0.102, 0.117, OPEN(100.0))),
    (0.0, (0.092, 0.102, 0.120, OPEN(100.0))),
    (0.020, (0.094, 0.106, 0.121, OPEN(100.0))),
    (0.040, (0.093, 0.105, 0.119, OPEN(100.0))),
    (0.072, (0.0875, 0.0975, 0.1115, OPEN(100.0))),
)
VEIL_CAP = (
    (0.060, (0.092, 0.104, 0.117)),
    (0.078, (0.085, 0.095, 0.108)),
    (0.100, (0.068, 0.074, 0.086)),
    (0.113, (0.048, 0.052, 0.060)),
    (0.119, (0.032, 0.034, 0.040)),
    (0.127, (0.018, 0.020, 0.024)),
    (0.133, (0.004, 0.004, 0.005)),
)
VEIL_CAP_BODY = 0.0085


def hildegard_headdress() -> list[Part]:
    wimple = (
        _closed(COLLAR, COLLAR_BODY, COLLAR_RINGS),
        cap_shell(FRAME, BODY, 0.0, 0.0, FRAME_ROWS),
        _closed(BAND, BAND_BODY, BAND_RINGS),
    )
    veil = (cap_shell(VEIL, BODY, 0.0, 0.0, *VEIL_GRID), _closed(VEIL_CAP, VEIL_CAP_BODY, VEIL_CAP_RINGS))
    return [
        *(part(shape, WIMPLE_WHITE, Material.CLOTH, NECK_BINDER) for shape in wimple),
        *(part(shape, VEIL_BLACK, Material.CLOTH, NECK_BINDER) for shape in veil),
    ]


def _closed(course: Sequence[Key], body: float, rings: int) -> Shape:
    """A closed solid over the course, which gives its half width, front depth and back depth by level above the eye
    line, standing `body` off it: it has no front edges to part and so no seam."""
    return trunk([(EYE_HEIGHT + level, tuple(radius + body for radius in radii)) for level, radii in course], rings)
