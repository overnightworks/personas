"""Hildegard's head covering: a white wimple that wraps the neck and the line of the chin, covers the ears and frames
the face between a band over the forehead and a gorget on the chest, and a black veil over the top and back of the head
that falls over the shoulders and down the upper back. No hair shows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_hair import cap_shell
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Material, part

if TYPE_CHECKING:
    from figures.figure_parts import Colour, Part

WIMPLE_WHITE: Colour = (240, 236, 226)
VEIL_BLACK: Colour = (20, 19, 22)
BODY = 0.006
# Rows of each shell; the wimple has more, for the edge that frames the face.
WIMPLE_ROWS = 64
VEIL_ROWS = 64
OPEN = math.radians
# Level above the eye line, then the wimple's half width, front depth, back depth and how far from the middle its front
# edges part. Below the chin it is closed and falls to the chest over the habit; from the chin to the forehead band it
# parts to leave the face free and covers the ears; above the band it is closed over the crown.
WIMPLE = (
    (-0.285, (0.156, 0.122, 0.126, 0.0)),
    (-0.250, (0.140, 0.112, 0.120, 0.0)),
    (-0.215, (0.122, 0.102, 0.112, 0.0)),
    (-0.185, (0.098, 0.086, 0.102, 0.0)),
    (-0.160, (0.082, 0.076, 0.094, 0.0)),
    (-0.145, (0.076, 0.072, 0.090, 0.0)),
    (-0.130, (0.062, 0.062, 0.078, 0.0)),
    (-0.118, (0.052, 0.056, 0.072, 0.0)),
    (-0.112, (0.052, 0.060, 0.072, 0.0)),
    (-0.106, (0.050, 0.080, 0.075, OPEN(62.0))),
    (-0.088, (0.058, 0.084, 0.082, OPEN(60.0))),
    (-0.066, (0.067, 0.087, 0.092, OPEN(57.0))),
    (-0.047, (0.072, 0.088, 0.098, OPEN(56.0))),
    (-0.025, (0.076, 0.088, 0.103, OPEN(55.0))),
    (0.0, (0.078, 0.088, 0.106, OPEN(54.0))),
    (0.020, (0.080, 0.092, 0.107, OPEN(54.0))),
    (0.040, (0.079, 0.091, 0.105, OPEN(54.0))),
    (0.052, (0.079, 0.091, 0.104, 0.0)),
    (0.078, (0.071, 0.081, 0.094, 0.0)),
    (0.100, (0.054, 0.060, 0.072, 0.0)),
    (0.113, (0.034, 0.038, 0.046, 0.0)),
    (0.119, (0.018, 0.020, 0.024, 0.0)),
    (0.127, (0.004, 0.004, 0.005, 0.0)),
)
# The same, for the veil: it lies over the wimple's crown and back, parts wide at the sides of the head and falls behind
# the shoulders to the upper back.
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
    (0.056, (0.093, 0.105, 0.118, OPEN(100.0))),
    (0.072, (0.089, 0.099, 0.113, 0.0)),
    (0.078, (0.085, 0.095, 0.108, 0.0)),
    (0.100, (0.068, 0.074, 0.086, 0.0)),
    (0.113, (0.048, 0.052, 0.060, 0.0)),
    (0.119, (0.032, 0.034, 0.040, 0.0)),
    (0.127, (0.018, 0.020, 0.024, 0.0)),
    (0.133, (0.004, 0.004, 0.005, 0.0)),
)


def hildegard_headdress() -> list[Part]:
    return [
        part(cap_shell(course, BODY, 0.0, 0.0, rows), colour, Material.CLOTH, NECK_BINDER)
        for course, colour, rows in ((WIMPLE, WIMPLE_WHITE, WIMPLE_ROWS), (VEIL, VEIL_BLACK, VEIL_ROWS))
    ]
