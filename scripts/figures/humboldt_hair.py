"""Humboldt's own hair: short and grey-brown, brushed back off a high forehead and a little tousled."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_hair import hair_cap, strands

if TYPE_CHECKING:
    from figures.figure_parts import Part

BROWN = (122, 104, 90)
GREY = (160, 152, 144)
# Level above the eye line, then the hair's half width, front depth and back depth there and how far from the middle
# its front edges part: it leaves the face free and meets over the middle of the forehead.
HAIR = (
    (-0.050, (0.071, 0.076, 0.097, math.radians(82.0))),
    (-0.020, (0.075, 0.081, 0.102, math.radians(76.0))),
    (0.010, (0.077, 0.085, 0.105, math.radians(66.0))),
    (0.040, (0.077, 0.088, 0.104, math.radians(52.0))),
    (0.065, (0.072, 0.083, 0.098, math.radians(32.0))),
    (0.085, (0.062, 0.070, 0.086, math.radians(8.0))),
    (0.100, (0.050, 0.056, 0.070, math.radians(0.0))),
    (0.118, (0.030, 0.034, 0.042, math.radians(0.0))),
    (0.130, (0.004, 0.004, 0.005, math.radians(0.0))),
)
BODY = 0.009
# Tufts stand out from the skull by up to TOUSLE more.
TOUSLE = 0.006
TUFTS = 9.0
# Grey-brown hair with greyer strands.
STRANDS = strands(BROWN, GREY, 0.3, 0.25, 30.0, 9.0)


def humboldt_hair() -> list[Part]:
    return hair_cap(HAIR, STRANDS, BODY, TOUSLE, TUFTS)
