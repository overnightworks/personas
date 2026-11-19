"""Humboldt's own hair: short and grey-brown, brushed back off a high forehead and a little tousled."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import open_point
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Colour, Material, Part, mixed, part
from figures.figure_skeleton import EYE_HEIGHT
from figures.figure_surfaces import shell, smoothstep, spread

if TYPE_CHECKING:
    from figures.figure_shapes import Vector

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
HAIR_COURSE = tuple((EYE_HEIGHT + level, keys) for level, keys in HAIR)
ROWS = 24
COLUMNS = 48
# The hair lies on the skull this far above the skin at its edge and stands out by up to TOUSLE more where it is tufted.
BASE = 0.003
BODY = 0.009
TOUSLE = 0.006
TUFTS = 9.0
# The height over which the hair thickens from its edge.
EDGE = 0.03


def humboldt_hair() -> list[Part]:
    outer = [[_hair_point(level, column, BODY) for column in range(COLUMNS + 1)] for level in _levels()]
    inner = [[_hair_point(level, column, 0.0) for column in range(COLUMNS + 1)] for level in _levels()]
    return [part(shell(outer, inner), _strands, Material.HAIR, NECK_BINDER)]


def _levels() -> list[float]:
    return spread(HAIR[0][0], HAIR[-1][0], ROWS)


def _hair_point(level: float, column: int, body: float) -> Vector:
    """The hair at a level above the eye line and a column round the head; tufts stand out from the skull, more towards
    the crown, and the hair is thinnest at its edge."""
    height = EYE_HEIGHT + level
    thickness = BASE + body * smoothstep(HAIR[0][0], HAIR[0][0] + EDGE, level)
    tuft = 0.5 - 0.5 * math.cos(TUFTS * 2.0 * math.pi * column / COLUMNS + 7.0 * math.sin(23.0 * level))
    lift = TOUSLE * tuft * smoothstep(-0.03, 0.08, level) if body else 0.0
    return open_point(HAIR_COURSE, height, column / COLUMNS, thickness + lift)


def _strands(position: Vector) -> Colour:
    """Grey-brown hair with greyer strands that wander as they lie."""
    x, height, z = position
    return mixed(BROWN, GREY, 0.3 + 0.25 * math.cos(30.0 * math.atan2(x, z) + 9.0 * height))
