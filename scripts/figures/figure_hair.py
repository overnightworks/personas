"""Hair that lies over the skull as one shell, its front edges parted to leave the face free."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import open_point
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Material, part
from figures.figure_skeleton import EYE_HEIGHT
from figures.figure_surfaces import shell, smoothstep, spread

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_parts import Paint, Part
    from figures.figure_shapes import Vector
    from figures.figure_surfaces import Key

ROWS = 24
COLUMNS = 48
# The hair lies on the skull this far above the skin at its edge.
BASE = 0.003
# The height over which the hair thickens from its edge.
EDGE = 0.03


def hair_cap(course: Sequence[Key], paint: Paint, body: float, tousle: float, tufts: float) -> list[Part]:
    """Hair over the skull. The course gives, by level above the eye line, its half width, front depth, back depth and
    how far from the middle its front edges part; `body` is how much it thickens from its edge, and tufts, as many as
    `tufts` round the head, stand out from the skull by up to `tousle` more."""
    heights = [(EYE_HEIGHT + level, keys) for level, keys in course]
    levels = spread(course[0][0], course[-1][0], ROWS)
    outer = [
        [_hair_point(course, heights, level, column, body, tousle, tufts) for column in range(COLUMNS + 1)]
        for level in levels
    ]
    inner = [
        [_hair_point(course, heights, level, column, 0.0, tousle, tufts) for column in range(COLUMNS + 1)]
        for level in levels
    ]
    return [part(shell(outer, inner), paint, Material.HAIR, NECK_BINDER)]


def _hair_point(
    course: Sequence[Key],
    heights: Sequence[Key],
    level: float,
    column: int,
    body: float,
    tousle: float,
    tufts: float,
) -> Vector:
    """The hair at a level above the eye line and a column round the head; tufts stand out from the skull, more towards
    the crown, and the hair is thinnest at its edge."""
    thickness = BASE + body * smoothstep(course[0][0], course[0][0] + EDGE, level)
    tuft = 0.5 - 0.5 * math.cos(tufts * 2.0 * math.pi * column / COLUMNS + 7.0 * math.sin(23.0 * level))
    lift = tousle * tuft * smoothstep(-0.03, 0.08, level) if body else 0.0
    return open_point(heights, EYE_HEIGHT + level, column / COLUMNS, thickness + lift)
