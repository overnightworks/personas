"""Hair that lies over the skull as one shell, its front edges parted to leave the face free."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_body import open_point
from figures.figure_head import NECK_BINDER
from figures.figure_parts import Colour, Material, mixed, part
from figures.figure_shapes import add, ellipsoid, scaled, subtract
from figures.figure_skeleton import EYE_HEIGHT
from figures.figure_surfaces import shell, smoothstep, spread, sweep

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.figure_parts import Paint, Part
    from figures.figure_shapes import Shape, Vector
    from figures.figure_surfaces import Key

ROWS = 24
COLUMNS = 48
# The hair lies on the skull this far above the skin at its edge.
BASE = 0.003
# The height over which the hair thickens from its edge.
EDGE = 0.03
RINGLET_STEPS = 36
# Over this share of its length a ringlet thins towards its tip.
RINGLET_TAPER = 0.35


def hair_cap(course: Sequence[Key], paint: Paint, body: float, tousle: float, tufts: float) -> list[Part]:
    """Hair over the skull; the course, `body`, `tousle` and `tufts` are as for `cap_shell`."""
    return [part(cap_shell(course, body, tousle, tufts), paint, Material.HAIR, NECK_BINDER)]


def cap_shell(
    course: Sequence[Key], body: float, tousle: float, tufts: float, rows: int = ROWS, columns: int = COLUMNS
) -> Shape:
    """A shell over the skull. The course gives, by level above the eye line, its half width, front depth, back depth
    and how far from the middle its front edges part; `body` is how much it thickens from its edge, and tufts, as many
    as `tufts` round the head, stand out from the skull by up to `tousle` more; it is lofted through `rows` levels and
    `columns` steps from one front edge to the other."""
    heights = [(EYE_HEIGHT + level, keys) for level, keys in course]
    levels = spread(course[0][0], course[-1][0], rows)
    outer = [
        [_hair_point(course, heights, level, column, columns, (body, tousle, tufts)) for column in range(columns + 1)]
        for level in levels
    ]
    inner = [
        [_hair_point(course, heights, level, column, columns, (0.0, tousle, tufts)) for column in range(columns + 1)]
        for level in levels
    ]
    return shell(outer, inner)


def _hair_point(
    course: Sequence[Key],
    heights: Sequence[Key],
    level: float,
    column: int,
    columns: int,
    grain: tuple[float, float, float],
) -> Vector:
    """The hair at a level above the eye line and a column round the head; tufts stand out from the skull, more towards
    the crown, and the hair is thinnest at its edge. `grain` is its body, tousle and tufts."""
    body, tousle, tufts = grain
    thickness = BASE + body * smoothstep(course[0][0], course[0][0] + EDGE, level)
    tuft = 0.5 - 0.5 * math.cos(tufts * 2.0 * math.pi * column / columns + 7.0 * math.sin(23.0 * level))
    lift = tousle * tuft * smoothstep(-0.03, 0.08, level) if body else 0.0
    return open_point(heights, EYE_HEIGHT + level, column / columns, thickness + lift)


def strands(
    first: Colour, second: Colour, share: float, swing: float, turns: float, wander: float
) -> Callable[[Vector], Colour]:
    """A paint of `first` hair with strands of `second` that wander as they lie: `share` of the way to the second on
    average, `swing` more or less, `turns` strands round the head and drifting with the height at `wander`."""

    def paint(position: Vector) -> Colour:
        x, height, z = position
        return mixed(first, second, share + swing * math.cos(turns * math.atan2(x, z) + wander * height))

    return paint


def knot(main: tuple[float, float], main_radii: Vector, coil: Vector, coil_radii: Vector, paint: Paint) -> list[Part]:
    """A knot of hair on the head: one rounded coil, given by its level above the eye line and how far behind the middle
    it lies, with a smaller one tucked against it, given by how far to the side, its level and how far behind."""
    level, behind = main
    side, low, back = coil
    coils = (
        ellipsoid((0.0, EYE_HEIGHT + level, behind), main_radii, 8, 14),
        ellipsoid((side, EYE_HEIGHT + low, back), coil_radii, 7, 12),
    )
    return [part(shape, paint, Material.HAIR, "head") for shape in coils]


def ringlet(top: Vector, bottom: Vector, turns: float, curl: float, thickness: float) -> Shape:
    """A lock of hair wound into a corkscrew round the line from `top` to `bottom`, `turns` times at a distance `curl`
    from it, as thick as `thickness` and thinning to its tip."""
    path: list[Vector] = []
    radii: list[tuple[float, float]] = []
    for step in range(RINGLET_STEPS + 1):
        along = step / RINGLET_STEPS
        angle = 2.0 * math.pi * turns * along
        axis = add(top, scaled(subtract(bottom, top), along))
        path.append(add(axis, (curl * math.cos(angle), 0.0, curl * math.sin(angle))))
        radius = thickness * (1.0 - 0.7 * smoothstep(1.0 - RINGLET_TAPER, 1.0, along))
        radii.append((radius, radius))
    return sweep(path, radii, (0.0, 1.0, 0.0), 10)
