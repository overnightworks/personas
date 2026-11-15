"""The furniture of Leonardo's workshop, each in its own frame, standing at the origin and facing +Z.

An easel with an unfinished panel, a stool, a trestle table strewn with his sheets, a shelf of pigment jars and the
frame of a flying machine's wing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_parts import mixed
from figures.figure_shapes import Shape, add, normalised, scaled, subtract
from figures.figure_surfaces import patch, smoothstep, spread, sweep
from figures.room_shapes import (
    ACROSS,
    BEHIND,
    BELOW,
    UP,
    Piece,
    box,
    cuts,
    lathe,
    panel,
    piece,
    placed,
    rotated,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.figure_parts import Colour
    from figures.figure_shapes import Vector

WALNUT = (112, 78, 50)
POPLAR = (178, 148, 104)
OAK = (126, 92, 60)
IRON = (52, 48, 46)
PAPER = (232, 220, 190)
INK = (98, 74, 52)
SANGUINE = (172, 82, 58)
LEATHER = (116, 70, 44)
LINEN = (214, 190, 146)
CORD = (150, 128, 96)
# How high a leg's rod starts, so its round end stays above the floor.
FOOT = 0.03
# The underpainting: a warm toned ground, bare gesso where it is not yet toned, umber drawing, blue-grey distance.
GROUND = (194, 156, 102)
GESSO = (236, 228, 208)
UMBER = (104, 70, 46)
DISTANCE = (128, 138, 136)
SKY = (166, 180, 174)

EASEL_LEAN = math.radians(11.0)
EASEL_HEIGHT = 2.05
EASEL_SPREAD = 0.4
LEDGE = 0.74
PANEL = (0.7, 0.96)
PANEL_CELL = 0.04
PROP_FOOT = -0.85

SEAT = 0.5
STOOL_TOP = ((0.0, 0.17), (0.045, 0.17), (0.045, 0.0))
STOOL_FEET = 3

TABLE = (2.0, 0.78, 0.78)
TABLE_TOP = 0.05
TRESTLE_AT = 0.72
TRESTLE_SPLAY = 0.3
# The sheets on the table: (x, z, turn) of each in the table's frame, plain notes in his mirror hand, or red chalk.
NOTES = ((-0.62, -0.05, 0.18), (-0.38, 0.12, -0.12), (0.1, -0.16, 0.4), (0.18, 0.14, -0.06))
STUDIES = ((-0.08, 0.04, 0.08), (0.52, 0.06, -0.22))
SHEET = (0.22, 0.31)
STUDY_CELL = 0.018
PINNED_CELL = 0.028
NOTE_LINES = 7
INKWELL = ((0.0, 0.035), (0.05, 0.04), (0.06, 0.02), (0.075, 0.018), (0.075, 0.0))

SHELF = (1.2, 0.28)
# The jars on the shelf: (x, turned profile of (height, radius), colour).
JARS = (
    (-0.5, ((0.0, 0.05), (0.03, 0.06), (0.15, 0.058), (0.18, 0.04), (0.2, 0.045), (0.21, 0.0)), (60, 88, 132)),
    (-0.36, ((0.0, 0.04), (0.08, 0.06), (0.12, 0.055), (0.15, 0.03), (0.17, 0.035), (0.175, 0.0)), (164, 96, 62)),
    (-0.22, ((0.0, 0.045), (0.02, 0.05), (0.13, 0.05), (0.15, 0.035), (0.16, 0.0)), (96, 116, 70)),
    (0.0, ((0.0, 0.07), (0.05, 0.08), (0.08, 0.085), (0.09, 0.0)), (186, 170, 146)),
    (0.2, ((0.0, 0.05), (0.03, 0.06), (0.2, 0.058), (0.23, 0.04), (0.25, 0.045), (0.26, 0.0)), (224, 214, 190)),
    (0.34, ((0.0, 0.035), (0.06, 0.05), (0.1, 0.045), (0.12, 0.025), (0.14, 0.03), (0.145, 0.0)), (138, 62, 46)),
)

# The wing's leading spar from its root out to its tip, and how far back each rib reaches from it.
SPAR = (
    (0.0, 0.0, 0.0),
    (-0.45, 0.04, 0.05),
    (-0.9, 0.07, 0.08),
    (-1.35, 0.08, 0.08),
    (-1.8, 0.07, 0.05),
    (-2.25, 0.04, 0.0),
    (-2.7, 0.0, -0.1),
)
RIBS = (1.1, 1.05, 0.95, 0.8, 0.6, 0.38, 0.12)
WING_TILT = 0.6
# A cord from each of these points of the spar up to the ceiling.
CORDS = (1, 4)


def easel() -> list[Piece]:
    """A leaning easel of walnut with an unfinished poplar panel on its ledge, the panel's face toward +Z."""
    width, height = PANEL
    uprights = [
        piece(_rod((x, FOOT, 0.0), (x * 0.8, EASEL_HEIGHT, 0.0), 0.024), WALNUT) for x in (-EASEL_SPREAD, EASEL_SPREAD)
    ]
    frame = [
        *uprights,
        piece(box((-EASEL_SPREAD, LEDGE - 0.04, -0.03), (EASEL_SPREAD, LEDGE, 0.09), omit=[BELOW]), WALNUT),
        piece(box((-0.36, 1.92, -0.03), (0.36, 1.97, 0.02)), WALNUT),
    ]
    board = box((-width / 2, LEDGE, 0.03), (width / 2, LEDGE + height, 0.055), omit=[BELOW])
    face = panel(
        (-width / 2, LEDGE, 0.056),
        ACROSS,
        UP,
        cuts((0.0, width), PANEL_CELL),
        cuts((0.0, height), PANEL_CELL),
    )

    def underpainting(position: Vector) -> Colour:
        return _underpainting((position[0] + width / 2) / width, (position[1] - LEDGE) / height)

    top = (0.0, EASEL_HEIGHT * math.cos(EASEL_LEAN), -EASEL_HEIGHT * math.sin(EASEL_LEAN))
    return [
        *(_leaning(each) for each in (*frame, piece(board, POPLAR), piece(face, underpainting))),
        piece(_rod(top, (0.0, FOOT, PROP_FOOT), 0.02), WALNUT),
    ]


def _leaning(upright: Piece) -> Piece:
    """The piece leant back with the easel about the line across its feet, painted where it stood."""
    return Piece(rotated(upright.shape, ACROSS, -EASEL_LEAN, (0.0, 0.0, 0.0)), upright.colours, upright.glows)


def easel_block() -> tuple[Vector, Vector]:
    """The middle and half sizes of the box the panel on the easel fills, for its shadow."""
    width, height = PANEL
    middle = LEDGE + height / 2
    return (0.0, middle, -middle * math.tan(EASEL_LEAN)), (width / 2, height / 2, 0.08)


def _underpainting(across: float, up: float) -> Colour:
    """A seated figure under a tree before blue mountains, drawn in umber on a toned ground, its lower corner bare."""
    width, height = PANEL
    x, y = across * width, up * height
    horizon = 0.58 + 0.03 * math.sin(across * 7.0)
    peaks = horizon + 0.08 * abs(math.sin(across * 5.0 + 1.0))
    colour = mixed(
        GROUND, DISTANCE, 0.8 * smoothstep(horizon - 0.06, horizon, up) * smoothstep(peaks + 0.03, peaks, up)
    )
    colour = mixed(colour, SKY, 0.7 * smoothstep(peaks, peaks + 0.15, up))
    head = math.hypot((x - 0.25) / 0.055, (y - 0.6) / 0.065)
    body = math.hypot((x - 0.27 - 0.25 * (0.5 - y)) / 0.12, (y - 0.36) / 0.2)
    trunk = abs(x - 0.56 + 0.04 * (y - 0.3)) / 0.025 + max(0.0, y - 0.72) * 20.0
    crown = math.hypot((x - 0.55) / 0.12, (y - 0.8) / 0.1)
    drawn = max(
        0.85 * smoothstep(1.15, 0.85, head),
        0.6 * smoothstep(1.15, 0.85, body),
        0.8 * smoothstep(1.2, 0.8, trunk),
        0.55 * smoothstep(1.15, 0.85, crown),
    )
    bare = smoothstep(0.04, -0.04, up - 0.25 - 0.6 * (across - 0.65))
    return mixed(mixed(colour, UMBER, drawn), GESSO, bare)


def stool() -> list[Piece]:
    """A three-legged stool with a round seat."""
    seat = lathe((0.0, SEAT - STOOL_TOP[1][0], 0.0), STOOL_TOP, 12)
    feet = [(math.sin(angle), 0.0, math.cos(angle)) for angle in spread(0.0, 2.0 * math.pi, STOOL_FEET)[:-1]]
    legs = [
        piece(_rod(add(scaled(foot, 0.22), (0.0, FOOT, 0.0)), add(scaled(foot, 0.1), (0.0, SEAT, 0.0)), 0.02), OAK)
        for foot in feet
    ]
    return [piece(seat, OAK), *legs]


def trestle_table() -> list[Piece]:
    """A plank top on two trestles, with notebooks, loose sheets, red chalk studies, an inkwell and a quill."""
    length, height, depth = TABLE
    top = height - TABLE_TOP
    pieces = [
        piece(box((-length / 2, top, -depth / 2), (length / 2, height, depth / 2), cell=0.5), OAK),
        piece(_rod((-TRESTLE_AT, 0.35, 0.0), (TRESTLE_AT, 0.35, 0.0), 0.025), OAK),
    ]
    for x in (-TRESTLE_AT, TRESTLE_AT):
        pieces += [
            piece(_rod((x, FOOT, side * TRESTLE_SPLAY), (x, top, side * 0.18), 0.03), OAK) for side in (-1.0, 1.0)
        ]
        pieces.append(piece(box((x - 0.04, top - 0.08, -0.32), (x + 0.04, top, 0.32)), OAK))
    for x, z, turn in NOTES:
        pieces += placed([_sheet(height, mirror_writing)], turn, (x, 0.0, z))
    for x, z, turn in STUDIES:
        pieces += placed([_sheet(height + 0.002, chalk_study)], turn, (x, 0.0, z))
    pieces += [
        *_codex((-0.82, height, -0.2), 0.0),
        *_codex((-0.78, height + 0.045, -0.22), 0.05),
        piece(lathe((0.72, height, -0.22), INKWELL, 10), IRON),
        piece(_rod((0.72, height + 0.07, -0.22), (0.6, height + 0.3, -0.05), 0.004), PAPER),
    ]
    return pieces


def table_block() -> tuple[Vector, Vector]:
    """The middle and half sizes of the box the table fills, for the shade on the floor under it."""
    length, height, depth = TABLE
    return (0.0, height / 2, 0.0), (length / 2, height / 2, depth / 2)


def _sheet(height: float, paint: Callable[[float, float], Colour]) -> Piece:
    """A sheet of paper lying on the table, its middle at the origin, painted by its place on the sheet."""
    width, length = SHEET
    surface = panel(
        (-width / 2, height + 0.001, length / 2),
        ACROSS,
        BEHIND,
        cuts((0.0, width), STUDY_CELL),
        cuts((0.0, length), STUDY_CELL),
    )
    return piece(surface, lambda position: paint(position[0] / width + 0.5, 0.5 - position[2] / length))


def mirror_writing(across: float, up: float) -> Colour:
    """Lines of small brown script, ragged at the left where his mirror hand ends them, and a little diagram."""
    line = up * NOTE_LINES
    end = 0.12 + 0.14 * abs(math.sin(math.floor(line) * 2.3))
    written = smoothstep(0.3, 0.9, math.cos(2.0 * math.pi * line)) * smoothstep(end, end + 0.04, across)
    margins = smoothstep(0.92, 0.88, across) * smoothstep(0.06, 0.1, up) * smoothstep(0.94, 0.9, up)
    diagram = smoothstep(0.03, 0.0, abs(math.hypot(across - 0.68, up - 0.3) - 0.12))
    clear = smoothstep(0.17, 0.2, math.hypot(across - 0.68, up - 0.3))
    return mixed(PAPER, INK, max(0.45 * written * margins * clear, 0.6 * diagram))


def chalk_study(across: float, up: float) -> Colour:
    """A head in profile in red chalk, its shadow hatched from upper left to lower right as a left hand draws."""
    head = math.hypot((across - 0.46) / 0.27, (up - 0.6) / 0.3)
    nose = math.hypot((across - 0.74) / 0.06, (up - 0.56) / 0.05)
    outline = smoothstep(0.2, 0.05, min(abs(head - 1.0), abs(nose - 1.0)))
    hatching = 0.5 + 0.5 * math.sin((across - up) * 40.0)
    shadow = smoothstep(1.0, 0.85, head) * smoothstep(0.5, 0.35, across) * hatching
    return mixed(PAPER, SANGUINE, max(0.8 * outline, 0.5 * shadow))


def drawing(width: float, height: float, paint: Callable[[float, float], Colour]) -> Piece:
    """A sheet pinned flat to a wall, facing +Z, its lower left corner at the origin, painted by its place on it."""
    surface = panel((0.0, 0.0, 0.0), ACROSS, UP, cuts((0.0, width), PINNED_CELL), cuts((0.0, height), PINNED_CELL))
    return piece(surface, lambda position: paint(position[0] / width, position[1] / height))


def _codex(corner: Vector, turn: float) -> list[Piece]:
    """A closed notebook bound in leather, its pages showing at the edge."""
    x, y, z = corner
    book = [
        piece(box((-0.11, 0.0, -0.08), (0.11, 0.008, 0.08)), LEATHER),
        piece(box((-0.105, 0.008, -0.075), (0.1, 0.032, 0.075), omit=[BELOW]), PAPER),
        piece(box((-0.11, 0.032, -0.08), (0.11, 0.04, 0.08), omit=[BELOW]), LEATHER),
    ]
    return placed(book, turn, (x, y, z))


def shelf() -> list[Piece]:
    """A plank shelf on two brackets against the wall behind it, with jars of pigment, a mortar and two books."""
    length, depth = SHELF
    pieces = [
        piece(box((-length / 2, -0.04, -depth), (length / 2, 0.0, 0.0), cell=0.6, omit=[BEHIND]), OAK),
        *(
            piece(box((x - 0.025, -0.3, -depth), (x + 0.025, -0.04, -depth + 0.2), omit=[BEHIND]), OAK)
            for x in (-0.45, 0.45)
        ),
    ]
    pieces += [piece(lathe((x, 0.0, -depth / 2), profile, 10), colour) for x, profile, colour in JARS]
    pieces += [
        piece(box((0.42, 0.0, -0.25), (0.46, 0.24, -0.05)), LEATHER),
        piece(box((0.47, 0.0, -0.24), (0.51, 0.21, -0.06)), (92, 60, 40)),
    ]
    return pieces


def wing(hook: float) -> list[Piece]:
    """The frame of a flying machine's wing, ribs of cane fanning back from its spar and linen stretched between.

    It hangs tilted by cords from `hook` above its root, its leading edge up, so its underside faces down toward
    +Z.
    """
    trailing = [add(point, (0.0, -0.02, -reach)) for point, reach in zip(SPAR, RIBS, strict=True)]
    rows = [_scalloped(SPAR, SPAR, 0.0), _scalloped(SPAR, trailing, 0.5), _scalloped(SPAR, trailing, 1.0)]
    shapes = [
        sweep(list(SPAR), [(0.024, 0.024)] * len(SPAR), UP, 6),
        *(_rod(point, end, 0.012) for point, end in zip(SPAR, trailing, strict=True)),
    ]
    cords = [_tilted(SPAR[index]) for index in CORDS]
    return [
        *(piece(_tilted_shape(shape), WALNUT) for shape in shapes),
        piece(_tilted_shape(patch(rows)), LINEN),
        *(piece(_rod(start, (start[0], hook, start[2]), 0.006), CORD) for start in cords),
    ]


def _tilted_shape(shape: Shape) -> Shape:
    return rotated(shape, ACROSS, -WING_TILT, (0.0, 0.0, 0.0))


def _tilted(point: Vector) -> Vector:
    return _tilted_shape(Shape((point,), (UP,), ())).positions[0]


def _scalloped(spar: Sequence[Vector], edge: Sequence[Vector], share: float) -> list[Vector]:
    """A row of the membrane `share` of the way from the spar to the trailing edge, its edge sagging between ribs."""
    row: list[Vector] = []
    for index, (front, back) in enumerate(zip(spar, edge, strict=True)):
        row.append(add(front, scaled(subtract(back, front), share)))
        if index < len(spar) - 1:
            middle_front = scaled(add(front, spar[index + 1]), 0.5)
            middle_back = scaled(add(back, edge[index + 1]), 0.5)
            sag = (1.0 - 0.18 * share) * share
            row.append(add(middle_front, scaled(subtract(middle_back, middle_front), sag)))
    return row


def _rod(start: Vector, end: Vector, radius: float) -> Shape:
    """A round rod from `start` to `end`."""
    along = normalised(subtract(end, start))
    reference = ACROSS if abs(along[1]) > 0.9 or abs(along[2]) > 0.7 else UP
    return sweep([start, end], [(radius, radius), (radius, radius)], reference, 6)
