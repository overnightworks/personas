"""Mozart's room: a salon in Vienna around 1785, in the warm light of a late afternoon.

Sage silk between gilt mouldings above a white dado, an oak parquet, two tall windows on the left with gold
curtains, a white double door on the right, a landscape over the middle, Walter's fortepiano by the windows and a
faience stove in the corner. The figure stands in the middle of the floor; the side toward the camera is open.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_parts import mixed
from figures.figure_shapes import add, ellipsoid, normalised, scaled
from figures.figure_surfaces import interpolated, patch, smoothstep
from figures.mozart_furniture import (
    GILT,
    chair,
    fortepiano,
    fortepiano_block,
    round_table,
    stove,
    stove_block,
)
from figures.room_light import Block, Light, Plane, Window, bake
from figures.room_shapes import ACROSS, AHEAD, BEHIND, UP, Wall, cuts, panel, piece, placed, turned

if TYPE_CHECKING:
    from figures.figure_parts import Colour
    from figures.figure_shapes import Vector
    from figures.room_light import Lit
    from figures.room_shapes import Piece

LEFT_X = -3.25
RIGHT_X = 3.25
BACK_Z = -3.2
FRONT_Z = 2.0
CEILING = 3.9
WIDTH = RIGHT_X - LEFT_X
DEPTH = FRONT_Z - BACK_Z
# The heights of the wall: skirting, dado rail, the silk's top under the frieze, and the cornice's steps.
SKIRTING = 0.16
RAIL = (0.9, 0.95)
FRIEZE = 3.45
CORNICE = ((3.52, 3.58, 0.05), (3.58, 3.72, 0.1), (3.72, CEILING, 0.16))
DADO_FRAME = (0.26, 0.8)
SILK_FRAME = (1.15, 3.3)
MOULDING = 0.03
WALL_CELL = 0.45
# Mouldings, rails and bars run long and thin, so their light changes slowly along them.
TRIM_CELL = 1.0
TILE = 0.65
JOINT = 0.012

OAK = ((178, 130, 84), (164, 118, 74))
JOINT_COLOUR = (124, 88, 56)
BOISERIE = (238, 232, 216)
SILK = (146, 164, 134)
SKIRTING_COLOUR = (128, 112, 100)
CEILING_COLOUR = (242, 238, 228)
DOOR = (232, 226, 210)
CURTAIN = (204, 160, 84)
GLASS_LOW = (255, 244, 220)
GLASS_HIGH = (232, 238, 244)
CANVAS = ((214, 196, 150), (150, 166, 176), (104, 118, 92), (58, 70, 46), (118, 98, 64))

BACK = Wall((LEFT_X, 0.0, BACK_Z), ACROSS)
LEFT = Wall((LEFT_X, 0.0, FRONT_Z), BEHIND)
RIGHT = Wall((RIGHT_X, 0.0, BACK_Z), AHEAD)

# The windows on the left wall, as ranges along it from the front, and their sill, head and the wall's thickness.
WINDOWS = ((0.9, 2.2), (3.3, 4.6))
SILL = RAIL[1]
HEAD = 3.1
REVEAL = 0.45
GLASS_DEPTH = 0.33
GLAZING_BAR = 0.025
PANE_ROWS = 5
# A curtain from its rod down: (height, how far it reaches from its outer edge, how deep its folds are).
CURTAIN_FALL = ((3.15, 0.52, 0.05), (2.2, 0.36, 0.06), (1.15, 0.13, 0.08), (0.55, 0.24, 0.07), (0.0, 0.34, 0.06))
CURTAIN_ROWS = 14
CURTAIN_COLUMNS = 13
CURTAIN_FOLDS = 4
CURTAIN_PIER = 0.36

BACK_BAYS = ((0.25, 1.95), (2.25, 4.25), (4.55, 6.25))
PAINTING = ((2.66, 3.84), (1.68, 2.92))
DOOR_LEAVES = (1.25, 2.75)
DOOR_HEAD = 2.7
ARCHITRAVE = 0.12
RIGHT_BAYS = ((3.15, 4.95),)

PIANO_AT: Vector = (-1.75, 0.0, -0.95)
PIANO_TURN = math.atan2(0.45, 0.89)
STOVE_AT: Vector = (2.72, 0.0, -2.67)
FURNITURE = (
    (chair, PIANO_TURN + 0.9, (-2.35, 0.0, -0.05)),
    (round_table, 0.0, (1.5, 0.0, -1.85)),
    (chair, 0.5, (1.0, 0.0, -2.3)),
)

SUN = normalised((0.86, -0.42, -0.30))
SIDES = (
    Plane((0.0, 0.0, 0.0), UP),
    Plane((0.0, CEILING, 0.0), (0.0, -1.0, 0.0)),
    Plane((LEFT_X, 0.0, 0.0), ACROSS),
    Plane((RIGHT_X, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    Plane((0.0, 0.0, BACK_Z), AHEAD),
)


def mozart_room() -> list[Lit]:
    """Every surface of the salon with its light baked in."""
    pieces = [
        *_floor(),
        piece(
            panel((LEFT_X, CEILING, BACK_Z), ACROSS, AHEAD, cuts((0.0, WIDTH), 0.8), cuts((0.0, DEPTH), 0.8)),
            CEILING_COLOUR,
        ),
        *_wall(BACK, WIDTH),
        *_wall(LEFT, DEPTH),
        *_wall(RIGHT, DEPTH),
        *(frame for bay in BACK_BAYS for frame in _bay(BACK, bay)),
        *(frame for bay in RIGHT_BAYS for frame in _bay(RIGHT, bay)),
        *(part for window in WINDOWS for part in _window(window)),
        *_painting(),
        *_door(),
        *_furniture(),
    ]
    return bake(pieces, _light())


def _light() -> Light:
    """The low sun of a late afternoon through both windows; the fortepiano and the stove cast shadows."""
    piano_centre, piano_half = fortepiano_block()
    stove_centre, stove_half = stove_block()
    return Light(
        sun=SUN,
        sun_colour=(3.4, 2.3, 1.25),
        sky_colour=(1.2, 1.15, 1.2),
        bounce_colour=(0.9, 0.6, 0.35),
        ambient=(0.21, 0.17, 0.13),
        windows=tuple(
            Window(
                (LEFT_X, (SILL + HEAD) / 2, FRONT_Z - (start + end) / 2),
                ACROSS,
                (0.0, (HEAD - SILL) / 2, (end - start) / 2),
                REVEAL,
            )
            for start, end in WINDOWS
        ),
        blocks=(
            Block(add(turned(piano_centre, PIANO_TURN), PIANO_AT), piano_half, PIANO_TURN),
            Block(add(stove_centre, STOVE_AT), stove_half),
        ),
        sides=SIDES,
        exposure=1.4,
    )


def _floor() -> list[Piece]:
    """Square oak panels of a Viennese parquet in two tones, each framed by a dark joint."""

    def tile_cuts(length: float) -> list[float]:
        stops = [0.0]
        for tile in range(round(length / TILE)):
            start = tile * TILE
            stops += [start + JOINT, start + TILE / 2, start + TILE - JOINT, start + TILE]
        return stops

    def oak(position: Vector) -> Colour:
        across, back = position[0] - LEFT_X, FRONT_Z - position[2]
        column, row = min(int(across / TILE), 9), min(int(back / TILE), 7)
        inside = min(across - column * TILE, (column + 1) * TILE - across, back - row * TILE, (row + 1) * TILE - back)
        if inside < JOINT / 2:
            return JOINT_COLOUR
        shade = ((column * 7 + row * 13) % 5 - 2) * 3
        red, green, blue = OAK[(column + row) % 2]
        return (red + shade, green + shade, blue + shade)

    return [piece(panel((LEFT_X, 0.0, FRONT_Z), ACROSS, BEHIND, tile_cuts(WIDTH), tile_cuts(DEPTH)), oak)]


def _wall(wall: Wall, length: float) -> list[Piece]:
    """A wall's white dado, its silk and its white frieze, the skirting, the dado rail and the stepped cornice."""
    openings = WINDOWS if wall is LEFT else ()

    def window_hole(along: float, up: float) -> bool:
        return SILL < up < HEAD and any(start < along < end for start, end in openings)

    along = cuts(sorted({0.0, length, *(edge for opening in openings for edge in opening)}), WALL_CELL)
    silk_up = cuts(sorted({RAIL[1], FRIEZE, *((SILL, HEAD) if openings else ())}), WALL_CELL)
    pieces = [
        piece(wall.panel(along, cuts((0.0, RAIL[1]), WALL_CELL)), BOISERIE),
        piece(wall.panel(along, silk_up, hole=window_hole), SILK),
        piece(wall.panel(along, cuts((FRIEZE, CEILING), WALL_CELL)), BOISERIE),
        piece(wall.box((0.0, length), (0.0, SKIRTING), (0.0, 0.022), TRIM_CELL), SKIRTING_COLOUR),
        piece(wall.box((0.0, length), RAIL, (0.0, 0.03), TRIM_CELL), BOISERIE),
    ]
    for bottom, top, out in CORNICE:
        pieces.append(
            piece(wall.box((0.0, length), (bottom, top), (0.0, out), TRIM_CELL), GILT if out < 0.06 else BOISERIE)
        )
    return pieces


def _frame(wall: Wall, along: tuple[float, float], up: tuple[float, float], out: float = 0.0) -> list[Piece]:
    """A gilt moulding round a rectangle on the wall."""
    (left, right), (bottom, top) = along, up
    reach = (out, out + 0.012)
    sides = (
        ((left, right), (bottom, bottom + MOULDING)),
        ((left, right), (top - MOULDING, top)),
        ((left, left + MOULDING), (bottom + MOULDING, top - MOULDING)),
        ((right - MOULDING, right), (bottom + MOULDING, top - MOULDING)),
    )
    return [piece(wall.box(span, rise, reach, TRIM_CELL), GILT) for span, rise in sides]


def _bay(wall: Wall, along: tuple[float, float]) -> list[Piece]:
    return [*_frame(wall, along, DADO_FRAME), *_frame(wall, along, SILK_FRAME)]


def _window(opening: tuple[float, float]) -> list[Piece]:
    """A tall casement window: its reveals, sill, glowing panes behind glazing bars, gilt pelmet and two curtains."""
    start, end = opening
    middle = (start + end) / 2
    glass_up = cuts((SILL, HEAD), 0.3)

    def sky(position: Vector) -> Colour:
        return mixed(GLASS_LOW, GLASS_HIGH, (position[1] - SILL) / (HEAD - SILL))

    rise = (SILL, HEAD)
    reveals = [
        panel(
            LEFT.at(start, SILL), scaled(LEFT.out, -1.0), UP, cuts((0.0, REVEAL), 0.3), cuts((0.0, HEAD - SILL), 0.4)
        ),
        panel(LEFT.at(end, SILL, -REVEAL), LEFT.out, UP, cuts((0.0, REVEAL), 0.3), cuts((0.0, HEAD - SILL), 0.4)),
        panel(
            LEFT.at(start, HEAD, -REVEAL), LEFT.along, LEFT.out, cuts((0.0, end - start), 0.4), cuts((0.0, REVEAL), 0.3)
        ),
    ]
    bars_out = (-GLASS_DEPTH - 0.02, -GLASS_DEPTH + 0.03)
    frame_width = 0.07
    bars = [
        ((start, start + frame_width), rise),
        ((end - frame_width, end), rise),
        ((start, end), (SILL, SILL + frame_width)),
        ((start, end), (HEAD - frame_width, HEAD)),
        ((middle - 0.03, middle + 0.03), rise),
    ]
    pane_bottom, pane_top = SILL + frame_width, HEAD - frame_width
    for row in range(1, PANE_ROWS):
        height = pane_bottom + (pane_top - pane_bottom) * row / PANE_ROWS
        bars.append(((start, end), (height - GLAZING_BAR / 2, height + GLAZING_BAR / 2)))
    for casement_middle in ((start + frame_width + middle - 0.03) / 2, (middle + 0.03 + end - frame_width) / 2):
        bars.append(((casement_middle - GLAZING_BAR / 2, casement_middle + GLAZING_BAR / 2), rise))
    return [
        *(piece(shape, BOISERIE) for shape in reveals),
        piece(LEFT.panel(cuts((start, end), 0.4), glass_up, -GLASS_DEPTH), sky, glows=True),
        *(piece(LEFT.box(span, height, bars_out, TRIM_CELL), BOISERIE) for span, height in bars),
        piece(LEFT.box((start - 0.05, end + 0.05), (SILL - 0.04, SILL), (-REVEAL, 0.05), TRIM_CELL), BOISERIE),
        piece(LEFT.box((start - 0.45, end + 0.45), (HEAD + 0.06, HEAD + 0.3), (0.0, 0.14), TRIM_CELL), GILT),
        *_curtain(start - CURTAIN_PIER, 1.0),
        *_curtain(end + CURTAIN_PIER, -1.0),
    ]


def _curtain(outer: float, toward: float) -> list[Piece]:
    """A curtain hung from its outer edge, gathered by a tasselled tie at about a third of its fall."""
    rows: list[list[Vector]] = []
    for row in range(CURTAIN_ROWS + 1):
        height = CURTAIN_FALL[0][0] * (1.0 - row / CURTAIN_ROWS)
        reach, depth = interpolated([(-key, values) for key, *values in CURTAIN_FALL], -height)
        points: list[Vector] = []
        for column in range(CURTAIN_COLUMNS + 1):
            share = column / CURTAIN_COLUMNS
            fold = 0.5 - 0.5 * math.cos(2.0 * math.pi * CURTAIN_FOLDS * share)
            points.append(LEFT.at(outer + toward * reach * share, max(height, 0.002), 0.04 + depth * fold))
        rows.append(points if toward < 0 else points[::-1])
    tie_height = CURTAIN_FALL[2][0]
    tassel = ellipsoid(LEFT.at(outer + toward * 0.14, tie_height, 0.12), (0.035, 0.05, 0.035), 6, 8)
    return [piece(patch(rows), CURTAIN), piece(tassel, GILT)]


def _painting() -> list[Piece]:
    """A landscape in a broad gilt frame: a pale evening sky over blue hills, dark trees, a sunlit foreground."""
    (left, right), (bottom, top) = PAINTING

    def landscape(position: Vector) -> Colour:
        across = (position[0] - BACK.at(left, 0.0)[0]) / (right - left)
        up = (position[1] - bottom) / (top - bottom)
        sky, hills, meadow, trees, ground = CANVAS
        horizon = 0.42 + 0.08 * math.sin(across * 5.0)
        crowns = 0.3 + 0.45 * max(0.0, 0.4 - across) + 0.5 * max(0.0, across - 0.75)
        if up > horizon:
            return mixed(hills, sky, smoothstep(horizon, horizon + 0.3, up))
        land = mixed(ground, meadow, smoothstep(0.0, horizon, up))
        return mixed(land, trees, smoothstep(crowns + 0.06, crowns - 0.06, up))

    canvas = BACK.panel(cuts((left, right), 0.08), cuts((bottom, top), 0.08), 0.02)
    border = (left - 0.1, right + 0.1), (bottom - 0.1, top + 0.1)
    return [
        piece(canvas, landscape),
        *_frame(BACK, *border),
        *_frame(BACK, (left - 0.06, right + 0.06), (bottom - 0.06, top + 0.06), 0.006),
    ]


def _door() -> list[Piece]:
    """A white double door with raised fields in gilt mouldings, its architrave, a cornice and an overdoor panel."""
    start, end = DOOR_LEAVES
    middle = (start + end) / 2
    pieces = [
        piece(RIGHT.box((start - ARCHITRAVE, start), (0.0, DOOR_HEAD + ARCHITRAVE), (0.0, 0.035), TRIM_CELL), BOISERIE),
        piece(RIGHT.box((end, end + ARCHITRAVE), (0.0, DOOR_HEAD + ARCHITRAVE), (0.0, 0.035), TRIM_CELL), BOISERIE),
        piece(RIGHT.box((start, end), (DOOR_HEAD, DOOR_HEAD + ARCHITRAVE), (0.0, 0.035), TRIM_CELL), BOISERIE),
        piece(
            RIGHT.box((start - 0.2, end + 0.2), (DOOR_HEAD + 0.12, DOOR_HEAD + 0.22), (0.0, 0.09), TRIM_CELL), BOISERIE
        ),
        *_frame(RIGHT, (start + 0.05, end - 0.05), (DOOR_HEAD + 0.3, FRIEZE - 0.1)),
    ]
    for left, right in ((start, middle - 0.004), (middle + 0.004, end)):
        pieces.append(piece(RIGHT.panel(cuts((left, right), 0.4), cuts((0.0, DOOR_HEAD), 0.4), 0.006), DOOR))
        for bottom, top in ((0.22, 0.95), (1.15, 2.45)):
            pieces += _frame(RIGHT, (left + 0.1, right - 0.1), (bottom, top), 0.006)
    for handle in (middle - 0.05, middle + 0.05):
        pieces.append(piece(ellipsoid(RIGHT.at(handle, 1.05, 0.03), (0.015, 0.03, 0.015), 5, 6), GILT))
    return pieces


def _furniture() -> list[Piece]:
    pieces = [*placed(fortepiano(), PIANO_TURN, PIANO_AT), *placed(stove(), 0.0, STOVE_AT)]
    for build, turn, at in FURNITURE:
        pieces += placed(build(), turn, at)
    return pieces
