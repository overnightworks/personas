"""Leonardo's room: his workshop in Milan around 1495, in the warm light of a late afternoon.

Lime plaster over stone above a dark painted band, a floor of terracotta tiles, a plank ceiling on heavy beams, and
one round-arched window on the right, its lower lights open and its arch screened with oiled linen, as a painter
who draws with his left hand wants his light. An easel with an unfinished panel stands in the sun on the left, a
trestle table of notes and red chalk studies under the window, a shelf of pigment jars and a plank door on the back
wall, and the frame of a flying machine's wing hangs from the ceiling. The side toward the camera is open.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_parts import mixed
from figures.figure_shapes import add, normalised
from figures.figure_surfaces import patch, smoothstep, spread
from figures.leonardo_furniture import (
    IRON,
    OAK,
    chalk_study,
    drawing,
    easel,
    easel_block,
    mirror_writing,
    shelf,
    stool,
    table_block,
    trestle_table,
    wing,
)
from figures.room_light import Block, Light, Plane, Window, bake
from figures.room_shapes import (
    ABOVE,
    ACROSS,
    AHEAD,
    BEHIND,
    UP,
    Floor,
    Leaf,
    Openings,
    Sheet,
    Trim,
    Wall,
    box,
    cuts,
    fitted,
    panel,
    piece,
    placed,
    tiled_floor,
    turned,
)
from figures.room_shapes import LEFT as LEFTWARD
from figures.room_shapes import RIGHT as RIGHTWARD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_parts import Colour
    from figures.figure_shapes import Vector
    from figures.room_light import Lit
    from figures.room_shapes import Piece

LEFT_X = -3.3
RIGHT_X = 3.2
BACK_Z = -3.5
FRONT_Z = 2.0
CEILING = 3.55
WIDTH = RIGHT_X - LEFT_X
DEPTH = FRONT_Z - BACK_Z
# The painted band along the foot of the walls and the stripe that ends it.
BAND = 0.95
STRIPE = (BAND, 1.0)
WALL_CELL = 0.45
TRIM_CELL = 1.0
TILE = 0.5
JOINT = 0.01
# The beams across the room, by where each lies front to back, and how deep and wide they are.
BEAMS = (-2.9, -1.7, -0.5, 0.7, 1.85)
BEAM = (0.26, 0.11)
CORBEL = (0.14, 0.2)

COTTO = ((172, 100, 70), (156, 90, 62))
MORTAR = (196, 176, 150)
PLASTER = (228, 214, 188)
SOOT = (176, 160, 136)
BAND_COLOUR = (142, 104, 78)
STRIPE_COLOUR = (118, 52, 40)
TIMBER = (88, 62, 42)
PLANKS = (104, 76, 52)
STONE = (154, 146, 132)
LINEN_LIT = (252, 232, 186)
# The view through the open lights: fields, blue hills, and a sky warm low down and paler high up.
FIELDS = (150, 146, 92)
HILLS = (156, 172, 192)
SKY_LOW = (255, 228, 176)
SKY_HIGH = (214, 226, 236)

BACK = Wall((LEFT_X, 0.0, BACK_Z), ACROSS)
LEFT = Wall((LEFT_X, 0.0, FRONT_Z), BEHIND)
RIGHT = Wall((RIGHT_X, 0.0, BACK_Z), AHEAD)

# The window in the right wall: its sides along the wall from the back, sill, where the arch springs, and the wall's
# thickness; the linen screen sits this deep in the arch.
WINDOW = (2.3, 3.5)
SILL = STRIPE[1]
SPRING = 2.25
RADIUS = (WINDOW[1] - WINDOW[0]) / 2
CROWN = SPRING + RADIUS
REVEAL = 0.55
SCREEN_DEPTH = 0.26
ARCH_STEPS = 12
SHUTTER = 0.6

DOOR = (5.0, 5.9)
DOOR_HEAD = 2.2
DOOR_PLANKS = 4
DOOR_CELL = 0.4

EASEL_AT: Vector = (-1.6, 0.0, -1.3)
EASEL_TURN = 0.45
TABLE_AT: Vector = (2.76, 0.0, -0.55)
TABLE_TURN = -math.pi / 2
WING_AT: Vector = (0.8, 2.68, -2.1)
WING_TURN = 0.2
FURNITURE = (
    (stool, 0.3, (-2.35, 0.0, -0.45)),
    (shelf, 0.0, (0.9, 1.5, BACK_Z + 0.28)),
)
# Sheets pinned to the walls: wall, along, up, width, height and what is drawn on each.
PINNED = (
    (LEFT, 2.5, 1.45, 0.42, 0.58, chalk_study),
    (BACK, 0.9, 2.0, 0.3, 0.4, mirror_writing),
    (BACK, 1.45, 1.95, 0.36, 0.48, chalk_study),
)

SUN = normalised((-0.9, -0.17, -0.24))
SIDES = (
    Plane((0.0, 0.0, 0.0), UP),
    Plane((0.0, CEILING, 0.0), (0.0, -1.0, 0.0)),
    Plane((LEFT_X, 0.0, 0.0), ACROSS),
    Plane((RIGHT_X, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    Plane((0.0, 0.0, BACK_Z), AHEAD),
)


def leonardo_room() -> list[Lit]:
    """Every surface of the workshop with its light baked in."""
    pieces = [
        *_floor(),
        *_ceiling(),
        *_wall(BACK, WIDTH),
        *_wall(LEFT, DEPTH),
        *_wall(RIGHT, DEPTH),
        *_window(),
        *_door(),
        *_furniture(),
    ]
    return bake(pieces, _light())


def _light() -> Light:
    """The low sun of a late afternoon through the open lights, across the easel; easel and table cast shade."""
    easel_centre, easel_half = easel_block()
    table_centre, table_half = table_block()
    start, end = WINDOW
    return Light(
        sun=SUN,
        sun_colour=(3.6, 2.25, 1.05),
        sky_colour=(1.9, 1.75, 1.6),
        bounce_colour=(1.1, 0.72, 0.4),
        ambient=(0.2, 0.155, 0.115),
        windows=(
            Window(
                RIGHT.at((start + end) / 2, (SILL + SPRING) / 2), RIGHT.out, (0.0, (SPRING - SILL) / 2, RADIUS), REVEAL
            ),
        ),
        blocks=(
            Block(add(turned(easel_centre, EASEL_TURN), EASEL_AT), easel_half, EASEL_TURN),
            Block(add(turned(table_centre, TABLE_TURN), TABLE_AT), table_half, TABLE_TURN),
        ),
        sides=SIDES,
        exposure=1.4,
    )


def _floor() -> list[Piece]:
    """Square terracotta tiles in lime mortar, each fired a little differently."""

    def cotto(column: int, row: int) -> Colour:
        shade = ((column * 11 + row * 7) % 7 - 3) * 5
        red, green, blue = COTTO[(column * 3 + row) % 2]
        return (red + shade, green + shade // 2, blue + shade // 2)

    floor = Floor((LEFT_X, 0.0, FRONT_Z), WIDTH, DEPTH, TILE, JOINT, MORTAR, (JOINT, TILE - JOINT, TILE), cotto)
    return [tiled_floor(floor)]


def _ceiling() -> list[Piece]:
    """Planks across heavy beams, each beam resting on a stone corbel at either wall."""
    pieces = [
        piece(panel((LEFT_X, CEILING, BACK_Z), ACROSS, AHEAD, cuts((0.0, WIDTH), 0.8), cuts((0.0, DEPTH), 0.8)), PLANKS)
    ]
    depth, half_width = BEAM
    bottom = CEILING - depth
    for z in BEAMS:
        beam = box(
            (LEFT_X, bottom, z - half_width),
            (RIGHT_X, CEILING, z + half_width),
            cell=TRIM_CELL,
            omit=[ABOVE, LEFTWARD, RIGHTWARD],
        )
        pieces.append(piece(beam, TIMBER))
        for wall, along in ((LEFT, FRONT_Z - z), (RIGHT, z - BACK_Z)):
            corbel = wall.box(
                (along - half_width, along + half_width), (bottom - CORBEL[0], bottom), (0.0, CORBEL[1]), TRIM_CELL
            )
            pieces.append(piece(corbel, STONE))
    return pieces


def _plaster(position: Vector) -> Colour:
    """Lime plaster, uneven where it was laid by hand, and darkened toward the ceiling by years of lamp smoke."""
    x, y, z = position
    unevenness = math.sin(1.7 * x + 2.3 * z + 0.9 * y) * math.sin(1.1 * y - 0.8 * x + 1.9 * z)
    red, green, blue = mixed(PLASTER, SOOT, smoothstep(2.2, CEILING, y))
    shade = round(8 * unevenness)
    return (red + shade, green + shade, blue + shade)


def _wall(wall: Wall, length: float) -> list[Piece]:
    """A wall's painted band, its stripe and the plaster above, open where the window is."""
    openings = Openings((WINDOW,), SILL, CROWN, (SPRING,)) if wall is RIGHT else None
    return fitted(
        wall,
        [
            Sheet(length, (0.0, BAND), BAND_COLOUR, WALL_CELL, openings),
            Trim((0.0, length), STRIPE, (0.0, 0.012), STRIPE_COLOUR, TRIM_CELL),
            Sheet(length, (STRIPE[1], CEILING), _plaster, WALL_CELL, openings, pierced=True),
        ],
    )


def _arch() -> list[tuple[float, float]]:
    """The window's round head as places (along, up) on the right wall, from its back side over to its front."""
    middle = (WINDOW[0] + WINDOW[1]) / 2
    angles = spread(0.0, math.pi, ARCH_STEPS)
    return [(middle - RADIUS * math.cos(angle), SPRING + RADIUS * math.sin(angle)) for angle in angles]


def _on_right_wall(places: Sequence[tuple[float, float]], out: float) -> list[Vector]:
    return [RIGHT.at(along, up, out) for along, up in places]


def _window() -> list[Piece]:
    """The arched window: plastered reveals, a stone sill, the view through the open lights, the linen in the arch
    behind a wooden transom and bars, the wall over the arch, and two shutters folded back against the wall."""
    start, end = WINDOW
    middle = (start + end) / 2
    half = (SILL + SPRING) / 2

    arch = _arch()
    opening = [(start, SILL), (start, half), *arch, (end, half), (end, SILL)]

    def view(position: Vector) -> Colour:
        up = (position[1] - SILL) / (SPRING - SILL)
        across = (position[2] - RIGHT.at(start, 0.0)[2]) / (end - start)
        horizon = 0.34 + 0.05 * math.sin(across * 3.0 + 1.0)
        land = mixed(FIELDS, HILLS, smoothstep(0.18, horizon, up))
        return mixed(
            land, mixed(SKY_LOW, SKY_HIGH, smoothstep(horizon, 1.0, up)), smoothstep(horizon - 0.02, horizon + 0.02, up)
        )

    spring_line = [(along, SPRING) for along, _ in arch]
    over_arch = [(along, CROWN) for along, _ in arch]
    bars_out = (-SCREEN_DEPTH - 0.03, -SCREEN_DEPTH + 0.03)
    rise = SPRING + 0.32
    reach = math.sqrt(RADIUS**2 - 0.32**2)
    return [
        piece(patch([_on_right_wall(opening, -REVEAL), _on_right_wall(opening, 0.0)]), PLASTER),
        piece(patch([_on_right_wall(arch, 0.0), _on_right_wall(over_arch, 0.0)]), _plaster),
        piece(RIGHT.panel(cuts((start, end), 0.08), cuts((SILL, SPRING), 0.08), -REVEAL), view, glows=True),
        piece(
            patch([_on_right_wall(spring_line, -SCREEN_DEPTH), _on_right_wall(arch, -SCREEN_DEPTH)]),
            LINEN_LIT,
            glows=True,
        ),
        piece(RIGHT.box((start, end), (SPRING - 0.04, SPRING + 0.03), bars_out, TRIM_CELL), OAK),
        piece(RIGHT.box((middle - 0.02, middle + 0.02), (SPRING + 0.03, CROWN), bars_out, TRIM_CELL), OAK),
        piece(RIGHT.box((middle - reach, middle + reach), (rise - 0.015, rise + 0.015), bars_out, TRIM_CELL), OAK),
        piece(RIGHT.box((start - 0.06, end + 0.06), (SILL - 0.07, SILL), (-REVEAL, 0.07), TRIM_CELL), STONE),
        *_shutter((start - SHUTTER - 0.04, start - 0.04)),
        *_shutter((end + 0.04, end + SHUTTER + 0.04)),
    ]


def _shutter(along: tuple[float, float]) -> list[Piece]:
    """A plank shutter folded flat against the wall, held by two iron straps."""
    left, right = along
    bottom, top = SILL + 0.03, SPRING - 0.05
    pieces = [piece(RIGHT.box(along, (bottom, top), (0.0, 0.035), 0.3), OAK)]
    for height in (bottom + 0.18, top - 0.18):
        pieces.append(
            piece(
                RIGHT.box((left + 0.03, right - 0.03), (height - 0.025, height + 0.025), (0.035, 0.042), TRIM_CELL),
                IRON,
            )
        )
    return pieces


def _door() -> list[Piece]:
    """A plank door in a plain stone frame, with an iron ring."""
    start, end = DOOR
    frame = 0.14
    width = (end - start) / DOOR_PLANKS
    parts: list[Trim | Leaf] = [
        Trim((start - frame, start), (0.0, DOOR_HEAD), (0.0, 0.06), STONE, TRIM_CELL),
        Trim((end, end + frame), (0.0, DOOR_HEAD), (0.0, 0.06), STONE, TRIM_CELL),
        Trim((start - frame - 0.04, end + frame + 0.04), (DOOR_HEAD, DOOR_HEAD + 0.2), (0.0, 0.08), STONE, TRIM_CELL),
        Leaf((start, end), DOOR_HEAD, 0.02, TIMBER, DOOR_CELL),
    ]
    for plank in range(DOOR_PLANKS):
        left = start + plank * width
        shade = (plank * 5 % 3 - 1) * 6
        colour = (PLANKS[0] + shade, PLANKS[1] + shade, PLANKS[2] + shade)
        parts.append(Leaf((left + 0.006, left + width - 0.006), DOOR_HEAD, 0.03, colour, DOOR_CELL))
    for height in (0.35, DOOR_HEAD - 0.35):
        parts.append(Trim((start + 0.05, end - 0.05), (height - 0.04, height + 0.04), (0.03, 0.05), OAK, TRIM_CELL))
    parts.append(Trim((end - 0.2, end - 0.14), (1.0, 1.1), (0.05, 0.075), IRON, TRIM_CELL))
    return fitted(BACK, parts)


def _furniture() -> list[Piece]:
    pieces = [
        *placed(easel(), EASEL_TURN, EASEL_AT),
        *placed(trestle_table(), TABLE_TURN, TABLE_AT),
        *placed(wing(CEILING - WING_AT[1]), WING_TURN, WING_AT),
    ]
    for build, turn, at in FURNITURE:
        pieces += placed(build(), turn, at)
    for wall, along, up, width, height, paint in PINNED:
        turn = math.atan2(wall.out[0], wall.out[2])
        pieces += placed([drawing(width, height, paint)], turn, wall.at(along, up, 0.004))
    return pieces
