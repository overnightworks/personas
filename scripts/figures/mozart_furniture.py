"""The furniture of Mozart's salon, each in its own frame, standing at the origin and facing +Z.

A Walter fortepiano with its black naturals and white sharps, a chair, a round table and a white tiled stove.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_surfaces import oval, rippled
from figures.room_shapes import (
    ACROSS,
    AHEAD,
    BEHIND,
    BELOW,
    LEFT,
    RIGHT,
    UP,
    Piece,
    banded,
    box,
    cuts,
    lathe,
    panel,
    piece,
    prism,
    rotated,
    turned_rings,
)

if TYPE_CHECKING:
    from figures.figure_parts import Colour
    from figures.figure_shapes import Vector
    from figures.figure_surfaces import Ring

WALNUT = (118, 74, 44)
WALNUT_DARK = (82, 50, 30)
EBONY = (34, 27, 24)
BONE = (238, 228, 206)
GILT = (206, 166, 82)
BRASS = (196, 158, 86)
CANDLE = (244, 238, 222)
PAPER = (238, 230, 208)
INK = (88, 76, 64)
SILK = (186, 112, 104)
MARBLE = (226, 220, 208)
FAIENCE = (240, 238, 230)

# The fortepiano's case in plan, from the keyboard's bass corner round by the treble side, the bent side and the
# tail back along the straight spine; the keys lie in a well in front of the nameboard.
CASE = (
    (-0.48, 0.0),
    (0.48, 0.0),
    (0.48, -0.42),
    (0.44, -0.80),
    (0.34, -1.20),
    (0.18, -1.62),
    (-0.02, -1.98),
    (-0.20, -2.22),
    (-0.48, -2.22),
)
KEY_WELL = 0.17
CASE_BOTTOM = 0.63
KEY_BED = 0.705
CASE_TOP = 0.90
CHEEK = 0.035
# Five octaves from FF to g''' as Walter built them: the naturals' letters from F, and those with a sharp above.
NATURALS = 37
LETTERS = "FGABCDE"
SHARPENED = "FGACD"
NATURAL_DROP = 0.02
SHARP_RISE = 0.012
SHARP_WIDTH = 0.013
SHARP_LENGTH = 0.095
KEY_GAP = 0.0016
LEGS = ((-0.43, -0.07), (0.43, -0.07), (0.40, -1.0), (-0.43, -1.3), (-0.30, -2.12))
# A tapered square leg with a spade foot: (height, half diagonal).
LEG = ((0.0, 0.034), (0.045, 0.034), (0.05, 0.028), (0.56, 0.05), (0.6, 0.058), (CASE_TOP - 0.27, 0.058))
DESK_TILT = math.radians(16.0)
DESK = (0.56, 0.3)
DESK_PLACE = -0.42
STAVES = 3
STAVE_LINES = 5
CANDLESTICK = (
    (0.0, 0.05),
    (0.012, 0.05),
    (0.02, 0.018),
    (0.13, 0.012),
    (0.15, 0.026),
    (0.165, 0.014),
)
CANDLE_RING = ((0.165, 0.011), (0.33, 0.011), (0.338, 0.004))
CANDLESTICKS = ((-0.36, -0.30), (0.36, -0.30))

SEAT = 0.46
CHAIR_LEG = ((0.0, 0.018), (0.03, 0.022), (0.05, 0.016), (0.30, 0.024), (0.34, 0.03), (SEAT - 0.08, 0.03))
CHAIR_LEGS = ((-0.2, 0.18), (0.2, 0.18), (-0.19, -0.18), (0.19, -0.18))

# The guéridon turned from its foot up: (height, radius) and the finish of the band above each ring.
TABLE = (
    ((0.0, 0.2), "gilt"),
    ((0.025, 0.2), "gilt"),
    ((0.06, 0.09), "gilt"),
    ((0.13, 0.05), "gilt"),
    ((0.56, 0.032), "gilt"),
    ((0.62, 0.05), "gilt"),
    ((0.70, 0.27), "marble"),
    ((0.735, 0.29), "marble"),
    ((0.745, 0.0), "marble"),
)
# The stove's drum, cornice and urn from the base up, the same way; the drum between its mouldings is fluted.
STOVE = (
    ((0.95, 0.36), "faience"),
    ((1.0, 0.33), "faience"),
    ((1.95, 0.33), "gilt"),
    ((2.0, 0.36), "faience"),
    ((2.1, 0.40), "gilt"),
    ((2.14, 0.38), "faience"),
    ((2.3, 0.3), "faience"),
    ((2.42, 0.14), "faience"),
    ((2.46, 0.06), "gilt"),
    ((2.52, 0.1), "gilt"),
    ((2.62, 0.13), "gilt"),
    ((2.72, 0.08), "gilt"),
    ((2.78, 0.0), "gilt"),
)
FLUTED = (1.0, 1.95)
STOVE_FLUTES = 16
STOVE_SEGMENTS = 32
FLUTE_DEPTH = 0.035
STOVE_BASE = 0.42
STOVE_BODY = 0.38
FINISHES = {"gilt": GILT, "marble": MARBLE, "faience": FAIENCE}


def fortepiano() -> list[Piece]:
    """Anton Walter's fortepiano of Mozart's Vienna years: the keyboard's front at z = 0, the tail toward -Z."""
    well = [(x, -KEY_WELL) for x, _ in CASE[:2]]
    return [
        piece(prism(CASE, CASE_BOTTOM, KEY_BED), WALNUT),
        piece(prism([*well, *CASE[2:]], KEY_BED, CASE_TOP), WALNUT),
        *(
            piece(box((x0, KEY_BED, -KEY_WELL), (x1, CASE_TOP - 0.04, 0.0), omit=[BELOW, BEHIND]), WALNUT_DARK)
            for x0, x1 in ((CASE[0][0], CASE[0][0] + CHEEK), (CASE[1][0] - CHEEK, CASE[1][0]))
        ),
        piece(
            panel((-0.09, 0.80, 0.001 - KEY_WELL), ACROSS, UP, (0.0, 0.18), (0.0, 0.035)),
            BONE,
        ),
        *_keys(),
        *(_leg(x, z) for x, z in LEGS),
        *_desk(),
        *(candle for x, z in CANDLESTICKS for candle in _candlestick((x, CASE_TOP, z))),
    ]


def fortepiano_block() -> tuple[Vector, Vector]:
    """The middle and half sizes of the box the fortepiano's case fills, for its shadow."""
    half_length = -CASE[-1][1] / 2
    return (0.0, (CASE_BOTTOM + CASE_TOP) / 2, -half_length), (0.48, (CASE_TOP - CASE_BOTTOM) / 2, half_length)


def _keys() -> list[Piece]:
    start = CASE[0][0] + CHEEK
    width = (CASE[1][0] - CHEEK - start) / NATURALS
    top = KEY_BED + NATURAL_DROP
    keys: list[Piece] = []
    for index in range(NATURALS):
        left = start + index * width
        natural = box(
            (left + KEY_GAP, KEY_BED, -KEY_WELL),
            (left + width - KEY_GAP, top, 0.006),
            omit=[BELOW, BEHIND, LEFT, RIGHT],
        )
        keys.append(piece(natural, EBONY))
        if LETTERS[index % len(LETTERS)] in SHARPENED and index < NATURALS - 1:
            middle = left + width
            sharp = box(
                (middle - SHARP_WIDTH / 2, top, -KEY_WELL),
                (middle + SHARP_WIDTH / 2, top + SHARP_RISE, SHARP_LENGTH - KEY_WELL),
                omit=[BELOW, BEHIND],
            )
            keys.append(piece(sharp, BONE))
    return keys


def _leg(x: float, z: float) -> Piece:
    return piece(lathe((x, 0.0, z), LEG, 4, math.pi / 4), WALNUT)


def _desk() -> list[Piece]:
    """The music desk standing on the lid, leaning back, with a page of music on it."""
    width, height = DESK
    pivot = (0.0, CASE_TOP, DESK_PLACE)
    board = box((-width / 2, CASE_TOP, DESK_PLACE - 0.012), (width / 2, CASE_TOP + height, DESK_PLACE), omit=[BELOW])
    paper_height = height - 0.05
    paper_bottom = CASE_TOP + 0.03
    sheet = panel(
        (-width / 2 + 0.03, paper_bottom, DESK_PLACE + 0.001),
        ACROSS,
        UP,
        cuts((0.0, width - 0.06), 0.25),
        _stave_cuts(paper_height),
    )

    def staves(position: Vector) -> Colour:
        return INK if _on_stave_line(position[1] - paper_bottom, paper_height) else PAPER

    return [
        _leaning(piece(board, WALNUT_DARK), pivot),
        _leaning(piece(sheet, staves), pivot),
    ]


def _leaning(upright: Piece, pivot: Vector) -> Piece:
    """The piece leant back about the line across through `pivot`, painted where it stood."""
    return Piece(rotated(upright.shape, ACROSS, -DESK_TILT, pivot), upright.colours, upright.glows)


def _stave_cuts(height: float) -> list[float]:
    """Cuts up the sheet that give every stave line its own thin band."""
    stops = [0.0, height]
    for line in _stave_lines(height):
        stops += [line - 0.0015, line, line + 0.0015]
    return sorted(stops)


def _stave_lines(height: float) -> list[float]:
    spacing = 0.008
    lines: list[float] = []
    for stave in range(STAVES):
        bottom = height * (stave + 0.6) / (STAVES + 0.4)
        lines += [bottom + line * spacing for line in range(STAVE_LINES)]
    return lines


def _on_stave_line(up: float, height: float) -> bool:
    return any(abs(up - line) < 0.0005 for line in _stave_lines(height))


def _candlestick(base: Vector) -> list[Piece]:
    return [piece(lathe(base, CANDLESTICK, 10), BRASS), piece(lathe(base, CANDLE_RING, 10), CANDLE)]


def chair() -> list[Piece]:
    """A gilt chair in the Louis XVI manner, its seat and oval-topped back upholstered in rose silk."""
    legs = [piece(lathe((x, 0.0, z), CHAIR_LEG, 8), GILT) for x, z in CHAIR_LEGS]
    stiles = [
        piece(box((x - 0.02, SEAT - 0.08, -0.22), (x + 0.02, SEAT + 0.5, -0.18), omit=[BELOW]), GILT)
        for x in (-0.19, 0.19)
    ]
    return [
        *legs,
        *stiles,
        piece(box((-0.24, SEAT - 0.08, -0.22), (0.24, SEAT - 0.03, 0.22), omit=[BELOW]), GILT),
        piece(box((-0.23, SEAT - 0.03, -0.21), (0.23, SEAT + 0.02, 0.21), omit=[BELOW]), SILK),
        piece(box((-0.17, SEAT + 0.48, -0.215), (0.17, SEAT + 0.52, -0.185)), GILT),
        piece(box((-0.17, SEAT + 0.1, -0.205), (0.17, SEAT + 0.47, -0.185), omit=[BEHIND]), SILK),
    ]


def round_table() -> list[Piece]:
    """A guéridon: a gilt pedestal under a white marble top, with a candlestick and a sheet of music on it."""
    top = TABLE[-2][0][0]
    return [
        *_turned((0.0, 0.0, 0.0), TABLE, 20),
        *_candlestick((0.12, top, -0.08)),
        piece(box((-0.2, top, -0.02), (0.02, top + 0.006, 0.14), omit=[BELOW]), PAPER),
    ]


def stove() -> list[Piece]:
    """A Viennese stove of white faience: a square base, a fluted drum, a cornice and a gilt urn on top."""
    rings: list[Ring] = []
    for (height, radius), _ in STOVE:
        ring = oval((0.0, height, 0.0), ACROSS, AHEAD, (radius, radius), STOVE_SEGMENTS)
        if FLUTED[0] < height < FLUTED[1]:
            ring = rippled(ring, (0.0, height, 0.0), STOVE_FLUTES, FLUTE_DEPTH)
        rings.append(ring)
    return [
        piece(box((-STOVE_BASE, 0.0, -STOVE_BASE), (STOVE_BASE, 0.12, STOVE_BASE), omit=[BELOW]), WALNUT_DARK),
        piece(
            box((-STOVE_BODY, 0.12, -STOVE_BODY), (STOVE_BODY, STOVE[0][0][0], STOVE_BODY), cell=0.2, omit=[BELOW]),
            FAIENCE,
        ),
        *banded(rings, [FINISHES[finish] for _, finish in STOVE[:-1]]),
    ]


def stove_block() -> tuple[Vector, Vector]:
    """The middle and half sizes of the box the stove fills, for its shadow."""
    top = STOVE[-1][0][0]
    return (0.0, top / 2, 0.0), (STOVE_BODY, top / 2, STOVE_BODY)


def _turned(base: Vector, profile: tuple[tuple[tuple[float, float], str], ...], count: int) -> list[Piece]:
    rings = turned_rings(base, [ring for ring, _ in profile], count)
    return banded(rings, [FINISHES[finish] for _, finish in profile[:-1]])
