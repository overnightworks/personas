"""The inside of the mouth: a dark hollow behind the lips and the upper teeth, seen when the jaw drops."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_head import LIP_LEVEL, MOUTH_HALF_WIDTH, face_point
from figures.figure_parts import Layer, Material, part
from figures.figure_surfaces import patch

if TYPE_CHECKING:
    from figures.figure_head import Face
    from figures.figure_parts import Part
    from figures.figure_shapes import Vector

HOLLOW = (58, 18, 20)
TEETH = (236, 230, 216)
HOLLOW_RINGS = 6
HOLLOW_SEGMENTS = 20
TEETH_HALF_WIDTH = 0.018
TEETH_COLUMNS = 12


def mouth(face: Face) -> list[Part]:
    lips = face_point(face, 0.0, LIP_LEVEL)
    return [
        part(patch(_hollow_rings(lips), wrap=True), HOLLOW, Material.MOUTH, "head", Layer.FACE),
        part(patch(_teeth_rows(lips)), TEETH, Material.MOUTH, "head", Layer.FACE),
    ]


def _hollow_rings(lips: Vector) -> list[list[Vector]]:
    """A bowl open towards the lips, its rim just behind them and its floor deep in the head."""
    x, y, z = lips[0], lips[1] - 0.003, lips[2] - 0.006
    rings: list[list[Vector]] = []
    for ring in range(HOLLOW_RINGS + 1):
        depth = 0.5 * math.pi * ring / HOLLOW_RINGS
        width, height = (MOUTH_HALF_WIDTH - 0.002) * math.cos(depth), 0.011 * math.cos(depth)
        rings.append(
            [
                (x + width * math.cos(turn), y + height * math.sin(turn), z - 0.022 * math.sin(depth))
                for turn in (2.0 * math.pi * step / HOLLOW_SEGMENTS for step in range(HOLLOW_SEGMENTS))
            ]
        )
    return rings


def _teeth_rows(lips: Vector) -> list[list[Vector]]:
    """The upper teeth as an arch behind the upper lip, their edge a little below the lip line."""
    rows: list[list[Vector]] = []
    for height in (lips[1] - 0.0045, lips[1] + 0.004):
        row: list[Vector] = []
        for column in range(TEETH_COLUMNS + 1):
            reach = 2.0 * column / TEETH_COLUMNS - 1.0
            row.append((reach * TEETH_HALF_WIDTH, height, lips[2] - 0.0035 - 0.012 * reach * reach))
        rows.append(row)
    return rows
