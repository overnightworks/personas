"""The face's expressions: each VRM preset as a pose of the jaw, lips, brows and lids, moved through the skin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from figures.figure_head import BROW_LEVEL, CHIN_LEVEL, LIP_LEVEL, MOUTH_HALF_WIDTH, Face, face_point, level_height
from figures.figure_parts import Layer, Part
from figures.figure_shapes import add, scaled
from figures.figure_surfaces import ZERO, bump, smoothstep

if TYPE_CHECKING:
    from figures.figure_shapes import Vector


class Preset(StrEnum):
    """The expression presets, in the order of their morph targets."""

    AA = "aa"
    IH = "ih"
    OU = "ou"
    EE = "ee"
    OH = "oh"
    BLINK = "blink"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    RELAXED = "relaxed"
    SURPRISED = "surprised"


@dataclass(frozen=True)
class Pose:
    """How far each part of the face moves: metres, except the lids, which give the share of their way."""

    jaw: float = 0.0
    stretch: float = 0.0
    pout: float = 0.0
    smile: float = 0.0
    brow_lift: float = 0.0
    brow_slant: float = 0.0
    upper_lid: float = 0.0
    lower_lid: float = 0.0


POSES = {
    Preset.AA: Pose(jaw=0.017, stretch=0.001),
    Preset.IH: Pose(jaw=0.006, stretch=0.004),
    Preset.OU: Pose(jaw=0.006, stretch=-0.007, pout=0.004),
    Preset.EE: Pose(jaw=0.008, stretch=0.006, smile=0.001),
    Preset.OH: Pose(jaw=0.012, stretch=-0.005, pout=0.003),
    Preset.BLINK: Pose(upper_lid=1.0),
    Preset.HAPPY: Pose(jaw=0.002, stretch=0.004, smile=0.007, brow_lift=0.0015, upper_lid=0.12, lower_lid=0.8),
    Preset.ANGRY: Pose(smile=-0.004, brow_lift=-0.002, brow_slant=-0.006, upper_lid=0.2, lower_lid=0.5),
    Preset.SAD: Pose(smile=-0.005, brow_slant=0.006, upper_lid=0.25),
    Preset.RELAXED: Pose(smile=0.003, upper_lid=0.35, lower_lid=0.2),
    Preset.SURPRISED: Pose(jaw=0.011, stretch=-0.003, brow_lift=0.007, upper_lid=-0.35),
}


@dataclass(frozen=True)
class Marks:
    """Where a face's movable features lie."""

    lips: Vector
    chin: float
    brow: float
    cheek: float


def marks(face: Face) -> Marks:
    return Marks(
        lips=face_point(face, 0.0, LIP_LEVEL),
        chin=level_height(face, CHIN_LEVEL),
        brow=level_height(face, BROW_LEVEL),
        cheek=level_height(face, -0.034),
    )


def morph_deltas(part: Part, pose: Pose, places: Marks) -> list[Vector]:
    """How far each vertex of a part moves when the face takes a pose."""
    if part.layer is Layer.UPPER_LID:
        return [scaled(travel, pose.upper_lid) for travel in part.lid_travel]
    if part.layer is Layer.LOWER_LID:
        return [scaled(travel, pose.lower_lid) for travel in part.lid_travel]
    return [_skin_delta(pose, places, position, part.layer) for position in part.shape.positions]


def _skin_delta(pose: Pose, places: Marks, position: Vector, layer: Layer) -> Vector:
    front = smoothstep(0.0, 0.04, position[2])
    if front == 0.0:
        return ZERO
    delta = add(_mouth_delta(pose, places, position), _brow_delta(pose, places, position))
    if layer in (Layer.JAW, Layer.BEARD):
        drop = pose.jaw * _jaw_share(places, position, hanging=layer is Layer.BEARD)
        delta = add(delta, (0.0, -drop, -0.25 * drop))
    return scaled(delta, front)


def _jaw_share(places: Marks, position: Vector, *, hanging: bool) -> float:
    """How much of the jaw's drop a place below the lip line takes: all at the chin, none at the mouth's corners."""
    x, height, _ = position
    reach = abs(x) / MOUTH_HALF_WIDTH
    lip = (1.0 - reach * reach) ** 0.8 if reach < 1.0 else 0.0
    below = smoothstep(0.0, 0.018, places.lips[1] - height)
    chin = bump(x / 0.085)
    if not hanging:
        chin *= smoothstep(places.chin - 0.04, places.chin - 0.008, height)
    return lip * (1.0 - below) + chin * below


def _mouth_delta(pose: Pose, places: Marks, position: Vector) -> Vector:
    x, height, _ = position
    reach = abs(x) / MOUTH_HALF_WIDTH
    beside = bump((abs(x) - MOUTH_HALF_WIDTH) / 0.03)
    above = height - places.lips[1]
    stretch = pose.stretch * math.copysign(reach if reach < 1.0 else beside, x) * bump(above / 0.022)
    corners = (reach * reach if reach < 1.0 else beside) * bump(above / 0.026)
    cheeks = bump(math.hypot((abs(x) - 0.042) / 0.03, (height - places.cheek) / 0.03)) * max(0.0, pose.smile)
    pout = pose.pout * bump(math.hypot(x / (MOUTH_HALF_WIDTH + 0.008), above / 0.016))
    return (stretch, pose.smile * corners + 0.4 * cheeks, pout - 0.5 * abs(stretch) + 0.3 * cheeks)


def _brow_delta(pose: Pose, places: Marks, position: Vector) -> Vector:
    x, height, _ = position
    level = bump((height - places.brow) / 0.022)
    whole = bump((abs(x) - 0.034) / 0.045)
    inner = bump((abs(x) - 0.012) / 0.03)
    outer = bump((abs(x) - 0.056) / 0.022)
    rise = pose.brow_lift * whole + pose.brow_slant * (inner - 0.3 * outer)
    return (math.copysign(0.3 * pose.brow_slant * inner, x) * level, rise * level, 0.0)
