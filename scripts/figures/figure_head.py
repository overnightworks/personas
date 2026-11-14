"""The head's skin: a skull lofted from level rings, with the face's relief raised on its front."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.figure_parts import Colour, Layer, Material, Part, chain, mixed, part
from figures.figure_skeleton import EYE_DEPTH, EYE_HEIGHT, EYE_SPACING
from figures.figure_surfaces import bump, interpolated, loft, mirrored, oval, patch, seamless, smoothstep

if TYPE_CHECKING:
    from figures.figure_shapes import Vector

COLUMNS = 96
FRONT_DENSITY = 0.5
LIP_LEVEL = -0.066
CHIN_LEVEL = -0.105
NECK_LEVEL = -0.2
BROW_LEVEL = 0.021
MOUTH_HALF_WIDTH = 0.024
CORNER_LIFT = 0.0018
NOSTRIL = (62, 30, 26)
# Above 1 the nose's ridge is a flat bridge between two flanks instead of an edge.
NOSE_BRIDGE = 1.5
# Level above the eye line, then half width, front depth, back depth and squareness of the skull's ring there.
SKULL = (
    (NECK_LEVEL, (0.052, 0.040, 0.060, 2.0)),
    (-0.140, (0.048, 0.038, 0.058, 2.0)),
    (-0.124, (0.046, 0.040, 0.060, 2.0)),
    (-0.116, (0.044, 0.050, 0.066, 2.0)),
    (CHIN_LEVEL, (0.045, 0.074, 0.070, 2.0)),
    (-0.088, (0.054, 0.080, 0.078, 2.1)),
    (LIP_LEVEL, (0.063, 0.083, 0.088, 2.2)),
    (-0.047, (0.068, 0.084, 0.094, 2.3)),
    (-0.025, (0.072, 0.084, 0.099, 2.3)),
    (0.0, (0.074, 0.084, 0.102, 2.3)),
    (0.020, (0.076, 0.087, 0.103, 2.3)),
    (0.048, (0.075, 0.086, 0.100, 2.25)),
    (0.078, (0.067, 0.076, 0.090, 2.15)),
    (0.100, (0.050, 0.056, 0.068, 2.05)),
    (0.113, (0.030, 0.034, 0.042, 2.0)),
    (0.119, (0.0, 0.0, 0.0, 2.0)),
)
# Bands of rows from the neck up: first level, last level and the rows between; the lips and the nose get the most.
JAW_BANDS = ((NECK_LEVEL, -0.124, 5), (-0.124, CHIN_LEVEL, 4), (CHIN_LEVEL, -0.080, 7), (-0.080, LIP_LEVEL, 9))
SKULL_BANDS = ((LIP_LEVEL, -0.054, 8), (-0.054, -0.036, 7), (-0.036, 0.035, 24), (0.035, 0.119, 12))
# Level, then how far the ear reaches to the back and the front of its middle and how far it stands off the head.
EAR = (
    (-0.046, (0.004, 0.006)),
    (-0.036, (0.008, 0.008)),
    (-0.016, (0.013, 0.009)),
    (0.004, (0.016, 0.009)),
    (0.012, (0.012, 0.007)),
    (0.017, (0.004, 0.004)),
)
NECK_BINDER = chain(1, "chest", (("neck", 1.43, 0.02), ("head", 1.485, 0.02)))


@dataclass(frozen=True)
class Face:
    """What tells one face from another: lengths in metres, ratios against an average face."""

    skin: Colour
    blush: Colour
    lips: Colour
    iris: Colour
    brow: Colour
    brow_weight: float
    length: float
    jaw: float
    nose_length: float
    nose_height: float
    nose_width: float
    brow_ridge: float
    eye_set: float
    cheek: float


def level_height(face: Face, level: float) -> float:
    """The height of a level of the face: the eye line is level 0, and the face below it stretches by its length."""
    return EYE_HEIGHT + level * (face.length if level < 0.0 else 1.0)


def skin_point(face: Face, level: float, angle: float) -> Vector:
    """The skin at a level and an angle around the head, 0 straight ahead and a quarter turn at the left ear."""
    half_width, front, back, squareness = interpolated(SKULL, level)
    half_width *= 1.0 + (face.jaw - 1.0) * smoothstep(-0.02, -0.07, level)
    across, along = math.sin(angle), math.cos(angle)
    x = half_width * math.copysign(abs(across) ** (2.0 / squareness), across)
    z = (front if along >= 0.0 else back) * math.copysign(abs(along) ** (2.0 / squareness), along)
    return (x, level_height(face, level), z + smoothstep(0.0, 0.04, z) * _relief(face, x, level, z))


def face_point(face: Face, x: float, level: float) -> Vector:
    """The skin on the front of the face at a distance from the midline and a level."""
    half_width, _, _, squareness = interpolated(SKULL, level)
    half_width *= 1.0 + (face.jaw - 1.0) * smoothstep(-0.02, -0.07, level)
    reach = min(1.0, abs(x) / half_width) ** (squareness / 2.0)
    return skin_point(face, level, math.copysign(math.asin(reach), x))


def head_skin(face: Face) -> list[Part]:
    """The neck and jaw below the lip line and the skull above it, meeting where the mouth opens."""

    def paint(position: Vector) -> Colour:
        return _skin_colour(face, position)

    jaw = patch([_ring(face, level) for level in _levels(JAW_BANDS)], wrap=True)
    skull = patch([_ring(face, level) for level in _levels(SKULL_BANDS)], wrap=True)
    halves = zip(seamless([jaw, skull]), (Layer.JAW, Layer.FACE), strict=True)
    return [part(shape, paint, Material.SKIN, NECK_BINDER, layer) for shape, layer in halves]


def ears(face: Face) -> list[Part]:
    """Both ears: a shell that leans back as it rises, widest above its middle."""
    rings = [
        oval(
            (0.0725, level_height(face, level), -0.004 - 0.14 * (level - EAR[0][0])),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            interpolated(EAR, level),
            10,
        )
        for level in _levels(((EAR[0][0], EAR[-1][0], 12),))
    ]
    colour = mixed(face.skin, face.blush, 0.3)
    return [part(shape, colour, Material.SKIN, "head") for shape in (loft(rings), mirrored(loft(rings)))]


def _levels(bands: tuple[tuple[float, float, int], ...]) -> list[float]:
    levels = [bands[0][0]]
    for start, end, rows in bands:
        levels += [start + (end - start) * row / rows for row in range(1, rows + 1)]
    return levels


def _ring(face: Face, level: float) -> list[Vector]:
    ring: list[Vector] = []
    for column in range(COLUMNS):
        turn = 2.0 * column / COLUMNS - 1.0
        angle = math.pi * turn * (FRONT_DENSITY + (1.0 - FRONT_DENSITY) * turn * turn)
        ring.append(skin_point(face, level, angle))
    return ring


def _relief(face: Face, x: float, level: float, depth: float) -> float:
    """How far the face stands out of the skull at a place on its front, where the skull is `depth` deep."""
    return (
        _eye_sockets(face, x, level, depth)
        + _nose(face, x, level)
        + _mouth(x, level)
        + face.cheek * bump(math.hypot((abs(x) - 0.042) / 0.03, (level + 0.034) / 0.03))
        + 0.004 * bump(math.hypot(x / 0.022, (level - CHIN_LEVEL - 0.014) / 0.016))
    )


def _eye_sockets(face: Face, x: float, level: float, depth: float) -> float:
    """A dish around each eye whose floor lies just in front of the eye's centre, so the lids stand out of the skin."""
    reach = math.hypot((abs(x) - EYE_SPACING) / 0.025, (level - 0.001) / 0.019)
    dish = 1.0 - smoothstep(0.45, 1.0, reach)
    ridge = bump(math.hypot((abs(x) - 0.034) / 0.034, (level - BROW_LEVEL - 0.004) / 0.011))
    return face.brow_ridge * ridge + dish * min(0.0, EYE_DEPTH + face.eye_set - depth)


def _nose(face: Face, x: float, level: float) -> float:
    base, tip, root = -0.047 * face.nose_length, -0.039 * face.nose_length, 0.006
    height = face.nose_height
    ridge = (
        (base - 0.006, (0.0, 0.014)),
        (base, (0.3 * height, 0.016)),
        (tip, (height, 0.014)),
        (tip + 0.3 * (root - tip), (0.66 * height, 0.011)),
        (root, (0.14 * height + 0.002, 0.009)),
        (root + 0.024, (0.0, 0.012)),
    )
    rise, half_width = interpolated(ridge, level)
    wing = bump(math.hypot((abs(x) - 0.0125 * face.nose_width) / 0.008, (level - base - 0.007) / 0.008))
    reach = abs(x) / (half_width * face.nose_width)
    return rise * bump(reach**NOSE_BRIDGE) + 0.42 * height * wing


def _mouth(x: float, level: float) -> float:
    above = _above_lip_line(x, level)
    muzzle = 0.003 * bump(math.hypot(x / 0.042, above / 0.03))
    upper = 0.0032 * bump(math.hypot(x / (MOUTH_HALF_WIDTH + 0.004), (above - 0.0055) / 0.0065))
    lower = 0.0045 * bump(math.hypot(x / (MOUTH_HALF_WIDTH * 0.85), (above + 0.0065) / 0.0075))
    crease = 0.0026 * bump(math.hypot(x / (MOUTH_HALF_WIDTH + 0.008), above / 0.0035))
    groove = 0.0028 * bump(math.hypot(x / 0.02, (above + 0.019) / 0.008))
    return muzzle + upper + lower - crease - groove


def _above_lip_line(x: float, level: float) -> float:
    """How far a place lies above the line where the lips meet, which rises a little towards the corners."""
    reach = min(1.2, abs(x) / MOUTH_HALF_WIDTH)
    return level - LIP_LEVEL - CORNER_LIFT * reach * reach


def _skin_colour(face: Face, position: Vector) -> Colour:
    x, height, z = position
    level = (height - EYE_HEIGHT) / (face.length if height < EYE_HEIGHT else 1.0)
    front = smoothstep(0.0, 0.03, z)
    base = -0.047 * face.nose_length
    blush = 0.5 * bump(math.hypot((abs(x) - 0.043) / 0.03, (level + 0.036) / 0.028))
    blush += 0.25 * bump(math.hypot(x / 0.012, (level - base - 0.008) / 0.012))
    blush += 0.18 * bump(math.hypot((abs(x) - EYE_SPACING) / 0.024, level / 0.015))
    nostril = 0.45 * bump(math.hypot((abs(x) - 0.0065 * face.nose_width) / 0.0036, (level - base - 0.0005) / 0.0024))
    colour = mixed(face.skin, face.blush, front * min(1.0, blush))
    colour = mixed(colour, face.lips, front * _lip_cover(x, level))
    return mixed(colour, NOSTRIL, front * nostril)


def _lip_cover(x: float, level: float) -> float:
    """How much of a place the lips colour: an upper lip with a bow, a fuller lower lip, both ending at the corners."""
    reach = abs(x) / MOUTH_HALF_WIDTH
    if reach >= 1.0:
        return 0.0
    above = _above_lip_line(x, level)
    fade = 1.0 - smoothstep(0.8, 1.0, reach)
    bow = 0.0072 * (1.0 - reach * reach) ** 0.6 * (1.0 - 0.22 * bump(x / 0.006))
    bottom = 0.0098 * (1.0 - reach * reach) ** 0.75
    edge = 0.002
    if above >= 0.0:
        return fade * (1.0 - smoothstep(bow - edge, bow + edge, above))
    return fade * (1.0 - smoothstep(bottom - edge, bottom + edge, -above))
