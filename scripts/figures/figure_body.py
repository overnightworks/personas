"""What both figures share below the neck: tubes along the arms and legs, a trunk, hands and shoes, bound to bones."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from figures.figure_parts import Material, chain, part
from figures.figure_shapes import add, normalised, scaled
from figures.figure_skeleton import BONE_POSITION, SHOULDER_HEIGHT
from figures.figure_surfaces import arc, interpolated, loft, mirrored, oval, spread, sweep

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.figure_parts import Binder, Binding, Colour, Paint, Part
    from figures.figure_shapes import Shape, Vector
    from figures.figure_surfaces import Key

SIDES = (("left", 1.0), ("right", -1.0))
LEG_SPACING = BONE_POSITION["leftUpperLeg"][0]
ACROSS: Vector = (1.0, 0.0, 0.0)
AHEAD: Vector = (0.0, 0.0, 1.0)
DOWN: Vector = (0.0, -1.0, 0.0)
TRUNK_SEGMENTS = 40
LIMB_SEGMENTS = 16
TRUNK_POWER = 2.3
BAND_RINGS = 6
# Each finger from the forefinger to the little finger: where it leaves the palm, its length and its thickness.
FINGERS = (
    ((0.785, 0.030), 0.072, 0.0088),
    ((0.790, 0.010), 0.080, 0.0090),
    ((0.787, -0.010), 0.074, 0.0085),
    ((0.778, -0.029), 0.058, 0.0076),
)
FINGER_CURL = (8.0, 24.0, 42.0)
PALM = (
    ((0.680, SHOULDER_HEIGHT, 0.0), (0.026, 0.019)),
    ((0.715, SHOULDER_HEIGHT, 0.002), (0.036, 0.017)),
    ((0.755, SHOULDER_HEIGHT, 0.002), (0.042, 0.014)),
    ((0.792, SHOULDER_HEIGHT - 0.001, 0.001), (0.040, 0.0105)),
)
# How far forward of the ankle, then the shoe's half width, half height and the height of its middle there.
SHOE = (
    (-0.060, (0.018, 0.022, 0.030)),
    (-0.045, (0.032, 0.036, 0.038)),
    (-0.010, (0.038, 0.038, 0.039)),
    (0.040, (0.041, 0.030, 0.031)),
    (0.100, (0.043, 0.020, 0.021)),
    (0.145, (0.036, 0.015, 0.016)),
    (0.166, (0.016, 0.008, 0.010)),
)


def arm_binder(prefix: str, side: float) -> Binder:
    shoulder, elbow, wrist = (BONE_POSITION[f"left{joint}"][0] * side for joint in ("UpperArm", "LowerArm", "Hand"))
    links = ((prefix + "UpperArm", shoulder, 0.05), (prefix + "LowerArm", elbow, 0.04), (prefix + "Hand", wrist, 0.015))
    return chain(0, "chest", links, side)


def leg_binder(prefix: str, _side: float) -> Binder:
    links = ((prefix + "LowerLeg", 0.09, 0.03), (prefix + "UpperLeg", 0.5, 0.04), ("hips", 0.92, 0.04))
    return chain(1, prefix + "Foot", links)


def _half_binder(prefix: str) -> Binder:
    links = ((prefix + "UpperLeg", 0.5, 0.05), ("hips", 0.9, 0.06), ("spine", 1.05, 0.05), ("chest", 1.2, 0.05))
    return chain(1, prefix + "LowerLeg", links)


LEFT_HALF, RIGHT_HALF = _half_binder("left"), _half_binder("right")


def trunk_binder(position: Vector) -> Binding:
    """The spine's bones by height; below the hips each half of a garment follows the leg on its own side."""
    return (LEFT_HALF if position[0] >= 0.0 else RIGHT_HALF)(position)


def arm_tube(course: Sequence[Key], rings: int, mouth: Sequence[Key] = ()) -> Shape:
    """A tube along the left arm in the bind pose; the course gives its (height, depth) radii by distance from the
    midline, and the mouth adds rings after its end, which may turn back inside to hollow the opening."""
    stations = [(x, interpolated(course, x)) for x in spread(course[0][0], course[-1][0], rings)]
    rings_along = [
        oval((x, SHOULDER_HEIGHT, 0.0), DOWN, AHEAD, radii, LIMB_SEGMENTS) for x, radii in (*stations, *mouth)
    ]
    return loft(rings_along)


def leg_tube(course: Sequence[Key], rings: int) -> Shape:
    """A tube up the left leg; the course gives its (width, depth) radii and how far forward it stands, by height."""
    path: list[Vector] = []
    radii: list[tuple[float, float]] = []
    for height in spread(course[0][0], course[-1][0], rings):
        width, depth, forward = interpolated(course, height)
        path.append((LEG_SPACING, height, forward))
        radii.append((width, depth))
    return sweep(path, radii, ACROSS, LIMB_SEGMENTS)


def trunk(course: Sequence[Key], rings: int) -> Shape:
    """A closed trunk from the bottom up; the course gives its (half width, front, back) radii by height."""
    return loft([_trunk_ring(course, height, 0.0) for height in spread(course[0][0], course[-1][0], rings)])


def open_rows(course: Sequence[Key], rings: int, columns: int, grown: float = 0.0) -> list[list[Vector]]:
    """Rows around the trunk from the bottom up that stay open down the front; the course gives the (half width, front,
    back) radii and the angle from the middle at which the edges part, by height, and `grown` moves the rows outward."""
    return [
        [open_point(course, height, column / columns, grown) for column in range(columns + 1)]
        for height in spread(course[0][0], course[-1][0], rings)
    ]


def open_point(course: Sequence[Key], height: float, share: float, grown: float = 0.0) -> Vector:
    """A point on an open trunk at a height and a share of the way round from one edge, 0, to the other, 1; `course`
    and `grown` are as for `open_rows`."""
    half_width, front, back, opening = interpolated(course, height)
    radii = (half_width + grown, front + grown, back + grown)
    angle = opening + (2.0 * math.pi - 2.0 * opening) * share
    return arc((0.0, height, 0.0), ACROSS, AHEAD, radii, [angle], TRUNK_POWER)[0]


def band(course: Sequence[Key], low: float, high: float, swell: float) -> Shape:
    """A belt around a trunk between two heights, standing `swell` off it at its middle."""
    return loft(
        [
            _trunk_ring(
                course,
                low + (high - low) * step / BAND_RINGS,
                swell * (0.4 + 0.6 * math.sin(math.pi * step / BAND_RINGS)),
            )
            for step in range(BAND_RINGS + 1)
        ]
    )


def _trunk_ring(course: Sequence[Key], height: float, grown: float) -> Sequence[Vector]:
    radii = [radius + grown for radius in interpolated(course, height)]
    return oval((0.0, height, 0.0), ACROSS, AHEAD, radii, TRUNK_SEGMENTS, TRUNK_POWER)


def both_sides(shape: Shape, paint: Paint, material: Material, binder: Callable[[str, float], Binder]) -> list[Part]:
    """A part of the left side and its mirror image on the right, each carried by its own side's bones."""
    return [
        part(shape if side > 0 else mirrored(shape), paint, material, binder(prefix, side)) for prefix, side in SIDES
    ]


def hands(skin: Colour) -> list[Part]:
    """Both hands, palm down as the bind pose holds them, the fingers a little curled."""
    shapes = [sweep([place for place, _ in PALM], [radii for _, radii in PALM], AHEAD, 14, 2.5), _thumb()]
    shapes += [_finger(root, length, thickness) for root, length, thickness in FINGERS]
    return [
        part(shape if side > 0 else mirrored(shape), skin, Material.SKIN, prefix + "Hand")
        for shape in shapes
        for prefix, side in SIDES
    ]


def shoes(colour: Colour) -> list[Part]:
    path = [(LEG_SPACING, centre, forward) for forward, (_, _, centre) in SHOE]
    shape = sweep(path, [(width, height) for _, (width, height, _) in SHOE], ACROSS, 14, 2.4)
    return [
        part(shape if side > 0 else mirrored(shape), colour, Material.LEATHER, prefix + "Foot")
        for prefix, side in SIDES
    ]


def _finger(root: tuple[float, float], length: float, thickness: float) -> Shape:
    place: Vector = (root[0], SHOULDER_HEIGHT - 0.001, root[1])
    path, radii = [place], [(thickness, thickness)]
    for curl, taper in zip(FINGER_CURL, (0.95, 0.86, 0.55), strict=True):
        bend = math.radians(curl)
        place = add(place, scaled((math.cos(bend), -math.sin(bend), 0.0), length / len(FINGER_CURL)))
        path.append(place)
        radii.append((thickness * taper, thickness * taper))
    return sweep(path, radii, AHEAD, 8)


def _thumb() -> Shape:
    root: Vector = (0.712, SHOULDER_HEIGHT - 0.006, 0.028)
    knuckle = add(root, scaled(normalised((0.55, -0.25, 0.8)), 0.036))
    tip = add(knuckle, scaled(normalised((0.78, -0.3, 0.5)), 0.03))
    return sweep([root, knuckle, tip], [(0.0125, 0.0125), (0.0105, 0.0105), (0.006, 0.006)], (0.0, 1.0, 0.0), 8)
