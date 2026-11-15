"""The light baked into a room's vertex colours: sun and sky through its windows, bounce, and shade in its corners.

A room's material is unlit, so these colours are what every device shows; they hold linear light from 0 to 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.figure_shapes import add, dot, multiply, normalised, scaled, subtract
from figures.figure_surfaces import smoothstep
from figures.glb_container import linear
from figures.room_shapes import turned

if TYPE_CHECKING:
    from collections.abc import Sequence

    from figures.figure_parts import Colour
    from figures.figure_shapes import Shape, Vector
    from figures.room_shapes import Piece

# How far a shadow's edge blurs for each metre the light travels past what casts it, and its least blur.
PENUMBRA = 0.035
LEAST_PENUMBRA = 0.06
# A ray toward the sun starts this far off its surface, so a surface does not shade itself.
RAY_START = 0.01
# How dark a corner gets at its deepest, and over what distance the shade fades out of it.
CORNER_DEPTH = 0.4
CORNER_REACH = 0.3
# How dark the floor gets under a block standing on it, and how far the shade reaches past its edge.
CONTACT_DEPTH = 0.45
CONTACT_REACH = 0.22
FLOOR_FACING = 0.7
# Keeps the sky's and the bounce's falloff finite at the window itself.
NEAR = 0.5
SUN_SHADES = (0.85, 1.0, 1.15)


@dataclass(frozen=True)
class Window:
    """An upright rectangle in a wall the light comes in through: its middle, the unit axis into the room, its
    half sizes along the axes (none along `inward`) and how thick the wall is."""

    centre: Vector
    inward: Vector
    half_size: Vector
    depth: float

    @property
    def area(self) -> float:
        sizes = [2.0 * half for half in self.half_size if half > 0.0]
        return sizes[0] * sizes[1]


@dataclass(frozen=True)
class Block:
    """A box that casts a shadow and shades the floor under it: its middle, half sizes and turn about the vertical."""

    centre: Vector
    half_size: Vector
    turn: float = 0.0

    def local(self, point: Vector) -> Vector:
        """The point in the block's own frame, its middle at the origin and its sides along the axes."""
        return turned(subtract(point, self.centre), -self.turn)


@dataclass(frozen=True)
class Plane:
    """A flat side of the room, floor, ceiling or wall: a point on it and its unit normal into the room."""

    point: Vector
    normal: Vector


@dataclass(frozen=True)
class Light:
    """One room's light: the direction the sunlight travels, the linear colours of sun, sky, bounce and the light
    that fills the room everywhere, the windows, what casts shadows, the room's sides, and the exposure that maps
    light to colour."""

    sun: Vector
    sun_colour: Vector
    sky_colour: Vector
    bounce_colour: Vector
    ambient: Vector
    windows: tuple[Window, ...]
    blocks: tuple[Block, ...]
    sides: tuple[Plane, ...]
    exposure: float


@dataclass(frozen=True)
class Lit:
    """A surface with the light baked into it: its colour at each vertex, linear from 0 to 1."""

    shape: Shape
    colours: tuple[Vector, ...]


def bake(pieces: Sequence[Piece], light: Light) -> list[Lit]:
    """Every piece with the light that falls on each vertex, or its own colour where it glows."""
    return [
        Lit(
            each.shape,
            tuple(
                _linear(colour) if each.glows else _shown(_linear(colour), _arriving(position, normal, light), light)
                for position, normal, colour in zip(each.shape.positions, each.shape.normals, each.colours, strict=True)
            ),
        )
        for each in pieces
    ]


def _linear(colour: Colour) -> Vector:
    return (linear(colour[0]), linear(colour[1]), linear(colour[2]))


def _shown(albedo: Vector, arriving: Vector, light: Light) -> Vector:
    """The colour a surface shows: what it reflects, rolled off softly toward white instead of clipping."""
    red, green, blue = multiply(albedo, arriving)
    return (_rolled_off(red, light), _rolled_off(green, light), _rolled_off(blue, light))


def _rolled_off(value: float, light: Light) -> float:
    return 1.0 - math.exp(-light.exposure * value)


def _arriving(position: Vector, normal: Vector, light: Light) -> Vector:
    toward_sun = scaled(light.sun, -1.0)
    facing_sun = max(0.0, dot(normal, toward_sun))
    sunlit = facing_sun * _sun_reaches(add(position, scaled(normal, RAY_START)), light) if facing_sun > 0.0 else 0.0
    total = add(light.ambient, scaled(light.sun_colour, sunlit))
    for window in light.windows:
        total = add(total, scaled(light.sky_colour, _sky_through(window, position, normal)))
        total = add(total, scaled(light.bounce_colour, _bounce_from(window, position, light)))
    return scaled(total, _corner_shade(position, normal, light) * _contact_shade(position, normal, light))


def _sun_reaches(start: Vector, light: Light) -> float:
    """How much of the sun a point sees: through a window's opening, and past every block."""
    toward_sun = scaled(light.sun, -1.0)
    through = max((_through(window, start, toward_sun) for window in light.windows), default=0.0)
    if through <= 0.0:
        return 0.0
    for block in light.blocks:
        through *= 1.0 - sum(_hits(block, start, toward_sun, scale) for scale in SUN_SHADES) / len(SUN_SHADES)
    return through


def _through(window: Window, start: Vector, direction: Vector) -> float:
    """How far the ray from `start` passes inside the window at both faces of the wall, 0 to 1, edges blurred."""
    outward = -dot(direction, window.inward)
    if outward <= 0.0:
        return 0.0
    share = 1.0
    for depth in (0.0, window.depth):
        face = subtract(window.centre, scaled(window.inward, depth))
        travel = dot(subtract(start, face), window.inward) / outward
        if travel <= 0.0:
            return 0.0
        offset = subtract(add(start, scaled(direction, travel)), face)
        blur = max(LEAST_PENUMBRA, PENUMBRA * travel)
        for axis in range(3):
            half = window.half_size[axis]
            if half > 0.0:
                share *= smoothstep(-blur, blur, half - abs(offset[axis]))
    return share


def _hits(block: Block, start: Vector, direction: Vector, scale: float) -> float:
    """1 where the ray from `start` passes through the block grown or shrunk by `scale`, else 0."""
    origin, heading = block.local(start), turned(direction, -block.turn)
    nearest, farthest = 0.0, math.inf
    for axis in range(3):
        half = block.half_size[axis] * scale
        if abs(heading[axis]) < 1e-12:
            if abs(origin[axis]) > half:
                return 0.0
            continue
        first, second = sorted(((-half - origin[axis]) / heading[axis], (half - origin[axis]) / heading[axis]))
        nearest, farthest = max(nearest, first), min(farthest, second)
    return 1.0 if nearest <= farthest else 0.0


def _sky_through(window: Window, position: Vector, normal: Vector) -> float:
    """The light of the sky through the window: an area light, falling off with the square of the distance."""
    reach = subtract(window.centre, position)
    distance_squared = dot(reach, reach)
    toward = normalised(reach)
    facing = max(0.0, dot(normal, toward)) * max(0.0, -dot(toward, window.inward))
    return window.area * facing / (math.pi * (distance_squared + NEAR))


def _bounce_from(window: Window, position: Vector, light: Light) -> float:
    """The sunlight thrown back from the patch the window's sun lays on the floor, softly in every direction."""
    if light.sun[1] >= 0.0:
        return 0.0
    patch = add(window.centre, scaled(light.sun, window.centre[1] / -light.sun[1]))
    reach = subtract(patch, position)
    return window.area / (math.pi * (dot(reach, reach) + NEAR))


def _corner_shade(position: Vector, normal: Vector, light: Light) -> float:
    """Darker where the surface meets another side of the room at an angle: along the skirting, in the corners."""
    shade = 1.0
    for side in light.sides:
        across = 1.0 - abs(dot(normal, side.normal))
        distance = max(0.0, dot(subtract(position, side.point), side.normal))
        shade *= 1.0 - CORNER_DEPTH * across * math.exp(-distance / CORNER_REACH)
    return shade


def _contact_shade(position: Vector, normal: Vector, light: Light) -> float:
    """Darker on the floor under and around what stands on it."""
    if normal[1] < FLOOR_FACING:
        return 1.0
    shade = 1.0
    for block in light.blocks:
        x, _, z = block.local(position)
        outside = math.hypot(max(0.0, abs(x) - block.half_size[0]), max(0.0, abs(z) - block.half_size[2]))
        height = max(0.0, block.centre[1] - block.half_size[1] - position[1])
        shade *= 1.0 - CONTACT_DEPTH * math.exp(-outside / CONTACT_REACH) / (1.0 + height)
    return shade
