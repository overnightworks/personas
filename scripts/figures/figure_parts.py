"""What a figure is assembled from: surfaces with a colour per vertex, a material and the bones that carry them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import TYPE_CHECKING

from figures.figure_surfaces import smoothstep

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.figure_shapes import Shape, Vector

type Colour = tuple[int, int, int]
type Binding = tuple[tuple[str, float], ...]
type Binder = Callable[[Vector], Binding]
type Paint = Colour | Callable[[Vector], Colour]


class Material(StrEnum):
    SKIN = "skin"
    EYELID = "eyelid"
    EYE = "eye"
    BROW = "brow"
    MOUTH = "mouth"
    WHISKERS = "whiskers"
    HAIR = "hair"
    CLOTH = "cloth"
    SATIN = "satin"
    LEATHER = "leather"
    GILT = "gilt"


ROUGHNESS = {
    Material.SKIN: 0.62,
    Material.EYELID: 0.62,
    Material.EYE: 0.18,
    Material.BROW: 0.8,
    Material.MOUTH: 0.5,
    Material.WHISKERS: 0.8,
    Material.HAIR: 0.78,
    Material.CLOTH: 0.92,
    Material.SATIN: 0.5,
    Material.LEATHER: 0.45,
    Material.GILT: 0.35,
}


class Layer(Enum):
    """How a part answers expressions: not at all, with the skull, with the jaw, hanging from the jaw, or as a lid."""

    BODY = "body"
    FACE = "face"
    JAW = "jaw"
    BEARD = "beard"
    UPPER_LID = "upperLid"
    LOWER_LID = "lowerLid"


@dataclass(frozen=True)
class Part:
    shape: Shape
    colours: tuple[Colour, ...]
    material: Material
    bindings: tuple[Binding, ...]
    layer: Layer = Layer.BODY
    lid_travel: tuple[Vector, ...] = ()


def part(shape: Shape, paint: Paint, material: Material, carrier: str | Binder, layer: Layer = Layer.BODY) -> Part:
    """A part painted one colour or by place, carried by one bone or by the bones its binder names by place."""
    positions = shape.positions
    colours = tuple(paint(position) for position in positions) if callable(paint) else (paint,) * len(positions)
    if callable(carrier):
        bindings = tuple(carrier(position) for position in positions)
    else:
        bindings = (((carrier, 1.0),),) * len(positions)
    return Part(shape, colours, material, bindings, layer)


def chain(axis: int, root: str, links: Sequence[tuple[str, float, float]], direction: float = 1.0) -> Binder:
    """Bones in a row along an axis: after the root, each (bone, start, softness) takes over eased around its start."""

    def binding(position: Vector) -> Binding:
        place = position[axis] * direction
        bone, weight, previous = root, 1.0, root
        for name, start, softness in links:
            share = smoothstep(start * direction - softness, start * direction + softness, place)
            if share > 0.0:
                bone, weight, previous = name, share, bone
        return ((bone, 1.0),) if weight >= 1.0 else ((bone, weight), (previous, 1.0 - weight))

    return binding


def mixed(first: Colour, second: Colour, share: float) -> Colour:
    """The colour a share of the way from the first to the second."""
    red, green, blue = (round(a + (b - a) * share) for a, b in zip(first, second, strict=True))
    return (red, green, blue)
