"""The wardrobes of the two personas: the faces, hair and clothes that tell Mozart and Leonardo apart."""

from __future__ import annotations

from typing import TYPE_CHECKING

from figures.figure_head import Face
from figures.humanoid_figure import Wardrobe
from figures.leonardo_hair import leonardo_hair
from figures.leonardo_robe import leonardo_clothes
from figures.mozart_coat import mozart_clothes
from figures.mozart_wig import mozart_wig

if TYPE_CHECKING:
    from figures.figure_parts import Part


def _dress_mozart(face: Face) -> list[Part]:
    return [*mozart_clothes(face), *mozart_wig()]


def _dress_leonardo(face: Face) -> list[Part]:
    return [*leonardo_clothes(face), *leonardo_hair(face)]


MOZART = Wardrobe(
    name="Mozart",
    face=Face(
        skin=(246, 222, 206),
        blush=(232, 160, 150),
        lips=(190, 105, 105),
        iris=(88, 118, 150),
        brow=(168, 138, 110),
        brow_weight=0.72,
        length=0.97,
        jaw=0.96,
        nose_length=1.05,
        nose_height=0.024,
        nose_width=0.95,
        brow_ridge=0.003,
        eye_set=0.0055,
        cheek=0.003,
    ),
    dress=_dress_mozart,
)
LEONARDO = Wardrobe(
    name="Leonardo",
    face=Face(
        skin=(224, 190, 165),
        blush=(200, 140, 125),
        lips=(160, 105, 100),
        iris=(110, 85, 60),
        brow=(215, 212, 205),
        brow_weight=1.15,
        length=1.06,
        jaw=1.0,
        nose_length=1.12,
        nose_height=0.026,
        nose_width=1.05,
        brow_ridge=0.005,
        eye_set=0.0068,
        cheek=0.0005,
    ),
    dress=_dress_leonardo,
)
