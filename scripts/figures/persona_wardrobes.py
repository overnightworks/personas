"""The wardrobes of the personas: the faces, hair and clothes that tell Mozart, Leonardo, Humboldt, Curie and Lovelace
apart."""

from __future__ import annotations

from typing import TYPE_CHECKING

from figures.curie_dress import curie_clothes
from figures.curie_hair import curie_hair
from figures.figure_head import Face
from figures.humanoid_figure import Wardrobe
from figures.humboldt_coat import humboldt_clothes
from figures.humboldt_hair import humboldt_hair
from figures.leonardo_hair import leonardo_hair
from figures.leonardo_robe import leonardo_clothes
from figures.lovelace_dress import lovelace_clothes
from figures.lovelace_hair import lovelace_hair
from figures.mozart_coat import mozart_clothes
from figures.mozart_wig import mozart_wig

if TYPE_CHECKING:
    from figures.figure_parts import Part


def _dress_curie(face: Face) -> list[Part]:
    return [*curie_clothes(face), *curie_hair()]


def _dress_lovelace(face: Face) -> list[Part]:
    return [*lovelace_clothes(face), *lovelace_hair()]


def _dress_mozart(face: Face) -> list[Part]:
    return [*mozart_clothes(face), *mozart_wig()]


def _dress_leonardo(face: Face) -> list[Part]:
    return [*leonardo_clothes(face), *leonardo_hair(face)]


def _dress_humboldt(face: Face) -> list[Part]:
    return [*humboldt_clothes(face), *humboldt_hair()]


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
HUMBOLDT = Wardrobe(
    name="Humboldt",
    face=Face(
        skin=(222, 190, 166),
        blush=(204, 144, 128),
        lips=(170, 110, 104),
        iris=(96, 112, 120),
        brow=(120, 104, 92),
        brow_weight=0.95,
        length=1.04,
        jaw=0.97,
        nose_length=1.08,
        nose_height=0.025,
        nose_width=0.98,
        brow_ridge=0.004,
        eye_set=0.0062,
        cheek=-0.001,
    ),
    dress=_dress_humboldt,
)
CURIE = Wardrobe(
    name="Curie",
    face=Face(
        skin=(236, 208, 188),
        blush=(214, 154, 142),
        lips=(176, 108, 106),
        iris=(110, 124, 128),
        brow=(112, 90, 68),
        brow_weight=0.85,
        length=1.02,
        jaw=0.94,
        nose_length=1.03,
        nose_height=0.023,
        nose_width=0.92,
        brow_ridge=0.003,
        eye_set=0.0060,
        cheek=0.001,
    ),
    dress=_dress_curie,
)
LOVELACE = Wardrobe(
    name="Lovelace",
    face=Face(
        skin=(240, 218, 204),
        blush=(222, 158, 150),
        lips=(184, 98, 102),
        iris=(58, 42, 34),
        brow=(56, 38, 30),
        brow_weight=0.9,
        length=1.0,
        jaw=0.93,
        nose_length=1.0,
        nose_height=0.022,
        nose_width=0.9,
        brow_ridge=0.002,
        eye_set=0.0062,
        cheek=0.002,
    ),
    dress=_dress_lovelace,
)
