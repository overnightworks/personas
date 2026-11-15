from dataclasses import replace

import pytest

from figures.figure_shapes import Shape
from figures.room_light import Block, Light, Window, bake
from figures.room_shapes import Piece

GREY = (200, 200, 200)
DARK = (0.0, 0.0, 0.0)
BRIGHT = (1.0, 1.0, 1.0)
# A level sun travelling toward -Z, in through the window of the wall at +Z.
LEVEL_SUN = (0.0, 0.0, -1.0)


def window_light(sun: tuple[float, float, float]) -> Light:
    return Light(
        sun=sun,
        sun_colour=BRIGHT,
        sky_colour=(0.1, 0.1, 0.1),
        bounce_colour=(0.1, 0.1, 0.1),
        ambient=DARK,
        windows=(Window(centre=(0.0, 1.0, 2.0), inward=(0.0, 0.0, -1.0), half_size=(1.0, 1.0, 0.0), depth=0.1),),
        blocks=(Block(centre=(0.0, 1.0, 0.0), half_size=(0.2, 0.5, 0.2)),),
        sides=(),
        exposure=1.0,
    )


def baked_colour(position: tuple[float, float, float], normal: tuple[float, float, float], light: Light):
    surface = Piece(Shape(positions=(position,), normals=(normal,), indices=()), (GREY,))
    (lit,) = bake([surface], light)
    return lit.colours[0]


def test_a_block_between_the_window_and_a_wall_keeps_the_sun_off_it():
    light = window_light(LEVEL_SUN)
    facing_the_window = (0.0, 0.0, 1.0)

    behind_the_block = baked_colour((0.0, 1.0, -2.0), facing_the_window, light)
    beside_the_block = baked_colour((0.7, 1.0, -2.0), facing_the_window, light)

    assert all(behind < beside for behind, beside in zip(behind_the_block, beside_the_block, strict=True))


@pytest.mark.parametrize(
    ("sun", "normal", "colour"),
    [
        pytest.param((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "sun_colour", id="sun along the window's wall comes not in"),
        pytest.param(LEVEL_SUN, (0.0, 0.0, 1.0), "bounce_colour", id="a level sun throws no bounce off the floor"),
    ],
)
def test_light_that_cannot_reach_the_room_adds_nothing(sun, normal, colour):
    light = window_light(sun)
    position = (0.7, 1.0, -2.0)

    with_it = baked_colour(position, normal, replace(light, **{colour: BRIGHT}))
    without_it = baked_colour(position, normal, replace(light, **{colour: DARK}))

    assert with_it == without_it
