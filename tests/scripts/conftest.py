import pytest

from figures.humanoid_figure import build_figure
from figures.persona_wardrobes import LEONARDO, MOZART
from figures.room_glb import build_room
from tests.support.glb import Glb, read_glb
from write_figures import ROOMS

PERSONAS = {wardrobe.name: wardrobe for wardrobe in (MOZART, LEONARDO)}
ROOMS_BY_PERSONA = {room.name: room for room in ROOMS}


@pytest.fixture(scope="session")
def figure_bytes() -> dict[str, bytes]:
    return {name: build_figure(wardrobe) for name, wardrobe in PERSONAS.items()}


@pytest.fixture(params=sorted(PERSONAS))
def figure(request: pytest.FixtureRequest, figure_bytes: dict[str, bytes]) -> Glb:
    return read_glb(figure_bytes[request.param])


@pytest.fixture(scope="session", params=sorted(ROOMS_BY_PERSONA))
def room(request: pytest.FixtureRequest) -> Glb:
    return read_glb(build_room(ROOMS_BY_PERSONA[request.param]))
