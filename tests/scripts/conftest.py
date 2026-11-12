import pytest

from figures.humanoid_figure import build_figure
from figures.persona_wardrobes import LEONARDO, MOZART
from tests.support.glb import Glb, read_glb

PERSONAS = {wardrobe.name: wardrobe for wardrobe in (MOZART, LEONARDO)}


@pytest.fixture(scope="session")
def figure_bytes() -> dict[str, bytes]:
    return {name: build_figure(wardrobe) for name, wardrobe in PERSONAS.items()}


@pytest.fixture(params=sorted(PERSONAS))
def figure(request: pytest.FixtureRequest, figure_bytes: dict[str, bytes]) -> Glb:
    return read_glb(figure_bytes[request.param])
