import pytest

from figures.figure_shapes import ellipsoid
from tests.support.surface import facing

CENTRE = (0.2, 0.3, 0.4)
RADII = (0.05, 0.08, 0.02)
# The bands are polygons, so a diameter falls short of the exact one by the cosine of half a segment.
POLYGON = 0.02


@pytest.fixture
def ball():
    return ellipsoid(CENTRE, RADII, 12, 18)


def test_an_ellipsoid_spans_its_radii_around_its_centre(ball):
    for axis in range(3):
        along = [position[axis] for position in ball.positions]

        assert max(along) - min(along) == pytest.approx(2 * RADII[axis], rel=POLYGON)
        assert min(along) == pytest.approx(CENTRE[axis] - RADII[axis], rel=POLYGON)


def test_every_triangle_of_an_ellipsoid_winds_outward_where_its_vertex_normals_point(ball):
    agreements = facing(ball.positions, ball.normals, ball.indices)

    assert len(agreements) > len(ball.indices) / 6
    assert min(agreements) > 0
