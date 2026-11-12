import math

import pytest

from figures.figure_shapes import Shape, Vector, cross, dot, ellipsoid, subtract, tapered_tube

LENGTH = 0.5
START_RADIUS = 0.1
END_RADIUS = 0.05
AXES = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
ORIGIN = (0.2, 0.3, 0.4)
# The rings are polygons, so a diameter falls short of the exact one by the cosine of half a segment.
POLYGON = 0.02
DIAGONAL = (0.3, 0.4, 0.5)


def tube_along(direction: Vector, depth: float = 1.0) -> Shape:
    end = tuple(origin + LENGTH * component for origin, component in zip(ORIGIN, direction, strict=True))
    return tapered_tube(ORIGIN, end, START_RADIUS, END_RADIUS, depth)  # type: ignore[arg-type]


def extent(shape: Shape, axis: int) -> float:
    values = [position[axis] for position in shape.positions]
    return max(values) - min(values)


@pytest.mark.parametrize("direction_name", AXES)
def test_a_tube_spans_its_length_along_its_axis_and_its_diameter_across(direction_name):
    direction = AXES[direction_name]
    shape = tube_along(direction)

    for axis, component in enumerate(direction):
        expected = LENGTH if component else 2 * START_RADIUS
        assert extent(shape, axis) == pytest.approx(expected, rel=POLYGON)


def test_a_flattened_tube_is_thinner_front_to_back_than_across():
    shape = tube_along(AXES["y"], depth=0.5)

    assert extent(shape, 2) == pytest.approx(START_RADIUS, rel=POLYGON)
    assert extent(shape, 0) == pytest.approx(2 * START_RADIUS, rel=POLYGON)


def test_an_ellipsoid_spans_its_radii_around_its_centre():
    radii = (0.05, 0.08, 0.02)
    shape = ellipsoid(ORIGIN, radii)

    for axis in range(3):
        assert extent(shape, axis) == pytest.approx(2 * radii[axis], rel=POLYGON)
        assert min(position[axis] for position in shape.positions) == pytest.approx(
            ORIGIN[axis] - radii[axis], rel=POLYGON
        )


@pytest.mark.parametrize(
    "shape",
    [
        *(tube_along(direction) for direction in AXES.values()),
        tube_along(AXES["y"], depth=0.6),
        tube_along(DIAGONAL),
        ellipsoid(ORIGIN, (0.05, 0.08, 0.02)),
    ],
    ids=[*AXES, "flattened", "diagonal", "ellipsoid"],
)
def test_every_triangle_winds_outward_where_its_vertex_normals_point(shape):
    triangles = list(zip(*[iter(shape.indices)] * 3, strict=True))
    checked = 0

    for first, second, third in triangles:
        winding = cross(
            subtract(shape.positions[second], shape.positions[first]),
            subtract(shape.positions[third], shape.positions[first]),
        )
        if math.sqrt(dot(winding, winding)) < 1e-9:
            continue
        mean_normal = tuple(sum(shape.normals[index][axis] for index in (first, second, third)) for axis in range(3))
        assert dot(winding, mean_normal) > 0  # type: ignore[arg-type]
        checked += 1

    assert checked > len(triangles) / 2
