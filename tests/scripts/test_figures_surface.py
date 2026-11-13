import math

import pytest

from figures.figure_shapes import dot, scaled, subtract
from figures.figure_surfaces import (
    arc,
    interpolated,
    loft,
    mirrored,
    oval,
    patch,
    seamless,
    sheet,
    shell,
    smoothstep,
    spread,
    sweep,
)
from tests.support.surface import facing

ACROSS, UP, AHEAD = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
DOWN = (0.0, -1.0, 0.0)
HEIGHTS = (0.0, 0.1, 0.2, 0.3)
CURVE = ((0.0, (1.0, 10.0)), (1.0, (3.0, 30.0)), (3.0, (2.0, 20.0)))
SEAM = 0.1


def rising_rings(middle=0.0):
    return [oval((middle, height, 0.0), ACROSS, AHEAD, (0.1, 0.1), 12) for height in HEIGHTS]


def rising_arcs(radius, angles):
    return [arc((0.0, height, 0.0), ACROSS, AHEAD, (radius, radius), angles) for height in HEIGHTS]


def normals_at_the_seam(shape):
    return {
        position: normal
        for position, normal in zip(shape.positions, shape.normals, strict=True)
        if position[1] == pytest.approx(SEAM)
    }


def outwardness(shape, along=UP, middle=0.0):
    """Per vertex off the axis through `middle` along `along`, how far its normal points away from that axis."""
    for position, normal in zip(shape.positions, shape.normals, strict=True):
        offset = subtract(position, (middle, 0.0, 0.0))
        radial = subtract(offset, scaled(along, dot(offset, along)))
        if dot(radial, radial) > 1e-12:
            yield dot(normal, radial)


def test_every_triangle_of_every_primitive_winds_with_its_vertex_normals(figure):
    agreements = [
        agreement
        for mesh in figure.document["meshes"]
        for primitive in mesh["primitives"]
        for agreement in facing(
            figure.accessor(primitive["attributes"]["POSITION"]),
            figure.accessor(primitive["attributes"]["NORMAL"]),
            [row[0] for row in figure.accessor(primitive["indices"])],
        )
    ]

    assert len(agreements) > 1000
    assert min(agreements) >= 0


def test_no_material_is_double_sided(figure):
    assert not any(material.get("doubleSided") for material in figure.document["materials"])


def test_a_loft_through_rising_rings_faces_away_from_its_axis():
    assert min(outwardness(loft(rising_rings(), capped=False))) > 0


def test_an_open_loft_keeps_one_vertex_for_each_point_of_its_rings():
    rings = rising_rings()

    assert len(loft(rings, capped=False).positions) == sum(len(ring) for ring in rings)


def test_a_capped_loft_closes_its_ends_facing_down_and_up():
    shape = loft(rising_rings())
    hubs = sorted(
        (position[1], normal[1])
        for position, normal in zip(shape.positions, shape.normals, strict=True)
        if math.hypot(position[0], position[2]) < 1e-6
    )

    assert hubs == [pytest.approx((HEIGHTS[0], -1.0)), pytest.approx((HEIGHTS[-1], 1.0))]


def test_a_capped_loft_keeps_the_edge_where_its_wall_meets_its_end():
    shape = loft(rising_rings())
    upward_on_the_top_edge = {
        round(normal[1], 6)
        for position, normal in zip(shape.positions, shape.normals, strict=True)
        if position[1] == pytest.approx(HEIGHTS[-1]) and math.hypot(position[0], position[2]) > 1e-6
    }

    assert upward_on_the_top_edge == {0.0, 1.0}


@pytest.mark.parametrize(
    ("along", "reference"),
    [(ACROSS, AHEAD), (UP, ACROSS), (DOWN, ACROSS), (AHEAD, ACROSS)],
    ids=["sideways", "up", "down", "ahead"],
)
def test_a_sweep_faces_away_from_its_middle_on_its_wall_and_its_ends_whichever_way_it_runs(along, reference):
    path = [scaled(along, distance) for distance in HEIGHTS]
    middle = scaled(along, (HEIGHTS[0] + HEIGHTS[-1]) / 2)

    shape = sweep(path, [(0.02, 0.03)] * len(path), reference, 8)

    away = [
        dot(normal, subtract(position, middle)) for position, normal in zip(shape.positions, shape.normals, strict=True)
    ]
    assert min(away) > 0


def test_a_patch_of_rising_arcs_faces_the_way_the_arcs_bulge():
    assert min(outwardness(patch(rising_arcs(0.1, spread(-1.0, 1.0, 8))))) > 0


def test_a_sheet_keeps_its_points_in_row_order_and_takes_the_normals_its_rule_gives():
    rows = rising_arcs(0.1, spread(-1.0, 1.0, 4))

    shape = sheet(rows, lambda _: AHEAD)

    assert list(shape.positions) == [point for row in rows for point in row]
    assert set(shape.normals) == {AHEAD}
    assert min(facing(shape.positions, shape.normals, shape.indices)) > 0


def test_seamless_surfaces_take_one_normal_at_each_place_they_share():
    angles = spread(-1.0, 1.0, 8)
    courses = ((0.0, 0.1), (SEAM, 0.1), (0.2, 0.15))
    rows = [arc((0.0, height, 0.0), ACROSS, AHEAD, (radius, radius), angles) for height, radius in courses]

    wall, flare = seamless([patch(rows[:2]), patch(rows[1:])])

    assert normals_at_the_seam(wall) == normals_at_the_seam(flare)
    assert max(normal[1] for normal in normals_at_the_seam(wall).values()) < 0


def test_a_shell_faces_outward_on_its_face_and_inward_on_its_lining():
    angles = spread(0.5, 2.0 * math.pi - 0.5, 12)
    face, lining = 0.1, 0.08

    face_rows, lining_rows = rising_arcs(face, angles), rising_arcs(lining, angles)
    open_edges = {row[end] for row in (*face_rows, *lining_rows) for end in (0, -1)}

    shape = shell(face_rows, lining_rows)

    within_the_edges = [
        (math.hypot(position[0], position[2]), dot(normal, (position[0], 0.0, position[2])))
        for position, normal in zip(shape.positions, shape.normals, strict=True)
        if HEIGHTS[0] < position[1] < HEIGHTS[-1] and position not in open_edges
    ]
    assert all((outward > 0) == (radius > (face + lining) / 2) for radius, outward in within_the_edges)
    assert {outward > 0 for _, outward in within_the_edges} == {True, False}


def test_a_mirrored_surface_faces_away_from_its_own_axis_and_winds_with_its_normals():
    shape = mirrored(loft(rising_rings(middle=0.5), capped=False))

    assert min(outwardness(shape, middle=-0.5)) > 0
    assert min(facing(shape.positions, shape.normals, shape.indices)) > 0


@pytest.mark.parametrize(
    ("at", "values"),
    [(-1.0, (1.0, 10.0)), (0.0, (1.0, 10.0)), (1.0, (3.0, 30.0)), (3.0, (2.0, 20.0)), (5.0, (2.0, 20.0))],
)
def test_a_curve_passes_through_its_keys_and_holds_its_ends_beyond_them(at, values):
    assert interpolated(CURVE, at) == pytest.approx(values)


def test_a_curve_runs_between_two_keys_without_a_step():
    just_before, at, just_after = (interpolated(CURVE, 1.0 + offset)[0] for offset in (-1e-6, 0.0, 1e-6))

    assert just_before == pytest.approx(at, abs=1e-4)
    assert just_after == pytest.approx(at, abs=1e-4)


@pytest.mark.parametrize(
    ("low", "high", "value", "share"),
    [(0.0, 1.0, -1.0, 0.0), (0.0, 1.0, 0.5, 0.5), (0.0, 1.0, 2.0, 1.0), (1.0, 0.0, 2.0, 0.0), (1.0, 0.0, -1.0, 1.0)],
)
def test_a_ramp_eases_from_nothing_at_its_low_end_to_all_at_its_high_end_either_way_round(low, high, value, share):
    assert smoothstep(low, high, value) == pytest.approx(share)
