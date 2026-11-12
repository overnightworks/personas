import math


def triangle_windings(glb, primitive):
    """Each triangle's geometric normal with the sum of its vertex normals; degenerate triangles are left out."""
    positions = glb.accessor(primitive["attributes"]["POSITION"])
    normals = glb.accessor(primitive["attributes"]["NORMAL"])
    indices = [row[0] for row in glb.accessor(primitive["indices"])]
    for start in range(0, len(indices), 3):
        first, second, third = indices[start : start + 3]
        edge_one = [positions[second][axis] - positions[first][axis] for axis in range(3)]
        edge_two = [positions[third][axis] - positions[first][axis] for axis in range(3)]
        winding = (
            edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
            edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
            edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
        )
        if math.sqrt(sum(component * component for component in winding)) < 1e-7:
            continue
        yield winding, [sum(normals[index][axis] for index in (first, second, third)) for axis in range(3)]


def test_every_triangle_of_every_primitive_winds_with_its_vertex_normals(figure):
    inward = 0
    checked = 0
    for mesh in figure.document["meshes"]:
        for primitive in mesh["primitives"]:
            for winding, normal in triangle_windings(figure, primitive):
                checked += 1
                inward += sum(w * n for w, n in zip(winding, normal, strict=True)) < 0

    assert checked > 1000
    assert inward == 0


def test_no_material_is_double_sided(figure):
    assert not any(material.get("doubleSided") for material in figure.document["materials"])
