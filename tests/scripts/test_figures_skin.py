import pytest

from figures.figure_parts import Material, chain

SIDES = [("left", 1), ("right", -1)]
# glTF stores the whole of a normalized unsigned byte as 255.
WHOLE = 255
# Around a joint the skin eases from one bone to the next; this far past the wrist the hand carries it alone.
PAST_THE_WRIST = 0.02


def carried_vertices(figure, mesh_name):
    """Per vertex of a mesh: its material, its position and the weight of each bone's node that carries it."""
    joints = figure.document["skins"][0]["joints"]
    for primitive in figure.primitives(mesh_name):
        attributes = primitive["attributes"]
        rows = zip(
            figure.accessor(attributes["POSITION"]),
            figure.accessor(attributes["JOINTS_0"]),
            figure.accessor(attributes["WEIGHTS_0"]),
            strict=True,
        )
        for position, joint_row, weight_row in rows:
            carriers = {joints[joint]: weight for joint, weight in zip(joint_row, weight_row, strict=True) if weight}
            yield figure.material_name(primitive), position, carriers


def bone_node(figure, bone):
    return figure.vrm()["humanoid"]["humanBones"][bone]["node"]


@pytest.mark.parametrize("mesh_name", ["body", "face"])
def test_every_vertex_is_carried_whole_by_bones_of_the_skeleton(figure, mesh_name):
    bones = {bone["node"] for bone in figure.vrm()["humanoid"]["humanBones"].values()}

    for _, _, carriers in carried_vertices(figure, mesh_name):
        assert set(carriers) <= bones
        assert sum(carriers.values()) == WHOLE


@pytest.mark.parametrize(("side", "direction"), SIDES)
def test_each_eyeball_turns_with_the_eye_bone_of_its_side(figure, side, direction):
    eyeball = [
        carriers
        for material, position, carriers in carried_vertices(figure, "body")
        if material == Material.EYE and position[0] * direction > 0
    ]

    assert eyeball
    assert all(carriers == {bone_node(figure, f"{side}Eye"): WHOLE} for carriers in eyeball)


@pytest.mark.parametrize(("side", "direction"), SIDES)
def test_each_hand_reaches_past_its_wrist_and_follows_its_hand_bone(figure, side, direction):
    wrist = figure.bone_positions()[f"{side}Hand"][0] * direction
    hand = [
        carriers
        for _, position, carriers in carried_vertices(figure, "body")
        if position[0] * direction > wrist + PAST_THE_WRIST
    ]

    assert hand
    assert all(carriers == {bone_node(figure, f"{side}Hand"): WHOLE} for carriers in hand)


@pytest.mark.parametrize(
    ("height", "carriers"),
    [
        (0.5, {"hips": 1.0}),
        (1.0, {"spine": 0.5, "hips": 0.5}),
        (1.5, {"spine": 1.0}),
        (2.0, {"chest": 0.5, "spine": 0.5}),
        (2.5, {"chest": 1.0}),
    ],
)
def test_a_chain_hands_a_place_from_bone_to_bone_along_its_axis(height, carriers):
    binder = chain(1, "hips", (("spine", 1.0, 0.1), ("chest", 2.0, 0.1)))

    assert dict(binder((0.0, height, 0.0))) == pytest.approx(carriers)


@pytest.mark.parametrize(("distance", "carrier"), [(0.0, "chest"), (-0.5, "rightUpperArm")])
def test_a_chain_that_runs_the_other_way_hands_over_towards_the_negative_side(distance, carrier):
    binder = chain(0, "chest", (("rightUpperArm", -0.2, 0.05),), -1.0)

    assert dict(binder((distance, 0.0, 0.0))) == {carrier: 1.0}
