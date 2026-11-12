import pytest

from figures.humanoid_figure import Preset

MOUTH_PRESETS = [Preset.AA, Preset.IH, Preset.OU, Preset.EE, Preset.OH]
EYE_MATERIALS = {"eyeWhite", "iris"}


def bound_target(figure, preset):
    binds = figure.vrm()["expressions"]["preset"][preset.value]["morphTargetBinds"]
    assert len(binds) == 1
    face_mesh = next(index for index, mesh in enumerate(figure.document["meshes"]) if mesh["name"] == "face")
    assert figure.document["nodes"][binds[0]["node"]]["mesh"] == face_mesh
    return binds[0]["index"]


def target_vertices(figure, preset):
    """Per material of the face, the (rest position, delta) of every vertex under the preset's target."""
    target = bound_target(figure, preset)
    return {
        figure.material_name(primitive): list(
            zip(
                figure.accessor(primitive["attributes"]["POSITION"]),
                figure.accessor(primitive["targets"][target]["POSITION"]),
                strict=True,
            )
        )
        for primitive in figure.primitives("face")
    }


def moves(figure, preset):
    """The same, for the vertices the target moves."""
    vertices = {
        material: [(rest, delta) for rest, delta in pairs if any(delta)]
        for material, pairs in target_vertices(figure, preset).items()
    }
    return {material: pairs for material, pairs in vertices.items() if pairs}


def test_the_figure_binds_every_preset_of_the_profile_to_a_target_of_its_own(figure):
    targets = [bound_target(figure, preset) for preset in Preset]

    assert len(set(targets)) == len(Preset)
    assert set(figure.vrm()["expressions"]["preset"]) == {preset.value for preset in Preset}


def test_every_face_primitive_carries_a_target_for_every_preset(figure):
    assert all(len(primitive["targets"]) == len(Preset) for primitive in figure.primitives("face"))


@pytest.mark.parametrize("preset", list(Preset))
def test_every_preset_moves_at_least_one_face_vertex(figure, preset):
    assert moves(figure, preset)


@pytest.mark.parametrize("preset", MOUTH_PRESETS)
def test_a_mouth_preset_moves_the_mouth_alone(figure, preset):
    assert set(moves(figure, preset)) == {"mouth"}


def test_no_two_mouth_presets_give_the_same_shape(figure):
    shapes = [tuple(delta for _, delta in moves(figure, preset)["mouth"]) for preset in MOUTH_PRESETS]

    assert len(set(shapes)) == len(MOUTH_PRESETS)


def test_blink_moves_the_eyes_alone_and_closes_them_to_a_sliver(figure):
    eye_whites = target_vertices(figure, Preset.BLINK)["eyeWhite"]

    rest_heights = [rest[1] for rest, _ in eye_whites]
    closed_heights = [rest[1] + delta[1] for rest, delta in eye_whites]
    assert set(moves(figure, Preset.BLINK)) == EYE_MATERIALS
    assert max(closed_heights) - min(closed_heights) < 0.15 * (max(rest_heights) - min(rest_heights))
