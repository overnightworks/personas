import pytest

from figures.figure_expressions import Preset
from figures.figure_parts import Material
from figures.persona_wardrobes import LEONARDO
from tests.support.glb import read_glb

MOUTH_PRESETS = [Preset.AA, Preset.IH, Preset.OU, Preset.EE, Preset.OH]
EYES = ["leftEye", "rightEye"]
# How far from an eye's middle a lid still lies in front of its pupil.
PUPIL_HALF_WIDTH = 0.002


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
def test_a_mouth_preset_moves_only_what_lies_below_the_eyes(figure, preset):
    eye_height = figure.bone_positions()["leftEye"][1]
    moved = moves(figure, preset)

    assert Material.SKIN in moved
    assert max(rest[1] for pairs in moved.values() for rest, _ in pairs) < eye_height


def test_no_two_mouth_presets_give_the_same_shape(figure):
    shapes = [tuple(delta for _, delta in target_vertices(figure, preset)[Material.SKIN]) for preset in MOUTH_PRESETS]

    assert len(set(shapes)) == len(MOUTH_PRESETS)


@pytest.mark.parametrize("eye", EYES)
def test_blink_moves_the_lids_alone_and_draws_them_down_past_the_line_of_sight(figure, eye):
    middle, line_of_sight, _ = figure.bone_positions()[eye]
    moved = moves(figure, Preset.BLINK)
    before_the_pupil = [
        (rest, delta) for rest, delta in moved[Material.EYELID] if abs(rest[0] - middle) < PUPIL_HALF_WIDTH
    ]

    assert set(moved) == {Material.EYELID}
    assert min(rest[1] for rest, _ in before_the_pupil) > line_of_sight
    assert min(rest[1] + delta[1] for rest, delta in before_the_pupil) < line_of_sight


def test_a_smile_raises_the_lids_a_blink_leaves_at_rest(figure):
    blinking = target_vertices(figure, Preset.BLINK)[Material.EYELID]
    smiling = target_vertices(figure, Preset.HAPPY)[Material.EYELID]
    lower_lids = [
        smile for (_, blink), (_, smile) in zip(blinking, smiling, strict=True) if any(smile) and not any(blink)
    ]

    assert lower_lids
    assert min(delta[1] for delta in lower_lids) > 0


def test_leonardos_beard_drops_as_far_as_his_chin(figure_bytes):
    open_mouth = moves(read_glb(figure_bytes[LEONARDO.name]), Preset.AA)
    drop = {material: min(delta[1] for _, delta in open_mouth[material]) for material in open_mouth}

    assert drop[Material.SKIN] < 0
    assert drop[Material.WHISKERS] == pytest.approx(drop[Material.SKIN])
