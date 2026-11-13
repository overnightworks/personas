import pytest

from figures.figure_parts import Material

PROFILE_BONES = {
    "hips", "spine", "head", "leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot",
    "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand",
}  # fmt: skip
LEVEL = 1e-6


def test_the_figure_names_every_bone_the_profile_requires(figure):
    assert set(figure.bone_positions()) >= PROFILE_BONES


def test_the_eyes_lie_in_front_of_the_head(figure):
    bones = figure.bone_positions()
    eyeballs = [
        position[2]
        for primitive in figure.primitives("body")
        if figure.material_name(primitive) == Material.EYE
        for position in figure.accessor(primitive["attributes"]["POSITION"])
    ]

    assert bones["leftEye"][2] > bones["head"][2]
    assert bones["rightEye"][2] > bones["head"][2]
    assert min(eyeballs) > bones["head"][2]


@pytest.mark.parametrize(("side", "direction"), [("left", 1), ("right", -1)])
def test_each_arm_is_held_out_sideways_at_shoulder_height(figure, side, direction):
    bones = figure.bone_positions()
    shoulder, hand = bones[f"{side}UpperArm"], bones[f"{side}Hand"]

    assert (hand[0] - shoulder[0]) * direction > 0.3
    assert abs(hand[1] - shoulder[1]) < LEVEL
    assert abs(hand[2] - shoulder[2]) < LEVEL
