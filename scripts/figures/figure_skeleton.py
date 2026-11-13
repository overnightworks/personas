"""The skeleton every figure shares: VRM's humanoid bones in the bind pose, arms out sideways, facing +Z."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from figures.figure_shapes import Vector

EYE_HEIGHT = 1.615
EYE_SPACING = 0.031
EYE_DEPTH = 0.064
SHOULDER_HEIGHT = 1.38


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str | None
    position: Vector


def _side_bones(prefix: str, side: float) -> tuple[Bone, ...]:
    return (
        Bone(prefix + "Eye", "head", (side * EYE_SPACING, EYE_HEIGHT, EYE_DEPTH)),
        Bone(prefix + "Shoulder", "chest", (side * 0.04, SHOULDER_HEIGHT, 0.0)),
        Bone(prefix + "UpperArm", prefix + "Shoulder", (side * 0.165, SHOULDER_HEIGHT, 0.0)),
        Bone(prefix + "LowerArm", prefix + "UpperArm", (side * 0.445, SHOULDER_HEIGHT, 0.0)),
        Bone(prefix + "Hand", prefix + "LowerArm", (side * 0.695, SHOULDER_HEIGHT, 0.0)),
        Bone(prefix + "UpperLeg", "hips", (side * 0.09, 0.92, 0.0)),
        Bone(prefix + "LowerLeg", prefix + "UpperLeg", (side * 0.09, 0.5, 0.0)),
        Bone(prefix + "Foot", prefix + "LowerLeg", (side * 0.09, 0.09, 0.0)),
    )


BONES = (
    Bone("hips", None, (0.0, 0.95, 0.0)),
    Bone("spine", "hips", (0.0, 1.05, 0.0)),
    Bone("chest", "spine", (0.0, 1.2, 0.0)),
    Bone("neck", "chest", (0.0, 1.42, 0.0)),
    Bone("head", "neck", (0.0, 1.5, 0.0)),
    *_side_bones("left", 1.0),
    *_side_bones("right", -1.0),
)
BONE_INDEX = {bone.name: index for index, bone in enumerate(BONES)}
BONE_POSITION = {bone.name: bone.position for bone in BONES}
