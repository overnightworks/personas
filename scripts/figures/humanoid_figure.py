"""The humanoid figure: the T-posed skeleton VRM 1.0 requires, a body and a face with morph targets, as one VRM file."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from figures.figure_shapes import (
    Shape,
    Vector,
    add,
    ellipsoid,
    multiply,
    scaled,
    subtract,
    tapered_tube,
)
from figures.glb_container import ROUNDING, ComponentType, GlbBuilder, JsonObject

if TYPE_CHECKING:
    from collections.abc import Mapping

type Colour = tuple[int, int, int]


class Bone(NamedTuple):
    name: str
    parent: str | None
    position: Vector


class Preset(StrEnum):
    """The expression presets of figure's profile; their order is the order of the face's morph targets."""

    AA = "aa"
    IH = "ih"
    OU = "ou"
    EE = "ee"
    OH = "oh"
    BLINK = "blink"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    RELAXED = "relaxed"
    SURPRISED = "surprised"


class FaceFeature(StrEnum):
    """What a morph target moves; a part of no feature stays where it is."""

    NONE = "none"
    EYE = "eye"
    MOUTH = "mouth"
    LEFT_BROW = "leftBrow"
    RIGHT_BROW = "rightBrow"


@dataclass(frozen=True)
class Part:
    """A shape worn rigidly by one bone, in one material."""

    shape: Shape
    material: str
    bone: str
    face_feature: FaceFeature = FaceFeature.NONE


@dataclass(frozen=True)
class Wardrobe:
    """What tells one persona from another: its name, its colours, its clothes and its extra parts."""

    name: str
    palette: Mapping[str, Colour]
    garment: str
    legwear: str
    long_robe: bool
    brow_size: Vector
    brow_material: str
    extras: tuple[Part, ...]


@dataclass(frozen=True)
class Sculpt:
    """How a morph target moves one feature: a scale about its centre, a shift, a smile curve, a brow tilt."""

    scale: Vector = (1.0, 1.0, 1.0)
    shift: Vector = (0.0, 0.0, 0.0)
    smile: float = 0.0
    tilt: float = 0.0


# World positions in metres, +Y up, the figure faces +Z; the arms are extended along X (VRM 1.0's T-pose).
BONES = (
    Bone("hips", None, (0.0, 0.95, 0.0)),
    Bone("spine", "hips", (0.0, 1.05, 0.0)),
    Bone("chest", "spine", (0.0, 1.2, 0.0)),
    Bone("neck", "chest", (0.0, 1.42, 0.0)),
    Bone("head", "neck", (0.0, 1.5, 0.0)),
    Bone("leftEye", "head", (0.036, 1.615, 0.072)),
    Bone("rightEye", "head", (-0.036, 1.615, 0.072)),
    Bone("leftShoulder", "chest", (0.04, 1.38, 0.0)),
    Bone("leftUpperArm", "leftShoulder", (0.17, 1.38, 0.0)),
    Bone("leftLowerArm", "leftUpperArm", (0.43, 1.38, 0.0)),
    Bone("leftHand", "leftLowerArm", (0.66, 1.38, 0.0)),
    Bone("rightShoulder", "chest", (-0.04, 1.38, 0.0)),
    Bone("rightUpperArm", "rightShoulder", (-0.17, 1.38, 0.0)),
    Bone("rightLowerArm", "rightUpperArm", (-0.43, 1.38, 0.0)),
    Bone("rightHand", "rightLowerArm", (-0.66, 1.38, 0.0)),
    Bone("leftUpperLeg", "hips", (0.09, 0.92, 0.0)),
    Bone("leftLowerLeg", "leftUpperLeg", (0.09, 0.5, 0.0)),
    Bone("leftFoot", "leftLowerLeg", (0.09, 0.09, 0.0)),
    Bone("rightUpperLeg", "hips", (-0.09, 0.92, 0.0)),
    Bone("rightLowerLeg", "rightUpperLeg", (-0.09, 0.5, 0.0)),
    Bone("rightFoot", "rightLowerLeg", (-0.09, 0.09, 0.0)),
)
BONE_INDEX = {bone.name: index for index, bone in enumerate(BONES)}
BONE_POSITION = {bone.name: bone.position for bone in BONES}

SKIN = "skin"
SHOE = "shoe"
EYE_WHITE = "eyeWhite"
IRIS = "iris"
MOUTH = "mouth"
ROUGHNESS = 0.85
LOOK_AT_RANGE = {"inputMaxValue": 90.0, "outputScale": 10.0}

MORPHS: dict[Preset, dict[FaceFeature, Sculpt]] = {
    Preset.AA: {FaceFeature.MOUTH: Sculpt(scale=(0.9, 4.5, 1.2))},
    Preset.IH: {FaceFeature.MOUTH: Sculpt(scale=(1.25, 1.8, 1.0))},
    Preset.OU: {FaceFeature.MOUTH: Sculpt(scale=(0.55, 2.8, 1.3))},
    Preset.EE: {FaceFeature.MOUTH: Sculpt(scale=(1.2, 2.6, 1.0))},
    Preset.OH: {FaceFeature.MOUTH: Sculpt(scale=(0.75, 3.8, 1.2))},
    Preset.BLINK: {FaceFeature.EYE: Sculpt(scale=(1.0, 0.08, 1.0))},
    Preset.HAPPY: {
        FaceFeature.MOUTH: Sculpt(smile=0.012),
        FaceFeature.LEFT_BROW: Sculpt(shift=(0.0, 0.004, 0.0)),
        FaceFeature.RIGHT_BROW: Sculpt(shift=(0.0, 0.004, 0.0)),
    },
    Preset.ANGRY: {
        FaceFeature.MOUTH: Sculpt(smile=-0.006),
        FaceFeature.LEFT_BROW: Sculpt(tilt=-0.009),
        FaceFeature.RIGHT_BROW: Sculpt(tilt=-0.009),
    },
    Preset.SAD: {
        FaceFeature.MOUTH: Sculpt(smile=-0.01),
        FaceFeature.LEFT_BROW: Sculpt(tilt=0.008),
        FaceFeature.RIGHT_BROW: Sculpt(tilt=0.008),
    },
    Preset.RELAXED: {FaceFeature.MOUTH: Sculpt(smile=0.005), FaceFeature.EYE: Sculpt(scale=(1.0, 0.55, 1.0))},
    Preset.SURPRISED: {
        FaceFeature.MOUTH: Sculpt(scale=(0.6, 3.0, 1.2)),
        FaceFeature.LEFT_BROW: Sculpt(shift=(0.0, 0.012, 0.0)),
        FaceFeature.RIGHT_BROW: Sculpt(shift=(0.0, 0.012, 0.0)),
        FaceFeature.EYE: Sculpt(scale=(1.15, 1.15, 1.0)),
    },
}

# The licence meta carries the repository's MIT LICENSE: it grants what MIT grants and forbids nothing MIT permits.
COPYRIGHT_INFORMATION = "Copyright (c) 2026 Felix Hummert; MIT License"
META: JsonObject = {
    "authors": ["overnightworks/personas"],
    "copyrightInformation": COPYRIGHT_INFORMATION,
    "licenseUrl": "https://vrm.dev/licenses/1.0/",
    "otherLicenseUrl": "https://github.com/overnightworks/personas/blob/main/LICENSE",
    "avatarPermission": "everyone",
    "allowRedistribution": True,
    "creditNotation": "unnecessary",
    "commercialUsage": "corporation",
    "modification": "allowModificationRedistribution",
    "allowExcessivelyViolentUsage": True,
    "allowExcessivelySexualUsage": True,
    "allowPoliticalOrReligiousUsage": True,
    "allowAntisocialOrHateUsage": True,
}


def build_figure(wardrobe: Wardrobe) -> bytes:
    """The VRM file of one persona."""
    builder = GlbBuilder()
    body_parts, face_parts = _body_parts(wardrobe), _face_parts(wardrobe)
    materials = sorted({part.material for part in [*body_parts, *face_parts]})
    meshes = [
        {"name": "body", "primitives": _primitives(builder, body_parts, materials, morphing=False)},
        {"name": "face", "primitives": _primitives(builder, face_parts, materials, morphing=True)},
    ]
    nodes = [*_bone_nodes(), {"name": "bodyMesh", "mesh": 0, "skin": 0}, {"name": "faceMesh", "mesh": 1, "skin": 0}]
    body_node, face_node = len(BONES), len(BONES) + 1
    document: JsonObject = {
        "asset": {"version": "2.0", "generator": "personas figure generator"},
        "extensionsUsed": ["VRMC_vrm"],
        "extensions": {"VRMC_vrm": _vrm_extension(wardrobe.name, face_node)},
        "scene": 0,
        "scenes": [{"nodes": [0, body_node, face_node]}],
        "nodes": nodes,
        "skins": [
            {"inverseBindMatrices": _inverse_bind_matrices(builder), "skeleton": 0, "joints": list(range(len(BONES)))}
        ],
        "meshes": meshes,
        "materials": [_material(name, wardrobe.palette[name]) for name in materials],
    }
    return builder.encode(document)


def _body_parts(wardrobe: Wardrobe) -> list[Part]:
    garment, legwear = wardrobe.garment, wardrobe.legwear
    parts = [
        Part(ellipsoid((0, 1.6, 0), (0.095, 0.115, 0.1)), SKIN, "head"),
        Part(ellipsoid((0.096, 1.6, 0.0), (0.014, 0.03, 0.02)), SKIN, "head"),
        Part(ellipsoid((-0.096, 1.6, 0.0), (0.014, 0.03, 0.02)), SKIN, "head"),
        Part(ellipsoid((0, 1.585, 0.101), (0.013, 0.022, 0.017)), SKIN, "head"),
        Part(tapered_tube((0, 1.36, 0), (0, 1.52, 0), 0.045, 0.043), SKIN, "neck"),
        Part(tapered_tube((0, 0.93, 0), (0, 1.2, 0), 0.15, 0.16, 0.66), garment, "spine"),
        Part(tapered_tube((0, 1.2, 0), (0, 1.39, 0), 0.16, 0.15, 0.66), garment, "chest"),
        Part(ellipsoid((0.15, 1.38, 0), (0.07, 0.055, 0.07)), garment, "chest"),
        Part(ellipsoid((-0.15, 1.38, 0), (0.07, 0.055, 0.07)), garment, "chest"),
    ]
    for side, prefix in ((1, "left"), (-1, "right")):
        parts += _limb_parts(garment, side, prefix)
        if not wardrobe.long_robe:
            parts += [
                Part(
                    tapered_tube((side * 0.09, 0.93, 0), (side * 0.09, 0.5, 0), 0.07, 0.055),
                    garment,
                    prefix + "UpperLeg",
                ),
                Part(
                    tapered_tube((side * 0.09, 0.5, 0), (side * 0.09, 0.09, 0), 0.05, 0.04),
                    legwear,
                    prefix + "LowerLeg",
                ),
            ]
    return [*parts, *wardrobe.extras]


def _limb_parts(garment: str, side: int, prefix: str) -> list[Part]:
    return [
        Part(tapered_tube((side * 0.17, 1.38, 0), (side * 0.43, 1.38, 0), 0.05, 0.045), garment, prefix + "UpperArm"),
        Part(ellipsoid((side * 0.43, 1.38, 0), (0.046, 0.046, 0.046)), garment, prefix + "UpperArm"),
        Part(tapered_tube((side * 0.43, 1.38, 0), (side * 0.63, 1.38, 0), 0.045, 0.04), garment, prefix + "LowerArm"),
        Part(ellipsoid((side * 0.7, 1.38, 0.005), (0.05, 0.022, 0.038)), SKIN, prefix + "Hand"),
        Part(ellipsoid((side * 0.09, 0.06, 0.04), (0.045, 0.035, 0.09)), SHOE, prefix + "Foot"),
    ]


def _face_parts(wardrobe: Wardrobe) -> list[Part]:
    parts: list[Part] = []
    for side, prefix, brow in ((1, "left", FaceFeature.LEFT_BROW), (-1, "right", FaceFeature.RIGHT_BROW)):
        eye = BONE_POSITION[prefix + "Eye"]
        iris_centre = add(eye, (0.0, 0.0, 0.016))
        parts += [
            Part(ellipsoid(eye, (0.018, 0.016, 0.018), 8, 12), EYE_WHITE, prefix + "Eye", FaceFeature.EYE),
            Part(ellipsoid(iris_centre, (0.009, 0.009, 0.004), 6, 10), IRIS, prefix + "Eye", FaceFeature.EYE),
            Part(
                ellipsoid((side * 0.038, 1.648, 0.091), wardrobe.brow_size, 6, 10), wardrobe.brow_material, "head", brow
            ),
        ]
    parts.append(Part(ellipsoid((0, 1.545, 0.088), (0.026, 0.006, 0.009), 6, 12), MOUTH, "head", FaceFeature.MOUTH))
    return parts


def _primitives(builder: GlbBuilder, parts: list[Part], materials: list[str], *, morphing: bool) -> list[JsonObject]:
    primitives: list[JsonObject] = []
    for index, material in enumerate(materials):
        chosen = [part for part in parts if part.material == material]
        if chosen:
            primitives.append(_primitive(builder, chosen, index, morphing=morphing))
    return primitives


def _primitive(builder: GlbBuilder, parts: list[Part], material: int, *, morphing: bool) -> JsonObject:
    positions: list[Vector] = []
    normals: list[Vector] = []
    joints: list[tuple[int, int, int, int]] = []
    indices: list[tuple[int]] = []
    for part in parts:
        base = len(positions)
        positions += part.shape.positions
        normals += part.shape.normals
        joints += [(BONE_INDEX[part.bone], 0, 0, 0)] * len(part.shape.positions)
        indices += [(base + index,) for index in part.shape.indices]
    primitive: JsonObject = {
        "attributes": {
            "POSITION": builder.add_accessor(positions, "VEC3", ComponentType.FLOAT, bounded=True),
            "NORMAL": builder.add_accessor(normals, "VEC3", ComponentType.FLOAT),
            "JOINTS_0": builder.add_accessor(joints, "VEC4", ComponentType.UNSIGNED_BYTE),
            "WEIGHTS_0": builder.add_accessor([(1.0, 0.0, 0.0, 0.0)] * len(positions), "VEC4", ComponentType.FLOAT),
        },
        "indices": builder.add_accessor(indices, "SCALAR", ComponentType.UNSIGNED_SHORT),
        "material": material,
    }
    if morphing:
        primitive["targets"] = [
            {"POSITION": builder.add_accessor(_morph_deltas(parts, preset), "VEC3", ComponentType.FLOAT, bounded=True)}
            for preset in Preset
        ]
    return primitive


def _morph_deltas(parts: list[Part], preset: Preset) -> list[Vector]:
    deltas: list[Vector] = []
    for part in parts:
        sculpt = MORPHS[preset].get(part.face_feature)
        deltas += _sculpt_deltas(part, sculpt) if sculpt else [(0.0, 0.0, 0.0)] * len(part.shape.positions)
    return deltas


def _sculpt_deltas(part: Part, sculpt: Sculpt) -> list[Vector]:
    centre = BONE_POSITION[part.bone] if part.face_feature is FaceFeature.EYE else _centre(part.shape)
    half_width = max(abs(position[0] - centre[0]) for position in part.shape.positions) or 1.0
    deltas: list[Vector] = []
    for position in part.shape.positions:
        offset = subtract(position, centre)
        delta = add(multiply(offset, subtract(sculpt.scale, (1.0, 1.0, 1.0))), sculpt.shift)
        across = offset[0] / half_width
        lift = sculpt.smile * across * across
        if sculpt.tilt:
            inner = -across if centre[0] > 0 else across
            lift += sculpt.tilt * max(0.0, inner)
        deltas.append((delta[0], delta[1] + lift, delta[2]))
    return deltas


def _centre(shape: Shape) -> Vector:
    total = (0.0, 0.0, 0.0)
    for position in shape.positions:
        total = add(total, position)
    return scaled(total, 1.0 / len(shape.positions))


def _bone_nodes() -> list[JsonObject]:
    nodes: list[JsonObject] = []
    for bone in BONES:
        parent = BONE_POSITION[bone.parent] if bone.parent else (0.0, 0.0, 0.0)
        node: JsonObject = {
            "name": bone.name,
            "translation": [round(value, ROUNDING) for value in subtract(bone.position, parent)],
        }
        children = [index for index, child in enumerate(BONES) if child.parent == bone.name]
        if children:
            node["children"] = children
        nodes.append(node)
    return nodes


def _inverse_bind_matrices(builder: GlbBuilder) -> int:
    matrices = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -x, -y, -z, 1.0) for _, _, (x, y, z) in BONES
    ]
    return builder.add_accessor(matrices, "MAT4", ComponentType.FLOAT)


def _material(name: str, colour: Colour) -> JsonObject:
    red, green, blue = (_linear(channel) for channel in colour)
    return {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [red, green, blue, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": ROUGHNESS,
        },
    }


def _linear(channel: int) -> float:
    """The linear value of an sRGB channel, as glTF's base colour factor holds it."""
    value = channel / 255
    return round(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4, 4)


def _vrm_extension(name: str, face_node: int) -> JsonObject:
    eye, head = BONE_POSITION["leftEye"], BONE_POSITION["head"]
    return {
        "specVersion": "1.0",
        "meta": {"name": name, **META},
        "humanoid": {"humanBones": {bone.name: {"node": BONE_INDEX[bone.name]} for bone in BONES}},
        "expressions": {"preset": {preset.value: _expression(face_node, index) for index, preset in enumerate(Preset)}},
        "lookAt": {
            "type": "bone",
            "offsetFromHeadBone": [0.0, round(eye[1] - head[1], ROUNDING), round(eye[2], ROUNDING)],
            "rangeMapHorizontalInner": LOOK_AT_RANGE,
            "rangeMapHorizontalOuter": LOOK_AT_RANGE,
            "rangeMapVerticalDown": LOOK_AT_RANGE,
            "rangeMapVerticalUp": LOOK_AT_RANGE,
        },
    }


def _expression(face_node: int, target: int) -> JsonObject:
    return {
        "morphTargetBinds": [{"node": face_node, "index": target, "weight": 1.0}],
        "isBinary": False,
        "overrideBlink": "none",
        "overrideLookAt": "none",
        "overrideMouth": "none",
    }
