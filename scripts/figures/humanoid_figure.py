"""The humanoid figure as one VRM 1.0 file: the skeleton, a body mesh and a face mesh with a morph target per preset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from figures.figure_expressions import POSES, Preset, marks, morph_deltas
from figures.figure_eyes import eyes
from figures.figure_head import ears, head_skin
from figures.figure_mouth import mouth
from figures.figure_parts import ROUGHNESS, Layer, Material
from figures.figure_shapes import subtract
from figures.figure_skeleton import BONE_INDEX, BONE_POSITION, BONES
from figures.glb_container import ROUNDING, ComponentType, GlbBuilder, JsonObject

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from figures.figure_expressions import Marks
    from figures.figure_head import Face
    from figures.figure_parts import Binding, Part
    from figures.figure_shapes import Vector

LOOK_AT_RANGE = {"inputMaxValue": 90.0, "outputScale": 10.0}
FULL = 255

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


@dataclass(frozen=True)
class Wardrobe:
    """What tells one persona from another: its name, its face, and the hair and clothes it wears."""

    name: str
    face: Face
    dress: Callable[[Face], Sequence[Part]]


def build_figure(wardrobe: Wardrobe) -> bytes:
    """The VRM file of one persona."""
    builder = GlbBuilder()
    face = wardrobe.face
    parts = [*head_skin(face), *ears(face), *eyes(face), *mouth(face), *wardrobe.dress(face)]
    still = [part for part in parts if part.layer is Layer.BODY]
    moving = [part for part in parts if part.layer is not Layer.BODY]
    materials = sorted({part.material for part in parts})
    meshes = [
        {"name": "body", "primitives": _primitives(builder, still, materials, None)},
        {"name": "face", "primitives": _primitives(builder, moving, materials, marks(face))},
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
        "materials": [_material(material) for material in materials],
    }
    return builder.encode(document)


def _primitives(
    builder: GlbBuilder, parts: list[Part], materials: list[Material], places: Marks | None
) -> list[JsonObject]:
    primitives: list[JsonObject] = []
    for index, material in enumerate(materials):
        chosen = [part for part in parts if part.material == material]
        if chosen:
            primitives.append(_primitive(builder, chosen, index, places))
    return primitives


def _primitive(builder: GlbBuilder, parts: list[Part], material: int, places: Marks | None) -> JsonObject:
    positions: list[Vector] = []
    normals: list[Vector] = []
    colours: list[tuple[int, int, int, int]] = []
    bindings: list[Binding] = []
    indices: list[tuple[int]] = []
    for part in parts:
        base = len(positions)
        positions += part.shape.positions
        normals += part.shape.normals
        colours += [(_linear(red), _linear(green), _linear(blue), FULL) for red, green, blue in part.colours]
        bindings += part.bindings
        indices += [(base + index,) for index in part.shape.indices]
    byte = ComponentType.UNSIGNED_BYTE
    primitive: JsonObject = {
        "attributes": {
            "POSITION": builder.add_accessor(positions, "VEC3", ComponentType.FLOAT, bounded=True),
            "NORMAL": builder.add_accessor(normals, "VEC3", ComponentType.FLOAT),
            "COLOR_0": builder.add_accessor(colours, "VEC4", byte, normalized=True),
            "JOINTS_0": builder.add_accessor([_joints(binding) for binding in bindings], "VEC4", byte),
            "WEIGHTS_0": builder.add_accessor(
                [_weights(binding) for binding in bindings], "VEC4", byte, normalized=True
            ),
        },
        "indices": builder.add_accessor(indices, "SCALAR", ComponentType.UNSIGNED_SHORT),
        "material": material,
    }
    if places is not None:
        primitive["targets"] = [
            {
                "POSITION": builder.add_accessor(
                    _deltas(parts, preset, places), "VEC3", ComponentType.FLOAT, bounded=True
                )
            }
            for preset in Preset
        ]
    return primitive


def _deltas(parts: list[Part], preset: Preset, places: Marks) -> list[Vector]:
    return [delta for part in parts for delta in morph_deltas(part, POSES[preset], places)]


def _joints(binding: Binding) -> tuple[int, int, int, int]:
    return (BONE_INDEX[binding[0][0]], BONE_INDEX[binding[-1][0]] if len(binding) > 1 else 0, 0, 0)


def _weights(binding: Binding) -> tuple[int, int, int, int]:
    """The weights as bytes that add up to the whole, so no renderer has to normalise them."""
    first = round(binding[0][1] * FULL) if len(binding) > 1 else FULL
    return (first, FULL - first, 0, 0)


def _linear(channel: int) -> int:
    """The linear byte of an sRGB channel, as glTF's vertex colours hold it."""
    value = channel / FULL
    return round(FULL * (value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4))


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
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -x, -y, -z, 1.0)
        for x, y, z in (bone.position for bone in BONES)
    ]
    return builder.add_accessor(matrices, "MAT4", ComponentType.FLOAT)


def _material(material: Material) -> JsonObject:
    """A material that leaves the colour to the vertices and gives the surface its finish."""
    return {
        "name": material.value,
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": ROUGHNESS[material],
        },
    }


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
