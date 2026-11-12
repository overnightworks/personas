"""Reads a GLB file the way a consumer would: the JSON chunk, the binary chunk and typed accessors."""

import json
import struct
from dataclasses import dataclass
from typing import Any

WIDTHS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}
FORMATS = {5126: "f", 5123: "H", 5121: "B"}


@dataclass(frozen=True)
class Glb:
    document: dict[str, Any]
    binary: bytes

    def accessor(self, index: int) -> list[tuple[float, ...]]:
        accessor = self.document["accessors"][index]
        view = self.document["bufferViews"][accessor["bufferView"]]
        width = WIDTHS[accessor["type"]]
        values = struct.unpack_from(
            "<" + FORMATS[accessor["componentType"]] * (accessor["count"] * width), self.binary, view["byteOffset"]
        )
        return [values[start : start + width] for start in range(0, len(values), width)]

    def vrm(self) -> dict[str, Any]:
        return self.document["extensions"]["VRMC_vrm"]

    def bone_positions(self) -> dict[str, tuple[float, float, float]]:
        """The world position of every humanoid bone, from the nodes' translations down the hierarchy."""
        nodes = self.document["nodes"]
        parents = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}

        def world(index: int) -> tuple[float, float, float]:
            x, y, z = nodes[index]["translation"]
            if index not in parents:
                return (x, y, z)
            parent_x, parent_y, parent_z = world(parents[index])
            return (parent_x + x, parent_y + y, parent_z + z)

        return {name: world(bone["node"]) for name, bone in self.vrm()["humanoid"]["humanBones"].items()}

    def primitives(self, mesh_name: str) -> list[dict[str, Any]]:
        mesh = next(mesh for mesh in self.document["meshes"] if mesh["name"] == mesh_name)
        return mesh["primitives"]

    def material_name(self, primitive: dict[str, Any]) -> str:
        return self.document["materials"][primitive["material"]]["name"]


def read_glb(data: bytes) -> Glb:
    json_length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + json_length])
    return Glb(document, data[20 + json_length + 8 :])
