"""Writes every persona's VRM figure and room into a directory: python3 scripts/write_figures.py <directory>."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from figures.humanoid_figure import Wardrobe, build_figure
from figures.persona_rooms import LEONARDO as LEONARDO_ROOM
from figures.persona_rooms import MOZART as MOZART_ROOM
from figures.persona_wardrobes import LEONARDO, MOZART
from figures.room_glb import Room, build_room

if TYPE_CHECKING:
    from collections.abc import Sequence

WARDROBES: tuple[Wardrobe, ...] = (MOZART, LEONARDO)
ROOMS: tuple[Room, ...] = (MOZART_ROOM, LEONARDO_ROOM)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write every persona's VRM figure and room.")
    parser.add_argument(
        "directory", type=Path, help="the directory the <name>.vrm and <name>.room.glb files are written to"
    )
    directory: Path = parser.parse_args(arguments).directory
    directory.mkdir(parents=True, exist_ok=True)
    for wardrobe in WARDROBES:
        (directory / f"{wardrobe.name.lower()}.vrm").write_bytes(build_figure(wardrobe))
    for room in ROOMS:
        (directory / f"{room.name.lower()}.room.glb").write_bytes(build_room(room))


if __name__ == "__main__":
    main()
