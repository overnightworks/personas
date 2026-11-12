"""Writes the VRM figure of every persona into a directory: python3 scripts/write_figures.py <directory>."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from figures.humanoid_figure import Wardrobe, build_figure
from figures.persona_wardrobes import LEONARDO, MOZART

if TYPE_CHECKING:
    from collections.abc import Sequence

WARDROBES: tuple[Wardrobe, ...] = (MOZART, LEONARDO)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the VRM figure of every persona.")
    parser.add_argument("directory", type=Path, help="the directory the <name>.vrm files are written to")
    directory: Path = parser.parse_args(arguments).directory
    directory.mkdir(parents=True, exist_ok=True)
    for wardrobe in WARDROBES:
        (directory / f"{wardrobe.name.lower()}.vrm").write_bytes(build_figure(wardrobe))


if __name__ == "__main__":
    main()
