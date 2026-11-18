"""Runs a shipped script from a tree a test built, the way CI runs it from the checkout."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_from(tree: Path, script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Runs the copy of ``script`` under the tree's ``scripts`` directory with ``arguments``."""
    return subprocess.run(  # noqa: S603 - a shipped script under test, at a path the test computes
        [str(tree / "scripts" / script.name), *arguments], capture_output=True, text=True, check=False
    )
