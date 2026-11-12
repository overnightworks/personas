import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "figure_release_notes.sh"
VERSION = "2026.11.12.1"
FIGURE_LINE = re.compile(r'^(?P<name>\w+): figure = \{ url = "(?P<url>[^"]+)", sha256 = "(?P<sha256>[0-9a-f]{64})" \}$')


def run_in(tree: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - the shipped script under test, at a path this module computes
        [str(tree / "scripts" / SCRIPT.name), *arguments], capture_output=True, text=True, check=False
    )


def tree_with(root: Path, figures: dict[str, bytes]) -> Path:
    (root / "scripts").mkdir()
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    characters = root / "src" / "characters"
    characters.mkdir(parents=True)
    for name, content in figures.items():
        (characters / name).write_bytes(content)
    return root


def test_the_notes_hold_one_line_per_figure_in_name_order_with_its_address_and_digest(tmp_path):
    figures = {"zeta.vrm": b"the bytes of zeta", "alpha.vrm": b"the bytes of alpha", "alpha.md": b"not a figure"}
    tree = tree_with(tmp_path, figures)

    result = run_in(tree, VERSION)

    lines = [FIGURE_LINE.fullmatch(line) for line in result.stdout.splitlines() if "figure =" in line]
    assert result.returncode == 0
    assert all(lines)
    assert [
        (line["name"], line["url"].endswith(f"/v{VERSION}/{line['name']}.vrm"), line["sha256"])
        for line in lines
        if line
    ] == [(name, True, hashlib.sha256(figures[f"{name}.vrm"]).hexdigest()) for name in ("alpha", "zeta")]


def test_the_notes_open_with_a_heading(tmp_path):
    tree = tree_with(tmp_path, {"alpha.vrm": b"the bytes of alpha"})

    result = run_in(tree, VERSION)

    assert result.stdout.startswith("#")


def test_the_shipped_figures_are_named_with_their_tracked_digests():
    root = SCRIPT.parents[1]

    result = subprocess.run(  # noqa: S603 - the shipped script under test
        [str(SCRIPT), VERSION], capture_output=True, text=True, check=False
    )

    tracked = {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "src/characters").glob("*.vrm")
    }
    named = {
        match["name"]: match["sha256"] for line in result.stdout.splitlines() if (match := FIGURE_LINE.fullmatch(line))
    }
    assert named == tracked
    assert tracked


def test_a_missing_version_is_refused_and_named(tmp_path):
    tree = tree_with(tmp_path, {"alpha.vrm": b"the bytes of alpha"})

    result = run_in(tree)

    assert result.returncode != 0
    assert "version" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("figures", [{}, {"alpha.md": b"a character, not a figure"}])
def test_a_tree_without_a_figure_file_is_refused_and_named(tmp_path, figures):
    tree = tree_with(tmp_path, figures)

    result = run_in(tree, VERSION)

    assert result.returncode != 0
    assert "figure" in result.stderr
    assert not result.stdout
