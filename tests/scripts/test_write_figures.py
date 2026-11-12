import hashlib
from typing import TYPE_CHECKING

from write_figures import main

if TYPE_CHECKING:
    from pathlib import Path


def digests(directory: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir())}


def test_the_writer_writes_one_file_per_persona(tmp_path):
    main([str(tmp_path / "characters")])

    assert sorted(path.name for path in (tmp_path / "characters").iterdir()) == ["leonardo.vrm", "mozart.vrm"]


def test_two_runs_write_the_same_bytes(tmp_path):
    main([str(tmp_path / "first")])
    main([str(tmp_path / "second")])

    assert digests(tmp_path / "first") == digests(tmp_path / "second")


def test_a_run_over_written_files_leaves_the_same_bytes(tmp_path):
    main([str(tmp_path)])
    first = digests(tmp_path)
    main([str(tmp_path)])

    assert digests(tmp_path) == first
