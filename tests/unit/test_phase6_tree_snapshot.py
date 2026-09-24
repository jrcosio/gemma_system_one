"""Referencia del árbol completo sin commit: determinista y sin lo que Git ignora."""

from __future__ import annotations

import importlib.util
import subprocess
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "snapshot_tree.py"


def _load():
    spec = importlib.util.spec_from_file_location("snapshot_tree", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tree_snapshot_is_deterministic_and_respects_gitignore(tmp_path):
    mod = _load()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("runs/\n")
    (repo / "a.py").write_text("x = 1\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "b.md").write_text("doc\n")
    (repo / "runs").mkdir()
    (repo / "runs" / "weights.bin").write_bytes(b"\0" * 10)
    first = mod.snapshot_tree(repo, tmp_path / "out")
    second = mod.snapshot_tree(repo, tmp_path / "out2")
    assert first["sha256"] == second["sha256"] and first["files"] == 3
    archive = tmp_path / "out" / f"{first['sha256']}.tar"
    assert archive.read_bytes() == (tmp_path / "out2" / f"{first['sha256']}.tar").read_bytes()
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == [".gitignore", "a.py", "docs/b.md"]
    (repo / "a.py").write_text("x = 2\n")
    assert mod.snapshot_tree(repo, tmp_path / "out")["sha256"] != first["sha256"]
