"""Copia determinista del árbol de trabajo completo, sin commit (revisión del cierre de fase 5).

A diferencia de ``snapshot_source.py`` (sólo código y configuración), incluye tests, documentación
e informes: todo lo que Git versionaría (``git ls-files --cached --others --exclude-standard``).
``data/``, ``runs/``, ``artifacts/`` y ``.venv/`` quedan fuera por ``.gitignore``.

Uso: uv run python scripts/snapshot_tree.py [directorio]   (por defecto artifacts/tree)
No sustituye a un commit; es una referencia recuperable del árbol mientras el usuario decide.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tree_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(p for p in out.decode().split("\0") if p and (root / p).is_file())


def snapshot_tree(root: Path = ROOT, out_dir: Path | None = None) -> dict[str, object]:
    files = tree_files(root)
    digest = hashlib.sha256()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in files:
            data = (root / rel).read_bytes()
            digest.update(rel.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n")
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    sha = digest.hexdigest()
    out_dir = Path(out_dir) if out_dir else root / "artifacts" / "tree"
    archive = out_dir / f"{sha}.tar"
    if not archive.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tree-", suffix=".tar")
        with os.fdopen(fd, "wb") as fh:
            fh.write(buf.getvalue())
        os.replace(tmp, archive)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    try:
        where = archive.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        where = archive.name
    return {"sha256": sha, "files": len(files), "git_head": head, "archive": where}


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(snapshot_tree(out_dir=out), indent=2))
