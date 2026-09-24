"""Manifiesto del entorno: hardware, sistema, versiones y commit del código."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = (
    "gemma-system-one",
    "torch",
    "transformers",
    "huggingface_hub",
    "hf-xet",
    "tokenizers",
    "safetensors",
    "numpy",
    "pydantic",
    "PyYAML",
    "psutil",
    "Pillow",
    "torchvision",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def sysctl(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    return _run(["sysctl", "-n", name])


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


SOURCE_GLOBS = ("src/**/*.py", "configs/*.yaml", "scripts/*.py", "pyproject.toml", "uv.lock")


def _source_files(root: Path) -> list[Path]:
    return sorted(
        {p for g in SOURCE_GLOBS for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    )


def _read_sources(root: Path) -> tuple[dict[str, object], list[tuple[str, bytes]]]:
    """Lee cada fuente una sola vez: el hash y la copia salen de los mismos bytes."""
    blobs = [(p.relative_to(root).as_posix(), p.read_bytes()) for p in _source_files(root)]
    digest = hashlib.sha256()
    for rel, data in blobs:
        digest.update(rel.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n")
    return {"sha256": digest.hexdigest(), "files": len(blobs), "globs": list(SOURCE_GLOBS)}, blobs


def source_fingerprint(root: Path = REPO_ROOT) -> dict[str, object]:
    """sha256 del código y la configuración tal como están en disco, estén o no versionados.

    Con el árbol sucio, ``git rev-parse HEAD`` no identifica el código ejecutado (hallazgo 4 de la
    revisión de fase 3); este hash sí, si se conservan los ficheros. No incluye datos ni pesos.
    """
    return _read_sources(root)[0]


SOURCE_ARCHIVE_DIR = Path("artifacts") / "source"


def snapshot_sources(root: Path = REPO_ROOT, out_dir: Path | None = None) -> dict[str, object]:
    """Hash de fuentes + copia exacta ``<out_dir>/<sha256>.tar`` (determinista; no sobrescribe).

    Revisión de fase 4 (hallazgo 5): los checkpoints registraban un hash cuya copia no se conservó.
    Desde la fase 5 cada inicio de run y cada checkpoint guardan la copia de lo que registran.
    """
    import io
    import os
    import tarfile
    import tempfile

    fp, blobs = _read_sources(root)
    if out_dir is None:
        out_dir = os.environ.get("GSO_SOURCE_ARCHIVE_DIR") or Path(root) / SOURCE_ARCHIVE_DIR
    out_dir = Path(out_dir)
    archive = out_dir / f"{fp['sha256']}.tar"
    if not archive.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel, data in blobs:
                info = tarfile.TarInfo(rel)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".src-", suffix=".tar")
        with os.fdopen(fd, "wb") as fh:
            fh.write(buf.getvalue())
        os.replace(tmp, archive)
    try:
        where = archive.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        where = mask_home(archive)
    return {**fp, "archive": where}


def git_state(root: Path = REPO_ROOT, *, snapshot: bool = False) -> dict[str, object]:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "source": snapshot_sources(root) if snapshot else source_fingerprint(root),
    }


def mask_home(path: str | Path) -> str:
    """Sustituye el HOME por '~' para no registrar el nombre de usuario en los reportes."""
    s = str(path)
    home = str(Path.home())
    return "~" + s[len(home) :] if s.startswith(home) else s


def environment_manifest() -> dict[str, object]:
    import torch

    mem = sysctl("hw.memsize")
    return {
        "machine": platform.machine(),
        "hw_model": sysctl("hw.model"),
        "cpu_brand": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(mem) if mem and mem.isdigit() else None,
        "os": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "macos_build": _run(["sw_vers", "-buildVersion"]) if sys.platform == "darwin" else None,
        "python": sys.version.split()[0],
        "python_executable": mask_home(sys.executable),
        "packages": package_versions(),
        "torch_mps_built": torch.backends.mps.is_built(),
        "torch_mps_available": torch.backends.mps.is_available(),
        # Un fallback silencioso a CPU invalidaría las medidas de MPS.
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        "git": git_state(snapshot=True),
    }
