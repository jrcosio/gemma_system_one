"""Descarga reproducible del checkpoint: snapshot del Hub con revisión fijada.

- No se cargan pesos ni se re-guardan con ``save_pretrained``.
- Antes de descargar se busca el snapshot en la caché local; si está completo no
  se toca la red para los pesos.
- La identidad de los ficheros LFS se comprueba contra el sha256 publicado por el Hub.
  En la caché de huggingface_hub el enlace del snapshot apunta a ``blobs/<sha256>``,
  lo que permite una comprobación barata; ``verify_hash=True`` recalcula el hash
  completo (siempre se hace tras una descarga nueva).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .env import git_state, mask_home, package_versions


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None
    lfs_sha256: str | None


@dataclass(frozen=True)
class RemoteMetadata:
    repo_id: str
    sha: str
    gated: object
    license: str | None
    files: tuple[RemoteFile, ...]

    def file(self, name: str) -> RemoteFile | None:
        return next((f for f in self.files if f.name == name), None)


@dataclass
class DownloadResult:
    snapshot_dir: Path
    cache_hit: bool
    remote: RemoteMetadata | None
    file_checks: list[dict[str, Any]] = field(default_factory=list)
    manifest_path: Path | None = None


class CheckpointIntegrityError(RuntimeError):
    pass


def fetch_remote_metadata(cfg: ProjectConfig) -> RemoteMetadata:
    from huggingface_hub import model_info

    info = model_info(cfg.model.repo_id, revision=cfg.model.revision, files_metadata=True)
    if info.sha != cfg.model.revision:
        raise CheckpointIntegrityError(
            f"El Hub resolvió {cfg.model.repo_id}@{cfg.model.revision} a {info.sha}"
        )
    files = tuple(
        RemoteFile(s.rfilename, s.size, s.lfs.sha256 if s.lfs is not None else None)
        for s in (info.siblings or [])
    )
    license_ = info.card_data.get("license") if info.card_data else None
    return RemoteMetadata(cfg.model.repo_id, info.sha, info.gated, license_, files)


def find_cached_snapshot(
    cfg: ProjectConfig,
    snapshot_fn: Callable[..., str] | None = None,
) -> Path | None:
    """Ruta del snapshot si está en caché con todos los ficheros requeridos; si no, None."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    if snapshot_fn is None:
        from huggingface_hub import snapshot_download as snapshot_fn
    try:
        path = Path(
            snapshot_fn(
                cfg.model.repo_id,
                revision=cfg.model.revision,
                cache_dir=cfg.paths.hf_cache,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        return None
    if all((path / name).is_file() for name in cfg.model.required_files):
        return path
    return None


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_files(
    snapshot_dir: Path,
    required: tuple[str, ...],
    remote: RemoteMetadata | None,
    verify_hash: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name in required:
        path = snapshot_dir / name
        if not path.is_file():
            raise CheckpointIntegrityError(f"Falta {name} en el snapshot")
        size = path.stat().st_size
        rf = remote.file(name) if remote else None
        entry: dict[str, Any] = {"file": name, "size": size}
        if rf is not None:
            if rf.size is not None and rf.size != size:
                raise CheckpointIntegrityError(f"{name}: tamaño local {size} != Hub {rf.size}")
            entry["hub_size_match"] = rf.size is not None
            if rf.lfs_sha256:
                entry["hub_sha256"] = rf.lfs_sha256
                # snapshots/<rev>/<file> -> blobs/<sha256> (huggingface_hub 1.x puede encadenar
                # otro enlace hacia un almacén global; sólo el primer salto lleva el sha256 LFS).
                blob_name = Path(os.readlink(path)).name if path.is_symlink() else None
                entry["blob_name_matches_sha256"] = blob_name == rf.lfs_sha256
                if verify_hash or not entry["blob_name_matches_sha256"]:
                    digest = sha256_file(path)
                    entry["sha256"] = digest
                    if digest != rf.lfs_sha256:
                        raise CheckpointIntegrityError(f"{name}: sha256 {digest} != Hub {rf.lfs_sha256}")
        checks.append(entry)
    return checks


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_path(cfg: ProjectConfig) -> Path:
    return cfg.paths.artifacts_dir / "manifests" / f"download_{cfg.model.revision}.json"


def download(
    cfg: ProjectConfig,
    *,
    verify_hash: bool = False,
    offline_ok: bool = True,
    snapshot_fn: Callable[..., str] | None = None,
    metadata_fn: Callable[[ProjectConfig], RemoteMetadata] = fetch_remote_metadata,
) -> DownloadResult:
    """Garantiza el snapshot fijado en la caché, descargando sólo si falta."""
    if snapshot_fn is None:
        from huggingface_hub import snapshot_download as snapshot_fn

    cached = find_cached_snapshot(cfg, snapshot_fn=snapshot_fn)
    remote: RemoteMetadata | None
    try:
        remote = metadata_fn(cfg)
    except CheckpointIntegrityError:
        raise
    except Exception as exc:  # red caída, DNS, etc.
        if cached is None or not offline_ok:
            raise
        if verify_hash:
            raise CheckpointIntegrityError(
                "No se puede verificar sha256 contra el Hub sin metadatos de referencia"
            ) from exc
        remote = None
        print(f"[download] metadatos del Hub no disponibles ({type(exc).__name__}); se usa la caché")

    if cached is not None:
        snapshot_dir, cache_hit = cached, True
    else:
        snapshot_dir = Path(
            snapshot_fn(cfg.model.repo_id, revision=cfg.model.revision, cache_dir=cfg.paths.hf_cache)
        )
        cache_hit = False
        # Tras una descarga nueva el hash completo se comprueba siempre.
        verify_hash = True

    checks = check_files(snapshot_dir, cfg.model.required_files, remote, verify_hash=verify_hash)
    result = DownloadResult(snapshot_dir, cache_hit, remote, checks)
    now = datetime.now(UTC).isoformat()

    mp = manifest_path(cfg)
    full_hash = {c["file"]: c["sha256"] for c in checks if "sha256" in c}
    if full_hash:
        last_full = {"verified_utc": now, "sha256": full_hash}
    else:
        # Conservar la última verificación completa si los tamaños no han cambiado.
        last_full = None
        if mp.exists():
            prev = json.loads(mp.read_text(encoding="utf-8"))
            prev_full = prev.get("last_full_hash_verification")
            sizes = {c["file"]: c["size"] for c in checks}
            prev_sizes = {c["file"]: c["size"] for c in prev.get("files", [])}
            if prev_full and all(prev_sizes.get(f) == sizes.get(f) for f in prev_full["sha256"]):
                last_full = prev_full

    payload: dict[str, Any] = {
        "created_utc": now,
        "repo_id": cfg.model.repo_id,
        "revision": cfg.model.revision,
        "snapshot_dir": mask_home(snapshot_dir),
        "cache_hit": cache_hit,
        "hub": None
        if remote is None
        else {"sha": remote.sha, "gated": remote.gated, "license": remote.license},
        "files": checks,
        "last_full_hash_verification": last_full,
        "packages": {k: v for k, v in package_versions().items() if k in ("huggingface_hub", "hf-xet")},
        "git": git_state(),
    }
    _atomic_write_json(mp, payload)
    result.manifest_path = mp
    return result


def require_snapshot(cfg: ProjectConfig) -> Path:
    """Snapshot local para cargar pesos; nunca descarga. Falla con un mensaje claro."""
    path = find_cached_snapshot(cfg)
    if path is None:
        raise FileNotFoundError(
            f"{cfg.model.repo_id}@{cfg.model.revision} no está en la caché local. "
            f"Ejecuta: uv run gso download --config <config>"
        )
    return path
