"""Contratos de descarga con un Hub simulado (doble de test): nunca se toca la red."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from gemma_system_one.config import ProjectConfig
from gemma_system_one.hub import (
    CheckpointIntegrityError,
    RemoteFile,
    RemoteMetadata,
    download,
    find_cached_snapshot,
    manifest_path,
)

FILES = {
    "config.json": b"{}",
    "processor_config.json": b"{}",
    "tokenizer.json": b"tok" * 10,
    "tokenizer_config.json": b"{}",
    "chat_template.jinja": b"tpl",
    "model.safetensors": b"weights" * 100,
}
LFS = {"tokenizer.json", "model.safetensors"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _remote(cfg: ProjectConfig, files=FILES, **overrides) -> RemoteMetadata:
    rfs = tuple(
        RemoteFile(
            n,
            overrides.get(n, {}).get("size", len(b)),
            overrides.get(n, {}).get("sha", _sha(b)) if n in LFS else None,
        )
        for n, b in files.items()
    )
    return RemoteMetadata(cfg.model.repo_id, cfg.model.revision, False, "apache-2.0", rfs)


class FakeHub:
    """Imita la caché de huggingface_hub: snapshots/<rev>/<file> -> blobs/<sha256|id>."""

    def __init__(self, root: Path, files=FILES, cached: bool = False):
        self.root = root
        self.files = files
        self.calls: list[dict] = []
        if cached:
            self._materialize()

    def _snapshot_dir(self, revision: str) -> Path:
        return self.root / "snapshots" / revision

    def _materialize(self, revision: str = "a" * 40) -> None:
        snap = self._snapshot_dir(revision)
        blobs = self.root / "blobs"
        snap.mkdir(parents=True, exist_ok=True)
        blobs.mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            blob = blobs / (_sha(data) if name in LFS else f"id-{name}")
            blob.write_bytes(data)
            link = snap / name
            if not link.exists():
                os.symlink(os.path.relpath(blob, snap), link)

    def __call__(self, repo_id, *, revision, cache_dir=None, local_files_only=False, **kw):
        self.calls.append({"revision": revision, "local_files_only": local_files_only})
        snap = self._snapshot_dir(revision)
        if local_files_only:
            if not snap.exists():
                raise LocalEntryNotFoundError("not cached")
            return str(snap)
        self._materialize(revision)
        return str(snap)

    @property
    def network_downloads(self) -> int:
        return sum(1 for c in self.calls if not c["local_files_only"])


@pytest.fixture
def cfg(cpu_cfg: ProjectConfig) -> ProjectConfig:
    return cpu_cfg.model_copy(update={"model": cpu_cfg.model.model_copy(update={"revision": "a" * 40})})


def test_cache_hit_does_not_download(cfg, tmp_path):
    hub = FakeHub(tmp_path / "hub", cached=True)
    res = download(cfg, snapshot_fn=hub, metadata_fn=lambda c: _remote(c))
    assert res.cache_hit is True
    assert hub.network_downloads == 0
    lfs = {c["file"]: c for c in res.file_checks if "hub_sha256" in c}
    assert all(c["blob_name_matches_sha256"] for c in lfs.values())
    assert all("sha256" not in c for c in lfs.values())  # sin rehash salvo que se pida


def test_miss_downloads_once_with_pinned_revision_and_hashes(cfg, tmp_path):
    hub = FakeHub(tmp_path / "hub")
    res = download(cfg, snapshot_fn=hub, metadata_fn=lambda c: _remote(c))
    assert res.cache_hit is False
    assert hub.network_downloads == 1
    assert [c["revision"] for c in hub.calls] == [cfg.model.revision] * len(hub.calls)
    full = {c["file"]: c["sha256"] for c in res.file_checks if "sha256" in c}
    assert set(full) == LFS
    # Segunda llamada: cache hit, ninguna descarga más y el manifiesto conserva el hash completo.
    res2 = download(cfg, snapshot_fn=hub, metadata_fn=lambda c: _remote(c))
    assert res2.cache_hit is True and hub.network_downloads == 1
    manifest = json.loads(manifest_path(cfg).read_text())
    assert manifest["last_full_hash_verification"]["sha256"] == full


def test_incomplete_snapshot_is_not_a_cache_hit(cfg, tmp_path):
    hub = FakeHub(tmp_path / "hub", cached=True)
    (hub._snapshot_dir(cfg.model.revision) / "model.safetensors").unlink()
    assert find_cached_snapshot(cfg, snapshot_fn=hub) is None


def test_size_mismatch_raises(cfg, tmp_path):
    hub = FakeHub(tmp_path / "hub", cached=True)
    remote = _remote(cfg, **{"config.json": {"size": 999}})
    with pytest.raises(CheckpointIntegrityError, match="tamaño"):
        download(cfg, snapshot_fn=hub, metadata_fn=lambda c: remote)


def test_sha_mismatch_raises_on_verify(cfg, tmp_path):
    hub = FakeHub(tmp_path / "hub", cached=True)
    remote = _remote(cfg, **{"model.safetensors": {"sha": "0" * 64}})
    with pytest.raises(CheckpointIntegrityError, match="sha256"):
        download(cfg, verify_hash=True, snapshot_fn=hub, metadata_fn=lambda c: remote)


def test_offline_with_cache_proceeds_offline_without_cache_fails(cfg, tmp_path):
    def offline(_):
        raise ConnectionError("sin red")

    cached = FakeHub(tmp_path / "hub1", cached=True)
    res = download(cfg, snapshot_fn=cached, metadata_fn=offline)
    assert res.cache_hit and res.remote is None
    with pytest.raises(ConnectionError):
        download(cfg, snapshot_fn=FakeHub(tmp_path / "hub2"), metadata_fn=offline)


def test_manifest_masks_home(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hub = FakeHub(tmp_path / "hub", cached=True)
    download(cfg, snapshot_fn=hub, metadata_fn=lambda c: _remote(c))
    manifest = json.loads(manifest_path(cfg).read_text())
    assert manifest["snapshot_dir"].startswith("~")
    assert str(tmp_path) not in json.dumps(manifest["snapshot_dir"])


def test_wrong_blob_with_same_size_is_rejected_without_verify(cfg, tmp_path):
    hub = FakeHub(tmp_path / "hub", cached=True)
    link = hub._snapshot_dir(cfg.model.revision) / "model.safetensors"
    link.unlink()
    wrong = tmp_path / "wrong-blob"
    wrong.write_bytes(b"x" * len(FILES["model.safetensors"]))
    link.symlink_to(wrong)
    with pytest.raises(CheckpointIntegrityError, match="sha256"):
        download(cfg, snapshot_fn=hub, metadata_fn=lambda c: _remote(c))
