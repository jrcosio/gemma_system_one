from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import gemma_system_one.checkpoint as ckpt_mod
from gemma_system_one.checkpoint import CheckpointMismatchError, load_head, save_head
from gemma_system_one.models.heads import NoulHead

REPO, REV, D = "google/gemma-4-E2B-it", "a" * 40, 16


def _save(path: Path, head: NoulHead) -> Path:
    return save_head(path, head, repo_id=REPO, revision=REV, backbone_dtype="bfloat16", prompt_template="t0")


def _head(seed: int = 0) -> NoulHead:
    torch.manual_seed(seed)
    return NoulHead(D)


def test_roundtrip_exact(tmp_path):
    head = _head()
    d = _save(tmp_path / "c", head)
    loaded, manifest = load_head(d, repo_id=REPO, revision=REV, hidden_size=D)
    x = torch.randn(5, D)
    assert torch.equal(head(x), loaded(x))
    assert manifest["base"] == {"repo_id": REPO, "revision": REV, "dtype": "bfloat16"}
    assert not loaded.training


def test_only_head_weights_are_saved(tmp_path):
    d = _save(tmp_path / "c", _head())
    assert sorted(load_file(d / "head.safetensors")) == ["proj.bias", "proj.weight"]
    assert sorted(p.name for p in d.iterdir()) == ["head.safetensors", "manifest.json"]


def test_refuses_overwrite(tmp_path):
    d = _save(tmp_path / "c", _head())
    with pytest.raises(FileExistsError):
        _save(d, _head(1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repo_id": "google/gemma-4-E4B-it", "revision": REV, "hidden_size": D},
        {"repo_id": REPO, "revision": "b" * 40, "hidden_size": D},
        {"repo_id": REPO, "revision": REV, "hidden_size": D * 2},
    ],
)
def test_rejects_mismatched_base(tmp_path, kwargs):
    d = _save(tmp_path / "c", _head())
    with pytest.raises(CheckpointMismatchError):
        load_head(d, **kwargs)


def test_rejects_tampered_weights(tmp_path):
    d = _save(tmp_path / "c", _head())
    state = load_file(d / "head.safetensors")
    state["proj.bias"] += 1
    save_file(state, d / "head.safetensors")
    with pytest.raises(CheckpointMismatchError, match="sha256"):
        load_head(d, repo_id=REPO, revision=REV, hidden_size=D)


def test_failed_save_leaves_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disco lleno")

    monkeypatch.setattr(ckpt_mod, "save_file", boom)
    with pytest.raises(OSError):
        _save(tmp_path / "c", _head())
    assert list(tmp_path.iterdir()) == []


def test_verify_in_fresh_process(tmp_path):
    head = _head().eval()
    d = _save(tmp_path / "c", head)
    pooled = torch.randn(3, D)
    with torch.inference_mode():
        logits = head(pooled)
    probe = tmp_path / "probe.safetensors"
    save_file({"pooled": pooled, "logits": logits.clone()}, probe)
    cmd = [
        sys.executable,
        "-m",
        "gemma_system_one.checkpoint",
        "verify",
        "--checkpoint",
        str(d),
        "--probe",
        str(probe),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"] and result["max_abs_diff"] == 0.0


def test_head_is_fp32_and_checks_dim():
    head = NoulHead(D)
    out = head(torch.randn(2, D, dtype=torch.bfloat16))
    assert out.dtype == torch.float32 and out.shape == (2,)
    with pytest.raises(ValueError):
        head(torch.randn(2, D + 1))
