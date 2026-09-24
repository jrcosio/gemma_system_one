"""Fase 1 en CPU: Gemma4Model real diminuto + procesador doble.

Valida orquestación, caché, recarga desde el texto y guardas de particiones. No es
evidencia de calidad ni del procesador real (eso está en tests/mps).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import gemma_system_one.features as features_mod
import gemma_system_one.training.pipeline as pipeline_mod
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate import write_dataset
from gemma_system_one.data.split import SplitError, make_split, split_path, write_split
from gemma_system_one.features import RepresentationCache, backbone_fingerprint, extract_pooled
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.encoding import InputTooLongError, build_chat_batch

FAKE_PROC_FP = {"files_sha256": {"fake": "0"}, "chat_kwargs": {}, "padding_side": "right"}
TEXTS = ["uno dos tres", "a b c d e f g h", "corto", "otra frase de longitud media aquí"]


@pytest.fixture
def backbone(cpu_cfg, tiny_checkpoint):
    return load_backbone(cpu_cfg, tiny_checkpoint)


def test_batch_forces_right_padding_and_rejects_long_rows(fake_processor):
    batch = build_chat_batch(fake_processor, TEXTS, max_length=64)
    assert fake_processor.calls[-1]["padding_side"] == "right"
    assert batch["attention_mask"][2].tolist()[0] == 1  # fila corta: tokens al principio
    with pytest.raises(InputTooLongError) as exc:
        build_chat_batch(fake_processor, TEXTS, max_length=5)
    assert [i for i, _ in exc.value.rows] == [1, 3]  # "uno dos tres" = BOS+3+EOS = 5 tokens


def test_extraction_preserves_order_and_matches_single_rows(backbone, fake_processor):
    pooled, stats = extract_pooled(backbone, fake_processor, TEXTS, max_length=64, microbatch_rows=3)
    assert pooled.shape == (4, 64) and pooled.dtype == torch.float32
    assert stats.rows == 4 and stats.backbone_forwards == 2
    lengths = build_chat_batch(fake_processor, TEXTS, max_length=64)["attention_mask"].sum(1)
    assert stats.valid_tokens == int(lengths.sum())
    for i, t in enumerate(TEXTS):
        alone, _ = extract_pooled(backbone, fake_processor, [t], max_length=64, microbatch_rows=1)
        torch.testing.assert_close(pooled[i], alone[0], atol=1e-5, rtol=1e-5)


def test_cache_hits_only_with_identical_fingerprint_and_rows(tmp_path, cpu_cfg, tiny_checkpoint, monkeypatch):
    monkeypatch.setattr(features_mod, "processor_fingerprint", lambda _: FAKE_PROC_FP)
    fp = backbone_fingerprint(cpu_cfg, tiny_checkpoint, 8)
    cache = RepresentationCache(tmp_path / "cache", fp)
    rows = ["h1", "h2"]
    tensor = torch.randn(2, 4)
    assert cache.get(rows) is None
    cache.put(rows, tensor, features_mod.ExtractionStats(rows=2))
    torch.testing.assert_close(cache.get(rows), tensor)
    assert cache.get(["h2", "h1"]) is None  # otro orden
    for change in (
        {"dtype": "bfloat16"},
        {"template_version": "gso-text-v2"},
        {"extraction": {"microbatch_rows": 4}},
    ):
        other = RepresentationCache(tmp_path / "cache", {**fp, **change})
        assert other.get(rows) is None


@pytest.fixture
def phase1_setup(tmp_path, tiny_checkpoint, fake_processor, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "model": {"repo_id": "google/gemma-4-E2B-it", "revision": "a" * 40, "dtype": "float32"},
                "runtime": {"device": "cpu", "max_length": 200},
            }
        )
    )
    ds_root = tmp_path / "ds"
    write_dataset(ds_root, 16, seed=5)
    dataset = load_dataset(ds_root)
    write_split(make_split(dataset, 0), split_path(ds_root, 0))
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        yaml.safe_dump(
            {
                "name": "tiny_noul",
                "base_config": str(base),
                "dataset": str(ds_root),
                "split_seed": 0,
                "extraction": {"microbatch_rows": 4},
                "cache_dir": str(tmp_path / "cache"),
                "runs_dir": str(tmp_path / "runs"),
                "train": {"epochs": 3, "lr": 0.01, "select_on": "validation"},
            }
        )
    )
    monkeypatch.setattr(pipeline_mod, "require_snapshot", lambda cfg: tiny_checkpoint)
    monkeypatch.setattr(pipeline_mod, "load_processor", lambda snap: fake_processor)
    monkeypatch.setattr(pipeline_mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP)
    monkeypatch.setattr(features_mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP)
    return train_cfg, ds_root


def test_train_then_reload_from_text_in_evaluate(phase1_setup):
    train_cfg, ds_root = phase1_setup
    report = pipeline_mod.run_train(train_cfg)
    run_dir = Path(report["run_dir"])
    for f in (
        "train_config.yaml",
        "env.json",
        "history.jsonl",
        "metrics.json",
        "predictions_validation.jsonl",
    ):
        assert (run_dir / f).is_file()
    assert report["splits_not_used_for_fitting"] == ["calibration", "test"]
    manifest = json.loads((run_dir / "checkpoint" / "manifest.json").read_text())
    assert manifest["extra"]["select_on"] == "validation"
    assert sorted(manifest["weights"]["keys"]) == ["proj.bias", "proj.weight"]

    # Ningún ejemplo de calibración/test aparece en las predicciones del entrenamiento.
    split = json.loads(split_path(ds_root, 0).read_text())
    held = {x["id"] for s in ("calibration", "test") for x in split["splits"][s]["examples"]}
    seen = {json.loads(line)["id"] for f in ("train", "validation")
            for line in (run_dir / f"predictions_{f}.jsonl").read_text().splitlines()}  # fmt: skip
    assert seen.isdisjoint(held)

    ev = pipeline_mod.run_evaluate(run_dir / "checkpoint", "validation", use_cache=False)
    assert ev["backbone_loaded"] is True
    assert ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
    cached = pipeline_mod.run_evaluate(run_dir / "checkpoint", "validation", use_cache=True)
    assert cached["backbone_loaded"] is False and cached["extraction"]["cache_hit"] is True

    with pytest.raises(SplitError, match="test"):
        pipeline_mod.run_evaluate(run_dir / "checkpoint", "test")
    final = pipeline_mod.run_evaluate(run_dir / "checkpoint", "test", allow_test=True, use_cache=False)
    assert final["final_test_used"] is True


def test_evaluate_refuses_modified_dataset(phase1_setup):
    train_cfg, ds_root = phase1_setup
    report = pipeline_mod.run_train(train_cfg)
    (ds_root / "dataset_manifest.json").unlink()
    with (ds_root / "examples.jsonl").open("a") as fh:
        line = (ds_root / "examples.jsonl").read_text().splitlines()[0]
        row = json.loads(line)
        row.update(id="nuevo-1", group_id="nuevo-grupo", state="estado añadido después")
        fh.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="dataset ha cambiado"):
        pipeline_mod.run_evaluate(Path(report["checkpoint"]), "validation")
