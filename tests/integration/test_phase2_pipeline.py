"""Fase 2 en CPU: Gemma4Model real diminuto + procesador doble (contratos, no calidad)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import gemma_system_one.features as features_mod
import gemma_system_one.training.decisions_pipeline as dp
import gemma_system_one.training.pipeline as p1
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import write_mixed_dataset
from gemma_system_one.data.split import SplitError, make_split, split_path, write_split

FAKE_PROC_FP = {"files_sha256": {"fake": "0"}, "chat_kwargs": {}, "padding_side": "right"}


@pytest.fixture
def setup(tmp_path, tiny_checkpoint, fake_processor, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "model": {"repo_id": "google/gemma-4-E2B-it", "revision": "a" * 40, "dtype": "float32"},
                "runtime": {"device": "cpu", "max_length": 250},
            }
        )
    )
    ds = tmp_path / "mixed"
    write_mixed_dataset(ds, 24, seed=4)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    transfer = tmp_path / "transfer"
    write_mixed_dataset(transfer, 6, seed=4, variant="transfer")
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "kind": "decision_heads",
                "name": "tiny_mixed",
                "base_config": str(base),
                "dataset": str(ds),
                "cache_dir": str(tmp_path / "cache"),
                "runs_dir": str(tmp_path / "runs"),
                "train": {"epochs": 3, "lr": 0.01},
            }
        )
    )
    for mod in (dp, p1):
        monkeypatch.setattr(mod, "require_snapshot", lambda cfg: tiny_checkpoint, raising=False)
        monkeypatch.setattr(mod, "load_processor", lambda snap: fake_processor, raising=False)
        monkeypatch.setattr(mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP, raising=False)
    monkeypatch.setattr(features_mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP)
    return cfg, ds, transfer


def test_train_evaluate_reload_robustness_and_guards(setup):
    cfg, ds, transfer = setup
    report = dp.run_train_decisions(cfg)
    run_dir = Path(report["run_dir"])
    m = report["metrics"]["models"]
    assert set(m) == {"gemma_heads", "prior", "bow"}
    val = m["gemma_heads"]["validation"]
    assert {"noul", "choice", "score"} & set(val)
    manifest = json.loads((run_dir / "checkpoint" / "manifest.json").read_text())
    assert manifest["kind"] == "decision_heads" and manifest["extra"]["calibrated"] is False
    assert sorted(k.split(".")[1] for k in manifest["weights"]["keys"] if k.endswith("weight")) == [
        "choice",
        "noul",
        "score",
    ]
    preds = [json.loads(line) for line in (run_dir / "predictions_validation.jsonl").read_text().splitlines()]
    for p in preds:
        a = p["answer"]
        if a["type"] == "choice":
            assert abs(sum(a["probabilities"].values()) - 1) < 1e-6 and a["choice"] in a["probabilities"]
        if a["type"] == "score":
            assert 0 <= a["score"] <= len(a["probabilities"]) - 1 and a["legend"]

    ev = dp.run_evaluate_decisions(run_dir / "checkpoint", "validation", use_cache=False, robustness=True)
    assert ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
    assert ev["usage"]["generated_tokens"] == 0
    assert ev["usage"]["expanded_rows"] >= ev["usage"]["questions"]
    rob = ev["robustness"]
    if rob["choice_id_rename_permutation"]["questions"]:
        r = rob["choice_id_rename_permutation"]
        assert r["rows_identical"] == r["questions"] and r["max_prob_diff_remapped"] == 0.0

    ext = dp.run_evaluate_decisions(run_dir / "checkpoint", "all", dataset=transfer, use_cache=False)
    assert ext["diagnostic_only"] and ext["external_leakage"]["errors"] == []
    with pytest.raises(SplitError):
        dp.run_evaluate_decisions(run_dir / "checkpoint", "test")
    with pytest.raises(ValueError, match="comparte"):
        dp.run_evaluate_decisions(run_dir / "checkpoint", "all", dataset=ds)  # mismo dataset: fuga


def test_overfit_gate_reports_per_primitive(setup, tmp_path):
    cfg, _, _ = setup
    raw = yaml.safe_load(cfg.read_text())
    raw["train"] = {
        "epochs": 150,
        "lr": 0.05,
        "weight_decay": 0.0,
        "select_on": "train",
        "max_train_questions": 12,
    }
    over = tmp_path / "over.yaml"
    over.write_text(yaml.safe_dump(raw))
    report = dp.run_train_decisions(over)
    check = report["metrics"]["overfit_check"]
    assert set(check["per_type"]) <= {"noul", "choice", "score"} and check["train_questions"] == 12
    assert check["passed"], check
