"""Fase 2 con procesador y pesos reales de E2B en MPS (skip con motivo si faltan)."""

from __future__ import annotations

import pytest
import torch
import yaml

from conftest import E2B_CONFIG
from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import generate_mixed, write_mixed_dataset
from gemma_system_one.data.split import make_split, split_path, write_split
from gemma_system_one.hub import find_cached_snapshot
from gemma_system_one.models.encoding import build_chat_batch, load_processor
from gemma_system_one.serialization import expand_example

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible")


@pytest.fixture(scope="module")
def snapshot():
    snap = find_cached_snapshot(load_config(E2B_CONFIG))
    if snap is None:
        pytest.skip("Pesos de E2B no están en caché: uv run gso download --config configs/e2b_text.yaml")
    return snap


@pytest.mark.weights
def test_real_processor_choice_and_score_rows(snapshot):
    """Filas Choice/Score reales: ≤512 tokens, sin IDs opacos y lectura tras el turno del modelo."""
    processor = load_processor(snapshot)
    examples, _ = generate_mixed(60, seed=9)
    rows = [(e, r) for e in examples if e.question.type != "noul" for r in expand_example(e)]
    batch = build_chat_batch(processor, [r.text for _, r in rows], max_length=512)
    tok = processor.tokenizer
    for i, (e, _row) in enumerate(rows):
        n = int(batch["attention_mask"][i].sum())
        ids = batch["input_ids"][i, :n].tolist()
        assert tok.convert_ids_to_tokens(ids[-3:]) == ["<|turn>", "model", "\n"]
        decoded = tok.decode(ids)
        if e.question.type == "choice":
            assert not any(k in decoded for k in e.question.criteria)  # IDs opacos fuera del prompt


@pytest.mark.mps
@pytest.mark.weights
@needs_mps
def test_phase2_overfit_and_reload_from_text_on_mps(snapshot, tmp_path):
    from gemma_system_one.training.decisions_pipeline import run_evaluate_decisions, run_train_decisions

    ds = tmp_path / "ds"
    write_mixed_dataset(ds, 16, seed=12)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    cfg = tmp_path / "overfit.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "kind": "decision_heads",
                "name": "mps_mixed_overfit",
                "base_config": str(E2B_CONFIG),
                "dataset": str(ds),
                "cache_dir": str(tmp_path / "cache"),
                "runs_dir": str(tmp_path / "runs"),
                "train": {
                    "epochs": 300,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "select_on": "train",
                    "max_train_questions": 16,
                },
            }
        )  # fmt: skip
    )
    report = run_train_decisions(cfg)
    assert report["metrics"]["overfit_check"]["passed"], report["metrics"]["overfit_check"]
    ext = report["extraction"][0]
    assert ext["cache_hit"] is False and ext["backbone_forwards"] == ext["rows"] > ext["questions"]
    ev = run_evaluate_decisions(report["checkpoint"], "train", use_cache=False, robustness=True)
    assert ev["backbone_loaded"] and ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
    r = ev["robustness"]["choice_id_rename_permutation"]
    assert r["rows_identical"] == r["questions"]
