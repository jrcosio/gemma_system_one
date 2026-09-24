"""Fase 1 con el procesador y los pesos reales de E2B en MPS (skip con motivo si faltan)."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from conftest import E2B_CONFIG
from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate import generate, write_dataset
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
def test_real_processor_rows_fit_and_end_at_model_turn(snapshot):
    """Filas reales: sin truncar, padding derecho, un BOS y lectura tras el turno del modelo."""
    processor = load_processor(snapshot)
    examples, _ = generate(40, seed=0)
    texts = [expand_example(e)[0].text for e in examples]
    batch = build_chat_batch(processor, texts, max_length=512)
    tok = processor.tokenizer
    mask = batch["attention_mask"]
    assert int(mask.sum(1).max()) <= 512
    for i in range(len(texts)):
        n = int(mask[i].sum())
        assert mask[i, :n].all() and not mask[i, n:].any()  # padding a la derecha
        ids = batch["input_ids"][i, :n].tolist()
        assert ids.count(tok.bos_token_id) == 1
        assert tok.convert_ids_to_tokens(ids[-3:]) == ["<|turn>", "model", "\n"]
        assert "<|think|>" not in tok.decode(ids)


@pytest.mark.mps
@pytest.mark.weights
@needs_mps
def test_phase1_overfit_and_reload_from_text_on_mps(snapshot, tmp_path):
    from gemma_system_one.training.pipeline import run_evaluate, run_train

    ds = tmp_path / "ds"
    write_dataset(ds, 12, seed=11)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    cfg_path = tmp_path / "overfit.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "mps_overfit",
                "base_config": str(E2B_CONFIG),
                "dataset": str(ds),
                "split_seed": 0,
                "cache_dir": str(tmp_path / "cache"),
                "runs_dir": str(tmp_path / "runs"),
                "train": {
                    "epochs": 200,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "select_on": "train",
                    "max_train_examples": 16,
                },
            }
        )  # fmt: skip
    )
    report = run_train(cfg_path)
    check = report["metrics"]["overfit_check"]
    assert check["passed"], check
    assert report["extraction"][0]["cache_hit"] is False and report["extraction"][0]["backbone_forwards"] > 0
    fp = json.loads(
        (tmp_path / "runs" / "mps_overfit").glob("*/checkpoint/manifest.json").__next__().read_text()
    )
    assert fp["extra"]["fingerprint"]["device"] == "mps" and fp["extra"]["fingerprint"]["dtype"] == "bfloat16"
    ev = run_evaluate(report["checkpoint"], "train", use_cache=False)
    assert ev["backbone_loaded"] and ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
