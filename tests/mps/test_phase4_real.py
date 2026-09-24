"""Fase 4 con procesador y pesos reales de E2B en MPS/BF16 (skip con motivo si faltan).

Filas reales con imagen (tokens visuales, orden y límites), sobreajuste + recarga desde
texto+imagen con ablación visual y un backward LoRA a través de filas con imagen con su memoria.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from conftest import REPO_ROOT
from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_vision import write_vision_dataset
from gemma_system_one.data.split import make_split, split_path, write_split
from gemma_system_one.hub import find_cached_snapshot
from gemma_system_one.images import load_image
from gemma_system_one.models.encoding import build_chat_batch, image_token_count, load_processor
from gemma_system_one.training.decisions import build_items

VISION_CONFIG = REPO_ROOT / "configs" / "e2b_vision.yaml"
needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible")


@pytest.fixture(scope="module")
def snapshot():
    snap = find_cached_snapshot(load_config(VISION_CONFIG))
    if snap is None:
        pytest.skip("Pesos de E2B no están en caché: uv run gso download --config configs/e2b_vision.yaml")
    return snap


@pytest.fixture(scope="module")
def vision_ds(tmp_path_factory):
    ds = tmp_path_factory.mktemp("vision") / "ds"
    write_vision_dataset(ds, 16, seed=12)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    return ds


@pytest.mark.weights
def test_real_processor_vision_rows(snapshot, vision_ds):
    """Filas reales: imagen antes del texto, ≤280 tokens visuales, dentro de max_length y sin IDs."""
    processor = load_processor(snapshot)
    tok = processor.tokenizer
    for it in build_items(load_dataset(vision_ds).examples[:12], vision_ds):
        image = load_image(it.image_file)
        for row in it.rows:
            b = build_chat_batch(processor, [row.text], max_length=768, images=[image])
            n = int(b["attention_mask"].sum())
            ids = b["input_ids"][0, :n].tolist()
            k = image_token_count(b)
            assert 0 < k <= 280 and b["pixel_values"].shape[0] == 1
            full = tok.decode(ids)
            assert full.index("<|image|>") < full.index("gso-eval")  # imagen antes del texto de la fila
            assert tok.convert_ids_to_tokens(ids[-3:]) == ["<|turn>", "model", "\n"]
            decoded = tok.decode([i for i in ids if i != processor.image_token_id])
            if it.primitive == "choice":
                assert not any(cid in decoded for cid in it.example.question.criteria)


@pytest.mark.mps
@pytest.mark.weights
@needs_mps
def test_phase4_overfit_reload_and_vision_ablation_on_mps(snapshot, vision_ds, tmp_path):
    from gemma_system_one.training.decisions_pipeline import run_evaluate_decisions, run_train_decisions

    cfg = tmp_path / "overfit.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "kind": "decision_heads",
                "name": "mps_vision_overfit",
                "base_config": str(VISION_CONFIG),
                "dataset": str(vision_ds),
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
        )
    )
    report = run_train_decisions(cfg)
    assert report["metrics"]["overfit_check"]["passed"], report["metrics"]["overfit_check"]
    ext = report["extraction"][0]
    assert ext["image_rows"] == ext["rows"] == ext["backbone_forwards"] and ext["image_tokens"] > 0
    ev = run_evaluate_decisions(report["checkpoint"], "train", use_cache=False, vision_ablation=True)
    assert ev["backbone_loaded"] and ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
    va = ev["vision_ablation"]
    assert va["image_omitted"]["all"]["questions"] == va["questions_with_image"] > 0
    assert va["image_swapped"]["all"]["questions"] > 0
    assert ev["memory"]["peaks_sampled"]["mps_driver_allocated_bytes"] < 32 * 1024**3


@pytest.mark.mps
@pytest.mark.weights
@needs_mps
def test_lora_backward_through_image_rows_on_mps(snapshot, vision_ds):
    """Autograd LoRA con filas con imagen: gradientes finitos, torre de visión congelada, memoria."""
    from gemma_system_one.models.backbone import load_backbone
    from gemma_system_one.models.heads import DecisionHeads
    from gemma_system_one.models.lora import LoraParams, apply_lora, lora_parameters, set_lora_mode
    from gemma_system_one.training.decisions import question_loss
    from gemma_system_one.training.lora import group_pooled

    cfg = load_config(VISION_CONFIG)
    bb = load_backbone(cfg, snapshot)
    processor = load_processor(snapshot)
    apply_lora(bb.model, LoraParams(), seed=0)
    assert not any(".lora_" in n for n, _ in bb.model.vision_tower.named_parameters())
    item = next(it for it in build_items(load_dataset(vision_ds).examples, vision_ds) if len(it.rows) >= 4)
    heads = DecisionHeads(bb.hidden_size).to(bb.device)
    lora = lora_parameters(bb.model)
    for _ in range(2):
        set_lora_mode(bb.model, training=True)
        for p in lora.values():
            p.grad = None
        pooled, tokens = group_pooled(
            bb,
            processor,
            [r.text for r in item.rows],
            max_length=768,
            microbatch_rows=1,
            image_file=item.image_file,
        )
        z = heads(item.primitive, pooled)
        graph_bytes = torch.mps.current_allocated_memory()
        question_loss(item.primitive, z, item.target).backward()
        torch.mps.synchronize()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in lora.values())
        assert tokens > 266 * len(item.rows)
        assert graph_bytes < 32 * 1024**3 and torch.mps.driver_allocated_memory() < 32 * 1024**3
        with torch.no_grad():
            for n, p in lora.items():
                if "lora_B" in n:
                    p.add_(torch.randn_like(p) * 1e-3)  # B ≠ 0 para que A reciba gradiente en el paso 2
    assert all(bool(p.grad.abs().sum() > 0) for p in lora.values())
    assert all(not p.requires_grad for p in bb.model.vision_tower.parameters())
    del bb
    torch.mps.empty_cache()
