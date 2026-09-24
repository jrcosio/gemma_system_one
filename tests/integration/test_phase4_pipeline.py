"""Fase 4 en CPU: Gemma 4 diminuto con torre de visión + procesador doble (contratos, no calidad).

Cubre: la representación depende de la imagen; caché con la imagen en la clave; entrenamiento
y recarga desde texto+imagen; ablación con imagen omitida/intercambiada; control sólo texto;
rechazo de microlotes con imagen; LoRA con autograd a través de filas con imagen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import gemma_system_one.features as features_mod
import gemma_system_one.training.decisions_pipeline as dp
import gemma_system_one.training.lora_pipeline as lp
import gemma_system_one.training.pipeline as p1
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_vision import write_vision_dataset
from gemma_system_one.data.split import make_split, split_path, write_split
from gemma_system_one.features import extract_pooled
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.training.decisions import build_items, flatten, row_images

FAKE_PROC_FP = {"files_sha256": {"fake": "0"}, "chat_kwargs": {}, "padding_side": "right"}


@pytest.fixture
def setup(tmp_path, tiny_vision_checkpoint, fake_processor, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "name": "tiny_vision",
                "model": {"repo_id": "google/gemma-4-E2B-it", "revision": "a" * 40, "dtype": "float32"},
                "runtime": {"device": "cpu", "max_length": 250},
            }
        )
    )
    ds = tmp_path / "vision"
    write_vision_dataset(ds, 40, seed=4)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    for mod in (dp, p1, lp):
        monkeypatch.setattr(mod, "require_snapshot", lambda cfg: tiny_vision_checkpoint, raising=False)
        monkeypatch.setattr(mod, "load_processor", lambda snap: fake_processor, raising=False)
        monkeypatch.setattr(mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP, raising=False)
    monkeypatch.setattr(features_mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP)

    def heads_cfg(name: str, **extra) -> Path:
        path = tmp_path / f"{name}.yaml"
        raw = {
            "kind": "decision_heads",
            "name": name,
            "base_config": str(base),
            "dataset": str(ds),
            "cache_dir": str(tmp_path / "cache"),
            "runs_dir": str(tmp_path / "runs"),
            "train": {"epochs": 3, "lr": 0.01},
            **extra,
        }
        path.write_text(yaml.safe_dump(raw))
        return path

    return base, ds, heads_cfg, tiny_vision_checkpoint


def test_representation_depends_on_the_image(setup, fake_processor):
    base, ds, _, ckpt = setup
    from gemma_system_one.config import load_config

    bb = load_backbone(load_config(base), ckpt)
    items = build_items(load_dataset(ds).examples[:6], ds)
    texts, _, _ = flatten(items)
    images = row_images(items)
    reps, stats = extract_pooled(bb, fake_processor, texts, max_length=250, microbatch_rows=1, images=images)
    assert stats.image_rows == len(texts) and stats.image_tokens == 4 * len(texts)
    assert stats.backbone_forwards == len(texts)
    donor = next(p for p in images if p != images[0])
    swapped, _ = extract_pooled(
        bb, fake_processor, texts[:1], max_length=250, microbatch_rows=1, images=[donor]
    )
    none, _ = extract_pooled(bb, fake_processor, texts[:1], max_length=250, microbatch_rows=1)
    assert not torch.allclose(reps[:1], swapped) and not torch.allclose(reps[:1], none)
    again, _ = extract_pooled(
        bb, fake_processor, texts[:1], max_length=250, microbatch_rows=1, images=images[:1]
    )
    assert torch.equal(reps[:1], again)
    with pytest.raises(ValueError, match="una fila por forward"):
        extract_pooled(bb, fake_processor, texts[:2], max_length=250, microbatch_rows=2, images=images[:2])


def test_train_reload_ablation_and_text_only_control(setup):
    _, ds, heads_cfg, _ = setup
    rep = dp.run_train_decisions(heads_cfg("tiny_vis_heads"))
    ext = rep["extraction"][0]
    assert ext["image_rows"] == ext["rows"] and ext["cache_hit"] is False
    ckpt = Path(rep["checkpoint"])
    assert json.loads((ckpt / "manifest.json").read_text())["extra"]["images"] == "use"
    ev = dp.run_evaluate_decisions(ckpt, "validation", use_cache=False, vision_ablation=True)
    assert ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
    assert ev["extraction"]["image_rows"] == ev["usage"]["expanded_rows"]
    va = ev["vision_ablation"]
    assert va["questions_with_image"] == ev["usage"]["questions"]
    assert va["image_omitted"]["all"]["questions"] == va["questions_with_image"]
    assert va["image_swapped"]["all"]["questions"] > 0
    preds = [json.loads(x) for x in Path(ev["predictions_path"]).read_text().splitlines()]
    assert all(len(p["input_sha256"]) == 64 for p in preds)

    # Mismo texto, sin imagen: otra clave de caché y filas sin imagen en entrenamiento y evaluación.
    ctl = dp.run_train_decisions(heads_cfg("tiny_vis_text_only", images="omit"))
    assert ctl["extraction"][0]["cache_hit"] is False and ctl["extraction"][0]["image_rows"] == 0
    cev = dp.run_evaluate_decisions(Path(ctl["checkpoint"]), "validation", use_cache=False)
    assert cev["reload_vs_training"]["ok"] and cev["extraction"]["image_rows"] == 0
    cached = dp.run_train_decisions(heads_cfg("tiny_vis_heads_again"))
    assert cached["extraction"][0]["cache_hit"] is True  # la imagen forma parte de la clave


def test_lora_backpropagates_through_image_rows(setup, tmp_path):
    base, ds, heads_cfg, _ = setup
    init = Path(dp.run_train_decisions(heads_cfg("tiny_vis_init"))["checkpoint"])
    cfg = tmp_path / "lora.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "kind": "lora_decision_heads",
                "name": "tiny_vis_lora",
                "base_config": str(base),
                "dataset": str(ds),
                "init_heads_from": str(init),
                "runs_dir": str(tmp_path / "runs"),
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
                "train": {"epochs": 1, "lr_lora": 5e-3, "questions_per_step": 8, "max_train_questions": 16},
            }
        )
    )
    rep = lp.run_train_lora(cfg)
    assert rep["global_steps"] == 2 and rep["base_params_unchanged_probe"]
    ev = dp.run_evaluate_decisions(Path(rep["checkpoint"]), "validation")
    assert ev["reload_vs_training"]["ok"] and ev["extraction"]["image_rows"] == ev["usage"]["expanded_rows"]


def test_checkpoint_records_modality_and_run_start_code(setup):
    _, _, heads_cfg, _ = setup
    rep = dp.run_train_decisions(heads_cfg("tiny_vis_modality"))
    extra = json.loads((Path(rep["checkpoint"]) / "manifest.json").read_text())["extra"]
    assert extra["input_modality"] == "text+image" and extra["run_start_code"]["source"]["archive"]
    ctl = dp.run_train_decisions(heads_cfg("tiny_vis_modality_text", images="omit"))
    assert (
        json.loads((Path(ctl["checkpoint"]) / "manifest.json").read_text())["extra"]["input_modality"]
        == "text"
    )
