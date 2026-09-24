"""Pruebas en hardware real. Se omiten (skip con motivo) sin MPS o sin pesos en caché;
una omisión no es un aprobado."""

from __future__ import annotations

import pytest
import torch

from gemma_system_one.hub import find_cached_snapshot
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.heads import NoulHead

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible")


@pytest.fixture
def mps_bf16_cfg(cpu_cfg):
    return cpu_cfg.model_copy(
        update={
            "runtime": cpu_cfg.runtime.model_copy(update={"device": "mps"}),
            "model": cpu_cfg.model.model_copy(update={"dtype": "bfloat16"}),
        }
    )


@pytest.mark.mps
@needs_mps
def test_tiny_gemma4_bf16_forward_backward_on_mps(mps_bf16_cfg, cpu_cfg, tiny_checkpoint):
    """Ruta Gemma4 completa en MPS/bf16 con gradiente hasta la capa 0, comparada con CPU/fp32."""
    mps = load_backbone(mps_bf16_cfg, tiny_checkpoint)
    ref = load_backbone(cpu_cfg, tiny_checkpoint)
    assert {p.device.type for p in mps.model.parameters()} == {"mps"}
    ids = torch.randint(3, 500, (2, 9), generator=torch.Generator().manual_seed(0))
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0
    batch = {"input_ids": ids, "attention_mask": mask}

    with torch.inference_mode():
        h_ref = ref.encode(batch).pooled()
    q0 = mps.text_model.layers[0].self_attn.q_proj.weight
    q0.requires_grad_(True)
    head = NoulHead(mps.hidden_size).to("mps")
    pooled = mps.encode(batch).pooled()
    loss = head(pooled).sum()
    loss.backward()
    torch.mps.synchronize()
    assert torch.isfinite(pooled).all()
    assert q0.grad is not None and torch.isfinite(q0.grad).all() and q0.grad.float().abs().sum() > 0
    cos = torch.cosine_similarity(pooled.float().cpu(), h_ref.float(), dim=1)
    assert (cos > 0.99).all(), cos


@pytest.mark.mps
@pytest.mark.weights
@needs_mps
def test_doctor_real_e2b(e2b_cfg, tmp_path):
    if find_cached_snapshot(e2b_cfg) is None:
        pytest.skip("Pesos de E2B no están en caché: uv run gso download --config configs/e2b_text.yaml")
    from gemma_system_one.doctor import Doctor

    cfg = e2b_cfg.model_copy(
        update={
            "paths": e2b_cfg.paths.model_copy(
                update={"artifacts_dir": tmp_path / "artifacts", "reports_dir": tmp_path / "reports"}
            )
        }
    )
    report = Doctor(cfg, skip_model=False).run()
    steps = {s["name"]: s for s in report["steps"]}
    failed = {n: s.get("error") for n, s in steps.items() if s["status"] == "fail"}
    assert not failed, failed
    assert report["verdict"] == "pass"
    load = steps["load_backbone"]["details"]
    assert load["class"] == "Gemma4Model" and not load["has_lm_head"]
    assert load["param_devices"] == ["mps:0"] and load["trainable_params"] == 0
    assert steps["text_forward"]["details"]["batch_shape"][2] == 1536
    assert steps["head_train"]["details"]["trainable_parameters"] == ["proj.weight", "proj.bias"]
    assert steps["save_reload"]["details"]["fresh_process_runs"]["mps"]["ok"]
    vis = steps["vision"]["details"]  # fase 4: repetición con una imagen real
    assert steps["vision"]["status"] == "pass" and 0 < vis["image_tokens"] <= 280
    assert vis["repeat_max_abs_diff"] == 0.0 and vis["other_image_max_abs_diff"] > 0
    assert {"pixel_values", "image_position_ids", "mm_token_type_ids"} <= set(vis["processor_fields"])
