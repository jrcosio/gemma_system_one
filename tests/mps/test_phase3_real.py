"""Fase 3 con pesos reales de E2B en MPS/BF16 (skip con motivo si faltan MPS o pesos).

Comprueba en el hardware objetivo lo que en CPU sólo se ve con el modelo diminuto: los 50
módulos objetivo reales, identidad exacta con ``lora_B = 0``, autograd real por la pérdida de
grupo, gradientes finitos, base intacta tras un paso de AdamW y adaptador recargado igual.
"""

from __future__ import annotations

import pytest
import torch

from conftest import E2B_CONFIG
from gemma_system_one.config import load_config
from gemma_system_one.data.generate_mixed import generate_mixed
from gemma_system_one.hub import find_cached_snapshot
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.encoding import load_processor
from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.models.lora import (
    LoraParams,
    apply_lora,
    load_lora_state,
    lora_parameters,
    lora_state,
    lora_summary,
    set_lora_mode,
)
from gemma_system_one.resources import memory_snapshot
from gemma_system_one.training.decisions import build_items, question_loss
from gemma_system_one.training.lora import group_pooled

pytestmark = [
    pytest.mark.mps,
    pytest.mark.weights,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible"),
]


@pytest.fixture(scope="module")
def real():
    cfg = load_config(E2B_CONFIG)
    snap = find_cached_snapshot(cfg)
    if snap is None:
        pytest.skip("Pesos de E2B no están en caché: uv run gso download --config configs/e2b_text.yaml")
    backbone = load_backbone(cfg, snap)
    yield cfg, backbone, load_processor(snap)
    del backbone
    torch.mps.empty_cache()


def test_lora_on_real_e2b_mps(real):
    cfg, bb, processor = real
    kw = {"max_length": cfg.runtime.max_length, "microbatch_rows": 1}
    examples, _ = generate_mixed(20, seed=21)
    items = [it for it in build_items(examples) if it.primitive == "choice"][:2]
    assert items and all(len(it.rows) >= 3 for it in items)
    with torch.no_grad():
        before, _ = group_pooled(bb, processor, [r.text for r in items[0].rows], **kw)

    targets = apply_lora(bb.model, LoraParams(), seed=0)
    summary = lora_summary(bb.model, targets, LoraParams())
    assert len(summary["layers_by_projection"]["q_proj"]) == 35
    assert summary["layers_by_projection"]["v_proj"] == list(range(15))
    assert summary["base_trainable_parameters"] == 0 and summary["trainable_parameters"] > 0
    assert all(t.startswith("language_model.layers.") for t in targets)
    lora = lora_parameters(bb.model)
    assert all(p.device.type == "mps" and p.dtype == torch.float32 for p in lora.values())
    with torch.no_grad():
        after, _ = group_pooled(bb, processor, [r.text for r in items[0].rows], **kw)
    assert torch.equal(before, after)  # B = 0: identidad exacta también en BF16/MPS

    heads = DecisionHeads(bb.hidden_size).to(bb.device)
    heads.standardizer.fit(before.float().cpu())
    heads.to(bb.device)
    probe = {
        n: p.detach().cpu().clone()
        for n, p in bb.model.named_parameters()
        if n.endswith(("layers.0.self_attn.q_proj.base_layer.weight", "embed_tokens.weight"))
    }
    opt = torch.optim.AdamW(list(lora.values()) + list(heads.heads.parameters()), lr=1e-3)
    losses = []
    for step in range(3):
        set_lora_mode(bb.model, training=True)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for it in items:
            pooled, _ = group_pooled(bb, processor, [r.text for r in it.rows], **kw)
            assert pooled.requires_grad and pooled.dtype == torch.bfloat16
            loss = question_loss("choice", heads("choice", pooled), it.target)
            (loss / len(items)).backward()
            total += float(loss.detach()) / len(items)
        losses.append(total)
        grads = {n: p.grad for n, p in lora.items()}
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
        assert all(float(g.abs().sum()) > 0 for n, g in grads.items() if "lora_B" in n or step > 0)
        torch.nn.utils.clip_grad_norm_(list(lora.values()), 1.0, error_if_nonfinite=True)
        opt.step()
    torch.mps.synchronize()
    assert losses[-1] < losses[0], losses
    assert all(torch.equal(p.detach().cpu(), probe[n]) for n, p in bb.model.named_parameters() if n in probe)
    mem = memory_snapshot()
    assert mem["swap_used_bytes"] >= 0 and mem["mps_driver_allocated_bytes"] < 32 * 1024**3

    # Estado del adaptador restaurado en el mismo modelo tras ponerlo a cero (eval, sin dropout).
    state = lora_state(bb.model)
    set_lora_mode(bb.model, training=False)
    with torch.no_grad():
        trained, _ = group_pooled(bb, processor, [r.text for r in items[0].rows], **kw)
    assert not torch.equal(trained, before)
    for p in lora.values():
        with torch.no_grad():
            p.zero_()
    load_lora_state(bb.model, state)
    with torch.no_grad():
        again, _ = group_pooled(bb, processor, [r.text for r in items[0].rows], **kw)
    assert torch.equal(trained, again)
