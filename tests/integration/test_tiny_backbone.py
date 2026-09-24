"""Ruta real de transformers (Gemma4Model) con un checkpoint diminuto aleatorio en CPU.

Valida contratos de carga, congelación, extracción y gradientes. No sustituye la
prueba con los pesos reales en MPS (tests/mps).
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from gemma_system_one.models.backbone import BackboneLoadError, load_backbone
from gemma_system_one.models.heads import NoulHead


@pytest.fixture
def backbone(cpu_cfg, tiny_checkpoint):
    return load_backbone(cpu_cfg, tiny_checkpoint)


def _batch(lengths: list[int], side: str = "right", seed: int = 0) -> tuple[dict, list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randint(3, 500, (n,), generator=g) for n in lengths]
    T = max(lengths)
    ids = torch.zeros(len(rows), T, dtype=torch.long)
    mask = torch.zeros(len(rows), T, dtype=torch.long)
    for i, r in enumerate(rows):
        sl = slice(0, len(r)) if side == "right" else slice(T - len(r), T)
        ids[i, sl] = r
        mask[i, sl] = 1
    return {"input_ids": ids, "attention_mask": mask}, rows


def test_loads_base_without_lm_head_frozen_and_eval(backbone):
    model = backbone.model
    assert type(model).__name__ == "Gemma4Model"
    assert not hasattr(model, "lm_head")
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert backbone.loading_info["missing_keys"] == []
    assert backbone.hidden_size == model.config.get_text_config().hidden_size == 64


def test_kv_shared_layers_have_no_kv_projections(backbone):
    # El checkpoint trae k/v de las capas con KV compartido; transformers las ignora a propósito.
    layers = backbone.text_model.layers
    shared = [i for i, layer in enumerate(layers) if layer.self_attn.is_kv_shared_layer]
    assert shared == [2, 3]
    assert all(
        not hasattr(layers[i].self_attn, "v_proj") or layers[i].self_attn.v_proj is None for i in shared
    )


def test_missing_text_weight_fails_loudly(cpu_cfg, tiny_checkpoint):
    f = tiny_checkpoint / "model.safetensors"
    state = load_file(f)
    del state["model.language_model.layers.0.mlp.up_proj.weight"]
    save_file(state, f, metadata={"format": "pt"})
    with pytest.raises(BackboneLoadError, match="missing"):
        load_backbone(cpu_cfg, tiny_checkpoint)


def test_encode_shapes_and_padding_invariance(backbone):
    batch, rows = _batch([7, 3, 5])
    with torch.inference_mode():
        out = backbone.encode(batch)
        assert out.last_hidden_state.shape == (3, 7, 64)
        pooled = out.pooled()
        for i, r in enumerate(rows):
            alone = backbone.encode({"input_ids": r[None], "attention_mask": torch.ones(1, len(r))}).pooled()[
                0
            ]
            torch.testing.assert_close(pooled[i], alone, atol=1e-5, rtol=1e-5)


def test_left_padding_pooling_matches_unpadded(backbone):
    batch, rows = _batch([6, 2], side="left")
    with torch.inference_mode():
        pooled = backbone.encode(batch).pooled()
        alone = backbone.encode({"input_ids": rows[1][None], "attention_mask": torch.ones(1, 2)}).pooled()[0]
    torch.testing.assert_close(pooled[1], alone, atol=1e-4, rtol=1e-4)


def test_encode_requires_mask(backbone):
    with pytest.raises(ValueError):
        backbone.encode({"input_ids": torch.ones(1, 3, dtype=torch.long)})


def test_head_training_leaves_backbone_untouched(backbone):
    batch, _ = _batch([5, 4, 6])
    before = {n: p.detach().clone() for n, p in backbone.model.named_parameters()}
    with torch.no_grad():
        pooled = backbone.encode(batch).pooled()
    head = NoulHead(backbone.hidden_size).train()
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-2)
    y = torch.tensor([1.0, 0.0, 1.0])
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(pooled), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(p.grad is None for p in backbone.model.parameters())
    assert all(torch.equal(before[n], p) for n, p in backbone.model.named_parameters())
    assert not backbone.model.training


def test_inference_mode_tensors_cannot_feed_head_backward(backbone):
    # Motivo de usar no_grad (y no inference_mode) al extraer h para entrenar el cabezal.
    batch, _ = _batch([4])
    with torch.inference_mode():
        pooled = backbone.encode(batch).pooled()
    head = NoulHead(backbone.hidden_size)
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        head(pooled).sum().backward()


def test_gradient_reaches_first_layer_only_where_enabled(backbone):
    """Sonda previa a LoRA: gradiente real a través de todas las capas hacia un único peso."""
    q0 = backbone.text_model.layers[0].self_attn.q_proj.weight
    q0.requires_grad_(True)
    try:
        batch, _ = _batch([6, 4])
        head = NoulHead(backbone.hidden_size)
        loss = head(backbone.encode(batch).pooled()).sum()
        loss.backward()
        assert q0.grad is not None and torch.isfinite(q0.grad).all() and q0.grad.abs().sum() > 0
        others = [n for n, p in backbone.model.named_parameters() if p.grad is not None and p is not q0]
        assert others == []
    finally:
        q0.requires_grad_(False)
        q0.grad = None
