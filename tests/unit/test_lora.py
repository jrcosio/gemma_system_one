"""Fase 3 en CPU con Gemma4Model diminuto real: objetivos LoRA, gradientes, identidad inicial,
microlotes frente a una fila por forward, scheduler y calibración (contratos, no calidad)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from conftest import tiny_gemma4_config
from gemma_system_one.calibration import MIN_QUESTIONS, fit_temperature, fit_temperatures
from gemma_system_one.checkpoint import CheckpointMismatchError, load_lora_decision, save_lora_decision
from gemma_system_one.models.backbone import Backbone
from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.models.lora import (
    LoraError,
    LoraParams,
    apply_lora,
    load_lora_state,
    lora_parameters,
    lora_state,
    set_lora_mode,
    text_attention_targets,
)
from gemma_system_one.training.decisions import question_loss
from gemma_system_one.training.lora import epoch_permutation, group_pooled, lr_factor


def _tiny_backbone(dtype=torch.float32) -> Backbone:
    from transformers import Gemma4Model

    torch.manual_seed(0)
    model = Gemma4Model(tiny_gemma4_config()).to(dtype)
    bb = Backbone(model, torch.device("cpu"), dtype, None)
    bb.freeze()
    return bb


def _ids(n=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 500, (1, n), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_targets_are_exact_text_attention_linears():
    bb = _tiny_backbone()
    targets = text_attention_targets(bb.model, ("q_proj", "v_proj"))
    # Tiny: 4 capas, las 2 últimas comparten K/V (sin v_proj), como las 15–34 de E2B.
    assert targets == [
        "language_model.layers.0.self_attn.q_proj",
        "language_model.layers.0.self_attn.v_proj",
        "language_model.layers.1.self_attn.q_proj",
        "language_model.layers.1.self_attn.v_proj",
        "language_model.layers.2.self_attn.q_proj",
        "language_model.layers.3.self_attn.q_proj",
    ]


def test_non_linear_target_is_rejected():
    class Clip(nn.Module):  # como Gemma4ClippableLinear: no es nn.Linear
        pass

    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.layers = nn.ModuleList([nn.Module()])
    root.language_model.layers[0].self_attn = nn.Module()
    root.language_model.layers[0].self_attn.q_proj = Clip()
    with pytest.raises(LoraError, match="no nn.Linear"):
        text_attention_targets(root, ("q_proj",))


def test_apply_lora_freezes_base_fp32_adapters_and_is_identity_at_init():
    bb = _tiny_backbone(torch.bfloat16)
    batch = _ids()
    with torch.no_grad():
        before = bb.encode(batch).pooled()
    targets = apply_lora(bb.model, LoraParams())
    lora = lora_parameters(bb.model)
    assert len(lora) == 2 * len(targets) and all(p.dtype == torch.float32 for p in lora.values())
    assert [n for n, p in bb.model.named_parameters() if p.requires_grad] == list(lora)
    assert all(float(p.detach().abs().max()) == 0 for n, p in lora.items() if "lora_B" in n)
    assert not bb.model.training
    with torch.no_grad():
        after = bb.encode(batch).pooled()
    assert torch.equal(before, after)  # B = 0: la inyección no cambia la representación
    with pytest.raises(LoraError, match="ya tiene"):
        apply_lora(bb.model, LoraParams())


def test_set_lora_mode_only_toggles_lora_dropout():
    bb = _tiny_backbone()
    apply_lora(bb.model, LoraParams())
    set_lora_mode(bb.model, training=True)
    drops = [m for n, m in bb.model.named_modules() if n.endswith("lora_dropout")]
    assert drops and all(m.training for m in drops)
    others = [m for n, m in bb.model.named_modules() if "lora_dropout" not in n]
    assert not any(m.training for m in others)
    set_lora_mode(bb.model, training=False)
    assert not any(m.training for m in bb.model.modules())


def test_lora_receives_real_gradients_and_base_stays_fixed():
    bb = _tiny_backbone()
    apply_lora(bb.model, LoraParams(dropout=0.0))
    base = {n: p.detach().clone() for n, p in bb.model.named_parameters() if ".lora_" not in n}
    lora = lora_parameters(bb.model)
    opt = torch.optim.AdamW(list(lora.values()), lr=1e-2)
    grads = []
    for _ in range(2):
        opt.zero_grad()
        bb.encode(_ids()).pooled().float().pow(2).sum().backward()
        grads.append({n: None if p.grad is None else float(p.grad.abs().sum()) for n, p in lora.items()})
        opt.step()
    # Paso 1: B recibe gradiente (A no, porque B = 0); paso 2, con B ≠ 0, también A.
    assert all(grads[0][n] > 0 for n in lora if "lora_B" in n)
    assert all(grads[1][n] > 0 for n in lora)
    assert all(math.isfinite(v) for g in grads for v in g.values())
    assert all(p.grad is None for n, p in bb.model.named_parameters() if ".lora_" not in n)
    assert all(torch.equal(p, base[n]) for n, p in bb.model.named_parameters() if ".lora_" not in n)


def test_group_loss_and_grads_equal_with_and_without_microbatch(fake_processor):
    """Integración §12: pérdida de pregunta igual con y sin microlote (FP32, CPU, tolerancia)."""
    texts = ["opción uno corta", "otra opción bastante más larga que la primera", "tercera", "cuarta a b c"]
    out = {}
    for mb in (1, 3, 4):
        bb = _tiny_backbone()
        apply_lora(bb.model, LoraParams(dropout=0.0))
        torch.manual_seed(1)
        for n, p in lora_parameters(bb.model).items():
            if "lora_B" in n:
                nn.init.normal_(p, std=0.05)  # B ≠ 0 para que A también tenga gradiente
        heads = DecisionHeads(bb.hidden_size)
        pooled, tokens = group_pooled(bb, fake_processor, texts, max_length=250, microbatch_rows=mb)
        z = heads("choice", pooled)
        loss = question_loss("choice", z, 2)
        loss.backward()
        out[mb] = (float(loss), {n: p.grad.clone() for n, p in lora_parameters(bb.model).items()}, tokens)
    ref_loss, ref_grads, ref_tokens = out[1]
    for mb in (3, 4):
        loss, grads, tokens = out[mb]
        assert tokens == ref_tokens  # el padding se recorta por microlote y no cuenta
        assert abs(loss - ref_loss) < 1e-5
        # Redondeo FP32 por otra forma de lote: tolerancia relativa al máximo de cada tensor.
        for n, g in grads.items():
            assert float((g - ref_grads[n]).abs().max()) <= 1e-4 * float(ref_grads[n].abs().max()) + 1e-7, n


def test_lr_schedule_warmup_and_linear_decay():
    f = [lr_factor(s, total=20, warmup=2) for s in range(20)]
    assert f[0] == 0.5 and f[1] == 1.0 and f[2] == 1.0
    assert all(a >= b for a, b in zip(f[2:], f[3:], strict=False))
    assert f[-1] == pytest.approx(1 / 18) and f[-1] > 0
    assert [lr_factor(s, total=5, warmup=0) for s in range(5)] == [1.0, 0.8, 0.6, 0.4, 0.2]
    assert epoch_permutation(10, 0, 1) == epoch_permutation(10, 0, 1) != epoch_permutation(10, 0, 2)


def test_lora_checkpoint_roundtrip_and_tamper(tmp_path):
    bb = _tiny_backbone()
    targets = apply_lora(bb.model, LoraParams())
    for p in lora_parameters(bb.model).values():
        with torch.no_grad():
            p.normal_(std=0.1)
    heads = DecisionHeads(bb.hidden_size)
    kw = dict(
        repo_id="google/gemma-4-E2B-it", revision="a" * 40, backbone_dtype="float32", prompt_template="t"
    )
    ckpt = save_lora_decision(tmp_path / "c", heads, lora_state(bb.model), {"targets": targets}, **kw)
    h2, state, manifest = load_lora_decision(
        ckpt, repo_id=kw["repo_id"], revision=kw["revision"], hidden_size=64
    )
    assert manifest["kind"] == "lora_decision_heads" and manifest["adapter"]["config"]["targets"] == targets
    bb2 = _tiny_backbone()
    apply_lora(bb2.model, LoraParams())
    load_lora_state(bb2.model, state)
    with torch.no_grad():
        assert torch.equal(bb.encode(_ids()).pooled(), bb2.encode(_ids()).pooled())
    with pytest.raises(LoraError, match="incompatible"):
        load_lora_state(bb2.model, {k: v for k, v in list(state.items())[1:]})
    (ckpt / "adapter.safetensors").write_bytes((ckpt / "adapter.safetensors").read_bytes() + b"x")
    with pytest.raises(CheckpointMismatchError, match="adaptador"):
        load_lora_decision(ckpt, repo_id=kw["repo_id"], revision=kw["revision"], hidden_size=64)


def test_temperature_recovers_known_scale_and_requires_enough_questions():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4000, generator=g) * 4
    y = (torch.rand(4000, generator=g) < torch.sigmoid(z / 2.0)).long().tolist()  # T verdadera = 2
    fit = fit_temperature("noul", [zi.reshape(1) for zi in z], y)
    assert abs(fit["temperature"] - 2.0) < 0.15 and fit["nll_after"] <= fit["nll_before"]
    zc = torch.randn(3000, 4, generator=g) * 3
    yc = [int(torch.multinomial(torch.softmax(r / 0.5, 0), 1, generator=g)) for r in zc]  # T = 0.5
    fit = fit_temperature("choice", list(zc), yc)
    assert abs(fit["temperature"] - 0.5) < 0.05

    class It:
        def __init__(self, p, t):
            self.primitive, self.target = p, t

    items = [It("noul", 1)] * (MIN_QUESTIONS - 1)
    out = fit_temperatures(items, [torch.tensor([1.0])] * len(items), ("noul", "choice"))
    assert out["noul"] == {"temperature": 1.0, "calibrated": False, "n": MIN_QUESTIONS - 1,
                           "reason": f"n < {MIN_QUESTIONS}"}  # fmt: skip
    assert out["choice"]["calibrated"] is False and out["choice"]["n"] == 0


def test_lora_init_is_seeded_and_independent_of_global_rng():
    """Regresión: la inicialización de lora_A dependía del RNG global al inyectar."""
    states = []
    for noise in (0, 7):
        bb = _tiny_backbone()
        torch.rand(noise)  # consume RNG global de forma distinta
        apply_lora(bb.model, LoraParams(), seed=3)
        states.append(lora_state(bb.model))
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    bb = _tiny_backbone()
    apply_lora(bb.model, LoraParams(), seed=4)
    assert any(not torch.equal(states[0][k], v) for k, v in lora_state(bb.model).items())


def test_layer_recomputation_gives_same_grads_and_keeps_base_in_eval():
    from gemma_system_one.models.lora import enable_layer_recomputation

    grads = {}
    for recompute in (False, True):
        bb = _tiny_backbone()
        apply_lora(bb.model, LoraParams(dropout=0.0), seed=0)
        lora = lora_parameters(bb.model)
        with torch.no_grad():
            for n, p in lora.items():
                if "lora_B" in n:
                    p.fill_(0.01)  # B ≠ 0 para que A también reciba gradiente
        if recompute:
            n_layers = enable_layer_recomputation(bb.model)
            assert n_layers == len(bb.model.language_model.layers)
            calls = []
            for layer in bb.model.language_model.layers:
                inner = layer._gradient_checkpointing_func

                def counted(*a, _inner=inner, _calls=calls, **k):
                    _calls.append(1)
                    return _inner(*a, **k)

                layer._gradient_checkpointing_func = counted
        set_lora_mode(bb.model, training=True)
        if recompute:
            marked = [m for m in bb.model.modules() if getattr(m, "_gso_recompute", False)]
            assert marked and all(m.training for m in marked)
            # Sólo la capa cambia de modo: atención, MLP y normas siguen en eval.
            others = [m for n, m in bb.model.named_modules() if "lora_dropout" not in n and m not in marked]
            assert not any(m.training for m in others)
        bb.encode(_ids()).pooled().float().pow(2).sum().backward()
        if recompute:
            assert len(calls) == n_layers  # cada capa pasó por torch.utils.checkpoint
        grads[recompute] = {n: p.grad.clone() for n, p in lora.items()}
        set_lora_mode(bb.model, training=False)
        assert not any(m.training for m in bb.model.modules())
    assert grads[False].keys() == grads[True].keys()
    for n in grads[False]:
        assert grads[True][n].abs().sum() > 0
        torch.testing.assert_close(grads[True][n], grads[False][n], rtol=1e-5, atol=1e-7)
