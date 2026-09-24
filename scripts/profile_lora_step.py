"""Fase 6: coste de pasos LoRA representativos (memoria y tiempo).

No entrena un modelo completo: ejecuta N pasos de 8 preguntas de train (forward con autograd de todo
el grupo, pérdida por pregunta, backward acumulado, clip 1,0 y AdamW sobre LoRA + cabezales),
sincroniza MPS y muestrea la memoria tras cada pregunta. Los cabezales se inicializan al azar con
un solo LR de 1e-4 y no se aplica el scheduler de ``pilot_lora_v3``; los tiempos y el presupuesto
son un perfil de esta carga, no una validación del entrenamiento E4B completo.

Uso: uv run python scripts/profile_lora_step.py CONFIG DATASET OUT.json [pasos] [--recompute]

``--check-equivalence`` (con ``--recompute``): antes de los pasos, gradientes LoRA de la primera
pregunta con y sin recomputación (dropout desactivado) y su diferencia relativa.
``--recompute``: recomputación de activaciones por capa (``enable_layer_recomputation``). Si se supera
el presupuesto de memoria de la configuración, se registra el punto exacto y el script termina con 2:
no se sube el límite (spec §2.3).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.split import load_split, split_path
from gemma_system_one.env import snapshot_sources
from gemma_system_one.hub import require_snapshot
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.encoding import load_processor
from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.models.lora import (
    RECOMPUTE_MARK,
    LoraParams,
    apply_lora,
    enable_layer_recomputation,
    lora_parameters,
    set_lora_mode,
)
from gemma_system_one.resources import GIB, MemoryTracker, synchronize
from gemma_system_one.training.decisions import build_items, question_loss
from gemma_system_one.training.lora import question_logits_with_grad

QUESTIONS_PER_STEP = 8


def main(
    cfg_path: str, dataset: str, out: str, steps: int = 6, recompute: bool = False, check: bool = False
) -> int:
    code = snapshot_sources()
    cfg = load_config(cfg_path)
    snap = require_snapshot(cfg)
    ds = load_dataset(dataset)
    train = load_split(ds, split_path(dataset, 0), ("train",))["train"]
    items = build_items(train)[: steps * QUESTIONS_PER_STEP]
    tracker = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
    tracker.sample("start")
    t0 = time.perf_counter()
    bb = load_backbone(cfg, snap)
    processor = load_processor(snap)
    synchronize(bb.device)
    load_s = time.perf_counter() - t0
    tracker.sample("backbone_loaded")
    targets = apply_lora(bb.model, LoraParams(r=8, alpha=16, dropout=0.05), seed=0)
    lora = lora_parameters(bb.model)
    heads = DecisionHeads(bb.hidden_size).to(bb.device)
    params = [*lora.values(), *heads.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    recomputed_layers = enable_layer_recomputation(bb.model) if recompute else 0
    set_lora_mode(bb.model, training=True)  # base en eval(); sólo el dropout de LoRA entrena
    tracker.sample("lora_applied")
    equivalence = None
    if check:
        if not recompute:
            raise SystemExit("--check-equivalence requiere --recompute")
        equivalence = _recompute_equivalence(bb, processor, heads, lora, items, cfg)
    step_rows, failure = [], None
    try:
        _run_steps(bb, processor, heads, lora, params, opt, items, steps, cfg, tracker, step_rows)
    except RuntimeError as exc:
        if "Presupuesto de memoria" not in str(exc):
            raise
        last = tracker.samples[-1]
        failure = {
            "error": str(exc),
            "at": last["label"],
            "completed_steps": len(step_rows),
            "mps_driver_allocated_bytes": last.get("mps_driver_allocated_bytes"),
            "mps_current_allocated_bytes": last.get("mps_current_allocated_bytes"),
            "question": _CURRENT.get("question"),
        }
    report = _report(cfg_path, cfg, bb, targets, lora, code, load_s, step_rows, tracker, recomputed_layers)
    report["budget_exceeded"] = failure
    report["recompute_equivalence"] = equivalence
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=1))
    return 2 if failure else 0


_CURRENT: dict = {}


def _recompute_equivalence(bb, processor, heads, lora, items, cfg) -> dict:
    """Mismo forward/backward con y sin recomputación; LoRA con B ≠ 0 temporal para que A tenga gradiente."""
    item = next(it for it in items if it.primitive != "noul")
    drops = [m for n, m in bb.model.named_modules() if n.endswith("lora_dropout")]
    layers = [m for m in bb.model.modules() if getattr(m, RECOMPUTE_MARK, False)]
    saved = {k: p.detach().clone() for k, p in lora.items() if "lora_B" in k}
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for k in saved:
            lora[k].copy_(torch.randn(lora[k].shape, generator=gen) * 0.02)
    for m in drops:
        m.eval()
    grads = {}
    for flag in (True, False):
        for layer in layers:
            layer.gradient_checkpointing = flag
        for p in [*lora.values(), *heads.parameters()]:
            p.grad = None
        z, _ = question_logits_with_grad(
            bb, processor, heads, item, max_length=cfg.runtime.max_length, microbatch_rows=1
        )
        question_loss(item.primitive, z.float(), item.target).backward()
        grads[flag] = torch.cat([lora[k].grad.flatten().float().cpu() for k in sorted(lora)])
    for layer in layers:
        layer.gradient_checkpointing = True
    for m in drops:
        m.train()
    with torch.no_grad():
        for k, v in saved.items():
            lora[k].copy_(v)
    for p in [*lora.values(), *heads.parameters()]:
        p.grad = None
    g1, g0 = grads[True], grads[False]
    return {
        "question": item.example.id,
        "rows": len(item.rows),
        "rel_grad_diff": float((g1 - g0).norm() / g0.norm()),
        "grad_cosine": float(torch.nn.functional.cosine_similarity(g1, g0, dim=0)),
        "nonzero_grads": int((g1 != 0).sum()),
        "grad_elements": int(g1.numel()),
    }


def _run_steps(bb, processor, heads, lora, params, opt, items, steps, cfg, tracker, step_rows) -> None:
    for s in range(steps):
        batch = items[s * QUESTIONS_PER_STEP : (s + 1) * QUESTIONS_PER_STEP]
        synchronize(bb.device)
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        rows = tokens = 0
        for qi, it in enumerate(batch):
            _CURRENT["question"] = {"step": s, "index": qi, "id": it.example.id, "rows": len(it.rows)}
            z, n_tok = question_logits_with_grad(
                bb, processor, heads, it, max_length=cfg.runtime.max_length, microbatch_rows=1
            )
            (question_loss(it.primitive, z.float(), it.target) / len(batch)).backward()
            rows += len(it.rows)
            tokens += n_tok
            tracker.sample(f"step{s}_question")
        missing = [name for name, p in lora.items() if p.grad is None]
        if missing:
            raise RuntimeError(f"LoRA sin gradiente en el paso {s}: {missing[:5]}")
        grads = [p.grad for p in lora.values()]
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        if not finite:
            raise RuntimeError(f"Gradiente LoRA no finito en el paso {s}")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        synchronize(bb.device)
        step_rows.append(
            {
                "step": s,
                "seconds": round(time.perf_counter() - ts, 3),
                "rows": rows,
                "tokens": tokens,
                "lora_grads": len(grads),
                "grads_finite": finite,
            }
        )
        tracker.sample(f"step{s}_done")
        if bb.device.type == "mps" and s % 2 == 1:
            torch.mps.empty_cache()


def _report(cfg_path, cfg, bb, targets, lora, code, load_s, step_rows, tracker, recomputed_layers):
    # El primer paso incluye compilación/calentamiento de kernels: se informa aparte.
    timed = step_rows[1:] or step_rows
    summ = tracker.summary()
    return {
        "config": cfg_path,
        "repo_id": cfg.model.repo_id,
        "revision": cfg.model.revision,
        "hidden_size": bb.hidden_size,
        "device": str(bb.device),
        "dtype": cfg.model.dtype,
        "lora_targets": len(targets),
        "recomputed_layers": recomputed_layers,
        "lora_trainable_params": int(sum(p.numel() for p in lora.values())),
        "code": {k: code[k] for k in ("sha256", "files", "archive")},
        "load_seconds": round(load_s, 2),
        "steps": step_rows,
        "median_step_seconds_excl_first": statistics.median(r["seconds"] for r in timed) if timed else None,
        "rows_per_second_excl_first": round(
            sum(r["rows"] for r in timed) / sum(r["seconds"] for r in timed), 3
        )
        if timed
        else None,
        "memory": {
            k: summ[k]
            for k in ("sampling", "budget_bytes", "peaks_sampled", "over_budget", "swap_delta_bytes")
        },
    }


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(
        main(
            pos[0],
            pos[1],
            pos[2],
            *(int(x) for x in pos[3:]),
            recompute="--recompute" in flags,
            check="--check-equivalence" in flags,
        )
    )
