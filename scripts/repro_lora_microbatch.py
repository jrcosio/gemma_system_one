"""Pérdida y gradientes LoRA de grupo con microlotes frente a una fila por forward, en E2B real.

Uso: uv run python scripts/repro_lora_microbatch.py configs/e2b_text.yaml data/pilot_v3 [n_preguntas]

Sólo lee preguntas Choice/Score de train. LoRA con B≠0 fijo (semilla) para que A también tenga
gradiente; cabezal Choice/Score aleatorio fijo. Mide |ΔL| y la diferencia relativa del gradiente
(norma de la diferencia / norma de referencia) para microbatch_rows = K frente a 1.
"""

from __future__ import annotations

import json
import sys

import torch

from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.split import load_split, split_path
from gemma_system_one.hub import require_snapshot
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.encoding import load_processor
from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.models.lora import LoraParams, apply_lora, lora_parameters
from gemma_system_one.training.decisions import build_items, question_loss
from gemma_system_one.training.lora import group_pooled


def main(cfg_path: str, dataset: str, n: int = 12) -> None:
    cfg = load_config(cfg_path)
    snap = require_snapshot(cfg)
    ds = load_dataset(dataset)
    train = load_split(ds, split_path(dataset, 0), ("train",))["train"]
    items = [it for it in build_items(train) if it.primitive != "noul"][:n]
    bb = load_backbone(cfg, snap)
    processor = load_processor(snap)
    apply_lora(bb.model, LoraParams(dropout=0.0), seed=0)
    lora = lora_parameters(bb.model)
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in lora.items():
            if "lora_B" in name:
                p.normal_(std=0.02)
    heads = DecisionHeads(bb.hidden_size).to(bb.device)
    rows = []
    for it in items:
        res = {}
        for mb in (1, len(it.rows)):
            for p in lora.values():
                p.grad = None
            pooled, tokens = group_pooled(
                bb,
                processor,
                [r.text for r in it.rows],
                max_length=cfg.runtime.max_length,
                microbatch_rows=mb,
            )
            loss = question_loss(it.primitive, heads(it.primitive, pooled), it.target)
            loss.backward()
            g = torch.cat([lora[k].grad.flatten().float().cpu() for k in sorted(lora)])
            res[mb] = (float(loss), g, tokens)
        (l1, g1, t1), (lk, gk, tk) = res[1], res[len(it.rows)]
        rows.append(
            {
                "type": it.primitive,
                "k": len(it.rows),
                "loss_1": l1,
                "abs_loss_diff": abs(lk - l1),
                "rel_grad_diff": float((gk - g1).norm() / g1.norm()),
                "grad_cosine": float(torch.nn.functional.cosine_similarity(gk, g1, dim=0)),
                "valid_tokens": [t1, tk],
            }
        )
    summary = {
        "questions": len(rows),
        "device": str(bb.device),
        "dtype": cfg.model.dtype,
        "max_abs_loss_diff": max(r["abs_loss_diff"] for r in rows),
        "max_rel_grad_diff": max(r["rel_grad_diff"] for r in rows),
        "min_grad_cosine": min(r["grad_cosine"] for r in rows),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:]))
