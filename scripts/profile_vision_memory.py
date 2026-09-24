"""Memoria y tiempo reales de una pregunta con imagen en E2B/MPS: extracción sin gradiente y
backward LoRA (grafo del grupo completo), frente a la misma pregunta sin imagen.

Uso: uv run python scripts/profile_vision_memory.py configs/e2b_vision.yaml data/vision_pilot_v1
Sólo lee preguntas de train. Muestras de memoria en puntos instrumentados, no picos exactos.
"""

from __future__ import annotations

import json
import sys
import time

import torch

from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.split import load_split, split_path
from gemma_system_one.hub import require_snapshot
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.encoding import load_processor
from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.models.lora import LoraParams, apply_lora, lora_parameters, set_lora_mode
from gemma_system_one.resources import memory_snapshot
from gemma_system_one.training.decisions import build_items, question_loss
from gemma_system_one.training.lora import group_pooled

GB = 1e9


def main(cfg_path: str, dataset: str) -> None:
    cfg = load_config(cfg_path)
    snap = require_snapshot(cfg)
    ds = load_dataset(dataset)
    train = load_split(ds, split_path(dataset, 0), ("train",))["train"]
    item = max(
        (it for it in build_items(train, ds.root) if it.primitive != "noul"), key=lambda it: len(it.rows)
    )
    bb = load_backbone(cfg, snap)
    processor = load_processor(snap)
    loaded = memory_snapshot()
    apply_lora(bb.model, LoraParams(), seed=0)
    heads = DecisionHeads(bb.hidden_size).to(bb.device)
    after_load = {k: loaded.get(k) for k in ("mps_current_allocated_bytes", "mps_driver_allocated_bytes")}
    out = {"question_rows": len(item.rows), "primitive": item.primitive, "after_load": after_load}
    for label, image in (("with_image", item.image_file), ("without_image", None)):
        kw = {"max_length": cfg.runtime.max_length, "microbatch_rows": 1, "image_file": image}
        texts = [r.text for r in item.rows]
        set_lora_mode(bb.model, training=False)
        torch.mps.empty_cache()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            group_pooled(bb, processor, texts, **kw)
        torch.mps.synchronize()
        t_nograd = time.perf_counter() - t0
        set_lora_mode(bb.model, training=True)
        for p in lora_parameters(bb.model).values():
            p.grad = None
        torch.mps.empty_cache()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        pooled, tokens = group_pooled(bb, processor, texts, **kw)
        loss = question_loss(item.primitive, heads(item.primitive, pooled), item.target)
        torch.mps.synchronize()
        graph = torch.mps.current_allocated_memory()
        loss.backward()
        torch.mps.synchronize()
        t_step = time.perf_counter() - t0
        snap_after = memory_snapshot()
        out[label] = {
            "valid_tokens": tokens,
            "seconds_forward_no_grad": round(t_nograd, 3),
            "seconds_forward_backward": round(t_step, 3),
            "mps_current_with_graph_bytes": graph,
            "graph_over_weights_gb": round((graph - loaded["mps_current_allocated_bytes"]) / GB, 3),
            "mps_driver_after_backward_bytes": snap_after["mps_driver_allocated_bytes"],
            "process_rss_bytes": snap_after["process_rss_bytes"],
            "swap_used_bytes": snap_after["swap_used_bytes"],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
