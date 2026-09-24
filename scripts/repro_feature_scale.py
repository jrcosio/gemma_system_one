# ruff: noqa: E501
"""Reproducción (decisión 0004): escala de las representaciones y variantes de optimización.

uv run python scripts/repro_feature_scale.py configs/pilot_ce.yaml [data/pilot_v1]

Usa sólo train y validación, y sólo representaciones ya cacheadas (falla si no están:
ejecutar antes ``gso train`` con esa configuración). No carga el backbone.
"""

import json
import sys
from pathlib import Path

from gemma_system_one.config import load_config
from gemma_system_one.features import RepresentationCache, backbone_fingerprint
from gemma_system_one.hub import require_snapshot
from gemma_system_one.training.decisions import DecisionTrainParams, flatten, train_decision_heads
from gemma_system_one.training.decisions_pipeline import _load, load_decision_config

tc = load_decision_config(Path(sys.argv[1]))
if len(sys.argv) > 2:
    tc = tc.model_copy(update={"dataset": Path(sys.argv[2])})
cfg = load_config(tc.base_config)
_, _, train, val = _load(tc)
cache = RepresentationCache(
    tc.cache_dir, backbone_fingerprint(cfg, require_snapshot(cfg), tc.extraction.microbatch_rows)
)


def reps(items):
    _, hashes, offsets = flatten(items)
    r = cache.get(hashes)
    if r is None:
        raise SystemExit("Representaciones no cacheadas: ejecuta gso train con esta configuración")
    return r, offsets


rt, ot = reps(train)
rv, ov = reps(val)
std, mean = rt.std(0), rt.mean(0)
print(json.dumps({
    "row_norm_mean": float(rt.norm(dim=1).mean()), "row_norm_max": float(rt.norm(dim=1).max()),
    "l1_mean": float(rt.abs().sum(1).mean()), "dim_absmean_max": float(rt.abs().mean(0).max()),
    "dim_absmean_median": float(rt.abs().mean(0).median()), "dim_std_max": float(std.max()),
    "dim_std_median": float(std.median()),
}))  # fmt: skip
for name, lr, fn in (
    ("raw", 1e-3, "none"),
    ("raw", 1e-4, "none"),
    ("std", 1e-3, "standardize_train"),
    ("std", 1e-4, "standardize_train"),
):
    p = DecisionTrainParams(
        epochs=10, lr=lr, weight_decay=0.01, questions_per_step=8, logit_chunk_rows=4, seed=0, feature_norm=fn
    )
    res = train_decision_heads(train, rt, ot, p, (val, rv, ov))
    tr = [round(h["train_nll"]["all"], 3) for h in res.history]
    va = [round(h["validation_nll"]["all"], 3) for h in res.history]
    print(
        f"{name}_lr{lr:g}: selected={res.selected_epoch} best_val={min(va):.4f}\n  train={tr}\n  val  ={va}"
    )
