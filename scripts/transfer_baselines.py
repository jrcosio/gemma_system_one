# ruff: noqa: E501
"""Baselines (ajustados sólo con train) evaluados en un dataset externo de diagnóstico.

uv run python scripts/transfer_baselines.py configs/pilot_ce.yaml data/pilot_transfer_v2
No usa el backbone; mismas métricas que ``gso evaluate``.
"""

import json
import sys
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.training.decisions import build_items
from gemma_system_one.training.decisions_pipeline import (
    _filter,
    _load,
    baseline_logits,
    evaluate_logits,
    load_decision_config,
)

tc = load_decision_config(Path(sys.argv[1]))
_, _, train, _ = _load(tc)
ext = build_items(_filter(load_dataset(sys.argv[2]).examples, tc.primitives))
base = baseline_logits(train, {"external": ext})
out = {}
for name, per in base.items():
    m = evaluate_logits(ext, per["external"], tc.train.threshold)
    out[name] = {
        "mean_nll_all": m["mean_nll_all"],
        **{p: {k: m[p][k] for k in ("n", "nll", "accuracy")} for p in ("noul", "choice", "score") if p in m},
    }
print(json.dumps(out, indent=1))
