# Uso: uv run python scripts/repro_batch_effect_probs.py
# Requiere pesos E2B en caché y un run de configs/noul_smoke.yaml.
"""Reproducción (decisión 0002): efecto del tamaño de microlote en probabilidades y decisiones."""

import glob
import json

import torch

from gemma_system_one.checkpoint import load_head
from gemma_system_one.config import load_config
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.split import load_split, split_path
from gemma_system_one.features import extract_pooled
from gemma_system_one.hub import require_snapshot
from gemma_system_one.models.backbone import load_backbone
from gemma_system_one.models.encoding import load_processor
from gemma_system_one.serialization import expand_example

cfg = load_config("configs/e2b_text.yaml")
snap = require_snapshot(cfg)
bb = load_backbone(cfg, snap)
proc = load_processor(snap)
ds = load_dataset("data/smoke_v1")
parts = load_split(ds, split_path("data/smoke_v1", 0), ("train", "validation"))
ck = sorted(glob.glob("runs/noul_smoke/*/checkpoint"))[-1]
head, _ = load_head(ck, repo_id=cfg.model.repo_id, revision=cfg.model.revision, hidden_size=1536)
res = {}
for split in ("train", "validation"):
    texts = [expand_example(e)[0].text for e in parts[split]]
    r = {mb: extract_pooled(bb, proc, texts, max_length=512, microbatch_rows=mb)[0] for mb in (1, 4, 8)}
    with torch.inference_mode():
        p = {mb: torch.sigmoid(head(v)) for mb, v in r.items()}
    res[split] = {
        f"mb{mb}_vs_mb1": {
            "max_abs_prob_diff": float((p[mb] - p[1]).abs().max()),
            "decision_flips@0.5": int(((p[mb] >= 0.5) != (p[1] >= 0.5)).sum()),
            "max_abs_logit_diff": float(
                (
                    torch.logit(p[mb].double().clamp(1e-12, 1 - 1e-12))
                    - torch.logit(p[1].double().clamp(1e-12, 1 - 1e-12))
                )
                .abs()
                .max()
            ),
        }
        for mb in (4, 8)
    }
    res[split]["n"] = len(texts)
print(json.dumps(res, indent=1))
