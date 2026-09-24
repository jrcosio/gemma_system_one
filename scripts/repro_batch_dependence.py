# Uso: uv run python scripts/repro_batch_dependence.py
# Requiere pesos E2B en caché y runs de configs/noul_smoke.yaml y configs/noul_overfit.yaml.
"""Reproducción: dependencia de la representación bf16 (MPS) respecto a la composición del microlote."""

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
ex = parts["train"][:16]
texts = [expand_example(e)[0].text for e in ex]
ref, _ = extract_pooled(bb, proc, texts, max_length=512, microbatch_rows=8)  # composición de entrenamiento
alone, _ = extract_pooled(bb, proc, texts, max_length=512, microbatch_rows=1)  # sin padding
mixed_texts = texts + [expand_example(e)[0].text for e in parts["train"][16:40]]
mixed, _ = extract_pooled(bb, proc, mixed_texts, max_length=512, microbatch_rows=8)  # otra composición
mixed = mixed[:16]
again, _ = extract_pooled(bb, proc, texts, max_length=512, microbatch_rows=8)


def cmp(a, b):
    d = (a - b).abs().max(dim=1).values
    return {
        "max_abs": float(d.max()),
        "min_cos": float(torch.cosine_similarity(a, b, dim=1).min()),
        "rows_identical": int((d == 0).sum()),
        "ref_norm_mean": float(a.norm(dim=1).mean()),
    }


out = {
    "repeat_same_composition": cmp(ref, again),
    "ref_vs_alone": cmp(ref, alone),
    "ref_vs_other_composition": cmp(ref, mixed),
}
for name in ("noul_smoke", "noul_overfit"):
    ck = sorted(glob.glob(f"runs/{name}/*/checkpoint"))[-1]
    head, _ = load_head(ck, repo_id=cfg.model.repo_id, revision=cfg.model.revision, hidden_size=1536)
    with torch.inference_mode():
        z = {k: head(v) for k, v in {"ref": ref, "alone": alone, "mixed": mixed}.items()}
    out[f"head_{name}"] = {
        "weight_norm": float(head.proj.weight.detach().norm()),
        "logit_absmax_ref": float(z["ref"].abs().max()),
        "max_logit_diff_alone": float((z["ref"] - z["alone"]).abs().max()),
        "max_logit_diff_other_composition": float((z["ref"] - z["mixed"]).abs().max()),
        "max_prob_diff_other_composition": float(
            (torch.sigmoid(z["ref"]) - torch.sigmoid(z["mixed"])).abs().max()
        ),
    }
print(json.dumps(out, indent=1))
