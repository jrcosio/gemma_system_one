"""Auditoría pequeña: recarga de V/C en validación y comparación histórica sin reabrir test.

Desde la raíz: .venv/bin/python scripts/review_phase4_artifacts.py
No entrena ni descarga. Libera cada backbone antes de cargar el siguiente.
"""

import gc
import json
from pathlib import Path

import torch

from gemma_system_one.env import git_state
from gemma_system_one.metrics import compare_predictions
from gemma_system_one.training.decisions_pipeline import _reload_check, prepare_evaluation


def main():
    out = {"code": git_state(), "reload": {}}
    for root in ("runs/vision_heads/20260923T144256Z", "runs/vision_text_only/20260923T150230Z"):
        ckpt = Path(root) / "checkpoint"
        ctx = prepare_evaluation(ckpt, use_cache=False)
        items = ctx.items_for("validation")[:2]
        logits, ext = ctx.logits("validation", items)
        check = _reload_check(ckpt, "validation", items, logits)
        assert check["ok"], check
        out["reload"][root] = {"check": check, "extraction": ext, "device": str(ctx.encoder.backbone.device)}
        del ctx, items, logits
        gc.collect()
        torch.mps.empty_cache()
    historical = json.loads(Path("reports/phase4/test_compare_vision_minus_text.json").read_text())
    a, b = [[json.loads(x) for x in Path(historical[k]).read_text().splitlines()] for k in ("a", "b")]
    comparison = compare_predictions(a, b, allow_different_inputs=True)
    out["stored_test_comparison_matches"] = comparison == {k: historical[k] for k in comparison}
    assert out["stored_test_comparison_matches"]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
