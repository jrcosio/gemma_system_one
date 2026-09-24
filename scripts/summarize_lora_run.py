"""Resumen de un run de fase 3: historial por época, deriva del estandarizador, pasos, tiempos y memoria.

Uso: uv run python scripts/summarize_lora_run.py runs/pilot_lora_v3/<ts>
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def main(run: str) -> None:
    run_dir = Path(run)
    steps = [json.loads(x) for x in (run_dir / "steps.jsonl").read_text().splitlines() if x.strip()]
    history = [json.loads(x) for x in (run_dir / "history.jsonl").read_text().splitlines() if x.strip()]
    metrics = (
        json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").is_file() else {}
    )
    secs = [s["seconds"] for s in steps]
    out = {
        "run_dir": str(run_dir),
        "steps": len(steps),
        "step_seconds": {
            "median": st.median(secs),
            "p95": sorted(secs)[int(0.95 * (len(secs) - 1))],
            "total": round(sum(secs), 1),
        },  # fmt: skip
        "rows_per_second": sum(s["rows"] for s in steps) / sum(secs),
        "tokens_per_second": sum(s["tokens"] for s in steps) / sum(secs),
        "grad_norm_before_clip": {
            "median": st.median(s["grad_norm_before_clip"] for s in steps),
            "fraction_clipped": sum(s["grad_norm_before_clip"] > 1.0 for s in steps) / len(steps),
        },
        "epochs": [
            {
                "epoch": h["epoch"],
                "eval_nll": {k: (None if v is None else round(v, 4)) for k, v in h["eval_nll"].items()},
                "train_loss_mean_steps": h.get("train_loss_mean_steps"),
                "feature_drift": h["feature_drift"],
                "eval_seconds": h["eval_seconds"],
            }
            for h in history
        ],
        "selected_epoch": metrics.get("selected_epoch"),
        "base_params_unchanged_probe": metrics.get("base_params_unchanged_probe"),
        "lora": metrics.get("lora"),
        "memory": metrics.get("memory", {}).get("peaks_sampled"),
        "swap_delta_bytes": metrics.get("memory", {}).get("swap_delta_bytes"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
