"""Fase 6: ¿agrupar las filas de una petición en un forward cambia las respuestas? (protocolo §Agrupación)

Para cada grupo (las preguntas de un caso, como una petición) de una partición que NO es test, extrae
las filas con la política vigente (1 fila por forward, decisión 0002) y con todas las filas de la
petición en un forward (padding derecho y orden por longitud: ``extract_pooled`` con
``microbatch_rows`` = filas). Misma carga, mismos cabezales y temperaturas. Informa |Δp| sobre todas
las probabilidades publicadas, cambios de decisión, NLL y el tiempo sincronizado de cada política.

Uso: uv run python scripts/measure_request_batching.py --checkpoint CKPT --calibration CAL \
       --split validation --out reports/phase6/batching_b2_validation.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gemma_system_one.calibration import load_calibration
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.split import load_split
from gemma_system_one.env import mask_home, snapshot_sources
from gemma_system_one.training.decisions import build_items, question_nll
from gemma_system_one.training.decisions_pipeline import _correct, _probs, load_decision_model

TOLERANCE = {"max_abs_dp": 0.02, "decision_changes": 0, "abs_delta_mean_nll": 0.005, "min_speedup": 1.5}


def _logits(heads, items, reps):
    out, start = [], 0
    for it in items:
        out.append(heads(it.primitive, reps[start : start + len(it.rows)]).double())
        start += len(it.rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--split", choices=["validation", "calibration"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    code = snapshot_sources()
    m = load_decision_model(args.checkpoint)
    ds = load_dataset(m.extra["dataset_root"])
    examples = load_split(ds, Path(m.extra["split_path"]), (args.split,))[args.split]
    examples = [e for e in examples if e.question.type in set(m.extra["primitives"])]
    temps = {"noul": 1.0, "choice": 1.0, "score": 1.0}
    if args.calibration is not None:
        temps.update(load_calibration(args.calibration, args.checkpoint)["temperatures"])
    heads = m.heads.cpu().eval()
    threshold = 0.5
    groups: dict[str, list] = defaultdict(list)
    for it in build_items(examples):
        groups[it.example.group_id].append(it)

    enc = m.encoder
    per_q, seconds = [], {"single": 0.0, "request": 0.0}
    forwards = {"single": 0, "request": 0}
    padding = {"single": 0, "request": 0}
    with torch.inference_mode():
        # Primera llamada fuera de la medida: carga (y LoRA) y calentamiento de kernels.
        warm = next(iter(groups.values()))
        enc.microbatch_rows = 1
        enc([r.text for it in warm for r in it.rows])
        for gid, items in groups.items():
            texts = [r.text for it in items for r in it.rows]
            res = {}
            # Orden alterno para no favorecer a una política con cachés calientes.
            order = ("single", "request") if len(per_q) % 2 == 0 else ("request", "single")
            for pol in order:
                enc.microbatch_rows = 1 if pol == "single" else len(texts)
                reps, st = enc(texts)
                seconds[pol] += st.seconds_synchronized
                forwards[pol] += st.backbone_forwards
                padding[pol] += st.padding_tokens
                res[pol] = _logits(heads, items, reps)
            for it, z1, zb in zip(items, res["single"], res["request"], strict=True):
                t = temps[it.primitive]
                p1, pb = np.array(_probs(it.primitive, z1 / t)), np.array(_probs(it.primitive, zb / t))
                per_q.append(
                    {
                        "id": it.example.id,
                        "group_id": gid,
                        "primitive": it.primitive,
                        "rows": len(it.rows),
                        "rows_in_request": len(texts),
                        "max_abs_dp": float(np.abs(p1 - pb).max()),
                        "max_abs_dlogit": float((z1 - zb).abs().max()),
                        # Decisión publicada: umbral en Noul, argmax en Choice/Score.
                        "decision_changed": bool((p1[0] >= threshold) != (pb[0] >= threshold))
                        if it.primitive == "noul"
                        else int(np.argmax(p1)) != int(np.argmax(pb)),
                        "correct_single": _correct(it, z1 / t, threshold),
                        "correct_request": _correct(it, zb / t, threshold),
                        "nll_single": question_nll(it.primitive, z1 / t, it.target),
                        "nll_request": question_nll(it.primitive, zb / t, it.target),
                    }
                )
    enc.microbatch_rows = 1
    dp = np.array([q["max_abs_dp"] for q in per_q])
    d_nll = float(np.mean([q["nll_request"] - q["nll_single"] for q in per_q]))
    speedup = seconds["single"] / seconds["request"] if seconds["request"] else None
    summary = {
        "questions": len(per_q),
        "requests": len(groups),
        "rows": int(sum(q["rows"] for q in per_q)),
        "max_abs_dp": float(dp.max()),
        "p95_abs_dp": float(np.quantile(dp, 0.95)),
        "median_abs_dp": float(np.median(dp)),
        "max_abs_dlogit": float(max(q["max_abs_dlogit"] for q in per_q)),
        "decision_changes": int(sum(q["decision_changed"] for q in per_q)),
        "mean_nll_single": float(np.mean([q["nll_single"] for q in per_q])),
        "mean_nll_request": float(np.mean([q["nll_request"] for q in per_q])),
        "delta_mean_nll": d_nll,
        "seconds_synchronized": {k: round(v, 3) for k, v in seconds.items()},
        "backbone_forwards": forwards,
        "padding_tokens": padding,
        "speedup_forward": None if speedup is None else round(speedup, 3),
    }
    by_prim = {}
    for prim in ("noul", "choice", "score"):
        sel = [q for q in per_q if q["primitive"] == prim]
        if sel:
            by_prim[prim] = {
                "n": len(sel),
                "max_abs_dp": max(q["max_abs_dp"] for q in sel),
                "decision_changes": sum(q["decision_changed"] for q in sel),
            }
    passed = (
        summary["max_abs_dp"] <= TOLERANCE["max_abs_dp"]
        and summary["decision_changes"] <= TOLERANCE["decision_changes"]
        and abs(d_nll) <= TOLERANCE["abs_delta_mean_nll"]
        and speedup is not None
        and speedup >= TOLERANCE["min_speedup"]
    )
    report = {
        "checkpoint": mask_home(args.checkpoint),
        "calibration": None if args.calibration is None else mask_home(args.calibration),
        "split": args.split,
        "temperatures": temps,
        "device": str(enc.backbone.device),
        "code": {k: code[k] for k in ("sha256", "files", "archive")},
        "tolerance_predeclared": TOLERANCE,
        "summary": summary,
        "by_primitive": by_prim,
        "adopt_as_option": passed,
        "per_question": per_q,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("summary", "by_primitive", "adopt_as_option")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
