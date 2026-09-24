# ruff: noqa: E501  (líneas de formato de informe)
"""Resumen legible de un run de fase 2: uv run python scripts/summarize_decisions_run.py runs/<name>/<ts>"""

import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
d = json.loads((run / "metrics.json").read_text())
print(f"run={run} época={d['selected_epoch']} ({d['select_on']}) counts={d['counts']}")
for e in d["extraction"]:
    print(
        f"  extracción {e['split']}: preguntas={e['questions']} filas={e['rows']} forwards={e['backbone_forwards']} "
        f"tokens={e['valid_tokens']} s={e['seconds_synchronized']} cache={e['cache_hit']}"
    )
print("  memoria:", d["memory"]["peaks_sampled"], "swapΔ", d["memory"]["swap_delta_bytes"])
for line in (run / "history.jsonl").read_text().splitlines():
    h = json.loads(line)
    print(
        f"  época {h['epoch']}: train nll={h['train_nll']['all']:.4f} val nll={h['validation_nll']['all']:.4f} "
        f"(noul {h['validation_nll']['noul']:.3f} choice {h['validation_nll']['choice']:.3f} score {h['validation_nll']['score']:.3f})"
    )
for model, per in d["metrics"]["models"].items():
    v = per["validation"]
    out = [f"{model:12s} val nll_all={v['mean_nll_all']:.4f}"]
    if "noul" in v:
        n = v["noul"]
        out.append(
            f"noul n={n['n']} nll={n['nll']:.3f} brier={n['brier']:.3f} acc={n['accuracy']:.3f} ece={n['ece_event_15bins']:.3f} f1={n['f1']} pair={n['within_group_pairwise']['pairwise_accuracy']}"
        )
    if "choice" in v:
        c = v["choice"]
        out.append(
            f"choice n={c['n']} nll={c['nll']:.3f} brier={c['brier_sum']:.3f} acc={c['accuracy']:.3f} ece={c['ece_top_label_15bins']:.3f} conc={c['mean_concentration']:.2f}"
        )
    if "score" in v:
        s = v["score"]
        out.append(
            f"score n={s['n']} nll={s['nll']:.3f} rps={s['rps']:.3f} mae={s['mae']:.3f} maeN={s['mae_norm']:.3f} acc={s['accuracy']:.3f} cumECE={s['cumulative_event_ece_15bins']:.3f}"
        )
    print("  " + " | ".join(out))
    for p in ("noul", "choice", "score"):
        if p in v:
            fam = {
                k: (x["n"], round(x["nll"], 3), round(x["accuracy"], 3))
                for k, x in v[p]["by_task_family"].items()
            }
            print(f"      {p} por familia (n, nll, acc): {fam}")
    if "choice" in v:
        print(
            "      choice por K:",
            {
                k: (x["n"], round(x["nll"], 3), round(x["accuracy"], 3), round(x["uniform_nll"], 3))
                for k, x in v["choice"]["by_k"].items()
            },
        )
    if "score" in v:
        print(
            "      score por M:",
            {
                k: (x["n"], round(x["nll"], 3), round(x["accuracy"], 3), round(x["mae"], 3))
                for k, x in v["score"]["by_m"].items()
            },
        )
b = d["metrics"]["validation_bootstrap"]
print(f"bootstrap: grupos={b['groups']} reps={b['reps']} seed={b['seed']}")
for k in sorted(b["intervals"]):
    if k.startswith(("gemma_heads", "bow.", "prior.")) and (
        k.endswith("nll") or k.endswith("accuracy") or k.endswith("nll_all")
    ):
        lo, hi = b["intervals"][k]
        print(f"  {k}: [{lo:.4f}, {hi:.4f}]")
