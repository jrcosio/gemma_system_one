"""Inspección manual de predicciones (spec §10: 50–100 antes de ampliar).

uv run python scripts/inspect_predictions.py <predictions.jsonl> <dataset_root> [n_correct]
Muestra todos los errores y una muestra determinista de aciertos por tipo.
"""

import json
import random
import sys

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.serialization import canonical_state

with open(sys.argv[1], encoding="utf-8") as fh:
    preds = [json.loads(line) for line in fh]
ds = load_dataset(sys.argv[2]).by_id()
n_ok = int(sys.argv[3]) if len(sys.argv) > 3 else 20
errors = [p for p in preds if not p["correct"]]
rng = random.Random(0)
ok = rng.sample([p for p in preds if p["correct"]], min(n_ok, sum(p["correct"] for p in preds)))
print(f"predicciones={len(preds)} errores={len(errors)} muestra_aciertos={len(ok)}")
for title, group in (("ERRORES", errors), ("ACIERTOS (muestra)", ok)):
    print(f"\n===== {title} =====")
    for p in sorted(group, key=lambda x: (x["type"], x["task_family"], -x["nll"])):
        e = ds[p["id"]]
        a = p["answer"]
        if a["type"] == "noul":
            pred = f"p={a['noul']:.3f}"
        elif a["type"] == "choice":
            probs = sorted(((v, e.question.criteria[k]) for k, v in a["probabilities"].items()), reverse=True)
            pred = "; ".join(f"{v:.2f} {desc}" for v, desc in probs[:3])
        else:
            pred = f"score={a['score']:.2f} p=" + ",".join(f"{x:.2f}" for x in a["probabilities"])
        print(f"[{p['type']}/{p['task_family']}] nll={p['nll']:.3f} objetivo={p['target_description']!r}")
        print(f"   Q: {e.question.instructions[:150]}")
        print(f"   S: {canonical_state(e.state)[:260]}")
        print(f"   pred: {pred}")
