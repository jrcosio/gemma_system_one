"""Fase 6e: métricas de tríos a partir de las predicciones de ``gso evaluate`` (sólo lectura).

Uso: uv run python scripts/analyze_triplets.py OUT.json NOMBRE=PREDICCIONES.jsonl [...]

Por fichero: accuracy global y por papel (real/other/none), proporción de tríos con los tres
aciertos e IC95 % por bootstrap de tríos (1000 repeticiones, semilla 0). Techo sin leer el estado:
accuracy 1/3 y 0 tríos completos (las tres preguntas de un trío son idénticas salvo el estado).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

REPS, SEED = 1000, 0
ROLES = frozenset(("real", "other", "none"))


def load_triplets(path: str) -> tuple[dict[str, dict[str, bool]], int]:
    """Aciertos por trío y papel, validados; y el número de filas leídas."""
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    trip: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in rows:
        tid, role = r["id"].rsplit("-", 1)
        if role not in ROLES or role in trip[tid] or type(r["correct"]) is not bool:
            raise ValueError(f"{path}: papel duplicado/desconocido o 'correct' no booleano: {r['id']}")
        trip[tid][role] = r["correct"]
    if not trip or any(set(v) != ROLES for v in trip.values()):
        raise ValueError(f"{path}: tríos vacíos o incompletos")
    return dict(trip), len(rows)


def analyze(path: str) -> dict:
    trip, n_rows = load_triplets(path)
    rows = range(n_rows)
    ids = sorted(trip)
    full = np.array([all(trip[t].values()) for t in ids], float)
    acc = np.array([np.mean(list(trip[t].values())) for t in ids], float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(REPS, len(ids)))
    ci = lambda x: [round(float(np.quantile(x[idx].mean(1), q)), 4) for q in (0.025, 0.975)]  # noqa: E731
    return {
        "triplets": len(ids),
        "questions": len(rows),
        "accuracy": round(float(acc.mean()), 4),
        "accuracy_ci": ci(acc),
        "by_role": {
            role: round(float(np.mean([trip[t][role] for t in ids])), 4) for role in ("real", "other", "none")
        },
        "full_triplets": round(float(full.mean()), 4),
        "full_triplets_ci": ci(full),
        "ceiling_without_state": {"accuracy": round(1 / 3, 4), "full_triplets": 0.0},
    }


if __name__ == "__main__":
    out = {name: analyze(path) for name, path in (a.split("=", 1) for a in sys.argv[2:])}
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump({"bootstrap": {"unit": "trío", "reps": REPS, "seed": SEED}, "results": out}, fh, indent=1)
    print(json.dumps(out, indent=1))
