"""Fase 6f (regla H): diferencia emparejada por trío entre dos modelos en los mismos tríos.

Uso: uv run python scripts/compare_triplets.py OUT.json A.jsonl B.jsonl

Devuelve b − a de la proporción de tríos completos y de la accuracy (global y por papel), con
IC95 % por bootstrap de tríos (1000 repeticiones, semilla 0). Exige las mismas preguntas,
grupos y etiquetas en ambos ficheros.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_triplets import REPS, ROLES, SEED, load_triplets  # noqa: E402


def _pairing_keys(path: str) -> dict[str, tuple[str, str, int]]:
    keys = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                keys[row["id"]] = (row["group_id"], row["input_sha256"], row["target_index"])
            except KeyError as exc:
                raise ValueError(f"{path}: falta campo de emparejamiento {exc}") from exc
    return keys


def compare(path_a: str, path_b: str) -> dict:
    a, _ = load_triplets(path_a)
    b, _ = load_triplets(path_b)
    if set(a) != set(b):
        raise ValueError("Los ficheros no contienen los mismos tríos")
    keys_a, keys_b = _pairing_keys(path_a), _pairing_keys(path_b)
    if keys_a != keys_b:
        raise ValueError("Los ficheros no contienen las mismas entradas, grupos y etiquetas por ID")
    ids = sorted(a)
    metrics = {
        "full_triplets": lambda t: float(all(t.values())),
        "accuracy": lambda t: float(np.mean(list(t.values()))),
        **{f"role_{r}": (lambda t, r=r: float(t[r])) for r in sorted(ROLES)},
    }
    idx = np.random.default_rng(SEED).integers(0, len(ids), size=(REPS, len(ids)))
    out = {"triplets": len(ids), "bootstrap": {"unit": "trío", "reps": REPS, "seed": SEED}}
    for name, f in metrics.items():
        va = np.array([f(a[t]) for t in ids])
        vb = np.array([f(b[t]) for t in ids])
        d = vb - va
        bs = d[idx].mean(1)
        out[name] = {
            "a": round(float(va.mean()), 4),
            "b": round(float(vb.mean()), 4),
            "b_minus_a": round(float(d.mean()), 4),
            "ci95": [round(float(np.quantile(bs, q)), 4) for q in (0.025, 0.975)],
        }
    return out


if __name__ == "__main__":
    res = {"a": sys.argv[2], "b": sys.argv[3], **compare(sys.argv[2], sys.argv[3])}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
