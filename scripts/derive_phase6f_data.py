"""Datos de la fase 6f (protocolo ``reports/phase6f-protocol.md``). Deterministas.

Uso: uv run python scripts/derive_phase6f_data.py   (tras generar data/pilot_v5 con gso generate-data)
"""

from __future__ import annotations

import json
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.derive import exclude_overlap, fault_kind_triplets, write_derived
from gemma_system_one.data.generate_mixed import generate_mixed

PREVIOUS = [
    Path(f"data/{p}")
    for p in (
        "pilot_v3",
        "pilot_v3_holdout6",
        "pilot_v3_calib7",
        "pilot_v3_final8",
        "pilot_v4",
        "pilot_v4_calib12",
        "pilot_v4_final13",
        "pilot_v4_kdiag11",
        "pilot_v4_kdiag14",
        "pilot_v4_trip15",
        "pilot_v5",
    )
]


def build(seed: int, cases: int, out: Path, refs: list[Path], note: str):
    ds = [load_dataset(p) for p in refs]
    examples, audit = generate_mixed(cases, seed, version="v5")
    kept, excluded = exclude_overlap(examples, [e for d in ds for e in d.examples])
    meta = {
        "generator_version": "support-mixed-v5",
        "seed": seed,
        "n_cases": cases,
        "references": [
            {"dataset": p.as_posix(), "examples_sha256": d.sha256} for p, d in zip(refs, ds, strict=True)
        ],
        "excluded_groups": excluded,
        "note": note,
    }
    return write_derived(out, kept, meta), kept, audit


if __name__ == "__main__":
    out = {}
    out["calib16"], _, _ = build(
        16, 400, Path("data/pilot_v5_calib16"), PREVIOUS, "Fase 6f: sólo calibración."
    )
    refs = [*PREVIOUS, Path("data/pilot_v5_calib16")]
    out["final17"], _, _ = build(
        17, 300, Path("data/pilot_v5_final17"), refs, "Fase 6f: test final (regla S, b)."
    )
    refs = [*refs, Path("data/pilot_v5_final17")]
    out["trip18"], kept, audit = build(
        18, 1000, Path("data/pilot_v5_trip18"), refs, "Fase 6f: origen de los tríos."
    )
    k4, _ = fault_kind_triplets(kept, audit, seed=0, n_triplets=150, version="v5")
    src = {
        "derived_from": {
            "dataset": "data/pilot_v5_trip18",
            "examples_sha256": out["trip18"]["examples_sha256"],
        }
    }
    out["trip18_K4"] = write_derived(
        Path("data/pilot_v5_trip18_K4"), k4, {**src, "note": "150 tríos v5, K4."}
    )
    summary = {
        k: {
            "examples": v["examples"],
            "sha256": v["examples_sha256"],
            "excluded": len(v.get("excluded_groups", [])),
        }
        for k, v in out.items()
    }
    print(json.dumps(summary, indent=1))
