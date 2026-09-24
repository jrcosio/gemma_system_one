"""Datos de la fase 6e (protocolo ``reports/phase6e-protocol.md``). Deterministas.

Uso: uv run python scripts/derive_phase6e_data.py
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
    )
]

if __name__ == "__main__":
    refs = [load_dataset(p) for p in PREVIOUS]
    ref_meta = [
        {"dataset": p.as_posix(), "examples_sha256": d.sha256} for p, d in zip(PREVIOUS, refs, strict=True)
    ]
    examples, audit = generate_mixed(1000, 15, version="v4")
    kept, excluded = exclude_overlap(examples, [e for d in refs for e in d.examples])
    meta = {"generator_version": "support-mixed-v4", "seed": 15, "n_cases": 1000, "references": ref_meta}
    src = write_derived(Path("data/pilot_v4_trip15"), kept, {**meta, "excluded_groups": excluded})
    k4, k8 = fault_kind_triplets(kept, audit, seed=0, n_triplets=150)
    base = {"derived_from": {"dataset": "data/pilot_v4_trip15", "examples_sha256": src["examples_sha256"]}}
    out = {
        "trip15": src,
        "K4": write_derived(
            Path("data/pilot_v4_trip15_K4"), k4, {**base, "note": "150 tríos K4, misma pregunta."}
        ),
        "K8": write_derived(
            Path("data/pilot_v4_trip15_K8"), k8, {**base, "note": "Los mismos tríos con 4 distractoras."}
        ),
    }
    print(
        json.dumps(
            {k: {"examples": v["examples"], "sha256": v["examples_sha256"]} for k, v in out.items()}, indent=1
        )
    )
    print("excluded", len(excluded))
