"""Datos de la fase 6d (protocolo ``reports/phase6d-protocol.md``). Deterministas.

Uso: uv run python scripts/derive_phase6d_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.derive import (
    balanced_fault_kind_pairs,
    exclude_overlap,
    swap_states,
    write_derived,
)
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
    )
]

if __name__ == "__main__":
    refs = [load_dataset(p) for p in PREVIOUS]
    ref_meta = [
        {"dataset": p.as_posix(), "examples_sha256": d.sha256} for p, d in zip(PREVIOUS, refs, strict=True)
    ]
    examples, audit = generate_mixed(400, 14, version="v4")
    kept, excluded = exclude_overlap(examples, [e for d in refs for e in d.examples])
    base = {"generator_version": "support-mixed-v4", "seed": 14, "n_cases": 400, "references": ref_meta}
    out = {
        "kdiag14": write_derived(Path("data/pilot_v4_kdiag14"), kept, {**base, "excluded_groups": excluded})
    }
    src = {
        "derived_from": {
            "dataset": "data/pilot_v4_kdiag14",
            "examples_sha256": out["kdiag14"]["examples_sha256"],
        }
    }
    k4, k8 = balanced_fault_kind_pairs(kept, audit, seed=0)
    for name, exs, note in (
        ("K4", k4, "Composición equilibrada: none, other y dos categorías reales."),
        ("K8", k8, "K4 más las cuatro distractoras."),
        ("K4swap", swap_states(k4, seed=0), "K4 con estados intercambiados y etiquetas originales."),
        ("K8swap", swap_states(k8, seed=0), "K8 con estados intercambiados y etiquetas originales."),
    ):
        out[name] = write_derived(Path(f"data/pilot_v4_kdiag14_{name}"), exs, {**src, "note": note})
    f13 = load_dataset("data/pilot_v4_final13")
    fault = [e for e in f13.examples if e.task_family == "fault_type"]
    out["final13_faultswap"] = write_derived(
        Path("data/pilot_v4_final13_faultswap"),
        swap_states(fault, seed=0),
        {"derived_from": {"dataset": "data/pilot_v4_final13", "examples_sha256": f13.sha256},
         "note": "fault_type de final13 con estados intercambiados (control descriptivo)."},
    )  # fmt: skip
    print(
        json.dumps(
            {k: {"examples": v["examples"], "sha256": v["examples_sha256"]} for k, v in out.items()}, indent=1
        )
    )
    print("excluded", len(excluded))
