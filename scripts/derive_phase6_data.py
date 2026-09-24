"""Datos derivados de la fase 6 (protocolo ``reports/phase6-protocol.md``). Deterministas.

Uso: uv run python scripts/derive_phase6_data.py
  data/pilot_v3_holdout6       -> data/pilot_v3_holdout6_clean     (sin estados presentes en pilot_v3)
  data/pilot_v3_holdout6_clean -> data/pilot_v3_holdout6_faultK    (sólo fault_type, opciones originales)
                               -> data/pilot_v3_holdout6_faultK8   (mismas preguntas, K ampliado hasta 8)
"""

from __future__ import annotations

import json
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.derive import exclude_overlap, widen_fault_kind, write_derived

SRC, REF = Path("data/pilot_v3_holdout6"), Path("data/pilot_v3")
CLEAN = Path("data/pilot_v3_holdout6_clean")
NARROW, WIDE = Path("data/pilot_v3_holdout6_faultK"), Path("data/pilot_v3_holdout6_faultK8")
WIDEN_SEED = 0

if __name__ == "__main__":
    src, ref = load_dataset(SRC), load_dataset(REF)
    kept, excluded = exclude_overlap(src.examples, ref.examples)
    base = {"derived_from": {"dataset": SRC.as_posix(), "examples_sha256": src.sha256}}
    clean = write_derived(
        CLEAN,
        kept,
        {
            **base,
            "excluded_groups": excluded,
            "reference": {"dataset": REF.as_posix(), "examples_sha256": ref.sha256},
            "note": "Holdout de fase 6: grupos con estado literal presente en la referencia excluidos.",
        },
    )
    fault = [e for e in kept if e.task_family == "fault_type"]
    cbase = {"derived_from": {"dataset": CLEAN.as_posix(), "examples_sha256": clean["examples_sha256"]}}
    narrow = write_derived(
        NARROW, fault, {**cbase, "note": "Sólo preguntas fault_type del holdout, sin cambios."}
    )
    wide = write_derived(
        WIDE,
        widen_fault_kind(kept, seed=WIDEN_SEED),
        {
            **cbase,
            "widen_seed": WIDEN_SEED,
            "target_k": 8,
            "note": "Mismas preguntas fault_type; opciones imposibles añadidas hasta K=8.",
        },
    )
    print(
        json.dumps(
            {
                "excluded_groups": excluded,
                "clean": clean["examples"],
                "narrow": narrow["examples"],
                "wide": wide["examples"],
                "sha": [clean["examples_sha256"], narrow["examples_sha256"], wide["examples_sha256"]],
            },
            indent=1,
        )
    )
