"""Datos de la fase 6c (protocolo ``reports/phase6c-protocol.md``). Deterministas.

Uso: uv run python scripts/derive_phase6c_data.py
Genera con el generador v4 (sin la pista de K) y excluye los grupos cuyo estado literal aparece en
cualquier conjunto anterior. ``pilot_v4`` se genera aparte con ``gso generate-data`` y ``gso split``.
"""

from __future__ import annotations

import json
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.derive import exclude_overlap, widen_fault_kind_by_facts, write_derived
from gemma_system_one.data.generate_mixed import generate_mixed

PREVIOUS = [
    Path(p)
    for p in (
        "data/pilot_v3",
        "data/pilot_v3_holdout6",
        "data/pilot_v3_calib7",
        "data/pilot_v3_final8",
        "data/pilot_v4",
    )
]
WIDEN_SEED = 0


def _refs(paths: list[Path]) -> tuple[list, list[dict]]:
    ds = [load_dataset(p) for p in paths]
    return [e for d in ds for e in d.examples], [
        {"dataset": p.as_posix(), "examples_sha256": d.sha256} for p, d in zip(paths, ds, strict=True)
    ]


def build(seed: int, cases: int, out: Path, refs: list[Path], note: str):
    examples, audit = generate_mixed(cases, seed, version="v4")
    ref_ex, ref_meta = _refs(refs)
    kept, excluded = exclude_overlap(examples, ref_ex)
    meta = {
        "generator_version": "support-mixed-v4",
        "variant": "main",
        "seed": seed,
        "n_cases": cases,
        "references": ref_meta,
        "excluded_groups": excluded,
        "note": note,
    }
    return write_derived(out, kept, meta), kept, audit


if __name__ == "__main__":
    summary = {}
    cal, _, _ = build(12, 400, Path("data/pilot_v4_calib12"), PREVIOUS, "Fase 6c: sólo calibración.")
    summary["calib12"] = cal
    refs = [*PREVIOUS, Path("data/pilot_v4_calib12")]
    fin, _, _ = build(13, 300, Path("data/pilot_v4_final13"), refs, "Fase 6c: test final de la regla S.")
    summary["final13"] = fin
    refs = [*refs, Path("data/pilot_v4_final13")]
    kd, kept, audit = build(
        11, 300, Path("data/pilot_v4_kdiag11"), refs, "Fase 6c: origen del diagnóstico K."
    )
    summary["kdiag11"] = kd
    base = {"derived_from": {"dataset": "data/pilot_v4_kdiag11", "examples_sha256": kd["examples_sha256"]}}
    fault = [e for e in kept if e.task_family == "fault_type"]
    summary["kdiag11_K"] = write_derived(
        Path("data/pilot_v4_kdiag11_K"), fault, {**base, "note": "fault_type con su K original (3–6)."}
    )
    summary["kdiag11_K8"] = write_derived(
        Path("data/pilot_v4_kdiag11_K8"),
        widen_fault_kind_by_facts(kept, audit, seed=WIDEN_SEED),
        {**base, "widen_seed": WIDEN_SEED, "note": "Mismas preguntas ampliadas por hechos a K = 8."},
    )
    print(
        json.dumps(
            {
                k: {x: v[x] for x in ("examples", "examples_sha256") if x in v}
                | {"excluded": len(v.get("excluded_groups", []))}
                for k, v in summary.items()
            },
            indent=1,
        )
    )
