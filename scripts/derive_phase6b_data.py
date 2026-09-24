"""Datos de la fase 6b (protocolo ``reports/phase6b-protocol.md``). Deterministas.

Uso: uv run python scripts/derive_phase6b_data.py   (tras generar data/pilot_v3_seed7 y data/pilot_v3_seed8)
  data/pilot_v3_seed7 -> data/pilot_v3_calib7  (sin estados de pilot_v3 ni del holdout de fase 6)
  data/pilot_v3_seed8 -> data/pilot_v3_final8  (sin estados de pilot_v3, holdout de fase 6 ni calib7)
"""

from __future__ import annotations

import json
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.derive import exclude_overlap, write_derived

REFS = [Path("data/pilot_v3"), Path("data/pilot_v3_holdout6")]


def derive(src: Path, out: Path, refs: list[Path], note: str) -> dict:
    ds = load_dataset(src)
    ref_ds = [load_dataset(r) for r in refs]
    kept, excluded = exclude_overlap(ds.examples, [e for r in ref_ds for e in r.examples])
    return write_derived(
        out,
        kept,
        {
            "derived_from": {"dataset": src.as_posix(), "examples_sha256": ds.sha256},
            "references": [
                {"dataset": r.as_posix(), "examples_sha256": d.sha256}
                for r, d in zip(refs, ref_ds, strict=True)
            ],
            "excluded_groups": excluded,
            "note": note,
        },
    )


if __name__ == "__main__":
    calib = derive(
        Path("data/pilot_v3_seed7"), Path("data/pilot_v3_calib7"), REFS, "Fase 6b: sólo calibración."
    )
    final = derive(
        Path("data/pilot_v3_seed8"),
        Path("data/pilot_v3_final8"),
        [*REFS, Path("data/pilot_v3_calib7")],
        "Fase 6b: test final independiente; una evaluación por modelo.",
    )
    print(
        json.dumps(
            {
                k: {x: v[x] for x in ("examples", "examples_sha256", "excluded_groups")}
                for k, v in (("calib7", calib), ("final8", final))
            },
            indent=1,
        )
    )
