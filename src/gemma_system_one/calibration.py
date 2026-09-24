"""Temperatura posterior por primitiva (spec §6.2).

- Pesos congelados: se leen los logits del checkpoint sobre la partición ``calibration``
  (y sólo esa) y se ajusta ``T = softplus(t) + ε`` por primitiva minimizando la NLL
  (BCE en Noul; entropía cruzada del grupo completo en Choice/Score). Una T por primitiva,
  nunca por consulta.
- Con menos de ``MIN_QUESTIONS`` preguntas de una primitiva: T = 1 y ``calibrated: false``.
- El artefacto es inmutable y queda vinculado al checkpoint (sha256 del manifiesto y de los
  pesos), al dataset y al split. Las métricas que incluye son *in-sample* (misma partición
  usada para ajustar); la medición independiente se hace en test.
- Fase 6b: también sobre un conjunto de calibración **externo** (``--dataset``, split ``all``) cuando
  la partición interna es pequeña. Se exige que no comparta grupos ni entradas con el dataset de
  entrenamiento; el artefacto registra su sha256 y ``evaluate`` rechaza evaluar sobre ese mismo
  conjunto con esas temperaturas.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .checkpoint import checkpoint_identity
from .env import git_state
from .training.decisions import QuestionItem, question_loss

CALIBRATION_FORMAT = 1
TEMPERATURE_EPS = 0.01
MIN_QUESTIONS = 30
METHOD = "temperature_softplus_lbfgs_v1"


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


def calibrated_nll(
    primitive: str, logits: list[torch.Tensor], targets: list[int], temperature
) -> torch.Tensor:
    losses = [
        question_loss(primitive, z.double() / temperature, y) for z, y in zip(logits, targets, strict=True)
    ]
    return torch.stack(losses).mean()


def fit_temperature(primitive: str, logits: list[torch.Tensor], targets: list[int]) -> dict[str, Any]:
    """T que minimiza la NLL media de ``primitive``; parte de T = 1."""
    if not logits:
        raise ValueError("Sin preguntas para ajustar la temperatura")
    t = torch.tensor(_inv_softplus(1.0 - TEMPERATURE_EPS), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.5, max_iter=200, tolerance_grad=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = calibrated_nll(primitive, logits, targets, torch.nn.functional.softplus(t) + TEMPERATURE_EPS)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(torch.nn.functional.softplus(t.detach()) + TEMPERATURE_EPS)
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"Temperatura no válida para {primitive}: {temperature}")
    with torch.no_grad():
        before = float(calibrated_nll(primitive, logits, targets, 1.0))
        after = float(calibrated_nll(primitive, logits, targets, temperature))
    if after > before + 1e-9:
        raise RuntimeError(f"La temperatura empeora la NLL de ajuste en {primitive}: {before} -> {after}")
    return {"temperature": temperature, "nll_before": before, "nll_after": after, "n": len(logits)}


def fit_temperatures(items: list[QuestionItem], logits: list[torch.Tensor], primitives) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in primitives:
        idx = [i for i, it in enumerate(items) if it.primitive == p]
        if len(idx) < MIN_QUESTIONS:
            out[p] = {
                "temperature": 1.0,
                "calibrated": False,
                "n": len(idx),
                "reason": f"n < {MIN_QUESTIONS}",
            }
            continue
        fit = fit_temperature(p, [logits[i] for i in idx], [items[i].target for i in idx])
        out[p] = {**fit, "calibrated": True}
    return out


def apply_temperatures(items: list[QuestionItem], logits: list[torch.Tensor], temps: dict[str, float]):
    for t in temps.values():
        if not math.isfinite(t) or t <= 0:
            raise ValueError("Temperaturas finitas y positivas")
    return [z / temps[it.primitive] for it, z in zip(items, logits, strict=True)]


def load_calibration(path: Path, checkpoint: Path) -> dict[str, Any]:
    """Lee un artefacto de calibración y exige que corresponda exactamente a ``checkpoint``."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    if cal.get("format") != CALIBRATION_FORMAT or cal.get("method") != METHOD:
        raise ValueError(f"Artefacto de calibración no soportado: {cal.get('format')}/{cal.get('method')}")
    ident = checkpoint_identity(checkpoint)
    bound = cal["checkpoint"]
    for key in ("manifest_sha256", "weights_sha256", "adapter_sha256"):
        if bound.get(key) != ident.get(key):
            raise ValueError(f"La calibración no pertenece a este checkpoint ({key} distinto)")
    if cal.get("split") not in CALIBRATION_SPLITS:
        raise ValueError("Las temperaturas deben ajustarse en la partición de calibración")
    if cal["split"] == EXTERNAL_SPLIT and not cal.get("external_dataset", {}).get("sha256"):
        raise ValueError("Calibración externa sin sha256 del conjunto usado")
    return cal


EXTERNAL_SPLIT = "external_calibration"
CALIBRATION_SPLITS = ("calibration", EXTERNAL_SPLIT)


def run_calibrate(
    checkpoint: Path, split: str, *, use_cache: bool = True, dataset: Path | None = None
) -> dict[str, Any]:
    from .data.dataset import load_dataset
    from .training.decisions_pipeline import (
        _external_leakage,
        _predictions,
        evaluate_logits,
        prepare_evaluation,
    )

    if dataset is None and split != "calibration":
        raise ValueError("gso calibrate sólo ajusta sobre la partición calibration (spec §5.3, §6.2)")
    if dataset is not None and split != "all":
        raise ValueError("Con --dataset de calibración externo se usa completo: --split all")
    checkpoint = Path(checkpoint)
    ctx = prepare_evaluation(checkpoint, use_cache=use_cache)
    external = None
    if dataset is not None:
        ext_ds = load_dataset(dataset)
        leakage = _external_leakage(ctx.train_ds, ext_ds.examples)
        if leakage["errors"]:
            raise ValueError(f"El conjunto de calibración comparte grupos o entradas: {leakage['errors']}")
        external = {"root": str(dataset), "sha256": ext_ds.sha256, "leakage": leakage}
    items = ctx.items_for("calibration", dataset=dataset)
    logits, ext = ctx.logits("calibration" if dataset is None else "all", items)
    fits = fit_temperatures(items, logits, ctx.extra["primitives"])
    temps = {p: fits[p]["temperature"] for p in fits}
    thr = ctx.extra["threshold"]
    scaled = apply_temperatures(items, logits, temps)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "format": CALIBRATION_FORMAT,
        "method": METHOD,
        "created_utc": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint_identity(checkpoint),
        "code": git_state(snapshot=True),
        "dataset_sha256": ctx.extra["dataset_sha256"],
        "split_manifest_sha256": ctx.extra["split_manifest_sha256"],
        "split": "calibration" if external is None else EXTERNAL_SPLIT,
        "external_dataset": external,
        "questions": len(items),
        "temperature_param": f"T = softplus(t) + {TEMPERATURE_EPS}, t0 tal que T0 = 1",
        "min_questions": MIN_QUESTIONS,
        "fits": fits,
        "temperatures": temps,
        "calibrated": {p: bool(fits[p]["calibrated"]) for p in fits},
        "in_sample_metrics": {
            "note": "misma partición que el ajuste: no es una evaluación independiente",
            "before": evaluate_logits(items, logits, thr),
            "after": evaluate_logits(items, scaled, thr),
        },
        "extraction": ext,
    }
    out_dir = checkpoint.parent / "calibration"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"calibration-{stamp}.json"
    if path.exists():
        raise FileExistsError(f"{path} ya existe; los artefactos de calibración no se sobrescriben")
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    preds = out_dir / f"calibration-{stamp}-predictions.jsonl"
    with preds.open("w", encoding="utf-8") as fh:
        for r in _predictions(items, logits, thr, temps):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": str(path), **artifact}
