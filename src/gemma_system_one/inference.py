"""Reconstrucción de respuestas tipadas a partir de logits por fila (spec §4.3–§4.5, §6.3, §9).

Se usa igual en evaluación y, más adelante, en el servidor. No genera texto: la
respuesta se construye en código a partir de tensores.

- Noul: ``noul = σ(z/T)``.
- Choice: softmax sobre las K filas; las probabilidades se devuelven por ID opaco.
- Score: softmax sobre las M filas; ``score = Σ m·p_m`` y ``legend`` = rúbrica en su orden.
- ``confidence`` = concentración ``1 − H(p)/ln K`` (definición propia, ``normalized_entropy_v1``);
  no es probabilidad de acierto.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .contracts import ChoiceQuestion, NoulQuestion, ScoreQuestion
from .serialization import Row

CONFIDENCE_METHOD = "normalized_entropy_v1"
PROB_SUM_ATOL = 1e-5  # tolerancia FP32 de la suma de probabilidades


class InvalidOutputError(RuntimeError):
    pass


def target_index(question, rows: list[Row], target) -> int:
    """Índice de la fila correcta dentro del grupo (remapeo por ID tras el orden canónico)."""
    if isinstance(question, NoulQuestion):
        return int(target.label)
    if isinstance(question, ChoiceQuestion):
        matches = [i for i, r in enumerate(rows) if r.candidate_id == target.class_id]
        if len(matches) != 1:
            raise ValueError(f"class_id {target.class_id!r} no identifica una única fila")
        return matches[0]
    if isinstance(question, ScoreQuestion):
        if not 0 <= target.level_index < len(rows):
            raise ValueError("level_index fuera de rango")
        return int(target.level_index)
    raise TypeError(type(question).__name__)


def concentration(p: torch.Tensor) -> float:
    k = p.numel()
    p64 = p.double()
    h = -float(torch.where(p64 > 0, p64 * torch.log(p64), torch.zeros_like(p64)).sum())
    return float(min(1.0, max(0.0, 1.0 - h / math.log(k))))


def group_probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("La temperatura debe ser finita y positiva")
    if not bool(torch.isfinite(logits).all()):
        raise InvalidOutputError("Logits no finitos")
    p = torch.softmax(logits.double() / temperature, dim=0)
    if not bool(torch.isfinite(p).all()):
        raise InvalidOutputError("Probabilidades no finitas")
    return p


def reconstruct(question, rows: list[Row], logits: torch.Tensor, temperature: float = 1.0) -> dict[str, Any]:
    """Respuesta pública de una pregunta a partir de los logits de sus filas (en orden de ``rows``)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("La temperatura debe ser finita y positiva")
    logits = logits.detach().flatten().cpu()
    if logits.numel() != len(rows):
        raise InvalidOutputError(f"{logits.numel()} logits para {len(rows)} filas")
    if isinstance(question, NoulQuestion):
        if not math.isfinite(float(logits[0])):
            raise InvalidOutputError("Logit no finito")
        p = float(torch.sigmoid(logits[0].double() / temperature))
        return {"type": "noul", "noul": p}
    p = group_probabilities(logits, temperature)
    if abs(float(p.sum()) - 1.0) > PROB_SUM_ATOL:
        raise InvalidOutputError("Probabilidades sin normalizar")
    conf = concentration(p)
    if isinstance(question, ChoiceQuestion):
        probs = {r.candidate_id: float(pi) for r, pi in zip(rows, p, strict=True)}
        choice = rows[int(torch.argmax(p))].candidate_id
        if choice not in question.criteria:
            raise InvalidOutputError("Opción elegida fuera del conjunto")
        return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": conf}
    if isinstance(question, ScoreQuestion):
        levels = torch.arange(len(rows), dtype=torch.float64)
        score = float((levels * p).sum())
        if not 0.0 <= score <= len(rows) - 1 + 1e-9:
            raise InvalidOutputError("score fuera de rango")
        return {
            "type": "score",
            "score": score,
            "probabilities": [float(x) for x in p],
            "legend": list(question.criteria),
            "confidence": conf,
        }
    raise TypeError(type(question).__name__)
