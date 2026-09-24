"""Etapa A de fase 2: cabezales Noul/Choice/Score sobre representaciones congeladas.

Pérdidas (spec §6.1):
- Noul: BCE sobre el logit de su única fila.
- Choice: entropía cruzada de la softmax sobre las K filas del grupo completo.
- Score: entropía cruzada sobre las M filas + ``rps_lambda`` · RPS (λ=0 por defecto;
  λ se elige sólo con validación).

La pérdida es por pregunta lógica, ponderada por ``type_weights`` (declarados) y dividida
por el número real de preguntas del paso, incluido el último paso parcial. Los logits de
un grupo pueden calcularse por trozos (``logit_chunk_rows``), pero la normalización y la
pérdida se hacen siempre sobre el grupo completo concatenado.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts import Example
from ..inference import target_index
from ..models.heads import PRIMITIVES, DecisionHeads
from ..serialization import Row, expand_example


class DecisionTrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=3, ge=1, le=10_000)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    questions_per_step: int = Field(default=8, ge=1)
    logit_chunk_rows: int = Field(default=4, ge=1)
    seed: int = 0
    threshold: float = Field(default=0.5, gt=0, lt=1)
    rps_lambda: float = Field(default=0.0, ge=0)
    type_weights: dict[str, float] = {"noul": 1.0, "choice": 1.0, "score": 1.0}
    select_on: Literal["validation", "train"] = "validation"
    max_train_questions: int | None = Field(default=None, ge=1)
    # Decisión 0004: estandarizar con media/desviación de las representaciones de train.
    feature_norm: Literal["none", "standardize_train"] = "standardize_train"

    @field_validator("type_weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != set(PRIMITIVES) or any(not math.isfinite(w) or w < 0 for w in v.values()):
            raise ValueError(f"type_weights debe definir pesos finitos ≥0 para {PRIMITIVES}")
        return v


@dataclass(frozen=True)
class QuestionItem:
    example: Example
    rows: list[Row]
    target: int
    image_file: Path | None = None  # ruta absoluta de la imagen de sus filas (misma para todas)

    @property
    def primitive(self) -> str:
        return self.example.question.type


ImageMode = Literal["use", "omit"]


def build_items(
    examples: list[Example], image_root: Path | None = None, images: ImageMode = "use"
) -> list[QuestionItem]:
    """Filas y objetivo por pregunta. Con imagen hace falta la raíz del dataset para resolverla.

    ``images="omit"`` quita la imagen de las filas (control sólo texto, decisión 0007): el texto de
    cada fila no cambia, así que el control difiere únicamente en la imagen.
    """
    items = []
    for e in examples:
        if e.image_path is not None and images == "omit":
            e = e.model_copy(update={"image_path": None})
        image_file = None
        if e.image_path is not None:
            if image_root is None:
                raise ValueError(f"{e.id}: tiene imagen y no se indicó la raíz del dataset")
            image_file = Path(image_root) / e.image_path
        rows = expand_example(e)
        items.append(QuestionItem(e, rows, target_index(e.question, rows, e.target), image_file))
    return items


def flatten(items: list[QuestionItem]) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    texts, hashes, offsets = [], [], []
    for it in items:
        offsets.append((len(texts), len(it.rows)))
        texts.extend(r.text for r in it.rows)
        hashes.extend(r.sha256 for r in it.rows)
    return texts, hashes, offsets


def row_images(items: list[QuestionItem]) -> list[Path | None] | None:
    """Imagen de cada fila (alineada con ``flatten``); ``None`` si ningún ítem tiene imagen."""
    if all(it.image_file is None for it in items):
        return None
    return [it.image_file for it in items for _ in it.rows]


def group_logits(heads: DecisionHeads, primitive: str, reps: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    """Logits del grupo completo; los trozos se concatenan antes de cualquier normalización."""
    parts = [heads(primitive, reps[i : i + chunk_rows]) for i in range(0, len(reps), chunk_rows)]
    return torch.cat(parts)


def rps_loss(logits: torch.Tensor, target: int) -> torch.Tensor:
    p = torch.softmax(logits, dim=0)
    m = p.numel()
    cdf = torch.cumsum(p, dim=0)[:-1]
    ind = (target <= torch.arange(m - 1, device=p.device)).to(p.dtype)
    return ((cdf - ind) ** 2).sum() / (m - 1)


def question_loss(primitive: str, logits: torch.Tensor, target: int, rps_lambda: float = 0.0) -> torch.Tensor:
    if primitive == "noul":
        if logits.numel() != 1:
            raise ValueError("Noul espera una fila")
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits[0], torch.tensor(float(target), dtype=logits.dtype, device=logits.device)
        )
    if logits.numel() < 2:
        raise ValueError(f"{primitive} necesita al menos 2 filas")
    ce = -torch.log_softmax(logits, dim=0)[target]
    if primitive == "choice":
        return ce
    if primitive == "score":
        return ce + rps_lambda * rps_loss(logits, target) if rps_lambda else ce
    raise ValueError(primitive)


def question_nll(primitive: str, logits: torch.Tensor, target: int) -> float:
    with torch.no_grad():
        return float(question_loss(primitive, logits.double(), target, 0.0))


def accumulate_step(
    heads: DecisionHeads,
    batch: list[int],
    items: list[QuestionItem],
    reps: torch.Tensor,
    offsets: list[tuple[int, int]],
    params: DecisionTrainParams,
) -> float:
    """Backward de un paso: Σ_q w_tipo·L_q / n_preguntas. Devuelve la pérdida media ponderada."""
    n = len(batch)
    if n == 0:
        raise ValueError("Paso sin preguntas")
    total = 0.0
    for qi in batch:
        it = items[qi]
        start, k = offsets[qi]
        z = group_logits(heads, it.primitive, reps[start : start + k], params.logit_chunk_rows)
        loss = params.type_weights[it.primitive] * question_loss(
            it.primitive, z, it.target, params.rps_lambda
        )
        (loss / n).backward()
        total += loss.detach().item()
    return total / n


def all_logits(heads: DecisionHeads, items, reps, offsets, chunk_rows: int = 64) -> list[torch.Tensor]:
    heads.eval()
    with torch.inference_mode():
        return [
            group_logits(heads, it.primitive, reps[s : s + k], chunk_rows).float()
            for it, (s, k) in zip(items, offsets, strict=True)
        ]


def mean_nll(items, logits: list[torch.Tensor]) -> dict[str, float | None]:
    per_type: dict[str, list[float]] = {p: [] for p in PRIMITIVES}
    for it, z in zip(items, logits, strict=True):
        per_type[it.primitive].append(question_nll(it.primitive, z, it.target))
    allv = [v for vs in per_type.values() for v in vs]
    out: dict[str, float | None] = {"all": sum(allv) / len(allv)}
    for p, vs in per_type.items():
        out[p] = sum(vs) / len(vs) if vs else None
    return out


@dataclass
class DecisionTrainResult:
    heads: DecisionHeads
    selected_epoch: int
    history: list[dict[str, Any]] = field(default_factory=list)


def train_decision_heads(
    items: list[QuestionItem],
    reps: torch.Tensor,
    offsets: list[tuple[int, int]],
    params: DecisionTrainParams,
    val: tuple[list[QuestionItem], torch.Tensor, list[tuple[int, int]]] | None = None,
) -> DecisionTrainResult:
    if params.select_on == "validation" and val is None:
        raise ValueError("select_on=validation requiere datos de validación")
    torch.manual_seed(params.seed)
    gen = torch.Generator().manual_seed(params.seed)
    heads = DecisionHeads(reps.shape[1])
    if params.feature_norm == "standardize_train":
        heads.standardizer.fit(reps)  # sólo filas de entrenamiento
    present = {it.primitive for it in items}
    # Sólo los evaluadores con preguntas de su tipo reciben gradiente; el resto queda en su inicialización.
    trainable = [p for name in sorted(present) for p in heads.heads[name].parameters()]
    opt = torch.optim.AdamW(trainable, lr=params.lr, weight_decay=params.weight_decay)
    history: list[dict[str, Any]] = []
    best_state, best_epoch, best_score = None, 0, math.inf
    for epoch in range(1, params.epochs + 1):
        heads.train()
        perm = torch.randperm(len(items), generator=gen).tolist()
        losses = []
        for start in range(0, len(perm), params.questions_per_step):
            batch = perm[start : start + params.questions_per_step]
            opt.zero_grad(set_to_none=True)
            losses.append(accumulate_step(heads, batch, items, reps, offsets, params))
            for p in trainable:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise RuntimeError(f"Gradiente no finito en la época {epoch}")
            opt.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "steps": len(losses),
            "train_loss_mean_steps": sum(losses) / len(losses),
            "train_nll": mean_nll(items, all_logits(heads, items, reps, offsets)),
        }
        if val is not None:
            v_items, v_reps, v_off = val
            record["validation_nll"] = mean_nll(v_items, all_logits(heads, v_items, v_reps, v_off))
        history.append(record)
        score = record["validation_nll"]["all"] if params.select_on == "validation" else -epoch
        if score < best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(heads.state_dict())
    heads.load_state_dict(best_state)
    heads.eval()
    return DecisionTrainResult(heads=heads, selected_epoch=best_epoch, history=history)
