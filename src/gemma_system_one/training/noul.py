"""Etapa A (spec §7.A): cabezal Noul sobre representaciones congeladas y cacheadas.

La pérdida se promedia por pregunta lógica. Un paso de optimizador acumula
``questions_per_step`` preguntas repartidas en microlotes de ``microbatch_rows`` filas;
cada microlote contribuye ``sum(loss)/n_preguntas_del_paso``, de modo que el
gradiente no depende del tamaño de microlote y el último paso parcial se normaliza
por su número real de preguntas.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from ..metrics import binary_nll_from_logits
from ..models.heads import NoulHead


class NoulTrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=3, ge=1, le=10_000)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    questions_per_step: int = Field(default=8, ge=1)
    microbatch_rows: int = Field(default=8, ge=1)
    seed: int = 0
    # Umbral de la decisión discreta: fijado antes de ver resultados (spec §4.3).
    threshold: float = Field(default=0.5, gt=0, lt=1)
    # "validation": elige la época por NLL de validación. "train": sólo para la prueba
    # de sobreajuste, donde se elige la última época y no se mide generalización.
    select_on: Literal["validation", "train"] = "validation"
    max_train_examples: int | None = Field(default=None, ge=1)


def accumulate_step(head: NoulHead, reps: torch.Tensor, labels: torch.Tensor, microbatch_rows: int) -> float:
    """Backward de un paso de optimizador; devuelve la pérdida media por pregunta."""
    n = len(labels)
    if n == 0:
        raise ValueError("Paso sin preguntas")
    total = 0.0
    for start in range(0, n, microbatch_rows):
        z = head(reps[start : start + microbatch_rows])
        y = labels[start : start + microbatch_rows]
        loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum")
        (loss_sum / n).backward()
        total += loss_sum.item()
    return total / n


@dataclass
class TrainResult:
    head: NoulHead
    selected_epoch: int
    history: list[dict[str, Any]] = field(default_factory=list)


def predict_logits(head: NoulHead, reps: torch.Tensor) -> torch.Tensor:
    head.eval()
    with torch.inference_mode():
        return head(reps).float()


def train_noul_head(
    train_reps: torch.Tensor,
    train_labels: torch.Tensor,
    params: NoulTrainParams,
    val_reps: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
) -> TrainResult:
    """Entrena en CPU/FP32. Devuelve el cabezal de la época seleccionada y el historial."""
    if params.select_on == "validation" and (val_reps is None or val_labels is None):
        raise ValueError("select_on=validation requiere datos de validación")
    torch.manual_seed(params.seed)
    gen = torch.Generator().manual_seed(params.seed)
    head = NoulHead(train_reps.shape[1])
    trainable = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=params.lr, weight_decay=params.weight_decay)
    y_train = train_labels.float()
    history: list[dict[str, Any]] = []
    best_state, best_epoch, best_score = None, 0, float("inf")
    n = len(y_train)
    for epoch in range(1, params.epochs + 1):
        head.train()
        perm = torch.randperm(n, generator=gen)
        step_losses = []
        for start in range(0, n, params.questions_per_step):
            idx = perm[start : start + params.questions_per_step]
            opt.zero_grad(set_to_none=True)
            loss = accumulate_step(head, train_reps[idx], y_train[idx], params.microbatch_rows)
            grads = [p.grad for p in trainable]
            if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError(f"Gradiente ausente o no finito en la época {epoch}")
            opt.step()
            step_losses.append(loss)
        record: dict[str, Any] = {
            "epoch": epoch,
            "steps": len(step_losses),
            "train_loss_mean_steps": sum(step_losses) / len(step_losses),
            "train_nll": binary_nll_from_logits(
                predict_logits(head, train_reps).numpy(), train_labels.numpy()
            ),
        }
        if val_reps is not None and val_labels is not None:
            record["validation_nll"] = binary_nll_from_logits(
                predict_logits(head, val_reps).numpy(), val_labels.numpy()
            )
        history.append(record)
        score = record["validation_nll"] if params.select_on == "validation" else -epoch
        if score < best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    head.eval()
    return TrainResult(head=head, selected_epoch=best_epoch, history=history)
