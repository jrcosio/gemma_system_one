"""Cabezales entrenables. Sin E/S ni HTTP; operan en FP32 sobre representaciones del backbone."""

from __future__ import annotations

import torch
from torch import nn


class NoulHead(nn.Module):
    """Logit binario ``z = w^T h + b``. La probabilidad y la temperatura se aplican fuera."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.shape[-1] != self.hidden_size:
            raise ValueError(f"Dimensión {pooled.shape[-1]} != {self.hidden_size}")
        return self.proj(pooled.float()).squeeze(-1)


PRIMITIVES = ("noul", "choice", "score")


class FeatureStandardizer(nn.Module):
    """Estandarización fija por dimensión, ``(h - media) / desviación``, ajustada sólo con train.

    Las representaciones de Gemma tienen norma ~230 y dimensiones atípicas con magnitud
    media ~100 (decisión 0004); sin escalar, AdamW con lr 1e-3 da pasos que hacen oscilar
    la pérdida de entrenamiento. Son buffers (no se entrenan) y viajan en el checkpoint.
    Sin ajustar es la identidad.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(hidden_size))
        self.register_buffer("scale", torch.ones(hidden_size))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, reps: torch.Tensor, eps: float = 1e-6) -> None:
        r = reps.double()
        self.mean.copy_(r.mean(0).float())
        self.scale.copy_(r.std(0, unbiased=False).clamp_min(eps).float())
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.scale


class DecisionHeads(nn.Module):
    """Tres evaluadores compartidos e independientes, uno por primitiva (spec §4.3–§4.5).

    Cada uno produce un logit por fila. Choice y Score puntúan cada candidato/nivel con el
    mismo evaluador y normalizan después sobre el grupo completo de la pregunta; ninguna
    posición de salida está ligada a un ID o índice fijo.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.standardizer = FeatureStandardizer(hidden_size)
        self.heads = nn.ModuleDict({p: NoulHead(hidden_size) for p in PRIMITIVES})

    def forward(self, primitive: str, pooled: torch.Tensor) -> torch.Tensor:
        return self.heads[primitive](self.standardizer(pooled))
