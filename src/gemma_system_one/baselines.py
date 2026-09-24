"""Baselines Noul (spec §8): prior de clase y clasificador textual ligero.

Ambos se ajustan sólo con entrenamiento y con hiperparámetros fijados de antemano
(no se ajustan en validación ni en test). El clasificador ve exactamente el mismo
texto serializado que el backbone. Limitación: es un modelo de etiquetas fijas por
bolsa de palabras; no representa interacciones pregunta×estado más allá de bigramas.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

import torch

BOW_DIM = 2**14
BOW_L2 = 1e-2  # fijo a priori
BOW_ITERS = 200
_TOKEN = re.compile(r"\w+", re.UNICODE)


class PriorBaseline:
    """Probabilidad constante = prevalencia de entrenamiento con suavizado de Laplace (k+1)/(n+2)."""

    def fit(self, labels: Sequence[int]) -> PriorBaseline:
        n, k = len(labels), int(sum(labels))
        self.p = (k + 1) / (n + 2)
        return self

    def logits(self, n: int) -> list[float]:
        z = math.log(self.p / (1 - self.p))
        return [z] * n


def _features(text: str) -> list[int]:
    tokens = _TOKEN.findall(text.casefold())
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    return sorted({int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % BOW_DIM for g in grams})


def _matrix(texts: Sequence[str]) -> torch.Tensor:
    x = torch.zeros(len(texts), BOW_DIM, dtype=torch.float64)
    for i, t in enumerate(texts):
        x[i, _features(t)] = 1.0
    return x


class HashedBowLogistic:
    """Regresión logística L2 sobre unigramas+bigramas hasheados (binarios), LBFGS en CPU."""

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> HashedBowLogistic:
        x = _matrix(texts)
        y = torch.tensor(labels, dtype=torch.float64)
        self.w = torch.zeros(BOW_DIM, dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=BOW_ITERS, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = x @ self.w + self.b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + BOW_L2 * (self.w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def logits(self, texts: Sequence[str]) -> list[float]:
        with torch.no_grad():
            return (_matrix(texts) @ self.w + self.b).tolist()


class UniformChoiceBaseline:
    """Choice: 1/K por pregunta. Con opciones dinámicas no hay un prior por ID con sentido."""

    def logits(self, sizes: Sequence[int]) -> list[list[float]]:
        return [[0.0] * k for k in sizes]


class LevelPriorBaseline:
    """Score: distribución marginal del nivel en train por cardinalidad M, con Laplace.

    Para una M no vista en train devuelve la uniforme.
    """

    def fit(self, sizes: Sequence[int], targets: Sequence[int]) -> LevelPriorBaseline:
        self.counts: dict[int, list[int]] = {}
        for m, t in zip(sizes, targets, strict=True):
            self.counts.setdefault(m, [0] * m)[t] += 1
        return self

    def logits(self, sizes: Sequence[int]) -> list[list[float]]:
        out = []
        for m in sizes:
            c = self.counts.get(m, [0] * m)
            out.append([math.log(ci + 1) for ci in c])
        return out


class HashedBowGroupScorer:
    """Puntuación lineal por fila sobre bolsa de palabras hasheada, softmax dentro del grupo.

    Mismo texto serializado que ve el backbone; hiperparámetros fijos (L2, iteraciones).
    Aplicable a opciones dinámicas porque puntúa cada fila, no posiciones fijas.
    """

    def _padded(self, groups: Sequence[Sequence[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        kmax = max(len(g) for g in groups)
        x = torch.zeros(len(groups), kmax, BOW_DIM, dtype=torch.float64)
        mask = torch.zeros(len(groups), kmax, dtype=torch.bool)
        for i, g in enumerate(groups):
            for j, t in enumerate(g):
                x[i, j, _features(t)] = 1.0
                mask[i, j] = True
        return x, mask

    def fit(self, groups: Sequence[Sequence[str]], targets: Sequence[int]) -> HashedBowGroupScorer:
        x, mask = self._padded(groups)
        y = torch.tensor(targets)
        self.w = torch.zeros(BOW_DIM, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([self.w], max_iter=BOW_ITERS, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = (x @ self.w).masked_fill(~mask, float("-inf"))
            loss = torch.nn.functional.cross_entropy(z, y) + BOW_L2 * (self.w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def logits(self, groups: Sequence[Sequence[str]]) -> list[list[float]]:
        with torch.no_grad():
            return [(_matrix(g) @ self.w).tolist() for g in groups]
