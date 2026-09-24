"""Métricas de evaluación (spec §6.1, §6.3, §8). Cálculo en float64 con numpy.

Convenciones:
- ECE: 15 bins fijos que cubren [0,1]; p=1 cae en el último bin; bins vacíos aportan 0.
- Noul: ECE del *evento* (media de p frente a frecuencia de y=1), no de la clase elegida.
- Brier multiclase: suma sobre clases sin dividir por K.
- Métricas no definidas (p. ej. precisión sin positivos predichos) se devuelven como None.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

N_BINS = 15


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bin_index(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    return np.minimum((np.asarray(p, dtype=np.float64) * n_bins).astype(np.int64), n_bins - 1)


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict[str, Any]]]:
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    idx = bin_index(p, n_bins)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        row: dict[str, Any] = {"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "count": n}
        if n:
            row["mean_p"] = float(p[m].mean())
            row["freq"] = float(y[m].mean())
            ece += n / len(p) * abs(row["mean_p"] - row["freq"])
        table.append(row)
    return float(ece), table


def binary_nll_from_logits(z: np.ndarray, y: np.ndarray) -> float:
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
    # -log σ(z) = softplus(-z); -log(1-σ(z)) = softplus(z)
    return float(np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)))


def _safe_div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def binary_metrics(logits: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> dict[str, Any]:
    z = np.asarray(logits, np.float64)
    y = np.asarray(labels, np.int64)
    if z.shape != y.shape or z.ndim != 1 or len(z) == 0:
        raise ValueError("logits y labels deben ser vectores 1D no vacíos de igual tamaño")
    if not np.isfinite(z).all():
        raise ValueError("logits no finitos")
    p = sigmoid(z)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    f1 = None if precision is None or recall is None else _safe_div(2 * tp, 2 * tp + fp + fn)
    ece, table = reliability(p, y)
    single_class = len(set(y.tolist())) < 2
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "nll": binary_nll_from_logits(z, y),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_event_15bins": ece,
        "accuracy": float((pred == y).mean()),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "single_class": single_class,
        "reliability": table,
    }


def multiclass_brier(probs: np.ndarray, labels: Sequence[int]) -> float:
    probs = np.asarray(probs, np.float64)
    onehot = np.eye(probs.shape[1])[np.asarray(labels)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ranked_probability_score(probs: np.ndarray, labels: Sequence[int]) -> float:
    probs = np.asarray(probs, np.float64)
    m = probs.shape[1]
    if m < 2:
        raise ValueError("RPS requiere M ≥ 2")
    cdf = np.cumsum(probs, axis=1)[:, :-1]
    y = np.asarray(labels)[:, None]
    ind = (y <= np.arange(m - 1)[None, :]).astype(np.float64)
    return float(np.mean(np.sum((cdf - ind) ** 2, axis=1) / (m - 1)))


def top_label_ece(probs: np.ndarray, labels: Sequence[int], n_bins: int = N_BINS) -> float:
    probs = np.asarray(probs, np.float64)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(labels)).astype(np.float64)
    return reliability(conf, correct, n_bins)[0]


def concentration(probs: np.ndarray) -> np.ndarray:
    """1 - H(p)/ln K con 0·log 0 = 0, recortado a [0,1] (definición propia, spec §6.3)."""
    probs = np.asarray(probs, np.float64)
    k = probs.shape[-1]
    if k < 2:
        raise ValueError("concentración requiere K ≥ 2")
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=-1)
    return np.clip(1 - h / np.log(k), 0.0, 1.0)


def bootstrap_by_group(
    groups: Sequence[str],
    fn: Callable[[np.ndarray], dict[str, float | None]],
    *,
    reps: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Intervalo percentil remuestreando grupos (no ejemplos dependientes).

    ``fn`` recibe los índices de ejemplos seleccionados y devuelve métricas escalares.
    """
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([by_group[keys[c]] for c in chosen])
        for name, value in fn(idx).items():
            if value is not None:
                samples[name].append(value)
    alpha = (1 - level) / 2
    return {
        "method": "bootstrap percentil por grupo",
        "reps": reps,
        "seed": seed,
        "groups": len(keys),
        "level": level,
        "intervals": {
            k: [float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha))] for k, v in samples.items()
        },
        "defined_reps": {k: len(v) for k, v in samples.items()},
    }


def within_group_pairwise(
    groups: Sequence[str], logits: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    """Pares (positivo, negativo) del mismo estado: ¿el modelo puntúa más al positivo?

    Diagnóstico de dependencia de la instrucción: un modelo que ignora la pregunta
    daría el mismo valor a todas las preguntas de un estado (empates = 0,5).
    """
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    z, y = np.asarray(logits, np.float64), np.asarray(labels)
    wins, pairs = 0.0, 0
    for idx in by_group.values():
        pos = [i for i in idx if y[i] == 1]
        neg = [i for i in idx if y[i] == 0]
        for a in pos:
            for b in neg:
                pairs += 1
                wins += 1.0 if z[a] > z[b] else 0.5 if z[a] == z[b] else 0.0
    return {"pairs": pairs, "pairwise_accuracy": None if pairs == 0 else wins / pairs}


def categorical_metrics(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> dict[str, Any]:
    """Choice con K variable por pregunta: NLL, Brier (suma sin dividir por K), accuracy,
    ECE top-label (15 bins) y concentración media; desglose por K (spec §6.1, §8)."""
    if len(probs) != len(targets) or not probs:
        raise ValueError("probs y targets deben tener igual tamaño no nulo")
    rows = []
    for p, t in zip(probs, targets, strict=True):
        p = np.asarray(p, np.float64)
        if not np.isfinite(p).all() or abs(p.sum() - 1) > 1e-6:
            raise ValueError("Distribución no finita o no normalizada")
        onehot = np.zeros_like(p)
        onehot[t] = 1.0
        rows.append(
            {
                "k": len(p),
                "nll": float(-np.log(max(p[t], 1e-300))),
                "brier": float(np.sum((p - onehot) ** 2)),
                "correct": float(np.argmax(p) == t),
                "conf": float(p.max()),
                "concentration": float(concentration(p[None])[0]),
            }
        )

    def agg(items):
        conf = np.array([r["conf"] for r in items])
        correct = np.array([r["correct"] for r in items])
        return {
            "n": len(items),
            "nll": float(np.mean([r["nll"] for r in items])),
            "brier_sum": float(np.mean([r["brier"] for r in items])),
            "accuracy": float(correct.mean()),
            "ece_top_label_15bins": reliability(conf, correct)[0],
            "mean_concentration": float(np.mean([r["concentration"] for r in items])),
            "uniform_nll": float(np.mean([np.log(r["k"]) for r in items])),
        }

    by_k = {str(k): agg([r for r in rows if r["k"] == k]) for k in sorted({r["k"] for r in rows})}
    return {**agg(rows), "by_k": by_k}


def ordinal_metrics(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> dict[str, Any]:
    """Score con M variable: NLL, RPS, MAE de la esperanza (bruto y /(M-1)), accuracy del
    argmax y ECE de eventos acumulados ``y ≤ j`` con probabilidad ``F_j`` (spec §6.1, §8)."""
    if len(probs) != len(targets) or not probs:
        raise ValueError("probs y targets deben tener igual tamaño no nulo")
    rows, cum_p, cum_y = [], [], []
    for p, t in zip(probs, targets, strict=True):
        p = np.asarray(p, np.float64)
        m = len(p)
        if not np.isfinite(p).all() or abs(p.sum() - 1) > 1e-6:
            raise ValueError("Distribución no finita o no normalizada")
        s = float(np.dot(np.arange(m), p))
        cdf = np.cumsum(p)[:-1]
        ind = (t <= np.arange(m - 1)).astype(np.float64)
        cum_p.extend(np.clip(cdf, 0, 1).tolist())
        cum_y.extend(ind.tolist())
        rows.append(
            {
                "m": m,
                "nll": float(-np.log(max(p[t], 1e-300))),
                "rps": float(np.sum((cdf - ind) ** 2) / (m - 1)),
                "mae": abs(s - t),
                "mae_norm": abs(s - t) / (m - 1),
                "correct": float(np.argmax(p) == t),
                "concentration": float(concentration(p[None])[0]),
            }
        )

    def agg(items):
        return {
            "n": len(items),
            **{k: float(np.mean([r[k] for r in items])) for k in ("nll", "rps", "mae", "mae_norm")},
            "accuracy": float(np.mean([r["correct"] for r in items])),
            "mean_concentration": float(np.mean([r["concentration"] for r in items])),
            "uniform_nll": float(np.mean([np.log(r["m"]) for r in items])),
        }

    by_m = {str(m): agg([r for r in rows if r["m"] == m]) for m in sorted({r["m"] for r in rows})}
    return {
        **agg(rows),
        "cumulative_event_ece_15bins": reliability(np.array(cum_p), np.array(cum_y))[0],
        "cumulative_events": len(cum_p),
        "by_m": by_m,
    }


def compare_predictions(
    a: Sequence[dict[str, Any]],
    b: Sequence[dict[str, Any]],
    *,
    reps: int = 1000,
    seed: int = 0,
    allow_different_inputs: bool = False,
) -> dict[str, Any]:
    """Diferencia emparejada ``b − a`` de NLL y acierto por pregunta (mismas preguntas), con IC
    por bootstrap de grupos. Espera filas de predicción con ``id``, ``group_id``, ``type``,
    ``nll`` y ``correct`` (las de ``gso evaluate``; con calibración, ya calibradas)."""
    by_a = {r["id"]: r for r in a}
    by_b = {r["id"]: r for r in b}
    if set(by_a) != set(by_b) or len(by_a) != len(a) or len(by_b) != len(b):
        raise ValueError("Las predicciones no cubren exactamente las mismas preguntas")
    ids = sorted(by_a)
    for i in ids:
        if (by_a[i]["group_id"], by_a[i]["type"]) != (by_b[i]["group_id"], by_b[i]["type"]):
            raise ValueError(f"Pregunta {i} con grupo o tipo distinto")
        keys = ("target_index", "target_description")
        if allow_different_inputs and len(by_a[i].get("row_logits", ())) != len(
            by_b[i].get("row_logits", ())
        ):
            # Conjunto de candidatos ampliado a propósito (fase 6, más opciones): el índice de la
            # respuesta cambia con el orden canónico; la etiqueta semántica debe ser la misma.
            if "target_description" not in by_a[i]:
                raise ValueError(f"Pregunta {i}: con otro número de candidatos hace falta target_description")
            keys = ("target_description",)
        for key in keys:
            if (key in by_a[i]) != (key in by_b[i]) or by_a[i].get(key) != by_b[i].get(key):
                raise ValueError(f"Pregunta {i} con etiqueta distinta ({key})")
        # Huella de lo que vio el modelo (filas serializadas + imagen); ausente en predicciones históricas.
        # Un control deliberado (p. ej. las mismas filas sin imagen) cambia la entrada a propósito:
        # sólo se admite con allow_different_inputs y siguen exigiéndose preguntas y etiquetas iguales.
        ha, hb = by_a[i].get("input_sha256"), by_b[i].get("input_sha256")
        if (ha is None) != (hb is None) or (ha != hb and not allow_different_inputs):
            raise ValueError(f"Pregunta {i} con entrada distinta (input_sha256)")
    types = np.array([by_a[i]["type"] for i in ids])
    d_nll = np.array([by_b[i]["nll"] - by_a[i]["nll"] for i in ids], np.float64)
    d_acc = np.array([float(by_b[i]["correct"]) - float(by_a[i]["correct"]) for i in ids], np.float64)
    nll_a = np.array([by_a[i]["nll"] for i in ids], np.float64)
    nll_b = np.array([by_b[i]["nll"] for i in ids], np.float64)

    def fn(idx):
        out: dict[str, float | None] = {
            "a.nll_all": float(nll_a[idx].mean()),
            "b.nll_all": float(nll_b[idx].mean()),
            "b_minus_a.nll_all": float(d_nll[idx].mean()),
            "b_minus_a.accuracy_all": float(d_acc[idx].mean()),
        }
        for t in sorted(set(types.tolist())):
            sel = idx[types[idx] == t]
            out[f"b_minus_a.{t}.nll"] = float(d_nll[sel].mean()) if len(sel) else None
            out[f"b_minus_a.{t}.accuracy"] = float(d_acc[sel].mean()) if len(sel) else None
        return out

    point = fn(np.arange(len(ids)))
    boot = bootstrap_by_group([by_a[i]["group_id"] for i in ids], fn, reps=reps, seed=seed)
    same_inputs = all(by_a[i].get("input_sha256") == by_b[i].get("input_sha256") for i in ids)
    return {"questions": len(ids), "same_inputs": same_inputs, "point": point, "bootstrap": boot}
