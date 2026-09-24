"""Fase 6d: ¿cuánto revela el conjunto de opciones sobre la respuesta, sin leer el estado?

Para una muestra grande de preguntas ``fault_type`` se calcula, sin modelos:
  - ``prior``: se elige, entre las opciones presentes, la categoría con mayor frecuencia global de
    ser la respuesta (prior global restringido a las opciones presentes);
  - ``bayes``: la respuesta más frecuente para cada composición exacta (conjunto de categorías), la
    mejor accuracy posible mirando sólo las opciones en esta misma muestra (cota sobreajustada).
La diferencia ``bayes − prior`` mide mejora de accuracy top-1, no información mutua: puede ser cero
aunque la composición cambie las probabilidades de las etiquetas.

Uso: uv run python scripts/probe_option_cue.py OUT.json [casos]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from gemma_system_one.data.derive import (
    DIAG_DISTRACTOR_OPTIONS,
    balanced_fault_kind_pairs,
    widen_fault_kind,
    widen_fault_kind_by_facts,
)
from gemma_system_one.data.generate_mixed import FAULT_DISTRACTOR_OPTIONS, FAULT_KIND_OPTIONS, generate_mixed


def _t2c(lang):
    cats = {**FAULT_KIND_OPTIONS[lang], **FAULT_DISTRACTOR_OPTIONS[lang], **DIAG_DISTRACTOR_OPTIONS[lang]}
    return {t: c for c, ts in cats.items() for t in ts}


def probe(examples) -> dict:
    rows = []
    for e in examples:
        if e.task_family != "fault_type":
            continue
        t2c = _t2c(e.language)
        rows.append(
            (
                frozenset(t2c[t] for t in e.question.criteria.values()),
                t2c[e.question.criteria[e.target.class_id]],
            )
        )
    freq = Counter(label for _, label in rows)
    by_sig: dict[frozenset, Counter] = defaultdict(Counter)
    for sig, label in rows:
        by_sig[sig][label] += 1
    prior_ok = sum(max(sig, key=lambda c: (freq[c], c)) == label for sig, label in rows)
    bayes_ok = sum(c.most_common(1)[0][1] for c in by_sig.values())
    n = len(rows)
    other = [lab == "other" for _, lab in rows]
    return {
        "questions": n,
        "signatures": len(by_sig),
        "other_rate": round(sum(other) / n, 4),
        "acc_prior_only": round(prior_ok / n, 4),
        "acc_options_bayes": round(bayes_ok / n, 4),
        "composition_gain": round((bayes_ok - prior_ok) / n, 4),
    }


def main(out: str, cases: int = 20000) -> None:
    ex3, _ = generate_mixed(cases, seed=31, version="v3")
    ex4, au4 = generate_mixed(cases, seed=31, version="v4")
    k4, k8 = balanced_fault_kind_pairs(ex4, au4, seed=0)
    report = {
        "cases": cases,
        "note": "Accuracies top-1 sin leer el estado; Bayes se ajusta en la misma muestra. Ver docstring.",
        "v3_generator_K3_6": probe(ex3),
        "v4_generator_K3_6": probe(ex4),
        "v3_widen_by_label_K8_phase6": probe(widen_fault_kind(ex3, seed=0)),
        "v4_widen_by_facts_K8_phase6c": probe(widen_fault_kind_by_facts(ex4, au4, seed=0)),
        "balanced_K4_phase6d": probe(k4),
        "balanced_K8_phase6d": probe(k8),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:]))
