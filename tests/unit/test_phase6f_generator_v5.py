"""Fase 6f: generador v5 (definiciones excluyentes, opciones independientes de los hechos)."""

from __future__ import annotations

import json
from collections import Counter

from gemma_system_one.data.derive import fault_kind_triplets
from gemma_system_one.data.generate_mixed import (
    FAULT_KIND_OPTIONS_V5,
    FAULT_TYPE_CLAUSE_V5,
    fault_categories,
    generate_mixed,
    true_fault_category,
)


def _cats(e):
    return {t: c for c, ts in fault_categories(e.language, "v5").items() for t in ts}


def test_v5_labels_follow_facts_and_options_always_include_none_and_other():
    ex, audit = generate_mixed(800, seed=51, version="v5")
    facts = {a["group_id"]: a["facts"] for a in audit}
    n = 0
    for e in ex:
        if e.task_family != "fault_type":
            continue
        n += 1
        t2c = _cats(e)
        present = {t2c[t] for t in e.question.criteria.values()}
        label = t2c[e.question.criteria[e.target.class_id]]
        true = true_fault_category(facts[e.group_id])
        assert {"none", "other"} <= present and 3 <= len(present) <= 6
        assert label == (true if true in present else "other")
        assert FAULT_TYPE_CLAUSE_V5[e.language] in e.question.instructions
    assert n > 200


def test_v5_option_composition_does_not_depend_on_facts():
    """Frecuencia de cada categoría listada, condicionada a la categoría verdadera: constante."""
    ex, audit = generate_mixed(6000, seed=52, version="v5")
    facts = {a["group_id"]: a["facts"] for a in audit}
    listed: dict[str, Counter] = {}
    total: Counter = Counter()
    for e in ex:
        if e.task_family != "fault_type":
            continue
        true = true_fault_category(facts[e.group_id])
        t2c = _cats(e)
        total[true] += 1
        listed.setdefault(true, Counter()).update(
            {t2c[t] for t in e.question.criteria.values()} - {"none", "other"}
        )
    rates = {
        t: {c: listed[t][c] / total[t] for c in ("network", "performance", "data", "application", "hardware")}
        for t in total
    }
    for c in ("network", "performance", "data", "application", "hardware"):
        vals = [rates[t][c] for t in rates]
        assert max(vals) - min(vals) < 0.08, (c, rates)


def test_v5_application_definition_excludes_slowness_and_data_loss():
    for lang, words in (("es", ("lentitud", "datos")), ("en", ("slowness", "data"))):
        main = FAULT_KIND_OPTIONS_V5[lang]["application"][0].lower()
        assert all(w in main for w in words)


def test_v3_and_v4_are_unchanged_by_v5():
    ex, _ = generate_mixed(30, seed=0, version="v4")
    assert json.dumps(ex[0].model_dump(mode="json"), sort_keys=True).startswith('{"group_id": "mix-s0-00000"')
    assert all("Clasifica por la naturaleza" not in e.question.instructions for e in ex)


def test_v5_triplets_use_v5_definitions():
    ex, audit = generate_mixed(1200, seed=53, version="v5")
    k4, _ = fault_kind_triplets(ex, audit, seed=0, n_triplets=60, version="v5")
    assert len(k4) == 180
    assert all(FAULT_TYPE_CLAUSE_V5[e.language] in e.question.instructions for e in k4)
    app = {t for lang in ("es", "en") for t in FAULT_KIND_OPTIONS_V5[lang]["application"]}
    assert any(t in app for e in k4 for t in e.question.criteria.values())
