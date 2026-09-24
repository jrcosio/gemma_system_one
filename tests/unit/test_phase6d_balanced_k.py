"""Revisión de la fase 6c: la composición del diagnóstico K8 no debe revelar la etiqueta."""

from __future__ import annotations

from collections import Counter

from gemma_system_one.data.derive import (
    DIAG_DISTRACTOR_OPTIONS,
    REAL_KINDS,
    balanced_fault_kind_pairs,
    swap_states,
)
from gemma_system_one.data.generate_mixed import (
    FAULT_DISTRACTOR_OPTIONS,
    FAULT_KIND_OPTIONS,
    generate_mixed,
    true_fault_category,
)


def _cats(lang):
    return {**FAULT_KIND_OPTIONS[lang], **FAULT_DISTRACTOR_OPTIONS[lang], **DIAG_DISTRACTOR_OPTIONS[lang]}


def _signature(e):
    t2c = {t: c for c, ts in _cats(e.language).items() for t in ts}
    return frozenset(t2c[t] for t in e.question.criteria.values()), t2c[
        e.question.criteria[e.target.class_id]
    ]


def test_structure_is_label_independent_and_labels_follow_facts():
    ex, audit = generate_mixed(3000, seed=21, version="v4")
    facts = {a["group_id"]: a["facts"] for a in audit}
    k4, k8 = balanced_fault_kind_pairs(ex, audit, seed=0)
    assert [e.id for e in k4] == [e.id for e in k8] and len(k4) > 1000
    pairs = {"other": Counter(), "real": Counter()}
    for a, b in zip(k4, k8, strict=True):
        sa, label = _signature(a)
        sb, label_b = _signature(b)
        assert (
            label == label_b
            and a.target == b.target
            and a.question.criteria.items() <= b.question.criteria.items()
        )
        assert len(sa) == 4 and {"none", "other"} <= sa and len(sa & set(REAL_KINDS)) == 2
        assert sb - sa == set(FAULT_DISTRACTOR_OPTIONS["es"]) | set(
            DIAG_DISTRACTOR_OPTIONS["es"]
        )  # siempre las 4
        true = true_fault_category(facts[a.group_id])
        if label == "other":
            assert true not in sa and true != "none"
        else:
            assert label == true and true in sa
        if true != "none":
            pairs["other" if label == "other" else "real"][frozenset(sa & set(REAL_KINDS))] += 1
    # El par de categorías reales es uniforme (6 pares) con respuesta «other» y con respuesta real.
    for c in pairs.values():
        n = sum(c.values())
        assert len(c) == 6 and all(abs(v / n - 1 / 6) < 0.05 for v in c.values()), c
    n_other, n_real = sum(pairs["other"].values()), sum(pairs["real"].values())
    assert abs(n_other / (n_other + n_real) - 0.5) < 0.05


def test_swap_states_is_a_derangement_and_keeps_options_and_labels():
    ex, audit = generate_mixed(200, seed=22, version="v4")
    k4, _ = balanced_fault_kind_pairs(ex, audit, seed=0)
    sw = swap_states(k4, seed=0)
    by_group = {e.group_id: e.state for e in k4}
    for o, s in zip(k4, sw, strict=True):
        assert s.question == o.question and s.target == o.target and s.group_id == o.group_id
        assert s.state != o.state and s.state in by_group.values()
    assert swap_states(k4, seed=0) == sw
