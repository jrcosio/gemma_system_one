"""Revisión de la fase 6b: en v3, K revelaba «other» en fault_type; v4 lo corrige."""

from __future__ import annotations

from collections import Counter

from gemma_system_one.contracts import CHOICE_MAX
from gemma_system_one.data.derive import widen_fault_kind_by_facts
from gemma_system_one.data.generate_mixed import (
    FAULT_DISTRACTOR_OPTIONS,
    FAULT_KIND_OPTIONS,
    fault_categories,
    generate_mixed,
    true_fault_category,
)


def _is_other(e) -> bool:
    return e.question.criteria[e.target.class_id] in FAULT_KIND_OPTIONS[e.language]["other"]


def _by_k(examples, audit):
    facts = {a["group_id"]: a["facts"] for a in audit}
    c = Counter()
    for e in examples:
        if e.task_family == "fault_type" and facts[e.group_id]["fault"] != "absent":
            c[(len(e.question.criteria), _is_other(e))] += 1
    return c


def test_v3_leaks_label_through_k_and_v4_does_not():
    ex3, au3 = generate_mixed(1500, seed=3, version="v3")
    c3 = _by_k(ex3, au3)
    assert c3[(6, True)] == 0 and c3[(6, False)] > 0  # la pista que encontró la revisión
    ex4, au4 = generate_mixed(4000, seed=3, version="v4")
    c4 = _by_k(ex4, au4)
    for k in range(3, 7):
        n = c4[(k, True)] + c4[(k, False)]
        assert n > 200 and abs(c4[(k, True)] / n - 0.2) < 0.07, (k, c4)


def test_v4_answers_are_consistent_with_facts():
    ex, audit = generate_mixed(600, seed=4, version="v4")
    facts = {a["group_id"]: a["facts"] for a in audit}
    for e in ex:
        if e.task_family != "fault_type":
            continue
        cats = fault_categories(e.language)
        text_to_cat = {t: c for c, ts in cats.items() for t in ts}
        present = {text_to_cat[t] for t in e.question.criteria.values()}
        label = text_to_cat[e.question.criteria[e.target.class_id]]
        true = true_fault_category(facts[e.group_id])
        assert 3 <= len(e.question.criteria) <= 6
        assert label not in FAULT_DISTRACTOR_OPTIONS[e.language]  # una distractora nunca es la respuesta
        if label == "other":
            assert true not in present and true != "none"
        else:
            assert label == true


def test_widening_by_facts_reaches_k8_for_every_label():
    ex, audit = generate_mixed(400, seed=5, version="v4")
    fault = {e.id: e for e in ex if e.task_family == "fault_type"}
    wide = widen_fault_kind_by_facts(ex, audit, seed=0)
    facts = {a["group_id"]: a["facts"] for a in audit}
    assert [w.id for w in wide] == list(fault)
    saw_other = False
    for w in wide:
        o = fault[w.id]
        assert len(w.question.criteria) == CHOICE_MAX  # K ya no depende de la etiqueta
        assert w.target == o.target and o.question.criteria.items() <= w.question.criteria.items()
        cats = fault_categories(w.language)
        text_to_cat = {t: c for c, ts in cats.items() for t in ts}
        added = {text_to_cat[t] for i, t in w.question.criteria.items() if i not in o.question.criteria}
        assert true_fault_category(facts[w.group_id]) not in added
        saw_other |= _is_other(w)
    assert saw_other
    assert [e.model_dump() for e in widen_fault_kind_by_facts(ex, audit, seed=0)] == [
        e.model_dump() for e in wide
    ]
