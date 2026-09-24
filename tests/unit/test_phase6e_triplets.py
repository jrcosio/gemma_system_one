"""Revisión de la fase 6d: tríos con la misma pregunta y estados con respuestas distintas."""

from __future__ import annotations

from collections import Counter, defaultdict

from gemma_system_one.data.derive import DIAG_DISTRACTOR_OPTIONS, REAL_KINDS, fault_kind_triplets
from gemma_system_one.data.generate_mixed import (
    FAULT_DISTRACTOR_OPTIONS,
    FAULT_KIND_OPTIONS,
    generate_mixed,
    true_fault_category,
)


def _t2c(lang):
    cats = {**FAULT_KIND_OPTIONS[lang], **FAULT_DISTRACTOR_OPTIONS[lang], **DIAG_DISTRACTOR_OPTIONS[lang]}
    return {t: c for c, ts in cats.items() for t in ts}


def test_triplets_share_question_differ_in_state_and_labels_follow_facts():
    ex, audit = generate_mixed(1500, seed=41, version="v4")
    facts = {a["group_id"]: a["facts"] for a in audit}
    k4, k8 = fault_kind_triplets(ex, audit, seed=0, n_triplets=120)
    assert len(k4) == len(k8) == 360
    trip = defaultdict(list)
    for a, b in zip(k4, k8, strict=True):
        assert (
            a.id == b.id
            and a.target == b.target
            and a.question.criteria.items() <= b.question.criteria.items()
        )
        assert len(a.question.criteria) == 4 and len(b.question.criteria) == 8
        trip[a.id.rsplit("-", 1)[0]].append(a)
    used_groups = Counter(e.group_id for e in k4)
    assert max(used_groups.values()) == 1  # cada estado se usa una sola vez
    roles = Counter()
    for members in trip.values():
        assert len(members) == 3
        q = members[0].question
        assert all(m.question == q for m in members)  # misma pregunta exacta (textos, IDs, orden)
        assert len({m.language for m in members}) == 1 and len({m.state.__repr__() for m in members}) == 3
        t2c = _t2c(members[0].language)
        present = {t2c[t] for t in q.criteria.values()}
        assert {"none", "other"} <= present and len(present & set(REAL_KINDS)) == 2
        answers = set()
        for m in members:
            label = t2c[q.criteria[m.target.class_id]]
            true = true_fault_category(facts[m.group_id])
            expected = true if true in present else "other"
            assert label == expected  # la etiqueta es la respuesta semántica de su propio estado
            answers.add(label)
            roles[m.provenance.label_method.split(":")[1]] += 1
        assert len(answers) == 3  # sin leer el estado, a lo sumo un acierto por trío
    assert roles == {"real": 120, "other": 120, "none": 120}
    assert fault_kind_triplets(ex, audit, seed=0, n_triplets=120)[0] == k4
