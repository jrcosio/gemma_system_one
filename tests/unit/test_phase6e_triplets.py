"""Revisión de la fase 6d: tríos con la misma pregunta y estados con respuestas distintas."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

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


def test_triplet_analysis_rejects_duplicate_role(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "trip0-0000-real", "correct": True},
        {"id": "trip0-0000-other", "correct": True},
        {"id": "trip0-0000-none", "correct": True},
        {"id": "trip0-0000-real", "correct": False},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    script = Path(__file__).resolve().parents[2] / "scripts/analyze_triplets.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "out.json"), f"bad={path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "duplicado" in result.stderr


def test_compare_triplets_pairs_by_triplet(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "compare_triplets", Path(__file__).resolve().parents[2] / "scripts" / "compare_triplets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def write(name, correct):
        p = tmp_path / name
        rows = [
            {"id": f"trip0-{t:04d}-{r}", "correct": c}
            for t, triple in enumerate(correct)
            for r, c in zip(("real", "other", "none"), triple, strict=True)
        ]
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        return str(p)

    a = write("a.jsonl", [(True, False, True), (True, True, True)])
    b = write("b.jsonl", [(True, True, True), (True, True, True)])
    res = mod.compare(a, b)
    assert res["full_triplets"]["a"] == 0.5 and res["full_triplets"]["b"] == 1.0
    assert res["full_triplets"]["b_minus_a"] == 0.5 and res["role_other"]["b_minus_a"] == 0.5
    c = write("c.jsonl", [(True, True, True)])
    try:
        mod.compare(a, c)
    except ValueError as exc:
        assert "mismos tríos" in str(exc)
    else:
        raise AssertionError("debía rechazar tríos distintos")
