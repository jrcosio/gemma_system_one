"""Fase 6: holdout sin solapes y más opciones en Choice sin cambiar la respuesta correcta."""

from __future__ import annotations

import pytest

from gemma_system_one.contracts import CHOICE_MAX
from gemma_system_one.data.derive import (
    NEVER_TRUE_FAULT_OPTIONS,
    _category_of,
    exclude_overlap,
    widen_fault_kind,
)
from gemma_system_one.data.generate_mixed import generate_mixed


def test_overlap_excludes_whole_groups():
    ref, _ = generate_mixed(40, seed=0)
    new, _ = generate_mixed(40, seed=1)
    # Forzar un solape: una pregunta del holdout con el estado literal de un caso de referencia.
    victim = new[4].group_id
    new = [e.model_copy(update={"state": ref[0].state}) if e.id == new[4].id else e for e in new]
    kept, excluded = exclude_overlap(new, ref)
    assert victim in excluded
    assert all(e.group_id not in excluded for e in kept)
    assert {e.group_id for e in kept} | set(excluded) == {e.group_id for e in new}


def test_widen_keeps_label_original_options_and_adds_only_wrong_ones():
    examples, _ = generate_mixed(200, seed=5)
    fault = {e.id: e for e in examples if e.task_family == "fault_type"}
    wide = widen_fault_kind(examples, seed=0)
    assert [e.id for e in wide] == list(fault)
    never = {t for opts in NEVER_TRUE_FAULT_OPTIONS.values() for v in opts.values() for t in v}
    saw_other = False
    for w in wide:
        o = fault[w.id]
        assert w.target == o.target and w.state == o.state
        assert w.question.instructions == o.question.instructions
        assert len(w.question.criteria) <= CHOICE_MAX
        # Las opciones originales, con sus IDs, siguen presentes; sólo se añaden otras.
        assert o.question.criteria.items() <= w.question.criteria.items()
        added = [t for i, t in w.question.criteria.items() if i not in o.question.criteria]
        label_cat = _category_of(o.question.criteria[o.target.class_id], o.language)
        if label_cat == "other":
            saw_other = True
            # La categoría real falta y no se conoce desde la pregunta: sólo categorías imposibles.
            assert all(t in never for t in added)
        else:
            present = {_category_of(t, o.language) for t in o.question.criteria.values()}
            for t in added:
                assert t in never or _category_of(t, o.language) not in present
        if added:
            assert len(w.question.criteria) == CHOICE_MAX or all(t in never for t in added)
    assert saw_other
    assert any(len(w.question.criteria) == CHOICE_MAX for w in wide)


def test_widen_is_deterministic():
    examples, _ = generate_mixed(60, seed=2)
    a = [e.model_dump() for e in widen_fault_kind(examples, seed=0)]
    b = [e.model_dump() for e in widen_fault_kind(examples, seed=0)]
    assert a == b


def test_comparison_with_more_candidates_requires_same_semantic_label():
    from gemma_system_one.metrics import compare_predictions

    base = {
        "id": "q",
        "group_id": "g",
        "type": "choice",
        "input_sha256": "a" * 64,
        "nll": 1.0,
        "correct": True,
    }
    a = [{**base, "target_index": 1, "target_description": "Red", "row_logits": [0.0, 1.0, 0.5]}]
    wide = {**base, "input_sha256": "b" * 64, "target_index": 3, "row_logits": [0.0] * 8, "nll": 1.5}
    ok = compare_predictions(a, [{**wide, "target_description": "Red"}], reps=5, allow_different_inputs=True)
    assert ok["point"]["b_minus_a.nll_all"] == 0.5
    with pytest.raises(ValueError, match="target_description"):
        compare_predictions(a, [{**wide, "target_description": "Otro"}], reps=5, allow_different_inputs=True)
    with pytest.raises(ValueError, match="target_index"):  # sin el control explícito, nunca
        compare_predictions(a, [{**wide, "target_description": "Red"}], reps=5)
    same_k = {**wide, "row_logits": [0.0] * 3, "target_description": "Red"}
    with pytest.raises(ValueError, match="target_index"):  # mismo K: el índice sigue exigiéndose
        compare_predictions(a, [same_k], reps=5, allow_different_inputs=True)
