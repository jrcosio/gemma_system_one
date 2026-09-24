"""Fase 3: generador v3 (política de acceso explícita) y límite del detector de casi duplicados."""

from __future__ import annotations

import pytest

from conftest import make_example
from gemma_system_one.data.generate_mixed import (
    ACCESS_POLICY_CLAUSE,
    FAULT_FAMILIES,
    generate_mixed,
)
from gemma_system_one.data.split import leakage_checks, near_duplicate_states


@pytest.mark.parametrize("variant", ["main", "transfer"])
def test_v3_only_adds_the_access_clause_to_fault_questions(variant):
    """v3 = v2 con la cláusula: mismos estados, etiquetas, grupos, IDs y criterios."""
    v2, audit2 = generate_mixed(200, 0, variant, version="v2")
    v3, audit3 = generate_mixed(200, 0, variant, version="v3")
    assert audit2 == audit3 and len(v2) == len(v3)
    changed = 0
    for a, b in zip(v2, v3, strict=True):
        assert (a.id, a.group_id, a.state, a.target, a.task_family) == (
            b.id,
            b.group_id,
            b.state,
            b.target,
            b.task_family,
        )
        assert getattr(a.question, "criteria", None) == getattr(b.question, "criteria", None)
        assert b.provenance.generator_version == "support-mixed-v3"
        if a.task_family in FAULT_FAMILIES:
            clause = ACCESS_POLICY_CLAUSE[a.language]
            assert b.question.instructions == f"{a.question.instructions} {clause}"
            changed += 1
        else:
            assert b.question.instructions == a.question.instructions
    assert changed > 0


def test_v3_locked_account_is_never_a_technical_failure_label():
    """La cláusula describe la regla con la que ya se etiquetaba: bloqueo sin fallo -> no hay fallo."""
    examples, audit = generate_mixed(400, 1, version="v3")
    facts = {a["group_id"]: a["facts"] for a in audit}
    locked_no_fault = [
        e
        for e in examples
        if e.task_family == "service_fault"
        and facts[e.group_id]["access"] == "locked"
        and facts[e.group_id]["fault"] != "active"
    ]
    assert locked_no_fault and all(e.target.label == 0 for e in locked_no_fault)


def test_near_duplicate_limit_is_applied_and_total_reported():
    base = "el cliente indica que la vpn se desconecta cada pocos minutos desde esta mañana"
    train = [make_example(id=f"a{i}", group_id=f"ga{i}", state=f"{base} caso {i}") for i in range(4)]
    val = [make_example(id=f"b{i}", group_id=f"gb{i}", state=f"{base} caso {i}.") for i in range(3)]
    near, total = near_duplicate_states({"train": train, "validation": val}, limit=2)
    assert len(near) == 2 and total > 2
    checks = leakage_checks({"train": train[:1], "validation": val[:1]})
    assert "near_duplicate_states_total" not in checks  # sin truncar: formato histórico intacto
