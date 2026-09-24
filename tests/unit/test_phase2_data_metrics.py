"""Métricas Choice/Score, generador mixto y duplicados semánticos (fase 2)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import make_example
from gemma_system_one.contracts import ChoiceQuestion
from gemma_system_one.data.dataset import serialized_input_hash, validate_examples
from gemma_system_one.data.generate_mixed import (
    FAULT_KIND_OPTIONS,
    NOUL_RULES,
    TEAM_RULE,
    generate_mixed,
    impact_level,
    write_mixed_dataset,
)
from gemma_system_one.data.split import leakage_checks, near_duplicate_states
from gemma_system_one.metrics import categorical_metrics, ordinal_metrics
from gemma_system_one.serialization import canonical_state


def test_categorical_metrics_hand_computed_and_by_k():
    probs = [[0.7, 0.2, 0.1], [0.5, 0.5]]
    m = categorical_metrics(probs, [0, 1])
    assert m["nll"] == pytest.approx((-np.log(0.7) - np.log(0.5)) / 2)
    assert m["brier_sum"] == pytest.approx(((0.09 + 0.04 + 0.01) + (0.25 + 0.25)) / 2)
    assert m["accuracy"] == pytest.approx(0.5)  # empate [0.5,0.5]: argmax=0 ≠ 1
    assert set(m["by_k"]) == {"2", "3"} and m["by_k"]["3"]["n"] == 1
    assert m["uniform_nll"] == pytest.approx((np.log(3) + np.log(2)) / 2)
    with pytest.raises(ValueError):
        categorical_metrics([[0.6, 0.6]], [0])


def test_ordinal_metrics_rps_mae_and_cumulative_events():
    m = ordinal_metrics([[0.2, 0.5, 0.3]], [1])
    assert m["rps"] == pytest.approx(0.065)
    assert m["mae"] == pytest.approx(abs(1.1 - 1))
    assert m["mae_norm"] == pytest.approx(0.05)
    assert m["cumulative_events"] == 2 and m["accuracy"] == 1.0
    perfect = ordinal_metrics([[0, 0, 1.0], [1.0, 0, 0, 0]], [2, 0])
    assert perfect["rps"] == 0 and perfect["mae"] == 0 and perfect["cumulative_event_ece_15bins"] == 0
    assert set(perfect["by_m"]) == {"3", "4"}


@pytest.mark.parametrize(
    "x,y,m,level", [(0, 10, 3, 0), (10, 10, 3, 2), (3, 10, 3, 1), (4, 10, 4, 1), (5, 10, 4, 2),
                    (10, 10, 4, 3), (2, 8, 5, 1), (3, 8, 5, 2), (4, 8, 5, 2), (5, 8, 5, 3), (8, 8, 5, 4)],
)  # fmt: skip
def test_impact_rule_boundaries(x, y, m, level):
    assert impact_level(x, y, m) == level


def test_mixed_generation_is_deterministic_and_valid(tmp_path):
    a = write_mixed_dataset(tmp_path / "a", 40, 3)
    b = write_mixed_dataset(tmp_path / "b", 40, 3)
    assert a["examples_sha256"] == b["examples_sha256"]
    examples, _ = generate_mixed(200, 0)
    report = validate_examples(Path("."), examples)
    assert report.ok, report.errors[:3]
    assert set(report.stats["by_type"]) == {"noul", "choice", "score"}
    assert len({canonical_state(e.state) for e in examples}) == 200  # un estado único por caso


def test_mixed_labels_follow_rules_and_ids_are_opaque():
    examples, audit = generate_mixed(300, 1)
    facts = {a["group_id"]: a["facts"] for a in audit}
    noul_kind = {"explicit_refund_request": "refund", "explicit_duplicate_charge": "double",
                 "active_fault_statement": "fault", "explicit_access_block": "locked",
                 "policy_and(duplicate_charge,refund_request)": "escalate"}  # fmt: skip
    labels_other = 0
    for e in examples:
        f = facts[e.group_id]
        q = e.question
        if q.type == "noul":
            assert e.target.label == int(NOUL_RULES[noul_kind[e.provenance.label_method]](f))
        elif q.type == "choice":
            assert all(k.startswith("o") and len(k) == 7 for k in q.criteria)  # IDs opacos
            desc = q.criteria[e.target.class_id]
            if e.task_family == "routing":
                assert len(q.criteria) == 4
                assert desc.split()[-1] in q.instructions or desc in q.instructions
                assert TEAM_RULE(f) in {"billing", "accounts", "technical", "general"}
            else:
                assert 3 <= len(q.criteria) <= 6
                lang = e.language
                cat = next(c for c, ds in FAULT_KIND_OPTIONS[lang].items() if desc in ds)
                truth = "none" if f["fault"] == "absent" else f["fault_kind"]
                if cat == "other":
                    labels_other += 1
                    present = {next(c for c, ds in FAULT_KIND_OPTIONS[lang].items() if d in ds)
                               for d in q.criteria.values()}  # fmt: skip
                    assert truth not in present  # "other" sólo si la categoría real no está
                else:
                    assert cat == truth
        else:
            m = len(q.criteria)
            assert 0 <= e.target.level_index < m
            if e.task_family == "service_impact":
                expected = impact_level(*f["impact"], m)
            else:
                expected = {"absent": 0, "resolved": 1, "active": 2}[f["fault"]]
            if e.provenance.label_method.endswith("_descending"):
                expected = m - 1 - expected
            assert e.target.level_index == expected
    assert labels_other > 0


def _surface(examples):
    texts = set()
    for e in examples:
        texts.add(e.question.instructions)
        if hasattr(e.question, "criteria"):
            vals = (
                e.question.criteria.values()
                if isinstance(e.question, ChoiceQuestion)
                else e.question.criteria
            )
            texts.update(vals)
    return texts


def test_transfer_variant_uses_disjoint_surface_templates():
    main, _ = generate_mixed(300, 0, "main")
    transfer, _ = generate_mixed(100, 0, "transfer")
    assert _surface(main).isdisjoint(_surface(transfer))
    assert {e.group_id for e in main}.isdisjoint({e.group_id for e in transfer})
    checks = leakage_checks({"main": main, "transfer": transfer})
    assert checks["errors"] == []
    families = lambda xs: Counter(e.task_family for e in xs)  # noqa: E731
    assert set(families(transfer)) <= set(families(main))


def test_semantic_duplicates_with_renamed_ids_are_detected():
    q1 = {"type": "choice", "instructions": "¿Equipo?", "criteria": {"a": "Cobros", "b": "Soporte"}}
    q2 = {"type": "choice", "instructions": "¿Equipo?", "criteria": {"y": "Soporte", "x": "Cobros"}}
    e1 = make_example(id="e1", group_id="g1", question=q1, target={"class_id": "a"})
    e2 = make_example(id="e2", group_id="g2", question=q2, target={"class_id": "x"})
    assert serialized_input_hash(e1) == serialized_input_hash(e2)
    r = validate_examples(Path("."), [e1, e2])
    assert any("misma serialización en grupos distintos" in err for err in r.errors)
    e3 = make_example(id="e3", group_id="g1", question=q2, target={"class_id": "y"})  # otra etiqueta
    r = validate_examples(Path("."), [e1, e3])
    assert any("etiquetas distintas" in err for err in r.errors)
    checks = leakage_checks({"train": [e1], "validation": [e2]})
    assert any("igual serialización" in err for err in checks["errors"])


def test_near_duplicates_consider_every_state_of_a_group():
    base = "el cliente indica que la vpn se desconecta cada pocos minutos desde esta mañana"
    a1 = make_example(
        id="a1", group_id="ga", state="texto totalmente distinto sobre facturas y cobros de marzo"
    )
    a2 = make_example(id="a2", group_id="ga", state=base)  # segundo estado del grupo
    b1 = make_example(id="b1", group_id="gb", state=base + " hoy")
    near, total = near_duplicate_states({"train": [a1, a2], "validation": [b1]})
    assert near and near[0]["groups"] == ["ga", "gb"] and total == 1


def test_split_manifest_for_mixed_dataset_is_leak_free(tmp_path):
    from gemma_system_one.data.dataset import load_dataset
    from gemma_system_one.data.split import load_split, make_split, split_path, write_split

    write_mixed_dataset(tmp_path, 60, 2)
    ds = load_dataset(tmp_path)
    m = make_split(ds, 0)
    assert m["checks"]["errors"] == []
    write_split(m, split_path(tmp_path, 0))
    parts = load_split(ds, split_path(tmp_path, 0), ("train", "validation"))
    assert sum(map(len, parts.values())) < len(ds.examples)
    json.dumps(m)  # serializable


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_mixed_generation_is_reproducible_across_processes(tmp_path, version):
    """Regresión P2-2: el orden de un set de str cambia con PYTHONHASHSEED entre procesos."""
    import os
    import subprocess
    import sys

    hashes = set()
    for seed in ("0", "12345"):
        out = tmp_path / f"{version}-{seed}"
        code = (
            "from pathlib import Path; from gemma_system_one.data.generate_mixed import write_mixed_dataset;"
            f"print(write_mixed_dataset(Path({str(out)!r}), 40, 0, version={version!r})['examples_sha256'])"
        )
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        hashes.add(res.stdout.strip().splitlines()[-1])
    assert len(hashes) == 1


def test_v2_rubrics_include_descending_orientation_with_consistent_labels():
    examples, audit = generate_mixed(300, 0, version="v2")
    facts = {a["group_id"]: a["facts"] for a in audit}
    desc = [e for e in examples if e.provenance.label_method.endswith("_descending")]
    asc = [
        e
        for e in examples
        if e.question.type == "score" and not e.provenance.label_method.endswith("_descending")
    ]
    assert desc and asc
    for e in desc:
        m = len(e.question.criteria)
        f = facts[e.group_id]
        if e.task_family == "fault_severity":
            asc_level = {"absent": 0, "resolved": 1, "active": 2}[f["fault"]]
        else:
            asc_level = impact_level(*f["impact"], m)
        assert e.target.level_index == m - 1 - asc_level
    v1, _ = generate_mixed(100, 0, version="v1")
    assert not any(e.provenance.label_method.endswith("_descending") for e in v1)
