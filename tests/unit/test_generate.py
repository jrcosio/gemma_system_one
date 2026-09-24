from collections import Counter

from gemma_system_one.data.dataset import validate_examples
from gemma_system_one.data.generate import (
    FEATURES,
    ITEMS,
    QUESTIONS,
    RULES,
    SENTENCES,
    generate,
    write_dataset,
)
from gemma_system_one.serialization import canonical_state


def test_generation_is_deterministic(tmp_path):
    m1 = write_dataset(tmp_path / "a", 20, seed=3)
    m2 = write_dataset(tmp_path / "b", 20, seed=3)
    assert m1["examples_sha256"] == m2["examples_sha256"]
    assert m1["examples_sha256"] != write_dataset(tmp_path / "c", 20, seed=4)["examples_sha256"]


def test_labels_follow_declared_rules_and_state_contains_the_facts():
    examples, audit = generate(40, seed=0)
    facts = {a["group_id"]: a["facts"] for a in audit}
    kind_of = {q.label_method: q.kind for q in QUESTIONS}
    for e in examples:
        f = facts[e.group_id]
        assert e.target.label == int(RULES[kind_of[e.provenance.label_method]](f))
        text = canonical_state(e.state)
        for name, value in f.items():
            if value == "absent":
                continue
            # Cada hecho presente aparece con alguna de sus frases ya formateadas.
            rendered = {
                t.format(item=i, feature=ft, Feature=ft[:1].upper() + ft[1:])
                for t in SENTENCES[e.language][name][value]
                for i in ITEMS[e.language]
                for ft in FEATURES[e.language]
            }
            assert any(r in text for r in rendered), (e.id, name, value)
        assert not any(ph in text for ph in ("{item}", "{feature}", "{Feature}"))


def test_dataset_has_contrastive_groups_negations_and_both_classes():
    examples, audit = generate(40, seed=0)
    report = validate_examples(__import__("pathlib").Path("."), examples)
    assert report.ok, report.errors
    assert report.stats["groups_with_contrasting_labels"] >= 10
    assert set(report.stats["label_distribution"]["noul"]) == {"0", "1"}
    values = Counter(v for a in audit for v in a["facts"].values())
    # Distractores con vocabulario compartido presentes en el fixture.
    assert values["declined"] > 0 and values["past"] > 0 and values["resolved"] > 0 and values["denied"] > 0
    assert Counter(e.language for e in examples) == {"es": 60, "en": 60}
