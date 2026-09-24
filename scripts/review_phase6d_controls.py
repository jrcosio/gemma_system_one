"""Revisión de los controles de fase 6d, sólo lectura de datasets y predicciones guardadas.

Uso: .venv/bin/python scripts/review_phase6d_controls.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.derive import DIAG_DISTRACTOR_OPTIONS
from gemma_system_one.data.generate_mixed import fault_categories, generate_mixed, true_fault_category


def category_map(lang: str) -> dict[str, str]:
    cats = {**fault_categories(lang), **DIAG_DISTRACTOR_OPTIONS[lang]}
    return {text: category for category, variants in cats.items() for text in variants}


def review(name: str, source: str, seed: int, cases: int) -> dict:
    original = [e for e in load_dataset(source).examples if e.task_family == "fault_type"]
    swapped_name = "final13_faultswap" if name == "final13" else f"{name}swap"
    swapped = {e.id: e for e in load_dataset(f"data/pilot_v4_{swapped_name}").examples}
    _, audit = generate_mixed(cases, seed, version="v4")
    facts = {row["group_id"]: row["facts"] for row in audit}
    groups = sorted({e.group_id for e in original})
    perm = groups[:]
    rng = random.Random("swap:0")
    while any(a == b for a, b in zip(groups, perm, strict=True)):
        rng.shuffle(perm)
    donor = dict(zip(groups, perm, strict=True))
    by_group = {e.group_id: e for e in original}
    semantic = {}
    conflicts = language_changes = 0
    signatures = []
    for e in original:
        d = by_group[donor[e.group_id]]
        assert swapped[e.id].state == d.state
        language_changes += e.language != d.language
        categories = category_map(e.language)
        present = frozenset(categories[text] for text in e.question.criteria.values())
        actual = true_fault_category(facts[d.group_id])
        expected = actual if actual in present else "other"
        semantic[e.id] = expected
        label = categories[e.question.criteria[e.target.class_id]]
        conflicts += label != expected
        signatures.append((present, label))
    result = {"questions": len(original), "target_conflicts": conflicts, "language_changes": language_changes}
    if name == "kdiag14_K4":
        n = len(signatures)
        by_signature = Counter(s for s, _ in signatures)
        by_label = Counter(y for _, y in signatures)
        joint = Counter(signatures)
        result["empirical_mutual_information_bits"] = round(
            sum(c / n * math.log2(c * n / (by_signature[s] * by_label[y])) for (s, y), c in joint.items()),
            4,
        )
        result["signatures"] = len(by_signature)
    model_results = {}
    by_id = {e.id: e for e in original}
    for model in ("a4v3", "a4v4", "a2v4"):
        path = Path(f"reports/phase6d/pred_{swapped_name}_{model}.txt").read_text().strip()
        predictions = [json.loads(line) for line in Path(path).read_text().splitlines()]
        correct_original = correct_donor = 0
        for row in predictions:
            e = by_id[row["id"]]
            categories = category_map(e.language)
            guess = categories[e.question.criteria[row["answer"]["choice"]]]
            correct_original += guess == categories[e.question.criteria[e.target.class_id]]
            correct_donor += guess == semantic[e.id]
        model_results[model] = {
            "accuracy_original_target": round(correct_original / len(predictions), 4),
            "accuracy_donor_semantic": round(correct_donor / len(predictions), 4),
        }
    result["models"] = model_results
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "kdiag14_K4": review("kdiag14_K4", "data/pilot_v4_kdiag14_K4", 14, 400),
                "kdiag14_K8": review("kdiag14_K8", "data/pilot_v4_kdiag14_K8", 14, 400),
                "final13": review("final13", "data/pilot_v4_final13", 13, 300),
            },
            indent=2,
        )
    )
