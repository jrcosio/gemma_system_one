import math

import numpy as np
import pytest

from gemma_system_one.metrics import (
    bin_index,
    binary_metrics,
    bootstrap_by_group,
    concentration,
    multiclass_brier,
    ranked_probability_score,
    reliability,
    top_label_ece,
    within_group_pairwise,
)


def _logit(p):
    return math.log(p / (1 - p))


def test_binary_metrics_hand_computed():
    p = np.array([0.9, 0.2, 0.6, 0.4])
    y = [1, 0, 0, 1]
    m = binary_metrics([_logit(x) for x in p], y)
    assert m["brier"] == pytest.approx(np.mean((p - np.array(y)) ** 2))
    nll = -np.mean([math.log(0.9), math.log(0.8), math.log(0.4), math.log(0.4)])
    assert m["nll"] == pytest.approx(nll)
    # umbral 0.5: pred = [1,0,1,0] -> tp=1, fp=1, fn=1
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5) and m["accuracy"] == pytest.approx(0.5)
    assert m["prevalence"] == 0.5


def test_nll_is_stable_for_extreme_logits():
    m = binary_metrics([80.0, -80.0], [1, 0])
    assert m["nll"] == pytest.approx(0.0, abs=1e-30)
    m = binary_metrics([80.0], [0])
    assert m["nll"] == pytest.approx(80.0)


def test_undefined_metrics_are_none_not_zero():
    m = binary_metrics([-5.0, -4.0], [0, 0])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["single_class"] is True


def test_ece_bins_cover_both_extremes():
    assert bin_index(np.array([0.0, 1.0, 1 / 15, 0.999999])).tolist() == [0, 14, 1, 14]
    ece, table = reliability(np.array([0.0, 1.0]), np.array([0, 1]))
    assert ece == 0.0
    assert sum(r["count"] for r in table) == 2 and len(table) == 15
    ece, _ = reliability(np.array([1.0, 1.0]), np.array([0, 0]))
    assert ece == pytest.approx(1.0)


def test_ece_empty_bins_contribute_zero():
    ece, table = reliability(np.array([0.7, 0.7]), np.array([1, 0]))
    assert ece == pytest.approx(0.2)
    assert sum(1 for r in table if r["count"]) == 1


def test_multiclass_brier_sum_convention_and_rps():
    probs = np.array([[1.0, 0.0, 0.0], [0.2, 0.5, 0.3]])
    assert multiclass_brier(probs, [0, 1]) == pytest.approx((0 + (0.04 + 0.25 + 0.09)) / 2)
    # RPS de [0.2,0.5,0.3] con y=1: F=[0.2,0.7]; ind=[0,1] -> (0.04+0.09)/2
    assert ranked_probability_score(probs[1:], [1]) == pytest.approx(0.065)
    assert ranked_probability_score(np.array([[0, 0, 1.0]]), [2]) == 0.0
    assert ranked_probability_score(np.array([[1.0, 0, 0]]), [2]) == pytest.approx(1.0)


def test_top_label_ece_uses_max_prob_and_argmax():
    assert top_label_ece(np.array([[0.9, 0.1], [0.9, 0.1]]), [0, 1]) == pytest.approx(0.4)


def test_concentration_definition():
    c = concentration(np.array([[1.0, 0.0], [0.5, 0.5], [0.25, 0.25, 0.25, 0.25][:2]]))
    assert c[0] == pytest.approx(1.0) and c[1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        concentration(np.array([[1.0]]))


def test_bootstrap_is_deterministic_and_group_based():
    groups = ["a", "a", "b", "b", "c"]
    vals = np.array([1.0, 1.0, 0.0, 0.0, 1.0])

    def fn(idx):
        return {"mean": float(vals[idx].mean())}

    r1 = bootstrap_by_group(groups, fn, reps=200, seed=1)
    r2 = bootstrap_by_group(groups, fn, reps=200, seed=1)
    assert r1 == r2 and r1["groups"] == 3
    lo, hi = r1["intervals"]["mean"]
    assert 0.0 <= lo <= hi <= 1.0


def test_within_group_pairwise_detects_instruction_blindness():
    groups = ["g", "g", "h", "h"]
    labels = [1, 0, 1, 0]
    assert within_group_pairwise(groups, [2.0, -1.0, 0.5, 0.1], labels)["pairwise_accuracy"] == 1.0
    # Mismo valor para todas las preguntas del estado: 0.5
    assert within_group_pairwise(groups, [1.0, 1.0, 3.0, 3.0], labels)["pairwise_accuracy"] == 0.5
    assert within_group_pairwise(["g"], [1.0], [1])["pairwise_accuracy"] is None
