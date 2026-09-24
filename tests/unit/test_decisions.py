"""Pérdida de grupo, acumulación por pregunta, reconstrucción y baselines de fase 2."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from conftest import make_example
from gemma_system_one.baselines import HashedBowGroupScorer, LevelPriorBaseline, UniformChoiceBaseline
from gemma_system_one.contracts import ChoiceQuestion, ScoreQuestion
from gemma_system_one.inference import InvalidOutputError, reconstruct, target_index
from gemma_system_one.metrics import ranked_probability_score
from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.serialization import expand
from gemma_system_one.training.decisions import (
    DecisionTrainParams,
    QuestionItem,
    accumulate_step,
    build_items,
    flatten,
    group_logits,
    question_loss,
    rps_loss,
    train_decision_heads,
)

CHOICE = {
    "type": "choice",
    "instructions": "¿Qué equipo?",
    "criteria": {"a1": "Cobros", "b2": "Soporte", "c3": "Cuentas"},
}
SCORE = {"type": "score", "instructions": "Alcance", "criteria": ["Nadie", "Algunos", "Todos"]}


def _items():
    return build_items(
        [
            make_example(id="n1", group_id="g1"),
            make_example(id="c1", group_id="g1", question=CHOICE, target={"class_id": "b2"}),
            make_example(id="s1", group_id="g1", question=SCORE, target={"level_index": 2}),
            make_example(id="n2", group_id="g2", state="otro estado", target={"label": 0}),
        ]
    )


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_group_loss_is_identical_with_and_without_chunking(chunk):
    torch.manual_seed(0)
    heads = DecisionHeads(6)
    reps = torch.randn(5, 6)
    full = question_loss("choice", group_logits(heads, "choice", reps, 5), 3)
    full.backward()
    g_full = [p.grad.clone() for p in heads.heads["choice"].parameters()]
    heads.zero_grad()
    chunked = question_loss("choice", group_logits(heads, "choice", reps, chunk), 3)
    chunked.backward()
    torch.testing.assert_close(chunked, full)
    for p, g in zip(heads.heads["choice"].parameters(), g_full, strict=True):
        torch.testing.assert_close(p.grad, g)


def test_softmax_is_over_the_whole_group_not_per_chunk():
    z = torch.tensor([2.0, 0.0, 1.0, -1.0])
    full = question_loss("choice", z, 0)
    assert full == pytest.approx(-torch.log_softmax(z, 0)[0])
    wrong = -torch.log_softmax(z[:2], 0)[0]  # normalizar sólo el trozo donde está el target
    assert not torch.isclose(full, wrong)


def test_rps_loss_matches_metric_and_lambda_zero_is_pure_ce():
    z = torch.tensor([0.3, -1.0, 2.0, 0.1], requires_grad=True)
    p = torch.softmax(z, 0).detach().numpy()[None]
    assert float(rps_loss(z, 1).detach()) == pytest.approx(ranked_probability_score(p, [1]), rel=1e-6)
    ce = question_loss("score", z, 1, 0.0)
    assert float(ce.detach()) == pytest.approx(float(-torch.log_softmax(z, 0)[1].detach()))
    both = question_loss("score", z, 1, 1.0)
    assert float(both.detach()) == pytest.approx(float(ce.detach()) + float(rps_loss(z, 1).detach()))
    both.backward()
    assert torch.isfinite(z.grad).all()


def test_accumulation_is_weighted_mean_per_question_including_partial_step():
    items = _items()
    texts, _, offsets = flatten(items)
    torch.manual_seed(1)
    reps = torch.randn(len(texts), 6)
    weights = {"noul": 1.0, "choice": 2.0, "score": 0.5}
    params = DecisionTrainParams(questions_per_step=8, logit_chunk_rows=2, type_weights=weights)
    heads = DecisionHeads(6)
    batch = [0, 1, 2]  # paso parcial: 3 preguntas
    loss = accumulate_step(heads, batch, items, reps, offsets, params)
    got = {n: p.grad.clone() for n, p in heads.named_parameters() if p.grad is not None}
    heads.zero_grad(set_to_none=True)
    manual = 0
    for qi in batch:
        s, k = offsets[qi]
        it = items[qi]
        manual = manual + weights[it.primitive] * question_loss(
            it.primitive, heads(it.primitive, reps[s : s + k]), it.target
        )
    (manual / 3).backward()
    assert loss == pytest.approx(float(manual.detach()) / 3)
    for n, p in heads.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(got[n], p.grad)


def test_zero_type_weight_gives_no_gradient_to_that_head():
    items = _items()
    texts, _, offsets = flatten(items)
    reps = torch.randn(len(texts), 6)
    params = DecisionTrainParams(type_weights={"noul": 1.0, "choice": 0.0, "score": 1.0})
    heads = DecisionHeads(6)
    accumulate_step(heads, [0, 1, 2], items, reps, offsets, params)
    assert all(float(p.grad.abs().sum()) == 0.0 for p in heads.heads["choice"].parameters())
    assert any(float(p.grad.abs().sum()) > 0 for p in heads.heads["score"].parameters())


def test_type_weights_must_cover_all_primitives():
    with pytest.raises(ValueError):
        DecisionTrainParams(type_weights={"noul": 1.0})


def test_target_index_remaps_by_id_after_canonical_ordering():
    q = ChoiceQuestion.model_validate(CHOICE)
    rows = expand("s", q)
    idx = target_index(q, rows, make_example(question=CHOICE, target={"class_id": "b2"}).target)
    assert rows[idx].candidate_id == "b2"
    renamed = ChoiceQuestion.model_validate(
        {**CHOICE, "criteria": {"zz": "Cuentas", "yy": "Soporte", "xx": "Cobros"}}
    )
    rows2 = expand("s", renamed)
    assert [r.text for r in rows2] == [r.text for r in rows]
    assert rows2[idx].candidate_id == "yy"


def _fake_logits(rows):
    """Puntuación determinista que depende sólo del texto de la fila (doble de test)."""
    return torch.tensor([int(hashlib.md5(r.text.encode()).hexdigest()[:4], 16) / 6553.6 for r in rows])


def test_reconstruct_choice_invariant_to_permutation_and_renaming():
    q1 = ChoiceQuestion.model_validate(CHOICE)
    q2 = ChoiceQuestion.model_validate(
        {**CHOICE, "criteria": {"k3": "Cuentas", "k1": "Cobros", "k2": "Soporte"}}
    )
    r1, r2 = expand("estado", q1), expand("estado", q2)
    a1, a2 = reconstruct(q1, r1, _fake_logits(r1)), reconstruct(q2, r2, _fake_logits(r2))
    mapping = {"a1": "k1", "b2": "k2", "c3": "k3"}
    for k, v in mapping.items():
        assert a1["probabilities"][k] == pytest.approx(a2["probabilities"][v], abs=0)
    assert mapping[a1["choice"]] == a2["choice"]
    assert sum(a1["probabilities"].values()) == pytest.approx(1.0)
    assert 0.0 <= a1["confidence"] <= 1.0 and a1["choice"] in q1.criteria


def test_reconstruct_score_expectation_legend_order_and_confidence():
    q = ScoreQuestion.model_validate(SCORE)
    rows = expand("s", q)
    ans = reconstruct(q, rows, torch.tensor([0.0, 0.0, 0.0]))
    assert ans["score"] == pytest.approx(1.0) and ans["legend"] == SCORE["criteria"]
    assert ans["confidence"] == pytest.approx(0.0, abs=1e-12)  # uniforme
    ans = reconstruct(q, rows, torch.tensor([-50.0, -50.0, 50.0]))
    assert ans["score"] == pytest.approx(2.0) and ans["confidence"] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(InvalidOutputError):
        reconstruct(q, rows, torch.tensor([0.0, float("nan"), 0.0]))
    with pytest.raises(InvalidOutputError):
        reconstruct(q, rows, torch.tensor([0.0, 1.0]))


def test_reversed_rubric_changes_rows_and_meaning():
    q = ScoreQuestion.model_validate(SCORE)
    rev = ScoreQuestion.model_validate({**SCORE, "criteria": list(reversed(SCORE["criteria"]))})
    assert [r.text for r in expand("s", q)] != [r.text for r in expand("s", rev)]
    assert '<evaluated-level index="0">Todos</evaluated-level>' in expand("s", rev)[0].text


def test_training_learns_all_primitives_and_leaves_absent_heads_untouched():
    torch.manual_seed(0)
    d = 8
    items, reps = [], []
    for i in range(60):
        k = 3
        target = i % k
        block = torch.randn(k, d) * 0.1
        block[target, 0] += 3.0  # señal: la fila correcta tiene la dimensión 0 alta
        reps.append(block)
        items.append(QuestionItem(make_example(id=f"c{i}", group_id=f"g{i}", question=CHOICE,
                                               target={"class_id": "a1"}), [None] * k, target))  # fmt: skip
    reps_t = torch.cat(reps)
    offsets = [(3 * i, 3) for i in range(60)]
    params = DecisionTrainParams(epochs=20, lr=0.05, weight_decay=0.0, select_on="train")
    before = {n: p.clone() for n, p in DecisionHeads(d).named_parameters()}
    res = train_decision_heads(items, reps_t, offsets, params)
    assert res.history[-1]["train_nll"]["choice"] < res.history[0]["train_nll"]["choice"]
    assert res.history[-1]["train_nll"]["noul"] is None
    torch.manual_seed(0)  # misma inicialización que dentro del entrenamiento
    init = DecisionHeads(d).state_dict()
    for name in ("noul", "score"):
        for key, v in res.heads.heads[name].state_dict().items():
            torch.testing.assert_close(v, init[f"heads.{name}.{key}"])
    assert before  # la inicialización es determinista con semilla


def test_group_baselines():
    assert UniformChoiceBaseline().logits([3, 2]) == [[0.0] * 3, [0.0] * 2]
    prior = LevelPriorBaseline().fit([3, 3, 3], [2, 2, 0])
    z = np.array(prior.logits([3])[0])
    p = np.exp(z) / np.exp(z).sum()
    assert p == pytest.approx(np.array([2, 1, 3]) / 6)
    assert prior.logits([5])[0] == [0.0] * 5  # M no vista: uniforme
    groups = [["pide cobro", "pide soporte"], ["soporte ya", "cobro ya"], ["cobro", "soporte"]]
    targets = [0, 1, 0]
    a = HashedBowGroupScorer().fit(groups, targets).logits(groups)
    b = HashedBowGroupScorer().fit(groups, targets).logits(groups)
    assert a == b and all(int(np.argmax(z)) == t for z, t in zip(a, targets, strict=True))


def test_standardizer_fits_only_given_train_reps_and_roundtrips(tmp_path):
    from gemma_system_one.checkpoint import (
        CheckpointMismatchError,
        load_decision_heads,
        read_manifest,
        save_decision_heads,
    )

    heads = DecisionHeads(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(heads.standardizer(x), x)  # sin ajustar: identidad
    train = torch.randn(50, 4) * torch.tensor([100.0, 1.0, 5.0, 0.1]) + 7
    heads.standardizer.fit(train)
    z = heads.standardizer(train)
    torch.testing.assert_close(z.mean(0), torch.zeros(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(z.std(0, unbiased=False), torch.ones(4), atol=1e-4, rtol=0)
    kw = dict(
        repo_id="google/gemma-4-E2B-it", revision="a" * 40, backbone_dtype="bfloat16", prompt_template="t"
    )
    d = save_decision_heads(tmp_path / "c", heads, **kw)
    loaded, _ = load_decision_heads(d, repo_id=kw["repo_id"], revision=kw["revision"], hidden_size=4)
    torch.testing.assert_close(loaded.standardizer.mean, heads.standardizer.mean)
    assert bool(loaded.standardizer.fitted)
    # Un checkpoint de formato anterior (sin estandarizador) se rechaza, sin migración silenciosa.
    m = read_manifest(d)
    m["format_version"] = 2
    (d / "manifest.json").write_text(__import__("json").dumps(m))
    with pytest.raises(CheckpointMismatchError, match="Formato"):
        load_decision_heads(d, repo_id=kw["repo_id"], revision=kw["revision"], hidden_size=4)


def test_training_standardizes_with_train_statistics_only():
    torch.manual_seed(3)
    items = _items()
    texts, _, offsets = flatten(items)
    reps = torch.randn(len(texts), 6) * 50
    res = train_decision_heads(items, reps, offsets, DecisionTrainParams(epochs=1, select_on="train"))
    torch.testing.assert_close(res.heads.standardizer.mean, reps.mean(0))
    off = train_decision_heads(
        items, reps, offsets, DecisionTrainParams(epochs=1, select_on="train", feature_norm="none")
    )
    assert not bool(off.heads.standardizer.fitted)
