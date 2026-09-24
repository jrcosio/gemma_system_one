import math

import pytest
import torch

from gemma_system_one.baselines import HashedBowLogistic, PriorBaseline
from gemma_system_one.models.heads import NoulHead
from gemma_system_one.training.noul import NoulTrainParams, accumulate_step, train_noul_head


def _grads(head):
    return torch.cat([p.grad.flatten().clone() for p in head.parameters()])


@pytest.mark.parametrize("n,mb", [(8, 3), (5, 2), (3, 8), (7, 1)])
def test_accumulated_gradient_independent_of_microbatch(n, mb):
    torch.manual_seed(0)
    reps, y = torch.randn(n, 6), (torch.rand(n) > 0.5).float()
    head = NoulHead(6)
    head.zero_grad()
    loss_full = accumulate_step(head, reps, y, microbatch_rows=n)
    g_full = _grads(head)
    head.zero_grad()
    loss_mb = accumulate_step(head, reps, y, microbatch_rows=mb)
    torch.testing.assert_close(_grads(head), g_full, atol=1e-6, rtol=1e-5)
    assert loss_mb == pytest.approx(loss_full, rel=1e-6)


def test_partial_last_step_is_normalized_by_its_real_size():
    torch.manual_seed(1)
    reps, y = torch.randn(3, 4), torch.tensor([1.0, 0.0, 1.0])
    head = NoulHead(4)
    head.zero_grad()
    accumulate_step(head, reps, y, microbatch_rows=2)
    got = _grads(head)
    head.zero_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(head(reps), y, reduction="mean").backward()
    torch.testing.assert_close(got, _grads(head))


def test_training_learns_separable_data_and_selects_on_validation():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = (x[:, 0] > 0).float()
    params = NoulTrainParams(epochs=30, lr=0.05, weight_decay=0.0, questions_per_step=8, seed=0)
    res = train_noul_head(x[:48], y[:48], params, x[48:], y[48:])
    assert 1 <= res.selected_epoch <= 30
    best = min(h["validation_nll"] for h in res.history)
    assert res.history[res.selected_epoch - 1]["validation_nll"] == best
    assert res.history[-1]["train_nll"] < res.history[0]["train_nll"]
    assert not res.head.training


def test_training_is_reproducible_with_seed():
    x, y = torch.randn(20, 5), (torch.rand(20) > 0.5).float()
    p = NoulTrainParams(epochs=3, select_on="train")
    a = train_noul_head(x, y, p).head.state_dict()
    b = train_noul_head(x, y, p).head.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_validation_selection_requires_validation_data():
    with pytest.raises(ValueError):
        train_noul_head(torch.randn(4, 3), torch.ones(4), NoulTrainParams(select_on="validation"))


def test_non_finite_representation_aborts_training():
    x = torch.randn(8, 3)
    x[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="no finito"):
        train_noul_head(x, torch.ones(8), NoulTrainParams(epochs=1, select_on="train"))


def test_prior_baseline_uses_laplace_train_prevalence():
    prior = PriorBaseline().fit([1, 0, 0, 0])
    assert prior.p == pytest.approx(2 / 6)
    assert prior.logits(2) == [pytest.approx(math.log(0.5))] * 2
    assert 0 < PriorBaseline().fit([0, 0]).p < 1  # nunca 0: NLL finita


def test_bow_baseline_fits_train_and_is_deterministic():
    texts = ["pide devolución", "no pide nada", "quiere devolución ya", "todo bien"]
    y = [1, 0, 1, 0]
    z1 = HashedBowLogistic().fit(texts, y).logits(texts)
    z2 = HashedBowLogistic().fit(texts, y).logits(texts)
    assert z1 == z2
    assert (z1[0] > 0) and (z1[1] < 0)
