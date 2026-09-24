from pathlib import Path

import pytest
import torch

from conftest import make_example
from gemma_system_one.contracts import ChoiceQuestion, NoulQuestion, ScoreQuestion
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import write_mixed_dataset
from gemma_system_one.data.split import SplitError, load_split, make_split, split_path, write_split
from gemma_system_one.inference import reconstruct
from gemma_system_one.serialization import expand
from gemma_system_one.training.decisions_pipeline import DecisionTrainConfig, _load


@pytest.fixture
def mixed(tmp_path):
    write_mixed_dataset(tmp_path, 12, seed=4)
    ds = load_dataset(tmp_path)
    write_split(make_split(ds, 0), split_path(tmp_path, 0))
    return ds


def test_training_rejects_requested_primitives_missing_after_subset(mixed):
    cfg = DecisionTrainConfig(
        kind="decision_heads",
        name="review",
        base_config=Path("unused"),
        dataset=mixed.root,
        train={"max_train_questions": 1},
    )
    with pytest.raises(ValueError, match="sin entrenar"):
        _load(cfg)


def test_unknown_split_has_explicit_error(mixed):
    with pytest.raises(SplitError, match="[Pp]artici"):
        load_split(mixed, split_path(mixed.root, 0), ("all",))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "question",
    [
        NoulQuestion(type="noul", instructions="q"),
        ChoiceQuestion(type="choice", instructions="q", criteria={"a": "A", "b": "B"}),
        ScoreQuestion(type="score", instructions="q", criteria=["A", "B"]),
    ],
)
def test_invalid_temperature_rejected_for_every_primitive(temperature, question):
    rows = expand("state", question)
    with pytest.raises(ValueError, match="temperatura"):
        reconstruct(question, rows, torch.ones(len(rows)), temperature)


def test_choice_statistics_do_not_use_opaque_ids(tmp_path):
    from gemma_system_one.data.dataset import validate_examples

    q1 = {"type": "choice", "instructions": "q", "criteria": {"x": "Red", "y": "Cobros"}}
    q2 = {"type": "choice", "instructions": "q", "criteria": {"new_x": "Red", "new_y": "Cobros"}}
    a = make_example(id="a", group_id="g", question=q1, target={"class_id": "x"})
    b = make_example(id="b", group_id="g", question=q2, target={"class_id": "new_x"})
    report = validate_examples(tmp_path, [a, b])
    assert report.stats["groups_with_contrasting_labels"] == 0


def test_requested_zero_weight_head_is_not_presented_as_trained(mixed):
    cfg = DecisionTrainConfig(
        kind="decision_heads",
        name="review",
        base_config=Path("unused"),
        dataset=mixed.root,
        train={"type_weights": {"noul": 1.0, "choice": 0.0, "score": 1.0}},
    )
    with pytest.raises(ValueError, match="sin entrenar"):
        _load(cfg)


def test_evaluation_rejects_untrained_head_before_loading_model(tmp_path):
    import json

    from gemma_system_one.training.decisions_pipeline import STAGE, run_evaluate_decisions

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "extra": {
                    "stage": STAGE,
                    "primitives": ["noul", "choice", "score"],
                    "trained_primitives": ["noul", "choice"],
                }
            }
        )
    )
    with pytest.raises(ValueError, match="sin entrenar"):
        run_evaluate_decisions(tmp_path, "validation")


def test_probability_overflow_is_rejected():
    from gemma_system_one.inference import InvalidOutputError, group_probabilities

    with pytest.raises(InvalidOutputError, match="finitas"):
        group_probabilities(torch.tensor([1.0, 2.0]), 1e-320)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_type_weights_must_be_finite(value):
    from gemma_system_one.training.decisions import DecisionTrainParams

    with pytest.raises(ValueError, match="finitos"):
        DecisionTrainParams(type_weights={"noul": value, "choice": 1.0, "score": 1.0})


@pytest.mark.parametrize("cases,questions", [(1, 100), (1, 0), (0, 3)])
def test_mixed_generator_rejects_impossible_sizes(cases, questions):
    from gemma_system_one.data.generate_mixed import generate_mixed

    with pytest.raises(ValueError):
        generate_mixed(cases, 0, questions_per_case=questions)
