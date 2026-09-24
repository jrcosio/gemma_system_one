import pytest
import torch

from gemma_system_one.models.backbone import last_valid_index, move_inputs, pool_last_valid


def test_right_padding():
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert last_valid_index(mask).tolist() == [2, 4]


def test_left_padding():
    # sum(mask)-1 daría [2, 4] aquí, que es incorrecto para la primera fila.
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert last_valid_index(mask).tolist() == [4, 4]


def test_mixed_and_interior_gaps():
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]])
    assert last_valid_index(mask).tolist() == [2, 1]


def test_empty_row_rejected():
    with pytest.raises(ValueError, match="sin tokens"):
        last_valid_index(torch.tensor([[1, 1], [0, 0]]))


def test_mask_must_be_2d():
    with pytest.raises(ValueError):
        last_valid_index(torch.ones(2, 3, 1))


def test_pool_selects_last_valid_vector_not_padding():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(pool_last_valid(h, right), torch.stack([h[0, 1], h[1, 3]]))
    assert torch.equal(pool_last_valid(h, left), torch.stack([h[0, 3], h[1, 3]]))
    # [:, -1] leería padding en la primera fila con padding derecho.
    assert not torch.equal(pool_last_valid(h, right), h[:, -1])


def test_pool_shape_mismatch():
    with pytest.raises(ValueError, match="no alinean"):
        pool_last_valid(torch.zeros(2, 4, 3), torch.ones(2, 5))


def test_move_inputs_keeps_integers_and_casts_floats():
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "mm_token_type_ids": torch.tensor([[0, 0]]),
        "pixel_values": torch.zeros(1, 3, dtype=torch.float32),
        "meta": "no-tensor",
    }
    out = move_inputs(batch, torch.device("cpu"), torch.bfloat16)
    assert out["input_ids"].dtype == torch.int64
    assert out["mm_token_type_ids"].dtype == torch.int64
    assert out["pixel_values"].dtype == torch.bfloat16
    assert out["meta"] == "no-tensor"
    assert set(out) == set(batch)
