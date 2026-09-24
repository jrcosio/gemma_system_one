import json

import pytest

from conftest import make_example
from gemma_system_one.contracts import parse_json_strict
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate import write_dataset
from gemma_system_one.data.split import SplitError, load_split, make_split
from gemma_system_one.features import backbone_fingerprint
from gemma_system_one.training.pipeline import _rows


def test_split_loader_rejects_group_crossing(tmp_path):
    write_dataset(tmp_path, 12, seed=11)
    ds = load_dataset(tmp_path)
    manifest = make_split(ds, 0)
    # Move one question, keeping its hash and the dataset hash unchanged.
    row = manifest["splits"]["train"]["examples"].pop()
    manifest["splits"]["validation"]["examples"].append(row)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SplitError):
        load_split(ds, path, ("train", "validation"))


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicad"):
        parse_json_strict('{"target":{"label":0,"label":1}}')


def test_text_pipeline_rejects_images():
    with pytest.raises(ValueError, match="imagen"):
        _rows([make_example(image_path="image.png")])


def test_cache_fingerprint_uses_resolved_device(cpu_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("gemma_system_one.features.processor_fingerprint", lambda _: {})
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    cfg = cpu_cfg.model_copy(
        update={"runtime": cpu_cfg.runtime.model_copy(update={"device": "mps", "allow_cpu_fallback": True})}
    )
    assert backbone_fingerprint(cfg, tmp_path, 1)["device"] == "cpu"


def test_cache_fingerprint_changes_with_context_limit(cpu_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("gemma_system_one.features.processor_fingerprint", lambda _: {})
    smaller = cpu_cfg.model_copy(update={"runtime": cpu_cfg.runtime.model_copy(update={"max_length": 16})})
    assert backbone_fingerprint(cpu_cfg, tmp_path, 1) != backbone_fingerprint(smaller, tmp_path, 1)


@pytest.mark.parametrize("mutation", ["duplicate", "omit", "group", "format"])
def test_split_loader_validates_whole_partition_manifest(tmp_path, mutation):
    write_dataset(tmp_path, 12, seed=11)
    ds = load_dataset(tmp_path)
    manifest = make_split(ds, 0)
    held = manifest["splits"]["test"]
    if mutation == "duplicate":
        held["examples"].append(held["examples"][0])
    elif mutation == "omit":
        held["examples"].pop()
    elif mutation == "group":
        row = manifest["splits"]["train"]["examples"].pop()
        manifest["splits"]["validation"]["examples"].append(row)
        source = ds.by_id()[row["id"]]
        manifest["splits"]["validation"]["groups"].append(source.group_id)
    else:
        manifest["format"] = 999
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SplitError):
        load_split(ds, path, ("train",))


def test_mps_fingerprint_rejects_silent_operator_fallback(cpu_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("gemma_system_one.features.processor_fingerprint", lambda _: {})
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cfg = cpu_cfg.model_copy(update={"runtime": cpu_cfg.runtime.model_copy(update={"device": "mps"})})
    with pytest.raises(RuntimeError, match="FALLBACK"):
        backbone_fingerprint(cfg, tmp_path, 1)


def test_f1_is_zero_when_both_classes_are_completely_misclassified():
    from gemma_system_one.metrics import binary_metrics

    assert binary_metrics([10.0, -10.0], [0, 1])["f1"] == 0.0
