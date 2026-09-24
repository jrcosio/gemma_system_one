from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gemma_system_one.config import ProjectConfig, load_config

BASE = {
    "name": "t",
    "model": {"repo_id": "google/gemma-4-E2B-it", "revision": "a" * 40},
}


def test_repo_config_is_pinned(e2b_cfg: ProjectConfig):
    assert e2b_cfg.model.repo_id == "google/gemma-4-E2B-it"
    assert len(e2b_cfg.model.revision) == 40
    assert e2b_cfg.model.backbone_class == "Gemma4Model"
    assert e2b_cfg.runtime.device == "mps"
    assert e2b_cfg.runtime.allow_cpu_fallback is False
    assert e2b_cfg.runtime.max_length == 512


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "g" * 40])
def test_revision_must_be_full_commit_sha(revision: str):
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({**BASE, "model": {**BASE["model"], "revision": revision}})


@pytest.mark.parametrize(
    "patch",
    [
        {"model": {**BASE["model"], "dtype": "int8"}},
        {"model": {**BASE["model"], "repo_id": "gemma"}},
        {"model": {**BASE["model"], "unknown": 1}},
        {"runtime": {"device": "cuda"}},
        {"runtime": {"memory_budget_gib": 64}},
        {"extra_top_level": True},
    ],
)
def test_invalid_or_unknown_fields_rejected(patch: dict):
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({**BASE, **patch})


def test_load_config_rejects_non_mapping(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(["x"]))
    with pytest.raises(ValueError):
        load_config(p)


def test_config_is_immutable(e2b_cfg: ProjectConfig):
    with pytest.raises(ValidationError):
        e2b_cfg.model.revision = "b" * 40  # type: ignore[misc]
