"""Pendientes de la revisión de fase 3: identidad del código, huella de entrada y durabilidad."""

from __future__ import annotations

import pytest

from gemma_system_one.env import source_fingerprint
from gemma_system_one.metrics import compare_predictions


def test_source_fingerprint_changes_with_any_source_file(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "c.yaml").write_text("a: 1\n")
    first = source_fingerprint(tmp_path)
    assert first["files"] == 2 and source_fingerprint(tmp_path) == first
    (tmp_path / "configs" / "c.yaml").write_text("a: 2\n")
    assert source_fingerprint(tmp_path)["sha256"] != first["sha256"]
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "a.cpython-311.py").write_text("")
    assert source_fingerprint(tmp_path)["files"] == 2  # la caché de bytecode no cuenta


def test_comparison_rejects_same_ids_with_different_inputs():
    a = [{"id": "q", "group_id": "g", "type": "noul", "input_sha256": "a" * 64, "nll": 1.0, "correct": False}]
    with pytest.raises(ValueError, match="input_sha256"):
        compare_predictions(a, [{**a[0], "input_sha256": "b" * 64}], reps=5)
    with pytest.raises(ValueError, match="input_sha256"):
        compare_predictions(a, [{k: v for k, v in a[0].items() if k != "input_sha256"}], reps=5)
    assert compare_predictions(a, [{**a[0], "nll": 0.5}], reps=5)["point"]["b_minus_a.nll_all"] == -0.5
    ctl = compare_predictions(a, [{**a[0], "input_sha256": "b" * 64}], reps=5, allow_different_inputs=True)
    assert ctl["same_inputs"] is False  # control deliberado: se admite y queda registrado
    with pytest.raises(ValueError, match="etiqueta"):
        compare_predictions(a, [{**a[0], "target_index": 1}], reps=5, allow_different_inputs=True)


def test_cli_exposes_every_evaluate_option():
    """Regresión: una opción de evaluate existía en el código pero no en el parser (fase 4)."""
    from gemma_system_one.cli import build_parser

    args = build_parser().parse_args(
        ["evaluate", "--checkpoint", "c", "--split", "test", "--final-test", "--no-cache", "--robustness",
         "--calibration", "cal.json", "--baselines", "--vision-ablation"]
    )  # fmt: skip
    assert args.vision_ablation and args.baselines and args.robustness and args.final_test
    cmp = build_parser().parse_args(["compare", "--a", "x", "--b", "y", "--allow-different-inputs"])
    assert cmp.allow_different_inputs
    gen = build_parser().parse_args(
        ["generate-data", "--kind", "vision", "--out", "d", "--variant", "transfer"]
    )
    assert gen.kind == "vision"


def test_snapshot_archive_reproduces_its_hash_and_checkpoints_record_it(tmp_path):
    """Hallazgo 5 de la revisión de fase 4: el hash registrado debe tener su copia exacta."""
    import io
    import tarfile

    from gemma_system_one.env import _read_sources, snapshot_sources

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text("print(1)\n")
    (root / "pyproject.toml").write_text("[project]\n")
    snap = snapshot_sources(root, tmp_path / "arch")
    archive = tmp_path / "arch" / f"{snap['sha256']}.tar"
    assert archive.is_file() and snapshot_sources(root, tmp_path / "arch") == snap  # idempotente
    extracted = tmp_path / "extracted"
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
        tar.extractall(extracted, filter="data")
    assert _read_sources(extracted)[0]["sha256"] == snap["sha256"]


def test_checkpoint_manifest_source_hash_has_an_archive(tmp_path, monkeypatch):
    import json
    import os

    from gemma_system_one.checkpoint import save_head
    from gemma_system_one.models.heads import NoulHead

    monkeypatch.setenv("GSO_SOURCE_ARCHIVE_DIR", str(tmp_path / "arch"))
    ckpt = save_head(
        tmp_path / "c",
        NoulHead(4),
        repo_id="a/b",
        revision="0" * 40,
        backbone_dtype="float32",
        prompt_template="t",
    )
    src = json.loads((ckpt / "manifest.json").read_text())["git"]["source"]
    assert os.path.isfile(tmp_path / "arch" / f"{src['sha256']}.tar")
