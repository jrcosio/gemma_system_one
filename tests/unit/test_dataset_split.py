from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_example
from gemma_system_one.data.dataset import DatasetError, load_dataset, validate_dataset
from gemma_system_one.data.generate import write_dataset
from gemma_system_one.data.split import (
    SPLITS,
    SplitError,
    assign_groups,
    leakage_checks,
    load_split,
    make_split,
    split_path,
    write_split,
)


def _write(root: Path, examples) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False) for e in examples]
    (root / "examples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_validate_reports_line_numbers_and_nan(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    good = json.dumps(make_example().model_dump(mode="json"))
    (root / "examples.jsonl").write_text(good + "\n" + good.replace('"label": 1', '"label": NaN') + "\n")
    report = validate_dataset(root)
    assert not report.ok
    assert any(e.startswith("línea 2") for e in report.errors)


def test_duplicate_ids_and_contradictory_labels(tmp_path):
    a = make_example(id="a")
    b = make_example(id="a", group_id="g2", state="otro")
    c = make_example(id="c", target={"label": 0})  # misma entrada que a, etiqueta distinta
    report = validate_dataset(_write(tmp_path / "ds", [a, b, c]))
    assert any("id duplicado" in e for e in report.errors)
    assert any("etiquetas distintas" in e for e in report.errors)


def test_same_input_in_different_groups_is_an_error(tmp_path):
    a = make_example(id="a", group_id="g1")
    b = make_example(id="b", group_id="g2")
    report = validate_dataset(_write(tmp_path / "ds", [a, b]))
    assert any("grupos distintos" in e for e in report.errors)


def test_image_paths_must_exist_inside_root(tmp_path):
    """Fase 4 (decisión 0007): images/<sha256>.png real, decodificable y dentro de la raíz."""
    import hashlib
    import io

    from PIL import Image

    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(buf, format="PNG")
    png = buf.getvalue()
    ok = f"images/{hashlib.sha256(png).hexdigest()}.png"
    (root / ok).write_bytes(png)
    outside = tmp_path / "secret.png"
    outside.write_bytes(png)
    link = f"images/{'1' * 64}.png"
    (root / link).symlink_to(outside)
    bad = b"no es una imagen"
    corrupt = f"images/{hashlib.sha256(bad).hexdigest()}.png"
    (root / corrupt).write_bytes(bad)
    wrong = f"images/{'2' * 64}.png"
    (root / wrong).write_bytes(png)
    exs = [
        make_example(id="a", state="s1", image_path=ok),
        make_example(id="b", state="s2", image_path=f"images/{'0' * 64}.png"),
        make_example(id="c", state="s3", image_path=link),
        make_example(id="d", state="s4", image_path="images/clase_positiva.png"),
        make_example(id="e", state="s5", image_path=corrupt),
        make_example(id="f", state="s6", image_path=wrong),
    ]
    report = validate_dataset(_write(root, exs))
    assert any(e.startswith("b:") and "no existe" in e for e in report.errors)
    assert any(e.startswith("c:") and "sale de la raíz" in e for e in report.errors)
    assert any(e.startswith("d:") and "dirección por contenido" in e for e in report.errors)
    assert any(e.startswith("e:") and "no decodificable" in e for e in report.errors)
    assert any(e.startswith("f:") and "no coincide" in e for e in report.errors)
    assert not any(e.startswith("a:") for e in report.errors)


def test_manifest_hash_mismatch_detected(tmp_path):
    root = tmp_path / "ds"
    write_dataset(root, 8, seed=1)
    with (root / "examples.jsonl").open("a") as fh:
        fh.write(json.dumps(make_example(id="extra", state="nuevo").model_dump(mode="json")) + "\n")
    assert any("no coincide" in e for e in validate_dataset(root).errors)
    with pytest.raises(DatasetError):
        load_dataset(root)


def test_group_assignment_is_deterministic_disjoint_and_complete():
    groups = [f"g{i}" for i in range(40)]
    a, b = assign_groups(groups, 0), assign_groups(list(reversed(groups)), 0)
    assert a == b
    assert set(a) == set(groups)
    counts = {s: sum(v == s for v in a.values()) for s in SPLITS}
    assert counts == {"train": 28, "validation": 4, "calibration": 4, "test": 4}
    assert assign_groups(groups, 1) != a
    with pytest.raises(SplitError):
        assign_groups(["g1", "g2", "g3"], 0)


@pytest.fixture
def smoke(tmp_path):
    root = tmp_path / "smoke"
    write_dataset(root, 40, seed=0)
    return load_dataset(root)


def test_split_keeps_groups_together_and_has_no_leaks(smoke):
    manifest = make_split(smoke, seed=0)
    assert manifest["checks"]["errors"] == []
    owner = {}
    for s in SPLITS:
        for g in manifest["splits"][s]["groups"]:
            assert owner.setdefault(g, s) == s
    ids = [x["id"] for s in SPLITS for x in manifest["splits"][s]["examples"]]
    assert sorted(ids) == sorted(e.id for e in smoke.examples)


def test_split_manifest_is_immutable_and_idempotent(smoke, tmp_path):
    path = split_path(smoke.root, 0)
    m = make_split(smoke, 0)
    assert write_split(m, path) is True
    assert write_split(make_split(smoke, 0), path) is False  # idéntico salvo fecha
    with pytest.raises(SplitError, match="no se sobrescriben"):
        write_split(make_split(smoke, 1), path)


def test_load_split_guards_test_and_modifications(smoke):
    path = split_path(smoke.root, 0)
    write_split(make_split(smoke, 0), path)
    with pytest.raises(SplitError, match="test"):
        load_split(smoke, path, ("train", "test"))
    parts = load_split(smoke, path, ("train", "validation"))
    assert {e.group_id for e in parts["train"]}.isdisjoint({e.group_id for e in parts["validation"]})
    other = type(smoke)(root=smoke.root, examples=smoke.examples, sha256="0" * 64)
    with pytest.raises(SplitError, match="otro contenido"):
        load_split(other, path, ("train",))


def test_leakage_checks_flag_shared_inputs_and_near_duplicates():
    a = make_example(id="a", group_id="g1", state="el mismo texto largo de un ticket repetido aquí")
    b = make_example(id="b", group_id="g2", state="el mismo texto largo de un ticket repetido aquí")
    c = make_example(id="c", group_id="g3", state="el mismo texto largo de un ticket repetido aquí hoy")
    parts = {"train": [a], "validation": [b], "calibration": [c], "test": []}
    checks = leakage_checks(parts)
    assert any("entradas idénticas" in e for e in checks["errors"])
    assert any(n["splits"] == ["train", "calibration"] for n in checks["near_duplicate_states"])
