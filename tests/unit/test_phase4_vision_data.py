"""Fase 4: datos visuales verificables, referencias por contenido y serialización con imagen."""

from __future__ import annotations

import io
import json
import re

import pytest

from conftest import make_example
from gemma_system_one.data.dataset import load_dataset, model_input_hash, validate_dataset
from gemma_system_one.data.generate_vision import (
    LEVEL_BOUNDS,
    RUBRIC,
    SERVICES,
    V1_STYLE_COUNT,
    generate_vision,
    level_of,
    write_vision_dataset,
)
from gemma_system_one.data.split import make_split
from gemma_system_one.serialization import expand_example


def _mentioned(text: str, names: list[str]) -> list[str]:
    """Servicios citados en la instrucción, en orden de aparición."""
    found = [(m.start(), n) for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", text)]
    return [n for _, n in sorted(found)]


@pytest.mark.parametrize("variant", ["main", "transfer"])
def test_labels_follow_the_drawn_values(variant):
    raws, images, audit = generate_vision(150, 3, variant)
    facts = {a["group_id"]: a for a in audit}
    assert len(images) == len(audit) == 150
    for r in raws:
        f = facts[r["group_id"]]
        val = dict(zip(f["services"], f["values"], strict=True))
        q, t, kind = r["question"], r["target"], r["task_family"].removeprefix("panel_")
        named = _mentioned(q["instructions"], f["services"])
        if kind == "any_over":
            assert t["label"] == int(max(f["values"]) > f["threshold"])
        elif kind == "service_over":
            assert t["label"] == int(val[named[0]] > f["threshold"])
        elif kind == "compare":
            a, b = named
            assert t["label"] == int(val[a] > val[b]) and abs(val[a] - val[b]) >= 12
        elif kind in ("top", "bottom"):
            pick = max if kind == "top" else min
            assert q["criteria"][t["class_id"]] == pick(f["services"], key=val.get)
            assert sorted(q["criteria"].values()) == sorted(f["services"])
        else:
            level = level_of(val[named[0]])
            asc = q["criteria"] == RUBRIC[r["language"]]
            assert t["level_index"] == (level if asc else len(q["criteria"]) - 1 - level)
            assert all(abs(val[named[0]] - b) >= 5 for b in LEVEL_BOUNDS)


def test_answer_is_only_in_the_image():
    raws, _, audit = generate_vision(80, 1)
    for r in raws:
        assert not re.search(r"\d", r["state"])  # sin valores en el estado
        assert not any(s in r["state"] for s in SERVICES)
    same_text = {}
    for r in raws:
        key = (r["state"], json.dumps(r["question"], sort_keys=True))
        same_text.setdefault(key, set()).add(json.dumps(r["target"]))
    assert any(len(v) > 1 for v in same_text.values())  # mismo texto, distinta respuesta
    assert {len(a["services"]) for a in audit} == {3, 4, 5}


def test_transfer_reserves_style_and_phrasing():
    main, _, main_audit = generate_vision(200, 0, "main")
    tr, _, tr_audit = generate_vision(60, 0, "transfer")
    reserved = V1_STYLE_COUNT - 1  # v1: último de sus 5 estilos (la lista global creció con v2)
    assert {a["style"] for a in tr_audit} == {reserved}
    assert reserved not in {a["style"] for a in main_audit} and max(a["style"] for a in main_audit) < reserved
    main_instr = {r["question"]["instructions"] for r in main}
    assert not any(
        r["question"]["instructions"] in main_instr for r in tr if "{" not in r["question"]["instructions"]
    )


def test_vision_dataset_is_reproducible_across_processes(tmp_path):
    import os
    import subprocess
    import sys

    out = set()
    for seed in ("0", "999"):
        path = tmp_path / seed
        code = (
            "from pathlib import Path\n"
            "from gemma_system_one.data.generate_vision import write_vision_dataset\n"
            f"m = write_vision_dataset(Path({str(path)!r}), 30, 0)\n"
            "print(m['examples_sha256'], m['images_sha256'])"
        )
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        out.add(res.stdout.strip().splitlines()[-1])
    assert len(out) == 1


def test_written_dataset_validates_splits_and_images_decode(tmp_path):
    from PIL import Image

    write_vision_dataset(tmp_path, 60, 2)
    report = validate_dataset(tmp_path)
    assert report.ok, report.errors[:3]
    assert report.stats["distinct_images"] == 60 and report.stats["with_image"] == report.stats["examples"]
    ds = load_dataset(tmp_path)
    for e in ds.examples[:5]:
        with Image.open(tmp_path / e.image_path) as im:
            assert im.size == (480, 360) and im.format == "PNG"
    m = make_split(ds, 0)
    assert m["checks"]["errors"] == [] and m["checks"]["near_duplicate_states"] == []
    groups = {s: {e["id"].rsplit("-", 1)[0] for e in v["examples"]} for s, v in m["splits"].items()}
    assert sum(map(len, groups.values())) >= 60


def test_image_rows_hash_the_image_and_text_rows_are_unchanged():
    text = make_example()
    (row,) = expand_example(text)
    assert row.image is None
    legacy = __import__("hashlib").sha256(f"gso-text-v1\n{row.text}".encode()).hexdigest()
    assert row.sha256 == legacy  # huellas de fases 1–3 intactas
    with_img = make_example(image_path=f"images/{'a' * 64}.png")
    (irow,) = expand_example(with_img)
    other = make_example(image_path=f"images/{'b' * 64}.png")
    assert irow.text == row.text and irow.sha256 != row.sha256
    assert irow.sha256 != expand_example(other)[0].sha256
    assert model_input_hash(text) != model_input_hash(with_img) != model_input_hash(other)
    payload = {"state": text.state, "question": text.question.model_dump(mode="json")}
    import hashlib

    assert (
        model_input_hash(text)
        == hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    )


def test_chat_batch_places_image_first_and_rejects_mixed(fake_processor):
    from PIL import Image

    from gemma_system_one.models.encoding import build_chat_batch, image_token_count

    img = Image.new("RGB", (40, 30), "blue")
    b = build_chat_batch(fake_processor, ["a b", "c"], max_length=64, images=[img, img])
    assert b["pixel_values"].shape == (2, 36, 768) and image_token_count(b) == 8
    with pytest.raises(ValueError, match="mezclar"):
        build_chat_batch(fake_processor, ["a", "b"], max_length=64, images=[img, None])
    with pytest.raises(ValueError, match="una entrada por fila"):
        build_chat_batch(fake_processor, ["a", "b"], max_length=64, images=[img])


def test_image_loader_rejects_tampered_and_oversized(tmp_path):
    import hashlib

    from PIL import Image

    from gemma_system_one.images import ImageError, check_image_file, load_image

    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")
    data = buf.getvalue()
    (tmp_path / "images").mkdir()
    good = tmp_path / "images" / f"{hashlib.sha256(data).hexdigest()}.png"
    good.write_bytes(data)
    assert load_image(good).size == (20, 10)
    bad = tmp_path / "images" / f"{'c' * 64}.png"
    bad.write_bytes(data)
    with pytest.raises(ImageError, match="hash"):
        load_image(bad)
    buf = io.BytesIO()
    Image.new("RGB", (5000, 3300)).save(buf, format="PNG")  # 16,5 MP > 16 MP
    big = buf.getvalue()
    rel = f"images/{hashlib.sha256(big).hexdigest()}.png"
    (tmp_path / rel).write_bytes(big)
    assert any("megap" in e or "píxeles" in e for e in check_image_file(tmp_path, rel))


def test_v2_styles_and_random_streams_are_separated_by_partition(tmp_path):
    """Revisión de fase 4 (hallazgo 4, spec §5.1): estilos disjuntos por partición del split real."""
    from gemma_system_one.data.generate_vision import STYLE_POOLS_V2
    from gemma_system_one.data.split import SPLITS, check_planned_split

    m = write_vision_dataset(tmp_path, 120, 5, version="v2", split_seed=0)
    assert m["generator_version"] == "support-vision-v2" and m["planned_split"]["seed"] == 0
    split = make_split(load_dataset(tmp_path), 0)
    check_planned_split(tmp_path, split)  # el split real coincide con el planificado
    audit = {a["group_id"]: a for a in map(json.loads, (tmp_path / "audit.jsonl").read_text().splitlines())}
    used = {}
    for s in SPLITS:
        styles = {audit[g]["style"] for g in split["splits"][s]["groups"]}
        assert styles <= set(STYLE_POOLS_V2[s]) and all(
            audit[g]["planned_split"] == s for g in split["splits"][s]["groups"]
        )
        used[s] = styles
    assert all(used[a].isdisjoint(used[b]) for a in SPLITS for b in SPLITS if a < b)
    assert all(
        set(STYLE_POOLS_V2[a]).isdisjoint(STYLE_POOLS_V2[b])
        for a in STYLE_POOLS_V2
        for b in STYLE_POOLS_V2
        if a < b
    )
    other = make_split(load_dataset(tmp_path), 1)
    other["seed"] = 0  # otra asignación presentada como la semilla planificada
    with pytest.raises(Exception, match="planificado"):
        check_planned_split(tmp_path, other)


def test_v1_output_is_unchanged_by_v2():
    a, _, _ = generate_vision(40, 0)
    b, _, _ = generate_vision(40, 0, version="v1")
    assert a == b and all(r["group_id"].startswith("vis-") for r in a)
    t, _, audit = generate_vision(20, 0, "transfer", version="v2")
    assert {x["style"] for x in audit} == {4} and all(x["planned_split"] == "transfer" for x in audit)
