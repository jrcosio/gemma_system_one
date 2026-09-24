"""E2E de la API con imagen: E2B real en MPS y el checkpoint V2 (cierre de fase 4).

Imagen obligatoria, identidad train/serve con imagen (logits de la API = los de ``gso evaluate``
para las mismas preguntas e imágenes de validación) y usage con tokens visuales reales.
"""

from __future__ import annotations

import base64
import gc
import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from gemma_system_one.api import create_app, load_serve_config
from gemma_system_one.inference import reconstruct
from gemma_system_one.serialization import expand

SERVE = REPO_ROOT / "configs" / "serve_vision.yaml"
pytestmark = [
    pytest.mark.mps,
    pytest.mark.weights,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible"),
]


@pytest.fixture(scope="module")
def served():
    cfg = load_serve_config(SERVE)
    ckpt = REPO_ROOT / cfg.checkpoint
    if not (ckpt / "manifest.json").is_file():
        pytest.skip(f"Falta el checkpoint entrenado {cfg.checkpoint}")
    cfg = cfg.model_copy(update={"checkpoint": ckpt})
    with TestClient(create_app(cfg)) as client:
        t0 = time.time()
        while time.time() - t0 < 600:
            r = client.get("/health/ready")
            if r.status_code == 200 or r.json().get("status") == "error":
                break
            time.sleep(0.5)
        assert r.status_code == 200, r.json()
        yield cfg, client, r.json()
    gc.collect()
    torch.mps.empty_cache()


def test_vision_api_matches_evaluation_and_requires_image(served):
    from gemma_system_one.data.dataset import load_dataset

    cfg, client, ready = served
    assert ready["modality"] == "text+image" and ready["device"].startswith("mps")
    ckpt = Path(cfg.checkpoint)
    extra = json.loads((ckpt / "manifest.json").read_text())["extra"]
    ds = load_dataset(REPO_ROOT / extra["dataset_root"])
    by_id = {e.id: e for e in ds.examples}
    saved = [json.loads(x) for x in (ckpt.parent / "predictions_validation.jsonl").read_text().splitlines()]
    groups: dict[str, list] = {}
    for p in saved:
        groups.setdefault(p["group_id"], []).append(p)
    for group in sorted(groups)[:3]:
        preds = groups[group]
        e0 = by_id[preds[0]["id"]]
        questions = {p["id"]: by_id[p["id"]].question.model_dump(mode="json") for p in preds}
        body = {"model": cfg.model_id, "state": e0.state, "questions": questions}
        r = client.post("/v1/decide", json=body)
        assert r.status_code == 422 and "imagen" in r.json()["error"]["message"]
        image = base64.b64encode((ds.root / e0.image_path).read_bytes()).decode()
        r = client.post("/v1/decide", json=body | {"image": image})
        assert r.status_code == 200, r.text
        usage = r.json()["usage"]
        assert usage["image_tokens"] == 266 * usage["expanded_rows"] and usage["generated_tokens"] == 0
        for p in preds:
            e = by_id[p["id"]]
            expected = reconstruct(
                e.question, expand(e.state, e.question, e.image_path), torch.tensor(p["row_logits"])
            )
            assert r.json()["answers"][p["id"]] == expected, p["id"]
