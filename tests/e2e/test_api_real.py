"""E2E de la API con E2B real en MPS y el checkpoint entrenado de fase 3 (skip con motivo si faltan).

Consulta real, identidad train/serve (logits de la API = los de ``gso evaluate`` para las mismas
preguntas), permutación de opciones, pregunta independiente, cambio de pregunta sobre el mismo
estado, modelo desconocido y usage real. No hay dobles: el servidor usa el motor real.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from gemma_system_one.api import create_app, load_serve_config
from gemma_system_one.calibration import load_calibration
from gemma_system_one.inference import reconstruct
from gemma_system_one.serialization import expand

SERVE = REPO_ROOT / "configs" / "serve_text.yaml"
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
    cfg = cfg.model_copy(update={"checkpoint": ckpt, "calibration": REPO_ROOT / cfg.calibration})
    with TestClient(create_app(cfg)) as client:
        t0 = time.time()
        while time.time() - t0 < 600:
            r = client.get("/health/ready")
            if r.status_code == 200 or r.json().get("status") == "error":
                break
            time.sleep(0.5)
        assert r.status_code == 200, r.json()
        yield cfg, client, r.json()


def test_ready_reports_real_device_and_calibration(served):
    _, _, ready = served
    assert ready["device"].startswith("mps") and ready["modality"] == "text"
    assert ready["checkpoint_id"].startswith("lora_decision_heads:") and "+cal:" in ready["checkpoint_id"]
    assert all(ready["calibrated"].values()) and ready["warmup"]["usage"]["generated_tokens"] == 0
    import os

    assert ready["pid"] == os.getpid() and len(ready["instance_id"]) == 32  # servidor en este proceso


def test_api_logits_match_evaluation_for_the_same_questions(served):
    """Serialización idéntica en entrenamiento y servicio (spec §12): mismas filas → mismos logits."""
    from gemma_system_one.data.dataset import load_dataset

    cfg, client, _ = served
    ckpt = Path(cfg.checkpoint)
    temps = load_calibration(cfg.calibration, ckpt)["temperatures"]
    extra = json.loads((ckpt / "manifest.json").read_text())["extra"]
    ds = {e.id: e for e in load_dataset(REPO_ROOT / extra["dataset_root"]).examples}
    saved = [json.loads(x) for x in (ckpt.parent / "predictions_validation.jsonl").read_text().splitlines()]
    by_group: dict[str, list] = {}
    for p in saved:
        by_group.setdefault(p["group_id"], []).append(p)
    for group in sorted(by_group)[:4]:
        preds = by_group[group]
        e0 = ds[preds[0]["id"]]
        questions = {p["id"]: ds[p["id"]].question.model_dump(mode="json") for p in preds}
        r = client.post("/v1/decide", json={"model": cfg.model_id, "state": e0.state, "questions": questions})
        assert r.status_code == 200, r.text
        assert r.headers["X-GSO-Instance-ID"] == r.json()["metadata"]["instance_id"]
        for p in preds:
            e = ds[p["id"]]
            expected = reconstruct(e.question, expand(e.state, e.question), torch.tensor(p["row_logits"]),
                                   temperature=temps[e.question.type])  # fmt: skip
            assert r.json()["answers"][p["id"]] == expected, p["id"]


def test_real_invariants_usage_and_errors(served):
    cfg, client, _ = served
    q = {
        "refund": {"type": "noul", "instructions": "¿Se solicita devolución?"},
        "route": {"type": "choice", "instructions": "¿Qué equipo debe atender la solicitud?",
                  "criteria": {"billing": "Cobros y devoluciones", "technical": "Errores de funcionamiento"}},
        "impact": {"type": "score", "instructions": "Evalúa el impacto económico según el estado.",
                   "criteria": ["No se indica problema económico", "Se indica un cobro incorrecto",
                                "Se indica pérdida económica que impide operar"]},
    }  # fmt: skip
    body = {
        "model": cfg.model_id,
        "state": "Se ha cobrado dos veces el pedido. Solicito devolución.",
        "questions": q,
    }
    r = client.post("/v1/decide", json=body)
    assert r.status_code == 200, r.text
    a, u = r.json()["answers"], r.json()["usage"]
    assert u["expanded_rows"] == u["backbone_forwards"] == 1 + 2 + 3 and u["generated_tokens"] == 0
    assert u["processed_input_tokens"] > 100 and u["image_tokens"] == 0
    renamed = {"route": {**q["route"], "criteria": {"t": q["route"]["criteria"]["technical"],
                                                    "b": q["route"]["criteria"]["billing"]}}}  # fmt: skip
    p2 = client.post("/v1/decide", json=body | {"questions": renamed}).json()["answers"]["route"][
        "probabilities"
    ]
    assert p2 == {"t": a["route"]["probabilities"]["technical"], "b": a["route"]["probabilities"]["billing"]}
    extra = q | {"lang": {"type": "noul", "instructions": "¿El mensaje está en inglés?"}}
    with_extra = client.post("/v1/decide", json=body | {"questions": extra}).json()["answers"]
    assert {k: v for k, v in with_extra.items() if k in q} == a
    other = {"x": {"type": "noul", "instructions": "¿Menciona el cliente un cobro duplicado?"}}
    assert client.post("/v1/decide", json=body | {"questions": other}).json()["answers"]["x"] != a["refund"]
    assert client.post("/v1/decide", json=body | {"model": "jev-1.13.0"}).status_code == 404
    image = next((REPO_ROOT / "data/vision_smoke_v1/images").iterdir()).read_bytes()
    png = __import__("base64").b64encode(image).decode()
    r = client.post("/v1/decide", json=body | {"image": png})
    assert r.status_code == 422 and "texto" in r.json()["error"]["message"]
