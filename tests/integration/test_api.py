"""Fase 5 en CPU: API HTTP con el motor real sobre Gemma 4 diminuto y checkpoints entrenados.

Contrato, invariantes (permutar opciones, pregunta independiente, cambio de pregunta), límites y
códigos de error, modalidad texto/imagen, modelo ausente. La saturación de la cola y el tiempo
agotado usan un motor doble lento (sólo aquí, nunca en el servidor real).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import gemma_system_one.features as features_mod
import gemma_system_one.training.decisions_pipeline as dp
import gemma_system_one.training.pipeline as p1
from gemma_system_one.api import ServeConfig, create_app
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import write_mixed_dataset
from gemma_system_one.data.generate_vision import write_vision_dataset
from gemma_system_one.data.split import make_split, split_path, write_split
from gemma_system_one.engine import DecisionResult

FAKE_PROC_FP = {"files_sha256": {"fake": "0"}, "chat_kwargs": {}, "padding_side": "right"}
MODEL = "gemma-system-one-e2b-v0.1"


def _patch(monkeypatch, snapshot, fake_processor):
    for mod in (dp, p1):
        monkeypatch.setattr(mod, "require_snapshot", lambda cfg: snapshot, raising=False)
        monkeypatch.setattr(mod, "load_processor", lambda snap: fake_processor, raising=False)
        monkeypatch.setattr(mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP, raising=False)
    monkeypatch.setattr(features_mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP)


def _train(tmp_path, ds: Path, name: str) -> Path:
    base = tmp_path / f"{name}_base.yaml"
    model = {"repo_id": "google/gemma-4-E2B-it", "revision": "a" * 40, "dtype": "float32"}
    base.write_text(
        yaml.safe_dump({"name": "tiny", "model": model, "runtime": {"device": "cpu", "max_length": 250}})
    )
    cfg = tmp_path / f"{name}.yaml"
    raw = {"kind": "decision_heads", "name": name, "base_config": str(base), "dataset": str(ds)}
    raw |= {"cache_dir": str(tmp_path / "cache"), "runs_dir": str(tmp_path / "runs")}
    cfg.write_text(yaml.safe_dump(raw | {"train": {"epochs": 2, "lr": 0.01}}))
    return Path(dp.run_train_decisions(cfg)["checkpoint"])


@pytest.fixture
def text_ckpt(tmp_path, tiny_checkpoint, fake_processor, monkeypatch):
    _patch(monkeypatch, tiny_checkpoint, fake_processor)
    ds = tmp_path / "mixed"
    write_mixed_dataset(ds, 24, seed=4)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    return _train(tmp_path, ds, "api_text")


def _client(cfg: ServeConfig, engine_factory=None) -> TestClient:
    return TestClient(create_app(cfg, engine_factory))


def _wait_ready(client: TestClient, timeout: float = 60) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get("/health/ready")
        if r.status_code == 200 or r.json().get("status") == "error":
            return r.json() | {"_code": r.status_code}
        time.sleep(0.05)
    raise AssertionError("no ready")


QUESTIONS = {
    "refund": {"type": "noul", "instructions": "¿Se solicita devolución?"},
    "route": {
        "type": "choice",
        "instructions": "¿Qué equipo debe atender la solicitud?",
        "criteria": {
            "billing": "Cobros y devoluciones",
            "technical": "Errores de funcionamiento",
            "other": "Otro",
        },
    },
    "impact": {
        "type": "score",
        "instructions": "Evalúa el impacto económico según el estado.",
        "criteria": [
            "No se indica problema económico",
            "Se indica un cobro incorrecto",
            "Pérdida que impide operar",
        ],
    },
}
STATE = "Se ha cobrado dos veces el pedido. Solicito devolución."


def _body(**over):
    return {"model": MODEL, "state": STATE, "questions": QUESTIONS, **over}


def test_real_engine_contract_invariants_and_errors(text_ckpt, tmp_path):
    cfg = ServeConfig(kind="serve", checkpoint=text_ckpt, max_body_bytes=64 * 1024)
    with _client(cfg) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = _wait_ready(client)
        assert (
            ready["_code"] == 200
            and ready["modality"] == "text"
            and ready["checkpoint_id"].startswith("decision_heads:")
        )
        r = client.post("/v1/decide", json=_body())
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["model"] == MODEL and set(data["answers"]) == set(QUESTIONS)
        a = data["answers"]
        assert 0 <= a["refund"]["noul"] <= 1
        assert a["route"]["choice"] in QUESTIONS["route"]["criteria"]
        assert abs(sum(a["route"]["probabilities"].values()) - 1) < 1e-6
        assert 0 <= a["impact"]["score"] <= 2 and a["impact"]["legend"] == QUESTIONS["impact"]["criteria"]
        u = data["usage"]
        assert u["questions"] == 3 and u["expanded_rows"] == u["backbone_forwards"] == 1 + 3 + 3
        assert u["generated_tokens"] == 0 and u["processed_input_tokens"] > 0 and u["image_tokens"] == 0
        assert data["metadata"]["confidence_method"] == "normalized_entropy_v1"
        # Identidad del proceso: la misma en ready y en cada respuesta (atribución del benchmark).
        assert (
            data["metadata"]["instance_id"] == ready["instance_id"]
            and data["metadata"]["pid"] == ready["pid"]
        )
        assert len(ready["code_sha256"]) == 64

        # Permutar el mapa y renombrar IDs: probabilidades remapeadas idénticas.
        route = QUESTIONS["route"]
        renamed = {"x" + k: route["criteria"][k] for k in reversed(list(route["criteria"]))}
        r2 = client.post("/v1/decide", json=_body(questions={"route": {**route, "criteria": renamed}}))
        p1_, p2 = a["route"]["probabilities"], r2.json()["answers"]["route"]["probabilities"]
        assert all(p1_[k] == p2["x" + k] for k in p1_)
        # Añadir una pregunta independiente no cambia las demás (una fila por forward).
        extra = {**QUESTIONS, "other": {"type": "noul", "instructions": "¿Menciona un producto?"}}
        r3 = client.post("/v1/decide", json=_body(questions=extra)).json()
        assert all(r3["answers"][k] == a[k] for k in QUESTIONS)
        # Misma pregunta y estado: respuesta idéntica; otra pregunta sobre el mismo estado: otras filas.
        assert client.post("/v1/decide", json=_body()).json()["answers"] == a
        q_other = {"x": {"type": "noul", "instructions": "¿El cliente está enfadado?"}}
        assert client.post("/v1/decide", json=_body(questions=q_other)).json()["answers"]["x"] != a["refund"]

        def err(resp, status, code):
            assert resp.status_code == status, resp.text
            assert resp.json()["error"]["code"] == code and resp.json()["error"]["request_id"]

        err(client.post("/v1/decide", json=_body(model="jev-1.13.0")), 404, "unknown_model")
        err(client.post("/v1/decide", content=b'{"model": "x", "state": NaN}'), 422, "invalid_json")
        err(client.post("/v1/decide", content=b'{"model": "a", "model": "b"}'), 422, "invalid_json")
        err(client.post("/v1/decide", json=_body(extra=1)), 422, "invalid_request")
        err(client.post("/v1/decide", json=_body(state="")), 422, "invalid_request")
        many = {f"q{i}": QUESTIONS["refund"] for i in range(9)}
        err(client.post("/v1/decide", json=_body(questions=many)), 422, "invalid_request")
        one = {"c": {"type": "choice", "instructions": "x", "criteria": {"a": "solo una"}}}
        err(client.post("/v1/decide", json=_body(questions=one)), 422, "invalid_request")
        big = client.post("/v1/decide", content=b"{" + b" " * (70 * 1024) + b"}")
        err(big, 413, "body_too_large")
        long_state = " ".join(f"palabra{i}" for i in range(400))
        err(client.post("/v1/decide", json=_body(state=long_state)), 422, "invalid_request")
        png = io.BytesIO()
        __import__("PIL.Image", fromlist=["Image"]).new("RGB", (8, 8)).save(png, format="PNG")
        err(client.post("/v1/decide", json=_body(image=base64.b64encode(png.getvalue()).decode())), 422,
            "invalid_request")  # fmt: skip
        err(client.post("/v1/decide", json=_body(image="no es base64!")), 400, "invalid_image")
        err(client.post("/v1/decide", json=_body(image=base64.b64encode(b"GIF89a....").decode())), 400,
            "invalid_image")  # fmt: skip
        payload = json.dumps(_body())
        assert '"state"' in payload  # la respuesta de error nunca repite el estado
        bad = client.post("/v1/decide", json=_body(extra="SECRETO")).text
        assert "SECRETO" not in bad and STATE not in bad


def test_image_limits_and_vision_modality(tmp_path, tiny_vision_checkpoint, fake_processor, monkeypatch):
    from PIL import Image

    _patch(monkeypatch, tiny_vision_checkpoint, fake_processor)
    ds = tmp_path / "vision"
    write_vision_dataset(ds, 40, seed=4)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    ckpt = _train(tmp_path, ds, "api_vision")
    with _client(ServeConfig(kind="serve", checkpoint=ckpt)) as client:
        ready = _wait_ready(client)
        assert ready["_code"] == 200 and ready["modality"] == "text+image"
        q = {"top": {"type": "choice", "instructions": "¿Qué servicio muestra más errores?",
                     "criteria": {"a": "API", "b": "Auth", "c": "Sync"}}}  # fmt: skip
        body = {"model": MODEL, "state": "Adjunto el panel.", "questions": q}
        r = client.post("/v1/decide", json=body)
        assert r.status_code == 422 and "imagen" in r.json()["error"]["message"]
        img = next((ds / "images").iterdir()).read_bytes()
        ok = client.post("/v1/decide", json=body | {"image": base64.b64encode(img).decode()})
        assert ok.status_code == 200, ok.text
        assert ok.json()["usage"]["image_tokens"] == 4 * 3 and ok.json()["usage"]["expanded_rows"] == 3
        other = client.post(
            "/v1/decide",
            json=body
            | {"image": base64.b64encode(sorted((ds / "images").iterdir())[-1].read_bytes()).decode()},
        )
        assert other.json()["answers"] != ok.json()["answers"]  # la imagen entra en la decisión
        huge = io.BytesIO()
        Image.new("RGB", (5000, 3300)).save(huge, format="PNG")  # 16,5 MP comprimidos en pocos KB
        r = client.post("/v1/decide", json=body | {"image": base64.b64encode(huge.getvalue()).decode()})
        assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"
        anim = io.BytesIO()
        frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
        frames[0].save(anim, format="PNG", save_all=True, append_images=frames[1:])
        r = client.post("/v1/decide", json=body | {"image": base64.b64encode(anim.getvalue()).decode()})
        assert r.status_code == 400 and "animada" in r.json()["error"]["message"]
        jpg = io.BytesIO()
        Image.new("RGB", (32, 32), "green").save(jpg, format="JPEG")
        trunc = base64.b64encode(jpg.getvalue()[:-40]).decode()
        r = client.post("/v1/decide", json=body | {"image": trunc})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"


def test_missing_model_is_not_ready_and_never_answers(tmp_path):
    cfg = ServeConfig(kind="serve", checkpoint=tmp_path / "no_existe")
    with _client(cfg) as client:
        ready = _wait_ready(client)
        assert ready["_code"] == 503 and ready["status"] == "error" and ready["reason"]
        r = client.post("/v1/decide", json=_body())
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        assert r.headers["X-GSO-Instance-ID"] == ready["instance_id"]


class _SlowEngine:
    """Doble de test: sólo para ejercitar cola y tiempo agotado sin modelo."""

    def __init__(self, delay: float):
        self.delay, self.calls = delay, 0
        self.gate = threading.Event()

    def warmup(self):
        return {"seconds": 0.0}

    def describe(self):
        return {"checkpoint_id": "double", "calibrated": {"noul": False, "choice": False, "score": False}}

    def decide(self, state, questions, image):
        self.calls += 1
        time.sleep(self.delay)
        answers = {k: {"type": "noul", "noul": 0.5} for k in questions}
        n = len(questions)
        usage = dict(questions=n, expanded_rows=n, backbone_forwards=n, processed_input_tokens=1)
        usage |= dict(image_tokens=0, padding_tokens=0, generated_tokens=0)
        return DecisionResult(answers, usage, {})


def test_queue_saturation_and_timeout_with_a_slow_double(tmp_path):
    import httpx

    engine = _SlowEngine(delay=0.6)
    cfg = ServeConfig(kind="serve", checkpoint=tmp_path / "x", max_queue=1, request_timeout_s=5)
    app = create_app(cfg, lambda: engine)
    body = _body(questions={"a": QUESTIONS["refund"]})

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                while (await c.get("/health/ready")).status_code != 200:
                    await asyncio.sleep(0.01)
                results = await asyncio.gather(*[c.post("/v1/decide", json=body) for _ in range(5)])
                return [(r.status_code, r.json().get("error", {}).get("code")) for r in results]

    results = asyncio.run(scenario())
    ok = [r for r in results if r[0] == 200]
    full = [r for r in results if r == (503, "queue_full")]
    assert ok and full and len(ok) + len(full) == 5, results
    # 1 en curso + 1 en cola como máximo simultáneos: nunca más de 2 aceptadas a la vez.
    assert len(ok) <= 2

    slow = _SlowEngine(delay=1.5)
    app2 = create_app(
        ServeConfig(kind="serve", checkpoint=tmp_path / "x", request_timeout_s=0.3), lambda: slow
    )
    with TestClient(app2) as client:
        _wait_ready(client)
        r = client.post("/v1/decide", json=body)
        assert r.status_code == 503 and r.json()["error"]["code"] == "timeout"
