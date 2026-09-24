"""Regresiones de revisión; dobles sólo para errores HTTP y planificación de trabajo."""

import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import gemma_system_one.api as api
from gemma_system_one.engine import DecisionEngine, DecisionResult


class Engine:
    def warmup(self):
        return {}

    def describe(self):
        return {"checkpoint_id": "test", "calibrated": {"noul": False}}

    def decide(self, state, questions, image):
        return DecisionResult(
            {k: {"type": "noul", "noul": 0.5} for k in questions},
            dict(
                questions=1,
                expanded_rows=1,
                backbone_forwards=1,
                processed_input_tokens=1,
                image_tokens=0,
                padding_tokens=0,
                generated_tokens=0,
            ),
        )


def body():
    return {
        "model": "gemma-system-one-e2b-v0.1",
        "state": "synthetic",
        "questions": {"q": {"type": "noul", "instructions": "Check?"}},
    }


def wait_ready(client):
    for _ in range(100):
        if client.get("/health/ready").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("not ready")


def test_deep_json_returns_422(tmp_path):
    app = api.create_app(api.ServeConfig(kind="serve", checkpoint=tmp_path), Engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        wait_ready(client)
        response = client.post("/v1/decide", content="[" * 2000 + "0" + "]" * 2000)
        assert response.status_code == 422


def test_validation_errors_do_not_echo_values(tmp_path):
    app = api.create_app(api.ServeConfig(kind="serve", checkpoint=tmp_path), Engine)
    with TestClient(app) as client:
        wait_ready(client)
        request = body()
        request["questions"]["q"]["type"] = "PRIVATE_SENTINEL"
        response = client.post("/v1/decide", json=request)
        assert response.status_code == 422
        assert "PRIVATE_SENTINEL" not in response.text


def test_image_decode_runs_in_worker_and_counts_toward_timeout(tmp_path, monkeypatch):
    threads = []

    def decode(_):
        threads.append(threading.current_thread().name)
        time.sleep(0.15)
        return None

    monkeypatch.setattr(api, "_decode_image", decode)
    app = api.create_app(api.ServeConfig(kind="serve", checkpoint=tmp_path, request_timeout_s=0.03), Engine)

    async def scenario():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
        ):
            while (await c.get("/health/ready")).status_code != 200:
                await asyncio.sleep(0.005)
            response = await c.post("/v1/decide", json=body() | {"image": "dummy"})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "timeout"
        assert threads == ["gso-inference"]

    asyncio.run(scenario())


@pytest.mark.parametrize("primitive", ["choice", "score"])
def test_warmup_uses_a_trained_primitive(primitive):
    engine = DecisionEngine.__new__(DecisionEngine)
    engine.modality = "text"
    engine.primitives = {primitive}

    def decide(state, questions, image):
        assert {q.type for q in questions.values()} <= engine.primitives
        return DecisionResult({}, {})

    engine.decide = decide
    engine.warmup()


def test_visual_split_cannot_change_planned_seed(tmp_path):
    from gemma_system_one.data.dataset import load_dataset
    from gemma_system_one.data.generate_vision import write_vision_dataset
    from gemma_system_one.data.split import SplitError, check_planned_split, make_split

    write_vision_dataset(tmp_path, 20, 3, version="v2", split_seed=0)
    ds = load_dataset(tmp_path)
    with pytest.raises(SplitError, match="planificado"):
        check_planned_split(tmp_path, make_split(ds, 1))


def test_benchmark_creates_output_directory_before_starting_server(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gemma_system_one import benchmark

    monkeypatch.setattr(
        benchmark,
        "load_serve_config",
        lambda _: SimpleNamespace(model_id="test", host="127.0.0.1", port=8000),
    )
    monkeypatch.setattr(benchmark, "build_requests", lambda *args: [{}])

    def stop_before_server(*args, **kwargs):
        assert (tmp_path / "new" / "result.server.log").exists()
        raise RuntimeError("server deliberately not started")

    monkeypatch.setattr(benchmark.subprocess, "Popen", stop_before_server)
    with pytest.raises(RuntimeError, match="deliberately not started"):
        benchmark.run_benchmark(tmp_path / "config.yaml", tmp_path, out=tmp_path / "new" / "result.json")
