"""Hallazgo 6 de la revisión de fase 5: atribución de respuestas al servidor lanzado y
contabilización completa de peticiones (también las que fallan en el transporte).

Los transportes simulados de httpx sólo existen aquí; no sustituyen al benchmark real."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import httpx
import pytest

from gemma_system_one import benchmark

INSTANCE = "a" * 32


def _ok(instance: str = INSTANCE, pid: int = 1) -> httpx.Response:
    body = {
        "metadata": {"instance_id": instance, "pid": pid, "timing_ms": {"server_total": 1.0}},
        "usage": {"questions": 1, "expanded_rows": 2, "processed_input_tokens": 3},
    }
    return httpx.Response(200, json=body)


def test_busy_port_is_rejected_before_launching_the_server(tmp_path, monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert not benchmark.port_is_free("127.0.0.1", port)
        cfg = SimpleNamespace(model_id="m", host="127.0.0.1", port=port, request_timeout_s=1, max_queue=1)
        monkeypatch.setattr(benchmark, "load_serve_config", lambda _: cfg)
        monkeypatch.setattr(benchmark, "build_requests", lambda *a: [{}])

        def never(*a, **k):
            raise AssertionError("no debe lanzarse el servidor si el puerto está ocupado")

        monkeypatch.setattr(benchmark.subprocess, "Popen", never)
        with pytest.raises(RuntimeError, match="ya está en uso"):
            benchmark.run_benchmark(tmp_path / "c.yaml", tmp_path)
    assert benchmark.port_is_free("127.0.0.1", port)


def test_identity_must_match_instance_and_pid():
    benchmark.check_identity({"instance_id": INSTANCE, "pid": 7}, INSTANCE, 7)
    with pytest.raises(benchmark.IdentityError):
        benchmark.check_identity({"instance_id": "otro", "pid": 7}, INSTANCE, 7)
    with pytest.raises(benchmark.IdentityError):
        benchmark.check_identity({"instance_id": INSTANCE, "pid": 8}, INSTANCE, 7)
    with pytest.raises(benchmark.IdentityError):
        benchmark.check_identity({}, INSTANCE)  # servidor sin identidad (p. ej. versión anterior)


def test_sequential_measurement_accounts_for_every_request():
    responses = [
        _ok(),
        httpx.ConnectError("caído"),
        httpx.Response(503, json={"error": {"code": "queue_full"}}, headers={"X-GSO-Instance-ID": INSTANCE}),
        httpx.Response(502, text="gateway", headers={"X-GSO-Instance-ID": INSTANCE}),
        _ok(),
    ]
    calls = iter(responses)

    def handler(request):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        m = benchmark.measure(httpx, client, "http://t", [{}] * 5, INSTANCE)
    assert m["statuses"] == {
        "200": 2,
        "transport_error:ConnectError": 1,
        "503:queue_full": 1,
        "502:invalid_body": 1,
    }
    assert len(m["usage_rows"]) == 2 and len(m["http_ms"]) == 4


def test_sequential_measurement_rejects_a_foreign_server():
    transport = httpx.MockTransport(lambda r: _ok(instance="ajeno"))
    with httpx.Client(transport=transport) as client, pytest.raises(benchmark.IdentityError):
        benchmark.measure(httpx, client, "http://t", [{}], INSTANCE)


def test_burst_records_transport_errors_and_foreign_answers():
    import itertools
    import threading

    lock = threading.Lock()
    seq = itertools.count()

    def handler(request):
        with lock:
            i = next(seq)
        if i % 3 == 0:
            raise httpx.ReadTimeout("sin respuesta")
        if i % 3 == 1:
            return httpx.Response(
                503, json={"error": {"code": "queue_full"}}, headers={"X-GSO-Instance-ID": INSTANCE}
            )
        return _ok(instance="ajeno" if i == 2 else INSTANCE)

    res = benchmark.run_burst(httpx, "http://t", {}, 9, 1.0, INSTANCE, transport=httpx.MockTransport(handler))
    assert sum(res["results"].values()) == 9
    assert res["results"]["transport_error:ReadTimeout"] == 3 and res["results"]["503:queue_full"] == 3
    assert res["results"]["200:identity_mismatch"] == 1 and res["valid"] is False


def test_sequential_rejects_unattributed_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(503, json={"error": {"code": "queue_full"}}))
    with httpx.Client(transport=transport) as client, pytest.raises(benchmark.IdentityError):
        benchmark.measure(httpx, client, "http://t", [{}], INSTANCE)


def test_burst_rejects_unattributed_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(503, json={"error": {"code": "queue_full"}}))
    result = benchmark.run_burst(httpx, "http://t", {}, 2, 1, INSTANCE, transport=transport)
    assert result["valid"] is False


@pytest.mark.parametrize(("requests", "warmup"), [(0, 0), (-1, 5), (1, -1)])
def test_invalid_counts_fail_before_loading_or_launching(tmp_path, monkeypatch, requests, warmup):
    def never(*args):
        raise AssertionError("must reject counts before loading configuration")

    monkeypatch.setattr(benchmark, "load_serve_config", never)
    with pytest.raises(ValueError, match="requests"):
        benchmark.run_benchmark(tmp_path / "missing.yaml", tmp_path, requests=requests, warmup=warmup)
