"""``gso benchmark``: servidor real en otro proceso, arranque en frío, warmup y consultas medidas.

Spec §8: modelo ya cargado, calentamiento explícito, ≥100 consultas medidas si es viable y misma
batería para variantes; arranque en frío aparte. Se registran preguntas, filas, tokens e imágenes.
La latencia HTTP (cliente) incluye cola, procesador y serialización; el forward sincronizado del
servidor se informa por separado. La batería sale de una partición **no reservada** (validation o
calibration): cada petición es un caso del dataset con todas sus preguntas. Test se rechaza.

La memoria del servidor se muestrea desde fuera (RSS con psutil tras cada petición); la memoria
MPS del driver no es visible desde otro proceso y no se afirma.
"""

from __future__ import annotations

import base64
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api import load_serve_config
from .data.dataset import load_dataset
from .data.split import load_split, split_path

READY_TIMEOUT_S = 900


def build_requests(dataset: Path, split: str, model_id: str, limit: int, split_seed: int = 0) -> list[dict]:
    if split not in ("validation", "calibration"):
        raise ValueError("La batería usa validation o calibration; test queda reservado")
    ds = load_dataset(dataset)
    examples = load_split(ds, split_path(dataset, split_seed), (split,))[split]
    by_group: dict[str, list] = defaultdict(list)
    for e in examples:
        by_group[e.group_id].append(e)
    requests = []
    for group in sorted(by_group):
        exs = by_group[group]
        body: dict[str, Any] = {
            "model": model_id,
            "state": exs[0].state,
            "questions": {f"q{i}": e.question.model_dump(mode="json") for i, e in enumerate(exs)},
        }
        if exs[0].image_path is not None:
            body["image"] = base64.b64encode((ds.root / exs[0].image_path).read_bytes()).decode()
        requests.append(body)
    if not requests:
        raise ValueError("Partición vacía")
    # Si hacen falta más consultas que casos, la batería se repite en el mismo orden.
    return [requests[i % len(requests)] for i in range(limit)]


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 2),
        "p50": round(_pct(values, 0.5), 2),
        "p95": round(_pct(values, 0.95), 2),
        "max": round(max(values), 2),
    }


class IdentityError(RuntimeError):
    """Respuesta de un proceso distinto del servidor lanzado por el benchmark."""


def port_is_free(host: str, port: int) -> bool:
    """True si nadie escucha en host:port (se intenta enlazar; no hay reutilización de dirección)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def check_identity(payload: dict[str, Any], instance_id: str, pid: int | None = None) -> None:
    """Exige que la respuesta venga del servidor lanzado (identificador aleatorio y PID)."""
    got_id, got_pid = payload.get("instance_id"), payload.get("pid")
    if got_id != instance_id or (pid is not None and got_pid != pid):
        raise IdentityError(f"Responde otro servidor (instance_id={got_id!r}, pid={got_pid!r})")


def run_benchmark(
    serve_config: Path,
    dataset: Path,
    *,
    split: str = "validation",
    requests: int = 100,
    warmup: int = 5,
    out: Path | None = None,
) -> dict[str, Any]:
    import uuid

    import httpx
    import psutil

    from .api import SERVER_INSTANCE_ENV
    from .env import snapshot_sources

    if requests < 1 or warmup < 0:
        raise ValueError("requests debe ser positivo y warmup no negativo")
    cfg = load_serve_config(serve_config)
    battery = build_requests(dataset, split, cfg.model_id, requests + warmup)
    base = f"http://{cfg.host}:{cfg.port}"
    if not port_is_free(cfg.host, cfg.port):
        # Sin esto, un servidor previo en el puerto contestaría en lugar del proceso lanzado.
        raise RuntimeError(f"{cfg.host}:{cfg.port} ya está en uso: detén ese servicio antes del benchmark")
    code = {"source": snapshot_sources()}  # sin subprocesos: hash + copia de las fuentes del cliente
    instance_id = uuid.uuid4().hex
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".server.log") if out else Path(os.devnull)
    t_spawn = time.perf_counter()
    with open(log_path, "w") as log_fh:  # noqa: PTH123
        proc = subprocess.Popen(
            [sys.executable, "-m", "gemma_system_one.cli", "serve", "--config", str(serve_config)],
            stdout=log_fh, stderr=subprocess.STDOUT, env={**os.environ, SERVER_INSTANCE_ENV: instance_id},
        )  # fmt: skip
        try:
            report = _drive(httpx, psutil, proc, base, cfg, battery, warmup, t_spawn, instance_id)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
    report.update(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "serve_config": str(serve_config),
            "dataset": str(dataset),
            "split": split,
            "server_exit_code": proc.returncode,
            "client_code": code,
            "same_code_as_server": code["source"]["sha256"] == report["server_identity"]["code_sha256"],
        }
    )
    if out is not None:
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _wait_ready(httpx, client, proc, base: str, t_spawn: float, instance_id: str) -> dict[str, Any]:
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"El servidor terminó antes de estar listo (código {proc.returncode})")
        if time.perf_counter() - t_spawn > READY_TIMEOUT_S:
            raise RuntimeError("El servidor no estuvo listo a tiempo")
        try:
            r = client.get(f"{base}/health/ready")
        except httpx.TransportError:
            time.sleep(0.5)
            continue
        payload = r.json()
        check_identity(payload, instance_id, proc.pid)  # también en 503: nunca un servidor ajeno
        if r.status_code == 200:
            return payload
        if payload.get("status") == "error":
            raise RuntimeError(f"El servidor no puede cargar el modelo: {payload.get('reason')}")
        time.sleep(0.5)


def measure(
    httpx, client, base: str, bodies: list[dict], instance_id: str, on_response=None
) -> dict[str, Any]:
    """Peticiones secuenciales; toda petición queda contabilizada (200, error HTTP o de transporte).

    Una respuesta de otra instancia invalida la medición (``IdentityError``)."""
    http_ms, server, usage_rows, statuses = [], defaultdict(list), [], defaultdict(int)
    t_run = time.perf_counter()
    for body in bodies:
        t0 = time.perf_counter()
        try:
            r = client.post(f"{base}/v1/decide", json=body)
        except httpx.TransportError as exc:
            statuses[f"transport_error:{type(exc).__name__}"] += 1
            continue
        http_ms.append((time.perf_counter() - t0) * 1000)
        if r.status_code == 200:
            data = r.json()
            check_identity(data["metadata"], instance_id)
            statuses["200"] += 1
            for k, v in data["metadata"]["timing_ms"].items():
                server[k].append(v)
            usage_rows.append(data["usage"])
        else:
            check_identity({"instance_id": r.headers.get("X-GSO-Instance-ID")}, instance_id)
            code = _error_code(r)
            statuses[f"{r.status_code}:{code}" if code else str(r.status_code)] += 1
        if on_response is not None:
            on_response()
    wall = time.perf_counter() - t_run
    if sum(statuses.values()) != len(bodies):
        raise RuntimeError("Peticiones sin contabilizar en la medición")
    return {"http_ms": http_ms, "server": server, "usage_rows": usage_rows, "statuses": dict(statuses),
            "wall": wall}  # fmt: skip


def _error_code(r) -> str | None:
    try:
        return r.json().get("error", {}).get("code")
    except ValueError:
        return "invalid_body"


def _drive(httpx, psutil, proc, base, cfg, battery, warmup, t_spawn, instance_id) -> dict[str, Any]:
    ps = psutil.Process(proc.pid)
    rss = [0]

    def sample():
        rss[0] = max(rss[0], ps.memory_info().rss)

    with httpx.Client(timeout=cfg.request_timeout_s + 30) as client:
        ready = _wait_ready(httpx, client, proc, base, t_spawn, instance_id)
        cold_start = time.perf_counter() - t_spawn
        warm = measure(httpx, client, base, battery[:warmup], instance_id)
        if warm["statuses"].get("200", 0) != warmup:
            raise RuntimeError(f"Warmup con errores: {warm['statuses']}")
        m = measure(httpx, client, base, battery[warmup:], instance_id, sample)
        burst = run_burst(httpx, base, battery[0], cfg.max_queue + 3, cfg.request_timeout_s + 30, instance_id)
    usage_rows, wall = m["usage_rows"], m["wall"]
    total = {k: sum(u[k] for u in usage_rows) for k in usage_rows[0]} if usage_rows else {}
    keys = ("checkpoint_id", "modality", "device", "cold_start_seconds", "warmup", "calibrated", "max_length")
    return {
        "server_ready": {k: ready.get(k) for k in keys},
        "server_identity": {k: ready.get(k) for k in ("instance_id", "pid", "code_sha256")},
        "cold_start_seconds_client": round(cold_start, 2),
        "measured_requests": len(battery) - warmup,
        "warmup_requests": warmup,
        "status_counts": m["statuses"],
        "http_latency_ms": _summary(m["http_ms"]) if m["http_ms"] else None,
        "server_timing_ms": {k: _summary(v) for k, v in m["server"].items()},
        "throughput": {
            "requests_per_s": round(len(usage_rows) / wall, 3),
            "questions_per_s": round(total.get("questions", 0) / wall, 3),
            "rows_per_s": round(total.get("expanded_rows", 0) / wall, 3),
            "tokens_per_s": round(total.get("processed_input_tokens", 0) / wall, 1),
        },
        "usage_totals": total,
        "server_rss_peak_bytes_sampled": rss[0],
        "burst": burst,
        "note": (
            "Secuencial salvo la ráfaga; RSS muestreado tras cada petición; "
            "memoria MPS no observable desde el cliente; identidad del servidor comprobada en cada respuesta"
        ),
    }


def run_burst(httpx, base, body, n: int, timeout: float, instance_id: str, transport=None) -> dict[str, Any]:
    """n peticiones simultáneas contra una cola de tamaño max_queue: se espera algún 503 queue_full.

    Cada hilo registra un resultado aunque falle (error de transporte o cuerpo no JSON); una
    respuesta 200 de otra instancia se cuenta aparte y se marca como inválida."""
    results: list[str] = []
    lock = threading.Lock()

    def one():
        try:
            with httpx.Client(timeout=timeout, transport=transport) as c:
                r = c.post(f"{base}/v1/decide", json=body)
            if r.status_code == 200:
                try:
                    check_identity(r.json()["metadata"], instance_id)
                    label = "200"
                except (IdentityError, ValueError, KeyError):
                    label = "200:identity_mismatch"
            else:
                try:
                    check_identity({"instance_id": r.headers.get("X-GSO-Instance-ID")}, instance_id)
                    code = _error_code(r)
                    label = f"{r.status_code}:{code}" if code else str(r.status_code)
                except IdentityError:
                    label = f"{r.status_code}:identity_mismatch"
        except Exception as exc:  # cada petición deja constancia, también las que fallan
            label = f"transport_error:{type(exc).__name__}"
        with lock:
            results.append(label)

    threads = [threading.Thread(target=one) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    counts: dict[str, int] = defaultdict(int)
    for label in results:
        counts[label] += 1
    if sum(counts.values()) != n:
        raise RuntimeError("Ráfaga con peticiones sin contabilizar")
    return {
        "concurrent_requests": n,
        "results": dict(counts),
        "valid": not any(k.endswith(":identity_mismatch") for k in counts),
    }
