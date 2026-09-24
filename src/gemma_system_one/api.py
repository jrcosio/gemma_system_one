"""API propia versión 1 (spec §9): ``POST /v1/decide``, ``GET /health/live``, ``GET /health/ready``.

- Un único hilo de inferencia con cola acotada; el endpoint async sólo encola y espera el
  resultado, nunca ejecuta cómputo bloqueante. Cola llena o tiempo agotado → 503. Un tiempo
  agotado cancela la petición si aún no empezó; un forward ya lanzado en la GPU no se interrumpe.
- El modelo se carga una vez en el ``lifespan`` y ``/health/ready`` sólo da 200 tras cargar,
  verificar el checkpoint (y su calibración) y completar un warmup real. Si falta el modelo, ready
  queda en 503 con el motivo: no hay respuestas simuladas.
- Límites antes de trabajar: cuerpo ≤ 8 MiB leído por trozos antes de parsear; JSON estricto (sin
  NaN/Infinity ni claves repetidas); contratos compartidos (``contracts.Question``, estado acotado);
  1–8 preguntas; imagen PNG/JPEG en base64 estricto, ≤ 5 MiB decodificados y ≤ 16 MP comprobados en
  la cabecera antes de decodificar, sin URL ni rutas del cliente.
- Errores: 422 esquema/rangos/contexto; 413 tamaño; 400 imagen corrupta o no admitida; 404 modelo
  desconocido; 503 no listo, cola llena o tiempo agotado; 500 fallo interno sin datos sensibles.
- Registro por petición: request_id, estado, duración y dimensiones; nunca estado ni imagen.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import math
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import FastAPI, Request  # a nivel de módulo: FastAPI resuelve aquí las anotaciones
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .contracts import Identifier, Question, parse_json_strict, validate_state
from .engine import DecisionResult, RequestError
from .env import mask_home
from .images import MAX_IMAGE_BYTES, ImageError, ImageTooLargeError, content_address, decode_image_bytes
from .inference import CONFIDENCE_METHOD

log = logging.getLogger("gso.api")
# Identidad del proceso servidor: quien lo lanza (p. ej. ``gso benchmark``) puede fijarla para
# comprobar que las respuestas vienen de ese proceso y no de otro que ocupe el puerto.
SERVER_INSTANCE_ENV = "GSO_SERVER_INSTANCE"
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_QUESTIONS = 8
MAX_IMAGE_B64_CHARS = 4 * math.ceil(MAX_IMAGE_BYTES / 3)


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["serve"]
    model_id: str = Field(default="gemma-system-one-e2b-v0.1", pattern=r"^[a-z0-9][a-z0-9.\-]{2,63}$")
    checkpoint: Path
    calibration: Path | None = None
    modality: Literal["text", "text+image"] | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_queue: int = Field(default=4, ge=1, le=256)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_body_bytes: int = Field(default=MAX_BODY_BYTES, ge=1024, le=MAX_BODY_BYTES)


def load_serve_config(path: Path) -> ServeConfig:
    return ServeConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --- contrato HTTP -----------------------------------------------------------


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    state: Any
    questions: dict[Identifier, Question]
    image: str | None = None  # PNG/JPEG en base64 estándar estricto

    @field_validator("state", mode="before")
    @classmethod
    def _state(cls, v: Any) -> Any:
        return validate_state(v)

    @field_validator("questions")
    @classmethod
    def _questions(cls, v: dict) -> dict:
        if not 1 <= len(v) <= MAX_QUESTIONS:
            raise ValueError(f"questions debe tener entre 1 y {MAX_QUESTIONS} preguntas")
        return v


Prob = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class NoulAnswer(_Out):
    type: Literal["noul"]
    noul: Prob


class ChoiceAnswer(_Out):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Prob]
    confidence: Prob


class ScoreAnswer(_Out):
    type: Literal["score"]
    score: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    probabilities: list[Prob]
    legend: list[str]
    confidence: Prob


class Usage(_Out):
    questions: int
    expanded_rows: int
    backbone_forwards: int
    processed_input_tokens: int
    image_tokens: int
    padding_tokens: int
    generated_tokens: Literal[0]


class DecideResponse(_Out):
    model: str
    checkpoint_id: str
    answers: dict[str, Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]]
    usage: Usage
    metadata: dict[str, Any]


# --- worker ------------------------------------------------------------------


@dataclass
class Job:
    state: Any
    questions: dict[str, Any]
    image: str | None  # base64 acotado; se decodifica sólo al salir de la cola
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    enqueued: float = field(default_factory=time.perf_counter)
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: float | None = None


def _resolve(fut: asyncio.Future, result=None, exc: BaseException | None = None) -> None:
    if fut.done():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(result)


class InferenceWorker(threading.Thread):
    """Único hilo que posee el modelo: carga, warmup y peticiones en orden de llegada."""

    def __init__(self, engine_factory, max_queue: int):
        super().__init__(name="gso-inference", daemon=True)
        self.engine_factory = engine_factory
        self.jobs: queue.Queue[Job | None] = queue.Queue(maxsize=max_queue)
        self.ready = threading.Event()
        self.engine = None
        self.error: str | None = None
        self.info: dict[str, Any] = {}
        self._stopping = threading.Event()

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            engine = self.engine_factory()
            warm = engine.warmup()
        except Exception as exc:  # el motivo queda en /health/ready; no se sirve nada simulado
            self.error = f"{type(exc).__name__}: {mask_home(str(exc))[:500]}"
            log.error("modelo no disponible: %s", self.error)
            return
        self.engine = engine
        self.info = {
            **engine.describe(),
            "warmup": warm,
            "cold_start_seconds": round(time.perf_counter() - t0, 3),
        }
        self.ready.set()
        log.info("modelo listo en %.2f s", self.info["cold_start_seconds"])
        while not self._stopping.is_set():
            job = self.jobs.get()
            if job is None:
                break
            if job.cancelled.is_set():
                continue
            job.started = time.perf_counter()
            image = None
            try:
                image = _decode_image(job.image) if job.image is not None else None
                if job.cancelled.is_set():
                    continue
                outcome = (engine.decide(job.state, job.questions, image), None)
            except Exception as exc:
                outcome = (None, exc)
            finally:
                if image is not None:
                    image[0].close()
            with contextlib.suppress(RuntimeError):  # bucle cerrado: el cliente ya no espera
                job.loop.call_soon_threadsafe(_resolve, job.future, *outcome)

    def submit(self, job: Job) -> None:
        self.jobs.put_nowait(job)  # queue.Full si la cola está llena

    def stop(self, timeout: float = 30.0) -> None:
        self._stopping.set()
        with contextlib.suppress(queue.Full):
            self.jobs.put(None, timeout=1.0)
        self.join(timeout=timeout)


# --- aplicación ----------------------------------------------------------------


def _error(status: int, code: str, message: str, request_id: str):
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message, "request_id": request_id}}
    )


async def _read_limited(request, limit: int) -> bytes | None:
    """Cuerpo completo o ``None`` si supera ``limit`` (se comprueba antes de parsear)."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        return None
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_image(b64: str):
    """(PIL RGB, referencia por contenido) o excepción tipada; nunca decodifica de más."""
    if len(b64) > MAX_IMAGE_B64_CHARS:
        raise ImageTooLargeError(f"imagen: supera {MAX_IMAGE_BYTES} bytes decodificados")
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("imagen: base64 no válido") from exc
    pil, ext = decode_image_bytes(data, name="imagen")
    return pil, content_address(data, ext)


def _validation_message(exc: ValidationError) -> list[dict[str, Any]]:
    """Errores de esquema sin reproducir la entrada (ni estado ni textos del cliente)."""
    # Pydantic incluye valores del cliente en msg (p. ej. discriminadores inválidos)
    # y claves arbitrarias en loc. Sólo publicar el campo superior conocido y el tipo.
    fields = {"model", "state", "questions", "image"}
    return [
        {
            "loc": [e["loc"][0] if e["loc"] and e["loc"][0] in fields else "request"],
            "msg": f"Valor no válido ({e['type']})",
        }
        for e in exc.errors()[:20]
    ]


def create_app(cfg: ServeConfig, engine_factory=None):
    if engine_factory is None:
        from .engine import DecisionEngine

        def engine_factory():
            return DecisionEngine(cfg.checkpoint, cfg.calibration, cfg.modality)

    @asynccontextmanager
    async def lifespan(app):
        worker = InferenceWorker(engine_factory, cfg.max_queue)
        app.state.worker = worker
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Gemma System One", version="1", lifespan=lifespan, docs_url=None, redoc_url=None)
    from .env import source_fingerprint

    identity = {
        "instance_id": os.environ.get(SERVER_INSTANCE_ENV) or uuid.uuid4().hex,
        "pid": os.getpid(),
        "code_sha256": source_fingerprint()["sha256"],
    }

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        w: InferenceWorker = request.app.state.worker
        if w.ready.is_set():
            return {"status": "ready", "model": cfg.model_id, **w.info, **identity}
        status = "error" if w.error else "loading"
        return JSONResponse(status_code=503, content={"status": status, "reason": w.error, **identity})

    @app.post("/v1/decide")
    async def decide(request: Request):
        rid = uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        w: InferenceWorker = request.app.state.worker
        dims: dict[str, Any] = {}

        def done(response, status: int):
            response.headers["X-GSO-Instance-ID"] = identity["instance_id"]
            log.info(
                "request_id=%s status=%d ms=%.1f questions=%s rows=%s tokens=%s image=%s",
                rid, status, (time.perf_counter() - t0) * 1000, dims.get("questions"), dims.get("rows"),
                dims.get("tokens"), dims.get("image"),
            )  # fmt: skip
            return response

        if not w.ready.is_set():
            return done(_error(503, "not_ready", "El modelo no está listo", rid), 503)
        raw = await _read_limited(request, cfg.max_body_bytes)
        if raw is None:
            return done(
                _error(413, "body_too_large", f"El cuerpo supera {cfg.max_body_bytes} bytes", rid), 413
            )
        try:
            payload = parse_json_strict(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError):
            return done(
                _error(422, "invalid_json", "JSON no válido (UTF-8, sin NaN ni claves repetidas)", rid), 422
            )
        try:
            req = DecideRequest.model_validate(payload)
        except ValidationError as exc:
            resp = JSONResponse(
                status_code=422,
                content={"error": {"code": "invalid_request", "message": "Petición no válida",
                                   "details": _validation_message(exc), "request_id": rid}},
            )  # fmt: skip
            return done(resp, 422)
        dims.update(questions=len(req.questions), image=req.image is not None)
        if req.model != cfg.model_id:
            return done(
                _error(404, "unknown_model", f"Modelo desconocido; este servidor sirve {cfg.model_id}", rid),
                404,
            )
        if req.image is not None and len(req.image) > MAX_IMAGE_B64_CHARS:
            return done(_error(413, "image_too_large", "Imagen demasiado grande", rid), 413)
        loop = asyncio.get_running_loop()
        job = Job(req.state, dict(req.questions), req.image, loop, loop.create_future())
        try:
            w.submit(job)
        except queue.Full:
            return done(_error(503, "queue_full", "Cola de inferencia llena; reintentar más tarde", rid), 503)
        try:
            result: DecisionResult = await asyncio.wait_for(asyncio.shield(job.future), cfg.request_timeout_s)
        except TimeoutError:
            job.cancelled.set()
            job.future.cancel()
            return done(
                _error(503, "timeout", "Tiempo agotado (un forward ya iniciado no se cancela)", rid), 503
            )
        except asyncio.CancelledError:
            job.cancelled.set()
            job.future.cancel()
            raise
        except ImageTooLargeError as exc:
            return done(_error(413, "image_too_large", str(exc), rid), 413)
        except ImageError as exc:
            return done(_error(400, "invalid_image", str(exc), rid), 400)
        except RequestError as exc:
            return done(_error(exc.status, exc.code, str(exc), rid), exc.status)
        except Exception as exc:
            log.error("request_id=%s fallo interno %s", rid, type(exc).__name__)
            return done(_error(500, "internal_error", "Fallo interno", rid), 500)
        dims.update(rows=result.usage["expanded_rows"], tokens=result.usage["processed_input_tokens"])
        started = job.started or job.enqueued
        timing = {
            "queue_wait": round((started - job.enqueued) * 1000, 2),
            **result.timing_ms,
            "server_total": round((time.perf_counter() - t0) * 1000, 2),
        }
        body = {
            "model": cfg.model_id,
            "checkpoint_id": w.info["checkpoint_id"],
            "answers": result.answers,
            "usage": result.usage,
            "metadata": {
                "confidence_method": CONFIDENCE_METHOD,
                "request_id": rid,
                "calibrated": {q.type: w.info["calibrated"][q.type] for q in req.questions.values()},
                "timing_ms": timing,
                "instance_id": identity["instance_id"],
                "pid": identity["pid"],
            },
        }
        try:
            DecideResponse.model_validate(body)
        except ValidationError:
            log.error("request_id=%s respuesta inválida", rid)
            return done(_error(500, "invalid_output", "La salida no cumple el contrato", rid), 500)
        return done(JSONResponse(content=body), 200)

    return app


def run_serve(config_path: Path) -> None:
    import uvicorn

    cfg = load_serve_config(config_path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Un solo proceso: varios duplicarían los pesos. Sin --reload (spec §9).
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, workers=1, reload=False, log_level="info")
