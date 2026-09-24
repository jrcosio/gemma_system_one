# Informe de fase 5: API local, cola, límites, E2E y benchmark

Fecha: 2026-09-23 · Mac M5 Pro (48 GB), E2B `3e22461f…` en MPS/BF16 · Sin commit · Hash de fuentes al cierre `910a39a4…` (69 ficheros, copia en `artifacts/source/`). Decisión: [0009](../docs/decisions/0009-local-api.md).

## Veredicto

**Criterio de salida de la fase 5 cumplido** (spec §10: «E2E con checkpoint entrenado, fallos claros y benchmark completo»):

- **E2E con checkpoints entrenados y E2B real en MPS** (`tests/e2e`, 4 tests, sin dobles):
  - **La API devuelve exactamente la misma respuesta que `gso evaluate`** para las mismas preguntas. Se comprobó en texto, con el artefacto B de la fase 3 (LoRA + calibración), y con imagen, con V2 del cierre de la fase 4. Es la identidad de serialización entre entrenamiento y servicio (§12).
  - Invariantes:
    - permutar y renombrar opciones da las mismas probabilidades remapeadas;
    - una pregunta independiente no altera las demás;
    - otra pregunta sobre el mismo estado cambia la respuesta.
  - Errores: modelo desconocido → 404; imagen en un checkpoint de texto → 422; falta la imagen en uno visual → 422.
  - `usage` real, con `generated_tokens = 0`.
- **Fallos claros** (tests CPU con el motor real sobre Gemma 4 diminuto):
  - errores de esquema, NaN o claves repetidas, límites de preguntas y opciones, cuerpo > límite, estado que excede el contexto;
  - imagen corrupta, truncada, animada, con formato no admitido o de más de 16 MP;
  - checkpoint ausente → `ready` 503 con motivo y `decide` 503;
  - cola llena y tiempo agotado, con un motor doble lento que sólo existe en el test.
  - Además, **cola llena con el modelo real** en la ráfaga del benchmark.
- **Benchmark completo** (`gso benchmark`): servidor real en otro proceso, arranque en frío aparte, 5 peticiones de calentamiento, **100 consultas medidas** por servicio, latencia HTTP y forward sincronizado por separado, y ráfaga concurrente.

**Límites:**

- Un solo worker y cola FIFO, sin batching entre peticiones.
- La latencia crece con las filas de la petición (K/M por pregunta, decisión 0002) y con la imagen (≈ 266 tokens visuales por fila).
- Sólo se sirve un checkpoint por proceso.
- No hay compatibilidad TypeSafe (opcional) ni umbrales de abstención.

## 1. Implementación

| Pieza | Archivo | Contrato |
|---|---|---|
| Carga verificada sin dataset | `training/decisions_pipeline.py` (`load_decision_model`) | Etapa, primitivas entrenadas, plantilla, procesador, huella; cabezales y LoRA. `prepare_evaluation` la reutiliza |
| Motor | `engine.py` | `expand` → una fila por forward (`extract_pooled`, imagen PIL ya validada) → cabezales CPU/FP32 → `reconstruct` con T calibrada; `inference_mode`; modalidad del checkpoint; invariantes de salida; `usage` y tiempos |
| HTTP | `api.py` | `POST /v1/decide`, `GET /health/live`, `GET /health/ready`. Un hilo de inferencia y cola acotada; límites de cuerpo, JSON estricto, contratos compartidos e imagen base64 estricta (≤ 5 MiB, ≤ 16 MP antes de decodificar); errores 422/413/400/404/503/500 sin reproducir la entrada; respuesta validada con Pydantic; logs sin contenido |
| Imagen | `images.py` (`decode_image_bytes`) | La misma verificación para dataset y API; `ImageTooLargeError` → 413 |
| Benchmark | `benchmark.py` | Batería de validation/calibration (test rechazado); servidor en subproceso; `ready`, warmup, N medidas, ráfaga; RSS del servidor muestreado |
| CLI | `cli.py` | `gso serve --config`, `gso benchmark --config --dataset --split` |
| Configuración | `configs/serve_text.yaml`, `configs/serve_vision.yaml` | Loopback 127.0.0.1:8000, un proceso, `max_queue` 4, timeout 120 s |

**Trazabilidad añadida:**

- los checkpoints nuevos registran `input_modality` y el código del inicio del run (`run_start_code`/`session_code`, con copia);
- `evaluate` y `calibrate` guardan también la copia de fuentes.

## 2. Pruebas

- **CPU:** **263 passed**. Nuevo `tests/integration/test_api.py`, 4 tests:
  - contrato e invariantes con el motor real;
  - límites y errores;
  - modalidad con imagen, limitada a 16 MP, sin animación y sin JPEG truncado;
  - modelo ausente;
  - cola llena y timeout con el doble.

  También `test_checkpoint_records_modality_and_run_start_code` y las regresiones de copia de fuentes.
- **MPS real:** `tests/mps`, **10 passed**.
- **E2E real:** `tests/e2e`, **4 passed** en 16,6 s: `test_api_real.py` (3) y `test_api_real_vision.py` (1).
- **Estilo y lock:** `ruff check`, `ruff format --check`, `uv lock --check` y `git diff --check` correctos.
- **Fallos encontrados y corregidos:**
  - FastAPI trataba `request: Request` como parámetro de consulta, por el import local con `from __future__ import annotations`: todas las rutas daban 422. Import a nivel de módulo; lo detectó el test de integración.
  - Aserción de dispositivo errónea en los E2E (`mps` y no `mps:0`), corregida en el test.

## 3. Benchmark (medido; `reports/phase5/benchmark_*.json`)

| Servicio | Texto: LoRA + calibración (fase 3) | Imagen: V2 (cierre fase 4) |
|---|---|---|
| Batería | validation de `pilot_v3`: 100 casos distintos | validation de `vision_pilot_v2`: 70 casos, 30 repetidos en el mismo orden |
| Peticiones medidas / warmup | 100 / 5 | 100 / 5 |
| Preguntas / filas / tokens totales | 300 / 781 / 188 609 | 300 / 832 / 372 878 (221 312 visuales) |
| Arranque en frío: cliente hasta `ready` / carga + warmup del servidor | 4,25 s / 3,56 s | 4,76 s / 3,94 s |
| Latencia HTTP media / p50 / p95 / máx. | 565 / 554 / 911 / 1196 ms | 2061 / 2231 / 3242 / 3530 ms |
| Forward sincronizado p50 / p95 | 532 / 896 ms | 2217 / 3227 ms |
| Espera en cola (secuencial) | 0,04 ms | 0,04 ms |
| Peticiones / preguntas / filas por segundo | 1,77 / 5,31 / 13,8 | 0,49 / 1,46 / 4,0 |
| RSS máximo muestreado del servidor | 1,42 GB | 1,10 GB |
| Ráfaga de 7 simultáneas (`max_queue` 4) | 5 × 200, 2 × 503 `queue_full` | 5 × 200, 2 × 503 `queue_full` |
| Estados HTTP | 100 × 200 | 100 × 200 |

**Lectura:**

- Casi toda la latencia es el forward (una fila por forward); la serialización, la validación y HTTP añaden ≈ 14–19 ms de media.
- Forward por token procesado: ≈ 0,29 ms en texto frente a ≈ 0,55 ms con imagen (suma de forwards / tokens válidos). Con imagen cada fila repite ≈ 266 tokens visuales y además pasa por la torre de visión.
- La memoria MPS del driver no es visible desde el cliente. En fases 3 y 4, la extracción sin gradiente midió ~11,4 GB de driver.

**Arranque en frío:** es el tiempo hasta `ready` en un proceso nuevo, tras ejecuciones previas en la misma sesión; la caché de ficheros del sistema probablemente estaba caliente, pero no se midió. No se midió un arranque tras reiniciar el sistema.

## 4. Comandos ejecutados (fase 5)

```bash
uv add "fastapi>=0.120" "uvicorn>=0.38" "httpx>=0.28"      # fastapi 0.141.1, uvicorn 0.53.0, starlette 1.7.0
.venv/bin/pytest tests/integration/test_api.py -q           # 4 passed (tras corregir el import de FastAPI)
.venv/bin/pytest tests/unit tests/integration -q            # 263 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -v -rs   # 10 + 2 passed, 2 fallos de aserción de dispositivo
caffeinate -i .venv/bin/pytest tests/e2e -v -rs             # 4 passed tras corregir la aserción
caffeinate -i .venv/bin/gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation \
  --requests 100 --warmup 5 --out reports/phase5/benchmark_text.json
caffeinate -i .venv/bin/gso benchmark --config configs/serve_vision.yaml --dataset data/vision_pilot_v2 --split validation \
  --requests 100 --warmup 5 --out reports/phase5/benchmark_vision.json
.venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check
.venv/bin/python scripts/snapshot_source.py                 # 910a39a4… (69 ficheros)
```

**Uso manual** (ilustrativo; este `curl` no se ejecutó en la sesión: las peticiones reales las hicieron los tests E2E y el benchmark):

```bash
uv run gso serve --config configs/serve_text.yaml    # 127.0.0.1:8000; esperar GET /health/ready == 200
curl -s -X POST http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' \
  -d '{"model":"gemma-system-one-e2b-v0.1","state":"Se ha cobrado dos veces el pedido. Solicito devolución.",
       "questions":{"refund":{"type":"noul","instructions":"¿Se solicita devolución?"}}}'
```

## 5. Cumplimiento de la spec §9 (resumen)

| Requisito | Estado |
|---|---|
| `POST /v1/decide` con la identidad `gemma-system-one-e2b-v0.1`; otro modelo → 404 | ✓ |
| Respuesta: `model`, `checkpoint_id`, `answers` con los mismos IDs, campos por primitiva, `confidence_method` y `usage` real con `generated_tokens = 0` | ✓ |
| Pydantic con uniones discriminadas y campos extra prohibidos; estado acotado y finito; 1–8 preguntas; Choice 2–8; Score 2–5 | ✓ (contratos compartidos) |
| Presupuesto de tokens sin truncar → 422 | ✓ |
| Imagen base64 estricta, 5 MiB, 16 MP antes de decodificar, cuerpo 8 MiB antes de parsear, sin URL ni ruta, sin animación | ✓ |
| Códigos 422 / 413 / 400 / 404 / 503 / 500 | ✓ |
| `/health/live` y `/health/ready` tras carga, verificación y warmup | ✓ |
| Carga única en `lifespan`, un worker, cola acotada, endpoint async sin cómputo bloqueante, `eval` + `inference_mode`, timeout sin cancelar la GPU | ✓ |
| Loopback 127.0.0.1:8000, sin reload, un proceso | ✓ |
| Logs con request ID, duración y dimensiones, sin contenido | ✓ |
| Compatibilidad TypeSafe | No implementada (opcional) |

## 6. Límites y riesgos

- **Rendimiento:** sin batching; una petición de 3 preguntas con imagen tarda ≈ 2 s. La latencia escala con K/M. `max_queue` y el timeout son valores iniciales, no ajustados.
- **Timeout:** con el modelo real sólo se comprobó en CPU con el doble.
- **Contrato:** sin abstención, sin umbrales de decisión en la API ni `confidence` en Noul (la spec no lo pide).
- **Modelos servidos:** V2 no está calibrado; B sí. La calidad servida es la medida en fases 3 y 4: datos sintéticos, sin generalización demostrada.
- **Registro:** avisos de bibliotecas en los logs (semáforo de `resource_tracker` al terminar con SIGTERM).

## 7. Cierre tras la revisión independiente (hallazgo 6: atribución y contabilización del benchmark)

La revisión ([phase5-review.md](phase5-review.md)) corrigió varios fallos de la API: split planificado con otra semilla, decodificación de imagen dentro del worker y del timeout, JSON profundo → 422, errores sin eco, warmup con una primitiva entrenada y log del benchmark en un directorio nuevo. Dejó pendiente el hallazgo 6: el benchmark podía medir a otro servidor que ocupara el puerto y no contabilizaba los errores de transporte de la ráfaga. Resuelto así (decisión 0009, sección «Cierre del hallazgo 6»):

| Cambio | Evidencia |
|---|---|
| Puerto comprobado antes de lanzar | `test_busy_port_is_rejected_before_launching_the_server`. **Reproducción real:** con un proceso escuchando en 127.0.0.1:8000, `gso benchmark --config configs/serve_text.yaml …` termina con código 1 y «127.0.0.1:8000 ya está en uso», sin lanzar el servidor ni escribir informe (`reports/phase5b/port_busy_check.log`) |
| Identidad del hijo (`GSO_SERVER_INSTANCE` aleatorio + PID) exigida en `ready`, warmup y cada respuesta | `test_identity_must_match_instance_and_pid`, `test_sequential_measurement_rejects_a_foreign_server`. Los tests de API (CPU y E2E) comprueban `instance_id`/`pid` en `ready` y en `metadata` |
| Cada petición contabilizada (200, HTTP con código, `invalid_body`, `transport_error:<tipo>`); sumas verificadas | `test_sequential_measurement_accounts_for_every_request`, `test_burst_records_transport_errors_and_foreign_answers` |

**Benchmarks repetidos con el código corregido** (hash `547b5658…`, igual en cliente y servidor: `same_code_as_server: true`; `reports/phase5b/benchmark_*.json`):

| Servicio | Texto (B, calibrado) | Imagen (V2) |
|---|---|---|
| Estados | 100 × 200 | 100 × 200 |
| Arranque hasta `ready` (cliente) | 4,90 s | 4,76 s |
| Latencia HTTP media / p50 / p95 / máx. | 567 / 552 / 927 / 1187 ms | 2059 / 2231 / 3247 / 3538 ms |
| Forward sincronizado p50 / p95 | 531 / 908 ms | 2217 / 3232 ms |
| Peticiones / preguntas / filas por segundo | 1,77 / 5,30 / 13,8 | 0,49 / 1,46 / 4,0 |
| Ráfaga de 7 | 5 × 200, 2 × 503 `queue_full`; `valid: true` | 5 × 200, 2 × 503 `queue_full`; `valid: true` |
| Identidad | `instance_id` y PID del hijo comprobados en todas | Ídem |

Las cifras coinciden con las de `910a39a4…` (§3) con diferencias de ≤ 16 ms en p50 y p95. **Las vigentes son las de esta sección.**

**Pruebas tras el cierre:**

- CPU: **275 passed**, que incluyen las 7 regresiones de la revisión y 5 del benchmark.
- MPS + E2E reales: **14 passed**, 0 omitidos, 131,6 s.
- Ruff, formato, lock y `git diff --check` correctos.

**Límites que se mantienen:**

- Un timeout no interrumpe un forward MPS ya iniciado.
- La cola limita el trabajo admitido, no las conexiones HTTP abiertas.
- El RSS muestreado no mide el pico de memoria unificada/MPS.
- El E2E usa ASGI (TestClient); el socket TCP real sólo lo ejercita el benchmark.

## 8. Cierre tras la segunda revisión (ráfaga real atribuida)

La segunda revisión ([phase5b-review.md](phase5b-review.md)) corrigió el arnés en dos puntos:

- la identidad se exige ahora también en los errores HTTP, mediante la cabecera `X-GSO-Instance-ID` en todas las respuestas del endpoint;
- las cantidades inválidas se rechazan antes de lanzar el servidor.

Dejó pendiente una ráfaga real con el arnés corregido. Se ejecutó con el código `73a72fdc…`, el mismo en cliente y servidor (`same_code_as_server: true`), y se guardó en `reports/phase5c/benchmark_*.json`.

| Servicio | Texto (B, calibrado) | Imagen (V2) |
|---|---|---|
| Medidas | 100 × 200 | 100 × 200 |
| Ráfaga de 7 | 5 × 200, **2 × 503 `queue_full` con identidad del hijo**; `valid: true` | Ídem |
| Log del servidor | 112 peticiones: 110 × 200 y 2 × 503 | 112 peticiones: 110 × 200 y 2 × 503 |
| Arranque hasta `ready` (cliente) | 4,24 s | 4,87 s |
| Latencia HTTP media / p50 / p95 / máx. | 575 / 561 / 940 / 1190 ms | 2063 / 2232 / 3248 / 3558 ms |
| Forward sincronizado p50 / p95 | 540 / 923 ms | 2217 / 3232 ms |
| Peticiones / preguntas / filas por segundo | 1,74 / 5,22 / 13,6 | 0,49 / 1,45 / 4,0 |

- El cliente contabilizó 112 peticiones: 5 de calentamiento + 100 medidas, todas 200, y una ráfaga de 7 con 5 × 200 y 2 × 503. Coinciden con las 112 registradas en el log del servidor lanzado.
- Cada 503 llevaba el `instance_id` aleatorio de ese hijo: `run_burst` lo exige y, si no, lo habría contado como `503:identity_mismatch` e invalidado la ráfaga.
- **Estas son las latencias vigentes** de la fuente actual. Las de §3 y §7 identifican `910a39a4…` y `547b5658…`, respectivamente.

Con esto queda satisfecho el último pendiente: **la fase 5 está cerrada**.

**Pruebas:** 280 CPU con el código `73a72fdc…`. Las 14 MPS/E2E se ejecutaron en la segunda revisión con ese mismo código y no se repitieron.
