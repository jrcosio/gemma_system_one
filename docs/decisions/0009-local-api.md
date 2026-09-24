# 0009 — API local v1: un worker, cola acotada, límites previos y motor compartido con la evaluación

Fecha: 2026-09-23 · Fase 5 · Estado: aceptada

## Contexto

La spec §9 fija `POST /v1/decide`, los campos de respuesta, `usage`, los límites de entrada, los códigos de error, la salud `live`/`ready`, un único proceso Uvicorn con un worker dedicado y cola limitada, y la escucha en loopback. §12 pide E2E con un checkpoint entrenado: cambio de pregunta, permutación de opciones, pregunta independiente, imagen requerida, saturación/timeout y modelo ausente.

## Decisión

1. **Dependencias:**
   - `fastapi` 0.141.1, `uvicorn` 0.53.0 y `httpx` 0.28.1;
   - `starlette` 1.7.0 llega como dependencia transitiva;
   - torch, transformers y peft no cambian.

   FastAPI se importa a nivel de módulo: con `from __future__ import annotations` resuelve las anotaciones en el ámbito del módulo. Un import local convertía `request: Request` en un parámetro de consulta y todas las rutas daban 422; lo detectó el test de integración.
2. **Motor compartido** (`engine.DecisionEngine`):
   - `load_decision_model` aplica las mismas verificaciones que la evaluación (etapa, primitivas entrenadas, plantilla, procesador, huella) sin necesitar el dataset;
   - serialización `expand`, una fila por forward con `extract_pooled`, cabezales en CPU/FP32 y `reconstruct` con la temperatura del artefacto de calibración vinculado;
   - `eval()` + `inference_mode()`, sin caché de representaciones y sin `generate()`: `generated_tokens` siempre vale 0.
3. **Modalidad fija por checkpoint.** Se lee de `extra.input_modality` o, en checkpoints anteriores, del dataset de entrenamiento. Con imagen, la imagen es obligatoria; en uno de texto se rechaza (422).
4. **Worker y cola:**
   - un hilo posee el modelo;
   - el endpoint async encola y espera con `asyncio.wait_for`, sin cómputo bloqueante en el bucle;
   - cola `max_queue` (4 por defecto): llena → 503 `queue_full`;
   - tiempo agotado → 503 `timeout`; la petición se descarta si aún no empezó, pero un forward ya iniciado en la GPU no se cancela;
   - `ready` sólo tras cargar, verificar y completar un warmup real; si falla, 503 con el motivo y nunca respuestas simuladas.
5. **Límites previos al trabajo:**
   - cuerpo ≤ 8 MiB, leído por trozos antes de parsear (413);
   - JSON estricto, sin NaN/Infinity ni claves repetidas (422);
   - `DecideRequest` con `extra="forbid"` y los contratos compartidos (`Question`, `validate_state`); 1–8 preguntas;
   - imagen en base64 estricto, con longitud acotada antes de decodificar;
   - `decode_image_bytes`, la misma función que el dataset: ≤ 5 MiB y ≤ 16 MP leídos en la cabecera (413), PNG/JPEG sin animación y decodificación completa (400);
   - filas que exceden `max_length` → 422, sin truncar.
6. **Respuesta:**
   - `model`, `checkpoint_id` (tipo + sha256 del manifiesto y, si hay, de la calibración), `answers` con los mismos IDs, `usage` real (preguntas, filas, forwards, tokens válidos, visuales y de padding) y `metadata` (`confidence_method`, `request_id`, calibración por primitiva y tiempos);
   - validada con modelos Pydantic: probabilidades finitas en [0, 1] y opción dentro del conjunto.
7. **Registro:** `request_id`, estado, duración y dimensiones; nunca estado, textos ni imagen. Los errores no reproducen la entrada del cliente.
8. **Benchmark** (`gso benchmark --config serve.yaml --dataset D --split validation|calibration`):
   - servidor real en otro proceso, con el arranque en frío medido aparte;
   - warmup explícito y ≥ 100 consultas medidas (cada una, un caso del dataset con todas sus preguntas);
   - latencia HTTP separada del forward sincronizado;
   - una ráfaga de `max_queue + 3` peticiones simultáneas para observar `queue_full`;
   - test queda excluido de la batería.

## Consecuencias

- Una petición Choice/Score cuesta K/M forwards; con imagen, cada fila repite ~266 tokens visuales (decisiones 0002 y 0007).
- La latencia depende del número total de filas de la petición.
- No hay batching entre peticiones: una sola cola FIFO.
- No se implementa compatibilidad TypeSafe (opcional en la spec).

## Revisión independiente (2026-09-23)

La decodificación de imagen se realiza dentro del único worker, después de admitir el base64 acotado en la cola; su tiempo queda incluido en el timeout. Se cierra la imagen tras el uso. El timeout cancela la espera, pero no puede interrumpir un forward MPS ya iniciado. Warmup utiliza una primitiva presente en el checkpoint. Se omiten valores y claves arbitrarias del cliente en errores de validación. Son correcciones del diseño existente, sin cambio de modelo, backend ni esquema público. Evidencia y limitaciones: `reports/phase5-review.md`.

## Cierre del hallazgo 6 de la revisión (2026-09-23)

- **Atribución.** El servidor expone `instance_id`, `pid` y `code_sha256` en `/health/ready` y `instance_id`/`pid` en `metadata` de cada respuesta. `instance_id` lo fija quien lo lanza con `GSO_SERVER_INSTANCE`, y si no, es aleatorio. `gso benchmark` comprueba antes de lanzar que el puerto está libre (si no, error sin iniciar el servidor), pasa un identificador aleatorio al hijo y exige ese identificador y el PID del hijo en `ready` y el identificador en cada respuesta. Una respuesta ajena invalida la medición (`IdentityError`).
- **Contabilización.** Cada petición del warmup, de la medición secuencial y de la ráfaga produce exactamente un resultado: 200, código HTTP con su `error.code`, `invalid_body` o `transport_error:<tipo>`. Las sumas se verifican. En la ráfaga, una respuesta 200 ajena se registra como `200:identity_mismatch` y marca `valid: false`.
- El informe del benchmark registra el hash de fuentes del cliente (con copia) y el que declara el servidor, en `same_code_as_server`.
- Evidencia: `tests/unit/test_phase5_benchmark.py` (5 casos) y reproducción real con el puerto 8000 ocupado (`reports/phase5b/port_busy_check.log`).

## Segunda revisión independiente: identidad de errores (2026-09-23)

Las respuestas de `/v1/decide` añaden `X-GSO-Instance-ID`, incluido cualquier error producido por el endpoint. El benchmark exige esa identidad en respuestas no 200; sin ella aborta la medición secuencial o invalida la ráfaga. Esto corrige la atribución de saturación a un servidor ajeno sin modificar el esquema JSON público, el modelo ni el backend. La cabecera es aditiva; un error de infraestructura sin identidad se rechaza conservadoramente. Las respuestas 200 conservan su validación por metadata. También se rechazan cantidades de benchmark inválidas antes de cargar configuración o lanzar procesos.
