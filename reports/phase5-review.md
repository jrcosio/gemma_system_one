# Revisión independiente: cierre visual 4b y API de fase 5

Fecha: 2026-09-23. Rama `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`.

Se leyeron AGENTS.md, README.md, ESPECIFICACION.md, AUDITORIA.md y STATUS.md, el diff y los ficheros reales (gran parte está sin seguimiento; `git diff` no representa toda la implementación). No se cambian modelo, backend ni contrato público. No se ejecutaron entrenamientos largos, descargas ni servicios externos.

## Hallazgos por gravedad

1. **Alta, corregido: se podía eludir el split visual planificado cambiando la semilla.** `check_planned_split` retornaba sin validar en ese caso y `load_split` no lo comprobaba. Reproducción con 20 grupos sintéticos v2 y semilla 1 frente al plan 0: se aceptaba. Ahora ambos caminos rechazan la inconsistencia. Los artefactos existentes usan el plan correcto: estilos train {0,1,2,3,5,6}, validation {7,8}, calibration {9,10}, test {11,12}, disjuntos. No se repitió inferencia sobre test ni se eligieron hiperparámetros con él.
2. **Alta, corregido: decodificación de imágenes en el event loop antes de la cola y el timeout.** Bloqueaba el servidor y permitía acumular imágenes expandidas fuera del límite pretendido. Una decodificación de 150 ms con timeout de 30 ms devolvía 200. Se encola base64 acotado, se decodifica sólo en el worker, se cierra PIL al acabar y el timeout abarca la decodificación. Se cancelan futuros al vencer o desconectarse el cliente. Regresión CPU con doble: 503 timeout y ejecución en `gso-inference`; no constituye una medición de timeout en MPS.
3. **Media, corregido: JSON muy anidado devolvía 500.** Un cuerpo con 2000 niveles provoca RecursionError. Ahora devuelve 422.
4. **Media, corregido: errores Pydantic repetían valores del cliente.** Un discriminador inválido aparecía literalmente en la respuesta; también podían aparecer claves arbitrarias. Se mantienen loc/msg, limitados a campos conocidos y tipo de error.
5. **Media, corregido: warmup exigía Noul aunque el checkpoint sólo tuviera Choice o Score.** Ambos casos fallaban; ahora selecciona una primitiva realmente entrenada.
6. **Media, pendiente: atribución y completitud del benchmark.** `_drive` puede consultar otro servicio que ya ocupe el puerto antes de detectar que el proceso hijo no pudo arrancar. `_burst` tampoco registra como resultado las excepciones de transporte de sus hilos. Hallazgos por inspección, no reproducidos lanzando servicios. Antes de considerar robusta la medición, comprobar identidad del servidor hijo y contabilizar todas las peticiones fallidas. No hay evidencia de que afectaran a los informes históricos, pero el arnés no lo garantiza.
7. **Baja, corregido: salida de benchmark en directorio nuevo fallaba al abrir el log.** Se crea el directorio antes de abrirlo. Prueba intercepta Popen: comprueba el log y detiene deliberadamente el arranque, sin servicio real.

## Evidencia y límites

- Inspección de la ruta real: carga de backbone sin generación, extracción y máscaras compartidas, preguntas/criterios en serialización, etiquetas fuera de entrada, scorer compartido y pérdidas por grupo; base congelada en eval e inferencia con inference_mode, autograd para LoRA. No se identificó un nuevo bloqueo en esos caminos. Las pruebas reales incluyen forwards BF16/MPS, gradientes, pequeñas fixtures de sobreajuste y recarga, imagen y LoRA.
- `.venv/bin/pytest tests/unit tests/integration -q`: antes, **263 passed** (27,47 s); después de las seis regresiones iniciales, **269 passed** (28,78 s). Dos avisos: deprecación Starlette/httpx y conversión a float en un test de LoRA.
- `.venv/bin/pytest tests/integration/test_phase5_review.py -q`: inicialmente **5 failed**; `.venv/bin/pytest tests/integration/test_phase5_review.py -k planned_seed -q`: **1 failed, 5 deselected**. Son los seis fallos reproducidos antes de corregir.
- `.venv/bin/pytest tests/integration/test_phase5_review.py tests/integration/test_api.py -q`: **10 passed** (6,19 s), tras correcciones iniciales.
- `.venv/bin/pytest tests/integration/test_phase5_review.py -q`: final **7 passed** (0,75 s), incluyendo la prueba adicional de directorio del benchmark. No se afirma que la suite completa de 270 casos se ejecutara: se ejecutó la de 269 y la adicional en este subconjunto.
- `.venv/bin/pytest tests/mps tests/e2e -q -rs`: **14 passed**, **0 omitidos**, 132,36 s. Diez tests MPS (incluyen una configuración pequeña) y cuatro E2E con checkpoints reales locales. E2E usa ASGI/TestClient, no prueba el socket TCP. Los tests CPU con dobles sólo prueban caminos de error.
- `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `uv lock --check`, `git diff --check`: correctos. Ruff comprueba 111 ficheros con formato correcto. Hubo un intento fallido de autofix SIM117 (se corrigió manualmente) y un intento con `python` ausente; se usó `.venv/bin/python` después.
- Auditoría local de metadatos: hash de dataset y splits al cargar train/validation/calibration; conjuntos de estilos disjuntos; recomputados los SHA256 de miembros ordenados de los tar `7f645568…` y `78d02416…`, coinciden con sus nombres. El inicio de V2 se acredita con env.json; su manifiesto antiguo no contiene `run_start_code`. No prueba por sí solo cuándo se redactó el protocolo.
- `.venv/bin/python scripts/snapshot_source.py`: fuente actual **ac7f751bc509025441ac23f9a581be0728fbf3f0f0087e5db42353cff6d92292**, 69 archivos, conservada en `artifacts/source/<hash>.tar`. Este hash excluye tests y documentación.

## Decisión y pendientes

La ejecución funcional real de la fase 5 y la separación visual v2 quedan verificadas con las correcciones anteriores. **No se otorga aprobación sin reservas al benchmark ni al cierre completo de la fase** hasta resolver su atribución y contabilización de fallos. Las latencias de los informes anteriores pertenecen al código `910a39a4…`, no al corregido; no se repitieron 100 peticiones. No se inicia fase 6.

Un timeout HTTP no interrumpe un forward MPS ya iniciado; la cola limita trabajo admitido, no todas las conexiones/cuerpos HTTP simultáneos. RSS muestreado no mide el pico total de memoria unificada/MPS. Sigue sin existir la copia histórica `70068e15…` de la fase 4 original. El resultado visual sólo avala esta familia sintética de gráficos, no generalización universal ni calibración universal.

Archivos cambiados en esta revisión: `src/gemma_system_one/api.py`, `engine.py`, `benchmark.py`, `data/split.py`, `tests/integration/test_phase5_review.py`, `docs/decisions/0009-local-api.md`, `docs/STATUS.md` y este informe. Sin commit ni modificación de pesos/datasets/checkpoints.
