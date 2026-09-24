# Segunda revisión independiente de fase 5

2026-09-23. Rama `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`. No se inicia fase 6.

Se contrastaron AGENTS, especificación, auditoría, README, STATUS, diff y código real. La comparación de fuentes con `artifacts/source/ac7f751b*.tar` muestra que el cierre anterior sólo cambió `api.py` y `benchmark.py`. El diff Git no incluye la mayoría de los archivos, aún sin seguimiento.

## Hallazgos

- **Media, corregido: atribución incompleta de errores del benchmark.** `measure` y `run_burst` comprobaban identidad sólo con HTTP 200. Un 503 `queue_full` sin identidad se contaba y la ráfaga figuraba válida. Dos regresiones con MockTransport fallaron antes del arreglo: la secuencial no lanzaba IdentityError y la ráfaga devolvía valid=True. Se añade cabecera `X-GSO-Instance-ID` a las respuestas del endpoint y se exige en los errores; una respuesta ajena invalida la medición. Decisión documentada en 0009, sin cambiar JSON público ni inferencia.
- **Baja, corregido: cantidades inválidas del benchmark sin rechazo temprano.** `requests <= 0` o `warmup < 0` llegaban a preparar la batería y podían iniciar un servidor sin una medición válida. Tres regresiones interceptan la carga de configuración y demuestran que antes se alcanzaba; ahora reciben ValueError antes de ese punto. No se lanzó un servidor para reproducirlo.
- **Afirmación no respaldada del cierre anterior:** «identidad comprobada en todas las respuestas» y «ráfaga válida» excedían la comprobación implementada para los errores. Los JSON de `reports/phase5b` contienen 100 respuestas 200 por modalidad con igualdad de hash cliente/servidor; esa evidencia permanece. Sus dos 503 por ráfaga no tienen atribución verificada por el arnés anterior. No se afirma que fueran ajenos, sino que faltaba comprobación.

## Revisión del camino de modelo

Sin cambios respecto a la fuente revisada: Gemma4Model local sin lm_head ni generación; movimiento de flotantes al dtype configurado preservando enteros; pooling por última máscara válida; plantilla incluye preguntas y criterios sin etiquetas ni IDs opacos; softmax por grupo completo; backward por pregunta normalizado por tamaño real del paso. La base permanece en eval y sólo dropout LoRA cambia de modo. Inferencia usa inference_mode y cabezales CPU/FP32, con extracción MPS según configuración. Se inspeccionaron las rutas de checkpoints y extracción compartidas; no se detectó nuevo bloqueo en ellas. Las pruebas MPS incluyen un modelo diminuto y pruebas con pesos E2B reales: no todas las 14 pruebas son modelos completos.

## Pruebas y comandos

```sh
.venv/bin/pytest tests/unit/test_phase5_benchmark.py -k unattributed -q
# Antes: 2 failed, 5 deselected.
.venv/bin/pytest tests/unit/test_phase5_benchmark.py tests/integration/test_api.py -q
# Después del arreglo de identidad: 11 passed, 5,86 s.
.venv/bin/pytest tests/unit/test_phase5_benchmark.py -k invalid_counts -q
# Antes del rechazo temprano: 3 failed, 7 deselected.
.venv/bin/pytest tests/unit tests/integration -q
.venv/bin/pytest tests/mps tests/e2e -q -rs
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv lock --check
git diff --check
.venv/bin/python scripts/snapshot_source.py
git branch --show-current
git rev-parse HEAD
```

Resultados finales: **280 CPU passed (30,80 s)** y **14 MPS/E2E passed (136,46 s), cero omitidos**. Ruff, formato, lock y diff correctos. Al terminar no se detectan procesos del proyecto ni listener en 8000. La primera ejecución CPU tras la corrección de identidad pasó 277 pruebas (29,59 s); la ejecución final incluye además los tres casos de cantidades inválidas. Las pruebas de errores HTTP/benchmark con dobles son CPU, no validación MPS. Los E2E usan ASGI, no un socket TCP externo.

Fuente final: `73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81` (69 ficheros), conservada en `artifacts/source/<hash>.tar`. Excluye tests/documentación. Se generó también una copia intermedia `cfc31e01…`; no es la final.

Archivos de esta revisión: `src/gemma_system_one/api.py`, `src/gemma_system_one/benchmark.py`, `tests/unit/test_phase5_benchmark.py`, `tests/integration/test_api.py`, `tests/e2e/test_api_real.py`, `docs/decisions/0009-local-api.md`, `docs/STATUS.md` y este informe.

## Riesgos y veredicto

La corrección de atribución está cubierta por regresiones y la cabecera por integración/E2E. No se repitió el benchmark de 100 consultas ni se publicó nueva latencia: los valores anteriores identifican `547b5658…`, no la fuente actual. Queda pendiente una ráfaga real con el arnés corregido para afirmar que los rechazos de saturación medidos pertenecen al servidor hijo. No se certifica el cierre completo de fase 5 con esa afirmación pendiente.

Permanecen los límites conocidos: timeout no interrumpe forward MPS, cola no limita todas las conexiones HTTP, RSS muestreado no es pico de memoria MPS, falta la copia histórica `70068e15…`, datos sintéticos limitados y V2 sin calibrar. No se reabrió test para elegir parámetros, no se modificaron checkpoints ni se hicieron descargas o entrenamientos largos.
