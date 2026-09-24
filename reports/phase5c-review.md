# Revisión independiente del cierre de la fase 5

Fecha: 2026-09-23. Rama `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`. Fuente revisada: `73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81` (69 archivos).

Se contrastaron AGENTS.md, README.md, `docs/ESPECIFICACION.md`, `docs/AUDITORIA.md`, `docs/STATUS.md`, el diff y los archivos reales. Git sólo muestra el cambio de README porque `src/`, `tests/`, `configs/` y otros directorios aún carecen de seguimiento. Se inspeccionaron la carga, serialización, extracción, entrenamiento, recarga, API, benchmark y sus pruebas. No se modificó código, modelo, backend ni contrato.

## Hallazgos por gravedad

- **Alta:** no se reprodujo un bloqueo nuevo de ejecución ni un defecto que invalide los resultados actuales. La prueba real de `Gemma4Model` E2B en MPS/BF16 y la prueba LoRA sobre E2B pasaron sin omisiones. Dos E2E reales recargaron B y V2 y comprobaron salud y respuesta visual contra predicciones guardadas.
- **Media, pendiente de trazabilidad:** las fases 0–5 siguen sin commit; `git diff` no contiene el código ejecutado y la huella de fuentes no incluye tests ni documentación. El tar de fuentes actual existe y su hash se recomputó correctamente; esto permite recuperar el código, pero no equivale a un historial versionado completo. No se cambió el estado de Git durante esta revisión.
- **Media, límite de evidencia:** el benchmark `phase5c` atribuye las respuestas 200 al servidor hijo y registra dos `503:queue_full` por modalidad con identidad comprobada por el código vigente. El informe conserva los conteos y la identidad global, no cada cabecera individual; los logs de servidor contienen 112 peticiones por modalidad (110 × 200 y 2 × 503). El muestreo RSS no mide el pico de memoria MPS ni toda la memoria unificada.
- **Baja, límites conocidos:** el timeout HTTP no detiene un forward MPS iniciado; la cola no acota todas las conexiones HTTP. El test visual cubre una familia sintética de gráficos y V2 no está calibrado. La copia de fuentes histórica `70068e15…` sigue ausente. Ninguno de estos límites se presenta como compatibilidad o calidad universal.

## Comprobaciones

- **Modelo y parámetros:** `Gemma4Model` se carga desde snapshot local sin cabeza generativa; `move_inputs` conserva enteros y mueve flotantes al dtype del modelo. El pooling usa la última posición válida de `attention_mask`, con padding izquierdo o derecho. La base permanece congelada y en eval; LoRA tiene autograd y gradientes finitos; el servicio usa `inference_mode` y cabezales FP32. Se leyó el código y se ejecutaron las pruebas reales indicadas abajo.
- **Entradas y pérdidas:** la plantilla contiene estado, pregunta, criterios y candidato/nivel. Los IDs opacos y etiquetas no aparecen en el texto; las descripciones duplicadas se rechazan. Choice/Score usan un evaluador compartido, softmax en el grupo completo y una pérdida por pregunta normalizada por el tamaño real del paso. Se ejecutaron pruebas de serialización, remapeo, gradientes y microbatch.
- **Datos y artefactos:** `load_split` comprobó cobertura y fugas por grupo para `pilot_v3` (3000 filas; 700/100/100/100 grupos) y `vision_pilot_v2` (2100 filas; 490/70/70/70 grupos), sin intersecciones. Los manifiestos B, V2 y C2 declaran Gemma 4 E2B, BF16, plantilla `gso-text-v1` y las tres primitivas entrenadas. Se recomputó el hash del tar de fuentes actual y se verificó que coincide con el hash cliente/servidor de ambos JSON `phase5c`.
- **Benchmark:** cada JSON informa 100 respuestas 200, `same_code_as_server: true` y ráfaga válida de siete peticiones (cinco 200, dos 503). La inspección independiente de los logs encontró 112 peticiones por servicio con esos estados. No se repitieron las 100 consultas ni se abrió un servicio TCP en esta revisión.

Comandos de prueba ejecutados:

```sh
.venv/bin/pytest tests/unit/test_phase5_benchmark.py tests/integration/test_phase5_review.py tests/integration/test_api.py -q
# 21 passed, 1 aviso Starlette/httpx, 6,23 s
.venv/bin/pytest tests/e2e/test_api_real.py::test_ready_reports_real_device_and_calibration tests/e2e/test_api_real_vision.py::test_vision_api_matches_evaluation_and_requires_image -q -rs
# 2 passed, 0 omitidos, 1 aviso Starlette/httpx, 13,33 s
.venv/bin/pytest tests/mps/test_mps_real.py::test_doctor_real_e2b tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs
# 2 passed, 0 omitidos, 15,36 s
.venv/bin/pytest tests/unit/test_pooling.py tests/unit/test_serialization.py tests/unit/test_lora.py tests/unit/test_dataset_split.py tests/unit/test_checkpoint.py -q
# 49 passed, 1 aviso de conversión a float en un test, 4,09 s
git diff --check
```

Las cuatro pruebas E2E/MPS ejecutadas aquí usaron pesos locales y MPS real; las pruebas CPU no se presentan como validación de hardware. Las 280 CPU y 14 MPS/E2E completas de STATUS pertenecen al cierre anterior con el mismo hash de código; no se repitieron en esta revisión. No se lanzaron entrenamientos largos, descargas ni servicios externos.

## Dictamen y continuación

La fase 5 cumple la puerta funcional de la especificación con checkpoint entrenado, E2E real y benchmark HTTP local atribuido al servidor hijo. Este dictamen se limita al prototipo y a los datos sintéticos examinados. No se propone una corrección de código porque no se reprodujo un error nuevo. Antes de avanzar, conservar un commit o una referencia versionada del árbol completo y mantener las limitaciones de memoria, timeout y alcance de datos visibles en los informes. No usar los tests reservados para escoger ajustes de la fase 6.
