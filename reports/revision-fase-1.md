# Revisión independiente de fase 1

2026-09-22 · Base `65d425025dd654fb4814062dc42b245062f151b1`, rama `main`, implementación y correcciones sin commit.

**Veredicto: puerta de fase 1 aprobada para el fixture textual Noul**, tras correcciones y repetición real en MPS. No se aprueba calidad general, calibración, Choice/Score entrenados, imagen, LoRA ni API.

## Hallazgos por gravedad

| Gravedad | Problema reproducido | Corrección |
|---|---|---|
| Alta | `load_split` aceptaba mover una pregunta de train a validation manteniendo hashes válidos y separando su grupo. También faltaban comprobaciones globales de IDs únicos/cobertura | Validación del manifiesto completo: formato, hashes, IDs, cobertura, grupos y fugas antes de devolver particiones |
| Alta | Huella registraba MPS solicitado aunque `allow_cpu_fallback` resolviera CPU; permitía confundir/reutilizar representaciones entre dispositivos. La ruta de entrenamiento tampoco rechazaba fallback silencioso de operadores | Dispositivo efectivo en huella y rechazo explícito de `PYTORCH_ENABLE_MPS_FALLBACK` para MPS |
| Media | Reducir `max_length` no invalidaba caché y un hit evitaba la tokenización que impone el límite | Límite incluido en la huella; cachés/checkpoints anteriores quedan como históricos, sin migración silenciosa |
| Media | Dataset con `image_path` válido entraba al entrenamiento textual ignorando la imagen | Rechazo explícito en la construcción de filas de fase 1, compartida por entrenamiento, baselines y evaluación |
| Media | JSON con `{"label":0,"label":1}` aceptaba silenciosamente la última etiqueta | Rechazo de claves repetidas a cualquier profundidad en el parser estricto |
| Media | F1 devolvía `None` con ambas clases presentes y todas las decisiones erróneas, aunque precision=recall=0 son valores definidos | F1=0 en ese caso, conservando el tratamiento existente de precision/recall no definidos |
| Baja | `splits_not_read` y documentación afirmaban que no se leían datos reservados, pero se parsea/valida todo el JSONL. La causa exacta de la dependencia del lote se presentaba como demostrada | Campo interno `splits_not_used_for_fitting`; documentación distingue inspección de ajuste/selección; causa de kernels documentada como hipótesis |

Las primeras cinco reproducciones dieron **5 failed** antes de corregir. Las de fallback de operadores y F1 dieron **2 failed** antes de corregir. Se añadieron cuatro casos adicionales de manifiesto manipulado. Las **11 regresiones** están en `tests/unit/test_phase1_review.py`.

Las pruebas con dobles verifican esas guardas. La evidencia de inferencia/entrenamiento proviene separadamente de los pesos reales.

## Evidencia y comandos

- `.venv/bin/pytest tests/unit tests/integration -q` fuera del sandbox, antes de cambios: **139 passed, 3,15 s**.
- `.venv/bin/pytest tests/unit/test_phase1_review.py -q`: reproducciones fallidas indicadas arriba; después incluidas en suite final.
- `.venv/bin/pytest tests/unit tests/integration -q` fuera del sandbox tras correcciones: **150 passed**.
- `.venv/bin/pytest tests/mps -v -rs` fuera del sandbox: **4 passed, 0 skipped, 16,52 s**. Incluye un Gemma diminuto en MPS, doctor E2B real, procesador real (sin forward GPU en esa prueba) y sobreajuste/recarga E2B real desde texto. No son cuatro entrenamientos reales ni cuatro pruebas exclusivamente GPU.
- `.venv/bin/python /private/tmp/gso-review-phase1.py`: repetición secuencial y breve de `run_train` con `configs/noul_smoke.yaml` y `configs/noul_overfit.yaml`, seguida de CLI de evaluación en procesos nuevos. Sólo cabezal CPU, extracción MPS, pesos existentes; sin descargas. Resumen local: `reports/doctor/phase1-review-runs.json` (ignorado por Git).
- `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `git diff --check`: correctos al cierre.

Equivalentes reproducibles de la sonda temporal (no requiere conservar el script):

```bash
.venv/bin/gso train --config configs/noul_smoke.yaml
.venv/bin/gso train --config configs/noul_overfit.yaml
# Sustituir los timestamps por los que produzca la ejecución:
.venv/bin/python -m gemma_system_one.cli evaluate --checkpoint runs/noul_smoke/20260922T204853Z/checkpoint --split validation --no-cache
.venv/bin/python -m gemma_system_one.cli evaluate --checkpoint runs/noul_overfit/20260922T204909Z/checkpoint --split train --no-cache
```

Los dos últimos comandos corresponden a las invocaciones del subproceso ejecutadas por la sonda. La sonda ejecutó los dos entrenamientos mediante `run_train(Path(config))`, no mediante esos comandos `gso train`.

## Resultados reales renovados

| Caso | Checkpoint nuevo | Resultado |
|---|---|---|
| Humo | `runs/noul_smoke/20260922T204853Z/checkpoint` | 84 train / 12 validation; NLL validation: cabezal 0,0076008, prior 0,569065, BoW 0,472011 |
| Sobreajuste | `runs/noul_overfit/20260922T204909Z/checkpoint` | 32 train; accuracy 1,0, NLL 2,73105e-10; puerta cumplida |
| Recarga humo sin caché | Proceso nuevo, 12 filas validation | Backbone cargado; diferencia máxima de logit **0,0**, tolerancia 1e-4 |
| Recarga sobreajuste sin caché | Proceso nuevo, 32 filas comparadas de train | Backbone cargado; diferencia máxima de logit **0,0**, tolerancia 1e-4 |

Las métricas coinciden con las originales para este fixture. El dataset y el split existentes pasan la validación reforzada. No se evaluaron test ni calibración. El cargador sí inspecciona todo el dataset para validarlo; no utiliza las etiquetas reservadas para ajustar pesos o elegir época. Los JSON de los runs emitidos antes del cambio de nombre conservan `splits_not_read` como campo histórico con esa interpretación.

## Código y metodología contrastados

- Gemma4Model local sin cabeza generativa, sin `generate`; dispositivo/dtype explícitos, caché de decodificación desactivada, máscaras/campos de procesador conservados y pooling por última posición válida.
- Serialización compartida: preguntas e instrucciones entran, Choice incorpora criterios completos/candidato, Score conserva orden. Etiquetas y metadatos quedan fuera. No se implementó ni certificó pérdida de grupo Choice/Score en esta fase.
- Extracción congelada con `no_grad`, base en eval; cabezal entrenado CPU/FP32 tras liberar backbone. La acumulación Noul divide cada suma de pérdidas por el número real de preguntas del paso, incluido el parcial; tests existentes verifican gradientes/equivalencia.
- Selección de época sólo con validación; sobreajuste explícito selecciona la última época. Baselines sólo se ajustan con train. Los cuatro grupos de validación y sus plantillas compartidas no miden transferencia.
- Revisión de manifiestos, métricas y código real, además de diff de README; todos los archivos nuevos siguen sin seguimiento y no aparecen en `git diff` normal.

## Decisiones y riesgos pendientes

[Decisión 0003](../docs/decisions/0003-extraction-fingerprint-validation.md): nueva huella interna y validación de manifiestos. Modelo/backend/contrato público permanecen iguales. No se sobrescribieron datos, splits, pesos o checkpoints anteriores. Los nuevos checkpoints son la referencia compatible; los originales se conservan como evidencia histórica.

- Muestras de memoria de fase 1 se toman al cargar y tras extracción, no en cada forward: el máximo observado puede omitir picos. No es un límite exacto del allocator ni benchmark de memoria.
- El detector de casi duplicados usa el primer estado de cada grupo; para grupos con paráfrasis múltiples puede omitir semejanzas en estados posteriores. El fixture revisado usa un estado por grupo. Reforzarlo antes de datasets con variantes.
- Hash de entrada del manifiesto usa estructura cruda; diferencias que desaparecen en serialización (p. ej. finales de línea, IDs de opciones renombrados) requieren mayor validación de duplicados semánticos antes de ampliar datos/Choice. No se identificó esa fuga en este fixture.
- Caché identifica revisión declarada/configuración de extracción, pero no vuelve a hashear todos los pesos base en cada uso. No detecta toda corrupción local de contenido por sí sola.
- Dependencia numérica del lote sigue siendo un riesgo si se cambia la política de una fila; sus causas exactas no se aislaron.
- Validación saturada, sesgo por selección de época, sin transferencia ni calibración; una NLL descendente en tres épocas no demuestra por sí sola que hagan falta más.
- Persisten riesgos de fases futuras: LoRA/autograd, imagen, FP16, estabilidad prolongada, contexto >512 y recursos del servicio. No se amplió el alcance para resolverlos aquí.
