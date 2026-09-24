# Informe de fase 1: contratos, datos, baselines y cabezal Noul

**Revisión posterior:** [revisión independiente](revision-fase-1.md). La ruta real se ha repetido y los fallos se han corregido; los runs de este informe son históricos y requieren la huella anterior. Calibración/test se inspeccionan al validar el dataset, pero no se usan para ajustar ni seleccionar.

Fecha: 2026-09-22 · Equipo objetivo (Apple M5 Pro, 48 GB, macOS 26.6.2) · Base git `65d4250` con cambios sin commit.

## Veredicto

**Puerta de fase 1 cumplida** (spec §10: «sobreajuste controlado de un fixture y baseline medido sin fugas»):

- **Sobreajuste:** 32 ejemplos de train con E2B congelado en MPS/bf16 y cabezal FP32. Accuracy de train = 1,0 y NLL de train = 2,7e-10. El criterio declarado era accuracy = 1 y NLL < 0,05.
- **Baselines:** prior y bolsa de palabras medidos en validación con los mismos splits. El split por grupos no tiene grupos compartidos, ni entradas idénticas entre particiones, ni casi duplicados (Jaccard ≥ 0,8).
- **Particiones sin leer:** calibración y test no se han leído en ningún entrenamiento ni evaluación de esta fase. `evaluate` rechaza test sin `--final-test`, y así se comprobó en ejecución real.

**Esto no es un resultado de calidad.** El fixture es sintético y usa un repertorio finito de frases. La validación tiene 12 preguntas en 4 grupos. La época se eligió en esa misma validación, y las plantillas de superficie se comparten entre particiones. Los números de §4 prueban que la ruta aprende y se recarga; no prueban generalización.

## 1. Qué se implementó

| Pieza | Archivo | Contrato |
|---|---|---|
| Esquema JSONL v1 | `src/gemma_system_one/contracts.py` | Unión discriminada Noul/Choice/Score con `extra="forbid"` y modo estricto. K de 2 a 8 y M de 2 a 5. Descripciones únicas tras normalizar. Etiqueta compatible con los criterios. Estado no vacío, finito y de profundidad ≤ 8. `image_path` relativa y sin `..`. JSON sin NaN ni Infinity |
| Plantilla `gso-text-v1` | `src/gemma_system_one/serialization.py` | Solo estado, tipo, instrucciones, criterios y candidato. Delimitadores escapados (`& < >`). Opciones de Choice en orden canónico y rúbrica de Score en su orden. Instrucción final fija. Nunca incluye IDs, grupo, familia, idioma, procedencia ni etiqueta |
| Validación | `src/gemma_system_one/data/dataset.py` | Errores con número de línea. IDs duplicados. Misma entrada con etiquetas distintas. Misma entrada en grupos distintos. Imágenes dentro de la raíz, incluidos enlaces simbólicos. Hash del manifiesto |
| Splits | `src/gemma_system_one/data/split.py` | 70/10/10/10 por `group_id`, deterministas con semilla. Manifiesto inmutable con hashes de entrada. Comprobaciones de fugas. Test protegido |
| Generador | `src/gemma_system_one/data/generate.py` | Tickets ES/EN con hechos controlados y reglas de etiqueta comprobables. Incluye negaciones y distractores («no quiero una devolución», «el año pasado me devolvieron…», «ya se solucionó»). Varias preguntas por estado |
| Lotes | `src/gemma_system_one/models/encoding.py` | Plantilla de chat oficial con `enable_thinking=False` y padding derecho. **Nunca trunca:** una fila demasiado larga produce un error con su índice |
| Extracción y caché | `src/gemma_system_one/features.py` | Último token válido y una fila por forward (decisión 0002). La clave de caché incluye checkpoint, dtype, atención, dispositivo, hash del procesador, plantilla, pooling, política de lotes, versiones y el hash de cada fila |
| Métricas | `src/gemma_system_one/metrics.py` | NLL, Brier, ECE del evento con 15 bins fijos, P/R/F1 (None si no están definidos), bootstrap por grupo y pares dentro del mismo estado. También Brier multiclase, RPS, ECE top-label y concentración, preparados para la fase 2 |
| Baselines | `src/gemma_system_one/baselines.py` | Prior de Laplace aprendido en train. Regresión logística L2 sobre unigramas y bigramas hasheados del mismo texto serializado. Hiperparámetros fijados de antemano |
| Entrenamiento | `src/gemma_system_one/training/noul.py`, `pipeline.py` | AdamW solo con los parámetros del cabezal. Pérdida por pregunta con acumulación normalizada por el número real de preguntas. Época elegida en validación. Checkpoint con procedencia completa |
| CLI | `src/gemma_system_one/cli.py` | `generate-data`, `validate-data`, `split`, `baselines`, `train` y `evaluate` |

Los splits dan a validación, calibración y test una partición cada uno. Esta fase solo lee train y validación.

## 2. Datos (evidencia: `gso validate-data`, `gso split`)

- `data/smoke_v1`: 120 preguntas Noul en 40 grupos (60 ES y 60 EN). Tres preguntas por estado. sha256 `2e48b1b5…705d`, generador `support-noul-v1`, semilla 0.
- Etiquetas: 37 positivas y 83 negativas. 20 de los 40 grupos tienen etiquetas distintas para preguntas sobre el mismo estado.
- Split con semilla 0, por grupos:

| Partición | Grupos | Preguntas | y=1 / y=0 |
|---|---|---|---|
| train | 28 | 84 | 25 / 59 |
| validation | 4 | 12 | 3 / 9 |
| calibration | 4 | 12 | 2 / 10 (no leída) |
| test | 4 | 12 | 7 / 5 (no leída) |

- Comprobaciones: 0 errores y 0 casi duplicados de estado entre particiones.
- **Limitación conocida:** todas las particiones comparten identificadores de formato (`es-text`, `en-json`…) y el mismo repertorio de frases. No hay un conjunto con plantillas no vistas (spec §5.3); queda para el piloto de la fase 2.
- Las filas reales tokenizadas miden como máximo 187 tokens, dentro del límite de 512 y sin truncar.

## 3. Hallazgos durante la fase

| Hallazgo | Reproducción | Resolución |
|---|---|---|
| `target.label: true` se aceptaba como 1 | `test_invalid_examples_rejected[overrides5]`: `Literal[0, 1]` admite `True` porque `True == 1` | `label` pasa a entero estricto con `ge=0` y `le=1` |
| La representación bf16 en MPS depende de la composición del microlote | `test_phase1_overfit_and_reload_from_text_on_mps`: logit desviado 2,56 al recalcular otro conjunto. Scripts en `scripts/repro_batch_*.py` | Una fila por forward (decisión 0002). Tras el cambio, la recarga da 0,0 |

Magnitud del segundo hallazgo: coseno ≥ 0,99946 entre composiciones y **Δp hasta 0,056** con el cabezal de humo, sin cambios de decisión.

## 4. Resultados

Métricas en validación: n = 12, 3 positivos y 4 grupos. Umbral fijado a priori en 0,5. Los intervalos son percentiles al 95 % de un bootstrap por grupo (1000 réplicas, semilla 0, solo 4 grupos: son muy anchos e inestables).

### Sobreajuste (`configs/noul_overfit.yaml`; run `runs/noul_overfit/20260922T203606Z`)

- 32 ejemplos de train, 200 épocas, lr 1e-2, `select_on: train`.
- Train: accuracy 1,000 y NLL 2,7e-10. **overfit_check: passed.**
- En validación el mismo cabezal tiene NLL 2,12 y accuracy 0,917: es un cabezal sobreconfiado, como se espera de un sobreajuste.

### Humo (`configs/noul_smoke.yaml`; run `runs/noul_smoke/20260922T203613Z`)

- 84 ejemplos de train, 3 épocas, lr 1e-3, wd 0,01 y 8 preguntas por paso. Son los valores iniciales de la spec §7.A.
- Época elegida: 3, la última. La NLL de validación aún bajaba (0,441 → 0,201 → 0,008): el máximo de 3 épocas limita el entrenamiento.

| Modelo | NLL val [IC] | Brier val | ECE evento | Accuracy val [IC] | P / R / F1 | Pares en el mismo estado |
|---|---|---|---|---|---|---|
| Prior (Laplace, p = 0,302) | 0,569 [0,430; 0,639] | 0,190 | 0,052 | 0,750 | — / 0,00 / — | 0,50 (6 pares) |
| BoW logístico | 0,472 [0,212; 0,646] | 0,170 | 0,261 | 0,750 [0,667; 0,917] | — / 0,00 / — | 0,83 (6 pares) |
| **E2B congelado + cabezal** | **0,008 [0,0001; 0,017]** | 0,0004 | 0,007 | **1,000 [1,0; 1,0]** | 1,00 / 1,00 / 1,00 | 1,00 (6 pares) |

- Diferencia de NLL entre el cabezal y BoW, con bootstrap emparejado: [−0,643; −0,195].
- En train: cabezal NLL 0,053 y accuracy 0,988 (1 error en 84); BoW NLL 0,156 y accuracy 0,964.
- «—» indica una métrica no definida: BoW y el prior no predicen ningún positivo en el umbral de 0,5.

### Inspección de predicciones

Revisé una a una las 12 de validación y las dudosas de train:

- El único error de train es un fallo activo mencionado al final de un estado JSON largo (p = 0,146, y = 1).
- Los casos más inciertos son la política compuesta «cobro duplicado Y devolución pedida» con el distractor «devolución pasada». En train quedan en p = 0,42 y 0,47, ambos con y = 0, del lado correcto pero cerca del umbral.

## 5. Recursos y reproducibilidad (evidencia medida)

- **Extracción en MPS/bf16**, una fila por forward:
  - 84 filas, 11 499 tokens válidos y 0 de padding, en 4,12 s sincronizados.
  - 12 filas en 0,56–0,69 s.
- **Memoria:**
  - MPS asignada: 10,22 GB (9,51 GiB) con el modelo cargado.
  - Driver MPS: máximo muestreado de 10,34 GB.
  - RSS del proceso: ≤ 1,0 GB.
  - Swap: 0 y sin variación. Presupuesto sin superar.
- **Entrenamiento del cabezal:** en CPU/FP32 sobre representaciones cacheadas, sin backbone cargado durante el bucle.
- **Recarga completa en un proceso nuevo** (`gso evaluate --no-cache`): checkpoint y procesador desde disco, texto re-serializado, backbone en MPS, logits comparados con los del entrenamiento.

  | Caso | Filas comparadas | Máx. dif. de logit |
  |---|---|---|
  | Humo, validación | 12 | 0,0 |
  | Humo, train | 84 | 0,0 |
  | Sobreajuste, train completo | 32 | 0,0 |

  Tolerancia declarada: 1e-4.
- **La caché no altera resultados:** en el run de humo, la validación salió de caché y coincide exactamente con el recálculo.

## 6. Pruebas

| Conjunto | Resultado | Qué cubre |
|---|---|---|
| `tests/unit` | 125 passed | Contratos (incluidos los ejemplos literales de la spec §5.1), serialización (etiquetas, IDs y procedencia ausentes; escape; permutación y renombrado de IDs; orden de la rúbrica; igualdad train/serve), validación, splits y fugas, generador (etiquetas = reglas), métricas con casos calculados a mano, acumulación normalizada, baselines |
| `tests/integration` | 14 passed | Gemma4Model real diminuto con procesador doble: padding derecho forzado, rechazo sin truncar, orden de extracción, validez de la caché, `train → evaluate` con recarga desde el texto, test protegido, dataset modificado rechazado |
| `tests/mps` | 4 passed | Doctor E2B (fase 0), E2B diminuto en bf16. Nuevos: filas con el procesador real (≤ 512 tokens, padding derecho, un BOS, termina en `<|turn>model\n`, sin `<|think|>`) y sobreajuste más recarga desde el texto con E2B real en MPS |

Total: `uv run pytest -rs` → 143 passed, 0 skipped. `ruff check` y `ruff format --check` sin errores.

## 7. Comandos ejecutados (en orden)

```bash
uv run gso generate-data --out data/smoke_v1 --cases 40 --seed 0
uv run gso validate-data --dataset data/smoke_v1
uv run gso split --dataset data/smoke_v1 --seed 0
uv run gso baselines --config configs/noul_smoke.yaml
uv run gso train --config configs/noul_overfit.yaml      # microlote 8 (sustituido, decisión 0002)
uv run gso train --config configs/noul_smoke.yaml        # microlote 8 (sustituido)
uv run gso evaluate --checkpoint runs/noul_smoke/20260922T203313Z/checkpoint --split validation --no-cache
uv run gso evaluate --checkpoint runs/noul_smoke/20260922T203313Z/checkpoint --split test   # rechazado (esperado)
uv run pytest tests/mps -v -rs                           # 1 fallo: dependencia del microlote
uv run python scripts/repro_batch_dependence.py          # (copia previa en el scratchpad)
uv run python scripts/repro_batch_effect_probs.py
# tras la decisión 0002 (microlote 1):
uv run gso train --config configs/noul_overfit.yaml      # runs/noul_overfit/20260922T203606Z
uv run gso train --config configs/noul_smoke.yaml        # runs/noul_smoke/20260922T203613Z
uv run gso evaluate --checkpoint runs/noul_smoke/20260922T203613Z/checkpoint --split validation --no-cache
uv run gso evaluate --checkpoint runs/noul_smoke/20260922T203613Z/checkpoint --split train --no-cache
uv run gso evaluate --checkpoint runs/noul_overfit/20260922T203606Z/checkpoint --split train --no-cache
uv run pytest -rs                                        # 143 passed
uv run python scripts/repro_batch_dependence.py          # vuelve a reproducir el efecto
```

## 8. Límites y lo que no se afirma

- No hay generalización demostrada. Queda pendiente un piloto de unos 1000 casos con plantillas y familias no vistas.
- La métrica de validación tiene sesgo optimista, porque la época se eligió con ella (3 candidatas). No hay calibración: la temperatura corresponde a la fase 3. ECE con 12 muestras no es informativo.
- **Calibración y test siguen intactos.** No usar test hasta tener el artefacto final congelado.
- Choice y Score están contratados, serializados y con métricas, pero no se han entrenado ni se ha probado la pérdida de grupo (fase 2).
- La recarga probada es de un cabezal de etapa A (`phase1_noul_frozen_backbone`) con temperatura 1. No es aún un artefacto de despliegue con calibración y límites.
