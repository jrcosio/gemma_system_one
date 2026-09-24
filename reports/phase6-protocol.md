# Protocolo predeclarado de la fase 6: E4B, más opciones y agrupación de filas

Fecha: 2026-09-23. Escrito **antes** de cargar E4B, de ver cualquier salida de un modelo sobre los conjuntos nuevos y de medir la agrupación. Su sha256 se guarda en `reports/phase6/protocol.sha256`. Spec §10, fase 6: «ganancia medida que justifique coste y complejidad». La fase es opcional; un resultado negativo medido también la cierra.

## Datos (ya generados; sólo datos, sin modelos)

| Conjunto | Origen | Preguntas / grupos | sha256 de `examples.jsonl` | Uso |
|---|---|---|---|---|
| `data/pilot_v3` | fase 3, split `seed0` | 3000 / 1000 | `f2519285…9c` | train (ajuste), validation (selección de época y decisiones de ingeniería), calibration (temperaturas). **Su test ya se usó en fase 3: no se vuelve a usar** |
| `data/pilot_v3_holdout6_clean` | `gso generate-data --kind mixed --generator-version v3 --variant main --seed 6 --cases 300`, sin los 5 grupos con estado literal presente en `pilot_v3` | 885 / 295 | `a0129b58…80` | **Test nuevo de fase 6.** Se evalúa una sola vez por modelo, al final |
| `data/pilot_v3_holdout6_faultK` | preguntas `fault_type` del holdout, sin cambios | 135 | `a2c1cc6b…c7` | Más opciones: referencia emparejada |
| `data/pilot_v3_holdout6_faultK8` | las mismas 135, con opciones imposibles añadidas hasta K = 8 (`widen_seed` 0) | 135 | `33be68c6…e8` | Más opciones: diagnóstico. Entrenamiento con K = 3–6 |

Derivación: `scripts/derive_phase6_data.py` (`data/derive.py`, tests `tests/unit/test_phase6_derive.py`).

## Modelos

| Clave | Checkpoint | Estado |
|---|---|---|
| **A2** | E2B congelado + cabezales, `runs/pilot_ce_v3/20260923T005721Z/checkpoint`, calibración `runs/pilot_ce_v3/20260923T005721Z/calibration/` (fase 3) | Existente |
| **B2** | E2B + LoRA + cabezales, `runs/pilot_lora_v3/20260923T012208Z/checkpoint` + `calibration-20260923T043504Z.json` (servido) | Existente |
| **A4** | E4B `ee0ef602…` congelado + cabezales, `configs/e4b_experiment.yaml`: mismos datos, split, semilla, plantilla, extracción (1 fila/forward) e hiperparámetros que A2; época por NLL de validación (1–30); calibración en `calibration` | Nuevo |

LoRA sobre E4B **no se entrena** en este turno: sólo se perfila un paso real (memoria y tiempo). Se entrena en un turno posterior únicamente si se cumple R1 y no R2.

## Métrica y reglas

- **Primaria:** NLL media por pregunta lógica **calibrada** en `pilot_v3_holdout6_clean` (885 preguntas), diferencia emparejada con `gso compare`: bootstrap de 295 grupos, 1000 repeticiones, semilla 0, IC95 %.
- **R1, ganancia del backbone:** A4 − A2 con límite superior del IC < 0 ⇒ «E4B mejora con cabezales congelados». Si el IC contiene 0 ⇒ no demostrada; si el inferior > 0 ⇒ empeora.
- **R2, adopción para el servicio:** A4 sustituye a B2 sólo si (a) A4 − B2 con límite superior < 0, (b) memoria del driver MPS en extracción ≤ 32 GiB (presupuesto de la spec) sin swap nuevo, y (c) p95 HTTP del benchmark de texto (misma batería que `reports/phase5c`: `pilot_v3` validation, 100 + warmup 5) ≤ 2 × el de B2 medido en el mismo turno. Si falla sólo (c), A4 queda como opción documentada, no como servicio por defecto.
- **Secundarias, sin regla:** NLL y accuracy por primitiva; NLL sin calibrar; Brier, ECE, RPS/MAE; validación (sesgada por la selección de época); coste de extracción (filas/s, memoria) y de servicio (p50/p95, arranque).

## Más opciones (diagnóstico, sin regla de adopción)

B2 (y A4 si existe) sobre `faultK` y `faultK8` con las temperaturas de su calibración; `gso compare --allow-different-inputs` (mismas preguntas y etiquetas, más opciones). Se informan ΔNLL y Δaccuracy con IC, accuracy por K y filas por pregunta. Lectura prevista: pérdida pequeña (Δaccuracy ≥ −0,05) ⇒ el scorer compartido tolera K fuera del rango de entrenamiento en esta familia; mayor ⇒ limitación documentada.

## Agrupación de filas por petición (optimización opcional)

Checkpoint B2, partición **validation** de `pilot_v3` (decisión de ingeniería; nunca test), agrupando por caso (las tres preguntas de un grupo, como una petición). Se compara la política vigente (1 fila por forward) con todas las filas de la petición en un forward (padding derecho, orden por longitud, `extract_pooled` con `microbatch_rows` = filas de la petición). Mismas temperaturas.

Se adopta como opción del servicio (no por defecto) sólo si se cumplen **todas**: máx. |Δp| ≤ 0,02 sobre todas las probabilidades publicadas; 0 cambios de decisión (argmax en Choice/Score, umbral 0,5 en Noul); |ΔNLL media| ≤ 0,005; y aceleración del forward sincronizado ≥ 1,5×. Si no, se mantiene la decisión 0002 y se documenta la medida.

## Lo que no se hace

- No se reutiliza el test de `pilot_v3`, ni el de `vision_pilot_v1/v2`, para decidir nada.
- No se ajusta ningún hiperparámetro de A4 distinto de A2; si A4 elige la época tope, no se amplía.
- No se cambian contrato público, plantilla ni backend.
