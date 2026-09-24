# 0004 — Estandarizar las representaciones antes de los cabezales

Fecha: 2026-09-22 · Fase 2 · Estado: aceptada

## Contexto

El primer piloto de la fase 2 (`runs/pilot_ce/20260922T211522Z`, dataset `pilot_v1`, lr 1e-3 y 3 épocas según la spec §7.A) empeoró en la época 3 en validación (NLL 0,448 → 0,585) **y también en train** (0,398 → 0,433). La subida de la pérdida de entrenamiento es compatible con inestabilidad de la optimización, pero por sí sola no demuestra que el paso sea demasiado grande ni excluye sobreajuste: la optimización por microlotes puede fluctuar. La elección se apoya en la comparación empírica siguiente.

## Evidencia

Medido sobre las representaciones cacheadas de E2B en train (5405 filas): norma media por fila 232 (máxima 497), norma L1 media 5454, dimensiones atípicas con |h| medio de hasta 97,6 (mediana 2,8) y desviación típica por dimensión de hasta 29,8 (mediana 2,6).

Con AdamW, cada peso se mueve del orden del lr por paso. Con una L1 de ~5000, un solo paso a lr 1e-3 puede cambiar un logit en unidades. Comparación de 10 épocas, **solo con train y validación** (`scripts/repro_feature_scale.py configs/pilot_ce.yaml data/pilot_v1`; mismas semillas y datos):

| Variante | Curva de NLL en train | Mejor NLL de validación (época) |
|---|---|---|
| Sin normalizar, lr 1e-3 | No monótona: 0,334 → 0,579 → 0,284 | 0,427 (9) |
| Sin normalizar, lr 1e-4 | Monótona | 0,377 (9) |
| **Estandarizada con train, lr 1e-3** | Casi monótona | **0,346 (9)** |
| Estandarizada con train, lr 1e-4 | Monótona, más lenta | 0,411 (10) |

## Decisión

- `DecisionHeads` incorpora `FeatureStandardizer`: `(h − μ)/σ` por dimensión. μ y σ se calculan **solo con las filas de train** (σ ≥ 1e-6) y se guardan como buffers en el checkpoint; no se entrenan. `feature_norm: standardize_train` es el valor por defecto de `DecisionTrainParams`.
- Se mantienen el lr (1e-3) y el weight decay (0,01) de la spec. El máximo pasa a 10 épocas y la época se elige por NLL de validación, como indica la spec §7.A («máximo inicial de 3 épocas, ajustado por validación»).
- El formato de checkpoint `decision_heads` pasa a la versión 3. Los checkpoints v2 (sin estandarizador) se rechazan con un error explícito, sin migración silenciosa. Los runs anteriores quedan como historial.
- El cabezal Noul de la fase 1 (`noul_head`) no cambia.

## Consecuencias y límites

- Se eligió entre 4 variantes mirando la validación, así que las métricas de validación de la fase 2 tienen sesgo optimista. La medición sin sesgo corresponde a test con el artefacto congelado, en la fase 3.
- La estandarización es un escalado afín fijo: no cambia lo que puede representar un cabezal lineal, solo el condicionamiento del problema.
- Con LoRA (fase 3), las representaciones cambian durante el entrenamiento y unas estadísticas fijas de train pueden quedar desfasadas. Allí habrá que medir si conviene recalcularlas o usar una normalización sin estadísticas de datos.
