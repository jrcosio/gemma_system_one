# 0011 — E4B congelado + cabezales (A4) como servicio de texto recomendado

Fecha: 2026-09-23 · Fase 6 · Estado: aceptada (por regla predeclarada)

## Contexto

La spec §1 fija `google/gemma-4-E4B-it` como candidato «sólo después del perfilado» y la fase 6 exige una «ganancia medida que justifique coste y complejidad». El protocolo `reports/phase6-protocol.md` (sha256 `c745cfea…`, escrito antes de cargar E4B) definió:
- **R1**, ganancia del backbone: A4 − A2;
- **R2**, adopción: A4 − B2, más límites de memoria y de latencia.

Ambas se evalúan con la NLL calibrada de un holdout nuevo de 295 grupos.

## Evidencia

- **Base:** `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`, apache-2.0, sin restricciones de acceso. Descarga única de 15,99 GB con el sha256 LFS verificado. Misma clase `Gemma4Model` y la misma ruta que E2B; hidden 2560 y 42 capas. `gso doctor`: 11/11 pasos en MPS/BF16.
- **A4:** `runs/e4b_experiment/20260923T204945Z`. Los mismos datos, split, plantilla, extracción e hiperparámetros que A2; época 7 de 30 elegida por validación. Calibración `calibration-20260923T210339Z.json` (T: Noul 1,69, Choice 0,71, Score 1,74).
- **Holdout** `data/pilot_v3_holdout6_clean` (885 preguntas, 295 grupos, bootstrap de 1000 repeticiones, semilla 0):

  | Comparación | ΔNLL calibrada [IC95 %] | Δaccuracy |
  |---|---|---|
  | A4 − A2 (R1) | **−0,218 [−0,274; −0,168]** | +0,092 [+0,069; +0,115] |
  | A4 − B2 (R2a) | **−0,182 [−0,236; −0,133]** | +0,085 [+0,061; +0,110] |

- **R2b, memoria:** pico muestreado del driver MPS de 17,1 GB en la extracción (presupuesto de 32 GiB) y swap sin cambio (delta 0).
- **R2c, latencia:** p95 HTTP de A4 1540 ms, frente a B2 909 ms en el mismo turno; el límite era 2 × 909 = 1817 ms.

## Decisión

- A4 es el modelo de texto recomendado para el servicio: `configs/serve_e4b_text.yaml`, `model_id` `gemma-system-one-e4b-heads-v0.1`.
- `configs/serve_text.yaml` (B2, E2B + LoRA) se conserva como opción de menor coste: p50 551 ms frente a 911 ms, arranque en frío de 5,9 s frente a 9,4 s y 1,78 peticiones/s frente a 1,06.
- El contrato público, la plantilla y el backend no cambian.
- El servicio visual sigue en E2B (V2); E4B con imagen no se ha medido.

## Consecuencias y límites

- La ganancia se ha medido en una sola familia sintética (soporte, generador `support-mixed-v3`) con las plantillas de `main`; no demuestra transferencia a datos reales.
- **Calibración de A4:** con sólo 300 preguntas de calibración, sus temperaturas empeoran la NLL del holdout en las tres primitivas (0,174 sin calibrar frente a 0,200 calibrada). La regla predeclarada usaba la calibrada; sin calibrar, A4 − B2 = −0,292 [−0,380; −0,217]. Hace falta una partición de calibración mayor o revisar el método antes de confiar en las probabilidades calibradas de A4.
- **Coste:** extracción 1,8 veces más lenta que E2B (8,5 frente a 15,6 filas/s) y 1,6 veces más latencia de servicio. Sin batching (decisión 0010).
- LoRA sobre E4B (entrenamiento de unas 2,6 h y recomputación obligatoria, decisión 0010) no se ha entrenado. Con A4 por delante de B2, ya no hace falta para superar a B2; sólo se justificaría si mejora a A4 en validación.

## Anexo 2026-09-24: confirmación en un test independiente (fase 6b)

La revisión de la fase 6 señaló que R2 eligió A4 con el mismo holdout con el que se medía. Con el protocolo predeclarado `reports/phase6b-protocol.md` (sha256 `209e9259…`):
- se ajustaron temperaturas nuevas con un conjunto de calibración externo de 1200 preguntas (`pilot_v3_calib7`);
- se evaluó una sola vez en un test final nuevo (`pilot_v3_final8`, 888 preguntas, 296 grupos).

**Resultados:**
- A4 − B2 = **−0,146 [−0,186; −0,104]**: por la regla C, **A4 queda confirmado**.
- El servicio usa ahora la calibración de `calib7`: A4 `calibration-20260924T045005Z.json` y B2 `calibration-20260924T044408Z.json`.
- La NLL calibrada de A4 en `final8` (0,2010) ya no empeora la sin calibrar (0,2021), aunque en Choice la temperatura global sigue empeorando.

Detalle: `reports/phase6b-final.md`. Este anexo sustituye la salvedad sobre la calibración de 300 preguntas.
