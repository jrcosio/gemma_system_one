# Fase 6: E4B, más opciones y optimización

Fecha: 2026-09-23. Mac M5 Pro con 48 GB, PyTorch/MPS en BF16 y cabezales CPU/FP32. Sin commit (HEAD `65d4250`).

- **Protocolo predeclarado:** [phase6-protocol.md](phase6-protocol.md), sha256 `c745cfea…`, escrito antes de cargar E4B.
- **Decisiones:** [0010](../docs/decisions/0010-phase6-recompute-and-batching.md) (recomputación y agrupación de filas) y [0011](../docs/decisions/0011-e4b-frozen-heads-text-service.md) (A4 como servicio de texto).
- **Evidencia bruta:** `reports/phase6/`.

## Veredicto

**Fase 6 cerrada según su criterio («ganancia medida que justifique coste y complejidad»), para esta tarea sintética:**

1. **E4B: ganancia medida y adoptada.**
   - En un holdout nuevo (885 preguntas, 295 grupos), E4B congelado con cabezales (A4) mejora la NLL calibrada frente a E2B congelado (A2): **−0,218 [−0,274; −0,168]**. También mejora frente al modelo servido hasta ahora, E2B + LoRA (B2): **−0,182 [−0,236; −0,133]**.
   - Cumple los límites de memoria (17,1 GB) y de latencia (p95 1540 ms ≤ 2 × 909 ms).
   - Por la regla R2, A4 pasa a ser el servicio de texto recomendado.
2. **Más opciones: el evaluador compartido tolera K = 7–8 en accuracy, pero la NLL de B2 empeora.**
   - Diagnóstico con 135 preguntas: Δaccuracy −0,030 en B2 y en A4.
   - ΔNLL: B2 +0,186 [+0,076; +0,310] y A4 +0,027 [−0,084; +0,137].
3. **Optimización:**
   - La **agrupación de filas por petición se rechaza**: Δp de hasta 0,149, 2 decisiones cambiadas y 1,00× de velocidad.
   - La **recomputación de activaciones** se añade como opción: da gradientes idénticos y hace viable LoRA sobre E4B dentro del presupuesto, que sin ella se supera.

## Datos

| Conjunto | Contenido | sha256 (`examples.jsonl`) |
|---|---|---|
| `data/pilot_v3_holdout6` | `generate-data --kind mixed --generator-version v3 --variant main --seed 6 --cases 300` (900 preguntas) | `0f91d9d4…0b19` |
| `data/pilot_v3_holdout6_clean` | Sin los 5 grupos con estado literal presente en `pilot_v3` (mix-s6-00108, 00236, 00248, 00267, 00284) | `a0129b58…6880` |
| `data/pilot_v3_holdout6_faultK` / `_faultK8` | 135 preguntas `fault_type` con sus opciones, y las mismas con opciones imposibles añadidas hasta 8 | `a2c1cc6b…ddc7` / `33be68c6…a5e8` |

- Las opciones añadidas son categorías del generador que no son la real (nunca en los casos `other`) y dos que ningún estado describe («hardware», «instalación»); tests en `tests/unit/test_phase6_derive.py`.
- `evaluate` no detecta fugas frente a `pilot_v3` (`errors: []`), pero sí 9 pares casi duplicados (Jaccard de 3-gramas ≥ umbral) en 8 grupos. Un análisis de sensibilidad posterior, no predeclarado, excluye esos grupos y obtiene lo mismo (287 grupos): A4 − A2 −0,217 [−0,274; −0,161]; A4 − B2 −0,180 [−0,232; −0,129].
- El test de `pilot_v3` (fase 3) no se ha vuelto a leer.

## 1. E4B

### Perfilado

| Paso | E2B | E4B |
|---|---|---|
| Revisión | `3e22461f…` | `ee0ef602…` (apache-2.0, sin restricciones de acceso) |
| Pesos | 10,25 GB | 15,99 GB (descarga única; sha256 LFS verificado) |
| Arquitectura del texto | hidden 1536, 35 capas | hidden 2560, 42 capas (7,94 × 10⁹ parámetros cargados, incluidos audio y visión) |
| `gso doctor` | pasa (fase 0) | **11/11 pasos**; pico muestreado del driver MPS 21,6 GB; presión de memoria máxima de nivel 2 (aviso) |
| Extracción sin gradiente, train (5377 filas) | 338 s (≈ 16 filas/s) | 616 s (8,7 filas/s); driver 17,1 GB; swap delta 0 |
| LoRA, 6 pasos reales (`scripts/profile_lora_step.py`) | 4,78 s/paso; driver 23,3 GB | **supera el presupuesto de 32 GiB** en el paso 3 (34,5 GB del driver, Choice K = 6) |
| LoRA con recomputación | 6,62 s/paso; 19,0 GB; gradientes idénticos (diferencia relativa 0,0) | 11,92 s/paso; 17,4 GB; 132 tensores LoRA con gradiente finito |

El perfilador reproduce el coste real de E2B en la fase 3 (4,5 s/paso). LoRA sobre E4B no se ha entrenado: costaría unas 2,6 h y el protocolo lo reservaba para el caso «R1 sí, R2 no».

### Calidad (NLL media por pregunta)

| Modelo | Validación, calibrado (descriptivo) | **Holdout, calibrado (primaria)** | Holdout, sin calibrar | Noul / Choice / Score (NLL · acc, holdout calibrado) |
|---|---|---|---|---|
| Prior | — | 1,046 | — | — |
| BoW | — | 0,967 | — | — |
| A2: E2B + cabezales | 0,410 | 0,419 | 0,463 | 0,202 · 0,939 / 0,529 · 0,797 / 0,608 · 0,764 |
| B2: E2B + LoRA | 0,375 | 0,383 | 0,466 | 0,172 · 0,941 / 0,471 · 0,822 / 0,588 · 0,756 |
| **A4: E4B + cabezales** | 0,287 | **0,200** | **0,174** | 0,055 · 0,980 / 0,315 · 0,897 / 0,283 · 0,919 |

**Diferencias emparejadas en el holdout** (`gso compare`, bootstrap de 295 grupos, 1000 repeticiones, semilla 0):

| | NLL calibrada | Accuracy | Noul NLL | Choice NLL | Score NLL |
|---|---|---|---|---|---|
| **A4 − A2 (R1)** | **−0,218 [−0,274; −0,168]** | +0,092 [+0,069; +0,115] | −0,148 [−0,226; −0,080] | −0,215 [−0,312; −0,108] | −0,326 [−0,422; −0,239] |
| **A4 − B2 (R2a)** | **−0,182 [−0,236; −0,133]** | +0,085 [+0,061; +0,110] | −0,117 [−0,177; −0,068] | −0,157 [−0,256; −0,062] | −0,305 [−0,412; −0,213] |
| B2 − A2 | −0,036 [−0,062; −0,013] | +0,007 [−0,010; +0,024] | — | — | — |

- **Sin calibrar:** A4 − A2 = −0,289 [−0,373; −0,217] y A4 − B2 = −0,292 [−0,380; −0,217].
- **Validación** (sesgada por la selección de época): A4 − A2 = −0,122 [−0,190; −0,058] y A4 − B2 = −0,088 [−0,156; −0,022].
- **Mejora de LoRA:** B2 − A2 en el holdout confirma la pequeña mejora medida en el test de la fase 3 (−0,075 allí).

**La calibración de A4 empeora el holdout** en las tres primitivas. Las temperaturas se estimaron con 300 preguntas: 1,69 para Noul, 0,71 para Choice y 1,74 para Score. En el holdout, la NLL sube:
- Noul: de 0,044 a 0,055;
- Choice: de 0,295 a 0,315;
- Score: de 0,225 a 0,283.

La regla primaria era la NLL calibrada y se cumple igualmente. Aun así, las probabilidades calibradas de A4 no deben tomarse como mejor calibradas.

### Coste de servicio (misma batería que `phase5c`: `pilot_v3` validation, 5 de warmup + 100 medidas + ráfaga de 7)

| | B2 (`serve_text.yaml`) | A4 (`serve_e4b_text.yaml`) |
|---|---|---|
| Respuestas medidas | 100 × 200 | 100 × 200 |
| HTTP p50 / p95 / máx. | 551 / 909 / 1184 ms | 911 / **1540** / 2043 ms |
| Forward sincronizado p50 / p95 | 529 / 893 ms | 890 / 1520 ms |
| Arranque en frío (cliente) | 5,85 s | 9,37 s |
| Peticiones/s · filas/s | 1,78 · 13,9 | 1,06 · 8,3 |
| Ráfaga de 7 | 5 × 200 + 2 × 503 `queue_full` con identidad del servidor lanzado; `valid: true` | Ídem |
| Log del servidor | 112 peticiones: 110 × 200 y 2 × 503 | Ídem |
| Código cliente = servidor | `2255651d…` (`same_code_as_server: true`) | Ídem |

R2c: 1540 ≤ 2 × 909 = 1817 ms ⇒ se cumple.

## 2. Más opciones (K fuera del rango de entrenamiento)

Se evalúan las 135 preguntas `fault_type` del holdout con sus opciones originales (K 3–6) y con opciones que no pueden ser la correcta añadidas hasta K = 8 (K medio 7,8; filas de 623 a 1054). Temperaturas de cada calibración.

| Modelo | NLL K original → ampliado | ΔNLL [IC] | Δaccuracy [IC] | Elige una opción añadida |
|---|---|---|---|---|
| B2 | 0,455 → 0,641 | **+0,186 [+0,076; +0,310]** | −0,030 [−0,096; +0,037] | 5/135 |
| A4 | 0,249 → 0,276 | +0,027 [−0,084; +0,137] | −0,030 [−0,074; +0,015] | 5/135 |

- **Accuracy por K original, B2:** K3 0,85 → 0,81; K4 0,79 → 0,76; K5 0,75 → 0,80; K6 0,89 → 0,77.
- **Accuracy por K original, A4:** K3 0,96 → 0,96; K4 0,88 → 0,79; K5 0,95 → 0,90; K6 0,89 → 0,91.
- **Lectura prevista** (Δaccuracy ≥ −0,05): se cumple en la estimación puntual de ambos, pero el IC de B2 no descarta pérdidas mayores.
- **B2 pierde calibración** con más opciones: reparte masa entre distractores imposibles. A4 apenas cambia.
- **Límites:** es una sola familia y los distractores son categorías; no son opciones largas ni de otras taxonomías.
- **`compare`:** para este control se admite un número distinto de filas con `--allow-different-inputs` si la etiqueta semántica (`target_description`) coincide. Con el mismo K se sigue exigiendo el mismo índice. Test: `test_comparison_with_more_candidates_requires_same_semantic_label`.

## 3. Optimización

- **Agrupar las filas de una petición en un forward** (B2, validación, 100 peticiones y 781 filas; `scripts/measure_request_batching.py`):
  - resultados: máx. |Δp| 0,149 (p95 0,054), 2 cambios de decisión, ΔNLL −0,0002, 55,5 → 55,4 s (1,00×; 781 → 100 forwards y +19 063 tokens de padding);
  - se rechaza según las tolerancias predeclaradas y se mantiene la decisión 0002.
- **Recomputación de activaciones** (`train.recompute_layers`, decisión 0010):
  - implementación: checkpoint no reentrante por capa del decodificador; sólo el indicador de la capa pasa a entrenamiento y los submódulos siguen en eval;
  - E2B en MPS real: gradientes LoRA idénticos;
  - test CPU: mismo entrenamiento (pesos e historial) con y sin ella;
  - coste: ×1,4 de tiempo por paso en E2B.

## Comandos ejecutados

```bash
.venv/bin/python scripts/snapshot_tree.py                     # referencia del árbol completo: 9f54e5ef… (213 ficheros)
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind mixed --generator-version v3 --variant main --seed 6 --cases 300 --out data/pilot_v3_holdout6
.venv/bin/python scripts/derive_phase6_data.py               # _clean, _faultK, _faultK8
for d in clean faultK faultK8; do .venv/bin/gso validate-data --dataset data/pilot_v3_holdout6_$d; done
shasum -a 256 reports/phase6-protocol.md > reports/phase6/protocol.sha256
.venv/bin/gso download --config configs/e4b_text.yaml        # 15,99 GB, una vez
caffeinate -i .venv/bin/python scripts/measure_request_batching.py --checkpoint runs/pilot_lora_v3/20260923T012208Z/checkpoint \
  --calibration runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json --split validation \
  --out reports/phase6/batching_b2_validation.json
caffeinate -i .venv/bin/gso doctor --config configs/e4b_text.yaml --json reports/phase6/doctor_e4b.json
caffeinate -i .venv/bin/python scripts/profile_lora_step.py configs/e2b_text.yaml data/pilot_v3 reports/phase6/lora_step_e2b.json 6
caffeinate -i .venv/bin/python scripts/profile_lora_step.py configs/e2b_text.yaml data/pilot_v3 reports/phase6/lora_step_e2b_recompute.json 6 --recompute --check-equivalence
caffeinate -i .venv/bin/python scripts/profile_lora_step.py configs/e4b_text.yaml data/pilot_v3 reports/phase6/lora_step_e4b_plain.json 6       # EXIT 2: presupuesto
caffeinate -i .venv/bin/python scripts/profile_lora_step.py configs/e4b_text.yaml data/pilot_v3 reports/phase6/lora_step_e4b_recompute.json 6 --recompute
caffeinate -i .venv/bin/gso train --config configs/e4b_experiment.yaml                        # runs/e4b_experiment/20260923T204945Z
caffeinate -i .venv/bin/gso calibrate --checkpoint runs/e4b_experiment/20260923T204945Z/checkpoint --split calibration
reports/phase6/run_evals.sh                                   # 10 evaluaciones: validación, holdout y K (ver el script)
.venv/bin/gso compare --a <A2|B2 holdout> --b <A4 holdout> --out reports/phase6/holdout_compare_a4_minus_{a2,b2}.json   # y b2−a2, validación
.venv/bin/gso compare --a <faultK> --b <faultK8> --allow-different-inputs --out reports/phase6/faultK8_minus_faultK_{b2,a4}.json
caffeinate -i .venv/bin/gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation --requests 100 --warmup 5 --out reports/phase6/benchmark_b2.json
caffeinate -i .venv/bin/gso benchmark --config configs/serve_e4b_text.yaml --dataset data/pilot_v3 --split validation --requests 100 --warmup 5 --out reports/phase6/benchmark_a4.json
.venv/bin/pytest tests/unit tests/integration -q               # 287 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs      # 14 passed, 0 omitidos
```

Además se hicieron un análisis de sensibilidad posterior (`reports/phase6/sensitivity/`) y las comparaciones sin calibrar (`reports/phase6/uncalibrated/`), a partir de los ficheros de predicciones.

**Código usado en cada paso** (todas las huellas tienen copia en `artifacts/source/`):

| Paso | Huella |
|---|---|
| Agrupación | `a847836a…` |
| Perfilado de E2B | `c81b4b94…` |
| Perfilado con recomputación y E4B | `cfde84ae…` |
| Entrenamiento de A4 | `cfde84ae…` |
| Evaluaciones | `241ccb25…` |
| Benchmarks | `2255651d…` |

Entre las dos últimas sólo cambió `metrics.compare_predictions`, que las evaluaciones no usan.

## Límites

- Los datos son sintéticos, de una sola familia de tareas y de las plantillas de `main`. La ganancia de E4B no está demostrada fuera de ella.
- La calibración de A4 se estima con 300 preguntas y no generaliza al holdout.
- La memoria es la de instantáneas muestreadas, no un pico exacto del driver.
- LoRA E4B y E4B con imagen no se han medido. Tampoco hay abstención ni batching.
