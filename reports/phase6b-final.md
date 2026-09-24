# Fase 6b: test final independiente del servicio elegido y calibración de A4

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. Sin commit (HEAD `65d4250`).

- **Protocolo predeclarado:** [phase6b-protocol.md](phase6b-protocol.md), sha256 `209e9259…` en `reports/phase6b/protocol.sha256`. Se escribió antes de generar los conjuntos y de ejecutar los modelos sobre ellos.
- **Evidencia bruta:** `reports/phase6b/`.

## Veredicto

Se cierran los pendientes metodológicos de la revisión de la fase 6:

1. **A4 confirmado como servicio de texto (regla C).**
   - Resultado: en un test final nuevo (`pilot_v3_final8`, 888 preguntas, 296 grupos), A4 − B2 = **−0,146 [−0,186; −0,104]** de NLL calibrada. No se usó para ajustar pesos ni temperaturas; la regla C predeclarada sí usa este test para confirmar o revertir la recomendación.
   - Lectura: el holdout de la fase 6 queda como conjunto de selección; esta es la medida independiente posterior a la elección.
2. **Calibración de A4: resuelta según el criterio declarado, pero de forma marginal.**
   - Datos: temperaturas ajustadas con 1200 preguntas externas (`pilot_v3_calib7`).
   - Resultado en `final8`: NLL 0,2010 frente a 0,2021 sin calibrar. Las T de 300 preguntas empeoraban la NLL (0,2089).
   - Salvedad: en Choice, la T nueva (1,28) sigue empeorando la NLL (0,279 → 0,296).
   - En A2 y B2, la calibración con `calib7` mejora claramente (0,414 → 0,384 y 0,412 → 0,347).
3. **Perfilador LoRA con la corrección del revisor, ejercitado en MPS:** 6/6 pasos de E4B con recomputación; 132 tensores LoRA con gradiente presente y finito en cada paso; 11,69 s/paso; driver 17,4 GB; swap delta 0.

## Datos

| Conjunto | Generación | Preguntas / grupos | sha256 |
|---|---|---|---|
| `data/pilot_v3_calib7` | seed 7, 400 casos; 0 grupos excluidos por solape con `pilot_v3` o `holdout6` | 1200 / 400 | `cdf1005c…2abd` |
| `data/pilot_v3_final8` | seed 8, 300 casos; excluidos 4 grupos por solape con `pilot_v3`, `holdout6` o `calib7` (mix-s8-00090, 00092, 00152, 00243) | 888 / 296 | `d9f1b2fb…bf39` |

- `evaluate` y `calibrate` no encontraron fugas frente a `pilot_v3` (`errors: []`).
- `final8` tiene 4 pares casi duplicados con `pilot_v3`. Se declaran; no hubo análisis adicional.

## Implementación

- **`gso calibrate --split all --dataset <conjunto>`** (calibración externa):
  - exige que no haya grupos ni entradas compartidos con el dataset de entrenamiento;
  - el artefacto registra `split: external_calibration` y el sha256 del conjunto;
  - Tras la revisión de 2026-09-24, `evaluate` rechaza también un conjunto distinto que comparta grupos o entradas con la calibración externa y comprueba su hash original.
  - Test: `tests/integration/test_phase3_pipeline.py::test_external_calibration_set_is_bound_checked_and_never_reused_as_test`.
- **Datos:** `scripts/derive_phase6b_data.py` reutiliza `data/derive.exclude_overlap`.
- **Servicio:** `configs/serve_e4b_text.yaml` (A4) y `configs/serve_text.yaml` (B2) usan ahora las calibraciones de `calib7`, como fijaba el protocolo.

## Resultados en `pilot_v3_final8` (evaluación única por modelo)

### Temperaturas

| Modelo | T de 300 preguntas (Noul / Choice / Score) | T de `calib7` (1200 preguntas) |
|---|---|---|
| A2 | 1,74 / 1,50 / 1,49 | 1,71 / 1,36 / 1,54 |
| B2 | 2,00 / 1,87 / 1,64 | 2,00 / 1,91 / 1,90 |
| A4 | 1,69 / **0,71** / **1,74** | 1,70 / **1,28** / **0,96** |

### NLL y accuracy (calibrado con `calib7`)

| Modelo | NLL (sin calibrar) | Noul NLL · acc | Choice NLL · acc | Score NLL · acc |
|---|---|---|---|---|
| Prior / BoW | 1,032 / 0,942 | — | — | — |
| A2 | 0,384 (0,414) | 0,181 · 0,926 | 0,501 · 0,813 | 0,573 · 0,789 |
| B2 | 0,347 (0,412) | 0,146 · 0,939 | 0,446 · 0,856 | 0,553 · 0,763 |
| **A4** | **0,201** (0,202) | 0,091 · 0,963 | 0,296 · 0,901 | 0,263 · 0,890 |

### Diferencias emparejadas (bootstrap de 296 grupos, 1000 repeticiones, semilla 0)

| | NLL calibrada | Accuracy | Noul NLL | Choice NLL | Score NLL |
|---|---|---|---|---|---|
| **A4 − B2 (regla C)** | **−0,146 [−0,186; −0,104]** | +0,057 [+0,035; +0,082] | −0,055 [−0,110; −0,010] | −0,150 [−0,223; −0,086] | −0,290 [−0,383; −0,192] |
| A4 − A2 | −0,183 [−0,227; −0,141] | +0,070 [+0,046; +0,096] | −0,090 [−0,154; −0,039] | −0,205 [−0,291; −0,134] | −0,310 [−0,407; −0,207] |
| B2 − A2 | −0,038 [−0,066; −0,012] | +0,012 [−0,007; +0,029] | −0,035 [−0,083; +0,004] | −0,055 [−0,096; −0,017] | −0,020 [−0,068; +0,031] |

Las magnitudes son algo menores que en el holdout de selección (A4 − B2: −0,182 allí, −0,146 aquí), como cabe esperar tras seleccionar. El signo y la conclusión se mantienen.

**Límite detectado en la revisión posterior:** el generador v3 de `fault_type` limita a cinco opciones los ejemplos cuya respuesta es `other`, aunque el sorteo de K pide seis. En `final8`, K6 en esa familia aparece 32 veces y ninguna tiene la etiqueta `other`. Esta pista estructural limita la interpretación fuera de estos datos sintéticos. Un análisis posterior, excluyendo las 141 preguntas `fault_type` y conservando los 296 grupos, da A4 − B2 = −0,131 con bootstrap por grupos [−0,175; −0,088] (1000 repeticiones, semilla 0). Es sensibilidad descriptiva hecha después de mirar el test, no una nueva prueba independiente ni una regla de selección. Los datasets v3 existentes se conservan para reproducibilidad.

### Variantes de calibración en `final8` (NLL media; sin decidir nada con ello)

| Modelo | Sin T | T de 300 | T de `calib7` |
|---|---|---|---|
| A2 | 0,4137 | 0,3860 | 0,3842 |
| B2 | 0,4120 | 0,3482 | 0,3467 |
| A4 | 0,2021 | 0,2089 | **0,2010** |

Por primitiva en A4:
- Noul: 0,107 → 0,091 con las dos T.
- Choice: 0,279 sin T → 0,287 con T de 300 → 0,296 con T de `calib7`.
- Score: 0,263 → 0,305 con T de 300 → 0,263 con T de `calib7`.

Una temperatura global por primitiva no es claramente útil en Choice para A4. Queda como límite; no se ha ajustado nada con `final8`.

### Servicio A4 con la calibración de `calib7`

| Medida | Resultado |
|---|---|
| Respuestas medidas | 100 × 200 (`pilot_v3` validation, 5 de warmup) |
| HTTP p50 / p95 | 910 / 1546 ms |
| Arranque en frío | 9,5 s |
| Ráfaga | 5 × 200 + 2 × 503 atribuidos; `valid: true` |
| Log del servidor | 112 peticiones |
| Identidad | `checkpoint_id` `decision_heads:97fbb4d1…+cal:f4213101…`; código cliente = servidor `a7f64500…` |

## Comandos ejecutados

```bash
shasum -a 256 reports/phase6b-protocol.md > reports/phase6b/protocol.sha256          # antes de generar datos
for s in 7:400 8:300; do PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind mixed --generator-version v3 --variant main \
  --seed ${s%%:*} --cases ${s##*:} --out data/pilot_v3_seed${s%%:*}; done
.venv/bin/python scripts/derive_phase6b_data.py                                      # calib7 y final8
.venv/bin/gso validate-data --dataset data/pilot_v3_calib7 ; .venv/bin/gso validate-data --dataset data/pilot_v3_final8
reports/phase6b/run_chain.sh      # 3 × gso calibrate --split all --dataset data/pilot_v3_calib7;
                                  # 3 × gso evaluate --split all --dataset data/pilot_v3_final8 --calibration <calib7> --baselines;
                                  # scripts/profile_lora_step.py configs/e4b_text.yaml data/pilot_v3 … 6 --recompute
.venv/bin/gso compare --a <final B2> --b <final A4> --out reports/phase6b/final_compare_a4_minus_b2.json   # y a4−a2, b2−a2
caffeinate -i .venv/bin/gso benchmark --config configs/serve_e4b_text.yaml --dataset data/pilot_v3 --split validation \
  --requests 100 --warmup 5 --out reports/phase6b/benchmark_a4_calib7.json
.venv/bin/pytest tests/unit tests/integration -q          # 288 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs # 14 passed, 0 omitidos
```

- Las variantes de calibración (`reports/phase6b/calibration_variants_final8.json`) se calcularon sin volver a usar el modelo, a partir de los `row_logits` de las predicciones.
- Código de calibraciones, evaluaciones y perfil: `9c31e85b…` (78 ficheros). Del benchmark: `a7f64500…`, que sólo añade el cambio de calibración en los YAML de servicio. Ambos con copia en `artifacts/source/`.

## Límites

- Los datos son sintéticos, de una sola familia y de las plantillas de `main`. `final8` es independiente de las decisiones, no de la distribución.
- La calibración de Choice en A4 no mejora con temperatura global.
- LoRA sobre E4B completo, E4B con imagen y abstención siguen sin hacerse.
- `final8` ya está usado: no debe servir para ajustar nada nuevo.
