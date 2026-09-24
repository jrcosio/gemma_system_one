# Informe de fase 3: LoRA, checkpoints, calibración y evaluación ciega

Fecha: 2026-09-23 · Mac M5 Pro (48 GB), E2B `3e22461f…` en MPS/BF16, LoRA y cabezales en FP32 · Base git `65d4250` con cambios sin commit. Decisiones: [0005](../docs/decisions/0005-lora-stage.md) (etapa LoRA) y [0006](../docs/decisions/0006-generator-v3-access-policy.md) (datos v3). Protocolo del test: [phase3-test-protocol.md](phase3-test-protocol.md).

## Veredicto

**Criterio de salida de la fase 3 cumplido** (spec §10: «comparación con cabezales congelados y recarga equivalente»), sobre datos sintéticos:

- **Comparación ciega en test** según un protocolo predeclarado (sha256 `a84f9220…`, escrito antes de ver la validación del run LoRA). NLL calibrada: LoRA + cabezales **0,345** frente a cabezales congelados **0,420**. Diferencia emparejada **−0,075 [−0,110; −0,039]** (IC95 % por bootstrap de 100 grupos). Por la regla predeclarada: **LoRA mejora**.
- **Recarga equivalente** desde el texto en un proceso nuevo y sin caché: diferencia máxima de logit **0,0** en los dos artefactos (300 preguntas y 781 forwards en validación cada uno) y en la puerta LoRA (48 preguntas).
- **Checkpoints separados**: despliegue (adaptador + cabezales, 5,2 MB, sin pesos base) y reanudación (optimizador, scheduler, RNG, posición del sampler). La reanudación es bit a bit en CPU (test) y se usó de verdad en MPS: el piloto se interrumpió dos veces y se reanudó.
- **Calibración** por temperatura sólo sobre `calibration`, con artefacto vinculado por sha256.

**Límites que condicionan la lectura:**

- En **validación** la misma comparación no fue concluyente: −0,002 de NLL, IC [−0,064; +0,075]. Con 100 grupos, validación y test pueden discrepar.
- **Sin calibrar**, la mejora en test tampoco es concluyente: −0,058 [−0,119; +0,008].
- En **transferencia** (plantillas no vistas, sólo diagnóstico) LoRA no mejora: +0,021 [−0,013; +0,058]. Mejora Choice, empeora Noul, y Score calibrado sigue peor que el prior en ambos modelos.
- No es un resultado de calidad general. Los datos son sintéticos, de repertorio finito, y la transferencia sólo cubre superficie.

## 1. Fallos del piloto resueltos antes de entrenar LoRA

| Fallo (fase 2) | Resolución | Evidencia |
|---|---|---|
| Ambigüedad «cuenta bloqueada» ≠ «fallo técnico»: la regla etiquetaba así, pero la instrucción no lo decía | Generador **v3**: cláusula explícita en las tres familias de fallo técnico. Misma secuencia aleatoria que v2: sólo cambian esas instrucciones | `tests/unit/test_phase3_data.py`. Mismo sha256 con `PYTHONHASHSEED` 1 y 999. Mismos grupos por split que v2 |
| `near_duplicate_states` ignoraba `limit` | Aplica el límite y devuelve el total. Sólo al truncar añade `near_duplicate_states_total`, así que los manifiestos existentes siguen idénticos | `gso split` sobre `pilot_v2` y `mixed_smoke_v2`: «ya existía idéntico» |
| La época elegida en v2 fue el máximo (10) | La referencia v3 usa un tope de 30 y selección por validación | Época 28 |

Efecto medido de v3, con el mismo tope de 10 épocas (diagnóstico `pilot_ce_v3_e10`, validación): NLL 0,438 [0,357; 0,519] frente a 0,445 [0,340; 0,548] en v2. Con cuenta bloqueada, los errores cambian en ambos sentidos (`fault_severity` 6/23 → 4/23; `fault_type` 2/14 → 4/14). **La ambigüedad queda resuelta en la especificación de la tarea; no hay mejora medible.**

## 2. Implementación

| Pieza | Archivo | Contrato |
|---|---|---|
| LoRA | `models/lora.py` | Objetivos por nombre exacto y tipo `nn.Linear` en `language_model.layers.*.self_attn.{q,v}_proj`: **35 q_proj + 15 v_proj** en E2B y 1 339 392 parámetros entrenables. `peft.inject_adapter_in_model` (0.21.0); pesos FP32; inicialización con semilla propia. Base congelada y en `eval()`; sólo el dropout de LoRA en `train` |
| Bucle | `training/lora.py` | Una fila por forward con autograd, pérdida de grupo completo y backward por pregunta. AdamW LoRA 1e-4 / cabezales 5e-4; warmup 5 %, decaimiento lineal, clip 1,0. Evaluación por época con la extracción compartida (cabezales en CPU/FP32). Reanudación atómica. Vaciado de la caché MPS cada 10 pasos |
| Pipeline | `training/lora_pipeline.py` | `gso train` con `kind: lora_decision_heads`, `--resume` y `--stop-after-steps`. Arranca de cabezales de fase 2 del mismo dataset, split y huella; la época 0 es exactamente ese modelo |
| Checkpoint | `checkpoint.py` | `lora_decision_heads` v1: `adapter.safetensors` + `head.safetensors` + manifiesto (base, huella, módulos, PEFT, dataset, split, época). Sha256 de ambos ficheros verificados |
| Evaluación | `training/decisions_pipeline.py` | `prepare_evaluation` común a fases 2 y 3 (con LoRA, sin caché). `--calibration` vinculada, `--baselines` ajustados en train y evaluados en la partición pedida. Predicciones con logits crudos, T, NLL calibrada y sin calibrar |
| Calibración | `calibration.py` | `gso calibrate --split calibration` (única partición admitida). T = softplus(t) + 0,01 por primitiva, LBFGS sobre la NLL; con menos de 30 preguntas, T = 1 y `calibrated: false`. Artefacto inmutable ligado al sha256 del manifiesto, de los pesos y del adaptador |
| Comparación | `metrics.compare_predictions`, `gso compare` | Diferencia emparejada b − a por pregunta (NLL y acierto), con bootstrap de grupos |

## 3. Pruebas

- **CPU:** `pytest tests/unit tests/integration` → **232 passed**.
- **Nuevos tests CPU de la fase:**
  - `test_lora.py`, con Gemma 4 diminuto: objetivos exactos y rechazo de módulos no `Linear`; base congelada y adaptadores FP32; identidad con B = 0; gradientes reales (B en el paso 1, A en el 2); base intacta; pérdida y gradiente con microlote igual que con una fila (FP32); scheduler; checkpoint con detección de manipulación; temperatura recuperada (T = 2 y 0,5 sintéticas); mínimo de preguntas; inicialización con semilla.
  - `test_phase3_pipeline.py`: época 0 igual a los cabezales de partida; **reanudación bit a bit** tras interrumpir; pasos perdidos apartados; configuración distinta rechazada; recarga; robustez; calibración sólo en `calibration` y rechazada con otro checkpoint; test protegido; baselines; comparación; puerta con sólo LoRA.
  - `test_phase3_data.py`: v3 = v2 + cláusula; límite de casi duplicados.
- **MPS real:** `pytest tests/mps -v -rs` → **7 passed, 0 skipped, 47,6 s**. Nuevo `test_phase3_real.py` con E2B:
  - 35 + 15 módulos;
  - identidad exacta con B = 0 en BF16;
  - autograd de grupo con gradientes finitos;
  - pérdida que baja en 3 pasos;
  - base intacta;
  - adaptador recargado idéntico.
- **Microlotes con autograd en E2B/MPS** (`scripts/repro_lora_microbatch.py`, 12 preguntas): |ΔL| hasta 0,194 y gradiente con hasta 16,6 % de diferencia relativa frente a una fila por forward. **Se mantiene una fila por forward** (decisión 0002).
- **Puerta de sobreajuste con sólo LoRA** (cabezales fijos, 48 preguntas): NLL de train 0,401 → 0,00016 y accuracy 1 por primitiva. Interrumpida en el paso 20 y reanudada. Recarga con 0,0 de diferencia.
- `ruff check`, `ruff format --check` y `uv lock --check`: correctos.

## 4. Datos y artefactos

| Artefacto | Detalle |
|---|---|
| `data/pilot_v3` | 3000 preguntas / 1000 grupos, sha256 `f2519285…9352`; split 700/100/100/100 grupos |
| `data/pilot_transfer_v3` | 600 preguntas, sha256 `a99c1844…c78f`; 0 fugas, 0 casi duplicados con el piloto |
| `data/mixed_smoke_v3` | 180 preguntas, sha256 `4d4365c0…ff46` |
| A: `runs/pilot_ce_v3/20260923T005721Z/checkpoint` | Cabezales congelados, época 28; calibración `calibration/calibration-20260923T043407Z.json` |
| B: `runs/pilot_lora_v3/20260923T012208Z/checkpoint` | LoRA + cabezales, época 1 de 0–3, 789 pasos; calibración `calibration/calibration-20260923T043504Z.json` |
| `runs/mixed_heads_v3/20260923T011017Z`, `runs/lora_overfit_v3/20260923T011059Z` | Punto de partida y puerta LoRA |
| `runs/pilot_ce_v3_e10/20260923T010432Z` | Diagnóstico v2 frente a v3 (no seleccionado) |

## 5. Entrenamiento LoRA del piloto (validación, 300 preguntas / 100 grupos)

| Época | NLL val (todas / Noul / Choice / Score) | Pérdida train (media de pasos) | Deriva estandarizada (media \|μ\| / σ media) |
|---|---|---|---|
| 0 (= A) | 0,4238 / 0,159 / 0,570 / 0,663 | — | 0,06 / 0,98 |
| **1 (elegida)** | **0,4222** / 0,182 / 0,596 / 0,601 | 0,200 | 0,17 / 1,17 |
| 2 | 0,4365 / 0,153 / 0,554 / 0,730 | 0,152 | 0,37 / 1,44 |
| 3 | 0,4303 / 0,128 / 0,492 / 0,804 | 0,087 | 0,50 / 1,61 |

- La pérdida de train baja mientras la validación de Score empeora: **sobreajuste a partir de la época 2**.
- El 77 % de los pasos recortó el gradiente (mediana de la norma: 4,3).
- **Diferencia B − A en validación:** −0,002 [−0,064; +0,075]. Está sesgada, porque la validación eligió la época.
- **Robustez en validación (B):**
  - IDs renombrados y mapa permutado: 84/84 filas idénticas, Δp = 0,0.
  - Rúbrica invertida: accuracy 0,742 → 0,708; NLL 0,601 → 0,791.
  - Choice sin un distractor: accuracy 0,774 → 0,786; predicción sin cambios en el 75 %.

## 6. Calibración (partición `calibration`, 300 preguntas; métricas in-sample)

| Artefacto | T Noul / Choice / Score | NLL antes → después (in-sample) |
|---|---|---|
| A | 1,744 / 1,497 / 1,490 | 0,440 → 0,406 |
| B | 2,003 / 1,870 / 1,635 | 0,406 → 0,345 |

Ambos modelos son sobreconfiados (T > 1 en todas las primitivas); B más que A.

## 7. Test ciego (300 preguntas / 100 grupos; una ejecución por artefacto; desde el texto, sin caché)

| Modelo | NLL todas | Noul NLL / acc | Choice NLL / acc | Score NLL / acc |
|---|---|---|---|---|
| Prior (train) | 1,048 | 0,655 / 0,647 | 1,432 / 0,147 | 1,144 / 0,367 |
| BoW (train) | 0,965 | 0,536 / 0,714 | 1,323 / 0,422 | 1,150 / 0,418 |
| A sin calibrar | 0,458 | 0,264 / 0,899 | 0,518 / 0,794 | 0,674 / 0,747 |
| A calibrado | 0,420 [0,342; 0,500] | 0,199 / 0,899 | 0,518 / 0,794 | 0,625 / 0,747 |
| B sin calibrar | 0,400 | 0,233 / 0,933 | 0,392 / 0,873 | 0,664 / 0,797 |
| **B calibrado** | **0,345** [0,275; 0,418] | 0,159 / 0,933 | 0,386 / 0,873 | 0,573 / 0,797 |

**Diferencias emparejadas B − A** (calibrados; `reports/phase3/test_compare_lora_minus_frozen.json`):

| Métrica | Diferencia [IC95 %] |
|---|---|
| NLL, todas (**primaria**) | **−0,075 [−0,110; −0,039]** |
| Accuracy, todas | +0,053 [+0,023; +0,083] |
| NLL Noul | −0,040 [−0,082; +0,007] |
| NLL Choice | −0,132 [−0,212; −0,056] |
| NLL Score | −0,053 [−0,118; −0,001] |
| NLL sin calibrar, todas | −0,058 [−0,119; +0,008] |

Frente a BoW, A queda en [−0,626; −0,463] de NLL y B en [−0,697; −0,536].

**Métricas secundarias:**

| Métrica | A sin calibrar → calibrado | B sin calibrar → calibrado |
|---|---|---|
| Noul: Brier | 0,075 → 0,066 | 0,057 → 0,051 |
| Noul: ECE del evento | 0,091 → 0,077 | 0,059 → 0,051 |
| Choice: Brier (suma) | 0,279 → 0,274 | 0,207 → 0,206 |
| Choice: ECE top-label | 0,076 → **0,100** | 0,078 → **0,126** |
| Score: RPS | 0,105 → 0,102 | 0,093 → 0,089 |
| Score: MAE | 0,393 → 0,433 | 0,343 → 0,374 |
| Score: ECE acumulativa | 0,085 → 0,103 | 0,076 → 0,078 |

- **Degradaciones de la calibración:** mejora NLL y Brier en ambos modelos, pero empeora el ECE top-label de Choice y el MAE de Score (el MAE sube porque la distribución se aplana). La temperatura optimiza la NLL, no el ECE.
- **Por familia (accuracy, A → B):**
  - `service_fault` 0,579 → 0,737;
  - `routing` 0,769 → 0,865;
  - `fault_severity` 0,879 → 0,948;
  - `service_impact` 0,381 → 0,381 (la comparación numérica de proporciones sigue sin resolverse).
- **Por M en Score (B):** M = 3: 0,905 (n = 63); M = 4: 0,3 (n = 10); M = 5: 0,5 (n = 6).

## 8. Transferencia (600 preguntas, plantillas reservadas; diagnóstico)

| Modelo | NLL todas (calibrado) | Noul | Choice | Score NLL / acc |
|---|---|---|---|---|
| Prior | 0,988 | 0,618 | 1,414 | 1,157 / 0,364 |
| A | 0,818 | 0,267 | 0,847 | 1,778 / 0,470 |
| B | 0,840 | 0,317 | 0,779 | 1,852 / 0,536 |

B − A: NLL +0,021 [−0,013; +0,058]; accuracy +0,027 [+0,005; +0,050]; Noul +0,050 [+0,015; +0,087]; Choice −0,068 [−0,111; −0,029].

En Score ambos modelos quedan peor que el prior en NLL incluso calibrados: una temperatura global no corrige la sobreconfianza fuera de distribución.

## 9. Recursos (medidos)

| Recurso | Medida |
|---|---|
| Entrenamiento LoRA | 789 pasos; mediana 4,5 s/paso (p95 13,9 s); 75 min sumando pasos sincronizados; 3,6 filas/s y 840 tokens/s con backward |
| Extracción sin gradiente (fase 2) | 16,6 filas/s |
| Evaluación por época | 781 filas en 54–132 s |
| Memoria asignada MPS | ~10,2 GB (pesos) |
| Memoria del driver MPS por paso, con vaciado cada 10 pasos | Mediana 22,6 GB, máximo 25,6 GB |
| RSS máximo muestreado | 3,4 GB |
| Swap | 0 |

**Incidencias:**

- **Caché del asignador MPS:** sin vaciado, el driver llegó a ~30 GB con el sistema en presión de memoria (0,5 GB libres, compresor con 8 GB). Vaciar en cada paso costaba un 35 % del hilo principal. Solución: cada 10 pasos (decisión 0005).
- **Reposo del Mac:** entró 148 veces en reposo de mantenimiento mientras entrenaba. El tiempo de reloj de la tercera sesión (2 h 05 min para 589 pasos) no es tiempo de cómputo. Lanzar los entrenamientos largos con `caffeinate -i`.
- **Checkpoint de despliegue:** 5,2 MB. Estado de reanudación: 21 MB.

## 10. Comandos ejecutados (en orden)

```bash
uv add "peft>=0.21"                                          # peft 0.21.0 + accelerate 1.15.0; torch/transformers sin cambios
PYTHONHASHSEED=1 uv run gso generate-data --kind mixed --generator-version v3 --out data/pilot_v3 --cases 1000 --seed 0
PYTHONHASHSEED=1 uv run gso generate-data --kind mixed --generator-version v3 --variant transfer --out data/pilot_transfer_v3 --cases 200 --seed 0
PYTHONHASHSEED=1 uv run gso generate-data --kind mixed --generator-version v3 --out data/mixed_smoke_v3 --cases 60 --seed 0
# repetición con PYTHONHASHSEED=999 en el scratchpad: sha256 idéntico
uv run gso validate-data --dataset data/{pilot_v3,pilot_transfer_v3,mixed_smoke_v3}     # 0 errores (tres invocaciones)
uv run gso split --dataset data/pilot_v3 --seed 0 ; uv run gso split --dataset data/mixed_smoke_v3 --seed 0
uv run gso split --dataset data/pilot_v2 --seed 0            # «ya existía idéntico» (también mixed_smoke_v2)
uv run gso train --config configs/pilot_ce_v3.yaml           # A: runs/pilot_ce_v3/20260923T005721Z
uv run gso train --config configs/pilot_ce_v3_e10.yaml       # diagnóstico v2/v3
uv run pytest tests/mps/test_phase3_real.py -v -rs           # 1 passed
uv run python scripts/repro_lora_microbatch.py configs/e2b_text.yaml data/pilot_v3 12 > reports/phase3/repro_lora_microbatch.json
uv run gso train --config configs/mixed_heads_v3.yaml
uv run gso train --config configs/lora_overfit_v3.yaml --stop-after-steps 20
uv run gso train --config configs/lora_overfit_v3.yaml --resume runs/lora_overfit_v3/20260923T011059Z   # overfit_check passed
uv run gso evaluate --checkpoint runs/lora_overfit_v3/20260923T011059Z/checkpoint --split train          # recarga 0,0
# protocolo del test escrito y hasheado (reports/phase3/test-protocol.sha256)
uv run gso train --config configs/pilot_lora_v3.yaml                                               # sesión 1, parada en el paso 185
uv run gso train --config configs/pilot_lora_v3.yaml --resume runs/pilot_lora_v3/20260923T012208Z  # sesiones 2 y 3
uv run gso evaluate --checkpoint runs/pilot_lora_v3/20260923T012208Z/checkpoint --split validation --robustness
uv run gso calibrate --checkpoint runs/pilot_ce_v3/20260923T005721Z/checkpoint --split calibration
uv run gso calibrate --checkpoint runs/pilot_lora_v3/20260923T012208Z/checkpoint --split calibration
caffeinate -i uv run gso evaluate --checkpoint <A> --split test --final-test --no-cache --calibration <cal A> --baselines
caffeinate -i uv run gso evaluate --checkpoint <B> --split test --final-test --calibration <cal B> --baselines
uv run gso compare --a <pred. test A> --b <pred. test B> --out reports/phase3/test_compare_lora_minus_frozen.json
caffeinate -i uv run gso evaluate --checkpoint <A|B> --split all --dataset data/pilot_transfer_v3 [--no-cache] --calibration <cal> --baselines
uv run gso compare --a <pred. transfer A> --b <pred. transfer B> --out reports/phase3/transfer_compare_lora_minus_frozen.json
caffeinate -i uv run gso evaluate --checkpoint <A> --split validation --no-cache          # recarga 0,0
uv run pytest tests/unit tests/integration -q ; caffeinate -i uv run pytest tests/mps -v -rs   # 232 + 7 passed
```

Los resúmenes de CLI están en `reports/phase3/*.json`. Los informes completos, en `runs/*/evaluations/` y `runs/*/calibration/`.

## 11. Límites y riesgos

- **Datos:** sintéticos y de repertorio finito. Test de 100 grupos: IC anchos. Validación y test discrepan en la conclusión sobre LoRA. **Test ya está usado**: cualquier cambio de diseño necesita un test nuevo, por ejemplo otra semilla.
- **Sobreajuste LoRA:** desde la época 2, con 2100 preguntas. No se exploraron hiperparámetros (lr, r, épocas) para no sesgar; la selección de la época 1 limita el daño.
- **Estandarizador fijo:** con deriva creciente. No se midió la alternativa de recalcularlo.
- **Calibración:** una T por primitiva mejora la NLL pero empeora el ECE de Choice y el MAE de Score. No corrige la transferencia de Score. Umbrales de abstención sin implementar (spec §8, fase 5).
- **Reproducibilidad:** en MPS no se ha verificado bit a bit una ejecución continua frente a otra reanudada, ni el efecto de `empty_cache` sobre los cálculos. En CPU sí coinciden.
- **Sin probar:** FP16, gradient checkpointing, K > 6, contexto > 512 con autograd, imagen, E4B.
