# Informe de fase 2: Choice y Score dinámicos, pérdida de grupo y piloto

**Revisión posterior (2026-09-23):** [hallazgos, correcciones y repetición real](revision-fase-2.md). El informe siguiente conserva sus métricas originales.

Fecha: 2026-09-22 · Mac M5 Pro (48 GB), E2B congelado en MPS/bf16, cabezales en CPU/FP32 · Base git `65d4250` con cambios sin commit.

## Veredicto

**Puerta de fase 2 cumplida** (spec §10: «tests de semántica, pérdida de grupo y orden; piloto reproducible»):

- **Semántica:** permutar el mapa de opciones y renombrar los IDs deja filas idénticas y probabilidades remapeadas exactas (Δp = 0,0 en 84/84 preguntas reales de validación). Cambiar la pregunta sobre el mismo estado cambia la fila y la respuesta; en validación, la exactitud de pares Noul del mismo estado es 1,00. Tests unitarios y de integración.
- **Pérdida de grupo:** softmax y entropía cruzada sobre las K o M filas completas. Pérdida y gradiente idénticos con y sin trozos, y la normalización por trozo se rechaza. Acumulación = Σ w_tipo·L_q / n_preguntas, incluido el paso parcial (tests en `tests/unit/test_decisions.py`).
- **Orden:** la rúbrica conserva su orden (tests de serialización y de rúbrica invertida). La robustez a la orientación se midió con el modelo real (§4.3).
- **Piloto reproducible:**
  - Generador `support-mixed-v2` determinista entre procesos: mismo sha256 con `PYTHONHASHSEED` 1 y 999, y test de regresión.
  - Split por grupos sin fugas.
  - Entrenamiento con procedencia completa.
  - Recarga desde el texto en un proceso nuevo con diferencia de logit **0,0** en 300 preguntas.
- **Sobreajuste:** 48 preguntas mixtas con accuracy = 1 y NLL < 0,05 **por primitiva** (NLL ≤ 2,1e-4).

**No es un resultado de calidad general.** Los datos son sintéticos y proceden de un repertorio finito de frases. La validación (300 preguntas, 100 grupos) se usó para elegir la época, la normalización (decisión 0004) y λ, así que sus cifras tienen sesgo optimista. El conjunto de transferencia solo reserva plantillas de superficie, no tareas nuevas. **Calibración y test siguen sin usarse.** No hay calibración: todas las temperaturas valen 1.

## 1. Implementación

| Pieza | Archivo | Contrato |
|---|---|---|
| Evaluadores compartidos | `models/heads.py` (`DecisionHeads`, `FeatureStandardizer`) | Tres evaluadores lineales distintos (Noul, Choice, Score), cada uno con un logit por fila. Ninguna salida está ligada a un ID o a una posición. Estandarización fija calculada con train (decisión 0004) |
| Pérdidas y bucle | `training/decisions.py` | BCE (Noul), CE de grupo completo (Choice), CE + λ·RPS (Score, λ = 0 por defecto). `type_weights` declarados (1/1/1). Los logits pueden calcularse por trozos, pero la normalización es siempre sobre el grupo |
| Reconstrucción | `inference.py` | Noul `p = σ(z)`. Choice: probabilidades por ID opaco y `choice` = argmax. Score: `score = Σ m·p_m` y `legend` = rúbrica en su orden. `confidence` = concentración `normalized_entropy_v1`. Comprueba que las probabilidades suman 1, que la opción pertenece al conjunto, que el score está en rango y que no hay NaN |
| Métricas | `metrics.py` | Choice: NLL, Brier (suma sin /K), accuracy, ECE top-label y concentración, por K. Score: NLL, RPS, MAE bruto y normalizado, accuracy y ECE de eventos acumulados, por M. Todo por familia de tarea |
| Baselines | `baselines.py` | Prior de Laplace (Noul), 1/K uniforme (Choice) y prior de nivel por M (Score). BoW hasheado: logística (Noul) y puntuador por fila con softmax de grupo (Choice y Score). Todos ajustados solo con train y con hiperparámetros fijos |
| Pipeline | `training/decisions_pipeline.py`, `cli.py` | `gso train` (con `kind: decision_heads`) y `gso evaluate` con `--robustness` y `--dataset` externo (`--split all`), que comprueba fugas contra el dataset de entrenamiento. Informa `usage` con `generated_tokens = 0` |
| Checkpoint | `checkpoint.py` | `decision_heads` v3: los tres evaluadores más el estandarizador, sin pesos base. Guarda procedencia, huella, dataset, split, IDs de train, época y temperaturas (1, `calibrated: false`). Los v2 se rechazan |
| Datos | `data/generate_mixed.py` | Choice: tipo de fallo (K = 3–6, con `other` si la categoría real no está y `none` si no hay fallo) y enrutado por política con prioridades (K = 4). Score: alcance por proporción de usuarios (M = 3–5) y severidad (M = 3). IDs opacos aleatorios. Variante `transfer` con plantillas reservadas. v2: rúbricas ascendentes o descendentes |
| Validación y fugas | `data/dataset.py`, `data/split.py` | Duplicados **semánticos** por hash de las filas serializadas (IDs renombrados, orden del mapa o de claves). Casi duplicados con **todos** los estados de cada grupo. Cierra los dos riesgos que dejó abierta la revisión de la fase 1 |

## 2. Datos (evidencia: `gso validate-data`, `gso split`)

| Dataset | Preguntas | Grupos | Noul / Choice / Score | sha256 |
|---|---|---|---|---|
| `data/pilot_v2` | 3000 | 1000 | 1320 / 919 / 761 | `5de5cb7b…be71` |
| `data/pilot_transfer_v2` (plantillas reservadas) | 600 | 200 | 272 / 177 / 151 | `05251b2c…1e72` |
| `data/mixed_smoke_v2` (sobreajuste) | 180 | 60 | 67 / 62 / 51 | `d7d2223c…f1e4` |

- **Split de `pilot_v2`:** 700 / 100 / 100 / 100 grupos (2100 / 300 / 300 / 300 preguntas). Sin errores de fuga; 5 pares de casi duplicados (Jaccard ≥ 0,8), que quedan como aviso. Entre el piloto y transferencia no hay grupos ni entradas compartidos, ni casi duplicados.
- **Filas con el procesador real:** máximo 322 tokens, media 226; ninguna supera 512.

## 3. Fallos del piloto y cómo se resolvieron

| ID | Fallo | Reproducción | Resolución |
|---|---|---|---|
| P2-1 | **Inestabilidad de optimización.** Con lr 1e-3 y 3 épocas (spec), la NLL de train **subía** en la época 3 (0,398 → 0,433) y la de validación también (0,448 → 0,585) | `scripts/repro_feature_scale.py configs/pilot_ce.yaml data/pilot_v1` | Estandarización con estadísticas de train. Máximo 10 épocas con selección en validación (**decisión 0004**) |
| P2-2 | **El generador no era reproducible entre procesos.** `_opaque_ids` ordenaba un `set` de `str`, cuyo orden depende de `PYTHONHASHSEED`, y la asignación ID → opción cambiaba. `pilot_v1` es un fichero válido, pero no regenerable | Regenerar `pilot_v1` daba otro sha256; `test_mixed_generation_is_reproducible_across_processes` | Lista en orden de generación. Hash idéntico con `PYTHONHASHSEED` 1 y 999 |
| P2-3 | **Atajo posicional en Score.** Con rúbricas siempre ascendentes (v1), invertir la rúbrica bajaba la accuracy de 0,776 a 0,635 y subía la NLL de 0,498 a 1,178 | `gso evaluate … --robustness` sobre `runs/pilot_ce/20260922T212358Z` | Generador v2 con orientación mezclada (la etiqueta sigue a la descripción). Después: 0,663 → 0,618 y NLL 0,763 → 0,856 |
| P2-4 | Duplicados en generación (estados literalmente iguales entre grupos) | `validate-data`: «entrada duplicada en grupos distintos» | Estados únicos por dataset. El validador lo detectó antes de entrenar |
| — | Cosmético: mayúscula tras un saludo con coma | Inspección | Corregido en v2 (v1 se conserva igual) |

Se revisaron manualmente **~90 predicciones**: en v1, los 50 errores de validación y 25 aciertos; en v2, los errores de Score. No encontré etiquetas incorrectas respecto a las reglas.

Sí encontré una **ambigüedad de política** («cuenta bloqueada» no cuenta como «fallo técnico» según la regla, y el modelo sí lo cuenta). Queda pendiente aclararla en la instrucción de una versión posterior del generador (spec §5.2).

## 4. Resultados del piloto v2

Configuración: `configs/pilot_ce.yaml` (lr 1e-3, wd 0,01, 8 preguntas por paso, estandarización, máximo 10 épocas, λ = 0). Run de referencia: **`runs/pilot_ce/20260922T212800Z`**, época elegida 10 (el máximo).

### 4.1 Validación (300 preguntas, 100 grupos; IC95 % por bootstrap de grupos, 1000 réplicas, semilla 0)

| Modelo | NLL todas [IC] | Noul NLL / acc [IC acc] | Choice NLL / acc [IC acc] | Score NLL / acc [IC acc] |
|---|---|---|---|---|
| Prior | 1,034 [0,977; 1,086] | 0,656 / 0,646 | 1,429 / 0,131 | 1,200 / 0,292 |
| BoW | 0,935 [0,863; 0,999] | 0,487 / 0,772 | 1,371 / 0,381 | 1,162 / 0,438 |
| **E2B + cabezales** | **0,445 [0,340; 0,548]** | **0,160 / 0,945** [0,901; 0,978] | **0,538 / 0,810** [0,725; 0,895] | **0,763 / 0,663** [0,568; 0,755] |

- **Diferencia emparejada de NLL frente a BoW:** [−0,589; −0,396].
- **Noul:** Brier 0,043, ECE del evento 0,059, F1 0,921. Pares del mismo estado: 1,00 (BoW 0,78; prior 0,50).
- **Choice:** Brier (suma) 0,273, ECE top-label 0,112, concentración media 0,69.

  | K | n | NLL | Accuracy | NLL uniforme |
  |---|---|---|---|---|
  | 3 | 13 | 0,829 | 0,769 | 1,099 |
  | 4 | 47 | 0,551 | 0,787 | 1,386 |
  | 5 | 13 | 0,495 | 0,846 | 1,609 |
  | 6 | 11 | 0,194 | 0,909 | 1,792 |

- **Score:** RPS 0,141, MAE 0,513 (normalizado 0,216) y ECE de eventos acumulados 0,114.

  | M | n | NLL | Accuracy |
  |---|---|---|---|
  | 3 | 67 | 0,502 | 0,776 |
  | 4 | 15 | 1,377 | 0,467 |
  | 5 | 7 | 1,945 | 0,000 |

- **Por familia:** Noul `service_fault` 0,815 (el más débil); Choice `fault_type` 0,837 y `routing` 0,780; Score `fault_severity` 0,828 y `service_impact` 0,355 (comparación numérica de proporciones).
- **CE + RPS (λ = 1, `runs/pilot_rps/20260922T213423Z`):** NLL 0,455 frente a 0,445, sin mejora. En v1 la diferencia emparejada de Score fue [−0,011; +0,006] de NLL. **Se elige λ = 0.**
- **Curva de validación:** ruidosa (0,540 → 0,458 → 0,511 → 0,445) y la mejor época es la última. No se ajustaron más hiperparámetros con validación para no aumentar el sesgo. En la fase 3 la spec prevé warmup y decaimiento lineal.

### 4.2 Recarga desde el texto (proceso nuevo, `--no-cache`)

`gso evaluate --checkpoint runs/pilot_ce/20260922T212800Z/checkpoint --split validation --no-cache --robustness`:

- Backbone cargado; 300 preguntas, 781 filas, 781 forwards, 180 849 tokens procesados, **0 generados**.
- **Diferencia máxima de logit: 0,0** (tolerancia 1e-4).

### 4.3 Robustez (validación, modelo real)

| Prueba | Resultado |
|---|---|
| IDs renombrados y mapa permutado | 84/84 filas idénticas; Δp remapeado = **0,0** |
| Rúbrica invertida (v2) | Accuracy 0,663 → 0,618; NLL 0,763 → 0,856 (en v1: 0,776 → 0,635 y 0,498 → 1,178) |
| Choice sin un distractor | Accuracy 0,810 → 0,774; NLL 0,538 → 0,431; predicción sin cambios en el 79,8 % |

### 4.4 Transferencia (plantillas de superficie no vistas; diagnóstico, no se usó para seleccionar)

`gso evaluate … --split all --dataset data/pilot_transfer_v2`: sin fugas y 0 casi duplicados. 600 preguntas, 1496 filas.

| Modelo | NLL todas | Noul acc | Choice acc | Score acc (NLL) |
|---|---|---|---|---|
| Prior | 0,988 | 0,691 | 0,136 | 0,364 (1,157) |
| BoW | 0,993 | 0,691 | 0,158 | 0,318 (1,153) |
| **E2B + cabezales** | **0,862** | **0,875** | **0,706** | 0,523 (**1,764**) |

- BoW cae al nivel del prior: dependía de la superficie.
- Los cabezales E2B generalizan mejor, pero pierden bastante frente a validación.
- En Score son **sobreconfiados**: su NLL es peor que la del prior aunque su accuracy sea mayor. Es un argumento para la calibración y LoRA de la fase 3; no es una garantía de que lo resuelvan.

## 5. Recursos (medidos)

- **Extracción** en MPS/bf16, una fila por forward y sin padding:
  - train: 5377 filas y 1 205 993 tokens en 324,8 s (≈ 16,6 filas/s);
  - validación: 781 filas en 48,4 s.
- **Run completo del piloto:** 6 min 23 s.
- **Por pregunta:** una pregunta Choice o Score cuesta K o M forwards; de media, 2,6 filas por pregunta en el piloto.
- **Memoria:**
  - MPS asignada: 10,22 GB.
  - Driver MPS: máximo muestreado de 11,36 GB.
  - RSS del proceso: máximo 2,57 GB (representaciones de train en CPU).
  - Swap: 0.
  - El muestreo se hace en puntos instrumentados; no es un pico exacto.
- **Cabezales:** se entrenan en CPU con representaciones cacheadas; los runs con caché tardan unos 6 s.

## 6. Pruebas

- **Total:** `uv run pytest` → **197 passed, 0 skipped** (unit 175, integration 16, mps 6). `ruff check` y `ruff format --check` sin errores; `uv lock --check` y `git diff --check` correctos.
- **Nuevos en la fase 2:**
  - `tests/unit/test_decisions.py`: pérdida de grupo con y sin trozos, rechazo de la normalización por trozo, RPS, acumulación ponderada con paso parcial, peso cero, remapeo por ID, invarianza de la reconstrucción, rúbrica invertida, cabezales ausentes sin tocar, baselines de grupo y estandarizador (ajuste solo con train, persistencia, rechazo de v2).
  - `tests/unit/test_phase2_data_metrics.py`: métricas calculadas a mano, fronteras de las reglas de alcance, etiquetas iguales a las reglas, IDs opacos, uso de `other`, plantillas de transferencia disjuntas, duplicados semánticos, casi duplicados con todos los estados, reproducibilidad entre procesos y orientación v2.
  - `tests/integration/test_phase2_pipeline.py`: E2B diminuto con procesador doble; train → evaluate con recarga, robustez, dataset externo y fugas, test protegido y sobreajuste por primitiva.
  - `tests/mps/test_phase2_real.py`: filas Choice y Score con el procesador real (≤ 512 tokens, sin IDs opacos, lectura tras el turno del modelo); sobreajuste mixto y recarga desde el texto con E2B real en MPS.

## 7. Comandos ejecutados (en orden, resumidos)

```bash
uv run gso generate-data --kind mixed --out data/mixed_smoke_v1 --cases 60 --seed 0           # v1 (historial)
uv run gso generate-data --kind mixed --out data/pilot_v1 --cases 1000 --seed 0                 # v1 (historial)
uv run gso generate-data --kind mixed --variant transfer --out data/pilot_transfer_v1 --cases 200 --seed 0
uv run gso split --dataset data/mixed_smoke_v1 --seed 0 ; uv run gso split --dataset data/pilot_v1 --seed 0
uv run gso train --config configs/mixed_overfit.yaml   # v1: passed (runs/mixed_overfit/20260922T211456Z)
uv run gso train --config configs/pilot_ce.yaml        # v1, sin normalizar, 3 épocas: P2-1 (runs/pilot_ce/20260922T211522Z)
uv run python scripts/inspect_predictions.py runs/pilot_ce/20260922T211522Z/predictions_validation.jsonl data/pilot_v1 25
# diagnóstico de escala (ahora scripts/repro_feature_scale.py) y decisión 0004
uv run gso train --config configs/mixed_overfit.yaml ; uv run gso train --config configs/pilot_ce.yaml ; uv run gso train --config configs/pilot_rps.yaml   # v1 + 0004
uv run gso evaluate --checkpoint runs/pilot_ce/20260922T212358Z/checkpoint --split validation --no-cache --robustness   # P2-3
# generador v2 + corrección P2-2; hash comprobado con PYTHONHASHSEED=1 y 999
uv run gso generate-data --kind mixed --generator-version v2 --out data/mixed_smoke_v2 --cases 60 --seed 0
uv run gso generate-data --kind mixed --generator-version v2 --out data/pilot_v2 --cases 1000 --seed 0
uv run gso generate-data --kind mixed --generator-version v2 --variant transfer --out data/pilot_transfer_v2 --cases 200 --seed 0
uv run gso split --dataset data/mixed_smoke_v2 --seed 0 ; uv run gso split --dataset data/pilot_v2 --seed 0
uv run gso train --config configs/pilot_ce.yaml        # runs/pilot_ce/20260922T212800Z (referencia)
uv run gso train --config configs/pilot_rps.yaml       # runs/pilot_rps/20260922T213423Z
uv run gso train --config configs/mixed_overfit.yaml   # runs/mixed_overfit/20260922T213430Z (puerta v2)
uv run gso evaluate --checkpoint runs/pilot_ce/20260922T212800Z/checkpoint --split validation --no-cache --robustness
uv run gso evaluate --checkpoint runs/pilot_ce/20260922T212800Z/checkpoint --split all --dataset data/pilot_transfer_v2 --no-cache
uv run python scripts/transfer_baselines.py configs/pilot_ce.yaml data/pilot_transfer_v2
uv run python scripts/repro_feature_scale.py configs/pilot_ce.yaml data/pilot_v1                # reproduce la decisión 0004
uv run pytest -rs                                                                               # 197 passed
```

## 8. Límites

- Datos sintéticos con reglas y repertorio finito. La transferencia solo cubre plantillas de superficie; no hay familias ni tareas nuevas.
- Las métricas de validación tienen sesgo optimista (se eligieron época, normalización y λ con ellas). La época elegida es la última del máximo.
- Sin calibración: Score es sobreconfiado en transferencia. **Calibración y test sin usar.**
- Los cabezales son lineales sobre el backbone congelado. Las tareas con composición de prioridades o comparación numérica (enrutado, alcance) son las más débiles, y queda un atajo parcial de orientación en Score (−4,5 puntos al invertir la rúbrica).
- Coste lineal en K o M forwards por pregunta con una fila por forward (decisión 0002). No se ha medido el batching en servicio.
- La memoria se muestrea en puntos instrumentados; no se ha medido una latencia de benchmark.
