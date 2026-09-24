# Fase 6d: diagnóstico de más opciones con composición equilibrada y estados intercambiados

## Revisión independiente (2026-09-24)

El resultado emparejado K8 − K4 sigue siendo válido como medición en estos 167 casos: K8 añade las mismas cuatro distractoras a cada K4 y el límite inferior de los tres IC no cumple el margen de −0,05. **No se demuestra tolerancia a K8.** Las afirmaciones originales de «información 0,000» y «no se explota ninguna pista» se retiran por los dos errores metodológicos siguientes. Los datos y logits históricos no se modifican.

1. `composition_gain = 0,000` es la diferencia de **accuracy top-1** entre dos predictores que sólo ven las opciones, calculada sobre la propia muestra. El predictor llamado `prior` ya restringe su elección a las categorías presentes, por lo que también usa la composición. La ganancia cero no demuestra independencia entre conjunto y etiqueta: una categoría real sólo puede ser respuesta si figura entre las dos elegidas. En las 167 preguntas K4, la información mutua empírica entre firma de categorías y etiqueta completa es 0,422 bits (estimación en la misma muestra, con sesgo por tamaño). El par es uniforme para `other`, `none` y las etiquetas reales **agregadas**, pero no para cada etiqueta real concreta. Esta información incluye disponibilidad legítima de candidatos; no se interpreta automáticamente como fuga indebida. El diseño emparejado K4/K8 controla la **adición** de distractoras fijas, no toda dependencia entre opciones y respuesta.
   El campo `note` del JSON archivado llama a la segunda cifra «accuracy máxima»; es una cifra ajustada en la propia muestra y se conserva sólo por trazabilidad. El script actual aclara la definición para futuros resultados.
2. `swap_states` conserva la pregunta, opciones y etiqueta originales pero sustituye el estado por el de otro grupo. En K4/K8swap, 125/167 etiquetas conservadas (74,9 %) contradicen la respuesta semántica del estado donante; 92/167 intercambios (55,1 %) cambian el idioma. En `final13_faultswap`, las cifras son 99/138 y 70/138. Por tanto, la accuracy frente a las etiquetas conservadas mide una perturbación con estados contradictorios; **no** mide un predictor que sólo lee opciones. En K4swap, la accuracy de A4v3 frente a la etiqueta conservada es 0,210 y frente a la respuesta semántica del estado donante es 0,814 (A4v4: 0,204/0,784; A2v4: 0,222/0,665). En K8swap: A4v3 0,234/0,820, A4v4 0,234/0,743 y A2v4 0,263/0,635. No puede concluirse de este control que los modelos no explotan pistas de opciones.

La comparación con el prior 0,347/0,427 en el control intercambiado queda retirada. Para aislar una pista de opciones haría falta un control que **mantuviera coherencia entre estado y etiqueta** o un predictor que recibiera sólo pregunta y opciones, entrenado y evaluado en grupos disjuntos. Es trabajo futuro con datos nuevos; los conjuntos de esta fase ya están abiertos.

**Reproducción de la revisión:** `.venv/bin/python scripts/review_phase6d_controls.py` recalcula los conflictos, cambios de idioma, información mutua empírica y accuracy frente al estado donante desde los datasets y las predicciones archivadas. `.venv/bin/pytest tests/unit tests/integration -q` → 293 passed; `.venv/bin/pytest tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs` → 1 passed, sin omisión. Una recarga de A4v3 desde `runs/e4b_experiment/20260923T204945Z/checkpoint` ejecutó una pregunta K4 sin caché: `Gemma4Model`, MPS/BF16, eval, cero parámetros base entrenables, sin `lm_head`, cuatro logits finitos y diferencia máxima 0,0 frente al JSONL guardado. Esta comprobación real de E4B es inferencia, no una nueva prueba de sus gradientes. `ruff check .`, `ruff format --check .`, `uv lock --check`, `git diff --check` y el sha256 del protocolo pasan. Las 14 pruebas MPS/E2E de la implementación se conservan como evidencia histórica; no se repitieron en esta revisión.

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. La fase se implementó en `fases-0-6` sobre `daa27ce` y quedó en el commit `9edc183`; la revisión independiente se realiza en `main` con correcciones sin commit.

- **Protocolo predeclarado:** [phase6d-protocol.md](phase6d-protocol.md), sha256 `474a32b4…` (en `reports/phase6d/protocol.sha256`). Se escribió antes de generar los datos y de evaluar.
- **Evidencia bruta:** `reports/phase6d/`.

## Veredicto

1. **La adición de opciones en K8 está controlada, pero no se ha demostrado ausencia de pistas.**
   - `balanced_fault_kind_pairs` usa el mismo par en K4 y K8 para cada pregunta; las cuatro distractoras añadidas son fijas. `composition_gain = 0,000` en 20 000 casos sólo mide una ganancia de accuracy top-1 en la misma muestra, no ausencia de información sobre la etiqueta.
   - El control con estados intercambiados no sirve para concluir ausencia de pistas: en muchos casos la etiqueta guardada contradice el estado donante.
2. **Tolerancia a K = 8: no demostrada para ningún modelo** (criterio: límite inferior del IC de Δaccuracy ≥ −0,05):
   - A4v3: −0,036 [−0,096; +0,024];
   - A4v4: −0,024 [−0,078; +0,030];
   - A2v4: −0,012 [−0,072; +0,054].
   Con 167 preguntas, los IC no descartan pérdidas de hasta 0,08–0,10. La respuesta `other` es el punto débil: con K8, su accuracy baja en todos los modelos.
3. **Composición del generador de entrenamiento v4:**
   - En el probe, la ganancia de accuracy top-1 dentro de la muestra es 0,050 (v3: 0,029). Incluye inferencia legítima: si están todas las categorías reales, `other` es imposible.
   - El resultado de `final13_faultswap` no determina si los modelos aprovechan esa composición. La construcción de un generador v5 y una prueba nueva queda pendiente.

## Diseño

- **`derive.balanced_fault_kind_pairs`:**
  - K4: `none`, `other` y dos categorías reales. Si la respuesta es `other`, ninguna de las dos es la verdadera; si no, una sí.
  - K8: K4 más siempre las cuatro distractoras, que nunca son la respuesta: «hardware», «instalación», «notificaciones» y la nueva «accesibilidad», comprobada por palabras clave; «screen» se evitó porque aparece en los estados.
  - `other` se sortea con probabilidad 0,5 si hay fallo. El par de categorías reales es aproximadamente uniforme de forma marginal, no condicionado a cada etiqueta real.
- **`derive.swap_states`:** cada pregunta recibe el estado de otro grupo (derangement), con las mismas opciones y la misma etiqueta.
- **Tests** (`tests/unit/test_phase6d_balanced_k.py`):
  - respuestas coherentes con los hechos, par marginal aproximadamente uniforme y proporción de `other` ≈ 0,5, con 3000 casos; esto no prueba independencia entre par y cada etiqueta concreta;
  - el intercambio de estados es un derangement determinista.
- **`scripts/probe_option_cue.py`:** dos accuracies top-1 sin leer el estado; la segunda se ajusta y evalúa en la misma muestra y puede sobreajustar (`reports/phase6d/option_cue_probe.json`).

| Diseño (20 000 casos) | Acc. prior restringido a opciones | Acc. por firma en la misma muestra | Ganancia top-1 |
|---|---|---|---|
| Generador v3 (K 3–6) | 0,322 | 0,351 | 0,029 |
| Generador v4 (K 3–6) | 0,427 | 0,477 | 0,050 |
| K8 de la fase 6 (con la etiqueta) | 0,260 | 0,401 | 0,141 |
| K8 de la fase 6c (con los hechos) | 0,275 | 0,294 | 0,019 |
| **Equilibrado K4 / K8 (6d)** | 0,377 | 0,377 | **0,000** |

En conjuntos pequeños, la «máxima» calculada dentro de la propia muestra sobreajusta: `reports/phase6d/option_cue_actual_sets.txt` da, por ejemplo, 0,93 en `final13`, con 120 composiciones para 138 preguntas. La muestra de 20 000 casos reduce ese sesgo, pero no convierte la ganancia top-1 en información mutua ni en una cota poblacional exacta.

## Datos

| Conjunto | Contenido | Preguntas | sha256 |
|---|---|---|---|
| `data/pilot_v4_kdiag14` | v4, seed 14, 400 casos; 12 grupos excluidos por solape con todos los conjuntos anteriores | 1164 | `b0273c33…` |
| `_K4` / `_K8` | `fault_type` equilibradas (58 `other`, 109 resto) | 167 / 167 | `e1a9cd92…` / `bd7af646…` |
| `_K4swap` / `_K8swap` | Las mismas, con estados intercambiados | 167 / 167 | `a498e79d…` / `b7093b49…` |
| `data/pilot_v4_final13_faultswap` | `fault_type` de `final13` con estados intercambiados (control descriptivo) | 138 | `4b6d7e18…` |

Modelos: A4v3 (servicio), A4v4 y A2v4, sin reentrenar, con sus temperaturas de `calib12`.

## Resultados

### Más opciones: K4 frente a K8, emparejado (bootstrap de 167 grupos, 1000 repeticiones, semilla 0)

| Modelo | NLL K4 → K8 | ΔNLL [IC] | **Δaccuracy [IC]** | Acc K4 → K8 | Acc `other` (58) K4 → K8 | Acc resto (109) K4 → K8 |
|---|---|---|---|---|---|---|
| A4v3 | 0,489 → 0,516 | +0,027 [−0,040; +0,104] | **−0,036 [−0,096; +0,024]** | 0,814 → 0,778 | 0,64 → 0,50 | 0,91 → 0,93 |
| A4v4 | 0,666 → 0,762 | +0,096 [−0,020; +0,218] | **−0,024 [−0,078; +0,030]** | 0,766 → 0,743 | 0,48 → 0,43 | 0,92 → 0,91 |
| A2v4 | 1,032 → 0,995 | −0,037 [−0,204; +0,117] | **−0,012 [−0,072; +0,054]** | 0,629 → 0,617 | 0,45 → 0,36 | 0,73 → 0,75 |

- **Tolerancia:** no demostrada según el criterio.
- **Aciertos:** son menores que en los diagnósticos anteriores; A4v3 en `other` tenía 0,71 en `kdiag11_K`. Aquí `none` y `other` están siempre presentes y la composición no ayuda.
- **`other`:** decidir que ninguna opción listada aplica es lo más difícil, y empeora con más distractoras.

### Control con estados intercambiados (mismas opciones y etiquetas, estado de otro grupo)

| Modelo | `K4swap` | `K8swap` | `final13_faultswap` |
|---|---|---|---|
| A4v3 | 0,210 | 0,234 | 0,406 |
| A4v4 | 0,204 | 0,234 | 0,384 |
| A2v4 | 0,222 | 0,263 | 0,377 |
| Predictor por prior restringido a opciones | 0,347 en este conjunto; 0,377 en la muestra grande | ídem | 0,427 en la muestra grande v4 |

- **Lecturas predeclaradas 2 y 3, retiradas:** la comparación contra el prior de las etiquetas originales no aísla una pista. El modelo puede responder coherentemente al estado donante y fallar frente a la etiqueta original. La accuracy 0,78 en `other` tampoco identifica por sí sola la causa del acierto.

## Comandos ejecutados

```bash
.venv/bin/pytest tests/unit/test_phase6d_balanced_k.py -q                       # 2 passed (antes del protocolo)
.venv/bin/python scripts/probe_option_cue.py reports/phase6d/option_cue_probe.json 20000
shasum -a 256 reports/phase6d-protocol.md > reports/phase6d/protocol.sha256
.venv/bin/python scripts/derive_phase6d_data.py
for d in kdiag14 kdiag14_K4 kdiag14_K8 kdiag14_K4swap kdiag14_K8swap final13_faultswap; do .venv/bin/gso validate-data --dataset data/pilot_v4_$d; done
reports/phase6d/run_chain.sh     # 15 × gso evaluate --split all --dataset … --calibration <calib12>
.venv/bin/gso compare --a <K4> --b <K8> --allow-different-inputs --out reports/phase6d/K8_minus_K4_<m>.json
.venv/bin/pytest tests/unit tests/integration -q          # 293 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs # 14 passed, 0 omitidos
```

**Código de las evaluaciones:** `fd340105…` (83 ficheros), con copia en `artifacts/source/`.

**Configuración de ruff:**
- el `.gitignore` corregido en `daa27ce` hizo visibles para ruff los generadores, que nunca se habían revisado;
- se añadió en `pyproject.toml` una excepción E501 y de formato para `generate.py`, `generate_mixed.py` y `generate_vision.py`, para no tocar el código que produce los datasets con hash registrado;
- `split.py` sólo cambió en una línea partida, sin cambio de comportamiento.

## Límites

- Sólo hay 167 preguntas (58 `other`), así que los IC son anchos.
- Una familia sintética y cuatro distractoras fijas.
- El generador de entrenamiento v4 muestra 0,050 de ganancia top-1 en el probe; el uso de esa composición por el modelo queda sin determinar. Un v5 requiere datos y evaluación nuevos.
- `kdiag14*` y `final13_faultswap` ya están usados.
