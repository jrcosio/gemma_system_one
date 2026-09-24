# Informe de fase 4: una imagen, uso visual y límites de memoria

Fecha: 2026-09-23 · Mac M5 Pro (48 GB), E2B `3e22461f…` en MPS/BF16, visión congelada; cabezales en CPU/FP32 · Base git `65d4250`, sin commit. Decisión: [0007](../docs/decisions/0007-single-image-path.md). Protocolo del test: [phase4-test-protocol.md](phase4-test-protocol.md).

## Veredicto de implementación

Revisión independiente posterior: [phase4-review.md](phase4-review.md). Ruta funcional verificada con correcciones; conformidad integral pendiente por estilos compartidos y fuentes de entrenamiento no conservados. Las cifras siguientes son históricas.

**Criterio de salida de la fase 4 cumplido** (spec §10: «evidencia de uso visual y límites de memoria reales»), sobre un banco sintético de paneles de barras.

- **Uso visual en test** (predeclarado; 210 preguntas en 70 grupos; una ejecución por artefacto, desde el texto+imagen, sin caché):

  | Modelo | NLL |
  |---|---|
  | V: E2B congelado + cabezales, **con imagen** | **0,418** [0,307; 0,536] |
  | C: mismas filas **sin imagen** | 1,112 |
  | Prior | 1,069 |
  | BoW | 1,071 |

  - V − C = **−0,694 [−0,816; −0,556]**.
  - Con la imagen de otro grupo, la NLL de V sube a 3,02 (+2,60) y la accuracy baja de 0,829 a 0,386.
  - Se cumplen las dos condiciones de la regla: **hay evidencia de uso visual**.
- **Límites de memoria reales** (MPS, muestreo en puntos instrumentados):

  | Ruta | MPS asignada | Driver MPS | RSS | Swap |
  |---|---|---|---|---|
  | Extracción con imagen para cabezales congelados | 10,2 GB | 11,5 GB | 1,7 GB | 0 |
  | Backward LoRA de una pregunta con imagen y K = 5 | **26,2 GB** (16,0 GB de grafo) | 26,6 GB | — | — |

  Se midió K = 5. **K > 5 no se probó**: la extrapolación para K = 8 ronda 36 GB y podría superar 32 GiB, pero no demuestra un límite exacto en K = 5 (corrección de revisión).
- **Recarga equivalente** con imagen en un proceso nuevo y sin caché: diferencia de logit **0,0** en 210 preguntas y 591 forwards.
- **Límites de la conclusión:**
  - Es un uso visual en gráficos de barras sintéticos generados por reglas; no demuestra comprensión visual general.
  - La transferencia sólo cubre un estilo visual y redacciones reservados.
  - La rúbrica de Score sigue siendo lo más débil: accuracy 0,71 en test.

## 1. Pendientes del piloto resueltos antes de la fase

| Pendiente (revisión de fase 3) | Resolución |
|---|---|
| `tests/mps` sin repetir tras los arreglos de la revisión | 7 passed, 0 omitidos, 58,2 s, con el código revisado |
| Trazabilidad del código (manifiestos con `dirty: true` y sin hash de fuentes) | `git_state()` añade `source.sha256` de `src/`, `configs/`, `scripts/`, `pyproject.toml` y `uv.lock`. Lo heredan checkpoints, entorno y descargas; ahora también `evaluate` y `calibrate`. `scripts/snapshot_source.py` guarda una copia exacta bajo ese hash. **El commit sigue pendiente de decisión del usuario**. Los checkpoints de fase 4 registran `70068e15…` |
| Huella de entrada en las predicciones | `input_sha256` por pregunta (filas + imagen). `compare` exige que coincida salvo `--allow-different-inputs`, que queda registrado |
| Durabilidad de la reanudación | `fsync` de los ficheros, del puntero `LATEST` y del directorio |
| Picos de memoria dentro del paso LoRA | `mps_current_before_backward_max_bytes` por paso (grafo vivo) |

Regresiones en `tests/unit/test_phase3_followups.py`.

## 2. Implementación

| Pieza | Archivo | Contrato |
|---|---|---|
| Imágenes | `images.py` | `images/<sha256>.png\|jpg` (dirección por contenido); PNG/JPEG sin animación; ≤ 5 MiB y ≤ 16 MP leídos en la cabecera; carga verificada contra el hash |
| Validación y fugas | `data/dataset.py`, `data/split.py` | Imágenes validadas una vez por fichero. `model_input_hash` y `serialized_input_hash` incluyen la imagen sólo si existe (hashes de fases 1–3 intactos). La misma imagen en dos particiones es una fuga. Casi duplicados sólo entre ejemplos con la misma imagen |
| Serialización | `serialization.py` | `Row.image`: la huella de la fila incluye la colocación y la imagen. El texto de la fila no cambia (`gso-text-v1`) |
| Procesador | `models/encoding.py` | Imagen antes del texto en el turno de usuario. No mezcla filas con y sin imagen. `image_token_count` usa `mm_token_type_ids == 1` |
| Extracción | `features.py` | Con imagen, una fila por forward (obligatorio) e imagen cargada al procesar su fila. `slice_batch` indexa todo tensor con dimensión de lote (antes sólo los 2D). Registra `image_rows` e `image_tokens` |
| Ítems y control | `training/decisions.py` | `build_items(..., image_root, images="use"\|"omit")`: `omit` quita la imagen dejando el mismo texto. `row_images` |
| Pipeline | `training/decisions_pipeline.py`, `lora*.py` | Modo de imagen en configuración y manifiesto. Robustez que conserva la imagen. `--vision-ablation` (imagen omitida y de otro grupo). LoRA con filas con imagen |
| Datos | `data/generate_vision.py` | `support-vision-v1`: paneles 480×360 con 3–5 servicios, eje 0–100 y umbral opcional. Márgenes que hacen las etiquetas inequívocas; estilo 5 y redacciones reservados para transferencia; `audit.jsonl` |
| Doctor | `doctor.py` | El paso `vision` deja de omitirse: campos, tokens, forward sincronizado, determinismo y sensibilidad a la imagen |
| Config | `configs/e2b_vision.yaml` | Misma base, `max_length: 768` |

## 3. Datos (`gso validate-data`, `gso split`)

| Dataset | Preguntas | Imágenes | Noul / Choice / Score | sha256 (ejemplos) |
|---|---|---|---|---|
| `data/vision_pilot_v1` | 2100 | 700 | 833 / 589 / 678 | `75932114…8b11` |
| `data/vision_transfer_v1` (estilo reservado) | 450 | 150 | 184 / 130 / 136 | `63e9d6b5…e629` |
| `data/vision_smoke_v1` | 180 | 60 | 84 / 48 / 48 | `dda20172…5756` |

- **Reproducibilidad:** mismos sha256 de ejemplos e imágenes con `PYTHONHASHSEED` 1 y 999.
- **Split del piloto:** 490/70/70/70 grupos (1470/210/210/210 preguntas). 0 errores, 0 imágenes compartidas entre particiones, 0 casi duplicados.
- **Filas:** 2,76 por pregunta; 266 tokens visuales por fila; máximo 491 tokens.
- **Verificabilidad:** el estado no contiene valores ni nombres de servicio, y el mismo texto aparece con respuestas distintas según la imagen. Las etiquetas se recalculan desde `audit.jsonl` en los tests.

## 4. Resultados

### 4.1 Puerta de sobreajuste (48 preguntas con imagen)

`runs/vision_overfit/20260923T144159Z`: accuracy 1 y NLL < 3e-4 por primitiva (overfit_check passed).

### 4.2 Validación (210 preguntas, 70 grupos; sesgada por la selección de época)

| Modelo | NLL todas [IC] | Noul NLL / acc | Choice NLL / acc | Score NLL / acc |
|---|---|---|---|---|
| Prior | 1,102 | 0,682 / 0,590 | 1,366 / 0,312 | 1,384 / 0,316 |
| BoW | 1,117 | 0,715 / 0,542 | 1,369 / 0,208 | 1,386 / 0,203 |
| C, sin imagen (época 1) | 1,160 [1,103; 1,216] | 0,825 / 0,446 | 1,413 / 0,229 | 1,357 / 0,316 |
| **V, con imagen (época 12)** | **0,424** [0,331; 0,530] | 0,207 / 0,928 | 0,352 / 0,917 | 0,696 / 0,671 |

**Ablación de V** (`gso evaluate --vision-ablation`, mismo texto; accuracy / NLL):

| Condición | Todas | Noul | Choice | Score |
|---|---|---|---|---|
| Original | 0,829 / 0,424 | 0,928 / 0,207 | 0,917 / 0,352 | 0,671 / 0,696 |
| Imagen omitida | 0,290 / 2,754 | 0,410 / 4,048 | 0,208 / 2,414 | 0,215 / 1,601 |
| Imagen de otro grupo | 0,367 / 2,865 | 0,518 / 1,754 | 0,271 / 5,448 | 0,266 / 2,464 |

**Robustez (V):**

- IDs renombrados y mapa permutado: Δp = 0,0 (48/48).
- Rúbrica invertida: accuracy 0,671 → 0,582.
- Choice sin un distractor: accuracy 0,917 → 0,917.

### 4.3 Test (predeclarado)

| Modelo | NLL todas [IC] | Noul NLL / acc | Choice NLL / acc | Score NLL / acc |
|---|---|---|---|---|
| Prior (train) | 1,069 | — | — | — |
| BoW (train) | 1,071 | — | — | — |
| C, sin imagen | 1,112 [1,041; 1,179] | 0,728 / 0,567 | 1,367 / 0,259 | 1,431 / 0,210 |
| **V, con imagen** | **0,418** [0,307; 0,536] | 0,333 / 0,878 | 0,331 / 0,879 | 0,622 / 0,710 |

**Diferencias emparejadas V − C** (`reports/phase4/test_compare_vision_minus_text.json`):

| Métrica | Todas | Noul | Choice | Score |
|---|---|---|---|---|
| NLL | **−0,694 [−0,816; −0,556]** | −0,395 [−0,576; −0,199] | −1,036 [−1,220; −0,810] | −0,809 [−1,021; −0,577] |
| Accuracy | +0,452 [+0,362; +0,538] | — | — | — |

**Ablación de V en test** (accuracy / NLL):

| Condición | Todas | Noul | Choice | Score |
|---|---|---|---|---|
| Original | 0,829 / 0,418 | — | — | — |
| Imagen omitida | 0,333 / 3,011 | — | — | — |
| Imagen de otro grupo | 0,386 / 3,020 | 0,511 | 0,345 | 0,242 |

**Métricas secundarias de V en test:**

- **Por familia (accuracy):**
  - `compare` 0,943; `top` 1,00; `bottom` 0,794;
  - `any_over` 0,917; `service_over` 0,720;
  - `level` 0,710.
- **Choice por K:** K = 3: 0,958; K = 4: 0,857; K = 5: 0,769.
- **Calibración:** ECE del evento de Noul 0,108; ECE top-label de Choice 0,096; RPS de Score 0,062. Sin calibrar: la calibración no forma parte de esta fase.

### 4.4 Transferencia a estilo reservado (450 preguntas, tema oscuro y redacciones no vistas; diagnóstico)

| Modelo | NLL | Accuracy | NLL Noul / Choice / Score |
|---|---|---|---|
| V | **0,433** | 0,840 | 0,430 / 0,180 / 0,678 |
| C | 1,142 | — | — |
| Prior | 1,103 | — | — |

- Ablación de V: imagen omitida 0,324 de accuracy; imagen de otro grupo 0,402.
- Noul es lo que peor transfiere (0,207 en validación → 0,430).

## 5. Recursos (medidos)

| Medida | Con imagen | Sin imagen |
|---|---|---|
| Extracción de train (4101 filas) | 1016,7 s (≈ 4,0 filas/s; 0,25 s/fila) | 208,9 s (≈ 19,6 filas/s) |
| Tokens por fila (validación y test) | ≈ 450 | ≈ 180 |
| Forward de una fila (doctor) | 0,20 s en caliente, 0,30 s en frío | 0,04 s |
| Extracción + cabezales (pico muestreado) | asignada 10,2 GB; driver 11,5 GB; RSS 1,7 GB; swap 0 | driver 10,3 GB |

**Backward LoRA de una pregunta Choice con K = 5** (`scripts/profile_vision_memory.py`; `reports/phase4/profile_vision_memory.json`):

| Medida | Con imagen | Sin imagen |
|---|---|---|
| Tokens válidos | 2215 | 875 |
| Grafo sobre los pesos | **16,0 GB** (≈ 3,2 GB por fila) | 6,1 GB |
| MPS asignada total | 26,2 GB | — |
| Driver MPS | 26,6 GB | — |
| Forward + backward | 2,34 s | 0,69 s |
| Forward sin gradiente | 1,42 s | — |

**Consecuencia:** con imagen, K = 5 filas en un solo grafo ocupa ~26 GB. Con K = 8 (límite de la API) se estimarían ~36 GB, por encima del presupuesto de 32 GiB. LoRA con imagen requerirá recomputación por fila o gradient checkpointing antes de entrenarse a escala. No se ha implementado ni medido.

Tamaños en disco: `vision_pilot_v1` 8,2 MB; checkpoint de cabezales 80 KB.

## 6. Pruebas

- **CPU:** **250 passed**.
  - `test_phase4_vision_data.py` (9): etiquetas recalculadas desde la auditoría; la respuesta sólo está en la imagen; estilos y redacciones reservados; reproducibilidad entre procesos; validación y split sin fugas; huellas con y sin imagen; imagen antes del texto; rechazo de lotes mixtos; imagen manipulada o de más de 16 MP.
  - `test_phase4_pipeline.py` (3), con Gemma 4 diminuto con torre de visión real y procesador doble que genera 36 parches: representación dependiente de la imagen, microlote rechazado, caché con la imagen en la clave, recarga, ablación, control sin imagen y LoRA con filas con imagen.
  - `test_phase3_followups.py` (3): hash de fuentes, huella de entrada, `--allow-different-inputs` y exposición de las opciones en la CLI.
  - Test de rutas de imagen de la fase 1 actualizado al contrato de dirección por contenido.
- **MPS real:** **10 passed, 0 omitidos, 116 s**. Nuevo `test_phase4_real.py`:
  - filas reales con 266 tokens visuales, imagen antes del texto y sin IDs;
  - sobreajuste, recarga y ablación con E2B;
  - backward LoRA con imagen: gradientes finitos, visión congelada, memoria < 32 GiB.

  `test_doctor_real_e2b` exige ahora el paso de visión.
- **`gso doctor --config configs/e2b_vision.yaml`:** pass en todos los pasos (`reports/doctor/phase4-vision.json`).
- **Fallos detectados durante la fase:**
  - las opciones `--vision-ablation` (evaluate) y `--allow-different-inputs` (compare) no llegaron al parser en el primer intento; ahora hay un test que ejerce la CLI;
  - `compare` rechazaba un control deliberado (desviación 1 del protocolo).

## 7. Comandos ejecutados (en orden)

```bash
caffeinate -i .venv/bin/pytest tests/mps -v -rs                       # 7 passed (código revisado de fase 3)
caffeinate -i .venv/bin/gso doctor --config configs/e2b_vision.yaml --json reports/doctor/phase4-vision.json
shasum -a 256 reports/phase4-test-protocol.md > reports/phase4/test-protocol.sha256   # antes de generar datos
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --out data/vision_pilot_v1 --cases 700 --seed 0
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --variant transfer --out data/vision_transfer_v1 --cases 150 --seed 0
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --out data/vision_smoke_v1 --cases 60 --seed 0
# repetición con PYTHONHASHSEED=999 en el scratchpad: mismos sha256
.venv/bin/gso validate-data --dataset data/vision_pilot_v1      # también vision_transfer_v1 y vision_smoke_v1
.venv/bin/gso split --dataset data/vision_pilot_v1 --seed 0 ; .venv/bin/gso split --dataset data/vision_smoke_v1 --seed 0
caffeinate -i .venv/bin/gso train --config configs/vision_overfit.yaml     # passed
caffeinate -i .venv/bin/gso train --config configs/vision_heads.yaml       # V: runs/vision_heads/20260923T144256Z
caffeinate -i .venv/bin/gso train --config configs/vision_text_only.yaml   # C: runs/vision_text_only/20260923T150230Z
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision_heads/20260923T144256Z/checkpoint --split validation --no-cache --vision-ablation --robustness --baselines
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision_heads/20260923T144256Z/checkpoint --split all --dataset data/vision_transfer_v1 --no-cache --baselines --vision-ablation
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision_text_only/20260923T150230Z/checkpoint --split all --dataset data/vision_transfer_v1 --no-cache
caffeinate -i .venv/bin/pytest tests/mps -v -rs                       # 10 passed
caffeinate -i .venv/bin/python scripts/profile_vision_memory.py configs/e2b_vision.yaml data/vision_pilot_v1 > reports/phase4/profile_vision_memory.json
shasum -a 256 -c reports/phase4/test-protocol.sha256
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision_heads/20260923T144256Z/checkpoint --split test --final-test --no-cache --baselines --vision-ablation
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision_text_only/20260923T150230Z/checkpoint --split test --final-test --no-cache --baselines
.venv/bin/gso compare --a runs/vision_text_only/20260923T150230Z/evaluations/test-20260923T153651Z-predictions.jsonl \
  --b runs/vision_heads/20260923T144256Z/evaluations/test-20260923T153618Z-predictions.jsonl \
  --allow-different-inputs --out reports/phase4/test_compare_vision_minus_text.json
.venv/bin/pytest tests/unit tests/integration -q ; .venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check
```

**Nota posterior (cierre de fase 4, decisión 0008):** estos comandos se ejecutaron cuando `--kind vision` generaba v1 por defecto. Ahora el valor por defecto es v2; para reproducir este piloto hay que añadir `--vision-version v1`, que da los mismos sha256.

Resúmenes en `reports/phase4/`. Informes completos en `runs/vision_*/evaluations/`.

## 8. Límites y riesgos

- **Banco:** gráficos de barras sintéticos con una sola familia visual y un estilo reservado. No hay fotos, documentos ni capturas reales, ni varias imágenes por fila.
- **Score:** la clasificación por tramos del eje es lo más débil (0,71) y la rúbrica invertida cuesta 9 puntos en validación. Noul transfiere peor al estilo oscuro.
- **LoRA con imagen:** probada (autograd con visión congelada), pero K > 5 está sin medir; ~3,2 GB por fila es una extrapolación del caso K = 5, no un límite demostrado.
- **Calibración:** no se calibró V; su ECE en test va de 0,10 a 0,11.
- **Fuente:** la de Pillow no dibuja acentos. Los textos de la imagen van sin ellos, y el título en español es menos natural.
- **Instrucción fija:** la tarea dice «using only the state»; con imagen, el estado incluye la imagen, pero la redacción no lo explicita.
- **Trazabilidad:** el código cambió tras entrenar V y C (CLI, `compare`, informes), con hash `70068e15…` al entrenar. Las evaluaciones de fase 4 se hicieron antes de que `evaluate` registrase el hash; a partir de ahora sí lo registra.
