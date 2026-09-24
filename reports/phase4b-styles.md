# Cierre de fase 4: estilos visuales separados por partición y trazabilidad de fuentes

Fecha: 2026-09-23 · E2B `3e22461f…` en MPS/BF16, visión congelada, cabezales CPU/FP32 · Sin commit. Decisión [0008](../docs/decisions/0008-vision-styles-per-split-and-source-archives.md). Protocolo [phase4b-test-protocol.md](phase4b-test-protocol.md), sha256 `05901ce5…fb95`, escrito antes de generar los datos.

## Veredicto

Se resuelven los dos pendientes que dejó abiertos la revisión independiente ([phase4-review.md](phase4-review.md)):

1. **Estilos separados por partición (spec §5.1): cumplido.**
   - Con el generador `support-vision-v2`, train usa los estilos 0, 1, 2, 3, 5 y 6; validation, 7 y 8; calibration, 9 y 10; test, 11 y 12. Comprobado cruzando el split real con `audit.jsonl`, y `gso split` verifica el reparto planificado.
   - En el **test nuevo** (estilos 11–12, nunca vistos en train):

     | Modelo | NLL |
     |---|---|
     | V2 (con imagen) | **0,415** [0,307; 0,526] |
     | C2 (mismas filas sin imagen) | 1,192 |
     | Prior | 1,085 |

     V2 − C2 = **−0,778 [−0,886; −0,660]**. Con la imagen de otro grupo, la NLL sube a 3,47 y la accuracy baja de 0,848 a 0,352.
   - Se cumplen las dos condiciones de la regla: **hay evidencia de uso visual con estilos no vistos**, dentro de la misma familia de gráficos de barras.
2. **Trazabilidad de fuentes: resuelta hacia delante, con un límite histórico.**
   - Cada inicio de run y cada checkpoint guardan ahora `artifacts/source/<sha256>.tar`. También lo hacen `evaluate` y `calibrate`, desde después de estas evaluaciones.
   - V2 y C2 lo demuestran: `7f645568…` al empezar V2, que es el código ejecutado, y `78d02416…` al guardar V2 y para C2 entero. Las evaluaciones de validación, test y transferencia registran `78d02416…`, que tiene copia.
   - **Límite:** la copia de `70068e15…` (V y C del piloto v1) no existe. No se fabricó ni se renombró otra.

El test del piloto v1 no se reutilizó ni se modificó. El piloto v1 sigue siendo reproducible byte a byte: la misma secuencia aleatoria da los mismos sha256 de ejemplos e imágenes.

## Datos

| Dataset | Preguntas | Imágenes | sha256 (ejemplos) | Notas |
|---|---|---|---|---|
| `data/vision_pilot_v2` | 2100 | 700 | `09b1d190…f604` | Split 490/70/70/70 grupos, 0 fugas, plan `3e4b7c98…`. Mismo hash con `PYTHONHASHSEED` 1 y 999 |
| `data/vision_transfer_v2` | 450 | 150 | `f7958bdb…29d8` | Estilo 4 y redacciones reservados; diagnóstico |

## Resultados

### Validación (estilos 7–8; sesgada por la selección)

| Modelo | NLL [IC] | Época elegida | Noul / Choice / Score (acc) |
|---|---|---|---|
| Prior | 1,092 | — | — |
| C2 | 1,121 [1,053; 1,178] | 1 | — |
| V2 | **0,338** [0,253; 0,438] | 30, el tope | 0,910 / 0,965 / 0,750 |

- **Recarga desde el texto+imagen** en proceso nuevo: Δlogit **0,0** (210 preguntas).
- **Ablación** (accuracy):

  | Condición | Accuracy |
  |---|---|
  | Original | 0,876 |
  | Imagen omitida | 0,343 |
  | Imagen de otro grupo | 0,362 |

- **Robustez:**
  - IDs renombrados y mapa permutado: Δp = 0,0 (57/57).
  - Rúbrica invertida: accuracy 0,750 → 0,719.
  - Sin un distractor: 0,965 → 0,895.

### Test (estilos 11–12; predeclarado)

| Modelo | NLL | Noul NLL / acc | Choice NLL / acc | Score NLL / acc |
|---|---|---|---|---|
| C2 | 1,192 | 0,881 / 0,429 | 1,368 / 0,262 | 1,432 / 0,197 |
| **V2** | **0,415** | 0,342 / 0,881 | 0,190 / 0,923 | 0,753 / 0,721 |

**V2 − C2 (NLL):**

| Primitiva | Diferencia [IC95 %] |
|---|---|
| Noul | −0,539 [−0,698; −0,345] |
| Choice | −1,178 [−1,324; −1,012] |
| Score | −0,679 [−0,910; −0,395] |
| Accuracy, todas | +0,538 [+0,476; +0,605] |

**Ablación en test (accuracy / NLL):**

| Condición | Todas | Noul | Choice | Score |
|---|---|---|---|---|
| Imagen omitida | 0,229 / 3,36 | 0,321 | 0,200 | 0,131 |
| Imagen de otro grupo | 0,352 / 3,47 | 0,500 | 0,215 | 0,295 |

**Por estilo de test:**

| Estilo | Preguntas | Accuracy | NLL |
|---|---|---|---|
| 11 | 126 | 0,881 | 0,332 |
| 12 | 84 | 0,798 | 0,539 |

El estilo 12 (horizontal, barras de un solo color, umbral punteado) cuesta más.

**Por familia (accuracy):** `compare` 0,956; `bottom` 0,941; `top` 0,903; `any_over` 0,867; `service_over` 0,750; `level` 0,721. Score por tramos sigue siendo lo más débil.

**Comparación descriptiva** con el V de fase 4 (estilos vistos; sin regla): NLL 0,418 frente a 0,415 de V2 con estilos no vistos. Son tests distintos y no emparejados.

### Transferencia (estilo 4; diagnóstico)

| Modelo | NLL | Accuracy |
|---|---|---|
| V2 | 0,363 | 0,842 |
| C2 | 1,145 | — |
| Prior | 1,092 | — |

Con la imagen de otro grupo, la accuracy de V2 baja a 0,367.

### Recursos

| Medida | Con imagen (V2) | Sin imagen (C2) |
|---|---|---|
| Extracción de train (4067 filas) | 1003 s (≈ 4,1 filas/s) | 203 s |
| Driver MPS | 11,4 GB | 10,3 GB |
| RSS | 1,6 GB | — |
| Swap | 0 | — |

## Comandos ejecutados

```bash
shasum -a 256 reports/phase4b-test-protocol.md > reports/phase4b/test-protocol.sha256      # antes de generar
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --vision-version v2 --out data/vision_pilot_v2 --cases 700 --seed 0 --split-seed 0
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --vision-version v2 --variant transfer --out data/vision_transfer_v2 --cases 150 --seed 0
# repetición con PYTHONHASHSEED=999 en el scratchpad: mismos sha256
.venv/bin/gso validate-data --dataset data/vision_pilot_v2 ; .venv/bin/gso validate-data --dataset data/vision_transfer_v2
.venv/bin/gso split --dataset data/vision_pilot_v2 --seed 0          # verifica el split planificado
caffeinate -i .venv/bin/gso train --config configs/vision2_heads.yaml       # runs/vision2_heads/20260923T161024Z
caffeinate -i .venv/bin/gso train --config configs/vision2_text_only.yaml   # runs/vision2_text_only/20260923T162937Z
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_heads/20260923T161024Z/checkpoint --split validation --no-cache --vision-ablation --robustness --baselines
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_heads/20260923T161024Z/checkpoint --split test --final-test --no-cache --baselines --vision-ablation
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_text_only/20260923T162937Z/checkpoint --split test --final-test --no-cache --baselines
.venv/bin/gso compare --a runs/vision2_text_only/20260923T162937Z/evaluations/test-20260923T164713Z-predictions.jsonl \
  --b runs/vision2_heads/20260923T161024Z/evaluations/test-20260923T164638Z-predictions.jsonl \
  --allow-different-inputs --out reports/phase4b/test_compare_vision_minus_text.json
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_heads/20260923T161024Z/checkpoint --split all --dataset data/vision_transfer_v2 --no-cache --baselines --vision-ablation
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_text_only/20260923T162937Z/checkpoint --split all --dataset data/vision_transfer_v2 --no-cache
```


## Límites

- Los estilos «no vistos» son variaciones de color, orientación, grosor, trazo y fuente dentro del mismo tipo de gráfico. No hay tipos de gráfico nuevos, fotos ni documentos.
- V2 eligió la época 30 (el tope); no se amplió para no ajustar más con validación.
- V2 no está calibrado.
- Test y validación tienen 70 grupos cada uno.
