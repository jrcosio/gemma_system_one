# Fase 6g: reproducibilidad desde un clon limpio y sensibilidad de la regla S

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16. Commit verificado: **`a556fd5`**, clonado desde `origin` (GitHub). No se cambió código del proyecto; las correcciones de la revisión de la 6f siguen sin commit en el repositorio de trabajo.

## Veredicto

1. **El commit publicado es reproducible de principio a fin en esta máquina, desde un clon limpio.**
   - Instalación con `uv sync --frozen`; tests CPU (301), ruff y formato.
   - Tests MPS reales (10): forward y backward de E2B, gradientes LoRA y recarga.
   - Generación de datos y split idénticos.
   - Entrenamiento **bit a bit idéntico** al original: pesos y logits de validación con |Δ| = 0,0.
   - Servicio HTTP real con benchmark de 100 peticiones.
2. **Lo que el clon no incluye, por diseño:**
   - **Checkpoints, datos y pesos:** no están en Git. Los 4 E2E se omiten avisando de que falta el checkpoint entrenado. Los pesos se obtienen con `gso download` (aquí, con acierto en la caché de Hugging Face, sin volver a descargarlos).
   - **Máquina nueva:** no se ha probado sin caché. `uv sync` tardó 0,6 s porque los paquetes ya estaban en la caché local de uv.
3. **La regla S de la fase 6f no depende de los casi duplicados:** sin los 11 grupos de `final17` con estados casi duplicados de `pilot_v3`, `pilot_v5` o `calib16`, A4v5 − A4v3 = −0,006 [−0,038; +0,028], frente a −0,007 [−0,037; +0,025]. S sigue sin cumplirse y el servicio sigue siendo A4v3. Es un análisis posterior y descriptivo.
4. **Bloqueo externo:** la generalización fuera de la familia sintética no puede medirse sin datos reales etiquetados. El trabajo restante queda descrito en la sección final.

## Clon limpio: comandos y resultados

Se ejecutó en un directorio temporal fuera del repositorio.

| Paso | Comando | Resultado |
|---|---|---|
| Clonar | `git clone https://github.com/<owner>/gemma_system_one.git` | HEAD `a556fd5`, árbol limpio |
| Instalar | `uv sync --frozen` | torch 2.14.0, transformers 5.17.0, peft 0.21.0; MPS disponible; el paquete se importa desde el clon |
| CPU | `uv run pytest tests/unit tests/integration -q -p no:cacheprovider` | **301 passed** (38,55 s) |
| Lint | `uv run ruff check .` · `uv run ruff format --check .` | Correctos (154 ficheros) |
| Pesos | `uv run gso download --config configs/e2b_text.yaml` | `cache hit (sin descarga)`; sha256 LFS verificado; manifiesto escrito en el clon |
| MPS | `uv run pytest tests/mps -q -rs` | **10 passed**, 0 omitidos (116 s) |
| E2E | `uv run pytest tests/e2e -q -rs` | **4 skipped**, con el motivo explícito «Falta el checkpoint entrenado runs/…»: los checkpoints no se versionan |
| Datos | `PYTHONHASHSEED=1 uv run gso generate-data --kind mixed --generator-version v5 --variant main --seed 0 --cases 1000 --out data/pilot_v5` | `examples.jsonl` idéntico byte a byte (sha256 `7b12f921…`) |
| Split | `uv run gso split --dataset data/pilot_v5 --seed 0` | Particiones idénticas; el fichero sólo difiere en `created_utc` |
| Entrenar | `uv run gso train --config configs/e2b_heads_v5.yaml` | Época 18 y NLL de validación 0,46420069…, iguales al original `runs/e2b_heads_v5/20260924T185841Z`; pesos |Δ| = 0,0; 300 logits de validación |Δ| = 0,0; misma huella de extracción y del dataset; extracción real (354 s en train, sin caché) |
| Servir | `uv run gso benchmark --config configs/serve_clone_check.yaml --dataset data/pilot_v5 --split validation --requests 100 --warmup 5` | 100 × 200 en MPS; p50/p95 HTTP 606/1010 ms; arranque 4,4 s; ráfaga 5 × 200 + 2 × 503 atribuidos (`valid: true`); 112 peticiones en el log del servidor; `same_code_as_server: true` |

- `configs/serve_clone_check.yaml` es temporal y sólo existe en el clon; apunta al checkpoint reentrenado allí.
- Evidencia copiada, con las rutas enmascaradas: `reports/phase6g/clean_clone/` (`train_clone.log`, `clone_run_metrics.json`, `bench_clone.json`, `bench_clone.stdout`, `bench_clone.server.log` y `serve_clone_check.yaml`).

## Sensibilidad de la regla S (`reports/phase6g/s_sensitivity_near_dups.{txt,json}`)

| Conjunto | Preguntas / grupos | A4v5 − A4v3, NLL calibrada [IC95 %] |
|---|---|---|
| `final17` completo (regla predeclarada) | 837 / 279 | −0,0074 [−0,0373; +0,0251] |
| Sin 11 grupos casi duplicados (6 frente a `pilot_v3`, 5 frente a `pilot_v5` y 2 frente a `calib16`, en unión) | 804 / 268 | −0,0063 [−0,0377; +0,0281] |

Los pares casi duplicados se detectan con Jaccard de 3-gramas de palabras (`data.split.near_duplicate_states`). El bootstrap es por grupos, con 1000 repeticiones y semilla 0.

## Bloqueo externo: generalización fuera de la familia sintética

- **Situación:** todos los datos del proyecto son sintéticos (`support-mixed-v1…v5`, `support-vision-v1/v2`). No hay datos reales etiquetados en el repositorio ni en la máquina, y generarlos exige anotación humana; no hay que inventarlos.
- **Trabajo restante, concreto y reproducible:**
  1. **Datos reales:** reunir de 100 a 300 incidencias reales anonimizadas y escribirlas en el esquema JSONL v1 (`docs/ESPECIFICACION.md` §5.1), con preguntas y criterios explícitos. Las etiquetas deben ponerlas personas, con `provenance.source: "human"`, y cada caso debe ser un `group_id` propio.
  2. **Validar:** `uv run gso validate-data --dataset data/real_v1`.
  3. **Predeclarar:** métrica y reglas en un protocolo nuevo antes de evaluar.
  4. **Evaluar** el servicio sin reentrenar: `uv run gso evaluate --checkpoint runs/e4b_experiment/20260923T204945Z/checkpoint --split all --dataset data/real_v1 --calibration runs/e4b_experiment/20260923T204945Z/calibration/calibration-20260924T045005Z.json --baselines`.
  5. **Controlar el uso del estado** con tríos o con un control equivalente adaptado a los datos reales.
- **Privacidad:** los datos reales no deben subirse a Git (`/data/` ya está ignorado) y los informes deben mostrar sólo métricas agregadas.
