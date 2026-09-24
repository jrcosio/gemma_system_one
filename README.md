# Gemma System One (local)

Evaluador local que responde **preguntas de decisión tipadas** sobre un mensaje o un estado JSON, usando Gemma 4 en un Mac con Apple Silicon (PyTorch/MPS). No genera texto: cada respuesta se calcula a partir de las representaciones del modelo y cabezales entrenados, y siempre respeta el esquema.

| Primitiva | Qué responde | Salida |
|---|---|---|
| **Noul** | Pregunta de sí o no | Probabilidad `noul` ∈ [0, 1] |
| **Choice** | Elegir una opción entre 2 y 8 descritas en la propia pregunta | `choice`, probabilidades por opción y `confidence` |
| **Score** | Nivel en una rúbrica ordenada de 2 a 5 niveles | `score` esperado, probabilidades por nivel, `legend` y `confidence` |

Las opciones y las rúbricas son **dinámicas**: van en cada petición, no están fijadas en el modelo. Una misma llamada puede incluir hasta 8 preguntas. `confidence` mide la concentración de la distribución (entropía normalizada); no garantiza acierto.

> **Alcance.** El modelo se ha entrenado y evaluado sólo con **datos sintéticos** de incidencias de soporte (facturación, acceso, fallos técnicos, alcance e impacto), en español e inglés. No se ha evaluado con datos reales, así que no debe darse por supuesta su generalización a otros dominios. En las preguntas sobre el tipo de fallo, los datos de entrenamiento del modelo de servicio tenían algunas regularidades estructurales en las opciones; los controles muestran que el modelo lee el mensaje, pero esas cifras de esa familia deben tomarse con cautela.

## Requisitos

- **Máquina:** Mac con Apple Silicon y MPS. Probado en un M5 Pro con 48 GB. El modelo de servicio (E4B) usa unos 17 GB de memoria MPS; la variante E2B, unos 11 GB.
- **Software:** Python 3.11 y [uv](https://docs.astral.sh/uv/).
- **Disco:** ~16 GB para los pesos de Gemma 4 E4B, ~10 GB para E2B (caché de Hugging Face) y unos pocos GB para datos y representaciones.
- **Pesos:** `google/gemma-4-E2B-it` y `google/gemma-4-E4B-it` son públicos (apache-2.0) y se descargan con revisión fijada.

## Instalación

```bash
git clone <url-del-repositorio> gemma_system_one && cd gemma_system_one
uv sync --frozen                                   # entorno exacto del lock (torch, transformers, peft, fastapi…)
uv run pytest tests/unit tests/integration -q      # comprobación rápida en CPU, sin pesos
```

## Puesta en marcha

Los datos, los pesos y los checkpoints no están en Git: se generan en local. Estos pasos reproducen el **modelo de servicio recomendado**, Gemma 4 E4B congelado + cabezales, con calibración externa.

**1. Pesos y diagnóstico del equipo** (descarga única, ~16 GB):

```bash
uv run gso download --config configs/e4b_text.yaml
uv run gso doctor   --config configs/e4b_text.yaml    # carga real, forward/backward, memoria y recarga en MPS
```

**2. Datos** (sintéticos y deterministas: los sha256 coinciden con los de los informes):

```bash
uv run gso generate-data --kind mixed --generator-version v3 --seed 0 --cases 1000 --out data/pilot_v3
uv run gso split --dataset data/pilot_v3 --seed 0
# Conjunto de calibración externo (1200 preguntas), sin solapes con el entrenamiento:
uv run gso generate-data --kind mixed --generator-version v3 --seed 6 --cases 300 --out data/pilot_v3_holdout6
uv run gso generate-data --kind mixed --generator-version v3 --seed 7 --cases 400 --out data/pilot_v3_seed7
uv run gso generate-data --kind mixed --generator-version v3 --seed 8 --cases 300 --out data/pilot_v3_seed8
uv run python scripts/derive_phase6b_data.py          # → data/pilot_v3_calib7 y data/pilot_v3_final8
```

**3. Entrenamiento y calibración** (~12 min de extracción con E4B; los cabezales tardan segundos):

```bash
caffeinate -i uv run gso train --config configs/e4b_experiment.yaml     # imprime runs/e4b_experiment/<ts>
uv run gso calibrate --checkpoint runs/e4b_experiment/<ts>/checkpoint --split all --dataset data/pilot_v3_calib7
uv run gso evaluate  --checkpoint runs/e4b_experiment/<ts>/checkpoint --split all --dataset data/pilot_v3_final8 \
  --calibration runs/e4b_experiment/<ts>/calibration/calibration-<fecha>.json --baselines
```

**4. Servicio.** Copia `configs/serve_e4b_text.yaml` (por ejemplo, a `configs/serve_local.yaml`) y ajusta `checkpoint` y `calibration` a tus rutas. Después:

```bash
uv run gso serve --config configs/serve_local.yaml     # API en http://127.0.0.1:8000 (sólo loopback)
curl -s http://127.0.0.1:8000/health/ready             # 200 cuando el modelo está cargado, verificado y caliente
```

**Variante ligera (E2B + cabezales, ~7 min).** Usa `configs/e2b_text.yaml` en el paso 1 y `configs/pilot_ce_v3.yaml` en el paso 3; calibra con `--split calibration` o con el conjunto externo, y crea la configuración de servicio igual que en el paso 4.

## Uso de la API

`POST /v1/decide` con `model` (el `model_id` de la configuración de servicio), `state` (texto o JSON), `questions` (de 1 a 8, cada una con su identificador) e `image` opcional (PNG o JPEG en base64, sólo para checkpoints entrenados con imagen).

```bash
curl -s -X POST http://127.0.0.1:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "model": "gemma-system-one-e4b-heads-v0.1",
  "state": "Hola, desde esta mañana la VPN se desconecta cada pocos minutos y además me han cobrado dos veces la cuota de junio. Solicito la devolución del cargo duplicado.",
  "questions": {
    "refund": {"type": "noul", "instructions": "¿El cliente pide una devolución?"},
    "fault": {"type": "choice", "instructions": "¿Qué tipo de fallo técnico describe el mensaje?",
              "criteria": {"net": "Red o conectividad (DNS, VPN, conexión)", "slow": "Lentitud o rendimiento",
                           "none": "No se describe ningún fallo técnico", "other": "Otro tipo de fallo técnico"}},
    "severity": {"type": "score", "instructions": "Evalúa la situación técnica descrita.",
                 "criteria": ["No se describe ningún fallo técnico", "Hubo un fallo técnico, pero ya está resuelto",
                              "Hay un fallo técnico activo"]}
  }
}'
```

Respuesta real del modelo de servicio (abreviada):

```json
{
  "model": "gemma-system-one-e4b-heads-v0.1",
  "checkpoint_id": "decision_heads:97fbb4d1b9cc7609+cal:f42131010813",
  "answers": {
    "refund":   {"type": "noul", "noul": 0.987},
    "fault":    {"type": "choice", "choice": "net",
                 "probabilities": {"net": 0.954, "slow": 0.025, "other": 0.013, "none": 0.007}, "confidence": 0.833},
    "severity": {"type": "score", "score": 1.70, "probabilities": [0.152, 0.000, 0.848],
                 "legend": ["No se describe ningún fallo técnico", "Hubo un fallo técnico, pero ya está resuelto",
                            "Hay un fallo técnico activo"], "confidence": 0.611}
  },
  "usage": {"questions": 3, "expanded_rows": 8, "backbone_forwards": 8, "processed_input_tokens": 1652,
            "image_tokens": 0, "padding_tokens": 0, "generated_tokens": 0},
  "metadata": {"confidence_method": "normalized_entropy_v1", "request_id": "…",
               "timing_ms": {"forward_synchronized": 1016.2, "server_total": 1030.5}}
}
```

- **Coste:** cada opción de Choice y cada nivel de Score es un forward del modelo. Una petición cuesta tantas filas como opciones y niveles tenga en total (aquí 1 + 4 + 3 = 8).
- **Errores:**

  | Código | Causa |
  |---|---|
  | 422 | Esquema inválido o modalidad incorrecta |
  | 413 | Cuerpo o imagen demasiado grandes |
  | 400 | Imagen ilegible |
  | 404 | Ruta desconocida |
  | 503 | Cola llena, tiempo agotado o modelo aún sin cargar |

- **Salud:** `/health/live` y `/health/ready`.
- **Límites operativos:**
  - un proceso con un worker y una cola acotada (`max_queue`);
  - el timeout no interrumpe un forward ya iniciado en la GPU.

## Resultados

Medidos en este Mac con datos sintéticos. El test final es un conjunto nuevo que no se usó para ajustar pesos ni temperaturas (888 preguntas, 296 grupos). NLL media por pregunta (menor es mejor) e IC95 % por bootstrap de grupos.

| Modelo | NLL calibrada | Acc. Noul / Choice / Score | Latencia HTTP p50 / p95 |
|---|---|---|---|
| Prior (baseline) | 1,032 | — | — |
| Bolsa de palabras (baseline) | 0,942 | — | — |
| E2B + cabezales | 0,384 | 0,93 / 0,81 / 0,79 | — |
| E2B + LoRA + cabezales (`configs/serve_text.yaml`) | 0,347 | 0,94 / 0,86 / 0,76 | 551 / 909 ms |
| **E4B + cabezales (servicio recomendado)** | **0,201** | **0,96 / 0,90 / 0,89** | 910 / 1546 ms |

- **E4B frente a E2B + LoRA:** E4B mejora la NLL en −0,146 [−0,186; −0,104], con ~1,6× de latencia y un arranque en frío de ~9,5 s. La latencia es por petición de 3 preguntas (~8 filas); rendimiento de ~1 petición/s con un worker.
- **Robustez:**
  - las respuestas no cambian al renombrar los IDs de las opciones ni al permutarlas;
  - en controles donde sólo cambia el estado del mensaje, el modelo cambia su respuesta, es decir, lee el estado y no sólo las opciones.
- **Imagen:** existe un servicio con una imagen por petición (`configs/serve_vision.yaml`, E2B) para paneles de barras sintéticos. Supera claramente al mismo modelo sin imagen en estilos visuales no vistos, pero no está calibrado.
- **Reproducibilidad:** desde un clon limpio, los datos se regeneran idénticos y el entrenamiento de cabezales se reproduce bit a bit.

## Comandos (`gso`)

| Comando | Para qué |
|---|---|
| `gso download --config …` | Descarga fijada de los pesos, con verificación sha256 |
| `gso doctor --config …` | Compatibilidad y recursos del equipo con el modelo real |
| `gso generate-data --kind mixed\|vision …` | Datasets sintéticos deterministas (`--generator-version`, `--seed`, `--cases`) |
| `gso validate-data --dataset …` | Esquema, ficheros y grupos |
| `gso split --dataset … --seed …` | Particiones train / validation / calibration / test por grupos, sin fugas |
| `gso train --config …` | Cabezales sobre la base congelada, o LoRA + cabezales (`kind` de la configuración) |
| `gso calibrate --checkpoint … --split calibration` o `--split all --dataset …` | Temperatura por primitiva |
| `gso evaluate --checkpoint … --split … [--calibration …] [--baselines]` | Métricas y predicciones auditables; `--split test` exige `--final-test` |
| `gso compare --a … --b …` | Diferencia emparejada entre dos ficheros de predicciones, con IC |
| `gso serve --config …` | API local con el modelo real |
| `gso benchmark --config … --dataset … --split validation` | Arranque, latencia y saturación de la API en un proceso aparte |

**Formato de los datos:** JSONL con `state`, `question` (`type`, `instructions`, `criteria`), `target` y `group_id` (esquema v1 en `docs/ESPECIFICACION.md` §5.1). Para evaluar el servicio con datos propios: `gso validate-data --dataset data/mis_datos` y después `gso evaluate --split all --dataset data/mis_datos`.

## Estructura

```text
src/gemma_system_one/   contratos, serialización, modelo, entrenamiento, calibración, API y CLI
configs/                configuraciones de modelo, entrenamiento y servicio
scripts/                derivación de datasets y análisis reproducibles
tests/                  unit, integration (CPU), mps (hardware real) y e2e (servicio con checkpoints)
docs/                   especificación y decisiones de diseño (docs/decisions/)
reports/                evidencia de cada experimento (métricas, protocolos y comandos)
```

- **Tests:** `uv run pytest tests/mps -rs` necesita MPS y los pesos descargados. `tests/e2e` necesita checkpoints entrenados; si faltan, los tests se omiten avisando del motivo.
- **Licencia:** propietaria (`pyproject.toml`). Los pesos de Gemma 4 se rigen por su propia licencia.
