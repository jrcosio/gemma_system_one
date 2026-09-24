# Informe de compatibilidad — fase 0

Fecha: 2026-09-22 · Ejecutado en el equipo objetivo · Código sobre el commit base `65d4250` (con cambios de fase 0 aún sin commit).

Revisión independiente posterior: [hallazgos, correcciones y pruebas](revision-fase-0.md). Se confirma la ruta real de texto; las cifras detalladas siguientes corresponden a la ejecución original.

**Veredicto por modalidad**

| Modalidad | Veredicto | Base |
|---|---|---|
| Texto (E2B, PyTorch/MPS, bf16) | **Compatible para fase 1** | `gso doctor` con `verdict: pass` y `tests/mps` 2/2 pasados, con los pesos reales |
| Imagen | **Pendiente (fase 4)** | No ejecutado. El procesador de imagen importa sin error, pero no se ha hecho ninguna pasada con imagen |
| Audio | Fuera de alcance | La torre se carga en memoria pero no se usa |
| LoRA/PEFT | **Pendiente (fase 3)** | PEFT no está instalado. Solo hay una sonda de gradiente real por el backbone, sin PEFT (ver §5) |

## 1. Entorno (evidencia: `environment` del doctor)

| Elemento | Valor |
|---|---|
| Máquina | `Mac17,8`, Apple M5 Pro, arm64 |
| Memoria | 51 539 607 552 B (48 GiB) unificada |
| macOS | 26.6.2 (25G83) |
| Python | 3.11.15 (gestionado por uv 0.11.6) |
| torch | 2.14.0 (MPS built y available) |
| transformers | 5.17.0 |
| huggingface_hub / hf-xet | 1.32.0 / 1.6.0 |
| tokenizers / safetensors | 0.23.2 / 0.8.0 |
| torchvision / Pillow | 0.29.0 / 12.3.0 |
| pydantic / numpy / PyYAML / psutil | 2.13.5 / 2.4.6 / 6.0.3 / 7.2.2 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | no definido, sin fallback silencioso a CPU |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | no definido, límites MPS por defecto |

Versiones exactas en `uv.lock` (66 paquetes). `uv lock --check` es correcto.

## 2. Checkpoint (evidencia: `gso download`, `checkpoint_metadata`)

- `google/gemma-4-E2B-it` en la revisión `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. El Hub devuelve `gated: False` y `license: apache-2.0` (según la ficha; condiciones completas en el README del modelo).
- `model.safetensors`: 10 246 621 918 B, sha256 `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`. El hash se recalculó entero tras la descarga y otra vez con `--verify-hash`, y coincide con el LFS del Hub. `tokenizer.json` también coincide (`cc8d3a0c…`).
- Descarga única: la primera ejecución de `gso download` descargó. Las tres siguientes dieron `cache hit (sin descarga)` y el espacio libre del disco no cambió (561 GiB). huggingface_hub 1.32 guarda el blob tras un segundo enlace simbólico, en `~/.cache/huggingface/blobs/`. Hay un solo ejemplar de los pesos.
- Configuración: `Gemma4ForConditionalGeneration` y `model_type gemma4`, generada con transformers `5.5.0.dev0`. Texto: `hidden_size=1536`, 35 capas, `num_kv_shared_layers=20`, `sliding_window=512`, vocabulario de 262 144 tokens, embeddings enlazados. Incluye visión (280 soft tokens por imagen) y audio.
- Parámetros según la cabecera safetensors, sin cargar pesos: 5 123 178 979 en total, todos en BF16. Los mayores bloques son `embed_tokens_per_layer` (2,349 B), `layers` (1,882 B), `embed_tokens` (0,403 B), audio (0,305 B) y visión (0,167 B).

## 3. Carga del backbone (evidencia: `load_backbone`)

- Clase `Gemma4Model` (MRO `Gemma4Model → Gemma4PreTrainedModel → PreTrainedModel → Module`). El modelo textual es `Gemma4TextModel` y **no hay `lm_head`**.
- `attn_implementation = sdpa`. Parámetros en `torch.bfloat16` y en `mps:0`. Buffers en bf16 y fp32.
- Informe de carga vacío (`missing`, `unexpected`, `mismatched` y `error_msgs`). Hay 5 104 297 504 parámetros cargados, con 0 entrenables y el modelo en `eval()`.
- Diferencia frente a la cabecera: 60 tensores K/V de capas con KV compartido que transformers descarta a propósito, más 963 buffers. Detalle en `docs/decisions/0001-backbone-load-path.md`.
- Tiempo de carga sincronizado: 2,42 s y 1,94 s en dos ejecuciones. **Supuesto:** el fichero estaba en la caché de páginas del sistema, porque se acababa de descargar y hashear. No es un arranque en frío desde disco.

## 4. Pasada textual y extracción (evidencia: `text_forward`)

- El procesador `Gemma4Processor` aplica la plantilla oficial con `add_generation_prompt=True` y `enable_thinking=False`, sin generar texto. Campos devueltos: `input_ids`, `attention_mask` y `mm_token_type_ids`. Todos se pasan al modelo. Hay un solo `<bos>` por fila.
- Lote de 3 filas ES/EN: forma `[3, 50, 1536]` bf16, 112 tokens válidos y 38 de padding. Todos los valores son finitos.
- Pooling en el último token válido (`max t: mask==1`):
  - Con padding derecho frente a la misma fila sin padding, la similitud coseno es ≥ 0,99991. La tolerancia declarada es 0,999.
  - Con padding izquierdo (el valor por defecto del tokenizer), la similitud es ≥ 0,99959, con posiciones por defecto o derivadas de la máscara.
  - El proyecto usa padding derecho.
- `use_cache=True` frente a `False` en el último token: diferencia 0,0. Repetir la pasada da diferencia 0,0.
- Dos entradas distintas dan similitud coseno de 0,79: la representación depende de la entrada.
- Tiempos sincronizados del lote: 1,42 s en la primera pasada de la primera ejecución (incluye preparación de kernels MPS) y 0,12 s en caliente. En ejecuciones posteriores: 0,08 s y 0,04 s. **No es un benchmark:** hay una sola repetición y 112 tokens.

## 5. Entrenamiento mínimo y gradientes (evidencia: `head_train`, `backbone_backward_probe`, `long_context_probe`)

- **Cabezal Noul** (`Linear(1536,1)` en FP32) con el backbone congelado y h extraído con `no_grad`. Cinco pasos de AdamW (lr 1e-3, wd 0,01), BCE en FP32:
  - La pérdida baja de 0,236 a 0,005 y los gradientes son finitos. Las normas de gradiente se registran en el informe JSON.
  - Entrenables: solo `proj.weight` y `proj.bias` (1537 parámetros en el optimizador).
  - Ningún parámetro del backbone tiene `.grad`. Tres tensores vigilados siguen bit a bit iguales (`layers.0.q_proj`, `layers.34.mlp.down_proj`, `norm`). El backbone sigue en `eval()`.
- **Sonda de backward por el backbone**, como riesgo previo a LoRA y sin PEFT: `requires_grad` activado solo en `layers.0.self_attn.q_proj.weight` y `layers.34.self_attn.q_proj.weight`. El backward en MPS/bf16 recorre las 35 capas. Normas de gradiente: 0,0269 y 0,0814, finitas. Ningún otro parámetro recibe gradiente. Tiempo: 0,15 s con 50 tokens.
- **Fila de 512 tokens** (contenido sintético recortado, solo para medir):
  - Forward de inferencia: 0,125 s.
  - Forward y backward hasta la capa 0: 0,63 s.
  - Con el grafo construido, la memoria MPS asignada pasa de 9,51 a 12,55 GiB (unos +3,0 GiB de activaciones para una fila, sin checkpointing). El driver llega a 12,83 GiB. El gradiente es finito.

## 6. Guardado y recarga mínima (evidencia: `save_reload`)

- Checkpoint `head/`: `head.safetensors` y `manifest.json`, 6919 B en total. No duplica pesos del backbone. El manifiesto guarda `repo_id`, revisión, dtype, `hidden_size`, plantilla, sha256 y git.
- Se recargó en **procesos nuevos** con `python -m gemma_system_one.checkpoint verify`. Diferencia máxima de logits sobre las representaciones guardadas: 1,9e-6 en CPU y 0,0 en MPS. La tolerancia declarada es 1e-4.
- **Limitación:** la recarga compara el cabezal sobre h guardados. No vuelve a ejecutar el backbone en el proceso nuevo, para no tener dos procesos de 10 GB a la vez. La igualdad del backbone entre pasadas se mide dentro del mismo proceso (diferencia 0,0).

## 7. Memoria (evidencia: `memory` y muestras del doctor)

| Punto | MPS asignada | Driver MPS | RSS del proceso | Disponible en el sistema |
|---|---|---|---|---|
| Inicio | 0 | 0 | 0,20 GiB | 28,98 GiB |
| Tras cargar | 9,51 GiB | 9,55 GiB | 0,49 GiB | 15,20 GiB |
| Grafo de 512 tokens | 12,55 GiB | 12,83 GiB | 0,83 GiB | 15,64 GiB |
| Final | 9,51 GiB | 10,57 GiB | 0,84 GiB | 18,49 GiB |

- Máximos muestreados: driver 12,83 GiB (13 774 045 184 B) y RSS 0,93 GiB. No se suman, porque pueden solaparse en memoria unificada.
- Swap: 0 B durante toda la ejecución. Presión (`kern.memorystatus_vm_pressure_level`) = 1 (normal) en todas las muestras.
- `torch.mps.recommended_max_memory` = 40 200 896 512 B (37,44 GiB). El presupuesto del proyecto es 32 GiB y no se superó.
- **Técnica:** instantáneas en puntos instrumentados. El pico real puede ser mayor.
- **Observación:** al arrancar solo había unos 29 GiB disponibles, porque otras aplicaciones del usuario estaban abiertas. El presupuesto de 32 GiB depende de lo que haya abierto en el Mac.

## 8. Incidencias encontradas y resueltas

| Incidencia | Reproducción | Resolución |
|---|---|---|
| `AutoProcessor` falla: `ImportError: Gemma4Processor requires the PIL library` | `AutoProcessor.from_pretrained(<snapshot>)` sin Pillow | Se añade `pillow>=11` |
| `ModuleNotFoundError: Could not import module 'Gemma4Processor'`, causado por `No module named 'torchvision'` | `from transformers.models.gemma4 import processing_gemma4` en transformers 5.17.0 (`image_processing_gemma4.py:17` importa torchvision) | Se añade `torchvision>=0.29` (0.29.0 exige `torch==2.14.0`) |
| Aviso `Kwargs passed to processor.__call__ have to be in processor_kwargs` | `apply_chat_template(..., padding=True)` | Se pasa `processor_kwargs={"padding": True}`. Se comprobó que el padding se aplicaba igual |
| El tokenizer usa `padding_side="left"` por defecto | `AutoTokenizer.from_pretrained(<snapshot>).padding_side` | El proyecto fija `right` de forma explícita |
| En huggingface_hub 1.32 el blob final no se llama por su sha256 | `readlink` en snapshot → `blobs/<sha256>` → `../../blobs/xx/<otro>` | La comprobación barata usa el primer enlace. El hash completo sigue siendo la referencia |

La revisión posterior corrigió fallos de integridad y monitorización; no detectó una incompatibilidad del backend de texto en la configuración probada.

## 9. Supuestos y límites de este informe

- No se ha medido: arranque en frío desde disco, benchmark de latencia (≥100 consultas), lotes mayores de 3 filas, filas de más de 512 tokens, estabilidad del entrenamiento en bf16 durante muchos pasos, FP16 del backbone, gradient checkpointing, PEFT/LoRA ni imagen.
- `mps_ops` compara con CPU pequeñas operaciones en fp32, bf16 y fp16. Las diferencias de gradiente son de 1e-5 en fp32, 0,125 en bf16 y 0,016 en fp16, dentro de lo esperado para cada precisión. No prueba la estabilidad de todo el modelo en fp16.
- No se ha comparado numéricamente el forward de E2B en MPS con una referencia en CPU del modelo completo.

## 10. Comandos para reproducir

```bash
uv venv --python 3.11 && uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest tests/unit tests/integration -q          # 49 pruebas en CPU, sin pesos
uv run gso download --config configs/e2b_text.yaml     # descarga una vez; después, cache hit
uv run gso download --config configs/e2b_text.yaml --verify-hash
uv run gso doctor --config configs/e2b_text.yaml --json reports/doctor/latest.json
uv run pytest tests/mps -v -rs                         # 2 pruebas reales en MPS (skip con motivo si faltan MPS o pesos)
```

El JSON completo del doctor (`reports/doctor/*.json`) y los artefactos (`artifacts/`) no se versionan, porque contienen rutas locales y checkpoints de prueba. Este documento resume esos datos.
