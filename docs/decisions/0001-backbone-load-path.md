# 0001 — Ruta de carga del backbone y dependencias del procesador

Fecha: 2026-09-22 · Fase 0 · Estado: aceptada

## Contexto

La especificación (§4.2) pide cargar una base sin cabeza de lenguaje, preferiblemente `Gemma4Model`, sin borrar `lm_head` a mano. El checkpoint `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7` declara `architectures: ["Gemma4ForConditionalGeneration"]` e incluye torres de visión y audio.

## Decisión

1. Cargar `transformers.Gemma4Model.from_pretrained(<snapshot local>, dtype=bfloat16, attn_implementation="sdpa", local_files_only=True)` y moverlo con `.to("mps")`. No se usa `device_map`.
2. Tras la carga se exige `missing_keys == []`, sin claves no coincidentes ni errores, y ninguna clave inesperada del modelo textual. Si no se cumple, se lanza `BackboneLoadError`.
3. Se carga el modelo multimodal completo (texto + visión + audio), no `Gemma4TextModel` solo: la fase 4 usará visión con el mismo objeto y el coste medido es aceptable.
4. El procesador oficial `Gemma4Processor` se usa desde fase 0. Esto obliga a añadir `Pillow` y `torchvision` (ver evidencia).
5. Convención de padding del proyecto: **derecha**, con `padding_side` explícito. El tokenizer trae `left` por defecto.

## Evidencia (medida en el M5 Pro, ver `reports/compatibility.md`)

- Clases en transformers 5.17.0: `Gemma4Model` → `language_model: Gemma4TextModel`, `vision_tower`, `audio_tower`, `embed_vision`, `embed_audio`; `hasattr(model, "lm_head") == False`.
- Informe de carga: `missing=[]`, `unexpected=[]`, `mismatched=[]`, `error_msgs=[]`.
- La cabecera safetensors tiene 5 123 178 979 valores. Los parámetros cargados suman 5 104 297 504. La diferencia está explicada al completo:
  - 60 tensores `self_attn.{k_proj,v_proj,k_norm}` de las 20 capas con KV compartido (15–34), con 18 880 512 valores. `Gemma4TextModel.__init__` los añade a `_keys_to_ignore_on_load_unexpected` (`modeling_gemma4.py:1606`), así que se descartan a propósito y en silencio.
  - 963 buffers persistentes: `input/output_min/max` de las `Gemma4ClippableLinear` y 35 `layer_scalar`.
- Memoria en MPS tras cargar: 9,51 GiB asignados (`current_allocated`) y 9,55 GiB de driver. El RSS del proceso es de 0,49 GiB.
- `Gemma4Processor` en transformers 5.17.0 importa `image_processing_gemma4.py`, que hace `from torchvision.transforms.v2 import functional as tvF` sin condición. Sin `torchvision`: `ModuleNotFoundError: Could not import module 'Gemma4Processor'`. Sin `Pillow`: `ImportError: Gemma4Processor requires the PIL library`.

## Consecuencias

- **LoRA (fase 3):** las capas 15–34 no tienen `k_proj` ni `v_proj`, porque reutilizan K/V de capas anteriores. Una lista de módulos objetivo "q_proj, v_proj" solo encuentra `v_proj` en las capas 0–14. La lista exacta se sacará de `named_modules()` y se registrará.
- Visión y audio ocupan unos 0,47 B parámetros (≈0,95 GB en bf16) aunque la fase 1 sea solo de texto. Si la memoria aprieta, la alternativa documentada es cargar `Gemma4TextModel` solo, pero antes hay que verificar que el mapeo de claves sea equivalente.
- `torchvision==0.29.0` está fijado por `torch==2.14.0` (su `requires_dist`). Al actualizar torch hay que actualizar ambos a la vez.
