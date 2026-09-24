# 0010 — Fase 6: recomputación de activaciones para LoRA grande; sin agrupar filas por petición

Fecha: 2026-09-23 · Fase 6 · Estado: aceptada

## Contexto

La fase 6 evalúa E4B, más opciones y optimizaciones opcionales con el criterio «ganancia medida que justifique coste y complejidad». El protocolo se predeclaró en `reports/phase6-protocol.md` (sha256 `c745cfea…`). Esta decisión recoge dos cambios de ingeniería medidos; la comparación de calidad de E4B está en `reports/phase6-e4b.md`.

## 1. LoRA sobre E4B: recomputación de activaciones por capa (opción, no por defecto)

**Evidencia** (`scripts/profile_lora_step.py`: 6 pasos reales de 8 preguntas de train de `pilot_v3`, la receta de `pilot_lora_v3`, MPS/BF16, 1 fila por forward):

| Modelo | Recomputación | Mediana s/paso (sin el 1.º) | Pico muestreado del driver MPS | Resultado |
|---|---|---|---|---|
| E2B | no | 4,78 | 23,3 GB | 6/6 pasos (fase 3 real: 4,5 s/paso) |
| E2B | sí | 6,62 | 19,0 GB | 6/6; gradientes LoRA **idénticos** (diferencia relativa 0,0; 1 339 392 de 1 339 392 elementos no nulos) |
| E4B | no | 7,29 (3 pasos) | **34,5 GB** | Se supera el presupuesto de 32 GiB en el paso 3 (Choice K = 6); `MemoryTracker` aborta |
| E4B | sí | 11,92 | 17,4 GB | 6/6; 132 tensores LoRA (42 `q_proj` + 24 `v_proj`) con gradiente finito; swap sin cambio |

El pico del driver incluye bloques retenidos por el asignador (decisión 0005 §8); la memoria asignada máxima fue 15,9 GB en ambos casos de E4B. No se sube el presupuesto ni se desactiva el límite de MPS.

**Decisión:**
- `models.lora.enable_layer_recomputation` marca cada `GradientCheckpointingLayer` del decodificador textual con `torch.utils.checkpoint` no reentrante.
- Transformers sólo recomputa si la capa está en `training`. `set_lora_mode` cambia el indicador sólo en esas capas, no en sus submódulos: atención (`attention_dropout` = 0,0 en E2B y E4B), MLP y normas siguen en `eval()` y el forward no cambia.
- Con el modo no reentrante, LoRA recibe gradiente aunque los embeddings congelados no lo requieran.
- Se activa con `train.recompute_layers: true` (por defecto `false`) y no actúa en evaluación ni servicio.
- Pruebas:
  - `tests/unit/test_lora.py::test_layer_recomputation_gives_same_grads_and_keeps_base_in_eval`: cada capa pasa por el checkpoint, los gradientes son iguales y la base queda en eval;
  - `tests/integration/test_phase3_pipeline.py::test_lora_training_with_layer_recomputation_matches_plain`: mismos pesos e historial;
  - la equivalencia en MPS real con E2B de la tabla anterior.

**Coste:** con E4B, ~11,9 s/paso; 789 pasos (3 épocas de `pilot_v3`) ≈ 2,6 h sincronizadas. No se entrena en este turno (regla del protocolo).

## 2. Agrupar las filas de una petición en un forward: rechazado

**Evidencia** (`scripts/measure_request_batching.py`, B2 calibrado, `pilot_v3` validation, 100 peticiones = 300 preguntas = 781 filas; `reports/phase6/batching_b2_validation.json`):

| Métrica | Tolerancia predeclarada | Medido |
|---|---|---|
| máx. \|Δp\| publicada | ≤ 0,02 | **0,149** (p95 0,054; mediana 0,0004) |
| Cambios de decisión | 0 | **2** (1 Choice, 1 Score) |
| \|ΔNLL media\| | ≤ 0,005 | 0,0002 |
| Aceleración del forward sincronizado | ≥ 1,5× | **1,00×** (55,5 s frente a 55,4 s; 781 → 100 forwards, +19 063 tokens de padding) |

**Decisión:** se mantiene la decisión 0002 (una fila por forward) en entrenamiento, evaluación y servicio. La agrupación con padding cambia probabilidades publicadas y decisiones, y no acelera en MPS con estas longitudes. Otras variantes quedan sin medir y no se adoptan: agrupar sólo filas de igual longitud, reutilizar el prefijo común con caché KV o forward en FP32.

## Consecuencias

- LoRA sobre E4B es ejecutable dentro del presupuesto sólo con `recompute_layers: true`. Su calidad no se ha medido.
- La latencia de servicio sigue siendo lineal en filas por petición (K/M forwards por pregunta).
