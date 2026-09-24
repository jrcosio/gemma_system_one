# 0005 — Etapa LoRA: módulos, arranque, estandarizador, autograd por fila y artefactos

Fecha: 2026-09-23 · Fase 3 · Estado: aceptada

## Contexto

La spec §7.B pide LoRA sobre proyecciones Q/V del transformer textual, verificar PEFT con la versión instalada, autograd real (sin `no_grad` ni caché), AdamW con grupos LoRA 1e-4 y cabezales 5e-4, warmup 5 %, decaimiento lineal y clip 1,0. §7.D separa artefacto de despliegue y de reanudación. La decisión 0004 dejó abierto qué hacer con el estandarizador fijo cuando las representaciones cambian.

## Decisión

1. **Biblioteca:** `peft==0.21.0` (publicada el 2026-09-15; `transformers 5.17.0` es del 2026-09-09). Añade `accelerate==1.15.0`; `torch`/`transformers` no cambian. Se usa `peft.inject_adapter_in_model`, que modifica el modelo en su sitio: sigue siendo `Gemma4Model` y la ruta de carga de la decisión 0001 no cambia.
2. **Módulos:** nombres completos obtenidos de `named_modules()`, `language_model.layers.{i}.self_attn.{q_proj|v_proj}` y tipo exacto `nn.Linear`. En E2B salen **35 `q_proj` + 15 `v_proj` (capas 0–14)**; las capas 15–34 comparten K/V. Un patrón genérico «q_proj, v_proj» adaptaría también 16+16 módulos de visión y 12+12 de audio (`Gemma4ClippableLinear`). La lista exacta se guarda en el manifiesto del checkpoint.
3. **LoRA:** r = 8, α = 16, dropout 0,05 (spec §2.3). Pesos LoRA en **FP32** sobre la base en BF16: PEFT los crea en el dtype de la base con `inject_adapter_in_model`, así que se convierten explícitamente. PEFT convierte la entrada al dtype de LoRA y devuelve el dtype de la base. `lora_A` se inicializa con un RNG propio con semilla (ver «Fallo corregido»).
4. **Modos:** la base siempre en `eval()`; sólo los módulos `lora_dropout` pasan a `train` durante el paso. La evaluación usa `eval()` y `no_grad`.
5. **Arranque (warm start):** la etapa B parte de un checkpoint de cabezales de fase 2 del **mismo dataset, split y huella de extracción** (se rechaza en otro caso). Como `lora_B = 0`, el modelo en el paso 0 es exactamente ese modelo congelado: la época 0 de validación es la comparación emparejada y un candidato más de la selección. Si LoRA no mejora en validación, el artefacto elegido es el de partida.
6. **Estandarizador:** se mantiene **fijo** con la media/desviación de train del backbone congelado (decisión 0004). Es un escalado afín fijo que no limita lo que el cabezal lineal puede representar. Se registra por época su deriva en la partición de evaluación (media |μ| por dimensión y desviación típica media de las características estandarizadas). Recalcularlo durante el entrenamiento cambiaría la función que los cabezales ya aprendieron.
7. **Autograd por fila:** cada pregunta hace un forward con gradiente por fila (`microbatch_rows = 1`, decisión 0002), concatena los K/M logits, calcula la pérdida del grupo completo y hace backward por pregunta, dividido por el número real de preguntas del paso. No hay gradient checkpointing: con una fila por forward y K ≤ 6 la memoria **asignada** se queda en ~10,2 GB (los pesos) entre pasos.
8. **Caché del asignador MPS:** con formas variables, el asignador retiene bloques del backward y la memoria del **driver** crece (25,2 GB en la puerta de sobreajuste; ~30 GB en el piloto, con el sistema en presión de memoria y el compresor activo). Se libera con `torch.mps.empty_cache()` cada `empty_mps_cache_every_steps = 10` pasos y se registra la memoria MPS en cada paso. Hacerlo en cada paso era caro (35 % del hilo principal en una muestra con 24 s/paso). No se desactiva ningún límite de memoria de MPS.
9. **Optimización:** AdamW, grupo LoRA (lr 1e-4, wd 0) y grupo cabezales (lr 5e-4, wd 0,01); warmup lineal del 5 % de los pasos y decaimiento lineal hasta 0; clip global 1,0; se exige gradiente LoRA presente y finito en cada paso. El wd 0 de LoRA es una elección inicial, no un valor de la spec.
10. **Artefactos:**
   - Despliegue (`checkpoint/`, `kind: lora_decision_heads`, formato 1): `adapter.safetensors` (sólo LoRA, FP32), `head.safetensors` (cabezales + estandarizador), manifiesto con base y revisión, huella, plantilla, lista de módulos, versión de PEFT, dataset, split, época y pasos. Sin pesos base. Guardado atómico, inmutable.
   - Reanudación (`resume/`): pesos entrenables actuales y mejores, estado del optimizador y del scheduler, RNG de CPU y MPS, época, posición en la permutación, paso global e historial. Un directorio nuevo por guardado y un puntero `LATEST` reemplazado con `os.replace`. `gso train --resume` exige la misma configuración, datos, split, punto de partida y módulos.
   - Las representaciones con LoRA nunca se cachean (`cache_dir: null` en el manifiesto).

## Evidencia

- `tests/mps/test_phase3_real.py` (E2B real, MPS/BF16): 35 + 15 módulos; identidad exacta con B = 0 (`torch.equal` de las representaciones); gradientes finitos y no nulos en autograd de grupo; pérdida que baja en 3 pasos; pesos base sin cambios; adaptador recargado con representaciones idénticas.
- **Microlotes con autograd** (`scripts/repro_lora_microbatch.py`, 12 preguntas Choice/Score de train, E2B/MPS/BF16, `reports/phase3/repro_lora_microbatch.json`): con todo el grupo en un microlote frente a una fila por forward, |ΔL| máximo **0,194** y diferencia relativa del gradiente LoRA de hasta **16,6 %** (coseno mínimo 0,989); incluso con |ΔL| = 0 el gradiente difiere un 0,8 %. En CPU/FP32 (modelo diminuto) coinciden dentro de 1e-4 relativo. Se mantiene una fila por forward.
- **Puerta de sobreajuste con sólo LoRA** (`configs/lora_overfit_v3.yaml`, cabezales fijos, 48 preguntas de train de `mixed_smoke_v3`, lr LoRA 2e-3): NLL de train 0,401 → 0,00016 y accuracy 1 por primitiva en 12 épocas (72 pasos); base sin cambios. Interrumpido en el paso 20 y reanudado con `--resume`. Recarga en proceso nuevo, sin caché: diferencia máxima de logit **0,0** (48 preguntas, 377 forwards).
- **Coste medido:** puerta, ~4,8 s por paso de 8 preguntas. Piloto (789 pasos, 2100 preguntas × 3 épocas): mediana 4,5 s/paso y p95 13,9 s; 75 min sumando pasos sincronizados, 3,6 filas/s y 840 tokens/s con backward, frente a ~16,6 filas/s de la extracción sin gradiente. El reloj fue mucho mayor porque el Mac entró 148 veces en reposo de mantenimiento: lanzar con `caffeinate -i`.
- **Memoria en el piloto con vaciado cada 10 pasos:** driver MPS por paso con mediana 22,6 GB y máximo 25,6 GB; asignada ~10,2 GB; swap 0.
- **Piloto** (`runs/pilot_lora_v3/20260923T012208Z`): época elegida 1 de 0–3 (validación 0,4222 frente a 0,4238 de la época 0). La pérdida de train bajó de 0,20 a 0,087 mientras la validación empeoraba en las épocas 2–3. La deriva de las características estandarizadas crece con las épocas (media |μ| por dimensión 0,06 → 0,17 → 0,37 → 0,50; desviación típica media 0,98 → 1,17 → 1,44 → 1,61). Test ciego y detalle en `reports/phase3-lora.md`.
- Tests CPU: `tests/unit/test_lora.py` y `tests/integration/test_phase3_pipeline.py` (reanudación bit a bit, recarga, calibración, test protegido, comparación).

## Fallo corregido durante la fase

`inject_adapter_in_model` inicializa `lora_A` con el RNG global. Dos runs idénticos en el mismo proceso divergían desde el paso 2 (en el paso 1 no se nota porque B = 0). Se detectó con el test de reanudación. Ahora `apply_lora(..., seed=…)` usa `torch.random.fork_rng` + semilla. Regresión: `test_lora_init_is_seeded_and_independent_of_global_rng`.

## Consecuencias y límites

- Coste de entrenamiento lineal en filas y unas 4–5 veces el de la extracción sin gradiente (3,6 frente a 16,6 filas/s).
- Estandarizador fijo: LoRA desplaza las representaciones (en el piloto, hasta 0,5 desviaciones de media y 1,6 de dispersión en la época 3; en la puerta, con lr 20× mayor, hasta ~2). No se ha medido si recalcularlo mejora; la época elegida (1) tiene deriva pequeña (0,17 / 1,17).
- Reproducibilidad: bit a bit en CPU entre ejecución continua y reanudada (test). En MPS no se ha comparado una ejecución continua con otra reanudada del mismo run largo; no se promete bit a bit entre dispositivos.
- No se ha probado FP16, gradient checkpointing, K > 6 ni contexto > 512 con autograd.
