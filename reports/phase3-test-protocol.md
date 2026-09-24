# Protocolo predeclarado del test ciego de fase 3

Redactado el 2026-09-23, **antes** de conocer el resultado en validación del run LoRA del piloto y sin haber leído nunca la partición test de `pilot_v2` ni de `pilot_v3` (mismos grupos). Este documento fija qué se mide, cómo y con qué regla de decisión. No se modifica después de ver test; cualquier desviación se añadirá al final como desviación declarada.

## Artefactos que se congelan antes de test

| Rol | Checkpoint | Selección |
|---|---|---|
| A: E2B congelado + cabezales (referencia) | `runs/pilot_ce_v3/20260923T005721Z/checkpoint` | Época 28/30 por NLL de validación |
| B: LoRA + cabezales | `runs/pilot_lora_v3/<único run>/checkpoint` (`configs/pilot_lora_v3.yaml`) | Época 0–3 por NLL de validación (la 0 es exactamente A) |

- Si el run B falla por un motivo técnico (memoria, error), se reanuda con `--resume`; si hubiera que relanzarlo, se registra aquí el motivo. No se lanzarán variantes de hiperparámetros para elegir la mejor.
- Cada checkpoint se calibra **una vez** con `gso calibrate --checkpoint … --split calibration` (una temperatura por primitiva, T = softplus(t)+0,01, mínimo 30 preguntas). El artefacto queda vinculado por sha256.

## Ejecución en test (una vez por artefacto)

```bash
uv run gso evaluate --checkpoint <A> --split test --final-test --no-cache --calibration <cal A> --baselines
uv run gso evaluate --checkpoint <B> --split test --final-test --no-cache --calibration <cal B> --baselines
uv run gso compare --a <predicciones test A> --b <predicciones test B> --out reports/phase3/test_compare_lora_minus_frozen.json
```

## Métricas

- **Primaria:** NLL media por pregunta lógica sobre todas las preguntas de test (300 en 100 grupos), **con calibración**.
- **Comparación:** diferencia emparejada B − A por pregunta; IC95 % percentil por bootstrap de grupos (1000 réplicas, semilla 0, `gso compare`).
- **Regla de decisión:** «LoRA mejora» si el extremo superior del IC < 0; «LoRA empeora» si el inferior > 0; en otro caso, «sin diferencia concluyente». Se informa el resultado sea cual sea.
- **Secundarias (sin regla de decisión):** NLL y accuracy por primitiva; Brier, ECE (15 bins), RPS/MAE de Score; métricas sin calibrar y calibradas de A y B (spec §8, baseline 4: antes y después de calibrar); prior y BoW ajustados en train sobre la misma partición test.
- **Recarga:** el test se evalúa desde el texto en un proceso nuevo y sin caché.

## Después de test

- No se cambia diseño, hiperparámetros ni datos a la vista de test. Si se cambiara algo, habría que declararlo y reservar otro test (p. ej., un dataset nuevo con otra semilla).
- La transferencia (`data/pilot_transfer_v3`) se evalúa como diagnóstico, nunca para seleccionar.
- Límites que se mantienen: datos sintéticos con repertorio finito; 100 grupos dan intervalos anchos; la calibración por temperatura no corrige sesgos por subgrupo ni cambios de distribución.

## Desviaciones declaradas

Añadidas después de redactar el protocolo (sha256 del texto original, antes de esta sección: `a84f9220…c6de`, en `reports/phase3/test-protocol.sha256`). Ninguna cambia artefactos, hiperparámetros, métricas ni regla de decisión, y todas son anteriores a leer test.

1. **El run B se interrumpió dos veces y se reanudó** (`--resume`), por gestión de memoria, no por resultados. Sesión 1: la caché del asignador MPS llevó la memoria del driver a ~30 GB y el sistema a presión de memoria; se paró en el paso 185 y se reanudó desde el estado del paso 150. Sesión 2: se añadió `torch.mps.empty_cache()` tras cada paso; una muestra del proceso mostró un 35 % del hilo principal en esa llamada, así que se paró en el paso 242 y se reanudó desde el 200 con vaciado cada 10 pasos. Los 77 pasos rehechos están en `steps.discarded.jsonl`. Vaciar la caché libera bloques no usados y no debería cambiar los cálculos, pero no se ha comprobado bit a bit en MPS; el cambio es de código (`empty_mps_cache_every_steps`), no de la configuración.
2. El Mac entró 148 veces en reposo de mantenimiento durante el entrenamiento; sólo afecta al tiempo de reloj. Las evaluaciones posteriores se lanzaron con `caffeinate -i`.
3. El test de A se ejecutó con `--no-cache` y el de B sin la opción, porque los checkpoints LoRA nunca usan caché; ambos se calcularon desde el texto (`use_cache: false` en los dos informes).
