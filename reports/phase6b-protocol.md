# Protocolo predeclarado de la fase 6b: test final independiente del servicio elegido y calibración de A4

Fecha: 2026-09-24. Escrito **antes** de generar los conjuntos nuevos y de ejecutar cualquier modelo sobre ellos. Su sha256 se guarda en `reports/phase6b/protocol.sha256`.

## Motivo

La revisión de la fase 6 (en `docs/STATUS.md`) deja dos pendientes:

1. **Selección y medida en el mismo conjunto.** La regla R2 usó `pilot_v3_holdout6_clean` para elegir A4 como servicio. Ese holdout pasa a ser un conjunto de selección, y falta una medida independiente posterior a la elección.
2. **Calibración de A4.** Sus temperaturas, ajustadas con las 300 preguntas de la partición `calibration`, empeoraron la NLL del holdout (0,174 → 0,200).

Elegir en validación entre «calibrar» y «no calibrar» estaría sesgado hacia no calibrar: la época de A4 se escogió minimizando la NLL de validación sin calibrar. El remedio sin selección es estimar las temperaturas con más datos, mediante un procedimiento fijado aquí.

## Datos (se generan después de escribir este protocolo)

| Conjunto | Generación | Uso |
|---|---|---|
| `data/pilot_v3_calib7` | `generate-data --kind mixed --generator-version v3 --variant main --seed 7 --cases 400`, sin los grupos con estado literal presente en `pilot_v3` o `pilot_v3_holdout6` | **Sólo** ajuste de temperaturas (calibración externa) |
| `data/pilot_v3_final8` | Ídem con `--seed 8 --cases 300`, sin los grupos con estado literal presente en `pilot_v3`, `pilot_v3_holdout6` o `pilot_v3_calib7` | **Test final.** Una evaluación por modelo, al final; no se ajusta nada con él |

Derivación: `scripts/derive_phase6b_data.py`. Los sha256 se registran en el informe.

## Modelos y procedimiento fijo

- **Checkpoints:** A2 (`runs/pilot_ce_v3/20260923T005721Z`), B2 (`runs/pilot_lora_v3/20260923T012208Z`) y A4 (`runs/e4b_experiment/20260923T204945Z`), sin reentrenar.
- **Calibración nueva de cada uno:** `gso calibrate --checkpoint … --split all --dataset data/pilot_v3_calib7`, con el mismo método (una T por primitiva; softplus + LBFGS).
- **Servicio:** la calibración que usará el servicio de A4 (y de B2) es la de `calib7`. Se fija aquí y no se elige después.

## Métrica primaria y regla

- **Primaria:** NLL media por pregunta en `pilot_v3_final8`, calibrada con `calib7`. Diferencia emparejada **A4 − B2** con `gso compare`: bootstrap por grupos, 1000 repeticiones, semilla 0, IC95 %.
- **Regla C:**
  - límite superior < 0 ⇒ **se confirma A4** como servicio de texto recomendado;
  - IC que contiene 0 ⇒ **no se confirma** y la recomendación vuelve a B2, más barato;
  - límite inferior > 0 ⇒ se revierte a B2.
- **Secundarias, sin regla:**
  - A4 − A2 y B2 − A2;
  - la misma comparación sin calibrar y con las temperaturas antiguas (300 preguntas);
  - NLL, accuracy, Brier/ECE y RPS por primitiva.
- **Calibración de A4** (se informa y no se decide nada con ello): NLL en `final8` sin temperatura, con T de 300 preguntas y con T de `calib7`. Se declara «resuelto» sólo si T(`calib7`) no empeora la NLL sin calibrar en `final8`.

## Otros

- Se repite en MPS el perfil de 6 pasos LoRA de E4B con recomputación, para ejercitar la corrección del revisor en `profile_lora_step.py`. Es un diagnóstico; no se entrena LoRA sobre E4B.
- `pilot_v3_holdout6*`, el test de `pilot_v3` y los tests visuales siguen sin usarse para decidir nada.
