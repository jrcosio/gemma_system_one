# Protocolo predeclarado de la fase 6f: generador v5 (definiciones excluyentes y opciones independientes de los hechos)

Fecha: 2026-09-24. Escrito **antes** de generar los datos v5 y de entrenar o evaluar ningún modelo sobre ellos. Su sha256 se guarda en `reports/phase6f/protocol.sha256`.

## Motivo

- **Hipótesis de la 6e:** la revisión dejó como hipótesis, no como causa demostrada, que la debilidad de `other` se debe a que la opción «aplicación» absorbe fallos de rendimiento y de datos. Los estados de esas categorías mencionan funciones de la app, y la definición de «aplicación» no los excluía.
- **Pendiente de la 6d:** la composición de las opciones del generador de entrenamiento v4 aporta información sobre la respuesta (0,050 de accuracy sólo con las opciones).
- **Indicación del revisor:** predeclarar definiciones y datos nuevos, usar el trío como unidad del IC y no reutilizar `trip15*`.

## Corrección (implementada y probada en CPU antes de este protocolo)

- **`support-mixed-v5`**, con `--generator-version v5`; v1–v4 siguen idénticos byte a byte:
  - **Definiciones excluyentes:**
    - «aplicación»: «Error, cierre o pantalla que no carga en una función (no incluye lentitud ni pérdida de datos)»;
    - «rendimiento»: «funciona, pero tarda mucho»;
    - la instrucción añade una cláusula de precedencia: «la lentitud es rendimiento y los datos perdidos o dañados son datos, aunque ocurran en una función de la aplicación».
  - **Composición independiente de los hechos:** siempre `none` y `other`, más K − 2 categorías (K de 3 a 6) sorteadas entre las 4 reales y las 3 distractoras sin mirar el estado. La etiqueta se deduce después.
- **Prueba sin modelos** (`reports/phase6f/option_cue_v5.txt`, 20 000 casos): ganancia por composición 0,012 (v4: 0,050); el residuo es semántico.
  - **Consecuencia conocida:** `other` pasa a 0,48 de las preguntas `fault_type` (v4: 0,15). Por eso **la métrica primaria son los tríos completos**, equilibrados por papel.
- **`fault_kind_triplets(version="v5")`:** tríos con las definiciones y la cláusula de v5.
- **Tests:** `tests/unit/test_phase6f_generator_v5.py`.

## Datos (se generan después de este protocolo)

Todos con v5 y `main`. Cada conjunto externo excluye los grupos cuyo estado literal aparezca en cualquier conjunto anterior (`pilot_v3`, `holdout6`, `calib7`, `final8`, `pilot_v4`, `calib12`, `final13`, `kdiag11`, `kdiag14`, `trip15`, `pilot_v5` y los v5 anteriores de esta lista).

| Conjunto | Semilla / casos | Uso |
|---|---|---|
| `data/pilot_v5` | 0 / 1000; `gso split --seed 0` | train y validation (selección de época). Su calibration y su test no se usan |
| `data/pilot_v5_calib16` | 16 / 400 | Sólo temperaturas (calibración externa) |
| `data/pilot_v5_final17` | 17 / 300 | Test final (parte b de la regla S) |
| `data/pilot_v5_trip18_K4` | 18 / 1000 → 150 tríos v5 (450 preguntas) | Métrica primaria (regla H) y parte a de la regla S |

Script: `scripts/derive_phase6f_data.py`.

## Modelos

- **A4v5:** E4B congelado + cabezales, `configs/e4b_v5.yaml`. Idéntico a `e4b_v4` salvo el dataset.
- **A2v5:** E2B congelado + cabezales, `configs/e2b_heads_v5.yaml`. Idéntico a `e2b_heads_v4` salvo el dataset.
- **A4v3:** el servicio actual, sin reentrenar.
- **Calibración:** los tres con `gso calibrate --split all --dataset data/pilot_v5_calib16`.

## Reglas

- **H, beneficio de v5.** Métrica: diferencia de la proporción de tríos completos **A4v5 − A4v3** en `trip18_K4`, emparejada por trío (bootstrap de 150 tríos, 1000 repeticiones, semilla 0, IC95 %).
  - Límite inferior > 0 ⇒ «v5 mejora el acierto conjunto de los tres papeles».
  - Si no ⇒ no demostrado.
- **S, servicio.** A4v5 sustituye a A4v3 en `configs/serve_e4b_text.yaml` sólo si se cumplen las dos partes, con un benchmark HTTP de confirmación:
  - **(a)** se cumple H;
  - **(b)** en `final17`, la diferencia de NLL calibrada A4v5 − A4v3 (`gso compare`, bootstrap por grupos) tiene el límite superior < +0,02.

  Si no, se mantiene A4v3.
- **Secundarias, sin regla:**
  - A2v5 − A4v3 y A2v5 frente a A4v5;
  - accuracy por papel (real, `other`, `none`);
  - número de errores `other` → «aplicación» y `other` → `none`;
  - métricas de `final17` por primitiva;
  - NLL sin calibrar.

## Lo que no se hace

- No se usan `trip15*`, `kdiag*`, `final8`, `final13`, `holdout6*` ni los tests de `pilot_v3`, `pilot_v4` y `pilot_v5` para decidir nada.
- Ningún hiperparámetro cambia respecto a v4. Si la época elegida es la tope, no se amplía.
- Un solo entrenamiento pesado a la vez.
