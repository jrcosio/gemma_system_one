# Fase 6f: generador v5 (definiciones excluyentes y opciones independientes de los hechos)

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. Rama `main`, sobre `0a958df`. Sin commit: los cambios de esta fase y las correcciones de la revisión de la 6e.

- **Protocolo predeclarado:** [phase6f-protocol.md](phase6f-protocol.md), sha256 `7a995f7f…` (en `reports/phase6f/protocol.sha256`). Se escribió antes de generar los datos v5 y de entrenar.
- **Decisión:** [0013](../docs/decisions/0013-generator-v5-exclusive-definitions.md).
- **Evidencia bruta:** `reports/phase6f/`.

## Veredicto

1. **Regla H: se cumple.** Entrenar con v5 mejora el acierto conjunto de los tres papeles en tríos v5 nuevos: tríos completos A4v5 − A4v3 = **+0,087 [+0,020; +0,160]** (bootstrap de 150 tríos, 1000 repeticiones, semilla 0).
2. **Regla S: no se cumple, así que el servicio sigue siendo A4v3.**
   - En `final17`, A4v5 − A4v3 = −0,007 [−0,037; +0,025] de NLL calibrada.
   - El límite superior supera el margen de no inferioridad (+0,02), igual que en la fase 6c.
   - A4v5 queda como alternativa: mejor en tríos, sin diferencia demostrada en NLL.
3. **Descriptivo, sin causalidad demostrada:**
   - **Con las definiciones excluyentes, el error `other` → «aplicación» casi desaparece:** 1 (A4v3) y 3 (A4v5) en `trip18`, frente a 31 de 55 errores de A4v3 en `trip15` (v4).
   - **La mejora aparece también en A4v3 sin reentrenar:** tríos completos 0,71 en `trip18` frente a 0,58 en `trip15`. Son conjuntos distintos, no emparejados.
   - **El error residual dominante en E4B** está en los fallos resueltos con respuesta `other`: A4v5 falla 13 de 48, frente a 2 de 102 con fallo activo. La hipótesis es que se lee «fallo ya resuelto» como «no se describe ningún fallo».

## Corrección

- **`support-mixed-v5`** (decisión 0013):
  - «aplicación» = «Error, cierre o pantalla que no carga en una función (no incluye lentitud ni pérdida de datos)»;
  - «rendimiento» = «funciona, pero tarda mucho»;
  - cláusula de precedencia en la instrucción;
  - opciones: siempre `none` y `other` más K − 2 categorías sorteadas sin mirar los hechos.

  v1–v4 siguen idénticos byte a byte.
- **Prueba sin modelos** (`reports/phase6f/option_cue_v5.txt`, 20 000 casos): ganancia por composición de 0,012 en v5 (v4: 0,050); `other` pasa a 0,478 de las preguntas `fault_type` (v4: 0,148).
- **`derive.fault_kind_triplets(version="v5")`** y **`scripts/compare_triplets.py`** (emparejado por trío; reutiliza la validación de `analyze_triplets.load_triplets`).
- **Tests:** `tests/unit/test_phase6f_generator_v5.py` y `tests/unit/test_phase6e_triplets.py::test_compare_triplets_pairs_by_triplet`.

## Datos (generador v5, `main`)

| Conjunto | Semilla / casos | Preguntas | Excluidos | sha256 |
|---|---|---|---|---|
| `data/pilot_v5` | 0 / 1000; split seed 0 | 3000 | — | `7b12f921…` |
| `data/pilot_v5_calib16` | 16 / 400 | 1140 | 20 grupos | `ce833663…` |
| `data/pilot_v5_final17` | 17 / 300 | 837 | 21 grupos | `d5e0db49…` |
| `data/pilot_v5_trip18` → `_K4` | 18 / 1000 → 150 tríos | 2868 → 450 | 44 grupos | `fdd0ec2d…` / `0d70be1c…` |

Los conjuntos externos excluyen los grupos cuyo estado literal aparece en cualquier conjunto anterior. `final17` da `errors: []` frente al entrenamiento y frente a `calib16`.

## Modelos

| Modelo | Checkpoint | Época | NLL validación | Extracción train / val (s) | Driver MPS | Swap delta | T (`calib16`) Noul / Choice / Score |
|---|---|---|---|---|---|---|---|
| A4v5 (E4B) | `runs/e4b_v5/20260924T184606Z` | 20 / 30 | 0,231 | 639 / 98 | 17,1 GB | +0,6 MB | 2,37 / 1,63 / 1,28 |
| A2v5 (E2B) | `runs/e2b_heads_v5/20260924T185841Z` | 18 / 30 | 0,464 | 348 / 53 | 11,4 GB | 0 | 1,42 / 1,48 / 1,45 |
| A4v3 (servicio) | `runs/e4b_experiment/20260923T204945Z` | — | — | — | — | — | 1,88 / 1,27 / 0,93 |

## Resultados

### Tríos v5 (`trip18_K4`, 150 tríos; bootstrap por trío)

| Modelo | Accuracy [IC] | real | `other` | `none` | Tríos completos [IC] |
|---|---|---|---|---|---|
| A4v3 | 0,896 [0,864; 0,922] | 0,96 | 0,87 | 0,86 | 0,713 [0,633; 0,787] |
| **A4v5** | **0,922** [0,896; 0,949] | 0,96 | 0,90 | 0,91 | **0,800** [0,740; 0,860] |
| A2v5 | 0,764 [0,729; 0,802] | 0,89 | 0,81 | 0,59 | 0,427 [0,347; 0,507] |

**Diferencias emparejadas por trío** (`reports/phase6f/H_trip18_a4v5_minus_a4v3.json`):

| | A4v5 − A4v3 | A2v5 − A4v3 |
|---|---|---|
| Tríos completos (regla H) | **+0,087 [+0,020; +0,160]** | −0,287 [−0,393; −0,187] |
| Accuracy | +0,027 [+0,002; +0,053] | — |
| `other` | +0,033 [−0,013; +0,080] | −0,053 [−0,133; +0,027] |
| `none` | +0,047 [0,000; +0,093] | — |
| real | 0,000 [−0,033; +0,040] | — |

### Test final `final17` (837 preguntas, 279 grupos)

| Modelo | NLL calibrada (sin calibrar) | Noul NLL · acc | Choice NLL · acc | Score NLL · acc |
|---|---|---|---|---|
| A4v3 | 0,256 (0,265) | 0,104 · 0,969 | 0,435 · 0,828 | 0,278 · 0,914 |
| **A4v5** | **0,249** (0,305) | 0,107 · 0,969 | 0,389 · 0,869 | 0,304 · 0,905 |
| A2v5 | 0,453 (0,488) | 0,167 · 0,941 | 0,690 · 0,737 | 0,625 · 0,781 |

| | NLL calibrada | Accuracy | Choice NLL | Choice acc |
|---|---|---|---|---|
| **A4v5 − A4v3 (regla S b)** | **−0,007 [−0,037; +0,025]** | +0,011 [−0,005; +0,026] | −0,047 [−0,118; +0,025] | +0,040 [0,000; +0,086] |
| A4v5 − A2v5 | −0,204 [−0,267; −0,140] | +0,086 [+0,059; +0,111] | −0,302 [−0,401; −0,204] | +0,131 [+0,081; +0,183] |

A4v5 sin calibrar (0,305) es peor que calibrado (0,249): sus temperaturas de `calib16` son altas, con Noul en 2,37.

### Errores en `trip18` (`reports/phase6f/errors_trip18.txt`)

| Modelo | Errores más frecuentes (papel → elegida) | `other` con fallo activo / resuelto (errores) |
|---|---|---|
| A4v3 | `none` → `other` 16; `other` → `none` 16; real → `other` 6; `other` → aplicación **1** | 6/102 · **14/48** |
| A4v5 | `none` → `other` 12; `other` → `none` 12; `other` → aplicación **3** | 2/102 · **13/48** |
| A2v5 | `none` → `other` 41; `other` → `none` 15; real → `other` 14; `other` → aplicación 7 | 16/102 · 12/48 |

- **`none` y el acceso a la cuenta:** en E2B, con cuenta bloqueada, el papel `none` falla 26 de 50, así que confunde el bloqueo con un fallo técnico pese a la cláusula. En E4B, falla 3 de 50 con cuenta bloqueada y 8 de 74 sin mención de acceso.

## Comandos ejecutados

```bash
.venv/bin/pytest tests/unit/test_phase6f_generator_v5.py tests/unit/test_phase6e_triplets.py tests/unit/test_phase6c_generator_v4.py -q  # antes del protocolo
.venv/bin/python - …                         # sonda sin modelos v4/v5 → reports/phase6f/option_cue_v5.txt
shasum -a 256 reports/phase6f-protocol.md > reports/phase6f/protocol.sha256
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind mixed --generator-version v5 --variant main --seed 0 --cases 1000 --out data/pilot_v5
.venv/bin/gso split --dataset data/pilot_v5 --seed 0
.venv/bin/python scripts/derive_phase6f_data.py
for d in pilot_v5 pilot_v5_calib16 pilot_v5_final17 pilot_v5_trip18 pilot_v5_trip18_K4; do .venv/bin/gso validate-data --dataset data/$d; done
reports/phase6f/run_chain.sh     # train e4b_v5, e2b_heads_v5; 3 × calibrate --dataset calib16; 6 × evaluate (final17, trip18_K4)
.venv/bin/python scripts/compare_triplets.py reports/phase6f/H_trip18_a4v5_minus_a4v3.json <trip18 A4v3> <trip18 A4v5>
.venv/bin/python scripts/analyze_triplets.py reports/phase6f/triplets.json a4v3=… a4v5=… a2v5=…
.venv/bin/gso compare --a <final17 A4v3> --b <final17 A4v5> --out reports/phase6f/final17_compare_a4v5_minus_a4v3.json
.venv/bin/pytest tests/unit tests/integration -q          # 301 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs # 14 passed, 0 omitidos
```

**Código de todas las ejecuciones:** `8d7c7d74…` (90 ficheros), el actual, con copia en `artifacts/source/`.

## Límites

- Los datos son sintéticos, de una familia.
- La proporción de `other` en v5 (0,48) difiere mucho de v4; por eso la métrica primaria son los tríos equilibrados, pero `final17` refleja ese sesgo.
- La comparación con `trip15` (v4) es descriptiva, con otros estados.
- `pilot_v5_final17` y `trip18*` ya están usados. El test y la partición de calibración de `pilot_v5` no se han leído.
