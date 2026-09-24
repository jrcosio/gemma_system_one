# Fase 6e: tríos con la misma pregunta (control coherente del uso del estado y diagnóstico de K)

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. Rama `main`, sobre `9edc183`. Sin commit: los cambios de esta fase y las correcciones de la revisión de la 6d.

- **Protocolo predeclarado:** [phase6e-protocol.md](phase6e-protocol.md), sha256 `d33ede5a…` (en `reports/phase6e/protocol.sha256`). Se escribió antes de generar los datos y de evaluar.
- **Evidencia bruta:** `reports/phase6e/`.

## Veredicto

1. **Los tres modelos usan el estado** (control coherente; techo sin estado: accuracy 1/3 y 0 tríos completos).

   | Modelo | Accuracy K4 | Tríos completos K4 |
   |---|---|---|
   | A4v3 | 0,847 | **0,58 [0,51; 0,66]** |
   | A4v4 | 0,822 | **0,50 [0,42; 0,58]** |
   | A2v4 | 0,684 | **0,21 [0,14; 0,27]** |

   El criterio (límite inferior > 0,10) se cumple en los tres.
2. **Tolerancia a K = 8** (criterio: límite inferior del IC de Δaccuracy ≥ −0,05):

   | Modelo | Δaccuracy [IC] | Lectura |
   |---|---|---|
   | A4v3 | −0,024 [−0,060; +0,011] | No demostrada |
   | A4v4 | −0,027 [−0,051; −0,002] | No demostrada, por 0,001 |
   | A2v4 | +0,013 [−0,024; +0,053] | **Tolera** |

   Con un bootstrap por trío, más conservador, el resultado es casi igual: A4v3 [−0,058; +0,011], A4v4 [−0,049; −0,004], A2v4 [−0,022; +0,051]. Así, A4v4 cumpliría por 0,001, pero la lectura predeclarada es la del bootstrap por grupos.
3. **La debilidad de `other` es sobre todo una ambigüedad de las categorías, no del número de opciones.**
   - **El error:** el más frecuente con respuesta `other` es elegir la opción de aplicación (A4v3 en K8: 31 de 55 errores).
   - **Qué estados fallan:** sobre todo los de rendimiento y de datos. Muchos estados de estas categorías se redactan como fallos de la aplicación («la aplicación va extremadamente lenta…»), y la opción «Error en una pantalla o función de la aplicación» no los excluye.
   - **Siguiente paso:** es un fallo de la definición de la tarea en el generador, que se corregiría con definiciones mutuamente excluyentes en un v5.

## Diseño y datos

- **`derive.fault_kind_triplets`:** cada trío comparte exactamente la misma pregunta (instrucción, textos, IDs y orden de las opciones `none`, `other` y dos categorías reales) y difiere sólo en el estado:
  - real: el fallo es de una categoría listada;
  - `other`: el fallo es de una categoría real no listada;
  - `none`: no hay fallo.

  Los tres estados salen de grupos distintos del mismo idioma y cada etiqueta sale de los hechos de su propio estado. K8 añade las mismas cuatro distractoras de la fase 6d. Tests: `tests/unit/test_phase6e_triplets.py`.
- **`scripts/analyze_triplets.py`:** accuracy por papel y tríos completos, con bootstrap por trío.

| Conjunto | Contenido | Preguntas | sha256 |
|---|---|---|---|
| `data/pilot_v4_trip15` | v4, seed 15, 1000 casos; 38 grupos excluidos por solape con todos los conjuntos anteriores | 2886 | `5380531f…` |
| `data/pilot_v4_trip15_K4` / `_K8` | 150 tríos (150 real, 150 `other`, 150 `none`) | 450 / 450 | `aa5c9c78…` / `272314ae…` |

Modelos: A4v3 (servicio), A4v4 y A2v4, sin reentrenar, con sus temperaturas de `calib12`.

## Resultados

| Modelo | K | Accuracy [IC por trío] | real | `other` | `none` | Tríos completos [IC] | NLL |
|---|---|---|---|---|---|---|---|
| A4v3 | 4 | 0,847 [0,816; 0,878] | 0,92 | 0,68 | 0,94 | 0,58 [0,51; 0,66] | 0,426 |
| A4v3 | 8 | 0,822 [0,789; 0,856] | 0,98 | 0,63 | 0,85 | 0,53 [0,45; 0,61] | 0,448 |
| A4v4 | 4 | 0,822 [0,793; 0,851] | 0,95 | 0,57 | 0,95 | 0,50 [0,42; 0,58] | 0,485 |
| A4v4 | 8 | 0,796 [0,760; 0,827] | 0,96 | 0,54 | 0,89 | 0,46 [0,38; 0,54] | 0,613 |
| A2v4 | 4 | 0,684 [0,651; 0,718] | 0,87 | 0,41 | 0,77 | 0,21 [0,14; 0,27] | 0,866 |
| A2v4 | 8 | 0,698 [0,660; 0,738] | 0,89 | 0,45 | 0,76 | 0,29 [0,21; 0,37] | 0,791 |

**K8 − K4, emparejado** (`gso compare`, bootstrap de 450 grupos, 1000 repeticiones, semilla 0):

| Modelo | ΔNLL [IC] | Δaccuracy [IC] |
|---|---|---|
| A4v3 | +0,022 [−0,020; +0,063] | −0,024 [−0,060; +0,011] |
| A4v4 | +0,129 [+0,056; +0,201] | −0,027 [−0,051; −0,002] |
| A2v4 | −0,074 [−0,161; +0,013] | +0,013 [−0,024; +0,053] |

**Accuracy con respuesta `other`, según la categoría real no listada** (K4):

| Modelo | aplicación | datos | red | rendimiento |
|---|---|---|---|---|
| A4v3 | 32/35 | 25/43 | 30/42 | **15/30** |
| A4v4 | 26/35 | 19/43 | 30/42 | **10/30** |
| A2v4 | 20/35 | 17/43 | 15/42 | **9/30** |

**Errores más frecuentes en K8** (papel → categoría elegida; `reports/phase6e/sensitivity_and_errors.txt`):

| Modelo | Errores más frecuentes |
|---|---|
| A4v3 | `other` → aplicación 31; `none` → `other` 16; `other` → `none` 11 |
| A4v4 | `other` → aplicación 36; `other` → `none` 14; `none` → `other` 11 |
| A2v4 | `other` → `none` 53; `none` → `other` 26; `other` → aplicación 20 |

- **E2B y E4B fallan de forma distinta:** E2B confunde «hay un fallo no listado» con «no hay fallo», mientras que E4B asigna el fallo a la categoría de aplicación.
- **Con K8, `none` baja en E4B** (0,94 → 0,85 en A4v3): parte de los estados sin fallo se asignan a `other`.

## Comandos ejecutados

```bash
.venv/bin/pytest tests/unit/test_phase6e_triplets.py -q                     # 1 passed (antes del protocolo)
shasum -a 256 reports/phase6e-protocol.md > reports/phase6e/protocol.sha256
.venv/bin/python scripts/derive_phase6e_data.py
for d in trip15 trip15_K4 trip15_K8; do .venv/bin/gso validate-data --dataset data/pilot_v4_$d; done
reports/phase6e/run_chain.sh     # 6 × gso evaluate --split all --dataset data/pilot_v4_trip15_{K4,K8} --calibration <calib12>
.venv/bin/python scripts/analyze_triplets.py reports/phase6e/triplets.json <modelo_K>=<predicciones> …
.venv/bin/gso compare --a <K4> --b <K8> --allow-different-inputs --out reports/phase6e/K8_minus_K4_<m>.json
.venv/bin/pytest tests/unit tests/integration -q          # 294 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs # 14 passed, 0 omitidos
```

El bootstrap por trío y el análisis de errores se calcularon sin volver a usar los modelos, a partir de las predicciones, y están en `reports/phase6e/sensitivity_and_errors.txt`.

**Código de las evaluaciones:** `c1ee7841…`. El actual, `a1e3f481…` (86 ficheros), sólo añade `scripts/analyze_triplets.py`. Ambos tienen copia en `artifacts/source/`.

## Límites

- Una familia sintética; 150 tríos, con IC de unos ±0,03 en accuracy.
- El bootstrap por grupos no tiene en cuenta que las preguntas de un trío comparten opciones; la sensibilidad por trío da casi lo mismo.
- `trip15*` ya está usado.
- La ambigüedad entre categorías afecta también a los datos de entrenamiento v3 y v4.
