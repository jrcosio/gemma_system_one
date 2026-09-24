# Fase 6e: tríos con la misma pregunta (control coherente del uso del estado y diagnóstico de K)

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. La fase 6e y la revisión de la 6d quedaron en el commit `0a958df` de `main`, sobre `9edc183`; la revisión independiente actual contiene correcciones sin commit.

- **Protocolo predeclarado:** [phase6e-protocol.md](phase6e-protocol.md), sha256 `d33ede5a…` (en `reports/phase6e/protocol.sha256`). Se escribió antes de generar los datos y de evaluar.
- **Evidencia bruta:** `reports/phase6e/`.

## Revisión independiente

Se comprobaron los 150 tríos de K4 y K8 contra los hechos regenerados: 450 grupos distintos por conjunto, cero etiquetas incoherentes y cero diferencias en el prompt que no sean el estado dentro de cada trío. Los seis archivos de predicciones tienen 450 filas cada uno y sus indicadores `correct` coinciden con el argmax de los logits y la etiqueta. Una recarga de A4v3 real en MPS/BF16, sin caché, reprodujo exactamente los cuatro logits archivados para una pregunta K4; esto verifica inferencia, no gradientes E4B nuevos. Pasaron 56 pruebas CPU focalizadas en máscaras, prompts, pérdida de grupo, LoRA y extracción diminuta, y una prueba de gradientes LoRA con E2B real en MPS (sin omisión).

La comprobación de fuga externa de A4v3 informa cero solapes exactos y tres estados **casi duplicados** del entrenamiento (`data/pilot_v3`), todos en el papel `none` de tríos distintos. Al excluir esos tres tríos de K4, los tríos completos son A4v3 0,571, A4v4 0,497 y A2v4 0,211 (frente a 0,580/0,500/0,207); el hallazgo descriptivo de uso del estado no depende de ellos. No se reestima el IC predeclarado tras la exclusión ni se presenta la ausencia de solapes exactos como ausencia de toda similitud.

`scripts/analyze_triplets.py` aceptaba silenciosamente una fila duplicada del mismo papel: con cuatro filas para un trío informaba `questions=4` pero calculaba accuracy sobre tres después de sobrescribir una. Se corrigió para rechazar papeles duplicados, desconocidos, incompletos, entradas vacías y `correct` no booleano. Los seis JSONL archivados son válidos y el resultado histórico no cambia. El diagnóstico causal de «ambigüedad» queda como hipótesis, según los recuentos del veredicto y la sección de límites.

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

   El bootstrap por trío respeta la dependencia entre las tres preguntas que comparten opciones: A4v3 [−0,058; +0,011], A4v4 [−0,049; −0,004], A2v4 [−0,022; +0,051]. A4v4 cruza el margen por sólo 0,001 con esa unidad, mientras falla por 0,001 con el bootstrap por grupo predeclarado. Su clasificación es **sensible a la unidad de remuestreo**; no se debe presentar ninguna de las dos como conclusión estable. A2v4 cumple con ambas unidades.
3. **`other` es difícil y hay una posible superposición entre categorías, aún sin causa demostrada.**
   - En A4v3, los errores `other` → «aplicación» son 18/48 errores en K4 y 31/55 en K8; `other` → `none` son 26/48 y 11/55. La composición de errores cambia al añadir opciones, por lo que no se puede descartar un efecto de K.
   - Muchos de los errores hacia «aplicación» corresponden a fallos de rendimiento o datos. Algunas redacciones del estado mencionan la aplicación, mientras la categoría «Error en una pantalla o función de la aplicación» no excluye explícitamente lentitud ni pérdida de datos. **Esto sugiere** superposición semántica; los recuentos de predicciones no demuestran que sea la causa principal.
   - Un v5 con definiciones excluyentes es una hipótesis de mejora pendiente de protocolo, datos nuevos y evaluación; no se le atribuye una mejora medida.

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

**Código de las evaluaciones:** `c1ee7841…` (85 ficheros). El cierre publicado fue `a1e3f481…` (86 ficheros; añadió `scripts/analyze_triplets.py`). La revisión actual, aún sin commit, es `72f55732…` (86 ficheros; valida la entrada del analizador). Las dos primeras huellas tienen copia en `artifacts/source/`; la revisión no se usó para generar los logits históricos.

## Límites

- Una familia sintética; 150 tríos, con IC de unos ±0,03 en accuracy.
- El bootstrap por grupos no tiene en cuenta que las preguntas de un trío comparten opciones. La lectura de A4v4 cambia en el margen entre ese bootstrap predeclarado y el bootstrap por trío, que respeta la unidad del diseño.
- `trip15*` ya está usado.
- Las mismas redacciones potencialmente solapadas aparecen en los datos de entrenamiento v3 y v4; su efecto causal sobre los errores no se ha aislado.
