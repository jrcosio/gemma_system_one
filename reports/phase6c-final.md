# Fase 6c: generador v4 sin la pista de K, reentrenamiento de cabezales y diagnóstico de más opciones

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. Rama `fases-0-6`; la implementación quedó en `daa27ce` sobre `1b5ad45`.

- **Protocolo predeclarado:** [phase6c-protocol.md](phase6c-protocol.md), sha256 `02d3ae6c…` (en `reports/phase6c/protocol.sha256`). Se escribió antes de generar los datos v4 y de entrenar.
- **Decisión:** [0012](../docs/decisions/0012-generator-v4-without-k-cue.md).
- **Evidencia bruta:** `reports/phase6c/`.

## Veredicto

1. **Pista determinista de K del piloto corregida.** El generador v4 sortea `other` con la misma probabilidad condicional para K = 3–6 y permite K = 6 con esa etiqueta. v1–v3 siguen idénticos byte a byte. La revisión posterior encontró otra pista en la composición de las opciones ampliadas a K = 8; véase más abajo.
2. **Regla S: se mantiene A4v3 como servicio.**
   - Resultado: en el test final `pilot_v4_final13` (882 preguntas, 294 grupos), A4v4 − A4v3 = **−0,007 [−0,034; +0,024]** de NLL calibrada.
   - El límite superior supera el margen de no inferioridad (+0,02), así que por la regla no hay sustitución.
   - Sin calibrar, la diferencia observada es −0,007 [−0,040; +0,030]. Los intervalos no demuestran equivalencia ni no inferioridad con el margen declarado.
3. **Más opciones: resultados observados, con diagnóstico de robustez no concluyente tras la revisión.** Δaccuracy K8 − K: A4v3 0,000 [−0,040; +0,033], A4v4 −0,020 [−0,060; +0,020], A2v4 −0,147 [−0,207; −0,087]. La NLL empeora en los tres. La composición de K8 da una pista parcial sobre `other`, por lo que se retira la afirmación de que A4v3 tolera K8 en una prueba sin fuga.
4. **La ventaja de E4B se mantiene sin la pista:** A4v4 − A2v4 = −0,149 [−0,195; −0,105].

## Datos (generador v4, `main`)

Los conjuntos externos excluyen los grupos cuyo estado literal aparece en `pilot_v3`, `holdout6`, `calib7`, `final8`, `pilot_v4` o en los conjuntos v4 anteriores.

| Conjunto | Semilla / casos | Preguntas | Excluidos | sha256 |
|---|---|---|---|---|
| `data/pilot_v4` | 0 / 1000; split seed 0 | 3000 | — | `19af2c2e…d02c` |
| `data/pilot_v4_calib12` | 12 / 400 | 1155 | 15 grupos | `6ab5ce5e…3638` |
| `data/pilot_v4_final13` | 13 / 300 | 882 | 6 grupos | `132bea94…6c9b` |
| `data/pilot_v4_kdiag11` → `_K` / `_K8` | 11 / 300 → 150 `fault_type` | 150 / 150 | 11 grupos | `f5de139b…` / `7ad123cc…` |

- **Respuesta `other` por K** (`reports/phase6c/other_by_k.txt`):
  - `pilot_v4`: K3 10/131, K4 17/116, K5 21/125, K6 17/111;
  - `final13`: K3 8/37, K4 3/31, K5 7/27, K6 5/43;
  - `kdiag11_K8`: todos K8, 17/150 `other`.
- **Fugas:** `final13` da `errors: []` frente al entrenamiento y frente a `calib12`, con 1 y 3 pares casi duplicados respectivamente.

## Modelos

| Modelo | Checkpoint | Época (validación) | NLL validación | Extracción train / val (s) | Driver MPS | Swap delta |
|---|---|---|---|---|---|---|
| A4v4 (E4B + cabezales) | `runs/e4b_v4/20260924T140932Z` | 7 / 30 | 0,289 | 626 / 86 | 17,1 GB | **+1,49 GB** |
| A2v4 (E2B + cabezales) | `runs/e2b_heads_v4/20260924T142144Z` | 25 / 30 | 0,352 | 339 / 47 | 11,4 GB | 0 |
| A4v3 (servicio) | `runs/e4b_experiment/20260923T204945Z` | 7 (fase 6) | — | — | — | — |

- Mismos hiperparámetros que `e4b_experiment` y `pilot_ce_v3`; sólo cambia el dataset.
- El swap creció 1,49 GB durante el entrenamiento de A4v4, sin superar el presupuesto de memoria de MPS. No se ha atribuido a una causa (puede ser otra actividad del sistema).

**Temperaturas de `calib12`** (Noul / Choice / Score):

| Modelo | Noul | Choice | Score |
|---|---|---|---|
| A4v4 | 1,15 | 1,16 | 1,20 |
| A2v4 | 2,02 | 1,28 | 1,32 |
| A4v3 | 1,38 | 1,46 | 1,00 |

## Test final `pilot_v4_final13` (una evaluación por modelo)

| Modelo | NLL calibrada (sin calibrar) | Noul NLL · acc | Choice NLL · acc | Score NLL · acc | `fault_type` acc: `other` (23) / resto (115) |
|---|---|---|---|---|---|
| Prior / BoW | 1,021 / 0,906 | — | — | — | — |
| A2v4 | 0,398 (0,424) | 0,156 · 0,926 | 0,587 · 0,787 | 0,601 · 0,739 | 0,652 / 0,843 |
| **A4v4** | **0,249** (0,258) | 0,071 · 0,969 | 0,352 · 0,850 | 0,442 · 0,824 | 0,739 / 0,948 |
| **A4v3** | **0,256** (0,265) | 0,098 · 0,967 | 0,362 · 0,865 | 0,409 · 0,865 | 0,739 / 0,930 |

**Diferencias emparejadas** (bootstrap de 294 grupos, 1000 repeticiones, semilla 0):

| | NLL calibrada | Accuracy | Noul NLL | Choice NLL | Score NLL |
|---|---|---|---|---|---|
| **A4v4 − A4v3 (regla S)** | **−0,007 [−0,034; +0,024]** | −0,014 [−0,032; +0,003] | −0,028 [−0,065; +0,006] | −0,011 [−0,059; +0,042] | +0,034 [−0,044; +0,109] |
| A4v4 − A2v4 | −0,149 [−0,195; −0,105] | +0,060 [+0,035; +0,085] | −0,085 [−0,127; −0,044] | −0,235 [−0,321; −0,150] | −0,158 [−0,290; −0,021] |

Sin calibrar, A4v4 − A4v3 = −0,007 [−0,040; +0,030]. A4v3 se entrenó con datos v3, con la pista y sin las categorías distractoras. En este test v4 no se detecta una pérdida material de A4v3, pero el IC calibrado admite que A4v4 sea hasta 0,024 peor y supera el margen de +0,02. No se puede atribuir causalmente la diferencia observada al uso o desuso de la pista.

## Más opciones: K original (3–6) frente a K = 8, emparejado

Las 150 preguntas son las mismas y la etiqueta semántica también (`gso compare --allow-different-inputs`). Temperaturas de `calib12`.

| Modelo | NLL K → K8 | ΔNLL [IC] | **Δaccuracy [IC]** | Acc `other` (17) K → K8 | Acc resto (133) K → K8 | Elige una opción añadida |
|---|---|---|---|---|---|---|
| A4v3 | 0,203 → 0,260 | +0,057 [+0,005; +0,112] | **0,000 [−0,040; +0,033]** | 0,71 → 0,65 | 0,97 → 0,98 | 4 |
| A4v4 | 0,215 → 0,278 | +0,063 [−0,036; +0,158] | **−0,020 [−0,060; +0,020]** | 0,59 → 0,47 | 0,96 → 0,96 | 8 |
| A2v4 | 0,282 → 0,604 | +0,322 [+0,197; +0,452] | **−0,147 [−0,207; −0,087]** | 0,65 → 0,24 | 0,96 → 0,84 | 17 |

**Accuracy por K original** (K → K8):

| Modelo | K3 | K4 | K5 | K6 |
|---|---|---|---|---|
| A4v4 | 0,95 → 0,98 | 0,93 → 0,89 | 0,95 → 0,90 | 0,84 → 0,82 |
| A2v4 | 0,98 → 0,81 | 0,89 → 0,82 | 0,90 → 0,81 | 0,89 → 0,66 |
| A4v3 | 0,95 → 0,95 | 0,96 → 0,96 | 0,93 → 0,93 | 0,92 → 0,92 |

**Lectura:**
- Las cifras describen estas 150 preguntas emparejadas. La NLL aumenta con K8; la comparación no aísla el efecto del número de candidatos de la composición de las opciones.
- Con E2B, más opciones cuesta decisiones, sobre todo en `other`, que exige descartar todas las opciones listadas.
- Son 17 casos `other` en total: IC anchos.

### Revisión posterior: pista en la composición de K8 (2026-09-24)

`widen_fault_kind_by_facts` excluye la categoría verdadera y amplía hasta 8 opciones de un universo de 9. Para los 17 ejemplos con etiqueta `other`, llegar a K8 exige incluir las **tres** categorías distractoras que nunca son respuesta. En los 133 ejemplos restantes, 41 tienen sólo dos de esas distractoras. La tabla observada es: dos distractoras → 41 no `other`, 0 `other`; tres distractoras → 92 no `other`, 17 `other`. Además, si falta `none` en K8, la respuesta tampoco puede ser `other` (7 casos más con `other` presente). Así, parte de la etiqueta se infiere de las opciones sin leer el estado. El número K ya no la revela, pero la prueba K8 no está libre de pistas estructurales. Se conservan los datasets y predicciones históricos; una nueva prueba requiere otro protocolo y un universo de distractoras más amplio o un diseño que iguale la composición condicional.

## Comandos ejecutados

```bash
.venv/bin/pytest tests/unit/test_phase6c_generator_v4.py -q                      # 3 passed (antes del protocolo)
shasum -a 256 reports/phase6c-protocol.md > reports/phase6c/protocol.sha256
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind mixed --generator-version v4 --variant main --seed 0 --cases 1000 --out data/pilot_v4
.venv/bin/gso split --dataset data/pilot_v4 --seed 0
.venv/bin/python scripts/derive_phase6c_data.py                                  # calib12, final13, kdiag11{,_K,_K8}
for d in pilot_v4 pilot_v4_calib12 pilot_v4_final13 pilot_v4_kdiag11_K pilot_v4_kdiag11_K8; do .venv/bin/gso validate-data --dataset data/$d; done
reports/phase6c/run_chain.sh     # gso train (e4b_v4, e2b_heads_v4); 3 × gso calibrate --split all --dataset data/pilot_v4_calib12;
                                 # 9 × gso evaluate --split all --dataset data/pilot_v4_{final13,kdiag11_K,kdiag11_K8} --calibration <calib12>
.venv/bin/gso compare --a <final13 A4v3> --b <final13 A4v4> --out reports/phase6c/final13_compare_a4v4_minus_a4v3.json   # y a4v4−a2v4
.venv/bin/gso compare --a <kdiag K> --b <kdiag K8> --allow-different-inputs --out reports/phase6c/kdiag_K8_minus_K_<m>.json
.venv/bin/pytest tests/unit tests/integration -q          # 291 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs # 14 passed, 0 omitidos
```

Código de todas las ejecuciones: `dca04dbd…` (81 ficheros), con copia en `artifacts/source/`.

**Comprobación posterior del commit:** `ruff check . --statistics` devuelve 123 `E501` y `ruff format --check .` identifica cuatro archivos sin formato tras incorporar `src/gemma_system_one/data/` al seguimiento de Git. La afirmación previa de que Ruff pasó correspondía al árbol anterior, cuando `.gitignore` excluía esos módulos. No afecta a los números anteriores, pero el gate de estilo del commit no está superado.

## Límites

- Los datos siguen siendo sintéticos, de una sola familia; las categorías distractoras son sólo tres.
- Hay 17 casos `other` en el diagnóstico de K, así que los IC son anchos.
- B2 (LoRA) no se ha reentrenado con v4.
- `final13` y `kdiag11*` ya están usados. Tampoco deben usarse `final8` ni `holdout6*`.
