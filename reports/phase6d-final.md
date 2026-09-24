# Fase 6d: diagnóstico de más opciones con composición equilibrada y control sin estado

Fecha: 2026-09-24. Mac M5 Pro con 48 GB, MPS/BF16 y cabezales CPU/FP32. Rama `fases-0-6`, sobre `daa27ce`; los cambios de esta fase están sin commit.

- **Protocolo predeclarado:** [phase6d-protocol.md](phase6d-protocol.md), sha256 `474a32b4…` (en `reports/phase6d/protocol.sha256`). Se escribió antes de generar los datos y de evaluar.
- **Evidencia bruta:** `reports/phase6d/`.

## Veredicto

1. **La pista de composición del diagnóstico K8 está corregida.**
   - Con `balanced_fault_kind_pairs`, la estructura de las opciones es la misma para cualquier etiqueta. La información sobre la respuesta que aporta la composición es **0,000** (sin modelos, 20 000 casos), frente a 0,019 en la fase 6c y 0,141 en la fase 6.
   - El control sin estado lo confirma con los modelos: con estados intercambiados, la accuracy (0,20–0,26) queda **por debajo** del prior del conjunto (0,347). No se explota ninguna pista.
2. **Tolerancia a K = 8: no demostrada para ningún modelo** (criterio: límite inferior del IC de Δaccuracy ≥ −0,05):
   - A4v3: −0,036 [−0,096; +0,024];
   - A4v4: −0,024 [−0,078; +0,030];
   - A2v4: −0,012 [−0,072; +0,054].
   Con 167 preguntas, los IC no descartan pérdidas de hasta 0,08–0,10. La respuesta `other` es el punto débil: con K8, su accuracy baja en todos los modelos.
3. **Composición del generador de entrenamiento v4:**
   - Sin modelos, deja 0,050 de información sobre la respuesta (v3: 0,029). Parte es inferencia legítima: si están todas las categorías reales, `other` es imposible.
   - Los modelos no la aprovechan más allá del prior. En `final13` con estados intercambiados aciertan 0,38–0,41, por debajo del 0,427 alcanzable sólo con el prior.
   - Aun así es una debilidad del generador, pendiente de corregir en una versión v5.

## Diseño

- **`derive.balanced_fault_kind_pairs`:**
  - K4: `none`, `other` y dos categorías reales. Si la respuesta es `other`, ninguna de las dos es la verdadera; si no, una sí.
  - K8: K4 más siempre las cuatro distractoras, que nunca son la respuesta: «hardware», «instalación», «notificaciones» y la nueva «accesibilidad», comprobada por palabras clave; «screen» se evitó porque aparece en los estados.
  - `other` se sortea con probabilidad 0,5 si hay fallo. El par de categorías reales es uniforme con cualquier etiqueta.
- **`derive.swap_states`:** cada pregunta recibe el estado de otro grupo (derangement), con las mismas opciones y la misma etiqueta.
- **Tests** (`tests/unit/test_phase6d_balanced_k.py`):
  - estructura independiente de la etiqueta, respuestas coherentes con los hechos, par uniforme y proporción de `other` ≈ 0,5, con 3000 casos;
  - el intercambio de estados es un derangement determinista.
- **`scripts/probe_option_cue.py`:** accuracy máxima alcanzable sin leer el estado, con el prior de la tarea y con cada composición exacta (`reports/phase6d/option_cue_probe.json`).

| Diseño (20 000 casos) | Acc. sólo con el prior | Acc. máxima sólo con las opciones | Ganancia por composición |
|---|---|---|---|
| Generador v3 (K 3–6) | 0,322 | 0,351 | 0,029 |
| Generador v4 (K 3–6) | 0,427 | 0,477 | 0,050 |
| K8 de la fase 6 (con la etiqueta) | 0,260 | 0,401 | 0,141 |
| K8 de la fase 6c (con los hechos) | 0,275 | 0,294 | 0,019 |
| **Equilibrado K4 / K8 (6d)** | 0,377 | 0,377 | **0,000** |

En conjuntos pequeños, la «máxima» calculada dentro de la propia muestra sobreajusta: `reports/phase6d/option_cue_actual_sets.txt` da, por ejemplo, 0,93 en `final13`, con 120 composiciones para 138 preguntas. Por eso se usan las cifras poblacionales.

## Datos

| Conjunto | Contenido | Preguntas | sha256 |
|---|---|---|---|
| `data/pilot_v4_kdiag14` | v4, seed 14, 400 casos; 12 grupos excluidos por solape con todos los conjuntos anteriores | 1164 | `b0273c33…` |
| `_K4` / `_K8` | `fault_type` equilibradas (58 `other`, 109 resto) | 167 / 167 | `e1a9cd92…` / `bd7af646…` |
| `_K4swap` / `_K8swap` | Las mismas, con estados intercambiados | 167 / 167 | `a498e79d…` / `b7093b49…` |
| `data/pilot_v4_final13_faultswap` | `fault_type` de `final13` con estados intercambiados (control descriptivo) | 138 | `4b6d7e18…` |

Modelos: A4v3 (servicio), A4v4 y A2v4, sin reentrenar, con sus temperaturas de `calib12`.

## Resultados

### Más opciones: K4 frente a K8, emparejado (bootstrap de 167 grupos, 1000 repeticiones, semilla 0)

| Modelo | NLL K4 → K8 | ΔNLL [IC] | **Δaccuracy [IC]** | Acc K4 → K8 | Acc `other` (58) K4 → K8 | Acc resto (109) K4 → K8 |
|---|---|---|---|---|---|---|
| A4v3 | 0,489 → 0,516 | +0,027 [−0,040; +0,104] | **−0,036 [−0,096; +0,024]** | 0,814 → 0,778 | 0,64 → 0,50 | 0,91 → 0,93 |
| A4v4 | 0,666 → 0,762 | +0,096 [−0,020; +0,218] | **−0,024 [−0,078; +0,030]** | 0,766 → 0,743 | 0,48 → 0,43 | 0,92 → 0,91 |
| A2v4 | 1,032 → 0,995 | −0,037 [−0,204; +0,117] | **−0,012 [−0,072; +0,054]** | 0,629 → 0,617 | 0,45 → 0,36 | 0,73 → 0,75 |

- **Tolerancia:** no demostrada según el criterio.
- **Aciertos:** son menores que en los diagnósticos anteriores; A4v3 en `other` tenía 0,71 en `kdiag11_K`. Aquí `none` y `other` están siempre presentes y la composición no ayuda.
- **`other`:** decidir que ninguna opción listada aplica es lo más difícil, y empeora con más distractoras.

### Control sin estado (mismas opciones y etiquetas, estado de otro grupo)

| Modelo | `K4swap` | `K8swap` | `final13_faultswap` |
|---|---|---|---|
| A4v3 | 0,210 | 0,234 | 0,406 |
| A4v4 | 0,204 | 0,234 | 0,384 |
| A2v4 | 0,222 | 0,263 | 0,377 |
| Cota sólo con las opciones | 0,347 (prior del conjunto; poblacional 0,377) | ídem | prior 0,427 / máxima 0,477 (v4, poblacional) |

- **Lectura predeclarada 2:** ninguna accuracy supera la cota + 0,10, así que no hay pistas no identificadas. Quedan incluso por debajo de la cota: con el estado cambiado, los modelos responden según el estado recibido.
- **Lectura 3:** en `final13_faultswap`, los modelos no superan lo alcanzable sólo con el prior. En los casos `other` aciertan 0,78 con estados cambiados porque la categoría real del estado donante tampoco suele estar entre las opciones, así que `other` sigue siendo semánticamente correcto; no es una pista del generador.

## Comandos ejecutados

```bash
.venv/bin/pytest tests/unit/test_phase6d_balanced_k.py -q                       # 2 passed (antes del protocolo)
.venv/bin/python scripts/probe_option_cue.py reports/phase6d/option_cue_probe.json 20000
shasum -a 256 reports/phase6d-protocol.md > reports/phase6d/protocol.sha256
.venv/bin/python scripts/derive_phase6d_data.py
for d in kdiag14 kdiag14_K4 kdiag14_K8 kdiag14_K4swap kdiag14_K8swap final13_faultswap; do .venv/bin/gso validate-data --dataset data/pilot_v4_$d; done
reports/phase6d/run_chain.sh     # 15 × gso evaluate --split all --dataset … --calibration <calib12>
.venv/bin/gso compare --a <K4> --b <K8> --allow-different-inputs --out reports/phase6d/K8_minus_K4_<m>.json
.venv/bin/pytest tests/unit tests/integration -q          # 293 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs # 14 passed, 0 omitidos
```

**Código de las evaluaciones:** `fd340105…` (83 ficheros), con copia en `artifacts/source/`.

**Configuración de ruff:**
- el `.gitignore` corregido en `daa27ce` hizo visibles para ruff los generadores, que nunca se habían revisado;
- se añadió en `pyproject.toml` una excepción E501 y de formato para `generate.py`, `generate_mixed.py` y `generate_vision.py`, para no tocar el código que produce los datasets con hash registrado;
- `split.py` sólo cambió en una línea partida, sin cambio de comportamiento.

## Límites

- Sólo hay 167 preguntas (58 `other`), así que los IC son anchos.
- Una familia sintética y cuatro distractoras fijas.
- El generador de entrenamiento v4 conserva 0,050 de información por composición (pendiente de un v5).
- `kdiag14*` y `final13_faultswap` ya están usados.
