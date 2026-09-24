# Protocolo predeclarado de la fase 6d: diagnóstico de más opciones con composición equilibrada y control sin estado

Fecha: 2026-09-24. Escrito **antes** de generar los datos de esta fase y de evaluar ningún modelo sobre ellos. Su sha256 se guarda en `reports/phase6d/protocol.sha256`.

## Fallo que se corrige

La revisión de la fase 6c detectó que, en el diagnóstico K8 (`widen_fault_kind_by_facts`, 8 de 9 categorías), la composición de las opciones daba una pista parcial sobre `other`. La prueba sin modelos (`scripts/probe_option_cue.py`, `reports/phase6d/option_cue_probe.json`, 20 000 casos) mide la ganancia de accuracy alcanzable sólo con la composición de las opciones:

| Diseño | Ganancia |
|---|---|
| K8 de la fase 6 (ampliación con la etiqueta) | 0,141 |
| K8 de la fase 6c (ampliación con los hechos) | 0,019 |
| Generador de entrenamiento v3 (K 3–6) | 0,029 |
| Generador de entrenamiento v4 (K 3–6) | 0,050 |

## Diseño (implementado y probado en CPU antes de este protocolo)

- **`derive.balanced_fault_kind_pairs`**, para cada pregunta `fault_type`:
  - **K4:** `none`, `other` y dos categorías reales;
  - **K8:** lo mismo más siempre las mismas **cuatro** distractoras («hardware», «instalación», «notificaciones» y «accesibilidad»).
  - `other` se sortea con probabilidad 0,5 cuando hay fallo; el par de categorías reales es uniforme con cualquier etiqueta.
  - Ganancia por composición medida: **0,000** en K4 y en K8 (6 composiciones posibles).
  - Tests: `tests/unit/test_phase6d_balanced_k.py`.
- **`derive.swap_states`:** control «sin leer el estado». Cada pregunta recibe el estado de otro grupo (derangement), con las mismas opciones y la misma etiqueta.

## Datos (se generan después de este protocolo)

- `data/pilot_v4_kdiag14`: `generate_mixed(400, seed=14, version="v4")`.
  - Excluye los grupos cuyo estado literal aparezca en `pilot_v3`, `holdout6`, `calib7`, `final8`, `pilot_v4`, `calib12`, `final13` o `kdiag11`.
  - Derivados, sólo con sus preguntas `fault_type`: `_K4`, `_K8`, `_K4swap` y `_K8swap` (`balanced_fault_kind_pairs` con seed 0; `swap_states` con seed 0).
- `data/pilot_v4_final13_faultswap`: las preguntas `fault_type` de `final13` con los estados intercambiados. `final13` ya está usado; esto es un control descriptivo, no una decisión.
- Script: `scripts/derive_phase6d_data.py`.

## Modelos y calibración

A4v3 (servicio), A4v4 y A2v4, sin reentrenar, con sus temperaturas de `calib12` (fase 6c).

## Lecturas predeclaradas

1. **Más opciones.** Δaccuracy de K8 − K4, emparejado (`gso compare --allow-different-inputs`, bootstrap por grupos, 1000 repeticiones, semilla 0).
   - «Tolera K = 8 en esta familia» si el **límite inferior** del IC es ≥ −0,05.
   - Se informan también ΔNLL y la accuracy separada para `other` y para el resto.
2. **Control sin estado.** Accuracy en `_K4swap` y `_K8swap` frente a la mejor alcanzable sólo con las opciones (el prior de la tarea en este diseño, del orden del porcentaje de `other`).
   - Si la supera en más de 0,10, se declara una pista no identificada.
   - Si no, el diagnóstico se considera libre de pistas explotadas.
3. **Composición del generador v4** (descriptiva). Accuracy de `final13_faultswap` frente a `acc_options_bayes` del generador v4 (0,477) y a su `acc_prior_only` (0,427), para ver si los modelos aprovechan la composición del entrenamiento.

## Lo que no se hace

- No se reentrena ni se cambia el servicio.
- No se usan `final8`, `holdout6*`, `final13` ni `kdiag11*` para decidir nada.
- Corregir la composición del generador de entrenamiento (un v5) queda fuera de esta fase y se deja como siguiente paso.
