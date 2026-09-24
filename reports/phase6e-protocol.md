# Protocolo predeclarado de la fase 6e: tríos con la misma pregunta (control coherente del uso del estado y diagnóstico de K)

Fecha: 2026-09-24. Escrito **antes** de generar los datos y de evaluar ningún modelo sobre ellos. Su sha256 se guarda en `reports/phase6e/protocol.sha256`.

## Motivo

La revisión de la fase 6d (`docs/STATUS.md`) retiró dos conclusiones:

- **El control con estados intercambiados** no aislaba el uso de las opciones: conservaba etiquetas que contradecían el estado donante (125/167).
- **La «ganancia 0,000»** era una accuracy dentro de la muestra, no información nula. La información mutua empírica entre la composición y la etiqueta en K4 es de 0,42 bits, en parte legítima: una respuesta real tiene que estar entre las opciones.

El revisor pide un control semánticamente coherente o un predictor que sólo vea la pregunta y las opciones. Además, el diagnóstico de K tenía poca potencia (167 preguntas, 58 `other`).

## Diseño (implementado y probado en CPU antes de este protocolo)

**`derive.fault_kind_triplets`:** cada trío comparte **exactamente** la misma pregunta (instrucción, textos, IDs y orden de las opciones: `none`, `other` y dos categorías reales). Difiere sólo en el estado, tomado de tres grupos distintos del mismo idioma:
- **real:** el fallo es de una categoría listada;
- **other:** el fallo es de una categoría real no listada;
- **none:** no hay fallo.

La etiqueta de cada pregunta es la respuesta semántica de su propio estado.

- **Techo sin leer el estado:** un predictor que sólo vea la pregunta y las opciones responde lo mismo a los tres. Su accuracy máxima es **1/3**, con **0** tríos completos. No hace falta estimarlo.
- **K8:** K4 más las cuatro distractoras de la fase 6d, con los mismos IDs y textos dentro del trío.
- **Tests:** `tests/unit/test_phase6e_triplets.py`.

## Datos (se generan después de este protocolo)

- `data/pilot_v4_trip15`: `generate_mixed(1000, seed=15, version="v4")`. Excluye los grupos cuyo estado literal aparezca en `pilot_v3`, `holdout6`, `calib7`, `final8`, `pilot_v4`, `calib12`, `final13`, `kdiag11` o `kdiag14`.
- `data/pilot_v4_trip15_K4` y `data/pilot_v4_trip15_K8`: 150 tríos, es decir, 450 preguntas y 150 respuestas `other` (`fault_kind_triplets` con seed 0).
- Script: `scripts/derive_phase6e_data.py`.

## Modelos

A4v3 (servicio), A4v4 y A2v4, sin reentrenar, con sus temperaturas de `calib12` (fase 6c).

## Lecturas predeclaradas

1. **Uso del estado** (control principal).
   - Métricas por modelo:
     - accuracy global en K4 y K8 frente al techo de 1/3;
     - proporción de tríos completos frente a 0, con IC por bootstrap de tríos (1000 repeticiones, semilla 0).
   - Se declara que el modelo **usa el estado** si el límite inferior del IC de los tríos completos es > 0,10.
2. **Más opciones.** Δaccuracy de K8 − K4, emparejado (`gso compare --allow-different-inputs`, bootstrap por grupos, 1000 repeticiones, semilla 0).
   - «Tolera K = 8» si el **límite inferior** del IC es ≥ −0,05.
   - Se informan también la accuracy por papel (real, `other`, `none`) y ΔNLL.
3. **`other`** (descriptivo): accuracy del papel `other` en K4 y K8 por modelo.

## Lo que no se hace

- No se reentrena ni se cambia el servicio (A4v3).
- No se usan `final8`, `holdout6*`, `final13`, `kdiag11*` ni `kdiag14*` para decidir nada.
- El generador de entrenamiento v5 queda fuera de esta fase.
