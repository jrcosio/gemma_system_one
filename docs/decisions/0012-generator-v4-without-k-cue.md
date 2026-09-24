# 0012 — Generador `support-mixed-v4`: sin la pista de K en `fault_type`

Fecha: 2026-09-24 · Fase 6c · Estado: aceptada

## Contexto

La revisión independiente de la fase 6b detectó dos pistas estructurales que revelaban la respuesta a partir del número de opciones:

- **En el generador v3:** los casos `fault_type` con respuesta `other` sólo disponen de 4 categorías no verdaderas, así que tienen como mucho K = 5; con K = 6 la respuesta nunca es `other`. La pista está en los datos de entrenamiento de A2, B2 y A4.
- **En la ampliación a K = 8 de la fase 6** (`derive.widen_fault_kind`): se decidía con la etiqueta qué añadir, y K = 8 implicaba «no `other`». Ese diagnóstico se retiró.

## Decisión

- **`support-mixed-v4`**, con `--generator-version v4`:
  - v3 más tres categorías distractoras que ningún estado describe: «hardware», «instalación» y «notificaciones», comprobado por palabras clave en `pilot_v3` y `final8`;
  - `other` se sortea con probabilidad 0,2 (con fallo presente), independiente de K de 3 a 6, y siempre hay categorías no verdaderas suficientes;
  - secuencia aleatoria propia; v1–v3 siguen idénticos byte a byte (comprobado con `pilot_v3` y `pilot_v3_seed8`).
- **`derive.widen_fault_kind_by_facts`:** amplía a K = 8 con los hechos del caso (nunca añade la categoría verdadera); todas las preguntas llegan a K = 8 sea cual sea la etiqueta. La revisión posterior detectó que la composición de las opciones sigue dando una pista parcial sobre la etiqueta `other`.
- **Datos nuevos:** los conjuntos se generan con v4. Los de v3 se conservan con sus hashes para la reproducibilidad.
- **Servicio:** por la regla S predeclarada (`reports/phase6c-protocol.md`) se mantiene A4v3. A4v4 − A4v3 en `pilot_v4_final13` = −0,007 [−0,034; +0,024]; el límite superior supera el margen de no inferioridad (+0,02). A4v4 queda como checkpoint alternativo, sin equivalencia demostrada.

## Evidencia

- **Tests:** `tests/unit/test_phase6c_generator_v4.py`.
  - En v3: 0 casos `other` con K = 6.
  - En v4: `other` ≈ 0,2 para cada K de 3 a 6 (n > 200 por K).
  - Respuestas coherentes con los hechos; ampliación a K = 8 para toda etiqueta y determinista.
- **Tabla de `other` por K** en los datos generados: `reports/phase6c/other_by_k.txt`. Por ejemplo, en `pilot_v4`: K3 10/131, K4 17/116, K5 21/125, K6 17/111.
- **Resultados:** `reports/phase6c-final.md`.

## Consecuencias

- **Más opciones con v4** (diagnóstico emparejado, 150 preguntas):
  - Las cifras emparejadas son A4v3 0,000 [−0,040; +0,033], A4v4 −0,020 [−0,060; +0,020] y A2v4 −0,147 [−0,207; −0,087] en Δaccuracy.
  - La lectura de robustez de A4v3 queda retirada: en K8 todos los casos `other` incluyen las tres distractoras imposibles, mientras 41 de 133 no `other` incluyen sólo dos. No es una prueba sin pistas de etiqueta; véase `reports/phase6c-final.md`.
  - La respuesta `other` es la más difícil en todos los modelos.
- La ventaja observada de E4B frente a E2B persiste en el test v4: A4v4 − A2v4 = −0,149 [−0,195; −0,105]. Esta comparación no atribuye causalmente la ventaja a una única pista del generador.

## Anexo 2026-09-24 (fase 6d): diagnóstico con composición equilibrada

- **Diseño:** `derive.balanced_fault_kind_pairs` sustituye a `widen_fault_kind_by_facts` para medir K8: K4 = `none` + `other` + dos categorías reales, y K8 = K4 + cuatro distractoras fijas.
- **Ganancia por composición:** 0,000 (`scripts/probe_option_cue.py`).
- **Control sin estado** (`derive.swap_states`): la accuracy queda por debajo del prior, así que no se explota ninguna pista.
- **Tolerancia a K = 8:** no demostrada para A4v3, A4v4 ni A2v4 (límite inferior del IC de Δaccuracy entre −0,072 y −0,096; 167 preguntas).
- **Generador de entrenamiento v4:** conserva 0,050 de información por composición, que los modelos no aprovechan más allá del prior en `final13`. Corregirlo exige un v5 con composición equilibrada (pendiente).

Detalle: `reports/phase6d-final.md`.
