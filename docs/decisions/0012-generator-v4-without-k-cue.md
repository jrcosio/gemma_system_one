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
- **`derive.widen_fault_kind_by_facts`:** amplía a K = 8 con los hechos del caso (nunca añade la categoría verdadera); todas las preguntas llegan a K = 8 sea cual sea la etiqueta.
- **Datos nuevos:** los conjuntos se generan con v4. Los de v3 se conservan con sus hashes para la reproducibilidad.
- **Servicio:** por la regla S predeclarada (`reports/phase6c-protocol.md`) se mantiene A4v3. A4v4 − A4v3 en `pilot_v4_final13` = −0,007 [−0,034; +0,024]; el límite superior supera el margen de no inferioridad (+0,02). A4v4 queda como checkpoint alternativo, prácticamente equivalente.

## Evidencia

- **Tests:** `tests/unit/test_phase6c_generator_v4.py`.
  - En v3: 0 casos `other` con K = 6.
  - En v4: `other` ≈ 0,2 para cada K de 3 a 6 (n > 200 por K).
  - Respuestas coherentes con los hechos; ampliación a K = 8 para toda etiqueta y determinista.
- **Tabla de `other` por K** en los datos generados: `reports/phase6c/other_by_k.txt`. Por ejemplo, en `pilot_v4`: K3 10/131, K4 17/116, K5 21/125, K6 17/111.
- **Resultados:** `reports/phase6c-final.md`.

## Consecuencias

- **Más opciones con v4** (diagnóstico emparejado, 150 preguntas):
  - A4v3 tolera K = 8 según la lectura predeclarada;
  - A4v4 no por poco: límite inferior de Δaccuracy −0,060;
  - A2v4 se degrada claramente.
  - La respuesta `other` es la más difícil en todos los modelos.
- La pista de v3 no explica la ventaja de E4B: con v4, A4v4 − A2v4 = −0,149 [−0,195; −0,105].
