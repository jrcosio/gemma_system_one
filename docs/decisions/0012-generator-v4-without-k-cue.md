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

## Anexo 2026-09-24 (fase 6d): diagnóstico con composición equilibrada, revisado

- **Diseño:** `derive.balanced_fault_kind_pairs` sustituye a `widen_fault_kind_by_facts` para medir K8: K4 = `none` + `other` + dos categorías reales, y K8 = K4 + cuatro distractoras fijas.
- **Ganancia top-1 del probe:** 0,000 (`scripts/probe_option_cue.py`) entre un prior restringido a opciones y un predictor por firma ajustado en la misma muestra. No implica información mutua cero ni independencia: cada etiqueta real sólo puede ser respuesta si está en el par. K8 añade cuatro distractoras fijas al mismo K4; ésa es la comparación controlada.
- **Control con estados intercambiados** (`derive.swap_states`): 125/167 etiquetas conservadas en K4/K8swap contradicen el estado donante. La accuracy frente a ellas no es una medida de uso exclusivo de opciones ni permite concluir que no se explota ninguna pista.
- **Tolerancia a K = 8:** no demostrada para A4v3, A4v4 ni A2v4 (límite inferior del IC de Δaccuracy entre −0,072 y −0,096; 167 preguntas).
- **Generador de entrenamiento v4:** el probe muestra 0,050 de ganancia top-1 dentro de la muestra. El uso de esa señal por el modelo no queda resuelto con `final13_faultswap`. Investigar un v5 con datos nuevos queda pendiente.

Detalle: `reports/phase6d-final.md`.

## Anexo 2026-09-24 (fase 6e): tríos con la misma pregunta

- **Diseño:** `derive.fault_kind_triplets` (protocolo `reports/phase6e-protocol.md`) da un control del uso del estado coherente con la semántica. Cada trío comparte exactamente las mismas opciones y sólo cambia el estado; sin leer el estado, el techo es 1/3 de accuracy y 0 tríos completos.
- **Uso del estado:** los tres modelos lo usan. Tríos completos: A4v3 0,58, A4v4 0,50 y A2v4 0,21, todos con el límite inferior del IC > 0,10.
- **Tolerancia a K = 8:** sólo A2v4 la cumple (Δaccuracy +0,013 [−0,024; +0,053]). En A4v3 (−0,024 [−0,060; +0,011]) y A4v4 (−0,027 [−0,051; −0,002]) no queda demostrada.
- **`other`:** en E4B son frecuentes las asignaciones de fallos de rendimiento y datos a «aplicación» (A4v3 K8: 31 de 55 errores de `other`). En K4, el error más frecuente de A4v3 es `other` → `none` (26 de 48), y la composición de errores cambia con K. El posible solapamiento de definiciones es una hipótesis para un v5, no una causa demostrada.

Detalle: `reports/phase6e-final.md`.
