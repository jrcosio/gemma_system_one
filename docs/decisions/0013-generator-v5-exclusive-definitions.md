# 0013 — Generador `support-mixed-v5`: definiciones excluyentes y opciones independientes de los hechos

Fecha: 2026-09-24 · Fase 6f · Estado: aceptada (datos); el servicio no cambia

## Contexto

- **Fase 6e:** con E4B, el error dominante en la respuesta `other` era elegir «aplicación» para fallos no listados de rendimiento o de datos. La revisión lo dejó como hipótesis: los estados de esas categorías mencionan funciones de la app, y la definición de «aplicación» no los excluía.
- **Fase 6d:** la sonda de opciones midió una diferencia de 0,050 en accuracy top-1 dentro de su muestra para v4; no midió información mutua.

## Decisión

- **`--generator-version v5`**, con v1–v4 idénticos byte a byte:
  - **Definiciones excluyentes:**
    - «aplicación» = «Error, cierre o pantalla que no carga en una función (no incluye lentitud ni pérdida de datos)»;
    - «rendimiento» = «funciona, pero tarda mucho»;
    - cláusula de precedencia en la instrucción: la lentitud es rendimiento y los datos perdidos o dañados son datos, aunque ocurran en una función.
  - **Composición independiente de los hechos:** siempre `none` y `other`, más K − 2 categorías (K de 3 a 6) sorteadas entre las 4 reales y las 3 distractoras. La etiqueta se deduce después.
- **`derive.fault_kind_triplets(version="v5")`** y **`scripts/compare_triplets.py`** (diferencia emparejada por trío).
- **Servicio:** se mantiene A4v3. La regla S predeclarada (`reports/phase6f-protocol.md`) exige H y no inferioridad en `final17`, y la parte b no se cumple.

## Evidencia (`reports/phase6f-final.md`)

- **Sin modelos:** diferencia de accuracy top-1 en la muestra de la sonda al usar composición de opciones: 0,012 en v5 (v4: 0,050). `other` pasa a 0,48 de las preguntas `fault_type`.
- **Regla H (se cumple):** tríos completos A4v5 − A4v3 = **+0,087 [+0,020; +0,160]**, con bootstrap por trío (150 tríos v5).
- **Regla S b (no se cumple):** NLL calibrada A4v5 − A4v3 en `final17` = −0,007 [−0,037; +0,025]; el límite superior supera +0,02.
- **Descriptivo:**
  - el error `other` → «aplicación» casi desaparece con v5 (A4v3 1 y A4v5 3 en `trip18`, frente a 31 de 55 errores de A4v3 en `trip15`), también en A4v3 sin reentrenar;
  - el error residual dominante en E4B está en los fallos **resueltos** con respuesta `other` (A4v5 falla 13 de 48, frente a 2 de 102 con fallo activo).

## Consecuencias

- Los datos nuevos se generan con v5; los de v3 y v4 se conservan por reproducibilidad.
- **Hipótesis siguiente,** para un v6 con protocolo nuevo: aclarar si un fallo ya resuelto cuenta como fallo, tanto en la opción `none` como en la instrucción.
- En E2B, las cuentas bloqueadas siguen confundiéndose con fallos técnicos pese a la cláusula de acceso.
