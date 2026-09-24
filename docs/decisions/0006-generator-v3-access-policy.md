# 0006 — Generador v3: política explícita de acceso a cuenta en las preguntas de fallo técnico

Fecha: 2026-09-23 · Fase 3 (antes de entrenar LoRA) · Estado: aceptada

## Contexto

La fase 2 dejó abierto un fallo del piloto: las preguntas de fallo técnico (Noul `service_fault`, Choice `fault_type`, Score `fault_severity`) se etiquetaban con la regla «un problema de acceso a la cuenta no es un fallo técnico», pero la instrucción no lo decía. En validación de `pilot_v2`, `fault_severity` fallaba 6/23 veces con cuenta bloqueada frente a 4/35 sin ella, y varios errores eran «cuenta bloqueada» → «fallo técnico». La especificación pide que la política de etiquetado sea explícita y comprobable (§5.2), y el STATUS pedía resolverlo con datos versionados nuevos, no editando etiquetas del piloto.

## Decisión

- `support-mixed-v3` = v2 + la cláusula, añadida al final de la instrucción de las tres familias de fallo técnico:
  - es: «Los problemas de acceso a la cuenta (inicio de sesión, contraseña o bloqueo) no cuentan como fallo técnico.»
  - en: «Account access problems (login, password or lockout) do not count as technical failures.»
- v3 **reutiliza la secuencia aleatoria de v2** y la cláusula no consume aleatoriedad. Estados, etiquetas, grupos, IDs, criterios y asignación de splits son idénticos a v2 (test `tests/unit/test_phase3_data.py`); sólo cambia el texto de esas instrucciones. Así la comparación v2/v3 es emparejada.
- `--generator-version` pasa a `v3` por defecto. v1 y v2 siguen disponibles; los artefactos v2 no se tocan.
- La cláusula es la misma en la variante de transferencia: es política, no superficie.

Además (riesgo abierto de fase 2): `near_duplicate_states` aplica ya su argumento `limit` y devuelve también el total. Sólo cuando hay truncamiento se añade `near_duplicate_states_total` a las comprobaciones, para que los manifiestos existentes (sin truncar) sigan siendo idénticos: `gso split` sobre `pilot_v2` y `mixed_smoke_v2` responde «ya existía idéntico». El coste sigue siendo cuadrático.

## Evidencia

- `data/pilot_v3`: sha256 `f2519285…9352`, idéntico con `PYTHONHASHSEED` 1 y 999; 3000 preguntas, 0 errores; split 700/100/100/100 grupos, los mismos que v2; 5 casi duplicados (aviso). `data/pilot_transfer_v3` `a99c1844…c78f`; `data/mixed_smoke_v3` `4d4365c0…ff46`.
- Cabezales congelados con el mismo tope de 10 épocas que v2 (`runs/pilot_ce_v3_e10/20260923T010432Z`, diagnóstico): NLL de validación **0,438** [0,357; 0,519] frente a 0,445 [0,340; 0,548] en v2. Con cuenta bloqueada: `fault_severity` 6/23 → 4/23 errores, `fault_type` 2/14 → 4/14, `service_fault` 1/12 → 0/12. **La diferencia no es distinguible del ruido** con 100 grupos de validación.

## Consecuencias

- Se resuelve la ambigüedad en la especificación de la tarea (lo que el modelo lee), no se demuestra una mejora de calidad.
- La referencia congelada de fase 3 se reentrena sobre v3 (`configs/pilot_ce_v3.yaml`) con tope de 30 épocas, porque en v2 la época elegida fue el máximo (10). Resultado: época 28, NLL de validación 0,424. Estas cifras de validación están sesgadas por la selección.
- El test de v3 corresponde a los mismos grupos que el test de v2, que nunca se usó.
