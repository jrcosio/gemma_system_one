# Protocolo predeclarado de la fase 6c: generador v4 sin la pista de K, reentrenamiento y prueba válida de más opciones

Fecha: 2026-09-24. Escrito **antes** de generar los datos v4 y de entrenar o evaluar cualquier modelo sobre ellos. Su sha256 se guarda en `reports/phase6c/protocol.sha256`.

## Fallo del piloto que se corrige

Lo encontró la revisión independiente de la fase 6b (`docs/STATUS.md`):

- **En el generador v3:** los casos `fault_type` cuya respuesta es `other` tienen como mucho K = 5, porque sólo quedan 4 categorías no verdaderas. K = 6 revela que la respuesta no es `other`. La pista está también en los datos de **entrenamiento** de A2, B2 y A4.
- **En la ampliación de la fase 6** (`widen_fault_kind`): se usaba la etiqueta para decidir qué añadir, y K = 8 revelaba lo mismo. Ese diagnóstico queda retirado.

## Corrección (ya implementada y probada en CPU antes de este protocolo)

- **Generador `support-mixed-v4`:**
  - v3 más tres categorías distractoras que ningún estado describe: «hardware», «instalación» y «notificaciones»;
  - `other` se sortea con la misma probabilidad (0,2, con fallo presente) para cualquier K de 3 a 6;
  - secuencia aleatoria propia; v1–v3 siguen idénticos byte a byte.
- **`derive.widen_fault_kind_by_facts`:** amplía a K = 8 usando los hechos del caso (nunca añade la categoría verdadera), así que todos los casos llegan a K = 8 sea cual sea la etiqueta.
- **Tests:** `tests/unit/test_phase6c_generator_v4.py`. Muestran la pista en v3, su ausencia en v4, la coherencia de las respuestas con los hechos y la ampliación a K8 para toda etiqueta.

## Datos (se generan después de escribir este protocolo)

Todos con `--generator-version v4 --variant main`. Cada conjunto externo excluye los grupos cuyo estado literal aparezca en `pilot_v3`, `pilot_v3_holdout6`, `pilot_v3_calib7`, `pilot_v3_final8`, `pilot_v4` o en los conjuntos v4 anteriores de esta lista (`scripts/derive_phase6c_data.py`).

| Conjunto | Semilla / casos | Uso |
|---|---|---|
| `data/pilot_v4` | 0 / 1000; `gso split --seed 0` | train (ajuste) y validation (selección de época). Su calibration y su test **no se usan** en esta fase |
| `data/pilot_v4_calib12` | 12 / 400 | Sólo temperaturas (calibración externa) |
| `data/pilot_v4_final13` | 13 / 300 | **Test final** de la regla S; una evaluación por modelo |
| `data/pilot_v4_kdiag11_K` / `_K8` | 11 / 300, sólo preguntas `fault_type`: originales (K 3–6) y ampliadas por hechos a K = 8 | Diagnóstico de más opciones |

## Modelos

- **A4v4:** E4B congelado + cabezales, `configs/e4b_v4.yaml`. Idéntico a `e4b_experiment` salvo el dataset: mismos hiperparámetros, época por validación (1–30).
- **A2v4:** E2B congelado + cabezales, `configs/e2b_heads_v4.yaml`. Idéntico a `pilot_ce_v3` salvo el dataset.
- **A4v3:** el servicio actual (`runs/e4b_experiment/20260923T204945Z`), sin reentrenar.
- **Calibración:** los tres se calibran con `gso calibrate --split all --dataset data/pilot_v4_calib12`. Es el procedimiento de la fase 6b, fijado aquí.
- B2 (LoRA) no se reentrena en esta fase.

## Regla S (servicio de texto)

- **Métrica:** NLL calibrada media en `pilot_v4_final13`, diferencia emparejada **A4v4 − A4v3** con `gso compare` (bootstrap por grupos, 1000 repeticiones, semilla 0, IC95 %).
- **Criterio:** v4 elimina una pista espuria, así que se prefiere salvo pérdida material (no inferioridad con margen 0,02).
  - Límite superior < +0,02 ⇒ **A4v4 sustituye a A4v3** en `configs/serve_e4b_text.yaml`, con un benchmark HTTP de confirmación.
  - Si no ⇒ se mantiene A4v3 y se documenta.
- **Secundarias, sin regla:** A4v4 − A2v4; NLL sin calibrar; métricas por primitiva; accuracy en `fault_type` según la etiqueta sea `other` o no.

## Más opciones (lectura predeclarada)

- **Datos:** para A4v4, A2v4 y A4v3, las 300 × ~0,15 preguntas `fault_type` de `kdiag11`, con su K original y ampliadas a K = 8.
- **Métrica:** `gso compare --allow-different-inputs`, que exige la misma etiqueta semántica, con las temperaturas de `calib12`.
- **Lectura:** «el evaluador compartido tolera K = 8 en esta familia» sólo si el **límite inferior** del IC de Δaccuracy (K8 − K) es ≥ −0,05. Si no, es una limitación documentada.
- **Complementos:** ΔNLL, accuracy por K original y accuracy en K8 separada para respuestas `other` y no `other`.
- `P(other | K)` se tabula en cada conjunto generado.

## Lo que no se hace

- No se usan `pilot_v3_final8`, `pilot_v3_holdout6*` ni los tests de `pilot_v3` y `pilot_v4` para decidir nada.
- No se cambia ningún hiperparámetro respecto a A2 y A4. Si la época elegida es la tope, no se amplía.
- No se cambian el contrato público, la plantilla ni el backend.
