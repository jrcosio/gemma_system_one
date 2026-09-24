# Protocolo predeclarado del test de cierre de fase 4 (estilos separados por partición)

Redactado el 2026-09-23, **antes** de generar `data/vision_pilot_v2` y de entrenar con él. Motivo: la revisión de fase 4 (hallazgo 4) comprobó que en el piloto v1 los estilos 0–3 aparecen en las cuatro particiones, en contra de la spec §5.1 («separando estilos y semillas entre particiones»). El test v1 ya está usado y no se reutiliza ni se modifica. Este protocolo no se cambia tras ver test; las desviaciones se añadirán al final.

## Datos (generador `support-vision-v2`, decisión 0008)

- **`data/vision_pilot_v2`:** `gso generate-data --kind vision --vision-version v2 --out data/vision_pilot_v2 --cases 700 --seed 0 --split-seed 0`, seguido de `gso split --dataset data/vision_pilot_v2 --seed 0`.
  - El split se planifica antes de muestrear y `gso split` verifica que coincide con el plan.
- **Estilos disjuntos por partición:**

  | Partición | Estilos |
  |---|---|
  | train | 0, 1, 2, 3, 5, 6 |
  | validation | 7, 8 |
  | calibration | 9, 10 |
  | test | 11, 12 |

  Cada partición tiene además su propio flujo aleatorio (semilla derivada de la partición). Las redacciones son las principales y comunes: sólo se separan los estilos visuales.
- **`data/vision_transfer_v2`:** 150 casos con el estilo 4 y las redacciones reservadas. Sólo diagnóstico.

## Modelos (mismos hiperparámetros que V/C de fase 4; sin ajustes nuevos)

| Rol | Configuración | Selección |
|---|---|---|
| V2: E2B congelado + cabezales, con imagen | `configs/vision2_heads.yaml` | Época por NLL de validación (estilos 7–8, no vistos en train) |
| C2: mismas filas sin imagen | `configs/vision2_text_only.yaml` | Época por NLL de validación |

No se entrenará ninguna otra variante para este test.

## Test (una vez por artefacto; desde texto+imagen; sin caché)

```bash
uv run gso evaluate --checkpoint <V2> --split test --final-test --no-cache --baselines --vision-ablation
uv run gso evaluate --checkpoint <C2> --split test --final-test --no-cache --baselines
uv run gso compare --a <predicciones test C2> --b <predicciones test V2> --allow-different-inputs \
  --out reports/phase4b/test_compare_vision_minus_text.json
```

- **Métrica primaria:** NLL media por pregunta en test (estilos 11–12), sin calibrar.
- **Regla de decisión:** hay «evidencia de uso visual con estilos no vistos» si se cumplen las dos condiciones:
  1. el extremo superior del IC95 % (bootstrap de grupos, 1000 réplicas, semilla 0) de V2 − C2 es < 0;
  2. la NLL de V2 con la imagen de otro grupo es mayor que con la original (media > 0).

  En otro caso: «sin evidencia suficiente con estilos no vistos».
- **Secundarias:**
  - accuracy y NLL por primitiva y por estilo de test;
  - imagen omitida;
  - prior y BoW;
  - transferencia v2 (diagnóstico);
  - comparación descriptiva con el V de fase 4 (estilos vistos), sin regla.
- **Límites:** es el mismo banco sintético de barras, y los estilos «no vistos» son variaciones de color, orientación, grosor, trazo y tamaño de fuente de la misma familia de gráfico. El resultado no implica comprensión visual general.

## Desviaciones declaradas

(Ninguna por ahora.)
