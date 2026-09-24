# Protocolo predeclarado del test de fase 4 (uso visual)

Redactado el 2026-09-23, **antes** de generar el piloto visual y de entrenar ningún modelo con imagen. Criterio de salida de la fase (spec §10): «evidencia de uso visual y límites de memoria reales». No se modifica tras ver test; las desviaciones se añaden al final.

## Datos

- `data/vision_pilot_v1`: `gso generate-data --kind vision --out data/vision_pilot_v1 --cases 700 --seed 0`, con split por grupos seed 0 (70/10/10/10). La respuesta sólo está en la imagen.
- `data/vision_transfer_v1`: 150 casos con estilo visual y redacciones reservados. Sólo diagnóstico, nunca selección.
- `data/vision_smoke_v1`: 60 casos para la puerta de sobreajuste.

## Modelos

| Rol | Configuración | Selección |
|---|---|---|
| V: E2B congelado + cabezales, **con imagen** | `configs/vision_heads.yaml` | Época por NLL de validación |
| C: control **sólo texto**, mismas filas sin imagen | `configs/vision_text_only.yaml` | Época por NLL de validación |

Si V no mostrara uso visual en **validación** (sin diferencia frente a C ni caída con la imagen omitida o intercambiada), podrá entrenarse una variante LoRA (visión congelada, LoRA textual como en fase 3) y sustituir a V **sólo si gana en validación**. Esa elección se registrará antes de abrir test.

## Test (una vez por artefacto, desde el texto+imagen, sin caché)

```bash
uv run gso evaluate --checkpoint <V> --split test --final-test --no-cache --baselines --vision-ablation
uv run gso evaluate --checkpoint <C> --split test --final-test --no-cache --baselines
uv run gso compare --a <predicciones test C> --b <predicciones test V> --out reports/phase4/test_compare_vision_minus_text.json
```

- **Métrica primaria:** NLL media por pregunta en test, sin calibrar; la calibración no forma parte del criterio de esta fase.
- **Comparación primaria:** V − C emparejada por pregunta, con IC95 % por bootstrap de grupos (1000 réplicas, semilla 0).
- **Regla de decisión:** hay «evidencia de uso visual» si se cumplen las dos condiciones:
  1. el extremo superior del IC de V − C es < 0;
  2. en la ablación de V en test, la NLL con la imagen **intercambiada** es mayor que con la original (diferencia emparejada media > 0).

  En otro caso: «sin evidencia suficiente».
- **Secundarias:**
  - accuracy y NLL por primitiva;
  - imagen omitida;
  - prior y BoW;
  - transferencia de estilo (diagnóstico);
  - memoria (driver/asignada MPS, RSS, swap) y tiempos sincronizados de extracción con imagen.
- **Límites:** es un banco sintético de gráficos de barras, y el uso visual en él no implica comprensión visual general. Las cifras de validación quedan sesgadas por la selección.

## Desviaciones declaradas

Añadidas tras redactar el protocolo. El sha256 del texto original, antes de esta sección, es `441a5e6d…37c7` y está en `reports/phase4/test-protocol.sha256`.

1. **Herramienta de comparación, después de ejecutar test.** `gso compare` rechazó la comparación primaria porque `input_sha256` difiere entre V y C, y eso es precisamente lo que se compara: C son las mismas filas sin imagen.
   - La comprobación de huella se había añadido en esta misma fase y no distinguía un control deliberado.
   - Se añadió `--allow-different-inputs`, que sigue exigiendo los mismos IDs, grupos, tipos y etiquetas, y deja `same_inputs: false` en el resultado.
   - La comparación se ejecutó sobre los mismos ficheros de predicciones de test ya producidos, sin inferencia nueva.
   - Comando real: `gso compare --a <C> --b <V> --allow-different-inputs --out reports/phase4/test_compare_vision_minus_text.json`. No cambia datos, modelos, métrica ni regla.
2. **Condición 2 de la regla.** Se calcula como la diferencia de las NLL medias de la ablación de V en test (imagen intercambiada − original, sobre las mismas 210 preguntas), que equivale a la media de la diferencia emparejada.
3. **Sin variante LoRA.** V mostró uso visual en validación, así que, como estaba previsto, no se entrenó la variante LoRA.
