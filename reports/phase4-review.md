# Revisión independiente de fase 4 — 2026-09-23

Se contrastaron AGENTS, README, ESPECIFICACION, AUDITORIA, STATUS, decisión 0007, diff de README y los fuentes sin seguimiento. `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`; no se hizo commit, cambio de rama ni entrenamiento largo. No hubo descargas ni servicios externos.

## Hallazgos por gravedad

1. **Alta, corregida: fuga de una imagen idéntica por alias de extensión.** `leakage_checks` comparaba rutas completas: los mismos bytes JPEG en `images/<sha>.jpg` y `images/<sha>.jpeg` se aceptaban en train y test, pese a pasar ambas la validación. Regresión `test_jpeg_extension_alias_cannot_cross_splits` falló antes de corregir. Ahora se compara el sha de contenido para detectar imágenes compartidas, sin cambiar los hashes históricos de filas, contratos o manifiestos. No afecta al piloto PNG. Recompresión o alteraciones de píxeles siguen fuera de este detector: no es un detector perceptual.
2. **Media, corregida: validación sin decodificación real.** Un JPEG al que se quitan los dos bytes finales pasaba `check_image_file` porque `PIL.Image.verify()` no asegura que se decodifiquen sus píxeles; podía fallar después al extraer representaciones, tras cargar el modelo. Ahora validación y extracción comparten una carga que verifica formato, animación, dimensiones, hash, integridad y decodificación completa. La lectura se acota a 5 MiB + 1 antes de materializar todo el fichero, y se decodifican los mismos bytes cuyo hash se verificó.
3. **Media, corregida: discrepancia entre validador y cargador.** `load_image` aceptaba BMP con nombre `.png` y PNG animado, tomando implícitamente un fotograma. Ambos fallos reproducidos. El pipeline habitual validaba antes el dataset, por lo que no se ha demostrado afectación del piloto; el cargador directo no cumplía su contrato. Ahora devuelve `ImageError` en ambas rutas para esas entradas.
4. **Media, pendiente metodológica: estilos compartidos entre splits.** ESPECIFICACION §5.1 pide separar estilos/semillas entre particiones. El generador elige estilos antes del split y usa los cuatro estilos 0–3 en train, validation, calibration y test (comprobado cruzando los grupos del manifiesto con `audit.jsonl`, sin analizar predicciones para decidir). Sólo transferencia reserva el estilo 4. La evidencia demuestra uso visual dentro del repertorio y un diagnóstico de transferencia; el test principal no mide estilos inéditos. No se modificaron datos o splits después de ver test. Cerrar este requisito exige declarar la desviación y su alcance o preparar un futuro benchmark con estilos separados y test nuevo, sin reutilizar el actual para selección.
5. **Media, pendiente: no se conserva la versión exacta de entrenamiento.** Ambos checkpoints registran `70068e151e9d4d9c317a023840be76c6405147056bb98c367bdbde9cb6b5967b`; no existe `artifacts/source/<ese hash>.tar`. La copia `aa718528…` corresponde al cierre posterior. El hash identifica, pero no reconstruye el código ausente. No se fabricó una copia histórica. El protocolo y sus desviaciones son coherentes con la comparación guardada, pero un hash local no certifica cronología.
6. **Baja, corregida documentalmente: límite de memoria no medido.** «K > 5 no escala/supera 32 GiB» no se deriva del perfil con K = 5. Se midieron 26,2 GB asignados; K = 8 ≈ 36 GB es una extrapolación lineal, no un experimento. Se corrigió en STATUS, informe y decisión 0007. El muestreo tampoco captura todos los temporales del backward.
7. **Baja, corregida en STATUS:** AGENTS no exige pedir permiso para hacer commit. Se elimina esa atribución; este trabajo no hace commit ni cambia el alcance.

## Código y metodología comprobados

- Carga local fijada de `Gemma4Model`, sin cabeza generativa ni `generate`; MPS/BF16 y pesos base congelados. Los campos `pixel_values`, `image_position_ids` y `mm_token_type_ids` se conservan; flotantes al dtype del modelo y enteros sin conversión de dtype.
- Imagen antes del texto; preguntas y criterios completos en cada fila; los IDs y etiquetas no entran al prompt. Referencias de imagen usadas como identidad externa, no como texto del modelo.
- Pooling por última posición válida y máscara alineada; rechazo de exceso de tokens. Extracción visual obligatoria de una fila por forward; uso de imagen en claves de caché. LoRA no usa caché de representaciones.
- Scorer compartido por primitiva, normalización sobre grupo completo y acumulación por pregunta. Autograd LoRA preservado, torre visual congelada, gradientes verificados en MPS. Cabezales FP32.
- V/C tienen el mismo dataset, split, texto e hiperparámetros; `images: omit` quita sólo la imagen. Cada modelo selecciona época por validación. Donantes de la ablación pertenecen a otro grupo de la misma partición; se mantiene la etiqueta original como diagnóstico, no como etiqueta de la imagen donante.
- Recarga y comparación histórica verificadas aparte; no se ejecutó inferencia nueva en test. API aún fuera de alcance (fase 5), por tanto no se afirma servidor real implementado.

## Pruebas y reproducción

```bash
.venv/bin/pytest tests/unit tests/integration -q
# Antes: 250 passed, 20,52 s. Después: 254 passed, 22,28 s.
.venv/bin/pytest tests/unit/test_phase4_review.py -q
# Antes de corregir: cuatro fallos (JPEG truncado, BMP, animación, alias jpg/jpeg).
.venv/bin/pytest tests/unit/test_phase4_review.py tests/unit/test_phase4_vision_data.py tests/unit/test_dataset_split.py -q
# Primera corrección: 22 passed, 1 fallo por texto de error; se preservó el texto esperado.
.venv/bin/pytest tests/mps -q -rs
# 10 passed, 117,78 s, 0 skips. Proceso iniciado antes de editar el cargador.
.venv/bin/python scripts/review_phase4_artifacts.py > reports/phase4/review-artifacts.json
# Proceso nuevo DESPUÉS de corregir: V/C, dos preguntas de validation cada uno, sin caché.
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv lock --check
git diff --check
```

Las 10 pruebas de `tests/mps` incluyen una prueba del procesador real que no ejecuta GPU; las restantes prueban MPS, incluida visión y backward LoRA. CPU utiliza modelos diminutos y dobles de procesador: no se presenta como validación del hardware. El warning existente de `float(loss)` en `test_lora.py` no se ocultó.

El script de revisión comprueba hashes y dataset al preparar V/C, recarga desde las primeras dos preguntas de validación con el procesador y pesos reales, y exige equivalencia con los logits guardados. Libera cada base antes de cargar la siguiente. Recalcula exclusivamente la comparación de JSONL históricos de test y exige igualdad exacta del informe. El resultado detallado queda en `phase4/review-artifacts.json`.

## Veredicto

**Ruta funcional de fase 4 verificada con E2B real; no se da aprobación integral contra la especificación mientras sigan pendientes la separación de estilos y la conservación del código exacto de entrenamiento.** Los fallos de imágenes están corregidos y no hay evidencia de que afectaran al piloto PNG existente. Se conserva la conclusión limitada de uso visual; no se afirma comprensión visual general, calibración universal ni un límite de K no medido.

Los cambios son de validación, detección de fugas y precisión documental: sin cambio de backend, base, prompt, contrato público, checkpoints entrenados ni datos. El próximo asistente debe resolver los pendientes metodológicos explícitamente antes de presentar la fase como plenamente conforme.
