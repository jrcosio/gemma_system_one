# Revisión independiente de fase 3 — 2026-09-23

Referencia: AGENTS.md, ESPECIFICACION (§4–8, §10–12), AUDITORIA, STATUS, decisiones 0001–0006, código y artefactos locales. HEAD: `65d425025dd654fb4814062dc42b245062f151b1`. README es el único diff seguido; la implementación completa de fases 0–3 está sin seguimiento. Se revisaron los archivos reales, no sólo el resumen anterior.

## Hallazgos por gravedad

1. **Alta, corregida: pérdida del estado válido al repetir un guardado de reanudación.** `save_resume` usaba un nombre determinado sólo por paso/época/posición y borraba ese directorio antes de publicar el nuevo. La coincidencia ocurre con `resume_every_steps=50` y `--stop-after-steps 50`. Una interrupción entre borrado y reemplazo deja `LATEST` apuntando a un directorio inexistente. La regresión inyectó un error en `os.replace` y reprodujo `FileNotFoundError`. Ahora cada guardado tiene nombre único y sólo se elimina el anterior después de publicar `LATEST`. Se siguen leyendo los estados antiguos. No se afirma durabilidad frente a pérdida de alimentación (no hay fsync).
2. **Media, corregida: un log cortado impide reanudar pesos válidos.** `_discard_steps_after` parseaba todas las líneas de `steps.jsonl` sin tolerar el último write incompleto. Reproducido con dos pasos completos y `{"step":` al final: `JSONDecodeError`. Ahora conserva ese fragmento en `steps.truncated.txt`, aparta los pasos posteriores al checkpoint y reemplaza el log de forma atómica. Sigue rechazando corrupción en líneas completas o intermedias.
3. **Media, corregida: comparación emparejada con etiquetas incompatibles.** `compare_predictions` sólo cotejaba IDs, grupos y tipos. Reproducido con el mismo ID y etiqueta 0 en A/1 en B: publicaba una aparente mejora. Ahora coteja también `target_index` y `target_description` cuando existen; rechaza su presencia asimétrica. Esto no demuestra identidad del contenido si ambos archivos omiten esos campos o reutilizan IDs con otros textos: falta una huella de entrada en el formato de predicciones. Los archivos reales de A/B pasan la comprobación y conservan exactamente sus resultados.
4. **Media, pendiente: trazabilidad del código histórico insuficiente.** Los dos checkpoints sólo identifican el commit documental inicial y `dirty: true`; no incluyen una copia o hash de los fuentes sin seguimiento. Los hashes de pesos, dataset y split sí existen. No puede reconstruirse inequívocamente el código exacto de cada sesión del piloto a partir de su manifiesto. Tampoco un hash local del protocolo prueba por sí mismo cuándo fue declarado. No encontré evidencia de selección con test, pero no certifico retrospectivamente su cronología. Antes del siguiente experimento, versionar el código revisado y conservar su identidad exacta; no modificar artefactos históricos para simular esa evidencia.
5. **Baja, corregida en comentario de test:** la prueba MPS decía recargar en «otro modelo», pero restaura el adaptador en el mismo objeto tras ponerlo a cero. El comentario ya describe lo que hace; la recarga de checkpoints reales desde un proceso nuevo se verifica aparte.

## Rutas contrastadas

- `Gemma4Model.from_pretrained` desde snapshot local fijado, sin `lm_head`, `generate`, `device_map=auto` ni descargas durante revisión. Verificación de claves de carga; base congelada y en eval.
- MPS/BF16 real; cabezales y adaptadores FP32. LoRA sólo en Q/V textuales; modo train limitado a su dropout; gradientes presentes y finitos y cambios de pesos comprobados. La prueba CPU no se usa como prueba MPS.
- Pooling por máximo índice válido, máscara alineada, rechazo de filas demasiado largas sin truncar. Los prompts contienen estado, instrucciones y criterios completos; las etiquetas y los IDs opacos no pasan a la serialización.
- Choice/Score comparten scorer por primitiva y normalizan sobre todo el grupo. Backward por pregunta dividido por el tamaño real del paso. Estandarizador ajustado en train y fijo en LoRA.
- Separación por grupos y comprobación de hashes; calibración sólo en su partición y vinculada al checkpoint; selección por validación. LoRA fuerza caché de representaciones desactivada.
- Memoria RSS y MPS separadas; presupuesto comprobado mediante muestras. No es un límite continuo: los picos durante backward pueden superar las muestras entre pasos. No se repitieron los benchmarks largos.

## Evidencia y comandos

Antes de corregir:

- `.venv/bin/pytest tests/unit tests/integration -q`: **232 passed**, 16,38 s; un aviso existente de conversión de tensor con gradiente a escalar en un test.
- `.venv/bin/pytest tests/mps -q -rs`: **7 passed**, 49,67 s, **0 skips**, E2B real en Apple M5 Pro/48 GiB, MPS/BF16.
- `.venv/bin/pytest tests/unit/test_phase3_review.py -q`: los dos primeros casos fallaron antes del arreglo; posteriormente el tercer caso reprodujo la aceptación de etiquetas distintas.

Después de corregir:

- Regresiones de recuperación y pipeline CPU: **5 passed**, 9,36 s (antes de añadir la tercera regresión).
- Suite CPU final y verificaciones estáticas: resultado actualizado en STATUS.
- Verificados ambos checkpoints con `load_decision_heads`/`load_lora_decision` y calibraciones con `load_calibration`: hashes correctos.
- Recarga real con `prepare_evaluation(checkpoint, use_cache=False)`, las primeras dos preguntas de **validation** y `_reload_check`, ejecutada en un Python nuevo; se libera cada backbone antes de cargar el siguiente. Resultado en `phase3/review-artifacts.json`.
- `compare_predictions` aplicado exclusivamente a los dos JSONL históricos referidos por `phase3/test_compare_lora_minus_frozen.json`: comparación exacta con el informe guardado. No se ejecutó inferencia nueva en test ni se eligieron hiperparámetros.

## Veredicto y límites

**Validación funcional de fase 3 superada con correcciones; cierre de reproducibilidad histórica pendiente.** Hay evidencia independiente con pesos reales, no sólo mocks. Los arreglos afectan recuperación y validación de comparaciones; no cambian backend, checkpoint base, prompts, contrato público ni la función aprendida. No requieren reentrenar ni volver a abrir test.

La mejora informada sigue limitada al piloto sintético calibrado. No demuestra mejora general; la transferencia no mejora. Permanecen pendientes la equivalencia continua/reanudada en MPS, la huella de entradas en las comparaciones, los picos de memoria dentro del paso y la trazabilidad señalada. Imagen y servidor siguen fuera de esta fase. No se lanzaron servicios externos, descargas ni entrenamientos largos.
