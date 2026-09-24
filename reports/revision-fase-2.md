# Revisión independiente de fase 2

Fecha de cierre: 2026-09-23 (Europe/Madrid; evaluación nueva con timestamp UTC del 23). Rama `main`, base `65d425025dd654fb4814062dc42b245062f151b1`; las fases 0–2 y esta revisión siguen sin commit.

## Alcance y veredicto

Se revisaron AGENTS.md, README, especificación, auditoría, STATUS, diff y archivos nuevos reales: entrenamiento/pipeline, inferencia, cabezales/estandarizador, checkpoints, datos/generador/splits, métricas y pruebas. Los nuevos archivos no aparecen en el diff normal porque siguen sin seguimiento.

**Puerta de fase 2 aprobada para el piloto sintético**, tras correcciones, pruebas CPU/MPS y recarga real del piloto desde texto en proceso nuevo sin caché. El contraste adicional se recoge al final. No se aprueban generalización, calibración, LoRA, imagen ni API. No se cambió backend, modelo, plantilla ni contrato de salida; tampoco formato de checkpoint, huella o asignación de splits. No se reentrenó el piloto.

## Hallazgos confirmados y correcciones, por gravedad

1. **Alta — evaluadores sin entrenamiento podían producir resultados presentados como aprendidos.** Recortar train con `max_train_questions=1` dejaba primitivas configuradas ausentes, pero se evaluaban en validación y podían influir en selección. El checkpoint distinguía `trained_primitives`, pero evaluación usaba `primitives`. Ahora el pipeline rechaza primitivas solicitadas sin preguntas de train o con peso cero antes de extraer; evaluación rechaza manifiestos que anuncian primitivas sin entrenar. Para entrenar un subconjunto hay que declararlo en `primitives`. Los tres checkpoints de referencia contienen las tres primitivas entrenadas y no están afectados.
2. **Media — temperaturas inválidas permitían respuestas inválidas.** Noul admitía cero/negativas/NaN/Inf; Choice podía devolver NaN con temperatura NaN porque comparar una suma NaN con la tolerancia no detectaba el error. Se exige temperatura finita y positiva en todas las rutas y probabilidades finitas tras softmax. El desbordamiento con una temperatura positiva extrema también falla explícitamente. No se añadió calibración: T=1 sigue siendo la configuración de fase 2.
3. **Media — generación podía no terminar.** `generate_mixed(1,0,questions_per_case=100)` agotaba las reglas posibles y seguía en el bucle de selección. Reproducción en un subproceso, cancelado automáticamente por timeout de dos segundos. Ahora se validan tamaños positivos, máximo de reglas y disponibilidad por caso. Los valores de los fixtures/piloto y su secuencia aleatoria no cambian.
4. **Baja — `--split all` sin dataset externo acababa en `KeyError`.** `load_split` ahora rechaza particiones desconocidas con `SplitError` explicativo, también para fase 1.
5. **Baja — estadísticas de validación Choice confundían IDs opacos con clases.** Renombrar una opción inflaba grupos con etiquetas contrastantes. `validate_examples` usa ahora la descripción elegida para su estadística descriptiva. Los resúmenes de etiquetas de los manifiestos de split existentes se conservan en su formato histórico (pueden contener IDs); no son métricas semánticas ni se usan para aprender. Se preservan hashes e idempotencia de splits, sin reescribir artefactos.
6. **Baja — afirmaciones causales excesivas.** Se matizó decisión 0004: una subida de NLL de train no demuestra por sí sola pasos demasiado grandes ni excluye sobreajuste; la decisión sigue apoyada en la comparación empírica registrada. Se corrigió el comentario que decía que el generador histórico v1 era regenerable.

Además, se rechazan pesos de pérdida NaN/Inf. Las guardas no cambian los cálculos del piloto con temperaturas/pesos válidos.

## Reproducciones y pruebas

- `.venv/bin/pytest tests/unit tests/integration -q` fuera del sandbox, antes de cambios: **191 passed, 5,06 s**.
- `.venv/bin/pytest tests/unit/test_phase2_review.py -q` antes de corregir: **11 failed, 4 passed**. Reproduce subconjunto sin primitivas, partición desconocida, temperaturas inválidas y estadísticas de IDs.
- Tras correcciones, se amplió ese archivo hasta **23 casos**: incluye rechazo temprano del checkpoint con cabezal no entrenado, peso cero solicitado, NaN/Inf en pesos, overflow de softmax y tamaños imposibles del generador.
- `.venv/bin/pytest tests/unit tests/integration -q` fuera del sandbox, final: **214 passed, 5,21 s**.
- `.venv/bin/pytest tests/mps -v -rs` fuera del sandbox: **6 passed, 0 skipped, 38,78 s**. Hay dos pruebas del procesador sin forward GPU; las otras incluyen Gemma diminuto MPS, doctor E2B y sobreajuste/recarga de fases 1 y 2 con E2B real. El test de fase 2 usa 16 preguntas y 300 pasos de época de un cabezal pequeño CPU; no entrenamiento largo del backbone. El modelo base sólo se usa para extracción MPS/BF16.
- `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `git diff --check`: correctos.

Los tests MPS se ejecutaron tras corregir guardas de entrenamiento/evaluación y temperaturas. Después sólo se añadieron límites a entradas imposibles del generador, compatibilidad del resumen histórico de splits y documentación; la suite CPU final cubre esos cambios. Los mocks de regresión no se presentan como evidencia MPS.

## Sospechas del relevo resueltas o acotadas

- **Estandarizador:** inspección del tensor real guardado en `runs/pilot_ce/20260922T212800Z/checkpoint/head.safetensors`: mínimo de escala **0,0038700933**, todas finitas, **0 de 1536** en el suelo de 1e-6. No se reprodujo amplificación por ese suelo en el piloto; sigue siendo riesgo para otros datasets. Se verificó en código y tests que se ajusta sólo con filas train y viaja en el state_dict seleccionado/recargado.
- **BoW:** sigue usando matrices densas FP64; es un coste real de memoria, no un fallo de la pérdida de grupo. No se cambió el baseline durante la revisión para no alterar la comparación.
- **Acumulación:** Σ pérdida ponderada por pregunta / número real de preguntas del paso. Choice/Score concatenan logits antes de softmax/CE; tests comparan pérdidas y gradientes con distintos trozos. Noul sigue CPU en esta etapa; adaptación LoRA sobre MPS requiere sus propias pruebas de autograd y dispositivo de targets.
- **Generador v2 y fugas:** tests entre procesos con semillas de hash distintas, duplicados por serialización, todos los estados del grupo y separación de grupos pasan. Las advertencias de cinco casi duplicados del piloto siguen siendo advertencias, no demostración de independencia semántica total.
- **Transferencia y selección:** el código de entrenamiento selecciona con validation. El dataset externo está marcado diagnóstico y no entra al optimizador. No se puede demostrar la ausencia de decisiones humanas basadas en él sólo leyendo código; no tratarlo como test ciego.
- **Política de etiquetas:** permanece la ambigüedad documentada entre cuenta bloqueada y fallo técnico. Aclararla exige nueva versión del generador/datos, no retocar etiquetas del piloto congelado en esta revisión.

## Artefactos y límites

Referencias conservadas: `runs/pilot_ce/20260922T212800Z/checkpoint`, `runs/pilot_rps/20260922T213423Z/checkpoint` y `runs/mixed_overfit/20260922T213430Z/checkpoint`; los tres son formato 3, con las tres primitivas entrenadas y huella MPS/BF16. No se reescribieron pesos, datasets, splits o manifiestos de checkpoint.

Las métricas históricas de validación tienen sesgo por selección de época, normalización y λ. Score M=5 es débil (0/7 en validación histórica); transferencia comparte familias/reglas y no demuestra nuevas tareas. Confidence sólo mide concentración. Persisten dependencia numérica del lote si se agrupa, picos de memoria no capturados por muestreo, y ausencia de pruebas de LoRA/FP16/visión/contexto >512. El límite `limit` del detector de casi duplicados no se aplica actualmente: coste cuadrático y salida potencialmente grande antes de ampliar datasets.

## Contraste final del piloto real (proceso nuevo)

Comando exacto ejecutado, código de salida 0:

```bash
.venv/bin/gso evaluate --checkpoint runs/pilot_ce/20260922T212800Z/checkpoint --split validation --no-cache > reports/doctor/phase2-review-evaluation.json
```

La CLI escribe dos objetos JSON consecutivos: stdout se renombró a `reports/doctor/phase2-review-evaluation.log`. El informe estructurado completo está en **`runs/pilot_ce/20260922T212800Z/evaluations/validation-20260923T004159Z.json`**. Un intento de parsear stdout como un único JSON falló con `Extra data`; fue un error de lectura de la revisión, no un fallo de inferencia. Las cifras siguientes se leyeron del informe estructurado.

- Backbone real cargado, cache_hit=false; 300 preguntas, 781 filas/forwards, 180849 tokens válidos, cero padding y cero tokens generados.
- Extracción sincronizada: **50,3826 s**; no es un benchmark de servicio.
- Recarga: **300 preguntas comparadas**, diferencia máxima de logit **0,0**, tolerancia 1e-4.
- NLL global **0,4448067161**. Accuracy Noul **0,94488189**, Choice **0,80952381**, Score **0,66292135**. Se reproducen las métricas del piloto histórico, sin reentrenarlo.
- Driver MPS máximo muestreado **11360894976 B**, RSS máximo muestreado **1500102656 B**, swap **0**; sin superar presupuesto. No se suman métricas que pueden solaparse ni se afirma un pico exacto.
- No se ejecutó test ni calibración. La recarga del piloto se hizo tras todas las correcciones funcionales finales.
- Cierre: `pgrep -fl 'gso|gemma_system_one|pytest|uvicorn'` fuera del sandbox, código 1 y sin salida; ningún proceso coincidente activo, sin sesiones de revisión pendientes.
