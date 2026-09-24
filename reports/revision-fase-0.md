# Revisión independiente de fase 0

Fecha: 2026-09-22. Base git: `65d4250`; implementación y correcciones aún sin commit.

## Dictamen

**Aprobada la puerta de fase 0 para texto E2B/MPS/BF16**, tras corregir los fallos descritos y ejecutar las pruebas reales. No se aprueban imagen, LoRA, calidad semántica, calibración ni servicio. La recarga comprobada es del cabezal sobre representaciones guardadas, no del pipeline completo desde texto.

Se contrastaron AGENTS.md, README, especificación, auditoría, STATUS, diff de README, todos los módulos de fase 0 y sus pruebas. Los archivos nuevos están sin seguimiento: no aparecen en `git diff`, por lo que se leyeron directamente. Se inspeccionaron también el JSON histórico `reports/doctor/latest.json` y la implementación instalada de Gemma4 en Transformers. No se hicieron commits, stash, descargas ni entrenamientos largos.

## Hallazgos por gravedad y correcciones

1. **Alta — el presupuesto no detenía las sondas.** `MemoryTracker.sample` sólo almacenaba muestras; el doctor comprobaba el límite al final, después del backward con contexto largo. Una muestra por encima del presupuesto permitía continuar aumentando memoria. Ahora cada muestra comprueba los máximos por métrica y lanza un error; las siguientes sondas tampoco pueden continuar tras una superación. Regresión: presupuesto 100 bytes, RSS simulado 101, antes no lanzaba excepción y ahora sí. Es una prueba del guardarraíl, no una medición de hardware. Límite pendiente: el muestreo no puede impedir un pico dentro de una operación ni sustituye los límites del driver.

2. **Media — verificación de integridad podía dar éxito sin verificar.** Con caché y metadatos remotos inaccesibles, `download(..., verify_hash=True)` devolvía éxito sin calcular ni comparar hashes. Ahora falla explícitamente si no puede obtener la referencia del Hub. Además, un enlace a un blob distinto con el mismo tamaño sólo registraba `blob_name_matches_sha256=false` y era aceptado. Ahora esa discrepancia obliga a calcular y comparar el contenido. Ambas reproducciones fallaron antes de la corrección y pasan después. No se ha añadido una descarga ni una dependencia. El modo offline sin verificación explícita sigue disponible; no certifica integridad actual.

3. **Media — fallo del monitor abortaba el doctor sin informe.** Las muestras inicial y final estaban fuera del bloque que captura errores. Se reprodujo de verdad un `OSError` de `psutil.swap_memory()` bajo el sandbox, además de una regresión pequeña inyectando el fallo. Ahora el paso queda en `fail` y el doctor puede guardar su informe. No se sustituye la métrica por un cero ficticio.

4. **Baja — documentación de relevo excesiva o contradictoria.** README todavía decía que no había software implementado; STATUS decía que no quedaba ningún fallo y recomendaba commit/stash obligatorio. Se actualiza el estado para distinguir evidencia histórica y revisión actual, manteniendo el trabajo previo sin alteraciones de git.

## Evidencia de ejecución

- Antes de cambios: `.venv/bin/pytest tests/unit tests/integration -q` dentro del sandbox: **48 passed, 1 failed**, por `psutil.swap_memory(): OSError`. MPS tampoco era visible dentro del sandbox. No se interpretó como fallo del hardware.
- Mismo comando fuera del sandbox, antes de cambios: **49 passed**.
- Regresiones nuevas, antes de corregir: **3 failed** en `test_review_regressions.py` y **1 failed** en `test_wrong_blob_with_same_size_is_rejected_without_verify`. Reproducciones sin pesos ni red.
- Después de las correcciones: `.venv/bin/pytest tests/unit tests/integration -q` fuera del sandbox: **53 passed**.
- `.venv/bin/pytest tests/mps -v -rs` fuera del sandbox: **2 passed, 0 skipped, 6,96 s**. `test_tiny_gemma4_bf16_forward_backward_on_mps` usa Gemma4 diminuto aleatorio; **sólo `test_doctor_real_e2b` usa el checkpoint E2B real en caché**. Ese segundo test ejecuta el doctor completo: carga, procesador, padding, estados, cabezal, gradientes a través del backbone, 512 tokens, recarga en procesos nuevos y memoria. Las correcciones del monitor estaban aplicadas durante esta ejecución; el ajuste posterior del enlace de caché no afecta la ruta del doctor.
- `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .` y `git diff --check`: correctos.

## Aspectos revisados y límites

- Carga: Gemma4Model real sin cabeza de lenguaje, checkpoint fijado, carga local; errores de claves faltantes/no coincidentes rechazados. Se conserva backend/modelo/contrato, sin decisión arquitectónica nueva.
- Dtype/dispositivo: la prueba E2B comprueba MPS y BF16; no se presenta CPU como MPS. No hay fallback silencioso autorizado ni protecciones de memoria desactivadas.
- Estados/máscaras: última posición válida, padding izquierdo/derecho y campos del procesador conservados. La sonda larga recorta deliberadamente contenido sintético para medir recursos; no es la serialización de producto.
- Gradientes: base congelada en eval para cabezal, extracción con no_grad, cabezal FP32; sondas con autograd explícito hacia capas primera/última y restauración posterior. La prueba diminuta compara todos los parámetros congelados; el doctor E2B vigila tres tensores y ausencia de gradientes, no un hash completo del backbone.
- Prompts: las sondas incluyen estado y pregunta, con etiquetas separadas; no hay dataset, IDs/procedencia o candidatos de producto. Plantilla con criterios, escape, igualdad train/serve, splits y normalización por grupo corresponden a fases 1–2: aún no validados.
- Checkpoint: guarda pesos del cabezal, identidad de base, dtype, plantilla y hash; recarga estricta de pesos. No valida todavía compatibilidad integral de plantilla/procesador/calibración ni recarga desde entrada original. Es un artefacto de diagnóstico, no de despliegue.
- No se ha repetido el hash completo de los 10 GB ni contrastado el contenido del Hub por red en esta revisión. Se usó la caché existente. La comprobación barata por nombre del blob no detecta una corrupción del contenido que conserve nombre y tamaño; para ello se requiere verificación completa explícita.
- No están validados memoria/estabilidad para contextos superiores a 512, LoRA, FP16 del backbone, visión, entrenamiento prolongado ni calidad. La sonda larga tiene texto de longitud finita y puede rechazar configuraciones mayores aunque el esquema permita hasta 8192; no se certifica ese rango.
