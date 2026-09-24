# Gemma System One local

Paquete de especificaciones corregidas para iniciar un proyecto de adaptación de Gemma 4 en un Mac M5 Pro con 48 GB, usando Claude Code con Fable 5.1 y Codex con GPT Astra.

**Recomendación:** empezar con Gemma 4 E2B, validar PyTorch/MPS y entrenar cabezales antes de LoRA. E4B es el siguiente candidato tras medir memoria y calidad. La salida serán decisiones Noul, Choice y Score sin generación de texto en inferencia.

El repositorio implementa la fase 0 (descarga, inspección y diagnóstico real de Gemma 4), la fase 1 (contratos, plantilla, datos por grupos, baselines y cabezal Noul), la fase 2 (Choice/Score con evaluadores compartidos y pérdida sobre el grupo completo, métricas por K/M, piloto de 1000 casos y conjunto de transferencia) la fase 3 (LoRA en la atención textual con autograd real, checkpoints de despliegue y de reanudación, calibración por temperatura y evaluación ciega en test) la fase 4 (una imagen por ejemplo con visión congelada, datos visuales verificables con estilos separados por partición, control sin imagen, ablaciones y límites de memoria) la fase 5 (API local con un worker, cola acotada, límites, E2E con checkpoints entrenados y benchmark) y la fase 6 (perfilado y adopción medida de E4B, más opciones en Choice, recomputación de activaciones para LoRA y agrupación de filas medida y descartada).

## Arranque en cinco pasos

1. Descomprime el paquete en una carpeta nueva que será la raíz del repositorio.
2. Lee la auditoría para entender qué se ha corregido respecto a los seis originales.
3. Abre la carpeta con Claude Code o Codex; ambos comparten `AGENTS.md`.
4. Copia el prompt de arranque de `PROMPTS.md`: implementará y comprobará la fase 0.
5. Usa el otro asistente para revisar la fase y continúa según sus criterios de aceptación.

## Fase 0: cómo ejecutar

Implementado: paquete `gemma_system_one` (CLI `gso`), configuración tipada, descarga fijada y comando doctor. Estado y evidencia en [docs/STATUS.md](docs/STATUS.md) y [reports/compatibility.md](reports/compatibility.md).

```bash
uv venv --python 3.11 && uv sync
uv run gso download --config configs/e2b_text.yaml   # ~10,25 GB una sola vez; después usa la caché
uv run gso doctor --config configs/e2b_text.yaml     # prueba real en MPS: carga, forward/backward, memoria, recarga
uv run pytest tests/unit tests/integration -q        # contratos en CPU (sin pesos)
uv run pytest tests/mps -rs                          # hardware real (skip con motivo si falta MPS o pesos)
```

## Fase 1: cómo ejecutar

Evidencia en [reports/phase1-noul.md](reports/phase1-noul.md). Fixture sintético de humo; no es un resultado de calidad.

```bash
uv run gso generate-data --out data/smoke_v1 --cases 40 --seed 0
uv run gso validate-data --dataset data/smoke_v1
uv run gso split --dataset data/smoke_v1 --seed 0            # manifiesto inmutable de 4 particiones por grupo
uv run gso baselines --config configs/noul_smoke.yaml         # prior y bolsa de palabras (sin Gemma)
uv run gso train --config configs/noul_overfit.yaml           # puerta: sobreajuste controlado de 32 ejemplos
uv run gso train --config configs/noul_smoke.yaml             # cabezal Noul, época elegida en validación
uv run gso evaluate --checkpoint runs/noul_smoke/<ts>/checkpoint --split validation --no-cache
```

`evaluate` rechaza `--split test` sin `--final-test`; calibración y test no se usan en esta fase.

## Fase 2: cómo ejecutar

Evidencia en [reports/phase2-decisions.md](reports/phase2-decisions.md). Datos sintéticos por reglas; no es un resultado de calidad general.

```bash
uv run gso generate-data --kind mixed --generator-version v2 --out data/pilot_v2 --cases 1000 --seed 0
uv run gso generate-data --kind mixed --generator-version v2 --variant transfer --out data/pilot_transfer_v2 --cases 200 --seed 0
uv run gso split --dataset data/pilot_v2 --seed 0
uv run gso train --config configs/mixed_overfit.yaml        # puerta: sobreajuste de 48 preguntas mixtas
uv run gso train --config configs/pilot_ce.yaml             # piloto (CE); configs/pilot_rps.yaml para CE+RPS
uv run gso evaluate --checkpoint runs/pilot_ce/<ts>/checkpoint --split validation --no-cache --robustness
uv run gso evaluate --checkpoint runs/pilot_ce/<ts>/checkpoint --split all --dataset data/pilot_transfer_v2
```

## Fase 3: cómo ejecutar

Evidencia en [reports/phase3-lora.md](reports/phase3-lora.md), protocolo del test en [reports/phase3-test-protocol.md](reports/phase3-test-protocol.md) y decisiones [0005](docs/decisions/0005-lora-stage.md) y [0006](docs/decisions/0006-generator-v3-access-policy.md). Datos sintéticos; no es un resultado de calidad general.

```bash
uv run gso generate-data --kind mixed --generator-version v3 --out data/pilot_v3 --cases 1000 --seed 0
uv run gso split --dataset data/pilot_v3 --seed 0
uv run gso train --config configs/pilot_ce_v3.yaml            # referencia: E2B congelado + cabezales
uv run gso train --config configs/pilot_lora_v3.yaml          # LoRA + cabezales, arranca de la referencia
uv run gso train --config configs/pilot_lora_v3.yaml --resume runs/pilot_lora_v3/<ts>   # tras una interrupción
uv run gso calibrate --checkpoint runs/<run>/<ts>/checkpoint --split calibration
uv run gso evaluate --checkpoint runs/<run>/<ts>/checkpoint --split test --final-test --no-cache \
    --calibration runs/<run>/<ts>/calibration/calibration-<ts>.json --baselines   # una vez, artefacto congelado
uv run gso compare --a <predicciones A> --b <predicciones B>                     # diferencia emparejada b − a
```

`gso calibrate` sólo acepta la partición `calibration`; el artefacto queda vinculado por sha256 al checkpoint y `evaluate --calibration` lo rechaza con otro checkpoint. `--stop-after-steps N` interrumpe un entrenamiento LoRA de forma controlada y guarda el estado de reanudación.

## Fase 4: cómo ejecutar

Evidencia en [reports/phase4-vision.md](reports/phase4-vision.md), protocolo en [reports/phase4-test-protocol.md](reports/phase4-test-protocol.md) y decisión [0007](docs/decisions/0007-single-image-path.md). Paneles de barras sintéticos; no es comprensión visual general.

```bash
uv run gso doctor --config configs/e2b_vision.yaml                              # incluye el paso de imagen
uv run gso generate-data --kind vision --vision-version v1 --out data/vision_pilot_v1 --cases 700 --seed 0   # piloto histórico
uv run gso generate-data --kind vision --vision-version v2 --out data/vision_pilot_v2 --cases 700 --seed 0 --split-seed 0
uv run gso split --dataset data/vision_pilot_v2 --seed 0        # comprueba el reparto planificado (estilos por partición)
uv run gso train --config configs/vision_overfit.yaml                           # puerta con imagen
uv run gso train --config configs/vision_heads.yaml                             # V: con imagen
uv run gso train --config configs/vision_text_only.yaml                         # C: mismas filas sin imagen
uv run gso evaluate --checkpoint runs/vision_heads/<ts>/checkpoint --split validation --no-cache --vision-ablation
uv run gso compare --a <predicciones C> --b <predicciones V> --allow-different-inputs
```

Las imágenes de un dataset se nombran por su sha256 (`images/<sha256>.png`); con imagen, la extracción usa una fila por forward. `scripts/snapshot_source.py` guarda una copia de las fuentes bajo el hash que registran los manifiestos.

## Fase 5: cómo ejecutar

Evidencia en [reports/phase5-api.md](reports/phase5-api.md) y decisión [0009](docs/decisions/0009-local-api.md). Cierre de la fase 4 con estilos separados por partición: [reports/phase4b-styles.md](reports/phase4b-styles.md) y [0008](docs/decisions/0008-vision-styles-per-split-and-source-archives.md).

```bash
uv run gso serve --config configs/serve_text.yaml      # API real en 127.0.0.1:8000 (un proceso, un worker)
curl -s http://127.0.0.1:8000/health/ready             # 200 sólo tras cargar, verificar y hacer warmup
uv run gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation --requests 100 --warmup 5
uv run pytest tests/e2e -v -rs                          # E2E con E2B real y checkpoints entrenados (MPS)
```

`POST /v1/decide` recibe `model`, `state`, `questions` (1–8) e `image` opcional (PNG/JPEG en base64, sólo en checkpoints entrenados con imagen). Devuelve `answers`, `usage` (`generated_tokens = 0`) y `metadata`. Errores: 422, 413, 400, 404, 503 y 500 según la spec §9.

## Fase 6: cómo ejecutar

Evidencia en [reports/phase6-e4b.md](reports/phase6-e4b.md), protocolo predeclarado en [reports/phase6-protocol.md](reports/phase6-protocol.md) y decisiones [0010](docs/decisions/0010-phase6-recompute-and-batching.md) y [0011](docs/decisions/0011-e4b-frozen-heads-text-service.md). En un holdout nuevo, E4B congelado + cabezales mejora a E2B + LoRA (NLL −0,18); es el servicio de texto recomendado, con ~1,6× de latencia.

```bash
uv run gso download --config configs/e4b_text.yaml          # 16 GB, revisión fijada; una sola vez
uv run gso doctor --config configs/e4b_text.yaml
uv run gso train --config configs/e4b_experiment.yaml       # A4: mismos datos e hiperparámetros que pilot_ce_v3
uv run gso calibrate --checkpoint runs/e4b_experiment/<ts>/checkpoint --split calibration
uv run python scripts/derive_phase6_data.py                 # holdout sin solapes y variante con K = 8
uv run gso serve --config configs/serve_e4b_text.yaml       # servicio de texto recomendado
uv run python scripts/profile_lora_step.py configs/e4b_text.yaml data/pilot_v3 out.json 6 --recompute
```

En el perfil de seis pasos de LoRA sobre E4B, la ejecución sin recomputación superó el presupuesto de 32 GiB y la ejecución con `train.recompute_layers: true` permaneció por debajo. No se ha medido un entrenamiento completo de E4B con LoRA.

Fase 6b ([reports/phase6b-final.md](reports/phase6b-final.md)): en un test final nuevo que no se usó para ajustar pesos ni temperaturas, A4 − B2 = −0,146 [−0,186; −0,104]. La regla C predeclarada usa ese test para confirmar A4. Las calibraciones se ajustan con un conjunto externo de 1200 preguntas (`gso calibrate --split all --dataset data/pilot_v3_calib7`). La revisión posterior detectó una fuga estructural de etiqueta en la familia sintética `fault_type`; véanse los límites del informe.

Fase 6c ([reports/phase6c-final.md](reports/phase6c-final.md), decisión [0012](docs/decisions/0012-generator-v4-without-k-cue.md)): el generador `--generator-version v4` elimina la pista determinista del número K. A4v4 − A4v3 = −0,007 [−0,034; +0,024] de NLL calibrada en el test v4; el margen predeclarado no se cumple y se mantiene A4v3. La revisión posterior encontró una pista parcial en la composición de las opciones ampliadas a K8; ese diagnóstico no demuestra robustez sin fuga.

Fase 6d ([reports/phase6d-final.md](reports/phase6d-final.md)): diagnóstico emparejado K4/K8 con cuatro distractoras fijas. La tolerancia a K = 8 no queda demostrada para ningún modelo (IC anchos; la respuesta `other` es el punto débil). La revisión retiró la conclusión de que las opciones no aportan pistas: 0,000 es una ganancia de accuracy top-1, no información mutua cero, y el control con estados intercambiados conserva etiquetas que suelen contradecir el estado donante. El generador de entrenamiento v4 muestra 0,050 de ganancia top-1 por composición en el probe; investigar un v5 queda pendiente.

Fase 6e ([reports/phase6e-final.md](reports/phase6e-final.md)): tríos con la misma pregunta y distinto estado (techo sin leer el estado: 1/3). Los tres modelos usan el estado en este diagnóstico sintético (tríos completos: A4v3 0,58; A2v4 0,21). La tolerancia a K = 8 sólo queda demostrada para A2v4 en este conjunto. Hay errores `other` hacia «aplicación» en fallos de rendimiento y datos; el posible solapamiento de categorías es una hipótesis para un v5, no una causa demostrada.

Fase 6f ([reports/phase6f-final.md](reports/phase6f-final.md), decisión [0013](docs/decisions/0013-generator-v5-exclusive-definitions.md)): generador `--generator-version v5` con definiciones excluyentes («aplicación» no incluye lentitud ni pérdida de datos) y opciones sorteadas sin mirar los hechos. Entrenar con v5 mejora los tríos completos de E4B (+0,087 [+0,020; +0,160]) y casi elimina el error `other` → «aplicación», pero no se demuestra la no inferioridad en NLL (−0,007 [−0,037; +0,025]), así que se mantiene A4v3 como servicio. El error residual está en los fallos ya resueltos.

## Documentos

| Archivo | Uso |
|---|---|
| [docs/ESPECIFICACION.md](docs/ESPECIFICACION.md) | Diseño completo: entorno, arquitectura, datos, entrenamiento, calibración, API, fases y tests |
| [docs/AUDITORIA.md](docs/AUDITORIA.md) | Revisión detallada de cada archivo original y correcciones |
| [AGENTS.md](AGENTS.md) | Reglas comunes de desarrollo y validación |
| [CLAUDE.md](CLAUDE.md) | Entrada de contexto de Claude Code |
| [PROMPTS.md](PROMPTS.md) | Prompts de arranque, revisión, continuación y relevo |
| [docs/FUENTES.md](docs/FUENTES.md) | Fuentes primarias, hechos contrastados y aspectos pendientes |
| [docs/STATUS.md](docs/STATUS.md) | Estado del trabajo, comandos ejecutados, resultados y relevo |

## Cambios esenciales

- Gemma 4 real como base, en lugar de PaliGemma 2.
- Adaptación supervisada y calibración posterior; no se presenta como reproducción de RLCD.
- Opciones y rúbricas variables representadas semánticamente, con coste explícito por candidato.
- Datos diversos y verificables; splits separados de entrenamiento, validación, calibración y test.
- Inferencia real y probabilidades calculadas, eliminando constantes y métricas inventadas.
- Perfilado de memoria en el equipo antes de aumentar contexto, candidatos o tamaño del modelo.
- API propia; compatibilidad con TypeSafe sólo como ampliación demostrada con tests.

## Qué significa éxito

Un proyecto que pueda entrenar un evaluador, demostrar mejora frente a baselines, medir su calibración, recargar sus pesos y responder por HTTP con resultados reales. Que una salida respete el esquema no garantiza que sea correcta. La calidad, latencia y memoria deberán quedar respaldadas por reportes reproducibles.

Fecha de revisión documental: 22 de septiembre de 2026.
