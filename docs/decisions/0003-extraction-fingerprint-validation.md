# 0003 — Identidad efectiva de extracción y validación de particiones

Fecha: 2026-09-22. Estado: aceptada en revisión de fase 1.

## Motivo y evidencia

La huella anterior usaba `runtime.device` solicitado. Con MPS no disponible y fallback autorizado se extraían representaciones CPU bajo identidad MPS. Además, cambiar `max_length` no cambiaba la clave: un acierto de caché podía saltarse el límite sin tokenizar. Ambas condiciones se reprodujeron en `tests/unit/test_phase1_review.py` antes de corregir.

## Decisión

- Registrar el dispositivo resuelto por la misma función que carga el backbone e incluir `max_length` en la huella.
- Rechazar extracción identificada como MPS con `PYTORCH_ENABLE_MPS_FALLBACK` activo, igual que el doctor. CPU explícita o fallback de dispositivo autorizado siguen identificados como CPU.
- No migrar silenciosamente huellas anteriores: cachés antiguas no se reutilizan y los checkpoints antiguos se rechazan al evaluar por discrepancia de huella. Conservarlos como historial y regenerar los fixtures bajo la configuración actual.
- Validar el manifiesto completo al cargar cualquier partición: formato, IDs únicos, cobertura, hashes, grupos declarados y ausencia de fugas exactas/de grupo. Esto inspecciona la estructura del dataset completo; sólo las particiones solicitadas se devuelven al consumidor.
- Sustituir el campo interno de reporte `splits_not_read` por `splits_not_used_for_fitting`: el cargador lee/valida el JSONL completo, pero no usa las particiones reservadas para gradientes ni selección de época. Los informes históricos mantienen el nombre antiguo y deben interpretarse con esa limitación.

No cambia el modelo E2B, PyTorch/MPS, dtype, política de una fila por forward, plantilla ni contrato público Noul/Choice/Score. Cambia la identidad interna de artefactos y una descripción incorrecta del reporte. No se borran ni sobrescriben artefactos previos.
