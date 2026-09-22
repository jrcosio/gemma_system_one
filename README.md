# Gemma System One local

Paquete de especificaciones corregidas para iniciar un proyecto de adaptación de Gemma 4 en un Mac M5 Pro con 48 GB, usando Claude Code con Fable 5.1 y Codex con GPT Astra.

**Recomendación:** empezar con Gemma 4 E2B, validar PyTorch/MPS y entrenar cabezales antes de LoRA. E4B es el siguiente candidato tras medir memoria y calidad. La salida serán decisiones Noul, Choice y Score sin generación de texto en inferencia.

Este paquete contiene Markdown para desarrollar el proyecto. No incluye software de entrenamiento implementado ni resultados obtenidos en tu Mac. Los comandos del diseño son requisitos para los agentes, no comandos ya disponibles al descomprimir.

## Arranque en cinco pasos

1. Descomprime el paquete en una carpeta nueva que será la raíz del repositorio.
2. Lee la auditoría para entender qué se ha corregido respecto a los seis originales.
3. Abre la carpeta con Claude Code o Codex; ambos comparten `AGENTS.md`.
4. Copia el prompt de arranque de `PROMPTS.md`: implementará y comprobará la fase 0.
5. Usa el otro asistente para revisar la fase y continúa según sus criterios de aceptación.

## Documentos

| Archivo | Uso |
|---|---|
| [docs/ESPECIFICACION.md](docs/ESPECIFICACION.md) | Diseño completo: entorno, arquitectura, datos, entrenamiento, calibración, API, fases y tests |
| [docs/AUDITORIA.md](docs/AUDITORIA.md) | Revisión detallada de cada archivo original y correcciones |
| [AGENTS.md](AGENTS.md) | Reglas comunes de desarrollo y validación |
| [CLAUDE.md](CLAUDE.md) | Entrada de contexto de Claude Code |
| [PROMPTS.md](PROMPTS.md) | Prompts de arranque, revisión, continuación y relevo |
| [docs/FUENTES.md](docs/FUENTES.md) | Fuentes primarias, hechos contrastados y aspectos pendientes |

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
