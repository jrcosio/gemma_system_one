# Fuentes y alcance de la verificación

Consulta realizada el 22 de septiembre de 2026. Fuentes primarias para contrastar producto, clases y metodología. Las decisiones de arquitectura, límites e hiperparámetros propuestos son del proyecto; no se atribuyen a estas fuentes.

| ID | Fuente | Qué se contrastó |
|---|---|---|
| S1 | [Google: Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | Checkpoint, familia y diferencia entre parámetros efectivos y totales |
| S2 | [Google: Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | Checkpoint, tamaño total con embeddings y modalidades |
| S3 | [Transformers: Gemma4](https://huggingface.co/docs/transformers/model_doc/gemma4) | Base multimodal sin cabeza de lenguaje, campos de entrada y clases |
| S4 | [PyTorch: MPS](https://docs.pytorch.org/docs/main/notes/mps.html) | Backend MPS; no certifica esta combinación de modelo y Mac |
| S5 | [TypeSafe: presentación de System One y Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Existencia del producto, denominación RLCD y claims del proveedor |
| S6 | [TypeSafe: confidence](https://docs.typesafe.ai/confidence) | Confianza como estadística de distribución, sin fórmula exacta en la página consultada |
| S7 | [Guo et al.: On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Calibración posterior y temperature scaling |
| S8 | [PEFT: LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora) | Configuración de adaptación; no certifica Gemma 4 + MPS en este equipo |
| S9 | [TypeSafe: Quick start](https://docs.typesafe.ai/introduction/quickstart) | Endpoint oficial y ejemplo del SDK Python |
| S10 | [TypeSafe: primitivas](https://docs.typesafe.ai/primitives) | Noul, Choice y Score, preguntas independientes y criterios |
| S11 | [Google: versiones de Gemma](https://ai.google.dev/gemma/docs/releases) | Existencia de Gemma 4 frente a la referencia antigua a PaliGemma 2 |

## Cómo interpretar la evidencia

- **Verificado documentalmente:** los checkpoints Gemma 4 citados existen; las clases documentadas permiten una base sin cabeza generativa; TypeSafe publica primitivas y SDK.
- **Derivado matemáticamente:** memoria mínima orientativa de pesos como número de parámetros por bytes; media y concentración obtenidas de distribuciones; estas cuentas no predicen el consumo completo.
- **Decisiones propuestas:** E2B primero, PyTorch/MPS, evaluador por candidato, límites de entrada y fases de entrenamiento.
- **Pendiente de ejecución:** combinación exacta de versiones, soporte de todos los operadores, backward, dtype estable, LoRA, memoria y rendimiento en el M5 Pro concreto.
- **No demostrado:** equivalencia con la arquitectura o RLCD de Jev, compatibilidad exacta del SDK, calibración universal o rendimiento de nivel frontera.

Las páginas `main` y otras URLs sin versión pueden cambiar. En fase 0 se debe guardar en el reporte la versión/commit realmente usado. No tomar la disponibilidad de una documentación como prueba de que las dependencias instaladas ya incorporan esa funcionalidad.

No se han usado sitios que imitan el nombre de Jev/TypeSafe como fuente oficial. Tampoco se han extrapolado cifras comerciales de Jev al Mac local.
