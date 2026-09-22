# Auditoría de los seis documentos originales

Revisión: 22 de septiembre de 2026. Esta auditoría distingue errores del código propuesto, afirmaciones sin evidencia y decisiones de diseño que necesitan experimentos. La documentación técnica revisada está en [FUENTES.md](FUENTES.md).

## Dictamen

La idea es viable como proyecto de investigación aplicado: usar representaciones de un modelo preentrenado para producir decisiones tipadas mediante cabezales aprendidos. Los originales no constituyen todavía una receta ejecutable de Gemma 4 ni justifican reproducir Jev. El riesgo principal sería conseguir una API que devuelve números plausibles sin haber demostrado aprendizaje, calibración o generalización.

La viabilidad específica del entrenamiento en M5 Pro de 48 GB requiere medir una ruta real. El plan revisado empieza con E2B y conserva E4B como candidato. No sustituye Gemma 4 por PaliGemma ni promete tiempos de entrenamiento.

## 1. `claude_code_agent_master_guide.md`

| Hallazgo | Consecuencia | Corrección |
|---|---|---|
| Presenta PaliGemma 2 como base de un proyecto llamado Gemma 4 | Arquitectura, clases y pesos no corresponden al objetivo | Gemma 4 E2B inicial y revisión fijada |
| Afirma reproducir principios de una arquitectura Jev | La semejanza de salidas no demuestra semejanza interna | Proyecto inspirado en el paradigma, con arquitectura propia |
| Llama deterministas a primitivas probabilísticas | Puede confundirse contrato de tipos con verdad o repetibilidad bit a bit | Contrato determinista; predicción estadística |
| Denomina RLCD a Brier y entropía | No hay política/recompensa/algoritmo RL especificado | Entrenamiento supervisado + calibración posterior |
| Promete una pasada para todo | Preguntas y candidatos necesitan representaciones diferentes | Contar forwards y filas; batching explícito |
| Incluye comandos que no están implementados | Falsa sensación de proyecto ejecutable | Marcar CLI como contrato a construir por fases |
| Sólo prepara contexto para Claude | Codex puede trabajar con reglas distintas | `AGENTS.md` común y `CLAUDE.md` que remite a él |

## 2. `configuraci_n_de_entorno_y_descarga.md`

| Hallazgo | Consecuencia | Corrección |
|---|---|---|
| Reserva 36–40 GB para GPU y promete >20 GB libres | Memoria unificada y activaciones dependen del sistema y entrada | Presupuesto conservador y medidas reales |
| Usa ~3,5B como base del cálculo | No representa los parámetros totales de E4B | Distinguir efectivos/totales y leer checkpoint |
| Asigna tamaños fijos a LoRA/AdamW | Dependen de módulos, rango, dtype y estados | Contar parámetros y medir después del primer paso |
| Dependencias con mínimos antiguos y sin lock | No garantizan soporte de Gemma 4 | Resolver, comprobar clases y fijar versiones |
| Instala SDKs y audio sin necesidad definida | Más conflictos sin beneficio para el MVP | Incorporación por fase |
| Descarga mediante carga + `save_pretrained` | Puede duplicar pesos y provocar carga en dispositivo innecesaria | Snapshot con revisión y caché |
| `device_map="auto"` en receta de entrenamiento | Distribución de dispositivos implícita | Dispositivo controlado y reporte |
| Test de una multiplicación | No valida modelo, procesador, backward ni PEFT | Smoke completo en hardware objetivo |

## 3. `especificaci_n_del_dataset_sint_tico_y_benchmarks.md`

| Hallazgo | Consecuencia | Corrección |
|---|---|---|
| Cuatro textos repetidos 500 veces | Memorización y fugas al dividir aleatoriamente | Casos únicos y splits por familias/grupos |
| Probabilidades inventadas más ruido uniforme | No son etiquetas calibradas ni resultados observados | Etiquetas duras verificables; distilación separada |
| Script sólo genera Noul | Choice y Score no se entrenan | Dataset discriminado por primitiva |
| CLI anunciada `--samples` no existe en el script | Instrucciones inconsistentes | CLI real con argumentos validados |
| `state` contiene una ruta de imagen | No incorpora contenido visual y puede filtrar etiquetas | Campo de imagen separado y archivo verificado |
| No hay val/calibration/test independientes | Ajuste y medición pueden contaminarse | Cuatro particiones y manifiestos |
| El título promete benchmarking pero no lo define | No hay comparación ni criterio de éxito | Baselines, métricas por tarea y rendimiento medido |

El ruido no soluciona la repetición. Un profesor LLM que dice «0,95» tampoco demuestra que el evento ocurra el 95% de las veces.

## 4. `especificaci_n_te_rica_y_formulaci_n_matem_tica.md`

| Hallazgo | Consecuencia | Corrección |
|---|---|---|
| Diagrama hace confluir texto en el proyector visual | Describe de forma equívoca la fusión multimodal | Procesador y backbone propios de Gemma 4 |
| Un único h genérico para cualquier pregunta | Si no contiene pregunta/criterios, no puede distinguir juicios | Serialización condicionada por pregunta |
| Cabezal fijo de K posiciones para opciones arbitrarias | No establece correspondencia semántica robusta | Scoring compartido por candidato |
| Softmax + esperanza llamado ordinal | No modela explícitamente distancias entre niveles | RPS acumulativo como experimento ordinal |
| Entropía tratada como confianza de acierto | Una predicción incorrecta puede tener entropía cero | Concentración y calibración separadas |
| Brier menos entropía presentado como regla propia | El término extra desplaza el óptimo | Baseline sin ese regularizador |
| ECE <0,03 presentado como objetivo suficiente | Depende de bins, muestra y distribución | NLL/Brier/RPS, ECE definido e incertidumbre |

Las fuentes consultadas no dan una receta reproducible de RLCD que respalde las fórmulas inventadas en los originales. Esto no afirma que no exista material adicional; delimita lo que se ha podido verificar.

## 5. `pipeline_de_entrenamiento_e_implementaci_n.md`

| Hallazgo | Consecuencia | Corrección |
|---|---|---|
| Elimina `language_model.lm_head` con `del` | El forward puede seguir esperando el atributo; ruta depende de versión | Usar base sin cabeza generativa |
| Ejecuta wrapper generativo para estados ocultos | Puede calcular logits masivos del vocabulario | Ruta base y última capa |
| `output_hidden_states=True` | Retiene todas las capas innecesariamente | Extraer sólo última representación |
| Pooling `[:, -1, :]` | Lee padding con lotes heterogéneos | Última posición válida según máscara |
| LoRA por nombres genéricos globales | Puede afectar módulos no deseados | Lista de módulos textuales inspeccionada |
| BF16 asumido válido | No demuestra estabilidad MPS de todos los operadores | Smoke completo por dtype |
| Bucle sólo Noul con `batch['label']` | No coincide con los campos del dataset original | Collator y targets por primitiva |
| Sin acumulación, checkpoints ni validación | Entrenamiento difícil de reproducir/reanudar | Pipeline por etapas con manifiestos |
| `model.train()` indiscriminado | Puede activar dropout del backbone congelado | Controlar modo por submódulo |

Además: en LoRA no se puede envolver la extracción en `no_grad()`. En entrenamiento de cabezales congelados sí puede hacerse. La distinción debe quedar cubierta por tests de gradientes.

## 6. `especificaci_n_y_servidor_de_inferencia.md`

| Hallazgo | Consecuencia | Corrección |
|---|---|---|
| Respuestas 0,96 / 0,89 / 1,84 codificadas | No existe inferencia | Cargar checkpoint real; 503 si falta |
| `usage.input_tokens=420` | Contabilidad ficticia | Medir entradas realmente procesadas |
| Devuelve identidad `jev-1.13.0` | Identifica incorrectamente el modelo local | Nombre propio vinculado a manifiesto |
| Promete contrato exacto | No incluye prueba de conformidad del SDK | API propia y adapter opcional probado |
| Tipo libre y criterios opcionales universales | Acepta estados inválidos | Unión discriminada y reglas por tipo |
| División `0.11/(K-1)` | Falla con K=1 | K mínimo validado; eliminar respuesta ficticia |
| Score fijo y leyenda variable | Rango/distribución pueden no coincidir con rúbrica | Score derivado de probabilidades reales |
| Endpoint async sin modelo/cola | Sustituir constantes por cómputo directo bloquearía event loop | Worker dedicado y cola limitada |
| Imágenes sin límite de bytes/píxeles | Agota recursos con entradas grandes | Límites previos a decodificación |
| `0.0.0.0 --reload` como receta común | Exposición y reinicios innecesarios para uso local | Loopback, un proceso, sin reload en mediciones |

La documentación oficial sí muestra `typesafe-sdk`, `TypeSafeClient` y `/v1/systemone`. No se eliminan por ser inexistentes: se retiran del núcleo porque no validan la compatibilidad de este servidor, su transporte, sus límites ni la definición de confianza.

## Cambios deliberados de alcance

- Entrenar E2B antes de E4B para reducir el coste de depurar; no renunciar al modelo mayor.
- Limitar inicialmente Choice a 8 opciones y Score a 5 niveles para medir coste y calidad.
- Texto primero, imagen después; ambas modalidades forman parte del objetivo final.
- Priorizar PyTorch/MPS para cabezales personalizados; MLX queda como alternativa técnica documentada, no como garantía sin pruebas.
- Reemplazar snippets incompletos por especificaciones implementables y criterios de aceptación. El paquete no contiene un entrenamiento supuestamente probado.
