# Gemma System One local — especificación de arranque

Versión 1.0 · 22 de septiembre de 2026 · Hardware objetivo: Mac M5 Pro, 48 GB de memoria unificada.

## 1. Objetivo y alcance

Crear desde cero el repositorio para adaptar un checkpoint preentrenado de Gemma 4 a decisiones probabilísticas tipadas. «Desde cero» se refiere al proyecto y a sus cabezales, no al preentrenamiento de miles de millones de parámetros.

Entrada: un estado textual o JSON, una imagen opcional y preguntas con criterios explícitos. Salida: probabilidades binarias (`noul`), selección entre opciones (`choice`) y distribución sobre una rúbrica ordenada (`score`). La inferencia del producto no usa `generate()`, no genera explicaciones ni cadenas de razonamiento y no analiza JSON escrito por el modelo. El código construye la respuesta a partir de tensores.

El objetivo es un prototipo local evaluable inspirado en System One. No es una reproducción demostrada de Jev ni de su entrenamiento RLCD. Las decisiones pueden ser incorrectas aunque siempre respeten el esquema. Una distribución concentrada tampoco garantiza acierto.

Primera versión: texto en español e inglés; después texto + una imagen. Audio, vídeo, decisiones encadenadas dentro de una llamada, herramientas, RAG y modelos de gran tamaño quedan para ampliaciones independientes. Primera familia de tareas: clasificación de incidencias de soporte con políticas explícitas. Es un banco de pruebas inicial; no se presupone que generalice a cualquier negocio.

### Decisiones iniciales

| Elemento | Decisión |
|---|---|
| Checkpoint inicial | `google/gemma-4-E2B-it`, fijando revisión concreta |
| Candidato posterior | `google/gemma-4-E4B-it`, sólo después del perfilado |
| Backend principal | PyTorch + Transformers sobre MPS |
| Backend alternativo | MLX, mediante decisión documentada si MPS bloquea el proyecto |
| Adaptación | Cabezales primero; LoRA en el transformer textual después |
| Entrenamiento | Supervisado; calibración posterior independiente |
| Opciones variables | Evaluador compartido condicionado por candidato |
| Servicio | FastAPI local, un proceso de inferencia y cola limitada |
| Gestión de entorno | Python 3.11 como punto inicial, `uv`, `pyproject.toml`, `uv.lock` |
| Desarrollo | Claude Code con Fable 5.1 y Codex con GPT Astra; mismos contratos |

Los nombres de asistentes son las preferencias del usuario, no identificadores de API que deban inventarse en scripts. Su uso para desarrollar no implica enviarles datos de entrenamiento ni contratarlos como generadores externos.

## 2. Viabilidad y puerta de entrada en el Mac

### 2.1 Lo conocido y lo pendiente

Las fichas oficiales distinguen parámetros efectivos y totales: E2B figura con aproximadamente 5,1B incluyendo embeddings y E4B con 8B. Los pesos a dos bytes por parámetro representan aproximadamente 10,2 y 16 GB decimales, respectivamente, como orientación; el checkpoint y los componentes realmente cargados mandan. No son cifras de memoria total de entrenamiento. [S1, S2]

La memoria unificada la comparten macOS, CPU y GPU. Hay que añadir activaciones, buffers, gradientes, estados del optimizador, procesador, copias temporales y otras aplicaciones. LoRA reduce parámetros entrenables, pero no elimina el coste de propagar gradientes por las capas adaptadas.

No se ha ejecutado este proyecto en el Mac del usuario. MPS disponible no demuestra que toda la ruta Gemma 4 + procesador + LoRA funcione. Tampoco se afirma una latencia, tamaño de lote o duración de entrenamiento antes de medirlos.

### 2.2 Prueba de compatibilidad obligatoria: fase 0

El agente debe crear un comando `gso doctor` que registre:

1. Arquitectura ARM64, modelo de máquina, memoria, macOS, Python y versiones instaladas.
2. Disponibilidad MPS y prueba de operaciones + backward con el dtype elegido.
3. Identidad y revisión del checkpoint, procesador y configuración, sin cargar pesos para inspeccionar metadatos.
4. Carga del backbone, pasada textual real y extracción del estado final.
5. Pérdida escalar, backward, actualización de un cabezal y de LoRA cuando se pruebe esa fase.
6. Parámetros entrenables por nombre, gradientes finitos y cambio de pesos esperado.
7. Memoria del proceso, memoria MPS disponible en la versión instalada, presión del sistema, swap y tiempo sincronizado.
8. Después: repetición con una imagen y guardado/recarga de un checkpoint mínimo.

`reports/compatibility.md` debe incluir comandos, salidas resumidas, versiones exactas, limitaciones y veredicto por modalidad. Un test con modelo diminuto valida contratos, pero no sustituye la prueba con los pesos reales.

### 2.3 Ajustes iniciales propuestos, no resultados

| Parámetro | Inicio conservador |
|---|---|
| Longitud de entrada | 512 tokens procesados por fila, incluyendo el prompt completo |
| Microbatch | 1 fila expandida |
| Acumulación | 8 preguntas lógicas; ver normalización de pérdida |
| Dtype del backbone | BF16 si la ruta completa pasa; FP16 sólo con prueba de estabilidad |
| Cabezales, logits y pérdidas | FP32 |
| LoRA | Rango 8, alpha 16, dropout 0,05 |
| Atención | Implementación que pase pruebas MPS; registrar cuál |
| Caché de decodificación | `use_cache=False` |
| DataLoader | `num_workers=0` inicialmente |
| Límite operativo provisional | Presupuesto conservador de 32 GiB para el trabajo ML, vigilando el sistema |

No desactivar límites de memoria MPS como receta para evitar OOM. No usar CPU fallback silencioso para presentar tiempos como aceleración completa. La ruta de diagnóstico puede usar CPU si queda identificada.

Si hay presión o swap creciente: reducir longitud, número de candidatos por microbatch y resolución/tokens visuales; activar checkpointing de gradientes cuando corresponda; probar E2B; entrenar sólo cabezales con representaciones cacheadas. Si persiste un operador incompatible, documentar y decidir un backend alternativo. No montar dos stacks completos desde el inicio.

Para medir inferencia en MPS sincronizar antes y después del intervalo. Diferenciar memoria del proceso y del driver: no sumarlas sin comprobar solapamientos. Registrar el pico observado y la técnica de muestreo; no presentarlo como pico exacto si el muestreo puede omitirlo.

## 3. Entorno reproducible

El primer agente implementará el paquete instalable `gemma_system_one`, con entrypoint `gso`. `src` es el directorio de distribución, no el nombre del paquete.

Dependencias iniciales: `torch`, `transformers`, `huggingface_hub`, `safetensors`, `numpy`, `pydantic`, `PyYAML` y `psutil`; incorporar `peft` en adaptación, `Pillow` en visión, `fastapi`, `uvicorn` y `httpx` en servicio. Desarrollo: `pytest` y `ruff`. Añadir dependencias que el procesador exacto necesite después de comprobarlas. No instalar `typesafe-sdk`, `pydantic-ai`, `torchaudio` o motores de agentes sin una función concreta.

No reutilizar mínimos de versiones de 2024 para una arquitectura nueva. Resolver versiones que expongan las clases Gemma 4, fijar el entorno con `uv.lock` y validar en el Mac. Si sólo funciona un commit de una biblioteca, fijar ese commit y justificarlo. Revalidar al actualizar.

Comandos de arranque del entorno, una vez creado `pyproject.toml`:

```bash
uv venv --python 3.11
uv sync
uv run gso doctor --config configs/e2b_text.yaml
```

Los comandos `gso` de este documento son un contrato a implementar; el paquete entregado contiene documentación, no una CLI ya construida.

La descarga debe utilizar una instantánea del Hub con revisión fijada y reutilizar la caché. No cargar el modelo en GPU para volver a guardar todos los pesos. No usar `device_map="auto"` para ocultar una distribución de dispositivos no controlada durante entrenamiento. Credenciales, si son necesarias, fuera de Git. Verificar licencia y condiciones del checkpoint real; no heredar automáticamente condiciones de PaliGemma.

Guardar un manifiesto del entorno y SHA del código en cada ejecución. Ignorar en Git pesos, datasets privados, cachés, entornos y secretos; conservar configs, esquemas, manifiestos y reportes sin información sensible.

## 4. Arquitectura de decisión

### 4.1 Módulos y responsabilidades

| Módulo | Responsabilidad |
|---|---|
| `contracts` | Preguntas, respuestas, ejemplos y validaciones |
| `data` | Ingesta, validación, separación por grupos, collator y serialización |
| `models/backbone` | Carga y estados ocultos de Gemma; detalles propios de Transformers |
| `models/heads` | Proyecciones entrenables, sin HTTP ni lectura de ficheros |
| `training` | Bucle, pérdidas, acumulación, validación y checkpoints |
| `calibration` | Temperaturas y artefactos de calibración |
| `evaluation` | Métricas, baselines, robustez y perfilado |
| `inference` | Expansión de preguntas, batching, reconstrucción y políticas de abstención |
| `api` | Validación HTTP, lifecycle, límites y adaptación del contrato |
| `cli` | Comandos reproducibles que reutilizan los módulos anteriores |

Monolito modular con composición de dependencias sencilla. No introducir microservicios, Redis o una base de datos para este prototipo.

### 4.2 Backbone sin generación

Transformers documenta `Gemma4Model` como base multimodal sin cabeza de lenguaje. Su integración concreta, carga desde el checkpoint y salida deben verificarse en la versión fijada. Es preferible a borrar atributos de `Gemma4ForConditionalGeneration`. [S3]

No llamar al `forward` generativo y después descartar logits de todo el vocabulario. No pedir estados de todas las capas si sólo se usa la última. No modificar pesos enlazados ni borrar embeddings para ahorrar memoria sin comprender sus dependencias.

El adapter de backbone devuelve `last_hidden_state` y la máscara alineada. El adapter del procesador conserva todos los campos multimodales requeridos, no sólo `input_ids` y `pixel_values`. Los enteros siguen siendo enteros; los tensores flotantes van al dtype compatible. Dimensión oculta leída de la configuración, nunca fijada como constante universal.

Cada fila termina con una instrucción fija de evaluación. Extraer el último token válido de la secuencia procesada: `max(t donde attention_mask[t] == 1)`. No usar `[:, -1, :]` si hay padding a la derecha ni `sum(mask)-1` con padding a la izquierda. Rechazar filas vacías. Verificar alineación con imágenes y que dicho token ve toda la pregunta.

La serialización incluye estado, tipo, instrucciones, criterios y candidato cuando proceda. Nunca incorpora etiquetas, origen del dato, ID del ejemplo o ID de pregunta como pistas. Usar plantilla versionada con escape de delimitadores; no añadir tokens especiales entrenables en el MVP. La plantilla del procesador puede añadir tokens de chat, pero no se genera una respuesta textual ni pensamiento.

### 4.3 Noul

Para una pregunta binaria, una fila produce `h` y un logit:

\[
z=w_n^Th+b_n,\qquad p=\sigma(z/T_n),\quad T_n>0.
\]

Durante entrenamiento, `T_n=1`. Etiqueta observada `y ∈ {0,1}`. Salida pública `noul=p`. La decisión discreta, cuando se necesite, se calcula en código con un umbral definido en desarrollo. `2|p-0,5|` puede describir concentración binaria, pero no es la probabilidad de acierto.

### 4.4 Choice con opciones dinámicas

Un `Linear(D,255)[:, :K]` no garantiza que sus posiciones aprendan significados variables. Se elige un evaluador compartido por candidato:

\[
h_i=f_\theta(\operatorname{serialize}(estado,pregunta,c_i)),\quad
z_i=w_c^Th_i+b_c,\quad p_i=\operatorname{softmax}(z/T_c)_i.
\]

Cada fila contiene los mismos criterios completos y destaca un candidato. El ID opaco sirve para reconstruir la respuesta; la descripción define su significado. Para que el orden del mapa no cambie las entradas, ordenar canónicamente por descripción normalizada y, en empates, por ID. Se recomiendan descripciones distintas; probar explícitamente las ambigüedades. La etiqueta se remapea por ID después de cualquier permutación.

Las K filas se agrupan en lotes o microbatches. La softmax y la pérdida se calculan sobre el grupo completo. No entrenar candidatos como K problemas binarios independientes cuando la decisión es mutuamente excluyente. No normalizar cada fragmento de candidatos por separado.

El coste crece con K y repite el estado; incluir todos los criterios en cada fila también aumenta tokens. Esto sacrifica eficiencia para tener una primera implementación clara y auditable. No promete la arquitectura ni la latencia de Jev. Inicialmente `2 ≤ K ≤ 8`; ampliar sólo con datos y benchmark. No anunciar 255 opciones por tener un validador que las permita.

### 4.5 Score con rúbricas variables

Aplicar un evaluador compartido distinto al de Choice. Una fila por nivel incluye estado, pregunta, rúbrica completa ordenada e índice del nivel evaluado. La rúbrica no se ordena alfabéticamente: el orden es parte de su significado.

\[
z_m=w_s^Th_m+b_s,\quad p_m=\operatorname{softmax}(z/T_s)_m,
\quad s=\sum_{m=0}^{M-1}m p_m.
\]

MVP: entre 2 y 5 niveles; posible ampliación a 10 después de validar. Valores numéricos `0…M-1`. Un score normalizado opcional se calcula como `s/(M-1)` y debe llevar otro nombre. Rúbricas no equidistantes requerirían valores explícitos y otra versión de contrato.

Predecir una distribución categórica y calcular su media no impone por sí mismo estructura ordinal. Para explotar orden, usar la pérdida acumulativa descrita más abajo. Conservar la distribución: una media intermedia puede esconder masa en ambos extremos.

### 4.6 Varias preguntas y significado de una pasada

Cada fila del backbone usa un forward sin decodificación token a token. Una petición completa puede requerir varias llamadas al backbone. Las preguntas se procesan de forma aislada y se agrupan por longitud/presupuesto; no concatenar preguntas con atención causal compartida si se pretende independencia.

Registrar número de preguntas, filas expandidas, microbatches y forwards. No afirmar «todas las preguntas en una sola pasada» cuando no sea cierto. Una optimización futura puede compartir el prefijo o usar posiciones de lectura múltiples, pero requiere demostrar equivalencia, aislamiento y ahorro antes de sustituir la referencia.

## 5. Datos y supervisión

### 5.1 Esquema JSONL versión 1

Un ejemplo corresponde a una pregunta lógica. Campos comunes obligatorios: `schema_version`, `id`, `group_id`, `task_family`, `language`, `state`, `question`, `target`, `provenance`. Imagen opcional en `image_path`, relativa a la raíz del dataset.

```json
{"schema_version":1,"id":"n-001","group_id":"case-001","task_family":"refund","language":"es","state":"Solicito que me devuelvan el cobro duplicado.","image_path":null,"question":{"type":"noul","instructions":"¿El cliente solicita una devolución?"},"target":{"label":1},"provenance":{"source":"synthetic_rule","generator_version":"v1","label_method":"explicit_request"}}
```

```json
{"schema_version":1,"id":"c-001","group_id":"case-002","task_family":"ticket_routing","language":"es","state":"No resuelve el DNS del servicio.","image_path":null,"question":{"type":"choice","instructions":"Selecciona el subsistema que presenta el fallo descrito.","criteria":{"network":"Resolución DNS o conectividad","auth":"Credenciales o permisos","other":"Ninguna de las anteriores"}},"target":{"class_id":"network"},"provenance":{"source":"synthetic_rule","generator_version":"v1","label_method":"explicit_fault"}}
```

```json
{"schema_version":1,"id":"s-001","group_id":"case-003","task_family":"service_impact","language":"es","state":{"affected_users":12,"total_users":12},"image_path":null,"question":{"type":"score","instructions":"Evalúa alcance de la incidencia según usuarios afectados.","criteria":["Ningún usuario","Algunos usuarios, pero no todos","Todos los usuarios"]},"target":{"level_index":2},"provenance":{"source":"synthetic_rule","generator_version":"v1","label_method":"affected_ratio"}}
```

Para visión, un ejemplo debe apuntar a una imagen real decodificable, con hash en el manifiesto y etiqueta verificada. La ruta no se copia en `state` como sustituto de contenido visual. El agente creará pequeños gráficos con datos conocidos y preguntas verificables, separando estilos y semillas entre particiones. Evitar que título o nombre del archivo revele la clase.

Validar tipos estrictos, claves únicas, etiquetas compatibles con criterios, rutas dentro de la raíz y archivos existentes. Etiquetas blandas requieren un tipo explícito y distribución válida; no reutilizar campos duros para decimales.

### 5.2 Cómo construir un dataset útil

Los cuatro textos repetidos del original sólo pueden servir de fixture. Plan incremental propuesto: 60–120 ejemplos de smoke, piloto de unos 1.000 casos únicos y expansión orientativa a 5.000–20.000 según errores y coste. Estos volúmenes no garantizan generalización.

Crear escenarios base con reglas de etiquetado comprobables; variar vocabulario, negaciones, distractores, políticas, datos contradictorios y ausencia de evidencia. Generar también múltiples preguntas sobre el mismo estado con respuestas diferentes, para detectar que el modelo ignora las instrucciones. Equilibrar tipos y revisar distribución de clases.

Un LLM puede parafrasear o proponer casos, pero sus porcentajes no constituyen probabilidades verdaderas. El consenso entre profesores se etiqueta como señal de distilación, no como calibración real. Conservar modelo/proveedor, revisión cuando esté disponible, prompt, semilla, coste y estado de revisión. No activar proveedores de pago automáticamente.

Definir política de etiquetado antes de generar: por ejemplo, «la solicitud aparece explícita» es comprobable; «riesgo real de fraude» necesita resultados observados o expertos. Ante falta de etiqueta fiable, excluir de evaluación supervisada y conservar en un conjunto de robustez. Añadir `other` cuando las opciones no sean exhaustivas y una pregunta de suficiencia de evidencia cuando el dominio la necesite.

### 5.3 Separación y prevención de fugas

Cuatro particiones por grupos: 70% entrenamiento, 10% validación, 10% calibración, 10% test como punto de partida. Asignar grupos antes de producir paráfrasis, aumentos de imagen o variantes de pregunta. Todos los derivados de una fuente, plantilla base o entidad correlacionada permanecen juntos. No dividir candidatos de una pregunta entre particiones.

El manifiesto fija IDs, hashes de contenido, grupos, semilla y versión del generador. Comprobar duplicados exactos y similitud entre particiones. No incluir la etiqueta en prompts o metadatos accesibles al modelo. Reservar además familias/plantillas y estilos visuales no vistos para medir transferencia. Un buen resultado dentro de la misma plantilla no demuestra comprensión general.

Validación decide hiperparámetros y checkpoint; calibración ajusta temperaturas; test sólo mide el artefacto congelado. Si se cambia el diseño tras ver test, declararlo y reservar otro test. Un conjunto de 100 casos por tipo da incertidumbre alta: publicar recuentos e intervalos.

## 6. Pérdidas y calibración

### 6.1 Entrenamiento supervisado

Noul: `BCEWithLogitsLoss` sobre logits FP32. Choice: entropía cruzada de logits del grupo completo y etiqueta de clase. Score: comenzar con entropía cruzada y comparar una pérdida ordinal acumulativa mediante validación.

Brier binario para evaluación:

\[
BS_n=\frac1N\sum_b(p_b-y_b)^2.
\]

Brier multiclase, convención de suma sin dividir por K:

\[
BS_c=\frac1N\sum_b\sum_k(p_{bk}-y_{bk})^2.
\]

Reportar por cardinalidad cuando K varía. No comparar esta escala con implementaciones que dividen por K sin indicar conversión.

Para Score, con acumuladas `F_j=Σ_{m≤j}p_m`:

\[
RPS=\frac1{M-1}\sum_{j=0}^{M-2}(F_j-\mathbb1[y\le j])^2.
\]

Comparar `CE` y `CE + λ·RPS`, con λ elegido sólo en validación. Inicialmente λ=0 y un experimento λ=1 si los datos lo justifican. Promediar pérdidas por pregunta lógica; si se mezclan tipos, explicitar sus pesos. Normalizar la acumulación por el número real de preguntas, incluido el último grupo parcial.

Brier puede ser una pérdida alternativa, pero cambiarla no convierte el entrenamiento en RL. Tampoco añadir `−λH(p)` reproduce RLCD. Esa penalización modifica el óptimo de una regla propia y puede introducir subconfianza; queda fuera del baseline.

### 6.2 Temperatura posterior

Congelar pesos del mejor checkpoint. Ajustar `T=softplus(t)+ε` por primitiva sobre calibración minimizando BCE/NLL. Guardar identidad del checkpoint, split y temperaturas. No ajustar una temperatura por consulta. Si una primitiva carece de datos suficientes, usar T=1 y marcarla como no calibrada.

La temperatura es un método sencillo documentado de calibración posterior [S7]; no garantiza corregir sesgos por subgrupo ni cambios de distribución. Comparar resultados antes y después, incluida degradación. Umbrales de abstención se fijan en validación y se evalúan de nuevo con el pipeline calibrado; si se optimizan sobre calibración, subdividirla o usar un conjunto dedicado para evitar presentar ese mismo ajuste como evaluación independiente.

### 6.3 Concentración y métricas de calibración

Si se publica `confidence` en Choice/Score, en este proyecto significa exclusivamente:

\[
concentration=1-\frac{-\sum_i p_i\ln p_i}{\ln K},\qquad K\ge2.
\]

Usar la convención `0 log 0 = 0`, cálculos FP32 y clamp numérico final a [0,1]. Es una definición propia; la documentación consultada de TypeSafe no da una fórmula exacta que justifique equivalencia. [S6]

Para ECE de clasificación usar `c=max(p)` y acierto de `argmax`, nunca concentración entrópica. Para Noul informar ECE de probabilidad del evento: agrupar por p y comparar media de p con frecuencia de y=1. Puede añadirse ECE de clase elegida, claramente separado. Score: RPS, MAE y calibración de eventos acumulados, además de top-label ECE si se informa.

En ECE usar 15 bins fijos que cubran [0,1], incluyendo ambos extremos; bins vacíos aportan cero. Publicar reglas, recuentos y reliability diagram. ECE depende del binning y tamaño de muestra: no usar `ECE < 0,03` como garantía contractual ni única condición de éxito.

## 7. Entrenamiento por etapas

### A. Cabezal con backbone congelado

Congelar todos los parámetros base y mantener backbone en `eval()`. Los cabezales se entrenan en modo `train()`. Extraer representaciones con `no_grad()`; no usar indiscriminadamente tensores de `inference_mode()` en un cálculo que necesita guardarlos para backward del cabezal. Cachear h en disco por hash de checkpoint, prompt, procesador y entrada cuando ahorre tiempo. Invalidar caché si cambia cualquiera o se activa LoRA.

AdamW sólo recibe parámetros entrenables. Learning rate inicial de cabezales: 1e-3; weight decay 0,01; máximo inicial de 3 épocas, ajustado por validación. Semillas registradas. Son hiperparámetros iniciales, no óptimos conocidos.

### B. LoRA + cabezales

Inspeccionar `named_modules()` y seleccionar explícitamente módulos de atención del transformer textual; empezar por proyecciones Q/V disponibles, sin una regex que adapte accidentalmente visión. La lista exacta queda en el reporte. Verificar compatibilidad de PEFT con el modelo base y versión elegida. [S8]

Congelar visión, audio, proyectores, embeddings y pesos base en esta fase. Descongelar sólo LoRA y cabezales. No envolver el backbone en `no_grad()` durante LoRA. Si se usa gradient checkpointing con embeddings congelados, demostrar que LoRA recibe gradientes; aplicar la configuración compatible necesaria de la biblioteca.

Grupos AdamW: LoRA 1e-4 y cabezales 5e-4 como inicio; warmup 5%, decaimiento lineal, clip global 1,0. Evaluar gradientes finitos. Backward en FP16 requiere una estrategia de escalado compatible y verificada; no copiar `torch.cuda.amp` a MPS por analogía.

Con candidatos microbatched, conservar el grafo hasta la pérdida de grupo o usar una estrategia de recomputación probada; fragmentar forward no garantiza ahorrar todas las activaciones de backward. Empezar con K pequeño y checkpointing. No hacer `detach()` de logits para ahorrar memoria: rompería el aprendizaje del backbone.

### C. Visión

Reutilizar el contrato, procesador oficial y visión congelada. Comenzar con una imagen y su configuración mínima válida; registrar tokens/patches/resolución realmente usados. No imponer 448×448 por herencia de PaliGemma. Entrenar y evaluar ejemplos que exijan mirar la imagen. Comparar con imagen omitida e imagen intercambiada; si rinde igual, investigar atajos antes de afirmar capacidad visual.

### D. Checkpoints

Separar artefacto de despliegue y de reanudación. Despliegue: adaptadores, cabezales, configuración, revisión base, procesador o referencia exacta, plantilla, temperaturas, métricas y límites. Reanudación añade optimizador, scheduler, RNG, sampler, época y paso.

No duplicar pesos congelados en cada checkpoint. Preferir safetensors para pesos; restaurar estados de entrenamiento sólo desde artefactos propios. Guardado atómico y comprobar recarga en proceso nuevo con tolerancia declarada. No prometer reproducibilidad bit a bit entre dispositivos.

## 8. Evaluación y aceptación

### Baselines

1. Prior de clases aprendido sólo en entrenamiento; constante para Noul y distribución marginal para Choice/Score cuando corresponda.
2. Clasificador textual ligero en tareas de etiquetas fijas, dejando clara su limitación frente a opciones dinámicas.
3. Gemma congelado + cabezal, con idénticos splits y prompts de evaluación.
4. LoRA + cabezal, antes y después de calibrar.
5. Opcional: Gemma generativo con salida restringida como comparación separada, incluyendo errores de formato y latencia completa. No forma parte de la inferencia del producto.

### Reporte mínimo

| Área | Evidencia |
|---|---|
| Noul | NLL, Brier, ECE del evento, precision/recall/F1 con umbral fijado, prevalencia |
| Choice | Accuracy, macro-F1 en taxonomías comparables, NLL, Brier, ECE top-label por K |
| Score | MAE bruto/normalizado, RPS, NLL y calibración acumulativa |
| Robustez | Preguntas nuevas, permutaciones, renombrado de IDs, negaciones, OOD y distractores |
| Visión | Texto+imagen frente a imagen omitida/intercambiada; estilos nuevos |
| Recursos | Mediana/p95, preguntas/s, filas/s, tokens procesados, memoria y swap |
| Abstención | Cobertura y error de decisiones aceptadas con umbrales congelados |

No calcular macro-F1 global mezclando IDs de categorías que significan cosas diferentes entre preguntas. Agrupar por taxonomía o usar acierto/NLL por tarea. Bootstrap por grupos, no por paráfrasis dependientes; indicar semilla, repeticiones y tamaño. En binarios con una sola clase, marcar métricas no definidas, sin fabricar cero.

Benchmark de rendimiento: modelo ya cargado, calentamiento explícito, al menos 100 consultas medidas si es viable y misma batería para variantes; medir cold start aparte. Registrar número de preguntas, candidatos, tokens e imágenes. La latencia HTTP incluye cola y procesador; la del forward se informa por separado. Nunca comparar sólo generación contra un forward omitiendo preprocesado y expansión.

Aceptación funcional: esquema válido en todas las pruebas, probabilidades finitas y normalizadas, salidas que dependen de la entrada, pesos recargables, ausencia de fugas detectadas y todas las invariantes críticas cubiertas. La calidad debe superar baselines en la métrica primaria predeclarada por tarea sin degradaciones materiales ocultas; publicar incertidumbre. Las metas numéricas de negocio y SLA se fijan después del piloto, antes de test final.

## 9. API propia versión 1

`POST /v1/decide`. Identidad inicial del artefacto: `gemma-system-one-e2b-v0.1`, resuelta desde manifest. Nunca responder `jev-1.13.0` ni aceptar silenciosamente cualquier nombre de modelo.

```json
{
  "model":"gemma-system-one-e2b-v0.1",
  "state":"Se ha cobrado dos veces el pedido. Solicito devolución.",
  "questions":{
    "refund":{"type":"noul","instructions":"¿Se solicita devolución?"},
    "route":{"type":"choice","instructions":"¿Qué equipo debe atender la solicitud?","criteria":{"billing":"Cobros y devoluciones","technical":"Errores de funcionamiento"}},
    "impact":{"type":"score","instructions":"Evalúa el impacto económico según el estado.","criteria":["No se indica problema económico","Se indica un cobro incorrecto","Se indica pérdida económica que impide operar"]}
  }
}
```

El esquema de respuesta contiene `model`, `checkpoint_id`, `answers` y `usage`. `answers` conserva exactamente los IDs de pregunta. Campos por primitiva: Noul (`type`, `noul`); Choice (`type`, `choice`, `probabilities`, `confidence`); Score (`type`, `score`, `probabilities`, `legend`, `confidence`). Se añade `confidence_method="normalized_entropy_v1"` en metadata de respuesta. No incluir un ejemplo de probabilidades como si fuera una inferencia real.

`usage`: `questions`, `expanded_rows`, `backbone_forwards`, `processed_input_tokens` y `generated_tokens=0`. Los tokens procesados son la suma de tokens válidos de las filas realmente evaluadas, incluido el prefijo repetido; no el tamaño del estado una sola vez ni el JSON de salida. Registrar padding y tokens visuales por separado si el procesador permite contarlos de forma fiable.

### Validaciones y límites propios iniciales

- Pydantic con uniones discriminadas por `Literal` y campos extra prohibidos.
- `state`: string, objeto o array JSON no vacío, con tamaño y profundidad acotados; números finitos.
- Instrucciones: texto no vacío; los IDs de pregunta sólo se usan para correlación.
- Entre 1 y 8 preguntas; Choice 2–8 opciones con descripciones no vacías; Score 2–5 niveles. Son límites iniciales del proyecto, no límites atribuidos a Jev.
- Rechazar entradas que excedan el presupuesto tokenizado. No truncar silenciosamente criterios ni texto decisivo.
- Imagen opcional JPEG/PNG base64 estricta; máximo inicial 5 MiB decodificados y 16 megapíxeles, máximo de cuerpo HTTP 8 MiB. Comprobar límite del cuerpo antes de parsear y píxeles antes de descomprimir por completo. Sin URL remota ni ruta arbitraria del cliente. Rechazar formatos animados.
- Respuesta: probabilidades en [0,1] con suma ≈1; opción seleccionada perteneciente al conjunto; score igual a la esperanza y dentro del rango; tolerancias FP32 documentadas. Error explícito ante NaN/Inf.

Errores: 422 para esquema/rangos/contexto; 413 para tamaño excesivo; 400 para imagen corrupta; 404 para modelo desconocido; 503 para modelo no listo o cola saturada; 500 ante fallo interno sin datos sensibles. `GET /health/live` informa vida del proceso; `GET /health/ready` sólo da 200 con checkpoint cargado, validado y warmup exitoso.

Cargar modelo una vez en lifespan. Un worker de inferencia dedicado con cola acotada; el endpoint async espera su resultado y no ejecuta directamente el cómputo bloqueante. Un solo proceso Uvicorn al inicio: varios procesos duplicarían pesos. `eval()` e `inference_mode()` en inferencia; timeout no implica que una operación GPU ya lanzada sea cancelable. Controlar backlog, no reintentar sin límite.

Escuchar por defecto en `127.0.0.1:8000`, sin `--reload` para benchmarks. No levantar servidor y entrenamiento pesado simultáneos. Logs con request ID, duración y dimensiones; no estados o imágenes completos.

### Compatibilidad TypeSafe, opcional

La documentación oficial confirma el endpoint `/v1/systemone` y el SDK Python `typesafe-sdk`. [S9] Eso no valida automáticamente el servidor adjunto. Si se necesita compatibilidad, crear adapter separado, fijar versión del SDK, inspeccionar su transporte/base URL y probar peticiones y respuestas reales con ese cliente. Documentar campos soportados, errores, límites y diferencias semánticas. No depender de `TYPESAFE_BASE_URL` sin verificar que la versión instalada la usa. No anunciar equivalencia de confianza, multimodalidad, latencia o calibración.

## 10. Fases de implementación y puertas de salida

| Fase | Trabajo | Criterio para avanzar |
|---|---|---|
| 0 | Paquete, lock, doctor, descarga e inspección de backbone | Forward/backward real y reporte del Mac; o bloqueo reproducible |
| 1 | Contratos, plantilla, dataset, split, baselines y Noul | Sobreajuste controlado de un fixture y baseline medido sin fugas |
| 2 | Choice/Score dinámicos, grupos de candidatos, métricas | Tests de semántica, pérdida de grupo y orden; piloto reproducible |
| 3 | LoRA, checkpoints, calibración y evaluación ciega | Comparación con cabezales congelados y recarga equivalente |
| 4 | Una imagen y benchmark visual | Evidencia de uso visual y límites de memoria reales |
| 5 | API, cola, límites y cliente HTTP | E2E con checkpoint entrenado, fallos claros y benchmark completo |
| 6 | E4B, más opciones y optimización opcional | Ganancia medida que justifique coste y complejidad |

En fase 1 basta con demostrar que la ruta aprende; sobreajustar 16–32 ejemplos es una prueba de implementación, no un resultado de calidad. En fase 2 no pasar directamente a 20.000 muestras: inspeccionar primero 50–100 predicciones y errores del piloto.

## 11. Estructura a implementar

```text
README.md
AGENTS.md
CLAUDE.md
PROMPTS.md
pyproject.toml
uv.lock
configs/
  e2b_text.yaml
  e2b_vision.yaml
  e4b_experiment.yaml
docs/
  ESPECIFICACION.md
  AUDITORIA.md
  FUENTES.md
  STATUS.md
  decisions/
src/gemma_system_one/
  contracts/
  data/
  models/
  training/
  calibration/
  evaluation/
  inference/
  api/
  cli.py
tests/
  unit/
  integration/
  mps/
  e2e/
scripts/
reports/
artifacts/
data/
```

El agente puede usar archivos sencillos en lugar de subpaquetes vacíos mientras el código sea pequeño. Los directorios de datos/artefactos no se versionan con sus contenidos grandes.

### Comandos que deberá ofrecer el proyecto

| Comando previsto | Resultado |
|---|---|
| `gso doctor --config …` | Compatibilidad y recursos |
| `gso download --config …` | Snapshot fijado y manifiesto |
| `gso validate-data --dataset …` | Errores de esquema, archivos y grupos |
| `gso split --dataset … --seed …` | Manifiestos inmutables de cuatro splits |
| `gso train --config …` | Checkpoint y registro de entrenamiento |
| `gso calibrate --checkpoint … --split …` | Artefacto de temperaturas vinculado |
| `gso evaluate --checkpoint … --split …` | Métricas y predicciones auditables |
| `gso benchmark --checkpoint … --suite …` | Latencia y memoria |
| `gso serve --checkpoint …` | API local con modelo real |

No considerar implementado un comando porque imprima un mensaje o retorne valores constantes. El objetivo final es una trayectoria completa de datos a entrenamiento, recarga y servicio.

## 12. Pruebas mínimas de alto valor

Unitarias CPU: contratos inválidos, K/M mínimo y máximo, probabilidades y score, fórmula de Brier/RPS con casos calculables, bins extremos de ECE, pooling con ambos paddings, remapeo de candidatos, split por grupos y rechazo de rutas externas.

Integración: serialización idéntica en train/serve, collator heterogéneo, pérdida de pregunta igual con y sin microbatch dentro de tolerancia, normalización del último paso de acumulación, identidad de parámetros congelados, LoRA con gradientes reales y checkpoint recargado.

MPS real: forward/backward textual, estabilidad del dtype, paso de optimizador, repetición visual y benchmark sincronizado. Marcar pruebas por hardware y no fingir que pasaron si se omitieron.

E2E: consulta real contra checkpoint entrenado; cambio de pregunta con mismo estado; permutar opciones; añadir una pregunta independiente; imagen requerida; saturación/timeout; modelo ausente. Las invariantes estructurales deben cumplirse; las pruebas de semántica se evalúan sobre conjuntos, no garantizando que cualquier cambio textual modifique el resultado.

No se exige cobertura global arbitraria. Se exige evidencia sobre estas rutas críticas y ausencia de mocks en las comprobaciones que afirman inferencia real.
