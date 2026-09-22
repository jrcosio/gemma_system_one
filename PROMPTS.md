# Prompts para iniciar y continuar el proyecto

Copiar los archivos del paquete a la raíz de una carpeta nueva. Elegir Fable 5.1 en Claude Code o GPT Astra en Codex mediante los controles disponibles en cada herramienta; estos prompts no dependen de flags ni identificadores de modelo supuestos.

## 1. Arranque con cualquiera de los dos asistentes

```text
Vamos a construir este proyecto desde cero para mi Mac M5 Pro con 48 GB de memoria unificada. Lee README.md, AGENTS.md y todos los documentos de docs antes de modificar nada.

El objetivo final es adaptar Gemma 4 para responder Noul, Choice y Score mediante cabezales aprendidos, sin generar texto durante la inferencia. Quiero un entrenamiento y una API reales. La documentación explica el alcance y los errores que debemos evitar.

Empieza implementando la fase 0 completa: paquete Python instalable con uv, configuración tipada, versiones verificadas, descarga reproducible del checkpoint Gemma 4 E2B y comando doctor. Inspecciona las clases y la ruta de carga en la versión que instales. Comprueba extracción de representación, forward/backward, parámetros entrenables, memoria y guardado/recarga mínima en este Mac. No descargues dos veces los mismos pesos.

No construyas todavía toda la API ni lances un entrenamiento grande. Crea tests CPU de los contratos que implementes y una prueba pequeña real en MPS. Si no estás ejecutándote en el Mac, completa lo verificable y deja comandos concretos para la comprobación local, sin darla por realizada.

Resuelve las decisiones rutinarias con la especificación. Si aparece una incompatibilidad, investiga y registra el error reproducible antes de proponer cambios de arquitectura. Termina con el código, los tests que hayas podido ejecutar, reports/compatibility.md y docs/STATUS.md. Distingue evidencia de supuestos y deja preparada la siguiente fase.
```

## 2. Revisión independiente del otro asistente

```text
Revisa la fase implementada de este repositorio tomando AGENTS.md y docs/ESPECIFICACION.md como referencia. Lee docs/STATUS.md y revisa el diff y el código real; no te limites al resumen del asistente anterior.

Busca fallos que impidan ejecutar el proyecto o invaliden sus resultados: carga real de Gemma 4, dispositivo y dtype, extracción de estados, máscaras, gradientes, parámetros congelados, prompts con preguntas/criterios, fuga de etiquetas, normalización por grupo, memoria, checkpoints y afirmaciones no respaldadas.

Reproduce los problemas con pruebas pequeñas cuando sea posible. Corrige errores claros y reversibles manteniendo el alcance de la fase. No cambies el backend, el modelo o el contrato público sin justificarlo en una decisión técnica. No lances entrenamientos largos ni servicios externos para revisar.

Entrega hallazgos por gravedad, correcciones realizadas, evidencia de pruebas y riesgos pendientes. Actualiza docs/STATUS.md. No apruebes la fase si sólo hay mocks donde se requiere modelo real o si se ha presentado una prueba CPU como prueba MPS.
```

## 3. Continuación de una fase

```text
Continúa a partir de docs/STATUS.md y de la evidencia existente. Lee AGENTS.md y la fase siguiente de docs/ESPECIFICACION.md.

Completa esa fase de principio a fin: implementación, pruebas relevantes, ejecución pequeña, reporte y actualización del estado. Reutiliza los contratos y la serialización compartidos. Antes de ampliar datos o entrenamiento, resuelve los fallos del piloto. Mantén separados validación, calibración y test.

No declares una fase terminada si su criterio de salida está pendiente. Si existe un bloqueo externo, deja el trabajo restante concreto y reproducible; continúa las partes independientes que sí puedas completar. Explica qué se ha ejecutado realmente y cuál es el siguiente paso.
```

## 4. Relevo entre herramientas

```text
Prepara el relevo para el otro asistente sin cambiar el alcance del proyecto. Actualiza docs/STATUS.md con fase, rama/commit, archivos modificados, decisiones, comandos exactos ejecutados, resultados, fallos reproducibles y tareas pendientes. Incluye qué procesos siguen activos y dónde están los checkpoints. No incluyas secretos ni datos privados.
```

## Orden sugerido

Claude Code puede implementar fase 0 y Codex revisarla; después intercambiar papeles si conviene. Ambos deben revisar la metodología de datos/calibración antes del primer entrenamiento relevante. En el Mac, mantener un único proceso pesado activo. La revisión externa es útil para detectar sesgos; no exige dos agentes editando a la vez.
