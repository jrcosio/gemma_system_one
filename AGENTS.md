# Instrucciones comunes para agentes de desarrollo

Este repositorio construye un evaluador local basado en Gemma 4 para un Mac M5 Pro con 48 GB. Lee `README.md`, `docs/ESPECIFICACION.md`, `docs/AUDITORIA.md` y `docs/STATUS.md` si existe antes de cambiar código.

## Objetivo

Adaptar un checkpoint preentrenado mediante cabezales y posteriormente LoRA. Inferencia tipada Noul/Choice/Score sin generación de texto. No preentrenar Gemma, no sustituir el proyecto por llamadas a Jev y no llamar RLCD a una pérdida supervisada.

## Reglas técnicas

1. Empezar por fase 0 y Gemma 4 E2B. E4B es un experimento posterior. Fijar revisión del checkpoint y dependencias verificadas.
2. PyTorch/MPS es la ruta principal. No introducir CUDA/bitsandbytes/QLoRA ni MLX como reemplazo automático sin comprobar soporte y documentar la decisión.
3. Nunca inventar clases de biblioteca, versiones, compatibilidad de SDK, métricas, tiempos o disponibilidad de hardware. Inspeccionar el código y las fuentes primarias actuales cuando haga falta.
4. Contratos compartidos entre dataset, entrenamiento e inferencia. Las preguntas y criterios entran en el modelo; etiquetas, IDs opacos y metadatos de procedencia no.
5. Choice y Score usan candidatos dinámicos con un scorer compartido. Normalizar sobre el grupo completo y acumular pérdidas por pregunta lógica.
6. Mantener splits de entrenamiento, validación, calibración y test sin fugas por grupos. No usar test para elegir hiperparámetros.
7. Cargar base sin cabeza de lenguaje; no borrar `lm_head` a ciegas ni llamar `generate()` en producción. Respetar campos del procesador y máscaras.
8. Verificar parámetros entrenables y gradientes. Base congelada en eval; LoRA necesita autograd; inferencia usa eval e inference_mode.
9. No devolver constantes o simulaciones en el servidor real. Los dobles viven en tests. Si falta el modelo, fallar claramente.
10. No afirmar compatibilidad Jev, calibración universal o ausencia de errores semánticos. `confidence` describe concentración según nuestra fórmula.
11. Comandos y resultados reproducibles, checkpoint recargable, logs sin secretos ni contenido sensible. No subir datasets, pesos o credenciales a Git.
12. No ejecutar descargas repetidas ni dos entrenamientos pesados a la vez. No desactivar protecciones de memoria como solución rutinaria.

## Forma de trabajar

Implementa una fase completa y verificable cada vez. Antes de entrenamientos grandes, ejecuta fixtures pequeños y prueba real del backend. Si no hay Mac/MPS, avanza contratos y tests CPU, dejando la prueba MPS como pendiente explícita. No presentes simulaciones como validación del hardware.

Mantén `docs/STATUS.md`: fase actual, commit, comandos ejecutados, resultados reales, archivos cambiados, bloqueos y siguiente paso. Registra cambios de arquitectura en `docs/decisions/` con motivo y evidencia. Las instrucciones del usuario prevalecen sobre este plan; explicita sus consecuencias técnicas.

Revisión: buscar errores funcionales y de metodología, no sólo estilo. Pruebas prioritarias: gradientes, candidatos/padding, separación de datos, calibración, recarga, servidor real y límites de recursos. No imponer cobertura arbitraria ni tests que sólo repitan un constructor.

Al entregar: qué cambió, por qué, cómo se probó, qué queda pendiente. Mostrar diff unificado cuando ayude a revisar. No afirmar que un test pasó si fue omitido por falta de hardware o pesos.

## Uso de Claude Code y Codex

Ambos trabajan bajo estas instrucciones. Se recomienda alternar implementación y revisión por fase; los papeles pueden invertirse. Si se usan simultáneamente, cada uno trabaja en una rama/worktree y módulos acordados, con un solo entrenamiento en el Mac. No sobrescribir trabajo ajeno ni modificar contrato y consumidor de forma incompatible. El revisor debe comprobar evidencia, no dar por buenas las conclusiones del implementador.
