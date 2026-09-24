# 0008 — Estilos visuales separados por partición y copia automática de las fuentes

Fecha: 2026-09-23 · Cierre de fase 4 · Estado: aceptada

## Contexto

La revisión independiente de la fase 4 dejó dos pendientes de conformidad:

1. **Estilos compartidos.** En `support-vision-v1` el estilo de cada caso se elige antes del split, así que los estilos 0–3 aparecen en train, validation, calibration y test. La spec §5.1 pide separar estilos y semillas entre particiones. El test v1 mide uso visual con estilos vistos, no con estilos inéditos.
2. **Fuentes no conservadas.** Los checkpoints V/C registran el hash de fuentes `70068e15…`, pero nadie guardó una copia. El hash identifica el código, pero no permite reconstruirlo.

## Decisión

1. **Generador `support-vision-v2`** (v1 se conserva byte a byte: la misma secuencia aleatoria reproduce los sha256 del piloto histórico).
   - Los `group_id` se fijan primero y el reparto se calcula con la misma `assign_groups` que usa `gso split` y la semilla indicada (`--split-seed`).
   - Cada caso se muestrea con el flujo aleatorio de su partición y con un estilo de su repertorio, disjunto de los demás: train 0, 1, 2, 3, 5, 6; validation 7, 8; calibration 9, 10; test 11, 12; transfer 4.
   - Los estilos nuevos varían fondo, colores, orientación, rejilla, tamaño de fuente, ancho de barra y trazo del umbral. v1 no los usa.
   - El manifiesto registra `style_pools` y `planned_split.assignment_sha256`, y `audit.jsonl` registra la partición planificada de cada caso.
   - `gso split` rechaza un reparto que no coincida con el planificado.
2. **El test v1 no se reutiliza ni se modifica.** El cierre se mide con un test nuevo, predeclarado en `reports/phase4b-test-protocol.md` antes de generar los datos.
3. **Copia automática de fuentes.**
   - `env.snapshot_sources()` lee cada fichero una sola vez: hash y copia salen de los mismos bytes.
   - Escribe `artifacts/source/<sha256>.tar`, determinista y atómico, sin sobrescribir.
   - `git_state(snapshot=True)` se usa en `environment_manifest()` (inicio de cada run) y en `_save_state` (cada checkpoint), de modo que cada hash registrado tiene su copia.
   - Los tests redirigen la copia con `GSO_SOURCE_ARCHIVE_DIR`.
4. **Límite histórico que se mantiene:** no existe la copia de `70068e15…` (búsqueda en el repo y en el directorio temporal de la sesión). No se fabrica ni se renombra otra copia.

## Consecuencias

- Validación (estilos 7–8) y test (11–12) miden transferencia a estilos no vistos dentro de la misma familia de gráficos.
- La selección de época se hace con estilos no vistos en train, lo que puede elegir una época distinta a la del piloto v1.
- Cada inicio de run y cada checkpoint añaden ~0,5 MB por versión distinta del código a `artifacts/source/`, un directorio ignorado por Git.
