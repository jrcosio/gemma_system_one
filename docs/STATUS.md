# Estado vigente — fase 6e cerrada: tríos con la misma pregunta (2026-09-24, 17:36 UTC)

Sustituye a las secciones siguientes (relevo y revisión de la fase 6d), que quedan como registro histórico. Resuelve su pendiente principal: un control semánticamente coherente del uso del estado, que sustituye al intercambio de estados, y un diagnóstico de K con más potencia. Incluye, sin cambios, las correcciones de esa revisión, que siguen sin commit:
- `src/gemma_system_one/data/derive.py` (docstrings);
- `scripts/probe_option_cue.py`, `scripts/derive_phase6d_data.py` y el nuevo `scripts/review_phase6d_controls.py`;
- `README.md`, `reports/phase6d-final.md` y `docs/decisions/0012-…md`.

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **6e cerrada.** No hay fases posteriores en la spec §10; es la corrección de fallos del piloto detectados en revisión. Informe: **`reports/phase6e-final.md`**. Protocolo `reports/phase6e-protocol.md`, sha256 `d33ede5a…` (en `reports/phase6e/protocol.sha256`), escrito antes de generar los datos y de evaluar |
| Diseño | `derive.fault_kind_triplets`: cada trío comparte exactamente la misma pregunta (`none`, `other` y 2 categorías reales; mismos textos, IDs y orden) y tiene tres estados de grupos distintos con respuestas real / `other` / `none`, derivadas de sus propios hechos. Sin leer el estado, el techo es 1/3 de accuracy y 0 tríos completos. K8 = K4 + las 4 distractoras de la 6d |
| Uso del estado | **Los tres modelos lo usan.** Tríos completos K4: A4v3 **0,58 [0,51; 0,66]**, A4v4 0,50 [0,42; 0,58], A2v4 0,21 [0,14; 0,27] (criterio: límite inferior > 0,10). Accuracy K4: 0,847 / 0,822 / 0,684 |
| Más opciones | Δaccuracy K8 − K4 (450 preguntas): A4v3 −0,024 [−0,060; +0,011] y A4v4 −0,027 [−0,051; −0,002], **no demostrada**; A2v4 +0,013 [−0,024; +0,053], **tolera**. El bootstrap por trío da casi lo mismo |
| Hallazgo | **La debilidad de `other` es sobre todo una ambigüedad de las categorías.** Con E4B, el error dominante es elegir «aplicación» para fallos no listados de rendimiento o de datos (A4v3 K8: 31 de 55 errores; `other` con rendimiento no listado: 15/30). Queda pendiente un generador v5 con definiciones excluyentes |
| Servicio | Sin cambios: A4v3 (`configs/serve_e4b_text.yaml`) |
| Rama / commit | `main` = `fases-0-6` = **`9edc183`**, en local y en `origin`. **Sin commit:** la revisión de la 6d y esta fase (§4). No se ha hecho push de nada nuevo |
| Código | Evaluaciones con `c1ee7841…`; el actual es **`a1e3f481609c6f830f88defd25cd5712f62012a20dcfbd602b2d80a4bb17f0a4`** (86 ficheros; sólo añade `scripts/analyze_triplets.py`). Ambos con copia en `artifacts/source/` |
| Pruebas | **294 CPU** (34,02 s) y **14 MPS/E2E reales** (130,74 s, 0 omitidos). Ruff y formato en verde (147 ficheros); `uv lock --check` y `git diff --check` correctos |

## 2. Procesos activos

`pgrep -fl 'gso|pytest|uvicorn'` terminó con código 1 y `lsof -nP -iTCP:8000 -sTCP:LISTEN` también: **ningún proceso del proyecto** y el puerto libre. En esta fase no se entrenó nada.

## 3. Checkpoints

Sin cambios: la tabla de la sección histórica siguiente (A4v3 servicio, A4v4, A2v4, B2, A2 y V2/C2, con sus calibraciones) sigue vigente.

**Datos nuevos**, ya usados: `data/pilot_v4_trip15{,_K4,_K8}`.

**Conjuntos que no deben servir para decidir:**
- el test de `pilot_v3`, los tests de `vision_pilot_v1/v2` y `holdout6*`;
- `final8`, `final13`, `kdiag11*`, `kdiag14*`, `final13_faultswap` y `trip15*`.

El test y la partición de calibración de `pilot_v4` siguen sin leer.

## 4. Archivos sin commit sobre `9edc183`

**De la revisión de la 6d:** los de la introducción de esta sección.

**Nuevos de esta fase:**
- `tests/unit/test_phase6e_triplets.py`;
- `scripts/derive_phase6e_data.py`, `scripts/analyze_triplets.py`;
- `reports/phase6e-protocol.md`, `reports/phase6e-final.md` y `reports/phase6e/` (logs, `pred_*.txt`, `triplets.json`, comparaciones, `sensitivity_and_errors.txt`, `run_chain.sh`).

**Modificados en esta fase:**
- `src/gemma_system_one/data/derive.py`: `TRIPLET_ROLES`, `fault_kind_triplets`;
- `README.md` y `docs/decisions/0012-…md` (anexo de la 6e);
- este STATUS.

## 5. Decisiones

- **0012, con anexo de la 6e:** el control con tríos sustituye al intercambio de estados, que queda retirado.
- Sin decisiones de arquitectura nuevas; 0001–0012 siguen vigentes.

## 6. Comandos exactos

Lista completa en `reports/phase6e-final.md` («Comandos ejecutados»):
- test de los tríos, protocolo y su sha256;
- `derive_phase6e_data.py` y `validate-data` (3 conjuntos);
- `reports/phase6e/run_chain.sh` (6 evaluaciones);
- `analyze_triplets.py` y 3 `gso compare`;
- `pytest` CPU y MPS/E2E;
- ruff, formato y lock.

## 7. Fallos reproducibles

- **Abierto, de definición de la tarea:** la opción «aplicación» («Error en una pantalla o función de la aplicación») absorbe estados de rendimiento y de datos redactados como fallos de la app (p. ej., «la aplicación va extremadamente lenta…»). Se reproduce con `reports/phase6e/sensitivity_and_errors.txt` (tabla por categoría real no listada).
- **Abierto:** la composición de opciones del generador de entrenamiento v4 (fase 6d) sigue pendiente.
- **Sin fallos de código nuevos.**

## 8. Pendientes, en orden

1. **Commit y push** de la revisión de la 6d y de esta fase, a decidir por el usuario.
2. **Revisión independiente de la 6e:**
   - `fault_kind_triplets` (pools por idioma y categoría, un estado por grupo);
   - el análisis por trío;
   - el diagnóstico de la ambigüedad de «aplicación».
3. **Generador v5**, con protocolo y datos nuevos antes de entrenar:
   - definiciones de categorías mutuamente excluyentes, por ejemplo que «aplicación» excluya la lentitud y la pérdida de datos, o una cláusula explícita en la instrucción;
   - composición de opciones equilibrada.

   Después, reentrenar los cabezales, evaluar con tríos y un test nuevo, y mantener A4v3 como servicio mientras no se supere su regla de selección.
4. **Clon limpio** con `uv sync`, sin probar.
5. **Opcionales:** calibración de Choice, LoRA sobre E4B, E4B con imagen y abstención.
6. **Riesgos que se mantienen:**
   - el timeout no interrumpe un forward MPS y la cola no limita las conexiones;
   - el RSS muestreado no es el pico de MPS;
   - datos sintéticos de una familia;
   - V2 sin calibrar.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Rutas:** las de `reports/phase6e/` están enmascaradas con `~`.

---

# Registro histórico — relevo tras revisión independiente de fase 6d (2026-09-24, 17:05 UTC)

**Verificación de este relevo (17:05 UTC).** `main`, `fases-0-6`, `origin/main` y `origin/fases-0-6` apuntan localmente a `9edc18340ab13fce5d9fe68ce7c7a2b28ff7ea1e`; no se hizo fetch ni push en este relevo. El árbol conserva siete archivos modificados (`README.md`, `docs/STATUS.md`, `docs/decisions/0012-generator-v4-without-k-cue.md`, `reports/phase6d-final.md`, `scripts/derive_phase6d_data.py`, `scripts/probe_option_cue.py`, `src/gemma_system_one/data/derive.py`) y uno nuevo sin seguimiento (`scripts/review_phase6d_controls.py`); stash vacío. No se ha hecho commit de la revisión. `git diff --stat` no cuenta el script sin seguimiento. El hash actual de fuentes sigue siendo `30cb81640a82f5d726db4eda6971ccdec61f284c88e69927e251eaf92ec540e3` (84 archivos); el hash de las evaluaciones 6d sigue siendo el histórico `fd340105…`.

En esta verificación se volvió a ejecutar el script de control: K4 y K8 tienen 125/167 etiquetas contradictorias con el estado donante; `final13_faultswap`, 99/138; la información mutua empírica K4 es 0,4223 bits. El hash del protocolo y `git diff --check` pasan. **No se repitieron** los 293 tests CPU, el test LoRA/E2B real en MPS ni la recarga E4B/MPS: sus resultados de la revisión anterior constan abajo. Los siete manifiestos de checkpoint de la tabla inferior existen; esta verificación sólo comprueba su presencia. El proceso visible que coincide con el filtro es un `caffeinate -i` auxiliar; no hay proceso GSO, pytest o uvicorn ni listener en TCP 8000. No se inició entrenamiento, descarga, evaluación de modelo ni servicio.

Comandos exactos de comprobación de este relevo, ejecutados desde la raíz:

```sh
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git stash list
git rev-parse refs/heads/fases-0-6 refs/remotes/origin/main refs/remotes/origin/fases-0-6
git diff --check
.venv/bin/python -c 'from gemma_system_one.env import source_fingerprint; x=source_fingerprint(); print(x["sha256"], x["files"])'
shasum -a 256 -c reports/phase6d/protocol.sha256
.venv/bin/python scripts/review_phase6d_controls.py
ps -axo pid,ppid,stat,etime,command | rg '[g]so|[g]emma_system_one|[p]ytest|[u]vicorn|[c]affeinate|[d]erive_phase6d|[p]robe_option_cue' | head -40
lsof -nP -iTCP:8000 -sTCP:LISTEN
.venv/bin/python - <<'PY'
from pathlib import Path
runs={
'A4v3':'runs/e4b_experiment/20260923T204945Z',
'A4v4':'runs/e4b_v4/20260924T140932Z',
'A2v4':'runs/e2b_heads_v4/20260924T142144Z',
'B2':'runs/pilot_lora_v3/20260923T012208Z',
'A2':'runs/pilot_ce_v3/20260923T005721Z',
'V2':'runs/vision2_heads/20260923T161024Z',
'C2':'runs/vision2_text_only/20260923T162937Z'}
for name,run in runs.items():
 p=Path(run)
 print(name, str(p/'checkpoint'), 'manifest=', (p/'checkpoint/manifest.json').is_file(), 'calibrations=', [q.name for q in sorted((p/'calibration').glob('*.json'))])
PY
```

La salida de `lsof` fue vacía (código 1). Todas las comprobaciones de manifiesto imprimieron `True`. Las rutas de calibración de la tabla siguiente son relativas al directorio de cada run. El relevo sólo modifica este documento y conserva las correcciones de revisión anteriores sin commit.

## Relevo actual

**Fase y dictamen.** Fase 6d implementada en el commit `9edc18340ab13fce5d9fe68ce7c7a2b28ff7ea1e`, rama `main` (también `fases-0-6` en el último relevo publicado). Esta revisión corrige dos conclusiones metodológicas del informe. **No aprueba** la afirmación de ausencia de pistas en las opciones ni el control «sin estado». La comparación emparejada K8 − K4 sí mide la adición de las mismas cuatro distractoras a cada K4, y **no demuestra tolerancia** según el margen predeclarado (límite inferior de Δaccuracy < −0,05 para los tres modelos). No se cambian pesos, datos, backend, modelo, contrato público ni servicio A4v3. No se inició entrenamiento, descarga ni servidor.

**Hallazgos reproducibles, por gravedad.**

| Gravedad | Hallazgo y evidencia | Estado |
|---|---|---|
| Alta, validez del control | `swap_states` conserva la etiqueta original con estado donante. En K4/K8swap, 125/167 etiquetas contradicen la respuesta semántica del donante y 92/167 cambian de idioma; en `final13_faultswap`, 99/138 y 70/138. A4v3 en K4swap: accuracy 0,210 frente a etiqueta original y 0,814 frente a respuesta del donante. La accuracy baja no demuestra ausencia de pistas. | Interpretación retirada en informe y decisión; datasets y logits históricos intactos. |
| Media, afirmación estadística | `composition_gain = 0,000` es una diferencia de accuracy top-1 **dentro de la misma muestra**. El predictor `prior` ya filtra por opciones presentes. Una etiqueta real requiere estar presente en el par; información mutua empírica firma/etiqueta K4 = 0,4223 bits (167 casos, estimador con sesgo). No equivale a información cero ni toda esa dependencia es fuga indebida. | Docstrings, informe y decisión corregidos. |
| Abierto, potencia | IC de Δaccuracy K8 − K4: A4v3 [−0,096; +0,024], A4v4 [−0,078; +0,030], A2v4 [−0,072; +0,054]. | Tolerancia a K8 no demostrada; sólo 167 preguntas, 58 `other`. |

**Correcciones de esta revisión, sin commit:** `src/gemma_system_one/data/derive.py` y `scripts/probe_option_cue.py` (descripciones precisas); `scripts/derive_phase6d_data.py` (nota de futuros manifiestos); nuevo `scripts/review_phase6d_controls.py` (diagnóstico de sólo lectura y reproducible); `README.md`, `reports/phase6d-final.md` y `docs/decisions/0012-generator-v4-without-k-cue.md` (retiro de las conclusiones inválidas); este `docs/STATUS.md`. La decisión 0012 sigue vigente para el generador v4 y A4v3 como servicio; sólo se corrige su anexo 6d. No hay nueva decisión de arquitectura. Había una edición previa de relevo en `docs/STATUS.md` al empezar esta revisión; se conserva como registro histórico a continuación.

**Evidencia de esta revisión.** `.venv/bin/python scripts/review_phase6d_controls.py` reproduce los conflictos, la información mutua y la accuracy de A4v3/A4v4/A2v4 frente a la respuesta semántica del donante. `.venv/bin/pytest tests/unit tests/integration -q` → **293 passed, 2 warnings, 34,56 s**. `.venv/bin/pytest tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs` → **1 passed, 9,46 s**, sin omisión, con E2B real, MPS y gradientes LoRA. Se recargó **A4v3/E4B real** desde checkpoint y se evaluó una pregunta K4 sin caché: `Gemma4Model`, `mps`, BF16, eval, cero parámetros base entrenables, sin `lm_head`, cuatro logits finitos, diferencia máxima **0,0** con el JSONL archivado; cuatro forwards y 971 tokens válidos. Es inferencia E4B real, no una nueva prueba de gradientes E4B. Los 14 tests MPS/E2E de la implementación son evidencia histórica y no se repitieron en esta revisión. El protocolo conserva su hash (`shasum … -c`: OK). Ruff, formato, lock y `git diff --check`: OK.

**Identidad del código.** El hash de fuentes usado en las evaluaciones 6d fue `fd340105be1c49de8e601bd57561318204d0058c3a67e756c9101a7940e378cc` (83 archivos, copia en `artifacts/source/`). Tras las correcciones de revisión y el script nuevo, `source_fingerprint()` da `30cb81640a82f5d726db4eda6971ccdec61f284c88e69927e251eaf92ec540e3` (84 archivos). Esta huella nueva **no** es la de las evaluaciones históricas; los cambios en rutas de fuentes son aclaraciones y el script de revisión, sin cambio en la inferencia.

**Procesos y checkpoints.** `ps -axo pid,ppid,stat,etime,command | rg '[g]so|[p]ytest|[u]vicorn|[c]affeinate'` mostró sólo un `caffeinate -i` ajeno al proyecto; no hay entrenamiento, evaluación, test o servicio GSO activo. `lsof -nP -iTCP:8000 -sTCP:LISTEN` no mostró listener. Existen los manifiestos en:

| Modelo | Checkpoint | Calibración relevante |
|---|---|---|
| A4v3, servicio | `runs/e4b_experiment/20260923T204945Z/checkpoint` | `calibration/calibration-20260924T045005Z.json`; comparación 6c: `…143729Z.json` |
| A4v4 | `runs/e4b_v4/20260924T140932Z/checkpoint` | `calibration/calibration-20260924T143416Z.json` |
| A2v4 | `runs/e2b_heads_v4/20260924T142144Z/checkpoint` | `calibration/calibration-20260924T143725Z.json` |
| B2 | `runs/pilot_lora_v3/20260923T012208Z/checkpoint` | `calibration/calibration-20260924T044408Z.json` |
| A2 | `runs/pilot_ce_v3/20260923T005721Z/checkpoint` | `calibration/calibration-20260924T044041Z.json` |
| V2 / C2 | `runs/vision2_heads/20260923T161024Z/checkpoint` / `runs/vision2_text_only/20260923T162937Z/checkpoint` | Sin calibración local |

Todos son locales e ignorados por Git. La prueba de esta revisión **sí recargó A4v3**; de los otros checkpoints sólo se verificó la existencia del manifiesto, salvo el test E2B real que cargó la base E2B local. No se tocaron los artefactos ni las particiones de test/calibración.

**Comandos exactos de comprobación desde la raíz** (sin entrenamientos largos):

```sh
git log -1 --format='%H %s'
git status --short
.venv/bin/python scripts/review_phase6d_controls.py
.venv/bin/pytest tests/unit tests/integration -q
.venv/bin/pytest tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs
shasum -a 256 -c reports/phase6d/protocol.sha256
.venv/bin/python -c 'from gemma_system_one.env import source_fingerprint; print(source_fingerprint())'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv lock --check
git diff --check
ps -axo pid,ppid,stat,etime,command | rg '[g]so|[p]ytest|[u]vicorn|[c]affeinate' | head -30
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

La recarga real de A4v3 se ejecutó con este bloque de sólo lectura:

```sh
.venv/bin/python - <<'PY'
import json, math
from pathlib import Path
import torch
from gemma_system_one.training.decisions_pipeline import prepare_evaluation
ctx = prepare_evaluation(Path('runs/e4b_experiment/20260923T204945Z/checkpoint'), use_cache=False)
item = ctx.items_for('all', dataset=Path('data/pilot_v4_kdiag14_K4'))[0]
logits, stat = ctx.logits('audit-one', [item])
bb = ctx.encoder.backbone
pred = Path(Path('reports/phase6d/pred_kdiag14_K4_a4v3.txt').read_text().strip())
saved = next(x for x in map(json.loads, pred.read_text().splitlines()) if x['id'] == item.example.id)
diff = max(abs(a-b) for a, b in zip(logits[0].tolist(), saved['row_logits'], strict=True))
print(type(bb.model).__name__, bb.device, bb.dtype, bb.model.training, hasattr(bb.model, 'lm_head'), diff, stat)
assert type(bb.model).__name__ == 'Gemma4Model' and bb.device.type == 'mps' and bb.dtype == torch.bfloat16
assert not bb.model.training and not hasattr(bb.model, 'lm_head') and diff < 1e-5
PY
```

**Pendientes.** Se necesita un diagnóstico nuevo de posibles pistas de opciones con un control semánticamente coherente o un predictor que sólo vea pregunta/opciones, separado por grupos entre ajuste y evaluación. No reutilizar `final13`, `kdiag14*` ni otros tests abiertos para elegir hiperparámetros. Si se desarrolla v5, fijar protocolo y datos nuevos antes de entrenar; mantener el servicio A4v3 mientras no se supere su regla de selección. Continúan los límites ya conocidos: tareas sintéticas, IC anchos, timeout MPS que no corta el forward, memoria pico MPS no medida por RSS. No se ha probado un clon limpio con descarga/instalación desde cero.

---

# Registro anterior — fase 6d publicada; relevo del implementador (2026-09-24, 16:43 UTC)

**Verificación del relevo** (sin cambiar código, datos ni alcance; sin lanzar trabajo pesado). Comandos exactos y resultados:

```sh
git branch --show-current; git fetch -q origin; git log --oneline -4
# main; 9edc183 → daa27ce → 1b5ad45 → 65d4250
git rev-parse origin/main origin/fases-0-6; git status --short; git stash list
# ambas 9edc18340ab13fce5d9fe68ce7c7a2b28ff7ea1e; árbol limpio; stash vacío
.venv/bin/python scripts/snapshot_source.py
# fd340105be1c49de8e601bd57561318204d0058c3a67e756c9101a7940e378cc, 83 ficheros (el de las evaluaciones de la 6d)
pgrep -fl 'gso|pytest|uvicorn'; lsof -nP -iTCP:8000 -sTCP:LISTEN
# ambos: código 1, sin salida. Ningún proceso del proyecto; puerto libre
# existencia (no recarga) de los 7 checkpoints de la tabla de §3 histórica y de la calibración de servicio de A4v3: ok
```

**Publicación** (hecha a petición del usuario):
- commit `9edc183` en `fases-0-6`;
- `git merge --ff-only fases-0-6` en `main` (65d4250 → 9edc183, sin commit de merge);
- `git push origin main`;
- `git push -u origin fases-0-6`.

Remoto: `origin` en GitHub. Antes del push se buscó en todos los commits (`git grep` sobre `git rev-list --all`) el usuario local, el dominio del correo y `/Users/`: sin coincidencias. **Este relevo modifica sólo `docs/STATUS.md`**, que queda sin commit.

**Para el otro asistente:**
- Tarea natural: revisión independiente de la fase 6d, con el prompt 2 de `PROMPTS.md`.
- Puntos que conviene comprobar:
  - `balanced_fault_kind_pairs` y `swap_states` en `data/derive.py`;
  - `scripts/probe_option_cue.py`: sus cotas poblacionales y el sobreajuste de la cota en conjuntos pequeños;
  - la lectura del control con estados intercambiados;
  - la excepción de ruff en `pyproject.toml`.
- Siguiente trabajo técnico propuesto: el generador v5 (§8.4).

Sustituye a las secciones siguientes (relevo y revisión de la fase 6c), que quedan como registro histórico. Resuelve sus pendientes:
- el nuevo diagnóstico K8 con la composición controlada;
- la comprobación de ruff, que queda en verde.

Incluye, sin cambios, las correcciones de esa revisión en `README.md`, `reports/phase6c-final.md` y `docs/decisions/0012-…md`.

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **6d cerrada.** No hay fases posteriores en la spec §10; es la corrección de fallos del piloto detectados en revisión. Informe: **`reports/phase6d-final.md`**. Protocolo `reports/phase6d-protocol.md`, sha256 `474a32b4…` (en `reports/phase6d/protocol.sha256`), escrito antes de generar los datos y de evaluar |
| Corrección | `derive.balanced_fault_kind_pairs`: K4 = `none` + `other` + 2 categorías reales; K8 = K4 + 4 distractoras fijas. Información de la composición sobre la respuesta: **0,000** (`scripts/probe_option_cue.py`, 20 000 casos), frente a 0,019 (6c) y 0,141 (fase 6). Control `derive.swap_states` con estados intercambiados |
| Más opciones | Δaccuracy K8 − K4 (167 preguntas, 58 `other`): A4v3 −0,036 [−0,096; +0,024]; A4v4 −0,024 [−0,078; +0,030]; A2v4 −0,012 [−0,072; +0,054]. **Tolerancia a K = 8 no demostrada para ninguno** (criterio: límite inferior ≥ −0,05). `other` es el punto débil (A4v3: 0,64 → 0,50) |
| Control sin estado | Accuracy con estados intercambiados 0,20–0,26, por debajo del prior del conjunto (0,347): **no se explota ninguna pista** |
| Generador v4 | Deja 0,050 de información por composición (v3: 0,029). Los modelos no la aprovechan más allá del prior: `final13` con estados intercambiados da 0,38–0,41, frente a 0,427 de prior. **Pendiente: un generador v5 con composición equilibrada** |
| Servicio | Sin cambios: A4v3 (`configs/serve_e4b_text.yaml`, calibración de `calib7`) |
| Rama / commit | **`main` = `fases-0-6` = `9edc183`**, en local y en `origin` (GitHub); historia `65d4250 → 1b5ad45 → daa27ce → 9edc183`. `1b5ad45` está incompleto (le falta `src/gemma_system_one/data/`); usar `daa27ce` o posterior |
| Código | **`fd340105be1c49de8e601bd57561318204d0058c3a67e756c9101a7940e378cc`** (83 ficheros), el de las evaluaciones; copia en `artifacts/source/` |
| Pruebas | **293 CPU** (34,49 s) y **14 MPS/E2E reales** (127,78 s, 0 omitidos). `ruff check .` y `ruff format --check .` **en verde** (141 ficheros), `uv lock --check` y `git diff --check` correctos |

## 2. Procesos activos

`pgrep -fl 'gso|pytest|uvicorn'` terminó con código 1 y `lsof -nP -iTCP:8000 -sTCP:LISTEN` también: **ningún proceso del proyecto** y el puerto libre.

## 3. Checkpoints

Sin cambios respecto a la sección histórica siguiente (tabla con A4v3, A4v4, A2v4, B2, A2 y V2/C2 y sus calibraciones). En esta fase no se entrenó nada.

**Datos nuevos**, todos ya usados:
- `data/pilot_v4_kdiag14` y `data/pilot_v4_kdiag14_{K4,K8,K4swap,K8swap}`;
- `data/pilot_v4_final13_faultswap`.

**Conjuntos que no deben servir para decidir:**
- el test de `pilot_v3`, los tests de `vision_pilot_v1/v2` y `pilot_v3_holdout6*`;
- `pilot_v3_final8`, `pilot_v4_final13`, `pilot_v4_kdiag11*` y `pilot_v4_kdiag14*`.

El test y la partición de calibración de `pilot_v4` siguen sin leer.

## 4. Archivos del commit `9edc183` (sobre `daa27ce`)

**De la revisión de la 6c:** `README.md`, `reports/phase6c-final.md` y `docs/decisions/0012-…md`. Esta fase añade encima una línea al README y un anexo a la 0012.

**Nuevos:**
- `tests/unit/test_phase6d_balanced_k.py`;
- `scripts/derive_phase6d_data.py`, `scripts/probe_option_cue.py`;
- `reports/phase6d-protocol.md`, `reports/phase6d-final.md` y `reports/phase6d/` (logs, `pred_*.txt`, comparaciones, `option_cue_probe.json`, `run_chain.sh`).

**Modificados:**
- `src/gemma_system_one/data/derive.py`: `DIAG_DISTRACTOR_OPTIONS`, `balanced_fault_kind_pairs`, `swap_states`;
- `src/gemma_system_one/data/split.py`: una línea partida, sin cambio de comportamiento;
- `pyproject.toml`: excepción E501 y de formato para los tres generadores;
- `docs/STATUS.md`.

## 5. Decisiones

- **0012, con anexo de la 6d:** el diagnóstico de K pasa a ser equilibrado.
- **Configuración de ruff:** los generadores quedan exentos de E501 y de formato, para no alterar el código que produce los datasets con hash registrado.
- 0001–0011 siguen vigentes.

## 6. Comandos exactos

Lista completa en `reports/phase6d-final.md` («Comandos ejecutados»):
- test del diseño equilibrado y `probe_option_cue.py`;
- protocolo y su sha256;
- `derive_phase6d_data.py` y `validate-data` (6 conjuntos);
- `reports/phase6d/run_chain.sh` (15 evaluaciones);
- 3 `gso compare`;
- `pytest` CPU y MPS/E2E;
- ruff, formato y lock.

## 7. Fallos reproducibles

- **Corregido:** la pista de composición del diagnóstico K8 de la 6c. Se reproduce con `scripts/probe_option_cue.py` (`v4_widen_by_facts_K8_phase6c`: ganancia 0,019; el equilibrado da 0,000).
- **Corregido:** la comprobación de ruff. Sin la excepción de `pyproject.toml`, `ruff check .` falla con E501 en los generadores, que eran invisibles para ruff hasta `daa27ce`.
- **Abierto:** composición del generador de entrenamiento v4, 0,050 (`v4_generator_K3_6` en `reports/phase6d/option_cue_probe.json`).

## 8. Pendientes, en orden

1. **Commit y publicación:** hechos (`9edc183` en `main` y `origin`). Sólo queda sin commit esta actualización de STATUS.
2. **Clon limpio sin probar:** `git clone … && uv sync && uv run pytest tests/unit tests/integration`. Los pesos y los datos se recrean con `gso download` y `gso generate-data`.
3. **Revisión independiente de la 6d:**
   - el diseño equilibrado;
   - la cota «sólo opciones»;
   - la lectura del control con estados intercambiados.
4. **Generador v5** con composición equilibrada también en entrenamiento, con protocolo y datos nuevos. Después, reentrenar los cabezales y medir si mejora `other`.
5. **Opcionales:**
   - un diagnóstico de K con más preguntas `other` para estrechar los IC;
   - calibración de Choice;
   - LoRA sobre E4B;
   - E4B con imagen;
   - abstención.
6. **Riesgos que se mantienen:**
   - el timeout no interrumpe un forward MPS y la cola no limita las conexiones;
   - el RSS muestreado no es el pico de MPS;
   - datos sintéticos de una familia;
   - V2 sin calibrar.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Rutas:** las de `reports/phase6d/` están enmascaradas con `~`.

---

# Registro histórico — relevo tras revisión independiente de la fase 6c (2026-09-24, 15:49 UTC)

**Fase, rama y commit.** Fase 6c implementada en `fases-0-6`, HEAD `daa27cee01481d3fd07991e7d0ed5486c8740076` sobre `1b5ad45`; sin merge ni push. La revisión de esta fase está documentada y **sin commit**. El árbol contiene cuatro archivos modificados: `README.md`, `reports/phase6c-final.md`, `docs/decisions/0012-generator-v4-without-k-cue.md` y `docs/STATUS.md`. Esta última pasada de relevo sólo modifica STATUS; stash vacío. El diff anterior al relevo era de 139 inserciones y 30 eliminaciones en esos cuatro archivos.

**Decisiones y resultado.** Se conserva PyTorch/MPS, Gemma 4, el contrato Noul/Choice/Score y A4v3 como servicio textual (`configs/serve_e4b_text.yaml`). La regla S predeclarada está bien aplicada: A4v4 − A4v3 = −0,0069407 de NLL calibrada, IC [−0,0341615; +0,0239739]; el límite superior excede +0,02, así que A4v4 no sustituye al servicio. La decisión 0012 mantiene el generador v4, que elimina la pista determinista del número K. La revisión retira la conclusión de robustez K8 porque la composición de opciones aún revela parte de la etiqueta. No se modifican pesos, datos, backend, modelo ni contrato público.

**Fallos reproducibles y pendientes.** El recuento K8 es `[((2, False), 41), ((3, False), 92), ((3, True), 17)]`, donde la primera cifra es cuántas de las tres distractoras imposibles aparecen y el booleano indica etiqueta `other`: con dos, `other` es imposible. El comando exacto está en la sección siguiente. `ruff check . --statistics` termina con 123 `E501`; `ruff format --check .` señala cuatro módulos `data/`. Un nuevo diagnóstico K8 exige datos y protocolo nuevos que controlen la composición; `final13` y `kdiag11*` ya están usados. Quedan el gate Ruff y un clon limpio con `uv sync`. No se considera demostrada la equivalencia A4v4/A4v3 ni la generalización fuera de la tarea sintética.

**Procesos actuales.** `pgrep -fl 'gso|pytest|uvicorn|profile_lora_step|measure_request_batching'` y `lsof -nP -iTCP:8000 -sTCP:LISTEN` terminaron con código 1, sin salida: no hay entrenamiento, descarga ni servicio del proyecto activo. No se inició ninguno durante este relevo.

**Checkpoints locales (existencia verificada, sin nueva recarga salvo la A4v4/MPS de la revisión anterior):**

| Modelo | Checkpoint | Calibración disponible |
|---|---|---|
| A4v3, servicio actual | `runs/e4b_experiment/20260923T204945Z/checkpoint` | `calibration/calibration-20260924T045005Z.json` (vigente del servicio); también `…143729Z.json` para la comparación 6c |
| A4v4 | `runs/e4b_v4/20260924T140932Z/checkpoint` | `calibration/calibration-20260924T143416Z.json` |
| A2v4 | `runs/e2b_heads_v4/20260924T142144Z/checkpoint` | `calibration/calibration-20260924T143725Z.json` |
| B2 | `runs/pilot_lora_v3/20260923T012208Z/checkpoint` | `calibration/calibration-20260924T044408Z.json` |
| A2 | `runs/pilot_ce_v3/20260923T005721Z/checkpoint` | `calibration/calibration-20260924T044041Z.json` |
| V2 / C2 | `runs/vision2_heads/20260923T161024Z/checkpoint`; `runs/vision2_text_only/20260923T162937Z/checkpoint` | Sin calibración local |

**Comandos exactos de este relevo**, desde la raíz, y resultados:

```sh
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git stash list
pgrep -fl 'gso|pytest|uvicorn|profile_lora_step|measure_request_batching'
lsof -nP -iTCP:8000 -sTCP:LISTEN
.venv/bin/python - <<'PY'
from pathlib import Path
for run in ('runs/e4b_experiment/20260923T204945Z','runs/e4b_v4/20260924T140932Z','runs/e2b_heads_v4/20260924T142144Z','runs/pilot_lora_v3/20260923T012208Z','runs/pilot_ce_v3/20260923T005721Z','runs/vision2_heads/20260923T161024Z','runs/vision2_text_only/20260923T162937Z'):
    root = Path(run)
    print(run, 'checkpoint', (root/'checkpoint').is_dir(), 'calibrations', [p.name for p in sorted((root/'calibration').glob('*.json'))])
PY
git diff --check
```

Resultados: rama/HEAD indicados, cuatro archivos modificados, stash vacío, siete checkpoints presentes, calibraciones de la tabla presentes, `git diff --check` correcto, ambos comandos de procesos sin salida (exit 1). La revisión anterior probó 291 tests CPU/integración, un test E2B/LoRA real en MPS y la recarga/forward de A4v4 real en MPS/BF16 con logits idénticos a los guardados. Esas pruebas no se repitieron en este relevo; los comandos, resultados y límites están inmediatamente después. El diff del relevo sólo contiene rutas relativas y no incluye secretos ni datos privados.

---

# Revisión anterior de la fase 6c (2026-09-24)

Rama `fases-0-6`, HEAD `daa27ce` (`daa27ce…`), sin merge ni push. El diff local ya modificaba este STATUS por el relevo anterior; esta revisión añade cambios en `README.md`, `reports/phase6c-final.md`, `docs/decisions/0012-generator-v4-without-k-cue.md` y `docs/STATUS.md`. No modifica modelo, backend, contrato público, pesos, datos ni artefactos de predicción. El bloque siguiente conserva el cierre de la fase como registro histórico, con las correcciones puntuales de las filas de resultados.

**Dictamen:** la regla S se aplicó correctamente y mantiene A4v3: en `final13`, A4v4 − A4v3 = −0,0069407 [−0,0341615; +0,0239739] de NLL calibrada (882 preguntas, 294 grupos); +0,02397 supera el margen +0,02. Esto **no demuestra equivalencia ni no inferioridad**. El generador v4 quita la pista determinista del número K, pero el diagnóstico de ampliación a K8 conserva una pista parcial en la **composición** de las opciones. Se retira la conclusión de que A4v3 tolera K8 en una prueba sin fuga. La comparación principal entre A4v4 y A2v4 en v4 sigue siendo una medición de esta tarea sintética, sin atribución causal exclusiva a la pista corregida.

**Hallazgos por gravedad:**

| Gravedad | Evidencia y decisión |
|---|---|
| Alta, metodología del diagnóstico K8 | En `data/pilot_v4_kdiag11_K8`, los 17 ejemplos con etiqueta `other` incluyen las tres distractoras imposibles; 41 de los 133 no `other` incluyen sólo dos. Si falta una distractora, `other` queda excluida sin leer el estado. Además, 7 casos con `other` presente y `none` ausente son necesariamente no `other`. `widen_fault_kind_by_facts` usa hechos verdaderos para excluir la categoría real; con sólo 9 categorías totales y K8, esta pista es estructural. Los 150 resultados numéricos son reproducibles, pero no prueban robustez sin pistas. Se corrigieron informe, README y decisión 0012; no se reescribe el test ya usado. |
| Media, afirmación estadística | El IC de la regla S llega a +0,024 y el margen es +0,02. «En la práctica equivalentes» no está respaldado por el criterio declarado. Corregido en informes y README; A4v3 sigue siendo el servicio. |
| Baja, entrega reproducible | Tras el arreglo de `.gitignore`, `ruff check . --statistics` falla con 123 `E501` en los módulos `data/` recién incluidos, y `ruff format --check .` señala 4 archivos. La afirmación anterior de que Ruff pasaba era del árbol previo que ignoraba ese paquete. Se registra como gate pendiente; no se reformatean archivos de los experimentos en esta revisión. |

**Pruebas ejecutadas:** `shasum -a 256 -c reports/phase6c/protocol.sha256` → OK; `rg -n 'EXIT|ALL DONE' reports/phase6c/run_chain.log` muestra 2 entrenamientos, 3 calibraciones y 9 evaluaciones con exit 0; `.venv/bin/pytest tests/unit tests/integration -q` → **291 passed**, 2 warnings, 35,57 s; `.venv/bin/pytest tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs` → **1 passed**, 12,05 s con pesos E2B reales en MPS. Una recarga nueva de A4v4 produjo `Gemma4Model` en `mps:0`, BF16, base en eval, 0 parámetros base entrenables, sin `lm_head`, 8 logits finitos para Noul/Choice/Score y diferencia máxima 0,0 frente a predicciones guardadas. Esta última prueba es inferencia real MPS; no es una nueva prueba de gradientes E4B. Los 14 MPS/E2E del cierre anterior y el perfil E4B LoRA son evidencia histórica, no repetida aquí. `uv lock --check` y `git diff --check` pasaron; los dos gates Ruff fallaron como arriba.

**Reproducción breve del hallazgo K8:**

```sh
.venv/bin/python - <<'PY'
from collections import Counter
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import FAULT_DISTRACTOR_OPTIONS, fault_categories
counts = Counter()
for e in load_dataset('data/pilot_v4_kdiag11_K8').examples:
    reverse = {t: c for c, texts in fault_categories(e.language).items() for t in texts}
    present = {reverse[t] for t in e.question.criteria.values()}
    label = reverse[e.question.criteria[e.target.class_id]]
    n = len(present & set(FAULT_DISTRACTOR_OPTIONS[e.language]))
    counts[(n, label == 'other')] += 1
print(sorted(counts.items()))
PY
# [((2, False), 41), ((3, False), 92), ((3, True), 17)]
.venv/bin/ruff check . --statistics
# 123 E501; exit 1
.venv/bin/ruff format --check .
# 4 archivos sin formato; exit 1
```

**Comando exacto de la recarga A4v4/MPS** (sin caché ni servicio):

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from gemma_system_one.training.decisions_pipeline import prepare_evaluation
ref = json.loads(Path('reports/phase6c/final13_compare_a4v4_minus_a4v3.json').read_text())
saved = {r['id']: r['row_logits'] for r in map(json.loads, Path(ref['b']).read_text().splitlines())}
ctx = prepare_evaluation(Path('runs/e4b_v4/20260924T140932Z/checkpoint'), use_cache=False)
items = ctx.items_for('all', dataset=Path('data/pilot_v4_final13'))
selected = [next(it for it in items if it.example.question.type == t) for t in ('noul','choice','score')]
with torch.inference_mode():
    logits, ext = ctx.logits('all', selected)
model = ctx.encoder.backbone.model
print(type(model).__name__, next(model.parameters()).device, next(model.parameters()).dtype)
print('eval', not model.training, 'base_trainable', sum(p.numel() for p in model.parameters() if p.requires_grad), 'lm_head', hasattr(model,'lm_head'))
print('rows', [len(it.rows) for it in selected], 'forwards', ext['backbone_forwards'], 'finite', all(torch.isfinite(z).all().item() for z in logits))
print('max_abs_logit_diff', max(max(abs(float(a)-float(b)) for a,b in zip(z.tolist(), saved[it.example.id], strict=True)) for it,z in zip(selected,logits,strict=True)))
PY
# Gemma4Model mps:0 torch.bfloat16; eval True base_trainable 0 lm_head False;
# rows [1, 4, 3] forwards 8 finite True; max_abs_logit_diff 0.0
```

**Separación de datos comprobada:** `leakage_checks` dio `errors: []` para `pilot_v4`/`calib12`, `pilot_v4`/`final13`, `calib12`/`final13` y `final13`/`kdiag11_K`; respectivamente, 7, 1, 3 y 0 pares de estados casi duplicados. Las particiones comparten plantillas sintéticas, por lo que no miden transferencia fuera de esa familia.

**Procesos y artefactos:** `pgrep -fl 'gso|pytest|uvicorn|profile_lora_step'` y `lsof -nP -iTCP:8000 -sTCP:LISTEN` devolvieron código 1 sin salida: ningún entrenamiento o servicio activo. Checkpoint vigente A4v3: `runs/e4b_experiment/20260923T204945Z/checkpoint` (calibración `…/calibration-20260924T045005Z.json`); A4v4: `runs/e4b_v4/20260924T140932Z/checkpoint`; A2v4: `runs/e2b_heads_v4/20260924T142144Z/checkpoint`. Resto y datasets en §3. No se descargaron pesos ni se inició entrenamiento largo o servicio externo.

**Pendientes:** una prueba nueva de K8 necesitará un universo más amplio de distractoras y un diseño que iguale su composición entre etiquetas, con protocolo y datos nuevos; `final13` y `kdiag11*` ya no sirven para decidir. Resolver el gate Ruff sobre los cuatro módulos `data/` recién versionados y probar un clon limpio con `uv sync`. Los límites de memoria (swap +1,49 GB en el entrenamiento histórico A4v4), calibración y servicio descritos abajo persisten. No se ha hecho commit en esta revisión.

---

# Cierre anterior — fase 6c con commit y relevo verificado (2026-09-24, 15:24 UTC)

Sustituye a las secciones siguientes (relevo y revisión de las fases 6/6b), que quedan como registro histórico. Resuelve la pista determinista del número K en el generador y ejecuta un diagnóstico de ampliación de candidatos que la revisión posterior limita por composición. Incluye, sin cambios, las correcciones de aquella revisión: la guarda de solapes entre la calibración externa y el test en `decisions_pipeline.py` y su test, y las correcciones de los informes de las fases 6/6b y del README. **Todo está en el commit `daa27ce`.**

**Verificación del relevo** (sin cambiar código, datos ni alcance; sin lanzar trabajo pesado). Comandos exactos y resultados:

```sh
git branch --show-current; git log --oneline -3; git status --short; git stash list
# fases-0-6; daa27ce → 1b5ad45 → 65d4250; árbol limpio; stash vacío
.venv/bin/python scripts/snapshot_source.py
# dca04dbdf2713ab3544ffdedad0f05f43c8e777bb8a755b224bdf854dee6bca7, 81 ficheros (el de todas las ejecuciones de la 6c)
pgrep -fl 'gso|pytest|uvicorn'; lsof -nP -iTCP:8000 -sTCP:LISTEN
# ambos: código 1, sin salida. Ningún proceso del proyecto; puerto libre
# existencia (no recarga) de 9 manifiestos y calibraciones de §3: 9 ok
```

**Comprobación del commit `daa27ce`** (hecha al crearlo):

```sh
git archive daa27ce | tar -x -C <scratchpad>/export_daa27ce
PYTHONPATH=<export>/src .venv/bin/python -m pytest tests/unit tests/integration -q -p no:cacheprovider   # desde <export>
# importa gemma_system_one desde <export>/src; 291 passed
```

Reutiliza el `.venv` del repositorio: **un clon limpio con `uv sync` sigue sin probarse**.

**Para el otro asistente:**
- Tarea natural: revisión independiente de la fase 6c, con el prompt 2 de `PROMPTS.md`, y del arreglo del `.gitignore` (§7).
- Puntos que conviene comprobar:
  - que v4 no introduce otra pista: qué categorías acompañan a `other` y la frecuencia de las distractoras por etiqueta;
  - `widen_fault_kind_by_facts`;
  - la aplicación de la regla S (margen +0,02).

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **6c cerrada.** No hay fases posteriores en la spec §10; es la corrección de un fallo del piloto detectado en revisión. Informe: **`reports/phase6c-final.md`**. Protocolo `reports/phase6c-protocol.md`, sha256 `02d3ae6c…` (en `reports/phase6c/protocol.sha256`), escrito antes de generar los datos v4 y de entrenar |
| Corrección | **Generador `support-mixed-v4`** (decisión 0012): 3 categorías distractoras que nunca son la respuesta; probabilidad condicional de `other` fijada para K3–K6. `derive.widen_fault_kind_by_facts` amplía a K = 8 desde los hechos, con la pista de composición descrita arriba. v1–v3 siguen idénticos byte a byte |
| Regla S (servicio) | En `pilot_v4_final13` (882 preguntas, 294 grupos), A4v4 − A4v3 = **−0,007 [−0,034; +0,024]**. El límite superior supera el margen de +0,02, así que **se mantiene A4v3** (`configs/serve_e4b_text.yaml` no cambia). No se demostró equivalencia |
| Más opciones | Diagnóstico de 150 preguntas, K 3–6 frente a K8, con pista residual en la composición de opciones. Δaccuracy observada: **A4v3 0,000 [−0,040; +0,033]**, **A4v4 −0,020 [−0,060; +0,020]**, **A2v4 −0,147 [−0,207; −0,087]**. No demuestra tolerancia a K8 sin fuga |
| Otras | A4v4 − A2v4 = −0,149 [−0,195; −0,105] en v4; la ventaja observada de E4B persiste, sin atribución causal exclusiva |
| Rama / commit | Rama **`fases-0-6`**, HEAD **`daa27ce`** («Fase 6c y corrección del .gitignore…»), sobre `1b5ad45` y `main` = `65d4250`. Árbol limpio. **Sin merge en `main` ni push** (decisión del usuario). `1b5ad45` está **incompleto**: le falta `src/gemma_system_one/data/` (§7); usar `daa27ce` |
| Código | **`dca04dbdf2713ab3544ffdedad0f05f43c8e777bb8a755b224bdf854dee6bca7`** (81 ficheros), el mismo con el que se ejecutó todo; copia en `artifacts/source/` |
| Pruebas | **291 CPU** (34,61 s) y **14 MPS/E2E reales** (129,61 s, 0 omitidos), con este código, según el cierre anterior. Tras incluir `src/gemma_system_one/data/`, Ruff y formato fallan en el commit; véase revisión vigente arriba. `uv lock --check` y `git diff --check` correctos |

## 2. Procesos activos

`pgrep -fl 'gso|pytest|uvicorn'` terminó con código 1 y `lsof -nP -iTCP:8000 -sTCP:LISTEN` también: **ningún proceso del proyecto** y el puerto libre. Sin entrenamientos, descargas ni servidores.

## 3. Checkpoints, calibraciones y datos

| Ruta | Uso |
|---|---|
| `runs/e4b_experiment/20260923T204945Z/checkpoint` + `calibration/calibration-20260924T045005Z.json` (`calib7`) | **A4v3, servicio de texto vigente** (sin cambios) |
| `…/e4b_experiment/20260923T204945Z/calibration/calibration-20260924T143729Z.json` | Calibración de A4v3 con `calib12`, usada sólo en la evaluación de la 6c; el servicio no la usa |
| `runs/e4b_v4/20260924T140932Z/checkpoint` + su `calibration/` (`calib12`) | **A4v4**, E4B + cabezales con datos v4; alternativa equivalente |
| `runs/e2b_heads_v4/20260924T142144Z/checkpoint` + su `calibration/` | A2v4, E2B + cabezales con datos v4 |
| `runs/pilot_lora_v3/20260923T012208Z/checkpoint` + `calibration-20260924T044408Z.json` | B2, E2B + LoRA (v3); opción de menor latencia |
| `runs/pilot_ce_v3/…`, `runs/vision2_heads/…`, `runs/vision2_text_only/…` | A2, V2 y C2, sin cambios |
| `data/pilot_v4`, `pilot_v4_calib12`, `pilot_v4_final13`, `pilot_v4_kdiag11{,_K,_K8}` | Datos v4; hashes en el informe. `final13` y `kdiag11*` **ya usados** |

**Conjuntos ya usados, que no deben servir para decidir:** el test de `pilot_v3`, los tests de `vision_pilot_v1/v2`, `pilot_v3_holdout6*`, `pilot_v3_final8`, `pilot_v4_final13` y `pilot_v4_kdiag11*`. El test y la partición de calibración de `pilot_v4` siguen **sin leer**.

## 4. Archivos del commit `daa27ce` (sobre `1b5ad45`: 49 nuevos y 8 modificados)

**Corrección del `.gitignore`:**
- `data/`, `runs/` y `artifacts/` pasan a `/data/`, `/runs/` y `/artifacts/`;
- entra por primera vez `src/gemma_system_one/data/` (`__init__`, `dataset`, `derive`, `generate`, `generate_mixed`, `generate_vision`, `split`).

**De la revisión anterior**, sin tocar aquí:
- `src/gemma_system_one/training/decisions_pipeline.py`;
- `tests/integration/test_phase3_pipeline.py`;
- `reports/phase6-e4b.md`, `reports/phase6b-final.md`.

**De esta fase:**
- **Nuevos:**
  - `tests/unit/test_phase6c_generator_v4.py`;
  - `scripts/derive_phase6c_data.py`;
  - `configs/e4b_v4.yaml`, `configs/e2b_heads_v4.yaml`;
  - `docs/decisions/0012-generator-v4-without-k-cue.md`;
  - `reports/phase6c-protocol.md`, `reports/phase6c-final.md` y `reports/phase6c/` (logs, comparaciones, `other_by_k.txt`, `run_chain.sh`).
- **Modificados:**
  - `src/gemma_system_one/data/generate_mixed.py`: v4, `FAULT_DISTRACTOR_OPTIONS`, `_choice_fault_kind_v4`, `fault_categories`, `true_fault_category`;
  - `src/gemma_system_one/data/derive.py`: `widen_fault_kind_by_facts`;
  - `src/gemma_system_one/cli.py`: `--generator-version v4`;
  - `README.md` y este STATUS.

## 5. Decisiones

- **0012:** generador v4 sin la pista de K; el servicio se mantiene en A4v3 por la regla S.
- 0001–0011 siguen vigentes.

## 6. Comandos exactos

Lista completa en `reports/phase6c-final.md` («Comandos ejecutados»):
- `pytest` del generador v4;
- protocolo y su sha256;
- `generate-data --generator-version v4 --seed 0`, `gso split` y `scripts/derive_phase6c_data.py`;
- `validate-data` de los 5 conjuntos;
- `reports/phase6c/run_chain.sh`: 2 entrenamientos, 3 calibraciones y 9 evaluaciones;
- 5 `gso compare`;
- `pytest` CPU y MPS/E2E.

## 7. Fallos reproducibles

- **Fallo corregido, de entrega:** `.gitignore` con `data/` sin anclar excluía también `src/gemma_system_one/data/`, así que `1b5ad45` no importaba `gemma_system_one.data`.
  - Reproducción: `git check-ignore -v src/gemma_system_one/data/split.py` con el `.gitignore` de `1b5ad45`, o `git ls-files src/gemma_system_one/data` en ese commit (vacío).
  - Corregido en `daa27ce`: `git ls-files src/gemma_system_one/data` lista 7 ficheros y los tests pasan desde la copia extraída.
- **Fallo corregido:** la pista de K en v3. Se reproduce con `pytest tests/unit/test_phase6c_generator_v4.py::test_v3_leaks_label_through_k_and_v4_does_not` (0 casos `other` con K = 6 en v3).
- **Revisión posterior:** el diagnóstico K8 conserva una pista de etiqueta por composición, y Ruff/format fallan en el commit; véase la sección vigente.
- **Observación:** el swap creció 1,49 GB durante el entrenamiento de A4v4 (`runs/e4b_v4/20260924T140932Z/metrics.json`), sin superar el presupuesto de MPS y sin causa atribuida.

## 8. Pendientes, en orden

1. **Integración, a decidir por el usuario:** fusionar `fases-0-6` en `main` (`git checkout main && git merge fases-0-6`) y, si se quiere, hacer push.
2. **Clon limpio:** `git clone … && uv sync && uv run pytest tests/unit tests/integration`. No se ha probado; los pesos y los datos no van en Git (`gso download` y `generate-data` los recrean).
3. **Revisión independiente de la fase 6c:**
   - que v4 no introduce otra pista (p. ej., qué categorías acompañan a `other`);
   - el diagnóstico de K por hechos;
   - la aplicación de la regla S.
4. **Opcionales, siempre con datos nuevos:**
   - calibración de Choice más allá de una T global;
   - mejorar la respuesta `other`, que es la más difícil (sobre todo con K alto);
   - LoRA sobre E4B con `recompute_layers: true`;
   - E4B con imagen;
   - abstención.
5. **Riesgos que se mantienen:**
   - el timeout no interrumpe un forward MPS y la cola no limita las conexiones;
   - el RSS muestreado no es el pico de MPS;
   - datos sintéticos de una familia;
   - V2 sin calibrar.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Rutas:** las de `reports/phase6c/` están enmascaradas con `~`.
- **Fuera de Git:** `reports/doctor/` sigue excluido por `.gitignore`.

---

# Registro histórico — relevo tras revisión independiente de fases 6/6b (2026-09-24, 13:44 UTC)

Esta sección sustituye el dictamen del relevo anterior; las secciones siguientes conservan el historial. Se revisaron `AGENTS.md`, `README.md`, `docs/ESPECIFICACION.md`, `docs/AUDITORIA.md`, el diff de Git, el código, el protocolo y las predicciones reales. **Rama:** `fases-0-6`; **HEAD:** `1b5ad45` (sobre `65d4250`). La revisión permanece **sin commit**. La fase 6b conserva A4 como servicio de texto según la regla C, dentro de estos datos sintéticos; no se aprueba la afirmación de robustez a K=8 de la fase 6. No se ha demostrado generalización fuera de la familia sintética.

**Relevo actual:** fase 6b revisada, sin fase nueva ni cambio de alcance. HEAD completo: `1b5ad45466748774f1f5adc255328ef654e251db`. Seis archivos modificados sin commit: `README.md`, `docs/STATUS.md`, `reports/phase6-e4b.md`, `reports/phase6b-final.md`, `src/gemma_system_one/training/decisions_pipeline.py` y `tests/integration/test_phase3_pipeline.py`; stash vacío. Esta última pasada de relevo sólo actualiza `docs/STATUS.md`. No se ha hecho merge ni push.

**Decisiones vigentes:** 0001–0011 no cambian de arquitectura. Se mantiene PyTorch/MPS, Gemma 4 E2B/E4B, el contrato Noul/Choice/Score, A4 para el servicio textual conforme a R2 y regla C, y la calibración de `calib7` fijada por el protocolo 6b. La corrección de solapes es una guarda de evaluación dentro de ese contrato. Los datos v3 y sus hashes no se alteran; el experimento de K ampliado queda sin conclusión de robustez. Una prueba nueva requiere versión de datos y protocolo nuevos, sin seleccionar con `holdout6*` ni `final8`.

**Comprobación de este relevo (comandos exactos adicionales):**

```sh
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git stash list
pgrep -fl 'gso|pytest|uvicorn|profile_lora_step|measure_request_batching'
lsof -nP -iTCP:8000 -sTCP:LISTEN
git diff --check
.venv/bin/python - <<'PY'
from pathlib import Path
paths = [
'runs/e4b_experiment/20260923T204945Z/checkpoint',
'runs/e4b_experiment/20260923T204945Z/calibration/calibration-20260924T045005Z.json',
'runs/pilot_lora_v3/20260923T012208Z/checkpoint',
'runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260924T044408Z.json',
'runs/pilot_ce_v3/20260923T005721Z/checkpoint',
'runs/pilot_ce_v3/20260923T005721Z/calibration/calibration-20260924T044041Z.json',
'runs/vision2_heads/20260923T161024Z/checkpoint',
'runs/vision2_text_only/20260923T162937Z/checkpoint',
]
for path in paths:
    print(path, Path(path).exists())
PY
```

Resultados: rama y HEAD anteriores; los seis archivos de la lista, stash vacío, `git diff --check` correcto; las dos comprobaciones de procesos devolvieron código 1 sin salida. **Ningún entrenamiento, descarga o servidor sigue activo**. Las ocho rutas de checkpoints y calibraciones enumeradas más abajo existen; se comprobó existencia, no integridad de todos los archivos ni recarga de todos los modelos. A4 sí se recargó en MPS durante la revisión. El relevo usa rutas relativas y no incluye secretos ni datos personales.

## Hallazgos y correcciones de esta revisión

| Gravedad | Hallazgo / resultado |
|---|---|
| Alta, metodología | `src/gemma_system_one/data/derive.py::widen_fault_kind` usa la etiqueta para decidir cuántas opciones puede añadir. En `pilot_v3_holdout6_faultK8`, K5/K6/K7 son 2/6/8 casos `other` y los 119 K8 no son `other`. K revela la respuesta; el diagnóstico «tolera K=7–8» se retira de `reports/phase6-e4b.md`. No se reescriben datos históricos ni hashes. |
| Media, fuga entre calibración y test | `run_evaluate_decisions` sólo rechazaba un dataset externo con el mismo sha256 que el de ajuste de temperaturas. Un conjunto con una pregunta compartida y otra nueva pasaba la guarda. Test de integración reproducido primero en rojo; el código ahora valida hash y solapes por grupo/entrada/imagen frente al conjunto de calibración. El mismo test pasa. Requiere conservar accesible el dataset de calibración, que ya está registrado en el artefacto. |
| Media, límite de datos | El generador v3 de `fault_type` trunca a K5 los casos `other` que sortearon K6. En `final8`, 32 ejemplos `fault_type` con K6 son no `other`. Se conserva v3 por reproducibilidad; cualquier corrección de generación requiere nueva versión, datasets y protocolo antes de otro test. |
| Baja, precisión del informe | `final8` no ajustó pesos ni temperaturas, pero **sí** se usó para la decisión de confirmación de la regla C. Corregidas las frases «no usado para ninguna decisión» en README e informe. |

**Resultado principal comprobado:** 888 pares de predicciones de A4 y B2 con entradas idénticas, 296 grupos; A4 − B2 = −0,1456947 de NLL calibrada, IC bootstrap [−0,1859615; −0,1043155] en `reports/phase6b/final_compare_a4_minus_b2.json`. Un análisis *posterior* sin las 141 preguntas `fault_type` da −0,13094 [−0,17500; −0,08785] en 747 preguntas y 296 grupos. Es sensibilidad descriptiva, no nuevo test de selección. Las comprobaciones de solape exacto train/calib7, train/final8, calib7/final8 y holdout6_clean/final8 dan `errors: []`; hay, respectivamente, 9, 4, 3 y 0 pares de estados casi duplicados, que limitan independencia semántica.

**Modelo real:** en esta revisión se recargó A4 desde el checkpoint local y se hicieron 8 forwards para una pregunta Noul, Choice y Score de `final8`: `Gemma4Model`, `mps:0`, BF16, base en eval, 0 parámetros base entrenables, sin `lm_head`, 8 logits finitos. La comparación adicional con las predicciones guardadas dio diferencia máxima de logits 0,0 en las tres preguntas. Esto verifica inferencia MPS real. También pasó `tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps` (1 passed, 11,77 s): gradientes finitos, base intacta y recarga LoRA, con pesos E2B reales. El perfil histórico `reports/phase6b/lora_step_e4b_recompute.json` registra seis pasos reales de E4B/MPS y 132 gradientes LoRA finitos por paso; no se repitió ese perfil en esta revisión. Los tests CPU no se presentan como prueba MPS.

**Archivos modificados desde HEAD:** `src/gemma_system_one/training/decisions_pipeline.py`, `tests/integration/test_phase3_pipeline.py`, `reports/phase6-e4b.md`, `reports/phase6b-final.md`, `README.md` y este `docs/STATUS.md` (que ya estaba modificado por el relevo previo). No cambió backend, modelo, contrato público, pesos, datos ni configuración de servicio. La huella de 78 fuentes tras la corrección es `b51bab4c25a1cb3806056942589cdfd907004c4e9052c9f8959cc9941d496900` (`artifacts/source/…tar`); la huella anterior `a7f645…` corresponde al cierre de 6b.

**Comandos de verificación ejecutados en esta revisión (desde la raíz):**

```sh
git status --short
git diff --stat
git diff -- src/gemma_system_one/training/decisions_pipeline.py tests/integration/test_phase3_pipeline.py
shasum -a 256 -c reports/phase6b/protocol.sha256
.venv/bin/pytest tests/integration/test_phase3_pipeline.py::test_external_calibration_set_is_bound_checked_and_never_reused_as_test -q
# Antes de corregir: FALLÓ, DID NOT RAISE ValueError. Después: 1 passed in 3.20s.
.venv/bin/pytest tests/unit tests/integration -q
# 288 passed, 2 warnings, 33.87s (CPU y dobles de prueba)
.venv/bin/pytest tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs
# 1 passed in 11.77s (E2B/MPS con pesos reales)
.venv/bin/ruff check src/gemma_system_one/training/decisions_pipeline.py tests/integration/test_phase3_pipeline.py
.venv/bin/ruff format --check src/gemma_system_one/training/decisions_pipeline.py tests/integration/test_phase3_pipeline.py
uv lock --check
git diff --check
# Todos correctos.
.venv/bin/python scripts/snapshot_source.py
# b51bab4c25a1cb3806056942589cdfd907004c4e9052c9f8959cc9941d496900; 78 ficheros.
pgrep -fl 'gso|pytest|uvicorn'
lsof -nP -iTCP:8000 -sTCP:LISTEN
# Ambos: código 1, sin salida; no hay entrenamiento ni servidor activo.
```

Comandos exactos de los recuentos y del smoke MPS, sin ficheros de salida:

```sh
.venv/bin/python - <<'PY'
from collections import Counter
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import FAULT_KIND_OPTIONS
for root in ('data/pilot_v3_holdout6_faultK8', 'data/pilot_v3_final8'):
    counts = Counter()
    for ex in load_dataset(root).examples:
        if ex.task_family != 'fault_type':
            continue
        label = ex.question.criteria[ex.target.class_id]
        other = label in FAULT_KIND_OPTIONS[ex.language]['other']
        counts[(len(ex.question.criteria), other)] += 1
    print(root, sorted(counts.items()))
PY
# faultK8: ((5, True), 2), ((6, True), 6), ((7, True), 8), ((8, False), 119)
# final8: ((3, False), 25), ((3, True), 4), ((4, False), 37), ((4, True), 8),
#         ((5, False), 32), ((5, True), 3), ((6, False), 32)
.venv/bin/python - <<'PY'
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.training.decisions_pipeline import _external_leakage
pairs = [('data/pilot_v3', 'data/pilot_v3_calib7'), ('data/pilot_v3', 'data/pilot_v3_final8'), ('data/pilot_v3_calib7', 'data/pilot_v3_final8'), ('data/pilot_v3_holdout6_clean', 'data/pilot_v3_final8')]
for a, b in pairs:
    d = _external_leakage(load_dataset(a), load_dataset(b).examples)
    print(a, b, d)
PY
# errors: [] en los cuatro pares; near_duplicate_states: 9, 4, 3, 0.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from gemma_system_one.metrics import compare_predictions
ref = json.loads(Path('reports/phase6b/final_compare_a4_minus_b2.json').read_text())
a = [json.loads(line) for line in Path(ref['a']).read_text().splitlines()]
b = [json.loads(line) for line in Path(ref['b']).read_text().splitlines()]
keep = lambda rows: [r for r in rows if r['task_family'] != 'fault_type']
result = compare_predictions(keep(a), keep(b), reps=1000, seed=0)
print(result['questions'], result['bootstrap']['groups'], result['same_inputs'], result['point']['b_minus_a.nll_all'], result['bootstrap']['intervals']['b_minus_a.nll_all'])
PY
# 747 296 True -0.13093669669246008 [-0.1750008002888237, -0.08785222366203808]
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from gemma_system_one.training.decisions_pipeline import prepare_evaluation
checkpoint = Path('runs/e4b_experiment/20260923T204945Z/checkpoint')
ctx = prepare_evaluation(checkpoint, use_cache=False)
items = ctx.items_for('all', dataset=Path('data/pilot_v3_final8'))
selected = [next(it for it in items if it.example.question.type == t) for t in ('noul', 'choice', 'score')]
with torch.inference_mode():
    logits, extraction = ctx.logits('all', selected)
backbone = ctx.encoder.backbone.model
saved = {r['id']: r['row_logits'] for r in map(json.loads, Path('runs/e4b_experiment/20260923T204945Z/evaluations/pilot_v3_final8-all-calibrated-20260924T050002Z-predictions.jsonl').read_text().splitlines())}
diffs = [max(abs(float(a)-float(b)) for a, b in zip(z.tolist(), saved[it.example.id], strict=True)) for it, z in zip(selected, logits, strict=True)]
print(type(backbone).__name__, next(backbone.parameters()).device, next(backbone.parameters()).dtype, 'eval', not backbone.training, 'trainable', sum(p.numel() for p in backbone.parameters() if p.requires_grad), 'lm_head', hasattr(backbone, 'lm_head'))
print('rows', [len(it.rows) for it in selected], 'forwards', extraction['backbone_forwards'], 'finite', all(torch.isfinite(z).all().item() for z in logits), 'max_abs_diff', max(diffs))
PY
# Gemma4Model mps:0 torch.bfloat16 eval True trainable 0 lm_head False
# rows [1, 4, 3] forwards 8 finite True max_abs_diff 0.0
```

No se inició entrenamiento largo, descarga ni servicio externo.

**Procesos y artefactos para el relevo:** la última inspección no encontró `gso`, `pytest`, `uvicorn` ni oyente en TCP:8000. Checkpoint A4: `runs/e4b_experiment/20260923T204945Z/checkpoint`; calibración vigente A4: `runs/e4b_experiment/20260923T204945Z/calibration/calibration-20260924T045005Z.json`. B2: `runs/pilot_lora_v3/20260923T012208Z/checkpoint` y `calibration/calibration-20260924T044408Z.json` bajo esa ejecución. A2: `runs/pilot_ce_v3/20260923T005721Z/checkpoint` y `calibration/calibration-20260924T044041Z.json`. V2 y C2: `runs/vision2_heads/20260923T161024Z/checkpoint` y `runs/vision2_text_only/20260923T162937Z/checkpoint`. Los datasets `data/`, pesos `runs/` y tar `artifacts/` están excluidos de Git; no hay secretos nuevos en el diff.

**Pendientes:** (1) diseñar, con datos/protocolo nuevos y sin reutilizar `final8` ni `holdout6*` para elegir, una versión del generador sin la pista de K y una prueba válida de ampliación de candidatos; (2) revisar en ejecución limpia el commit tras incorporar esta corrección; (3) los límites ya declarados del servicio y la calibración de Choice siguen vigentes. No se ha hecho commit, merge ni push en esta revisión.

---

# Relevo anterior — fases 0–6 completas y con commit (2026-09-24, 13:25 UTC)

Sustituye a las secciones siguientes (relevo y revisión de la fase 6), que quedan como registro histórico. La fase 6b resolvió sus pendientes 2 y 3 (medida independiente del servicio elegido y calibración de A4) y ejercitó en MPS la corrección del perfilador (pendiente 4). El pendiente 1 (commit) está hecho: `1b5ad45`, en la rama `fases-0-6`.

**Verificación del relevo** (sin cambiar código, datos ni alcance; sin lanzar trabajo pesado). Comandos exactos y resultados:

```sh
git branch --show-current; git log --oneline -3; git status --short; git stash list
# fases-0-6; 1b5ad45 (Fases 0–6 …) sobre 65d4250; árbol limpio; stash vacío
.venv/bin/python scripts/snapshot_source.py
# a7f6450022aea2af933ee0583415420bcf25c92689d3ffb7d3e3ec148b39d682, 78 ficheros (igual que al cierre de 6b)
pgrep -fl 'gso|pytest|uvicorn'; lsof -nP -iTCP:8000 -sTCP:LISTEN
# ambos: código 1, sin salida. Ningún proceso del proyecto; puerto libre
# existencia (no recarga) de los 8 manifiestos y calibraciones de §3: 8 ok
```

**Para el otro asistente:**
- Tarea natural: revisión independiente de la fase 6b, con el prompt 2 de `PROMPTS.md`.
- Puntos que conviene comprobar:
  - la guarda de la calibración externa (`calibration.run_calibrate`, `EXTERNAL_SPLIT`, y el rechazo en `run_evaluate_decisions`);
  - `scripts/derive_phase6b_data.py` (exclusiones);
  - la aplicación de la regla C.
- El commit es la referencia; basta con `git show --stat 1b5ad45`.
- No usar `pilot_v3_final8` ni `pilot_v3_holdout6*` para decidir nada.

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **6 completa, incluida la 6b.** No hay fases posteriores en la spec §10. Informes: `reports/phase6-e4b.md` (selección) y **`reports/phase6b-final.md`** (medida final). Protocolo `reports/phase6b-protocol.md`, sha256 `209e9259…` (en `reports/phase6b/protocol.sha256`), escrito antes de generar los datos y ejecutar los modelos |
| Resultado | **Regla C:** en el test final nuevo `pilot_v3_final8` (888 preguntas, 296 grupos, no usado para ajustar pesos ni temperaturas; sí usado en la regla C), A4 − B2 = **−0,146 [−0,186; −0,104]** de NLL calibrada, así que **A4 queda confirmado** como servicio de texto para esta tarea sintética. A4 − A2 = −0,183 [−0,227; −0,141]; B2 − A2 = −0,038 [−0,066; −0,012] |
| Calibración | Temperaturas nuevas con 1200 preguntas externas (`pilot_v3_calib7`). NLL de A4 en `final8`: 0,2021 sin T, 0,2089 con T de 300 preguntas y **0,2010 con T de `calib7`**. Resuelto según el criterio declarado, de forma marginal: en Choice la T global sigue empeorando (0,279 → 0,296). En A2 y B2 mejora claramente |
| Servicio | `configs/serve_e4b_text.yaml` (A4) y `configs/serve_text.yaml` (B2) apuntan ahora a las calibraciones de `calib7`. Benchmark de A4: 100 × 200, p50/p95 910/1546 ms, ráfaga válida y log con 112 peticiones |
| Rama / commit | **`fases-0-6`, commit `1b5ad45`** («Fases 0–6: evaluador Gemma 4…»), sobre `main` = `65d4250`. 298 ficheros (297 nuevos + `README.md`); árbol limpio tras el commit. **Sin fusionar en `main` ni push** (decisión del usuario). Fuera de Git por `.gitignore`: `data/`, `runs/`, `artifacts/`, `.venv/`, `*.safetensors` y `reports/doctor/` (el JSON de fase 0 con la ruta absoluta queda fuera). Antes del commit se buscaron rutas personales, correos y tokens: ninguno |
| Referencia del árbol | `artifacts/tree/76a3562d….tar` (304 ficheros, generado antes de esta línea); anteriores `9f54e5ef…` y `e7b7a322…`. No sustituye a un commit |
| Código | Actual **`a7f6450022aea2af933ee0583415420bcf25c92689d3ffb7d3e3ec148b39d682`** (78 ficheros), con copia en `artifacts/source/`. Calibraciones, evaluaciones y perfil se ejecutaron con `9c31e85b…`; el benchmark, con `a7f64500…` (sólo cambian los YAML de servicio) |
| Pruebas | **288 CPU** (34,41 s) y **14 MPS/E2E reales** (128,30 s, 0 omitidos), con el código actual. Ruff, formato (128 ficheros), `uv lock --check` y `git diff --check` correctos |

## 2. Procesos activos

`pgrep -fl 'gso|pytest|uvicorn'` terminó con código 1 y `lsof -nP -iTCP:8000 -sTCP:LISTEN` también: **ningún proceso del proyecto** y el puerto libre. El servidor de benchmark terminó por SIGTERM. No hay entrenamientos ni descargas.

## 3. Checkpoints, calibraciones y datos

| Ruta | Uso |
|---|---|
| `runs/e4b_experiment/20260923T204945Z/checkpoint` | **A4**, E4B + cabezales (servicio de texto recomendado) |
| `…/e4b_experiment/20260923T204945Z/calibration/calibration-20260924T045005Z.json` | **Calibración vigente de A4** (`calib7`); la anterior (`…210339Z`, 300 preguntas) queda como historial |
| `runs/pilot_lora_v3/20260923T012208Z/checkpoint` + `calibration/calibration-20260924T044408Z.json` | B2 (opción de menor latencia) y su calibración vigente (`calib7`) |
| `runs/pilot_ce_v3/20260923T005721Z/checkpoint` + `calibration/calibration-20260924T044041Z.json` | A2 (comparador) |
| `runs/vision2_heads/20260923T161024Z/checkpoint`; `runs/vision2_text_only/20260923T162937Z/checkpoint` | V2, servicio visual E2B; C2, control textual |
| Caché HF | E2B `3e22461f…` y E4B `ee0ef602…` (una copia de cada) |
| `data/pilot_v3_calib7` (`cdf1005c…`) | Sólo calibración externa |
| `data/pilot_v3_final8` (`d9f1b2fb…`) | Test final de la fase 6b, **ya usado** |
| `data/pilot_v3_seed7`, `data/pilot_v3_seed8` | Generados sin filtrar (origen de los anteriores) |

**Conjuntos ya usados, que no deben servir para decidir:** el test de `pilot_v3`, los tests de `vision_pilot_v1/v2`, `pilot_v3_holdout6*` (conjunto de selección de la fase 6) y `pilot_v3_final8`.

## 4. Archivos de esta fase (6b)

**Nuevos:**
- `scripts/derive_phase6b_data.py`;
- `reports/phase6b-protocol.md`, `reports/phase6b-final.md`;
- `reports/phase6b/`: logs, `run_chain.sh`, comparaciones, `calibration_variants_final8.json`, benchmark y perfil.

**Modificados:**
- `src/gemma_system_one/calibration.py`: calibración externa, `EXTERNAL_SPLIT`;
- `src/gemma_system_one/training/decisions_pipeline.py`: `evaluate` rechaza el mismo conjunto de calibración;
- `src/gemma_system_one/cli.py`: `calibrate --split all --dataset`;
- `configs/serve_e4b_text.yaml`, `configs/serve_text.yaml`: calibración de `calib7`;
- `tests/integration/test_phase3_pipeline.py`: un test;
- `README.md`: párrafo de la fase 6b;
- `docs/decisions/0011-…md`: anexo de confirmación.

Los ficheros de la fase 6 están en la sección histórica siguiente. `data/`, `runs/`, `artifacts/` y `.venv/` quedan fuera de Git.

## 5. Decisiones

- **0011, con anexo:** A4 confirmado por la regla C; el servicio usa la calibración de `calib7`.
- 0001–0010 siguen vigentes.
- La calibración externa es una extensión del procedimiento de la spec §6.2: mismo método, otro conjunto sin fugas y vinculado por sha256.

## 6. Comandos exactos

Lista completa en `reports/phase6b-final.md` («Comandos ejecutados»):
- protocolo y su sha256;
- `generate-data` con semillas 7 y 8, `scripts/derive_phase6b_data.py` y `validate-data`;
- `reports/phase6b/run_chain.sh`: 3 calibraciones externas, 3 evaluaciones finales y el perfil LoRA de E4B;
- 3 `gso compare`;
- `gso benchmark` de `serve_e4b_text.yaml`;
- `pytest` CPU y MPS/E2E.

## 7. Fallos reproducibles

No aparecieron fallos de código nuevos. Límite metodológico que persiste: una temperatura global en Choice empeora la NLL de A4 en `final8`. Se reproduce con `reports/phase6b/calibration_variants_final8.json`, calculado desde los `row_logits` de `pred_a4`.

## 8. Pendientes, en orden

1. **Integración, a decidir por el usuario:** fusionar `fases-0-6` en `main` (`git checkout main && git merge fases-0-6`) y, si se quiere, hacer push. **No verificado:** un clon limpio del commit con `uv sync` y tests; no se ha probado.
2. **Revisión independiente de la fase 6b:**
   - calibración externa y su guarda;
   - datos sin solapes (4 grupos excluidos de `final8`; 4 pares casi duplicados declarados);
   - aplicación de la regla C.
3. **Opcionales:**
   - calibración de Choice más allá de una T global, decidida con datos nuevos y no con `final8`;
   - LoRA sobre E4B con su configuración real (`recompute_layers: true`);
   - E4B con imagen;
   - abstención con umbrales congelados;
   - Score por tramos.
4. **Riesgos que se mantienen:**
   - el timeout no interrumpe un forward MPS y la cola no limita las conexiones;
   - el RSS muestreado no es el pico de MPS;
   - datos sintéticos de una familia; V2 sin calibrar;
   - falta la copia histórica `70068e15…`;
   - `reports/doctor/20260922T195301Z.json` contiene una ruta absoluta del usuario.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Rutas:** las de `reports/phase6b/` están enmascaradas con `~`.
- **Credenciales:** no se usó ningún token.

---

# Registro histórico — relevo tras la revisión de fase 6 (2026-09-24, 04:31 UTC)

Este relevo **sólo modifica `docs/STATUS.md`**. Mantiene el alcance, PyTorch/MPS, los modelos y el contrato público. La revisión técnica vigente empieza tras esta sección; los cierres anteriores son históricos.

| Campo | Estado para el siguiente asistente |
|---|---|
| Fase y dictamen | Fase 6 implementada y revisada. E4B/BF16 tuvo una recarga y un forward reales en MPS en la revisión anterior. La comparación A4 − B2 se reprodujo desde 885 pares de predicciones. **No es una evaluación final independiente del servicio A4 elegido**, porque R2 usó ese holdout para escogerlo. |
| Rama y commit | `main`, `65d425025dd654fb4814062dc42b245062f151b1`; stash vacío. |
| Árbol modificado | `README.md` modificado; `.gitignore`, `.python-version`, `configs/`, `docs/STATUS.md`, `docs/decisions/`, `pyproject.toml`, `reports/`, `scripts/`, `src/`, `tests/` y `uv.lock` sin seguimiento. `git diff --stat` sólo muestra README y no representa los archivos nuevos. En este relevo no se tocó código, datos, pesos ni configuraciones. |
| Decisiones | 0001–0009 siguen vigentes. 0010 mantiene una fila por forward y ofrece recomputación para LoRA. 0011 recomienda A4 para texto según R2, con la salvedad del holdout usado para selección. Ninguna decisión técnica cambió en este relevo. |
| Huella | Las 77 fuentes siguen en `125c93fe8b0b54abcb9bd33d39c8a0c04af44c5d6c62115607fcfd7e741e0f2a`; el tar correspondiente existe en `artifacts/source/`. Los tar `artifacts/tree/9f54e5ef…` y `e7b7a322…` están presentes. Son referencias locales, no un commit. |
| Procesos | La inspección por ejecutable devolvió `[]` para `gso`, `pytest` y `uvicorn`. `lsof -nP -iTCP:8000 -sTCP:LISTEN` terminó con código 1 y sin salida. No se inició entrenamiento, descarga ni servicio en este relevo. |

**Checkpoints locales:** sólo se verificó su existencia ahora; no se repitió su recarga o integridad.

| Modelo | Ruta |
|---|---|
| A4, E4B + cabezales | `runs/e4b_experiment/20260923T204945Z/checkpoint` |
| Calibración A4 | `runs/e4b_experiment/20260923T204945Z/calibration/calibration-20260923T210339Z.json` |
| B2, E2B + LoRA | `runs/pilot_lora_v3/20260923T012208Z/checkpoint` |
| Calibración B2 | `runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json` |
| A2, E2B + cabezales | `runs/pilot_ce_v3/20260923T005721Z/checkpoint` |
| V2, E2B visual; C2, control textual | `runs/vision2_heads/20260923T161024Z/checkpoint`; `runs/vision2_text_only/20260923T162937Z/checkpoint` |

**Comandos exactos de esta verificación y resultados:**

```sh
git branch --show-current && git rev-parse HEAD && git status --short && git diff --stat && git stash list
# main; 65d425025dd654fb4814062dc42b245062f151b1; estado según tabla; stash vacío
date -u '+%Y-%m-%d %H:%M:%S UTC'
# 2026-09-24 04:31:46 UTC
.venv/bin/python -c 'import os,psutil; print([(p.pid,p.info["name"]) for p in psutil.process_iter(["name","cmdline"]) if p.pid!=os.getpid() and (os.path.basename((p.info["cmdline"] or [""])[0]) in {"gso","pytest","uvicorn"})])'
# []
lsof -nP -iTCP:8000 -sTCP:LISTEN
# código 1, sin salida
.venv/bin/python -c 'from pathlib import Path; from gemma_system_one.env import source_fingerprint; x=source_fingerprint(); print(x["sha256"], x["files"], (Path("artifacts/source")/(x["sha256"]+".tar")).is_file())'
# 125c93fe8b0b54abcb9bd33d39c8a0c04af44c5d6c62115607fcfd7e741e0f2a 77 True
shasum -a 256 -c reports/phase6/protocol.sha256
# reports/phase6-protocol.md: OK
git diff --check
# sin errores
.venv/bin/python -c 'from pathlib import Path; p=["runs/e4b_experiment/20260923T204945Z/checkpoint/manifest.json","runs/e4b_experiment/20260923T204945Z/calibration/calibration-20260923T210339Z.json","runs/pilot_lora_v3/20260923T012208Z/checkpoint/manifest.json","runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json","runs/pilot_ce_v3/20260923T005721Z/checkpoint/manifest.json","runs/vision2_heads/20260923T161024Z/checkpoint/manifest.json","runs/vision2_text_only/20260923T162937Z/checkpoint/manifest.json"]; print({x:Path(x).is_file() for x in p})'
# siete rutas: True
.venv/bin/python -c 'from pathlib import Path; print({p.name:p.is_file() for p in Path("artifacts/tree").glob("*.tar")})'
# dos tar: True
```

También se comprobó con sondas Python de sólo lectura la presencia de las siete rutas de la tabla y de los dos tar de `artifacts/tree/`; todas estaban presentes. **No se ejecutaron tests CPU/MPS en este relevo.** Los 287 tests CPU y la sonda E4B/MPS consignados debajo pertenecen a la revisión anterior; los 14 MPS/E2E del cierre de fase 6 son aún anteriores.

**Fallos reproducibles y tareas pendientes:**

1. Desde `HEAD` solo no se puede ejecutar el proyecto; `git status --short` muestra la implementación como `??`. Versionar el árbol cuando se decida el commit, excluyendo `data/`, `runs/`, `artifacts/` y credenciales.
2. R2 eligió A4 usando `pilot_v3_holdout6_clean`. Tratar ese holdout y sus variantes `faultK*` como ya usados; reservar un conjunto nuevo e intacto para medir el servicio elegido. No reajustar con ellos.
3. A4 empeoró la NLL del holdout tras temperatura (0,174 → 0,200). Estudiar más datos de calibración o un método distinto usando validación para decidir; medir después en el conjunto nuevo.
4. El perfil de seis pasos LoRA E4B no valida un entrenamiento completo. El perfilador ya falla si falta un gradiente LoRA, pero esa corrección no se volvió a ejercitar en MPS. Si se prosigue con LoRA E4B, usar la configuración real y medir memoria/recarga sin entrenamientos simultáneos.

---

# Registro de la revisión independiente de fase 6 (2026-09-24)

**Veredicto:** la carga y la inferencia E4B de la fase 6 tienen evidencia real en MPS, y las cifras archivadas de la comparación coinciden con los ficheros de predicciones. La mejora medida es válida como resultado de este conjunto sintético. **No se aprueba todavía como evaluación final independiente del servicio elegido:** la regla R2 empleó el propio holdout de fase 6 para escoger A4 como servicio por defecto. La regla estaba escrita antes y evita una elección improvisada, pero ese holdout ya es un conjunto de selección; hace falta uno nuevo, intacto, para medir el servicio elegido tras la selección. La fase tampoco es reproducible desde `HEAD` solo: casi todo el código y los informes siguen sin seguimiento en Git. Este dictamen sustituye al veredicto de cierre que figura abajo como historial.

| Gravedad | Hallazgo y evidencia | Estado |
|---|---|---|
| Alta, metodología | `reports/phase6-protocol.md` define R2 sobre `pilot_v3_holdout6_clean` como test y `docs/decisions/0011` adopta A4 usando ese resultado. La NLL A4 − B2 de −0,182 [−0,236; −0,133] es una comparación real y predeclarada, pero no una medida independiente posterior a la elección del servicio. Los 9 pares casi duplicados se declaran y el análisis de sensibilidad conserva el signo. | Pendiente: test nuevo sin usar para decidir, o presentar el holdout actual explícitamente como conjunto de selección. No reutilizarlo para ajustar nada. |
| Alta, entrega | `HEAD=65d4250` contiene esencialmente documentación inicial; `git diff` muestra README y el resto de implementación figura como `??`. Un clon del commit no ejecuta `gso`. Los archivos `artifacts/tree/` y `artifacts/source/` ayudan a auditar esta máquina, pero no sustituyen un commit. | Pendiente de versionar el árbol, respetando la decisión del usuario sobre el commit. |
| Media, alcance | `scripts/profile_lora_step.py` usa cabezales aleatorios, un LR único de 1e-4 y sin scheduler, mientras `configs/pilot_lora_v3.yaml` inicia cabezales desde A2 con LR 5e-4 y scheduler. Los seis pasos reales de E4B con recomputación (132 tensores LoRA con gradiente; driver muestreado 17,4 GB) demuestran esa carga, no el entrenamiento E4B completo ni una duración garantizada de 2,6 h. Sin recomputación, el perfil superó 32 GiB después de 3 pasos. | Aclarado en el docstring del perfilador y README; un entrenamiento completo queda pendiente si se adopta LoRA E4B. |
| Media, calibración | A4 empeora su NLL en este holdout al aplicar las temperaturas estimadas con 300 preguntas: 0,174 → 0,200. La ventaja frente a B2 persiste, pero «calibrado» describe el procedimiento, no una mejora demostrada de calibración fuera de la partición usada para ajustarla. | Pendiente; no ajustar temperaturas con este holdout. |
| Baja, verificación | El perfilador calculaba `all()` sobre la lista de gradientes presentes: una lista vacía habría dado `grads_finite=true`. | Corregido: ahora falla si falta cualquier gradiente LoRA o si alguno no es finito. El bucle de entrenamiento principal ya comprobaba los gradientes faltantes. |

**Pruebas de esta revisión:**

- Código real: `Gemma4Model` de Transformers 5.17, BF16 y MPS, sin `lm_head`, 0 parámetros base entrenables. Recarga de `runs/e4b_experiment/20260923T204945Z/checkpoint` y una pregunta de validación sin caché: 1 forward, logit finito y diferencia máxima frente al guardado **0,0**. Esto prueba esa ruta real; no se repitió el `doctor` completo ni el entrenamiento.
- Datos reales en disco: `pilot_v3` 3000 preguntas; holdout limpio 885; variantes estrecha y K ampliado 135 cada una. `leakage_checks`: 0 errores exactos y 9 pares casi duplicados. Regenerando hechos con semilla 6, ninguna opción añadida coincide con la categoría verdadera; K ampliado en el rango 5–8. Una media directa de los 885 pares de predicciones A4/B2 reproduce `−0,182046917372638`, con 295 grupos y la misma huella de entrada en cada par.
- `pytest tests/unit tests/integration -q`: **287 passed, 2 warnings** (CPU, 32,61 s). Selección focalizada de 30 tests CPU: **30 passed**. `ruff check .`, `ruff format --check .`, `uv lock --check`, `git diff --check` y `py_compile` del perfilador: correctos. **No** se repitió la suite MPS/E2E de 14 pruebas ni el perfil LoRA tras la corrección; sus resultados en el informe son históricos.
- Los dos tar de `artifacts/tree/` se compararon por contenido: 67 archivos añadidos, 9 cambiados y ninguno borrado entre el inicio y el cierre archivado de fase 6. El hash del protocolo actual coincide con `reports/phase6/protocol.sha256`; el hash por sí solo no acredita cuándo se redactó.

**Comandos ejecutados:** `git status --short`, `git diff --stat`, `git diff -- README.md`, `shasum -a 256 -c reports/phase6/protocol.sha256`, `.venv/bin/pytest tests/unit/test_phase6_derive.py tests/unit/test_lora.py tests/unit/test_pooling.py tests/unit/test_review_regressions.py tests/integration/test_phase3_pipeline.py -q`, `.venv/bin/pytest tests/unit tests/integration -q`, `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `uv lock --check`, `git diff --check`, `.venv/bin/python -m py_compile scripts/profile_lora_step.py`, `.venv/bin/python scripts/snapshot_source.py`. Además, pequeñas sondas Python de sólo lectura recargaron A4 sin caché, comprobaron las cuatro colecciones de datos y compararon los tar de `artifacts/tree/`; sus resultados concretos figuran arriba. No se lanzaron entrenamientos, descargas ni servidores.

**Archivos cambiados en esta revisión:** `scripts/profile_lora_step.py`, `README.md` y este `docs/STATUS.md`. El código de modelo, el backend y el contrato público siguen como estaban. Las predicciones, pesos y datasets no se modificaron. Huella de las 77 fuentes actuales: `125c93fe8b0b54abcb9bd33d39c8a0c04af44c5d6c62115607fcfd7e741e0f2a`, archivada en `artifacts/source/`.

**Siguiente paso:** versionar el árbol cuando se decida el commit y reservar un test nuevo para medir A4 como servicio elegido. Si se quiere afirmar viabilidad de entrenamiento LoRA E4B, ejecutar una prueba completa con su configuración real antes de extrapolar los seis pasos.

---

# Registro histórico — fase 6 cerrada; relevo verificado (2026-09-24, 04:20 UTC)

Sustituye a las secciones siguientes, que quedan como registro histórico (relevo y revisión independiente de la fase 5 y sus cierres anteriores).

**Verificación del relevo** (sin cambiar código, datos ni alcance; sin lanzar trabajo pesado):
- `git rev-parse HEAD`: `65d4250…`; rama `main`; `git stash list` vacío; `git status --short` igual que al cierre.
- `.venv/bin/python scripts/snapshot_source.py`: `4fe63db1…` (77 ficheros), el mismo código del cierre.
- `pgrep -fl 'gso|pytest|uvicorn'` y `lsof -nP -iTCP:8000 -sTCP:LISTEN`: código 1 y sin salida. **Ningún proceso del proyecto; puerto libre.**
- Existen los checkpoints A4, B2, A2 y V2 de §3, la calibración de A4 y `artifacts/tree/e7b7a322….tar`. Sólo se comprobó que existen: no se recargaron ni se revalidó su integridad.

**Para el otro asistente:**
- Leer esta sección, `reports/phase6-e4b.md`, `reports/phase6-protocol.md` y las decisiones 0010 y 0011.
- Tarea natural: revisión independiente de la fase 6, con el prompt 2 de `PROMPTS.md`.
- Puntos que conviene comprobar:
  - que el holdout no tiene fugas (`data/derive.py`, `exclude_overlap`);
  - que las opciones añadidas nunca son correctas (`widen_fault_kind`);
  - el relajamiento de `compare_predictions` con otro número de filas;
  - que la recomputación no cambia el forward (sólo marca la capa, no sus submódulos);
  - que la regla R2 se aplicó tal como estaba predeclarada.
- No reutilizar `pilot_v3_holdout6*` ni los tests ya abiertos para decidir nada.

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **6 cerrada** según su criterio («ganancia medida que justifique coste y complejidad»), para la tarea sintética del proyecto. Informe: `reports/phase6-e4b.md`. Protocolo predeclarado `reports/phase6-protocol.md` (sha256 `c745cfea…`, en `reports/phase6/protocol.sha256`), escrito antes de cargar E4B |
| Resultado | **E4B congelado + cabezales (A4) es el servicio de texto recomendado** (decisión 0011). En el holdout nuevo, NLL calibrada: A4 − A2 −0,218 [−0,274; −0,168] y A4 − B2 −0,182 [−0,236; −0,133]. Memoria 17,1 GB; p95 HTTP 1540 ms frente a 909 ms de B2 (límite 1817). Agrupar filas por petición se descarta; la recomputación de activaciones se añade como opción para LoRA (decisión 0010). Con K = 8, la accuracy baja 0,03 en B2 y en A4, pero la NLL de B2 empeora +0,19 |
| Siguiente | Fuera de las fases de la spec. Revisión independiente de la fase 6 y commit a decisión del usuario. Mejoras candidatas en §8 |
| Rama / commit | `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`. **Fases 0–6 sin commit**; stash vacío |
| Referencia del árbol | Tar determinista del árbol completo (incluye tests, documentación e informes) con `scripts/snapshot_tree.py`: `artifacts/tree/9f54e5ef….tar` al empezar la fase (213 ficheros); al cierre, `artifacts/tree/e7b7a322….tar` (280 ficheros; generado justo antes de escribir esta línea en STATUS, que por tanto no incluye). No sustituye a un commit |
| Código | Actual **`4fe63db1a38de5b3e589c10fab96d2d598b228adbe8b4157ff262b9d4fb1c2b4`** (77 ficheros). Todas las huellas usadas en la fase tienen copia en `artifacts/source/` (tabla en el informe). Los benchmarks usaron `2255651d…`, que sólo difiere en un comentario de `configs/serve_e4b_text.yaml` |
| Pruebas | **287 CPU** (32,44 s) y **14 MPS/E2E reales** (128,46 s, 0 omitidos) al cierre. Ruff, formato (125 ficheros), `uv lock --check` y `git diff --check` correctos |

## 2. Procesos activos

- `pgrep -fl 'gso|pytest|uvicorn'` terminó con código 1 y `lsof -nP -iTCP:8000 -sTCP:LISTEN` también: **ningún proceso del proyecto** y el puerto 8000 está libre.
- Los servidores de benchmark terminaron por SIGTERM (−15).
- Sin descargas ni entrenamientos en curso.

## 3. Checkpoints, pesos y datos

| Ruta | Uso |
|---|---|
| `runs/e4b_experiment/20260923T204945Z/checkpoint` + `calibration/calibration-20260923T210339Z.json` | **A4**, E4B + cabezales, época 7/30; servido por `configs/serve_e4b_text.yaml` |
| `runs/pilot_lora_v3/20260923T012208Z/checkpoint` + `calibration-20260923T043504Z.json` | B2, E2B + LoRA; `configs/serve_text.yaml` (opción de menor latencia) |
| `runs/pilot_ce_v3/20260923T005721Z/checkpoint` + `calibration-20260923T043407Z.json` | A2, E2B + cabezales (comparador) |
| `runs/vision2_heads/20260923T161024Z/checkpoint` | V2, servicio visual E2B (`configs/serve_vision.yaml`); E4B con imagen sin medir |
| Caché HF | E2B `3e22461f…` y **E4B `ee0ef602…`** (15,99 GB, descargado una vez; manifiesto `artifacts/manifests/download_ee0ef602….json`) |
| Otros `runs/*/*/checkpoint` | Pruebas de humo, sobreajuste y pilotos de fases 1–4 (`noul_*`, `mixed_*`, `pilot_ce*`, `pilot_rps`, `lora_overfit_v3`, `vision_*`, `vision2_text_only`); evidencia de sus informes, no se sirven |
| `artifacts/source/`, `artifacts/tree/` | Copias por hash del código (todas las huellas de la fase) y del árbol completo (`9f54e5ef…` al empezar, `e7b7a322…` al cierre) |
| `artifacts/cache/representations/` | Representaciones de E4B de train/validation/calibration de `pilot_v3` (clave con la revisión E4B) |
| `data/pilot_v3_holdout6{,_clean,_faultK,_faultK8}` | Holdout de fase 6 **ya usado**: no usarlo para decidir nada nuevo |

**Tests ya usados, que no deben servir para decidir:** `pilot_v3` test, `vision_pilot_v1/v2` test y `pilot_v3_holdout6*`.

## 4. Archivos de esta fase

**Nuevos:**
- código y scripts:
  - `src/gemma_system_one/data/derive.py`;
  - `scripts/{snapshot_tree,derive_phase6_data,measure_request_batching,profile_lora_step}.py`;
- configuración: `configs/{e4b_text,e4b_experiment,serve_e4b_text}.yaml`;
- tests: `tests/unit/test_phase6_{derive,tree_snapshot}.py`;
- documentación e informes:
  - `docs/decisions/0010-phase6-recompute-and-batching.md`, `docs/decisions/0011-e4b-frozen-heads-text-service.md`;
  - `reports/phase6-protocol.md`, `reports/phase6-e4b.md`;
  - `reports/phase6/`: JSON, logs, `run_evals.sh`, `sensitivity/`, `uncalibrated/`.

**Modificados:**
- `src/gemma_system_one/models/lora.py`: `enable_layer_recomputation` y `set_lora_mode` con capas marcadas;
- `src/gemma_system_one/training/lora.py`: `recompute_layers`;
- `src/gemma_system_one/training/lora_pipeline.py`;
- `src/gemma_system_one/metrics.py`: `compare` con otro número de candidatos exige la misma `target_description`;
- tests: `tests/unit/test_lora.py` y `tests/integration/test_phase3_pipeline.py` (un test cada uno);
- `README.md`: sección de la fase 6.

Fuera de Git por `.gitignore`: `data/`, `runs/`, `artifacts/` y `.venv/`.

## 5. Decisiones

- **0010:** recomputación de activaciones como opción, con gradientes idénticos medidos; agrupar filas por petición se rechaza (Δp 0,149, 2 decisiones cambiadas, 1,00×).
- **0011:** A4 como servicio de texto recomendado, por la regla R2 predeclarada.
- 0001–0009 siguen vigentes.

## 6. Comandos exactos

La lista completa está en `reports/phase6-e4b.md` («Comandos ejecutados»). En resumen:
- `generate-data` con semilla 6 y `scripts/derive_phase6_data.py`;
- `gso download` y `gso doctor` de `configs/e4b_text.yaml`;
- `scripts/measure_request_batching.py` y `scripts/profile_lora_step.py` (×4);
- `gso train --config configs/e4b_experiment.yaml` y `gso calibrate`;
- `reports/phase6/run_evals.sh` (10 evaluaciones);
- `gso compare` (validación, holdout, K);
- `gso benchmark` de B2 y A4;
- `pytest` CPU y MPS/E2E.

## 7. Fallos reproducibles encontrados

1. **LoRA sobre E4B supera el presupuesto de 32 GiB** sin recomputación.
   - Reproducción: `scripts/profile_lora_step.py configs/e4b_text.yaml data/pilot_v3 out.json 6` sale con código 2 en el paso 3 (driver 34,5 GB). Se resuelve con `--recompute` (o `train.recompute_layers: true`). No se subió el límite.
2. **`gso compare` rechazaba el control de más opciones** (el índice de la etiqueta cambia con K). Corregido: se exige `target_description` igual con otro número de filas; con el mismo K se sigue exigiendo el índice. Test: `tests/unit/test_phase6_derive.py::test_comparison_with_more_candidates_requires_same_semantic_label`.
3. **La calibración de A4 empeora el holdout** (0,174 sin calibrar frente a 0,200 calibrada). No es un fallo de código, sino un límite metodológico: temperaturas estimadas con 300 preguntas. Pendiente en §8.

## 8. Pendientes, en orden

1. **Revisión independiente de la fase 6** y **commit de las fases 0–6, a decidir por el usuario**.
2. **Calibración de A4:** una partición de calibración mayor (dataset nuevo con split planificado) o un método por primitiva más robusto, decidido en validación. El holdout de fase 6 no debe usarse para ello.
3. **Opcionales:**
   - LoRA sobre E4B con `recompute_layers: true` (unas 2,6 h), sólo si se busca mejorar a A4 en validación;
   - E4B con imagen (perfilar memoria con 266 tokens visuales por fila);
   - Score por tramos y abstención con umbrales fijados en validación.
4. **Riesgos que se mantienen:**
   - el timeout no interrumpe un forward MPS iniciado y la cola no limita las conexiones;
   - el RSS muestreado no es el pico de MPS;
   - datos sintéticos de una familia; V2 sin calibrar;
   - falta la copia histórica `70068e15…`;
   - `reports/doctor/20260922T195301Z.json` (fase 0) contiene una ruta absoluta del usuario.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Rutas:** las rutas de `reports/phase6/` están enmascaradas con `~`.
- **Descargas:** se hicieron sin token HF (aviso de peticiones no autenticadas).

---

# Registro histórico — relevo tras la revisión de la fase 5 (2026-09-23, 20:29 UTC)

**Fase y decisión:** fase 5 revisada y funcional para el prototipo local E2B; fase 6 no iniciada. Siguen vigentes las decisiones 0001–0009, PyTorch/MPS, Gemma 4 E2B y el contrato API. Este relevo sólo actualiza `docs/STATUS.md`; no cambia el alcance ni el código. La revisión técnica vigente figura inmediatamente debajo y en [el informe](../reports/phase5c-review.md).

**Rama/commit y archivos:** `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`, stash vacío. `README.md` está modificado; `.gitignore`, `.python-version`, `configs/`, `docs/STATUS.md`, `docs/decisions/`, `pyproject.toml`, `reports/`, `scripts/`, `src/`, `tests/` y `uv.lock` siguen sin seguimiento. En la revisión previa sólo se añadieron `reports/phase5c-review.md` y texto a STATUS; en este turno sólo se edita STATUS. `git diff --stat` muestra únicamente README porque el resto no está versionado. Huella de código sin cambios: `73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81` (69 fuentes), archivo `artifacts/source/<hash>.tar` existente.

**Resultados anteriores, no repetidos en este relevo:** 21 pruebas CPU de API/benchmark, 49 CPU de máscaras/serialización/grupos/gradientes/checkpoints, 2 E2E reales y 2 MPS reales aprobaron sin omisiones en la revisión independiente. Antes, el cierre completo había pasado 280 CPU y 14 MPS/E2E con el mismo código. Los informes `reports/phase5c/benchmark_{text,vision}.json` registran 100 respuestas 200 por servicio, misma huella cliente/servidor y ráfagas 5 × 200 + 2 × 503; esta verificación sólo comprobó que la huella actual no ha cambiado. No presentar esas pruebas CPU como MPS ni estos resultados como calidad general.

**Fallos reproducibles y riesgos:** no hubo fallo nuevo en la última revisión ni en este relevo. Los fallos ya corregidos de atribución de errores HTTP y cantidades inválidas del benchmark se reproducen con los comandos `pytest ... -k unattributed` y `pytest ... -k invalid_counts` documentados en [phase5b-review](../reports/phase5b-review.md). Pendiente principal: versionar el árbol completo; la huella de fuentes excluye tests/documentación y falta la copia histórica `70068e15…`. El timeout no interrumpe forward MPS, la cola no limita conexiones HTTP, RSS muestreado no es pico de MPS, V2 no está calibrado y los datos son sintéticos. No usar tests ya abiertos para seleccionar nuevos ajustes.

**Procesos y checkpoints:** el filtro por ejecutable/módulo devolvió `project_processes: []`; `lsof -nP -iTCP:8000 -sTCP:LISTEN` no mostró listener (código 1). Existen B `runs/pilot_lora_v3/20260923T012208Z/checkpoint`, su calibración `runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json`, V2 `runs/vision2_heads/20260923T161024Z/checkpoint` y C2 `runs/vision2_text_only/20260923T162937Z/checkpoint`. Aquí se comprobó existencia, no se recargaron pesos ni se revalidó integridad. No queda proceso lanzado por esta revisión.

**Comandos exactos de este relevo:**

```sh
git branch --show-current && git rev-parse HEAD && git status --short && git diff --stat && git stash list
date -u '+%Y-%m-%d %H:%M:%S UTC' && .venv/bin/python -c 'from gemma_system_one.env import source_fingerprint as f; x=f(); print(x["sha256"], x["files"])' && git diff --check
.venv/bin/python - <<'PY'
from pathlib import Path
paths = ('runs/pilot_lora_v3/20260923T012208Z/checkpoint', 'runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json', 'runs/vision2_heads/20260923T161024Z/checkpoint', 'runs/vision2_text_only/20260923T162937Z/checkpoint', 'artifacts/source/73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81.tar')
for raw in paths:
 p=Path(raw)
 print(raw, 'exists' if p.exists() else 'MISSING')
PY
.venv/bin/python - <<'PY'
import os, psutil
found=[]
for p in psutil.process_iter(['pid','name','cmdline']):
 try:
  if p.pid == os.getpid(): continue
  args=p.info.get('cmdline') or []
  exe=os.path.basename(args[0]) if args else ''
  if exe in {'gso','pytest','uvicorn'} or ('-m' in args and any(a in {'gemma_system_one.cli','pytest','uvicorn'} for a in args)):
   found.append({'pid':p.pid,'name':p.info.get('name')})
 except (psutil.NoSuchProcess,psutil.AccessDenied): pass
print('project_processes:',found)
PY
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Resultado: `2026-09-23 20:29:06 UTC`, misma huella de 69 fuentes, cinco rutas presentes, ningún proceso del proyecto ni listener. No se imprimieron argumentos de otros procesos, secretos ni datos privados.

---

# Registro de la revisión independiente de la fase 5 (2026-09-23, 20:26 UTC)

**Veredicto:** la fase 5 supera la puerta funcional de la especificación para este prototipo: API y benchmark HTTP local con checkpoints E2B reales, texto e imagen, sin respuestas simuladas en producción. No se detectó un nuevo fallo de ejecución o metodología que requiera cambiar código. El alcance de los datos sigue siendo sintético y limitado. Detalle: [revisión independiente](../reports/phase5c-review.md).

| Punto | Estado verificado |
|---|---|
| Fase / rama / commit | Fase 5 revisada; fase 6 no iniciada. `main`, `65d425025dd654fb4814062dc42b245062f151b1`. Fases 0–5 aún sin commit |
| Archivos de esta revisión | Nuevo `reports/phase5c-review.md` y esta actualización de `docs/STATUS.md`; sin cambios de código, tests, configs, datos o pesos |
| Código y decisión | Huella `73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81` de 69 fuentes, igual a los dos informes `phase5c`. Se conservan el backend PyTorch/MPS, el modelo E2B, el contrato API y las decisiones 0001–0009 |
| Evidencia nueva CPU | 21 pruebas focalizadas de benchmark/API y 49 de máscaras, serialización, grupos, gradientes y checkpoints; todas aprobaron |
| Evidencia nueva MPS/E2E | 2 E2E reales (B y V2) y 2 pruebas MPS reales (doctor E2B/BF16 y LoRA con gradientes), todas aprobaron, cero omitidas. No se hicieron pasar pruebas CPU por MPS |
| Benchmark previo comprobado | JSON y logs `reports/phase5c/`: 100 respuestas 200 por modalidad; ráfagas de 5 × 200 y 2 × 503 atribuidas por el arnés; 112 registros HTTP por servicio (110 × 200, 2 × 503); hash del cliente y servidor igual al actual. No se repitieron las 100 consultas |
| Riesgos pendientes | El árbol sigue mayoritariamente sin seguimiento; la huella de fuentes excluye tests y docs. RSS muestreado no es pico MPS; timeout no interrumpe forward; cola no limita conexiones; datos sintéticos y V2 sin calibrar; falta archivo de fuentes histórico `70068e15…` |
| Checkpoints | B `runs/pilot_lora_v3/20260923T012208Z/checkpoint` y calibración `runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json`; V2 `runs/vision2_heads/20260923T161024Z/checkpoint`; C2 `runs/vision2_text_only/20260923T162937Z/checkpoint` |

**Reproducción y resultados de esta revisión:**

```sh
.venv/bin/pytest tests/unit/test_phase5_benchmark.py tests/integration/test_phase5_review.py tests/integration/test_api.py -q
# 21 passed, 1 warning, 6,23 s
.venv/bin/pytest tests/e2e/test_api_real.py::test_ready_reports_real_device_and_calibration tests/e2e/test_api_real_vision.py::test_vision_api_matches_evaluation_and_requires_image -q -rs
# 2 passed, 0 skipped, 13,33 s
.venv/bin/pytest tests/mps/test_mps_real.py::test_doctor_real_e2b tests/mps/test_phase3_real.py::test_lora_on_real_e2b_mps -q -rs
# 2 passed, 0 skipped, 15,36 s
.venv/bin/pytest tests/unit/test_pooling.py tests/unit/test_serialization.py tests/unit/test_lora.py tests/unit/test_dataset_split.py tests/unit/test_checkpoint.py -q
# 49 passed, 1 warning, 4,09 s
git diff --check
# correcto
```

No se reprodujeron fallos nuevos. Los fallos reproducibles corregidos en revisiones anteriores, con sus comandos, permanecen en el histórico y en `reports/phase5b-review.md`. El siguiente paso es versionar el árbol completo para cerrar la brecha de trazabilidad antes de iniciar cualquier cambio de fase; no se usan los tests ya abiertos para elegir nuevos ajustes.

**Procesos:** al terminar las pruebas, la inspección por ejecutable/módulo devolvió `project_processes: []`; `lsof -nP -iTCP:8000 -sTCP:LISTEN` terminó sin salida. No quedó servidor, entrenamiento ni test activo de esta revisión.

---

# Registro histórico — fase 5 cerrada; relevo verificado (2026-09-23, 20:23 UTC)

Sustituye a las secciones siguientes, que quedan como registro histórico (segunda revisión de la fase 5, cierre anterior y relevos).

**Verificación del relevo (20:23 UTC, sin cambiar código ni lanzar trabajo pesado):**
- `git rev-parse HEAD` → `65d4250…`; rama `main`, `git stash list` vacío.
- `.venv/bin/python scripts/snapshot_source.py` → `73a72fdc…`, igual que en el cierre.
- `pgrep -fl 'gso|pytest|uvicorn'` y `lsof -nP -iTCP:8000 -sTCP:LISTEN` → código 1, sin salida.

**Para empezar:** leer este estado y `reports/phase5-api.md` §8. Después, o revisar el cierre, o esperar la decisión del usuario sobre el commit y la fase 6. No reabrir tests ya usados.

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **5 cerrada.** El último pendiente de la segunda revisión, una ráfaga real con el arnés corregido y los rechazos atribuidos al servidor lanzado, está ejecutado y es válido |
| Siguiente | Fase 6 (E4B, más opciones, optimización opcional; «ganancia medida que justifique coste y complejidad»). **No iniciada.** Requiere descargar E4B (~16 GB) y perfilar antes de entrenar. Conviene una revisión de este cierre primero |
| Rama / commit | `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`. **Fases 0–5 sin commit**; sin stash ni cambio de rama. El diff de Git sólo muestra `README.md`: el resto está sin seguimiento |
| Código | **`73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81`** (69 ficheros; excluye tests y documentación). Copia en `artifacts/source/73a72fdc….tar`. Es el código de la segunda revisión y el de los benchmarks vigentes. Este cierre no cambió código |
| Pruebas | **280 CPU** (28,05 s) en este cierre. **14 MPS/E2E** (136,46 s, 0 omitidos) en la segunda revisión, con el mismo código; no repetidas. Ruff, formato, `uv lock --check` y `git diff --check` correctos |

**Evidencia del cierre** (`reports/phase5-api.md` §8; `reports/phase5c/benchmark_*.json`):

| Servicio | Medidas | Ráfaga de 7 | Latencia HTTP p50 / p95 | Log del servidor |
|---|---|---|---|---|
| Texto (B, calibrado) | 100 × 200 | 5 × 200 + 2 × 503 `queue_full` con identidad del hijo; `valid: true` | 561 / 940 ms | 112 peticiones: 110 × 200 y 2 × 503 |
| Imagen (V2) | 100 × 200 | Ídem | 2232 / 3248 ms | 112 peticiones: 110 × 200 y 2 × 503 |

- `same_code_as_server: true` en ambos.
- Las latencias vigentes son éstas; las de `reports/phase5b` corresponden a `547b5658…`.

## 2. Procesos activos

- Al cierre, `pgrep -fl 'gso|pytest|uvicorn'` terminó con código 1, sin salida, y `lsof -nP -iTCP:8000 -sTCP:LISTEN` también (código 1, sin salida).
- **Ningún proceso del proyecto.** Los dos servidores de benchmark terminaron por SIGTERM (−15).
- Los procesos auxiliares ajenos (`caffeinate`, servidor MCP de otra herramienta) no se han tocado. Sin descargas ni entrenamientos.

## 3. Checkpoints y artefactos

| Ruta | Uso |
|---|---|
| `runs/pilot_lora_v3/20260923T012208Z/checkpoint` + `calibration/calibration-20260923T043504Z.json` | **B**, servido por `configs/serve_text.yaml` |
| `runs/vision2_heads/20260923T161024Z/checkpoint` | **V2**, servido por `configs/serve_vision.yaml` |
| `runs/vision2_text_only/20260923T162937Z/checkpoint` | C2 (control sin imagen) |
| Fases 3 y 4 originales | Sin cambios |
| Otros `runs/*/*/checkpoint` | Pruebas de humo, sobreajuste y pilotos de fases 1–4 (`noul_*`, `mixed_*`, `pilot_ce*`, `pilot_rps`, `lora_overfit_v3`, `vision_*`), conservados como evidencia de sus informes; ninguno se sirve |
| `reports/phase5c/` | Benchmarks vigentes y logs del servidor (rutas enmascaradas con `~`) |
| `artifacts/source/73a72fdc….tar` | Copia del código vigente |

Pesos base en la caché de HF, una copia, revisión fijada `3e22461f…`.

**Tests usados, que no deben usarse para decidir:** `pilot_v3`, `vision_pilot_v1` y `vision_pilot_v2`.

## 4. Archivos modificados

**Frente a `65d4250`, todo sin commit:**
- modificado: `README.md`;
- sin seguimiento:
  - `.gitignore`, `.python-version`, `pyproject.toml`, `uv.lock`;
  - `configs/`, `scripts/`, `src/`, `tests/`;
  - `docs/STATUS.md`, `docs/decisions/` (0001–0009);
  - `reports/`.

**Excluidos por `.gitignore`, fuera del commit:** `data/`, `runs/`, `artifacts/` y `.venv/`.

**En este cierre** no hubo cambios de código ni de tests. Documentación:

- `reports/phase5-api.md` §8;
- `reports/phase5c/`: `benchmark_{text,vision}.json`, `.stdout`, `.server.log` y `chain.log`;
- esta sección de STATUS.

## 5. Decisiones

Sin decisiones nuevas; 0001–0009 vigentes. La 0009 incluye la cabecera `X-GSO-Instance-ID` de la segunda revisión.

## 6. Comandos exactos de este cierre

```bash
head -30 docs/STATUS.md ; cat reports/phase5b-review.md ; sed -n 263,310p src/gemma_system_one/benchmark.py
.venv/bin/pytest tests/unit/test_phase5_benchmark.py tests/integration/test_api.py -q      # 14 passed
pgrep -fl 'gso|pytest|uvicorn' ; lsof -nP -iTCP:8000 -sTCP:LISTEN                          # libres
caffeinate -i .venv/bin/gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation \
  --requests 100 --warmup 5 --out reports/phase5c/benchmark_text.json
caffeinate -i .venv/bin/gso benchmark --config configs/serve_vision.yaml --dataset data/vision_pilot_v2 --split validation \
  --requests 100 --warmup 5 --out reports/phase5c/benchmark_vision.json
grep -c 'request_id=' reports/phase5c/benchmark_{text,vision}.server.log                 # 112 cada uno
.venv/bin/pytest tests/unit tests/integration -q                                         # 280 passed
.venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check
```

## 7. Fallos reproducibles

No aparecieron fallos nuevos en este cierre. Los de la segunda revisión (identidad no exigida en los errores; cantidades inválidas) están corregidos y cubiertos por `tests/unit/test_phase5_benchmark.py`: `-k unattributed` y `-k invalid_counts`.

## 8. Pendientes, en orden

1. **Revisión de este cierre (opcional pero recomendable)** y **commit de las fases 0–5, a decidir por el usuario**.
2. **Fase 6**, sólo si una ganancia medida lo justifica:
   - perfilar E4B (descarga única) frente a E2B con la misma batería;
   - batching con equivalencia demostrada (decisión 0002);
   - recomputación o checkpointing para LoRA con imagen (K > 5 no probado).
3. **Mejoras candidatas**, sin usar tests ya abiertos:
   - calibrar V2 (partición de calibración, estilos 9–10);
   - Score por tramos;
   - abstención con umbrales fijados en validación.

### Riesgos abiertos

- **Timeout:** no interrumpe un forward MPS ya iniciado.
- **Cola:** limita el trabajo admitido, no las conexiones HTTP.
- **Memoria:** el RSS muestreado no mide el pico de MPS.
- **E2E:** usa ASGI; el socket TCP sólo lo ejercita el benchmark.
- **Trazabilidad:** falta la copia histórica `70068e15…`.
- **Datos y servicio:** datos sintéticos de una sola familia visual; V2 sin calibrar; sin abstención ni batching.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Logs del servidor:** sólo request_id, estado, duración y dimensiones; rutas de avisos enmascaradas.
- **Pendiente anterior:** `reports/doctor/20260922T195301Z.json` (fase 0) contiene una ruta absoluta del usuario.

---

# Registro histórico

## Segunda revisión de la fase 5 y relevo de las 20:03 UTC (histórico; su pendiente quedó cerrado arriba)

**Fase:** segunda revisión independiente de la fase 5. La ejecución funcional está verificada; queda pendiente una ráfaga real con el arnés corregido antes de certificar la atribución de los rechazos por saturación. Fase 6 no iniciada. Este relevo sólo actualiza `docs/STATUS.md`; no cambia código, datos, pesos ni alcance.

**Rama y commit:** `main`, `65d425025dd654fb4814062dc42b245062f151b1`; sin nuevo commit ni stash. `git status --short` sigue mostrando `README.md` modificado y `.gitignore`, `.python-version`, `configs/`, `docs/STATUS.md`, `docs/decisions/`, `pyproject.toml`, `reports/`, `scripts/`, `src/`, `tests/` y `uv.lock` sin seguimiento. Por ello `git diff --stat` sólo muestra `README.md`; se debe revisar el código real. La huella vigente de 69 fuentes es `73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81`, con copia en `artifacts/source/73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81.tar`.

**Archivos de la última revisión:** `src/gemma_system_one/api.py`, `src/gemma_system_one/benchmark.py`, `tests/unit/test_phase5_benchmark.py`, `tests/integration/test_api.py`, `tests/e2e/test_api_real.py`, `docs/decisions/0009-local-api.md`, `reports/phase5b-review.md` y este STATUS. En el presente relevo sólo cambia STATUS. La decisión 0009 documenta la cabecera aditiva `X-GSO-Instance-ID` para atribuir también los errores HTTP; no cambia el modelo, PyTorch/MPS ni el JSON público.

**Resultados comprobados en la última revisión, sin repetirlos en este relevo:** 280 pruebas CPU aprobadas (30,80 s) y 14 MPS/E2E aprobadas (136,46 s, cero omitidas). Las regresiones con transporte simulado son CPU; las E2E usan checkpoints E2B reales mediante ASGI. Ruff, formato, lock y `git diff --check` pasaron. Los benchmarks previos de 100 respuestas 200 por modalidad pertenecen a la fuente `547b5658…`; sus respuestas 503 de ráfaga carecían de atribución verificada. No atribuir esas latencias a la fuente actual.

**Fallos reproducibles:** antes de las correcciones, dos tests `-k unattributed` fallaban porque un HTTP 503 sin identidad contaba como respuesta del hijo; tres tests `-k invalid_counts` fallaban porque cantidades inválidas alcanzaban la carga de configuración. Ahora pasan dentro de la suite CPU. Véase [informe de revisión](../reports/phase5b-review.md). En esta verificación no aparecieron fallos nuevos. El primer filtro de procesos de este relevo dio falsos positivos al detectar el texto de su propio comando; el filtro por ejecutable y módulo que figura abajo devolvió `project_processes: []`.

**Procesos y checkpoints:** no hay proceso `gso`, `pytest`, `uvicorn` ni módulo `gemma_system_one.cli` activo según el filtro corregido; `lsof -nP -iTCP:8000 -sTCP:LISTEN` terminó con código 1 y sin salida. Están presentes `runs/pilot_lora_v3/20260923T012208Z/checkpoint` y su calibración `runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json`, `runs/vision2_heads/20260923T161024Z/checkpoint` y `runs/vision2_text_only/20260923T162937Z/checkpoint`. Se comprobó existencia, no integridad ni nueva recarga. No se lanzaron entrenamientos, servicios ni descargas en este relevo.

**Tareas pendientes:** ejecutar una ráfaga real breve con el arnés actual y revisar identidad en cada error; repetir el benchmark de 100 consultas sólo si se quieren publicar latencias de la fuente actual. Conservar el test reservado para medición, no para elegir ajustes. Persisten los límites documentados de timeout de MPS, memoria muestreada, datos sintéticos y la copia histórica ausente `70068e15…`. No iniciar fase 6 con la atribución pendiente.

**Comandos exactos de este relevo** (las pruebas de la revisión previa figuran inmediatamente después y en el informe):

```sh
git branch --show-current && git rev-parse HEAD && git status --short && git diff --stat && git stash list
date -u '+%Y-%m-%d %H:%M:%S UTC' && .venv/bin/python -c 'from gemma_system_one.env import source_fingerprint as f; x=f(); print(x["sha256"], x["files"])' && git diff --check
.venv/bin/python - <<'PY'
from pathlib import Path
for name in (
    'runs/pilot_lora_v3/20260923T012208Z/checkpoint',
    'runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json',
    'runs/vision2_heads/20260923T161024Z/checkpoint',
    'runs/vision2_text_only/20260923T162937Z/checkpoint',
    'artifacts/source/73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81.tar',
):
    path = Path(name)
    print(path.as_posix(), 'exists' if path.exists() else 'MISSING')
PY
.venv/bin/python - <<'PY'
import os, psutil
found=[]
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        if p.pid == os.getpid(): continue
        args=p.info.get('cmdline') or []
        exe=os.path.basename(args[0]) if args else ''
        is_project=exe in {'gso','pytest','uvicorn'} or ('-m' in args and any(a in {'gemma_system_one.cli','pytest','uvicorn'} for a in args))
        if is_project: found.append({'pid':p.pid,'name':p.info.get('name')})
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
print('project_processes:',found)
PY
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Resultado: `2026-09-23 20:03:58 UTC`, huella `73a72fdc…d14c81` (69 archivos), cinco rutas presentes, `project_processes: []`, puerto 8000 sin listener. No se imprimieron argumentos de procesos ajenos, secretos ni datos privados.

---

# Registro de la segunda revisión independiente de fase 5 (2026-09-23)

Este bloque sustituye el veredicto de cierre anterior, conservado debajo como histórico. **Fase 5 funcional verificada; cierre completo pendiente de una ráfaga real con el arnés corregido. Fase 6 no iniciada.**

- **Rama/commit:** `main`, `65d425025dd654fb4814062dc42b245062f151b1`, sin commit nuevo. Gran parte del árbol sigue sin seguimiento; el diff Git por sí solo no representa la implementación.
- **Fuente final:** `73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81`, 69 ficheros; copia `artifacts/source/73a72fdccf27fb6c2f38af57fd273cff1cdd3d73b0ee6feac447a15a85d14c81.tar`. Excluye tests/documentación. Se conservó también una copia intermedia `cfc31e01…`.
- **Media, corregido:** las respuestas no 200 no tenían atribución verificada. Un 503 sin identidad se contaba como queue_full y la ráfaga figuraba válida. Dos pruebas fallaron antes del arreglo. Se añade cabecera `X-GSO-Instance-ID` a `/v1/decide` y el benchmark la exige en errores; respuestas no atribuibles abortan la medición secuencial o invalidan la ráfaga.
- **Baja, corregido:** `requests <= 0` y `warmup < 0` no se rechazaban antes de preparar el benchmark. Tres pruebas interceptando configuración fallaron antes y ahora pasan; se rechazan sin arrancar servicio.
- **Decisión técnica:** extensión aditiva de cabecera documentada en 0009; sin modificar JSON público, modelo ni backend. Se mantiene alcance de fase 5.
- **Archivos modificados:** `src/gemma_system_one/api.py`, `src/gemma_system_one/benchmark.py`, `tests/unit/test_phase5_benchmark.py`, `tests/integration/test_api.py`, `tests/e2e/test_api_real.py`, `docs/decisions/0009-local-api.md`, este STATUS y nuevo `reports/phase5b-review.md`.
- **Resultados:** suite final **280 CPU passed, 30,80 s**, dos avisos conocidos (Starlette/httpx y float de tensor en test LoRA). **14 MPS/E2E passed, 136,46 s, cero omitidos**, un aviso Starlette. Incluyen Gemma E2B local real y un test de configuración diminuta; los dobles de errores no se presentan como MPS. E2E usa ASGI. Ruff, formato (113 archivos), lock y diff correctos. No se repitió el benchmark de 100 peticiones.
- **Evidencia histórica matizada:** se inspeccionaron los JSON `reports/phase5b/benchmark_*.json`: 100 respuestas 200 por modalidad, mismo hash cliente/servidor `547b5658…`, salida del servidor −15. Las latencias pertenecen a esa fuente. Sus dos 503 por ráfaga no fueron verificados por identidad: no se certifica la afirmación anterior «todas las respuestas». No hay prueba de que provinieran de otro proceso.
- **Procesos al terminar:** filtro psutil de ejecutables del proyecto devuelve `[]`; `lsof -iTCP:8000 -sTCP:LISTEN` retorna 1 sin salida. Las dos sesiones de pytest terminaron. Sin servicios, descargas ni entrenamientos largos lanzados en esta revisión.
- **Checkpoints sin cambios:** B `runs/pilot_lora_v3/20260923T012208Z/checkpoint`, calibración `runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json`; V2 `runs/vision2_heads/20260923T161024Z/checkpoint`; C2 `runs/vision2_text_only/20260923T162937Z/checkpoint`. Los E2E recargan B y V2, no se atribuye a C2 una nueva recarga.
- **Siguiente paso:** verificar una ráfaga real con el arnés actual antes de certificar la atribución de saturación. Repetir 100 consultas sólo si se publican latencias de la fuente nueva. No usar tests abiertos para ajuste. Límites pendientes: timeout no cancela forward, cola no limita conexiones, RSS muestreado no es pico MPS, fuente histórica `70068e15…` ausente, generalización sintética limitada y V2 sin calibrar.

Informe: [reports/phase5b-review.md](../reports/phase5b-review.md). Comandos de pruebas y verificación ejecutados:

```sh
.venv/bin/pytest tests/unit/test_phase5_benchmark.py -k unattributed -q
# Antes: 2 failed, 5 deselected.
.venv/bin/pytest tests/unit/test_phase5_benchmark.py tests/integration/test_api.py -q
# Tras identidad: 11 passed, 5,86 s.
.venv/bin/pytest tests/unit/test_phase5_benchmark.py -k invalid_counts -q
# Antes: 3 failed, 7 deselected.
.venv/bin/pytest tests/unit tests/integration -q
# Intermedia: 277 passed; final: 280 passed.
.venv/bin/pytest tests/mps tests/e2e -q -rs
# 14 passed, sin omisiones.
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv lock --check
git diff --check
.venv/bin/python scripts/snapshot_source.py
git branch --show-current
git rev-parse HEAD
lsof -iTCP:8000 -sTCP:LISTEN
```

---

# Registro histórico — cierre de la fase 5 (2026-09-23, 17:41 UTC; relevo verificado a las 19:49 UTC)

Esta sección sustituye a las siguientes. Debajo, como registro histórico, quedan la revisión independiente de 4b/5 y los relevos anteriores.

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **5 cerrada** (spec §10: «E2E con checkpoint entrenado, fallos claros y benchmark completo»). Se resolvió el hallazgo 6, el único pendiente de la revisión independiente: atribución al servidor lanzado y contabilización completa del benchmark. Benchmarks repetidos con el código corregido |
| Siguiente fase | 6 (E4B, más opciones, optimización). **No iniciada**: primero, revisión independiente de este cierre |
| Rama / commit | `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`. **Fases 0–5 sin commit**; sin stash ni cambio de rama. Gran parte del proyecto está sin seguimiento: revisar los ficheros reales, no sólo `git diff` |
| Identidad del código | **`547b5658f2342c6281db94c90a50cac8db9d2a0065a0c49cc81e17d771e38243`** (69 ficheros de `src`, `configs`, `scripts`, `pyproject.toml` y `uv.lock`; excluye tests y documentación). Copia en `artifacts/source/547b5658….tar`. Es el código de los benchmarks vigentes (`same_code_as_server: true`) y no ha cambiado desde entonces |
| Pruebas | **275 CPU**, que incluyen 7 regresiones de la revisión y 5 nuevas del benchmark. **14 MPS + E2E reales** en 131,6 s, 0 omitidos. Ruff, formato, `uv lock --check` y `git diff --check` correctos |
| Dependencias | Sin cambios desde la fase 5 (fastapi 0.141.1, uvicorn 0.53.0, starlette 1.7.0, httpx 0.28.1; torch 2.14.0, transformers 5.17.0, peft 0.21.0) |

**Resultados vigentes** (`reports/phase5-api.md` §7; `reports/phase5b/benchmark_*.json`):

| Servicio | Respuestas | Latencia HTTP p50 / p95 | Ráfaga de 7 |
|---|---|---|---|
| Texto (B de fase 3, calibrado) | 100 × 200 | 552 / 927 ms | 5 × 200 + 2 × 503 `queue_full`, válida |
| Imagen (V2) | 100 × 200 | 2231 / 3247 ms | 5 × 200 + 2 × 503 `queue_full`, válida |

- Identidad del servidor hijo comprobada en todas las respuestas.
- E2E: la API devuelve exactamente la respuesta de `gso evaluate` en texto y con imagen (sin cambios).
- Cierre de la fase 4 (V2 frente a C2 con estilos no vistos): sin cambios; ver `reports/phase4b-styles.md`.

## 2. Procesos activos

- En el relevo (19:49 UTC), `pgrep -fl 'gso|gemma_system_one|pytest|uvicorn'` terminó con código 1, sin salida. `lsof -iTCP:8000 -sTCP:LISTEN` terminó con código 1, sin salida (puerto libre). Lo mismo se había comprobado al cierre.
- **Auxiliares ajenos, que no se han tocado** (inspección con `psutil` de nombre y proceso padre, sin argumentos):
  - un `caffeinate` de una terminal del usuario;
  - otro `caffeinate` y un shell, ambos del cliente Claude Code;
  - un servidor MCP en Python lanzado por `uv` desde Codex.
- Ningún proceso del proyecto; ningún servidor. Los servidores de benchmark terminaron por SIGTERM (−15).
- El proceso auxiliar que ocupó el puerto 8000 en la reproducción se detuvo (`kill`), y se comprobó el puerto libre.
- No se han tocado procesos ajenos. Sin descargas.

## 3. Checkpoints y artefactos

| Ruta | Uso |
|---|---|
| `runs/pilot_lora_v3/20260923T012208Z/checkpoint` + `calibration/calibration-20260923T043504Z.json` | **B**, servido por `configs/serve_text.yaml` |
| `runs/vision2_heads/20260923T161024Z/checkpoint` | **V2**, servido por `configs/serve_vision.yaml` |
| `runs/vision2_text_only/20260923T162937Z/checkpoint` | C2 (control) |
| Fases 3 y 4 originales | Sin cambios (ver §3 histórico) |
| `reports/phase5b/` | Benchmarks vigentes, logs del servidor (rutas enmascaradas) y `port_busy_check.log` |
| `artifacts/source/` | 8 copias de fuentes:<br>• `aa718528…`: cierre de la fase 4 original<br>• `f414efce…`: revisión de la fase 4<br>• `457c66c3…`: primera copia tras añadir `snapshot_sources`<br>• `7f645568…`: inicio de V2<br>• `78d02416…`: V2 al guardar, C2 y evaluaciones de 4b<br>• `910a39a4…`: primera versión de la fase 5<br>• `ac7f751b…`: revisión de la fase 5<br>• **`547b5658…`: vigente, código de los benchmarks actuales**<br>No existe la copia `70068e15…` |

Pesos base, una sola copia en la caché de HF, revisión fijada `3e22461f…`. **Tests ya usados, que no deben usarse para decidir:** `pilot_v3`, `vision_pilot_v1` y `vision_pilot_v2`.

## 4. Archivos de este cierre

- **Código:**
  - `src/gemma_system_one/benchmark.py`:
    - `port_is_free`, `check_identity`, `IdentityError`;
    - `measure` (secuencial, cuenta todo);
    - `run_burst` (cuenta errores de transporte y respuestas ajenas);
    - `_wait_ready` con identidad;
    - copia de fuentes del cliente y `same_code_as_server`.
  - `src/gemma_system_one/api.py`: `SERVER_INSTANCE_ENV`; `instance_id`, `pid` y `code_sha256` en `ready` (incluso en 503); `instance_id`/`pid` en `metadata`.
- **Tests:**
  - nuevo `tests/unit/test_phase5_benchmark.py` (5 casos);
  - `tests/integration/test_api.py` y `tests/e2e/test_api_real.py`: identidad.
- **Documentación:** `docs/decisions/0009-local-api.md` (sección «Cierre del hallazgo 6»), `reports/phase5-api.md` §7 y esta sección.

## 5. Decisiones

Sin decisiones de arquitectura nuevas: 0001–0009 vigentes.

## 6. Comandos exactos de este cierre

```bash
cat reports/phase5-review.md ; sed -n 70,260p src/gemma_system_one/benchmark.py
.venv/bin/pytest tests/integration/test_phase5_review.py -q        # 1 fallo (Popen interceptado atrapaba git) → 7 passed tras usar snapshot_sources
.venv/bin/pytest tests/unit/test_phase5_benchmark.py -q            # 5 passed
.venv/bin/pytest tests/unit tests/integration -q                   # 275 passed, 27,75 s
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -q -rs          # 14 passed, 131,59 s
# Reproducción real: un proceso Python escuchando en 127.0.0.1:8000 (socket.listen), y:
.venv/bin/gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation --requests 3 --warmup 1 \
  --out reports/phase5b/port_busy_check.json                       # exit 1: «ya está en uso»; sin servidor ni informe
.venv/bin/python scripts/snapshot_source.py                        # 547b5658… (69)
caffeinate -i .venv/bin/gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation \
  --requests 100 --warmup 5 --out reports/phase5b/benchmark_text.json
caffeinate -i .venv/bin/gso benchmark --config configs/serve_vision.yaml --dataset data/vision_pilot_v2 --split validation \
  --requests 100 --warmup 5 --out reports/phase5b/benchmark_vision.json
.venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check
pgrep -fl 'gso|pytest|uvicorn' ; lsof -iTCP:8000 -sTCP:LISTEN
```

## 7. Fallos reproducibles de este cierre

| Fallo | Reproducción | Estado |
|---|---|---|
| El benchmark podía medir a otro servidor en el puerto (hallazgo 6a) | Ocupar 127.0.0.1:8000 y ejecutar `gso benchmark` | Rechazo previo sin lanzar el servidor + identidad en cada respuesta; test y reproducción real |
| Errores de transporte de la ráfaga sin registrar (hallazgo 6b) | `test_burst_records_transport_errors_and_foreign_answers` (transporte simulado) | Cada petición cuenta y las sumas se verifican |
| Mi primera versión llamaba a `git_state()` (subprocesos) antes de lanzar el servidor, y un test que intercepta `Popen` lo detectó | `tests/integration/test_phase5_review.py::test_benchmark_creates_output_directory_before_starting_server` | El benchmark usa `snapshot_sources()`, sin subprocesos |

## 8. Pendientes, en orden

1. **Revisión independiente de este cierre**: benchmark (`port_is_free`, identidad, contabilización) e identidad en la API.
2. **Commit o rama con las fases 0–5, a decidir por el usuario.**
3. **Fase 6** (spec §10), sólo con una ganancia medida que justifique coste y complejidad:
   - E4B (descarga única y perfilado de memoria antes de entrenar);
   - batching con equivalencia demostrada (decisión 0002);
   - recomputación o checkpointing para LoRA con imagen.
4. **Mejoras candidatas**, sin usar tests ya abiertos:
   - calibrar V2;
   - Score por tramos;
   - abstención.

### 8.1 Relevo (19:49 UTC; sólo verificación; sólo cambia este documento)

```bash
date -u ; git rev-parse HEAD ; git branch --show-current ; git status --short ; git stash list | wc -l
find src tests scripts configs docs reports README.md pyproject.toml uv.lock -newer docs/STATUS.md -type f   # nada
pgrep -fl 'gso|gemma_system_one|pytest|uvicorn' ; lsof -iTCP:8000 -sTCP:LISTEN                            # ambos código 1, sin salida
.venv/bin/python -c "from gemma_system_one.env import source_fingerprint as f; d=f(); print(d['sha256'], d['files'])"   # 547b5658…, 69
ls artifacts/source                                                   # 8 copias (§3)
.venv/bin/pytest tests/unit tests/integration -q                      # 275 passed, 27,96 s
.venv/bin/pytest --collect-only -q tests/mps tests/e2e                # 14 recogidos; no ejecutados (código sin cambios)
.venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check   # correctos
shasum -a 256 -c reports/phase4b/test-protocol.sha256                 # OK
```

- Existen los checkpoints de §3: B con su calibración, V2, C2, V de la fase 4 y la referencia congelada de la fase 3.
- No se recargaron pesos ni se recalcularon hashes de checkpoints: esa evidencia es la del cierre y la de la revisión.

### Riesgos abiertos

- **Timeout:** no interrumpe un forward MPS ya iniciado.
- **Cola:** limita el trabajo admitido, no las conexiones HTTP abiertas.
- **Memoria:** el RSS muestreado no mide el pico de memoria unificada/MPS.
- **E2E:** usa ASGI; el socket TCP sólo lo ejercita el benchmark.
- **Trazabilidad:** falta la copia histórica `70068e15…`.
- **Datos y servicio:**
  - datos sintéticos y una sola familia visual;
  - V2 sin calibrar;
  - sin abstención ni batching.

### Privacidad

- **Contenido:** sin secretos ni datos personales.
- **Logs:** los del servidor sólo registran request_id, estado, duración y dimensiones; las rutas absolutas de avisos de bibliotecas se enmascararon con `~`.
- **Pendiente anterior:** `reports/doctor/20260922T195301Z.json` (fase 0) contiene una ruta absoluta del usuario.

---

# Registro histórico

## Revisión independiente de 4b y 5 (histórico; su pendiente quedó cerrado arriba)

Esta sección sustituye el veredicto del relevo anterior conservado debajo como registro histórico.

- **Fase:** revisión de 4b y 5. Ejecución funcional real verificada tras correcciones; **cierre completo pendiente** por robustez del benchmark (atribución al proceso hijo y contabilización de errores de transporte). No iniciar fase 6 con esta revisión como aprobación sin reservas.
- **Rama/commit:** `main`, `65d425025dd654fb4814062dc42b245062f151b1`; sin commit nuevo. Gran parte del proyecto sigue sin seguimiento; revisar código real además del diff.
- **Fuente actual:** `ac7f751bc509025441ac23f9a581be0728fbf3f0f0087e5db42353cff6d92292`, 69 archivos, copia `artifacts/source/ac7f751bc509025441ac23f9a581be0728fbf3f0f0087e5db42353cff6d92292.tar`. Excluye documentación/tests. `910a39a4…` identifica sólo el cierre y benchmark anteriores.
- **Correcciones:** rechazo de semilla distinta al split visual planificado y validación al cargarlo; decodificación de imagen dentro de cola/timeout; cancelación del futuro y cierre PIL; JSON profundo → 422; errores sin eco del cliente; warmup para primitiva entrenada; directorio del log de benchmark creado antes de abrirlo.
- **Archivos modificados en esta revisión:** `src/gemma_system_one/{api.py,engine.py,benchmark.py,data/split.py}`, nuevo `tests/integration/test_phase5_review.py`, `docs/decisions/0009-local-api.md`, este STATUS y `reports/phase5-review.md`.
- **Decisión:** mantener modelo, PyTorch/MPS, alcance y contrato público. Detalle de gravedad, reproducciones, comandos exactos y límites en [revisión independiente](../reports/phase5-review.md).
- **Pruebas:** 263 CPU antes; 269 CPU después de seis regresiones; subconjunto final de siete regresiones aprobado (incluye una adicional). **14 MPS/E2E aprobados, cero omitidos, 132,36 s**, con pesos locales. No presentar los dobles CPU de timeout/errores como MPS. Ruff, formato, lock y diff correctos. No se repitió benchmark de 100 consultas con el código corregido.
- **Procesos al terminar:** inspección con psutil filtrando ejecutables Python/gso/pytest/uvicorn del proyecto: lista vacía; `lsof -iTCP:8000 -sTCP:LISTEN` sin salida. Las sesiones de tests terminaron. No se lanzó entrenamiento largo, descarga ni servicio externo. No se afirma nada sobre procesos ajenos.
- **Checkpoints conservados:** V2 `runs/vision2_heads/20260923T161024Z/checkpoint`; C2 `runs/vision2_text_only/20260923T162937Z/checkpoint`; B `runs/pilot_lora_v3/20260923T012208Z/checkpoint`, calibración `runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json`. Ninguno modificado. Pesos base en caché local, revisión fijada; no redescargar.
- **Pendientes del relevo:** corregir y probar identidad del proceso y fallos de transporte del benchmark; repetir mediciones sólo si se pretende publicar rendimiento del código actual. No reutilizar test para elegir parámetros. Mantener explícitos límites del timeout (no cancela forward), memoria muestreada y copia histórica ausente `70068e15…`. La evidencia de estilos v2 y los archivos fuente `7f645568…`/`78d02416…` sí se comprobaron.

Comandos principales ejecutados (resultados y fallos iniciales en el informe):

```sh
.venv/bin/pytest tests/unit tests/integration -q
.venv/bin/pytest tests/integration/test_phase5_review.py -q
.venv/bin/pytest tests/integration/test_phase5_review.py -k planned_seed -q
.venv/bin/pytest tests/integration/test_phase5_review.py tests/integration/test_api.py -q
.venv/bin/pytest tests/mps tests/e2e -q -rs
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv lock --check
git diff --check
.venv/bin/python scripts/snapshot_source.py
git status --short
git branch --show-current
git rev-parse HEAD
lsof -iTCP:8000 -sTCP:LISTEN
```

## Relevo comprobado a las 17:27 UTC (2026-09-23)

En este turno **sólo se modifica `docs/STATUS.md`**. No se repiten tests, benchmark ni entrenamientos; los resultados anteriores pertenecen a la revisión inmediatamente anterior. Rama, HEAD y fingerprint permanecen iguales. Se confirmó la existencia de los tres directorios de checkpoint y del JSON de calibración indicados arriba; esta comprobación no es una nueva validación de integridad ni recarga.

**Procesos:** ningún proceso del proyecto detectado por el filtro de abajo; puerto 8000 sin listener (`lsof` retorna 1, sin salida). No quedan procesos lanzados por este asistente. No se listan argumentos de procesos ajenos, variables de entorno, credenciales ni contenidos de datos.

**Fallos reproducibles conservados** en `tests/integration/test_phase5_review.py` (ahora se espera que pasen):

| Caso | Antes de corregir | Resultado esperado actual |
|---|---|---|
| Split v2 planificado con seed 0, solicitar seed 1 | Aceptaba cambio | `SplitError` |
| Decodificador de 150 ms, timeout 30 ms | HTTP 200 fuera del plazo | HTTP 503 `timeout`, decodificador en worker |
| JSON con 2000 niveles | HTTP 500 | HTTP 422 |
| Discriminador inválido con marcador sintético | Repetía el marcador | Respuesta sin ese valor |
| Warmup de checkpoint sólo Choice / sólo Score | Fallaban ambos | Usa una primitiva entrenada |
| Benchmark hacia directorio inexistente | Apertura del log fallaba | Directorio y log creados antes de Popen |

Los seis primeros casos (Choice/Score cuentan por separado) se observaron fallar antes del arreglo. El caso del directorio se probó después con Popen interceptado; no se presenta como reproducción previa ni arranque real. Reejecutar: `.venv/bin/pytest tests/integration/test_phase5_review.py -q`.

**Orden de continuación, dentro de fase 5:**

1. Leer AGENTS, especificación, auditoría, este estado y `reports/phase5-review.md`; inspeccionar `src/gemma_system_one/benchmark.py`.
2. Corregir y probar atribución de respuestas al proceso lanzado y contabilización de errores en `_burst`. Ambos pendientes proceden de inspección, no de una reproducción con servicios.
3. Ejecutar las pruebas pertinentes y actualizar evidencia. Sólo repetir benchmark real si se requiere publicar latencias del código corregido; no iniciar fase 6 ni usar test para decisiones de ajuste.
4. Conservar pesos y checkpoints actuales, no redescargar ni subir artefactos privados. No se ha creado commit en este relevo.

### Comandos exactos del relevo

Inspección inicial:

```sh
pwd
git branch --show-current
git rev-parse HEAD
git status --short
sed -n '1,85p' docs/STATUS.md
cat AGENTS.md
```

Comprobación de fuentes, existencia de artefactos y procesos (sin imprimir argumentos):

```sh
.venv/bin/python - <<'PY_CHECK'
from datetime import datetime, timezone
from pathlib import Path
import os
import psutil
from gemma_system_one.env import source_fingerprint
print('verified_utc:', datetime.now(timezone.utc).isoformat())
print('source:', source_fingerprint())
for name in ('runs/vision2_heads/20260923T161024Z/checkpoint', 'runs/vision2_text_only/20260923T162937Z/checkpoint', 'runs/pilot_lora_v3/20260923T012208Z/checkpoint', 'runs/pilot_lora_v3/20260923T012208Z/calibration/calibration-20260923T043504Z.json'):
    print(name, 'exists:', Path(name).exists())
found=[]
for p in psutil.process_iter(['pid','name','cmdline']):
    if p.pid == os.getpid():
        continue
    name=p.info['name'] or ''
    if 'python' in name.lower() or name in ('gso','uvicorn','pytest'):
        args=p.info['cmdline'] or []
        if any('gemma_system_one' in a or a.endswith('/gso') or a.endswith('/pytest') or a == 'uvicorn' for a in args):
            found.append({'pid':p.pid,'name':name})
print('project_processes:', found)
PY_CHECK
lsof -iTCP:8000 -sTCP:LISTEN
git diff --check
```

Resultado: fingerprint `ac7f751b…d92292`, cuatro rutas existentes, `project_processes: []`; sin listener. El delimitador del heredoc se llama aquí `PY_CHECK` para separarlo del script de edición; el cuerpo es el ejecutado. `git diff --check` no valida los archivos sin seguimiento, incluido STATUS: se revisa también el contenido del documento.

---

# Registro histórico — estado del trabajo — relevo al siguiente asistente

Actualizado: 2026-09-23, 17:12 UTC. Relevo verificado a las 17:15 UTC sin cambios de código, datos ni artefactos; sólo cambia este documento (§6.1).

**Resuelto en esta sesión:**

- **Cierre de la fase 4:** estilos separados por partición y trazabilidad de fuentes, los pendientes de la revisión independiente.
- **Fase 5 completada:** API local con E2E reales y benchmark.

**Pendientes:** revisión independiente del cierre de la fase 4 y de la fase 5; commit, a decidir por el usuario.

| Documento | Uso |
|---|---|
| [reports/phase5-api.md](../reports/phase5-api.md) | Informe de la fase 5 (referencia actual) |
| [reports/phase4b-styles.md](../reports/phase4b-styles.md) | Cierre de la fase 4 (estilos por partición, test nuevo) |
| [reports/phase4b-test-protocol.md](../reports/phase4b-test-protocol.md) | Protocolo predeclarado del test de cierre (sha256 `05901ce5…fb95`, sin desviaciones) |
| [0008](decisions/0008-vision-styles-per-split-and-source-archives.md), [0009](decisions/0009-local-api.md) | Decisiones nuevas |
| [reports/phase4-vision.md](../reports/phase4-vision.md), [reports/phase4-review.md](../reports/phase4-review.md), [0007](decisions/0007-single-image-path.md) | Fase 4 original y su revisión |
| [reports/phase3-lora.md](../reports/phase3-lora.md), [reports/phase3-review.md](../reports/phase3-review.md), [0005](decisions/0005-lora-stage.md), [0006](decisions/0006-generator-v3-access-policy.md) | Fase 3 |

## 1. Fase, rama y veredicto

| Campo | Estado |
|---|---|
| Fase | **5 completada** (spec §10: «E2E con checkpoint entrenado, fallos claros y benchmark completo»). **Pendientes de la revisión de fase 4, resueltos**: estilos por partición con un test nuevo; copia de fuentes automática (con el límite histórico de `70068e15…`) |
| Siguiente fase | 6: E4B, más opciones y optimización opcional, «ganancia medida que justifique coste y complejidad». Antes, la revisión independiente |
| Rama / commit | `main`, HEAD `65d425025dd654fb4814062dc42b245062f151b1`. **Fases 0–5 sin commit**; sin stash ni cambio de rama. No hago commit sin petición del usuario |
| Identidad del código | Al cierre: **`910a39a4eab76542e9e4c98bc6208ed427d831b65d767ff14c6c2ea9d2a212b7`** (69 ficheros de `src`, `configs`, `scripts`, `pyproject.toml` y `uv.lock`). Copia: `artifacts/source/910a39a4….tar`. Es el código con que se ejecutaron los E2E y los benchmarks: no se cambió desde entonces |
| Árbol de trabajo | Modificado: `README.md`. Sin seguimiento: `.gitignore`, `.python-version`, `configs/`, `docs/STATUS.md`, `docs/decisions/`, `pyproject.toml`, `reports/`, `scripts/`, `src/`, `tests/`, `uv.lock` |
| Pruebas | **263 CPU**, **10 tests/mps** y **4 tests/e2e** con E2B real, 0 omitidos. Ruff, formato, `uv lock --check` y `git diff --check` correctos |
| Dependencias nuevas | `fastapi` 0.141.1, `uvicorn` 0.53.0 y `starlette` 1.7.0 (transitiva); `httpx` 0.28.1 ya estaba. torch 2.14.0, transformers 5.17.0 y peft 0.21.0 sin cambios |

### Cierre de la fase 4 (test nuevo, estilos 11–12 no vistos; predeclarado)

| Modelo | NLL en test |
|---|---|
| **V2**, con imagen | **0,415** |
| **C2**, mismas filas sin imagen | 1,192 |
| Prior | 1,085 |

- V2 − C2 = **−0,778 [−0,886; −0,660]**.
- Con la imagen de otro grupo, la NLL de V2 sube a 3,47 y la accuracy baja de 0,848 a 0,352.
- Se cumple la regla predeclarada: **uso visual con estilos no vistos**, dentro de la misma familia de gráficos de barras.
- Recarga desde el texto+imagen: Δlogit 0,0.

### Fase 5

- **E2E con E2B real:** la API devuelve **exactamente** la respuesta de `gso evaluate` para las mismas preguntas, en texto (B de fase 3, calibrado) y con imagen (V2). Además pasan las pruebas de permutación, pregunta independiente, cambio de pregunta, 404, modalidad y `usage` real.
- **Benchmark** (100 consultas medidas por servicio, servidor en otro proceso):

  | Servicio | Latencia HTTP p50 / p95 | Arranque hasta `ready` |
  |---|---|---|
  | Texto | 554 / 911 ms | 4,3 s |
  | Imagen | 2231 / 3242 ms | 4,8 s |

  En la ráfaga de 7 peticiones simultáneas: 5 × 200 y 2 × 503 `queue_full`.

## 2. Procesos activos

Verificado de nuevo en el relevo (17:15 UTC):

- `pgrep -fl 'gso|gemma_system_one|pytest|uvicorn'` terminó con código 1, sin salida.
- `lsof -iTCP:8000 -sTCP:LISTEN` no muestra ningún proceso escuchando.
- **Ningún proceso del proyecto** (entrenamiento, evaluación, servidor ni tests); no hay pesos cargados. Los servidores de benchmark se detuvieron con SIGTERM al terminar (código de salida −15).
- **Auxiliares ajenos, que no se han tocado** (inspección con `psutil` de nombre y proceso padre, sin argumentos):
  - un `caffeinate` de una terminal del usuario;
  - otro `caffeinate` y un shell, ambos del cliente Claude Code;
  - un servidor MCP en Python lanzado por `uv` desde Codex.

  Los PID son una instantánea y no se registran.
- No hubo descargas de pesos. `uv add` sí descargó los paquetes de servicio (fastapi, uvicorn, starlette).

## 3. Pesos, datos, checkpoints y artefactos (ignorados por Git)

| Ruta | Contenido |
|---|---|
| `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f…/` | Pesos base, una sola copia. **No volver a descargar** |
| `data/vision_pilot_v2` (+ split y `audit.jsonl`) | **Cierre de fase 4**: 2100 preguntas, 700 imágenes; estilos por partición (train 0–3, 5–6 · val 7–8 · cal 9–10 · test 11–12). sha256 `09b1d190…f604` |
| `data/vision_transfer_v2` | 450 preguntas con el estilo 4 (diagnóstico) |
| **`runs/vision2_heads/20260923T161024Z/checkpoint`** | **V2**: `decision_heads` v3, con imagen, época 30/30. Lo sirve `configs/serve_vision.yaml` |
| `runs/vision2_text_only/20260923T162937Z/checkpoint` | **C2**: control sin imagen |
| `runs/vision2_*/evaluations/test-*` | **Test de cierre usado** (una vez por artefacto). No reutilizar |
| **`runs/pilot_lora_v3/20260923T012208Z/checkpoint`** + `calibration/calibration-20260923T043504Z.json` | **B** (fase 3, LoRA calibrado). Lo sirve `configs/serve_text.yaml` |
| `runs/vision_heads/…`, `runs/vision_text_only/…`, `data/vision_pilot_v1` | Fase 4 original (estilos compartidos). Test v1 usado; sus fuentes `70068e15…` no se conservaron |
| `runs/pilot_ce_v3/…`, `runs/pilot_lora_v3/…`, `data/pilot_v3` | Fase 3 |
| `artifacts/source/*.tar` | Copias de fuentes por hash:<br>• `aa718528…`: cierre de la fase 4 original<br>• `f414efce…`: revisión de la fase 4<br>• `457c66c3…`: primera copia tras añadir `snapshot_sources`<br>• `7f645568…`: inicio de V2, el código que ejecutó<br>• `78d02416…`: V2 al guardar, C2 y evaluaciones de cierre de la fase 4<br>• `910a39a4…`: cierre de la fase 5, E2E y benchmarks<br>No existe la copia `70068e15…` (V/C de la fase 4 original) |
| `reports/phase4b/`, `reports/phase5/` | Resúmenes CLI, comparaciones, benchmarks y logs del servidor (rutas enmascaradas con `~`) |

## 4. Archivos de esta sesión

**Nuevos:**

- Código:
  - `src/gemma_system_one/engine.py`
  - `src/gemma_system_one/api.py`
  - `src/gemma_system_one/benchmark.py`
- Configuraciones: `configs/{vision2_heads,vision2_text_only,serve_text,serve_vision}.yaml`
- Tests:
  - `tests/integration/test_api.py`
  - `tests/e2e/test_api_real.py` y `tests/e2e/test_api_real_vision.py`
- Documentación:
  - `docs/decisions/0008-…`, `0009-local-api.md`
  - `reports/phase4b-test-protocol.md`, `reports/phase4b-styles.md`, `reports/phase5-api.md`
  - `reports/phase4b/`, `reports/phase5/`

**Modificados:**

- **Código:**
  - `data/generate_vision.py`: v2 con split planificado, estilos 5–12 y `STYLE_POOLS_V2`; v1 intacto.
  - `data/split.py`: `check_planned_split`.
  - `env.py`: `snapshot_sources`; `git_state(snapshot=…)`; `GSO_SOURCE_ARCHIVE_DIR`.
  - `checkpoint.py`: el manifiesto guarda la copia de fuentes.
  - `images.py`: `decode_image_bytes` e `ImageTooLargeError`.
  - `features.py`: acepta imágenes ya decodificadas.
  - `training/decisions_pipeline.py`:
    - `load_decision_model`;
    - `prepare_evaluation` la reutiliza;
    - `input_modality`, `run_start_code`;
    - `code` con copia.
  - `training/lora_pipeline.py`: `input_modality`, `session_code`.
  - `calibration.py`: `code` con copia.
  - `cli.py`:
    - `--vision-version`, `--split-seed`;
    - `gso split` verifica el plan;
    - `serve`, `benchmark`.
  - `scripts/snapshot_source.py`: delega en `env`.
  - `pyproject.toml` y `uv.lock`.
- **Tests:**
  - `tests/conftest.py`: copia de fuentes a un directorio temporal.
  - `tests/unit/test_phase3_followups.py`: copia reproducible y manifiesto con copia.
  - `tests/unit/test_phase4_vision_data.py`: estilos por partición y v1 intacto.
  - `tests/integration/test_phase4_pipeline.py`: modalidad y código de inicio.
- **Documentación:**
  - `README.md` (fase 5 y comandos de visión con versión explícita);
  - `reports/phase4-vision.md` (nota sobre `--vision-version v1`);
  - este archivo.

## 5. Decisiones vigentes

- **0008:**
  - datos visuales v2 con estilos disjuntos por partición y flujo aleatorio propio; el split se planifica con `assign_groups` y `gso split` lo verifica;
  - copia automática de las fuentes en cada run, checkpoint, evaluación y calibración;
  - límite histórico: la copia `70068e15…` no existe.
- **0009:**
  - API v1 con un worker y cola acotada (503 si está llena o se agota el tiempo);
  - `ready` sólo tras un warmup real;
  - límites previos a parsear y decodificar;
  - motor compartido con la evaluación (`load_decision_model`, `expand`, `extract_pooled`, `reconstruct`);
  - modalidad fija por checkpoint;
  - `usage` real y logs sin contenido.
- **Se mantienen:** 0001–0007.

## 6. Comandos exactos de esta sesión (en orden)

Detalle en `reports/phase4b-styles.md` y `reports/phase5-api.md`.

```bash
.venv/bin/python scripts/snapshot_source.py         # copias de fuentes
shasum -a 256 reports/phase4b-test-protocol.md > reports/phase4b/test-protocol.sha256     # antes de generar
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --vision-version v2 --out data/vision_pilot_v2 --cases 700 --seed 0 --split-seed 0
PYTHONHASHSEED=1 .venv/bin/gso generate-data --kind vision --vision-version v2 --variant transfer --out data/vision_transfer_v2 --cases 150 --seed 0
.venv/bin/gso validate-data --dataset data/vision_pilot_v2 ; .venv/bin/gso validate-data --dataset data/vision_transfer_v2
.venv/bin/gso split --dataset data/vision_pilot_v2 --seed 0
caffeinate -i .venv/bin/gso train --config configs/vision2_heads.yaml
caffeinate -i .venv/bin/gso train --config configs/vision2_text_only.yaml
uv add "fastapi>=0.120" "uvicorn>=0.38" "httpx>=0.28"
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_heads/20260923T161024Z/checkpoint --split validation --no-cache --vision-ablation --robustness --baselines
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_heads/20260923T161024Z/checkpoint --split test --final-test --no-cache --baselines --vision-ablation
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_text_only/20260923T162937Z/checkpoint --split test --final-test --no-cache --baselines
.venv/bin/gso compare --a runs/vision2_text_only/20260923T162937Z/evaluations/test-20260923T164713Z-predictions.jsonl \
  --b runs/vision2_heads/20260923T161024Z/evaluations/test-20260923T164638Z-predictions.jsonl --allow-different-inputs \
  --out reports/phase4b/test_compare_vision_minus_text.json
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_heads/20260923T161024Z/checkpoint --split all --dataset data/vision_transfer_v2 --no-cache --baselines --vision-ablation
caffeinate -i .venv/bin/gso evaluate --checkpoint runs/vision2_text_only/20260923T162937Z/checkpoint --split all --dataset data/vision_transfer_v2 --no-cache
.venv/bin/pytest tests/unit tests/integration -q                       # 263 passed
caffeinate -i .venv/bin/pytest tests/mps tests/e2e -v -rs              # 10 + 2 passed; 2 fallos de aserción de dispositivo (corregidos)
caffeinate -i .venv/bin/pytest tests/e2e -v -rs                        # 4 passed
caffeinate -i .venv/bin/gso benchmark --config configs/serve_text.yaml --dataset data/pilot_v3 --split validation --requests 100 --warmup 5 --out reports/phase5/benchmark_text.json
caffeinate -i .venv/bin/gso benchmark --config configs/serve_vision.yaml --dataset data/vision_pilot_v2 --split validation --requests 100 --warmup 5 --out reports/phase5/benchmark_vision.json
.venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check
.venv/bin/python scripts/snapshot_source.py                            # 910a39a4… (69 ficheros)
pgrep -fl 'gso|pytest|uvicorn' ; lsof -iTCP:8000 -sTCP:LISTEN          # sin salida
```

### 6.1 Este relevo (sólo verificación; sólo cambia `docs/STATUS.md`)

```bash
date -u ; git rev-parse HEAD ; git branch --show-current ; git status --short ; git stash list | wc -l
find src tests scripts configs docs reports README.md pyproject.toml uv.lock -newer docs/STATUS.md -type f   # nada posterior
pgrep -fl 'gso|gemma_system_one|pytest|uvicorn'          # código 1, sin salida
lsof -iTCP:8000 -sTCP:LISTEN                             # sin salida
.venv/bin/python -c "from gemma_system_one.env import source_fingerprint as f; d=f(); print(d['sha256'], d['files'])"   # 910a39a4…, 69
ls artifacts/source/                                     # 6 copias (§3)
.venv/bin/pytest tests/unit tests/integration -q         # 263 passed, 27,80 s
.venv/bin/pytest --collect-only -q tests/mps tests/e2e   # 14 recogidos, no ejecutados (código sin cambios desde su última pasada)
.venv/bin/ruff check . ; .venv/bin/ruff format --check . ; uv lock --check ; git diff --check   # correctos
shasum -a 256 -c reports/phase4b/test-protocol.sha256    # OK (protocolo de cierre sin desviaciones)
```

También se comprobó que existen los manifiestos de V2, C2, B (con su calibración), V de la fase 4 y la referencia congelada de la fase 3, y se inspeccionaron los procesos auxiliares con `psutil` (§2).

## 7. Fallos encontrados en esta sesión (reproducibles)

| Fallo | Reproducción | Estado |
|---|---|---|
| Estilos visuales compartidos entre particiones (revisión de fase 4, hallazgo 4) | Cruzar `data/vision_pilot_v1/splits/seed0.json` con `audit.jsonl` | Resuelto con v2 y un test nuevo; v1 queda como historial |
| Hash de fuentes sin copia (hallazgo 5) | `ls artifacts/source/70068e15*` (no existe) | Copia automática desde ahora; límite histórico documentado |
| FastAPI interpretaba `request: Request` como parámetro de consulta (import local con `from __future__ import annotations`) | Mover `from fastapi import Request` dentro de `create_app` → `/health/ready` da 422 | Import a nivel de módulo; cubierto por `tests/integration/test_api.py` |
| La copia de fuentes de los tests se escribía en `artifacts/` del repo | Ejecutar tests sin `GSO_SOURCE_ARCHIVE_DIR` | Fixture automático en `conftest.py` |
| El test de estilo reservado suponía «el último de la lista» | `test_transfer_reserves_style_and_phrasing` con la lista ampliada | Usa `V1_STYLE_COUNT` |
| Aserción de dispositivo (`mps:0` frente a `mps`) en los E2E | `tests/e2e` | Corregida en el test |
| Hash de fuentes durante un run: el registrado al guardar el checkpoint puede diferir del ejecutado si se edita el código mientras corre (V2: `7f645568…` al empezar, `78d02416…` al guardar) | Manifiesto frente a `env.json` de `runs/vision2_heads/20260923T161024Z` | Desde ahora el checkpoint registra también `run_start_code`. Los dos tienen copia |

## 8. Tareas pendientes, en orden

1. **Revisión independiente** del cierre de fase 4 (`reports/phase4b-styles.md`) y de la fase 5 (`reports/phase5-api.md`). Comprobar en particular:
   - que el protocolo `05901ce5…` es anterior a los datos v2 y que el test se ejecutó una vez por artefacto;
   - que los estilos son disjuntos en el split real;
   - la identidad train/serve de los E2E;
   - los límites y códigos de la API;
   - el benchmark.
2. **Commit, a decidir por el usuario.**
3. **Fase 6** (spec §10): E4B, más opciones y optimización. Sólo con una ganancia medida que justifique el coste. Antes de optimizar el servicio:
   - batching con equivalencia demostrada (decisión 0002);
   - recomputación o checkpointing para LoRA con imagen (K > 5 no probado).
4. **Mejoras candidatas**, sin usar los tests ya abiertos:
   - calibrar V2 con la partición de calibración (estilos 9–10);
   - Score por tramos;
   - umbrales de abstención (spec §6.2, §8);
   - parada temprana o LoRA más corto.

**Tests ya usados, que no deben usarse para decidir:** fase 3 (`pilot_v3`), fase 4 v1 (`vision_pilot_v1`) y cierre de fase 4 (`vision_pilot_v2`).

### Riesgos abiertos

- **Datos:** sintéticos, de una sola familia visual. Los estilos «no vistos» son variaciones del mismo gráfico. Particiones de 70–100 grupos.
- **API:**
  - sin batching, un checkpoint por proceso;
  - el timeout con el modelo real sólo se probó con un doble en CPU;
  - V2 no está calibrado;
  - sin abstención.
- **MPS:** reanudación y `empty_cache` no verificados bit a bit; FP16, gradient checkpointing y contexto > 512 con autograd, sin probar.
- **Plantilla:** «using only the state» también con imagen.

## 9. Privacidad

- **Contenido:** datos sintéticos por reglas; imágenes generadas con valores conocidos. Sin credenciales ni datos personales.
- **Logs:** los del servidor (`reports/phase5/*.server.log`) sólo registran request_id, estado, duración y dimensiones, nunca estado, textos ni imágenes. Las rutas absolutas de avisos de bibliotecas se enmascararon con `~`.
- **Versionado:** datasets, imágenes, runs y artefactos no se versionan.
- **Pendiente anterior:** `reports/doctor/20260922T195301Z.json` (fase 0) contiene una ruta absoluta del usuario.
