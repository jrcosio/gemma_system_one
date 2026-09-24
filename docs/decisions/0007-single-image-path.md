# 0007 — Una imagen por ejemplo: referencia por contenido, colocación, extracción y ablaciones

Fecha: 2026-09-23 · Fase 4 · Estado: aceptada

## Contexto

La spec §7.C pide reutilizar el contrato, el procesador oficial y la visión congelada, empezar con una imagen, registrar los tokens y la resolución reales, no imponer 448×448 y comparar con imagen omitida e intercambiada. §5.1 exige imagen real decodificable, hash en el manifiesto y que ni el título ni el nombre del fichero revelen la clase. §9 fija límites iniciales de 5 MiB y 16 MP.

## Evidencia previa medida (E2B, transformers 5.17.0, MPS/BF16)

- `Gemma4Processor` añade a cada fila con imagen `pixel_values` [1, 2520, 768], `image_position_ids` [1, 2520, 2] y `mm_token_type_ids`, que vale 1 en los tokens visuales.
- `max_soft_tokens = 280`, parche de 16 px y agrupación 3×3: la imagen se reescala conservando proporción.
  - 480×360 → **266 tokens visuales**;
  - 768×768 → 256;
  - 1600×900 → 264.
- `Gemma4Model.forward` acepta esos campos por la ruta de carga de la decisión 0001; la torre de visión carga en BF16 y congelada.
- Forward con imagen: ~0,20 s en caliente (0,30 s en frío), frente a 0,04 s sólo con texto. Repetición idéntica. Otra imagen da otra representación (máx. |Δ| 2,47; coseno 0,9965). Memoria del driver ~11,4 GB, frente a 10,2 GB asignados por los pesos.
- **Fallo que se habría producido:** la extracción anterior sólo recortaba tensores 2D. Con imagen, `pixel_values` e `image_position_ids` (3D) pasaban sin indexar por fila: lotes erróneos con microlote > 1 y, con el lote completo, `pixel_values` de todo el conjunto en memoria (≈ 7,7 MB por fila en FP32).

## Decisión

1. **Referencia por contenido.** `image_path = images/<sha256>.<png|jpg>`. El validador comprueba:
   - el hash;
   - que la ruta quede dentro de la raíz;
   - formato PNG/JPEG sin animación;
   - ≤ 5 MiB y ≤ 16 MP, estos últimos leídos en la cabecera, antes de decodificar;
   - que el fichero decodifique.

   Así el hash de entrada y el de cada fila identifican la imagen sin leerla, y el nombre no revela la clase. Otro nombre se rechaza: el test de rutas de la fase 1 se actualizó a este contrato.
2. **Hashes.** Sólo cambian cuando hay imagen, así que las filas, cachés, splits y checkpoints de fases 1–3 conservan su identidad:
   - `Row.sha256` y `serialized_input_hash` incluyen `IMAGE_PLACEMENT` y la referencia de la imagen;
   - `model_input_hash` añade `image`.
   - La misma imagen en dos particiones es un error de fuga.
   - Los casi duplicados de estado sólo se comparan entre ejemplos con la misma imagen.
3. **Colocación.** La imagen va antes del texto de la fila en el mismo turno de usuario (`image_before_text_v1`). El texto de la fila (plantilla `gso-text-v1`) **no cambia**: omitir la imagen es una ablación limpia, que sólo quita la imagen. Límite: la instrucción fija dice «using only the state»; con imagen, el estado incluye la imagen, pero la redacción no lo explicita.
4. **Extracción.** Con imagen, una fila por forward: es obligatorio y rechaza `microbatch_rows > 1` (decisión 0002). Cada imagen se carga y verifica al procesar su fila. `slice_batch` indexa todo tensor con dimensión de lote y recorta sólo los de secuencia. Se registran `image_rows` e `image_tokens`. La caché no necesita otra huella porque la imagen ya está en el hash de cada fila.
5. **`configs/e2b_vision.yaml`.** Misma base que `e2b_text`, con `max_length: 768`. Las filas del piloto visual miden como máximo 490 tokens.
6. **Datos `support-vision-v1`:**
   - paneles de barras dibujados con la fuente incrustada de Pillow, que es determinista entre máquinas pero no dibuja vocales acentuadas; por eso los textos de la imagen van sin acentos;
   - valores conocidos con márgenes, de modo que la respuesta sólo está en la imagen;
   - estilo visual y redacciones reservados para transferencia;
   - `audit.jsonl` con los hechos de cada caso para verificar las etiquetas; el modelo no lo lee.
7. **Uso visual:**
   - control `images: omit`: mismas filas sin imagen, con entrenamiento y selección idénticos;
   - `gso evaluate --vision-ablation`: misma pregunta con la imagen omitida o con la de otro grupo de la misma partición, conservando la etiqueta original.

   El modo de imagen queda en el manifiesto del checkpoint; evaluación y arranque LoRA exigen el mismo modo.

## Resultado (detalle en `reports/phase4-vision.md`)

- **Test predeclarado:** V (con imagen) 0,418 de NLL frente a 1,112 de C (mismas filas sin imagen). V − C = −0,694 [−0,816; −0,556]. Con la imagen de otro grupo, la NLL de V sube a 3,02 y la accuracy baja de 0,829 a 0,386.
- **Recarga** con imagen desde el texto en proceso nuevo: Δlogit 0,0.
- **Extracción con imagen:** ≈ 4 filas/s; driver MPS 11,5 GB.
- **Backward LoRA de una pregunta con imagen y K = 5:** 16,0 GB de grafo, 26,2 GB asignados.

## Consecuencias

- Coste: ~266 tokens más por fila y unas 5 veces más tiempo por forward. Una pregunta Choice/Score repite la imagen en sus K/M filas.
- LoRA con imagen funciona (visión congelada, LoRA textual; decisión 0005), pero con ~3,2 GB de grafo por fila. K > 5 no se midió: la extrapolación para K = 8 ronda 36 GB y podría superar 32 GiB. Debe medirse antes de escalar; no se ha demostrado un límite exacto en K = 5 (corrección de revisión).
- La comparación con el control sin imagen cambia la entrada a propósito: `gso compare --allow-different-inputs`.
- No se ha probado JPEG real, varias imágenes por fila ni imágenes fuera de 480×360 en entrenamiento.


## Aclaraciones de revisión (2026-09-23)

La validación ahora decodifica píxeles además de `verify()` y comparte controles con `load_image`; las fugas comparan el sha de la referencia sin diferenciar `.jpg`/`.jpeg`. No cambian contratos, hashes históricos ni arquitectura. Los cuatro splits del piloto comparten estilos 0–3: sólo transferencia reserva el 4. Esto no satisface la separación de estilos por partición de ESPECIFICACION §5.1; queda como desviación pendiente, no como prueba de generalización a estilos inéditos en el test principal. Véase `reports/phase4-review.md`.
