# 0002 — Una fila por microlote en la extracción de representaciones

Fecha: 2026-09-22 · Fase 1 · Estado: aceptada

## Contexto

La primera versión de la fase 1 extraía representaciones en microlotes de 8 filas, ordenadas por longitud y con padding derecho. La prueba MPS `test_phase1_overfit_and_reload_from_text_on_mps` falló al recargar: `max_abs_logit_diff = 2.558` con `atol = 1e-4`. `evaluate` había recalculado todas las filas de train, no solo las 16 del subconjunto de entrenamiento. Al cambiar el conjunto cambian los microlotes, y con ellos la anchura del padding de cada fila.

## Evidencia

Reproducción: `scripts/repro_batch_dependence.py` y `scripts/repro_batch_effect_probs.py`. Configuración: E2B, bf16, MPS, SDPA, con 16 y 84 filas reales del fixture `data/smoke_v1`.

| Comparación de representaciones (fp32 tras pooling) | Máx. dif. abs. | Coseno mínimo | Filas idénticas |
|---|---|---|---|
| Mismo lote repetido | 0,0 | 1,0 | 16/16 |
| Microlote de 8 frente a fila sola | 2,5 | 0,99946 | 0/16 |
| Microlote de 8 frente a otra composición de 8 | 2,0 | 0,99975 | 14/16 |

La norma media de las representaciones es ~228. Las diferencias son compatibles con efectos numéricos de BF16 a través de 35 capas; no se ha aislado su causa exacta. Un cabezal lineal las amplifica:

- Cabezal de humo: hasta 0,93 de logit.
- Cabezal sobreajustado: hasta 9,1 de logit.

Con el cabezal de humo, pasar de microlotes de 4 u 8 a 1 fila cambia **p hasta 0,056**. No hubo cambios de decisión en el umbral de 0,5, ni en las 84 filas de train ni en las 12 de validación.

Con padding derecho, la máscara y el pooling ya eran correctos (tests de fase 0). Las pruebas apuntan a un efecto numérico asociado a la anchura de secuencia. Un cambio de orden de acumulación de los kernels es una hipótesis, no una causa demostrada mediante instrumentación.

## Decisión

- `extraction.microbatch_rows = 1` por defecto y en `configs/noul_*.yaml`. Coincide con el valor inicial de la especificación (§2.3: «Microbatch: 1 fila expandida»).
- La política de extracción forma parte de la huella de la caché y del checkpoint (`fingerprint.extraction`). Un cambio invalida la caché y hace que `evaluate` rechace la huella.
- Con 1 fila por forward, la representación de cada fila no depende del resto del conjunto. Tras el cambio, la recarga desde el texto en un proceso nuevo dio **0,0** de diferencia en tres casos:
  - validación: 12 filas;
  - train: 84 filas;
  - checkpoint de sobreajuste evaluado sobre las 84 filas de train, aunque se entrenó con 32.

## Consecuencias

- **Coste:** un forward por fila (84 filas en 4,1 s sincronizados en MPS, sin padding). Aceptable para fixtures y pilotos. Choice y Score multiplican las filas por K o M.
- **Servicio (fase 5):** usar la misma política que el entrenamiento, o demostrar con evaluación que el batching mantiene probabilidades y decisiones dentro de una tolerancia declarada. Opciones a medir:
  - agrupar solo filas de igual longitud, sin padding;
  - forward en fp32;
  - entrenar el cabezal con representaciones de varias composiciones.
- **LoRA (fase 3):** la misma fuente de variación existe en entrenamiento con gradiente. La pérdida de grupo con microlotes deberá medirse también frente a una fila por forward.
- Los runs anteriores a la decisión (`runs/noul_overfit/20260922T203258Z`, `runs/noul_smoke/20260922T203313Z`, con microlote 8) quedan como historial. No son la referencia.
