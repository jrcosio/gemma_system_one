"""Construcción de lotes con el procesador oficial (compartida por doctor, entrenamiento y servicio).

- Plantilla de chat oficial con ``add_generation_prompt=True`` y ``enable_thinking=False``;
  no se genera texto: el último token válido es el que se lee.
- Padding derecho explícito (el tokenizer trae ``left`` por defecto).
- Nunca se trunca: una fila que supera ``max_length`` es un error con su índice.
- Imagen (fase 4): antes del texto, en el mismo turno de usuario (decisión 0007).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

CHAT_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}
PADDING_SIDE = "right"
PROCESSOR_FILES = ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


class InputTooLongError(ValueError):
    def __init__(self, rows: list[tuple[int, int]], max_length: int):
        self.rows = rows
        preview = ", ".join(f"fila {i}: {n} tokens" for i, n in rows[:5])
        super().__init__(f"{len(rows)} filas superan max_length={max_length} ({preview}); no se trunca")


def load_processor(snapshot_dir: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(snapshot_dir)


def processor_fingerprint(snapshot_dir: Path) -> dict[str, Any]:
    files = {}
    for name in PROCESSOR_FILES:
        files[name] = hashlib.sha256((Path(snapshot_dir) / name).read_bytes()).hexdigest()
    return {"files_sha256": files, "chat_kwargs": CHAT_KWARGS, "padding_side": PADDING_SIDE}


def build_chat_batch(
    processor,
    texts: list[str] | tuple[str, ...],
    *,
    max_length: int | None,
    padding_side: str = PADDING_SIDE,
    images: list | None = None,
) -> dict[str, torch.Tensor]:
    """Tokeniza filas como turno de usuario; devuelve todos los campos del procesador.

    ``images`` (opcional, alineada con ``texts``): imagen PIL o ``None`` por fila; la imagen va
    antes del texto (``IMAGE_PLACEMENT``). Un lote no mezcla filas con y sin imagen.
    """
    if not texts:
        raise ValueError("Lote vacío")
    if images is not None:
        if len(images) != len(texts):
            raise ValueError("images debe tener una entrada por fila")
        if len({im is None for im in images}) > 1:
            raise ValueError("Un lote no puede mezclar filas con y sin imagen")
    processor.tokenizer.padding_side = padding_side
    conversations = []
    for i, t in enumerate(texts):
        content = [{"type": "text", "text": t}]
        if images is not None and images[i] is not None:
            content.insert(0, {"type": "image", "image": images[i]})
        conversations.append([{"role": "user", "content": content}])
    batch = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        **CHAT_KWARGS,
    )
    batch = dict(batch)
    if max_length is not None:
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        too_long = [(i, int(n)) for i, n in enumerate(lengths) if n > max_length]
        if too_long:
            raise InputTooLongError(too_long, max_length)
    return batch


def image_token_count(batch: dict[str, torch.Tensor]) -> int:
    """Tokens visuales del lote (``mm_token_type_ids == 1`` en Gemma 4); 0 si no hay imagen."""
    if "pixel_values" not in batch or "mm_token_type_ids" not in batch:
        return 0
    return int((batch["mm_token_type_ids"] == 1).sum())
