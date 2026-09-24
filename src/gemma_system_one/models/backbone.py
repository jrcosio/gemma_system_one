"""Backbone Gemma 4 sin cabeza de lenguaje.

Se carga ``Gemma4Model`` (base multimodal documentada sin ``lm_head``) directamente
desde el checkpoint ``Gemma4ForConditionalGeneration``: transformers elimina el
prefijo ``model.`` y el ``lm_head`` está enlazado a los embeddings, así que no hay
que borrar atributos a mano ni calcular logits del vocabulario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..config import ProjectConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class BackboneLoadError(RuntimeError):
    pass


def torch_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def resolve_device(cfg: ProjectConfig) -> torch.device:
    want = cfg.runtime.device
    if want == "mps" and not torch.backends.mps.is_available():
        if not cfg.runtime.allow_cpu_fallback:
            raise RuntimeError("MPS no disponible y allow_cpu_fallback=false")
        return torch.device("cpu")
    return torch.device(want)


def last_valid_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Índice ``max(t : mask[t] == 1)`` por fila; válido con padding a izquierda o derecha."""
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask debe ser [B, T], no {tuple(attention_mask.shape)}")
    valid = attention_mask.bool()
    if not bool(valid.any(dim=1).all()):
        empty = (~valid.any(dim=1)).nonzero().flatten().tolist()
        raise ValueError(f"Filas sin tokens válidos: {empty}")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return torch.where(valid, positions, -1).amax(dim=1)


def pool_last_valid(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Estado del último token válido de cada fila: [B, T, D] -> [B, D]."""
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError(f"hidden {tuple(hidden.shape)} y mask {tuple(attention_mask.shape)} no alinean")
    idx = last_valid_index(attention_mask)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def move_inputs(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Mueve todos los campos del procesador; enteros siguen enteros, flotantes al dtype del modelo."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        out[key] = value
    return out


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor

    def pooled(self) -> torch.Tensor:
        return pool_last_valid(self.last_hidden_state, self.attention_mask)


class Backbone:
    """Envoltorio fino: posee el modelo congelado y devuelve estados finales alineados con la máscara."""

    def __init__(self, model: nn.Module, device: torch.device, dtype: torch.dtype, loading_info: dict | None):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.loading_info = loading_info or {}

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.get_text_config().hidden_size)

    @property
    def text_model(self) -> nn.Module:
        return self.model.language_model

    def freeze(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(self, batch: dict[str, Any]) -> EncoderOutput:
        if "attention_mask" not in batch:
            raise ValueError("El lote debe incluir attention_mask")
        inputs = move_inputs(batch, self.device, self.dtype)
        out = self.model(**inputs, use_cache=False, return_dict=True)
        return EncoderOutput(out.last_hidden_state, inputs["attention_mask"])


def load_backbone(cfg: ProjectConfig, snapshot_dir: Path, device: torch.device | None = None) -> Backbone:
    """Carga ``Gemma4Model`` desde un snapshot local (sin red), congelado y en ``eval()``."""
    from transformers import Gemma4Model

    device = device or resolve_device(cfg)
    dtype = torch_dtype(cfg.model.dtype)
    model, info = Gemma4Model.from_pretrained(
        snapshot_dir,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        output_loading_info=True,
        local_files_only=True,
    )
    missing = sorted(info.get("missing_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    if missing or mismatched or info.get("error_msgs"):
        raise BackboneLoadError(
            f"Carga incompleta: missing={missing[:10]} mismatched={mismatched[:10]} "
            f"errors={info.get('error_msgs')}"
        )
    unexpected = sorted(info.get("unexpected_keys", []))
    if any(k.startswith(("language_model.", "model.language_model.")) for k in unexpected):
        raise BackboneLoadError(f"Claves inesperadas del modelo textual: {unexpected[:10]}")
    model.to(device)
    backbone = Backbone(
        model, device, dtype, {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()}
    )
    backbone.freeze()
    return backbone
