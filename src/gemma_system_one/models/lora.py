"""LoRA sobre proyecciones de atención del transformer textual (spec §7.B, decisión 0005).

- Objetivos por nombre completo, enumerados con ``named_modules()``:
  ``language_model.layers.{i}.self_attn.{q_proj|v_proj}`` y de tipo ``nn.Linear``. Nunca un
  patrón genérico: en Gemma 4 visión y audio también tienen ``q_proj``/``v_proj``
  (``Gemma4ClippableLinear``). En E2B las capas 15–34 comparten K/V y no tienen ``v_proj``
  (decisión 0001), así que salen 35 ``q_proj`` + 15 ``v_proj``.
- ``peft.inject_adapter_in_model`` modifica el modelo en su sitio: sigue siendo ``Gemma4Model``.
- Pesos LoRA en FP32 (la base sigue en BF16); PEFT convierte la entrada al dtype de LoRA.
- Con ``lora_B`` a cero (inicialización de PEFT) la salida es idéntica a la de la base.
- Modos: la base queda siempre en ``eval()``; sólo el dropout de LoRA pasa a ``train``.
"""

from __future__ import annotations

import re
from importlib import metadata
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

LORA_MARK = ".lora_"
TEXT_ATTENTION = r"^language_model\.layers\.(\d+)\.self_attn\.({projs})$"


class LoraError(RuntimeError):
    pass


class LoraParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    r: int = Field(default=8, ge=1, le=256)
    alpha: float = Field(default=16.0, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    projections: tuple[str, ...] = ("q_proj", "v_proj")


def text_attention_targets(model: nn.Module, projections: tuple[str, ...]) -> list[str]:
    """Nombres completos de las proyecciones del transformer textual; error si el tipo no es Linear."""
    if not projections or any(not re.fullmatch(r"[a-z_]+", p) for p in projections):
        raise LoraError(f"Proyecciones no válidas: {projections}")
    pattern = re.compile(TEXT_ATTENTION.format(projs="|".join(projections)))
    targets = []
    for name, module in model.named_modules():
        if pattern.match(name):
            if type(module) is not nn.Linear:
                raise LoraError(f"{name} es {type(module).__name__}, no nn.Linear")
            targets.append(name)
    if not targets:
        raise LoraError("No hay módulos objetivo en el transformer textual")
    return targets


def lora_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if LORA_MARK in n}


def apply_lora(model: nn.Module, params: LoraParams, *, seed: int = 0) -> list[str]:
    """Inyecta LoRA en los objetivos textuales y verifica qué queda entrenable. Devuelve los objetivos.

    ``lora_A`` se inicializa al azar (``lora_B`` a cero): se usa un RNG propio con ``seed`` para que
    la inicialización no dependa del estado global (antes, dos runs iguales divergían en el paso 2).
    """
    from peft import LoraConfig, inject_adapter_in_model

    if lora_parameters(model):
        raise LoraError("El modelo ya tiene adaptadores LoRA")
    if any(p.requires_grad for p in model.parameters()):
        raise LoraError("La base debe estar congelada antes de inyectar LoRA")
    targets = text_attention_targets(model, params.projections)
    config = LoraConfig(
        r=params.r, lora_alpha=params.alpha, lora_dropout=params.dropout, target_modules=targets, bias="none"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        inject_adapter_in_model(config, model)
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B"):
            module.to(torch.float32)
    lora = lora_parameters(model)
    wrapped = sorted({n.split(LORA_MARK)[0] for n in lora})
    if wrapped != sorted(targets):
        raise LoraError("Los módulos adaptados no coinciden con los objetivos")
    extra = [n for n, p in model.named_parameters() if p.requires_grad and LORA_MARK not in n]
    frozen = [n for n, p in lora.items() if not p.requires_grad]
    if extra or frozen:
        raise LoraError(f"Entrenables inesperados: base={extra[:5]} lora_congelados={frozen[:5]}")
    set_lora_mode(model, training=False)
    return targets


def set_lora_mode(model: nn.Module, *, training: bool) -> None:
    """Base en ``eval()`` siempre; el dropout de LoRA sólo se activa al entrenar.

    Con ``enable_layer_recomputation``, las capas marcadas recomputan sólo al entrenar: se cambia el
    indicador de la capa, no el de sus submódulos (atención y dropout de la base siguen en eval)."""
    model.eval()
    for name, module in model.named_modules():
        if name.endswith("lora_dropout"):
            module.train(training)
        elif getattr(module, RECOMPUTE_MARK, False):
            module.training = training


RECOMPUTE_MARK = "_gso_recompute"


def enable_layer_recomputation(model: nn.Module) -> int:
    """Recomputación de activaciones (checkpoint no reentrante) en cada capa del decodificador textual.

    Spec §2.3/§7.B: reduce la memoria de activaciones del backward de LoRA. Transformers sólo recomputa
    si la capa está en modo entrenamiento (``GradientCheckpointingLayer``); ``set_lora_mode`` activa ese
    indicador sólo en la capa, así que no cambia ningún cálculo del forward. Con el modo no reentrante
    los adaptadores reciben gradiente aunque las entradas (embeddings congelados) no lo requieran.
    Devuelve el número de capas marcadas."""
    from functools import partial

    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_layers import GradientCheckpointingLayer

    text = getattr(model, "language_model", model)
    layers = [m for m in text.modules() if isinstance(m, GradientCheckpointingLayer)]
    if not layers:
        raise LoraError("El modelo no tiene capas con recomputación de activaciones")
    for layer in layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        setattr(layer, RECOMPUTE_MARK, True)
    return len(layers)


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", torch.float32).clone() for n, p in lora_parameters(model).items()}


@torch.no_grad()
def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = lora_parameters(model)
    if set(params) != set(state):
        missing, unexpected = sorted(set(params) - set(state)), sorted(set(state) - set(params))
        raise LoraError(f"Estado LoRA incompatible: faltan {missing[:3]}, sobran {unexpected[:3]}")
    for name, p in params.items():
        if tuple(state[name].shape) != tuple(p.shape):
            raise LoraError(f"Forma distinta en {name}: {tuple(state[name].shape)} != {tuple(p.shape)}")
        p.copy_(state[name].to(p.device, p.dtype))


def lora_summary(model: nn.Module, targets: list[str], params: LoraParams) -> dict[str, Any]:
    lora = lora_parameters(model)
    by_proj: dict[str, list[int]] = {}
    for t in targets:
        layer, proj = re.match(TEXT_ATTENTION.format(projs="|".join(params.projections)), t).groups()
        by_proj.setdefault(proj, []).append(int(layer))
    return {
        "peft_version": metadata.version("peft"),
        "method": "peft.inject_adapter_in_model",
        "r": params.r,
        "alpha": params.alpha,
        "scaling": params.alpha / params.r,
        "dropout": params.dropout,
        "targets": targets,
        "layers_by_projection": by_proj,
        "trainable_parameters": int(sum(p.numel() for p in lora.values())),
        "weight_dtype": "float32",
        "base_trainable_parameters": int(
            sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and LORA_MARK not in n)
        ),
    }
