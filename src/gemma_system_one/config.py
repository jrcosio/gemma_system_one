"""Configuración tipada del proyecto (YAML -> Pydantic).

Cualquier campo desconocido es un error: una errata en el YAML no debe
convertirse silenciosamente en un valor por defecto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DType = Literal["bfloat16", "float16", "float32"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    repo_id: str = Field(min_length=3)
    # Revisión fijada: un SHA de commit completo, nunca una rama como "main".
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backbone_class: Literal["Gemma4Model"] = "Gemma4Model"
    dtype: DType = "bfloat16"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    # Ficheros que la descarga debe traer; el resto del repo (README, etc.) también
    # se descarga, pero éstos se verifican explícitamente.
    required_files: tuple[str, ...] = (
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
    )

    @field_validator("repo_id")
    @classmethod
    def _repo_id_has_owner(cls, v: str) -> str:
        if v.count("/") != 1 or any(not part for part in v.split("/")):
            raise ValueError("repo_id debe tener la forma 'owner/name'")
        return v


class RuntimeConfig(_Strict):
    device: Literal["mps", "cpu"] = "mps"
    # Si el dispositivo pedido no está disponible, fallar en vez de caer a CPU.
    allow_cpu_fallback: bool = False
    max_length: int = Field(default=512, ge=16, le=8192)
    memory_budget_gib: float = Field(default=32.0, gt=0, le=48)
    seed: int = 0


class PathsConfig(_Strict):
    # None = caché estándar de huggingface_hub (HF_HUB_CACHE).
    hf_cache: Path | None = None
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")


class ProjectConfig(_Strict):
    name: str = Field(min_length=1)
    model: ModelConfig
    runtime: RuntimeConfig = RuntimeConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: se esperaba un mapeo YAML en la raíz")
    return ProjectConfig.model_validate(raw)
