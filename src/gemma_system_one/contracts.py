"""Contratos compartidos por dataset, entrenamiento e inferencia (esquema JSONL versión 1).

Un ``Example`` es una pregunta lógica. Sólo ``state`` y ``question`` llegan al modelo
(vía ``serialization``); ``id``, ``group_id``, ``task_family``, ``language``, ``target``
y ``provenance`` son metadatos y nunca se serializan en el prompt.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

# Límites iniciales del proyecto (spec §4.4, §4.5, §9); no son límites de Jev.
CHOICE_MIN, CHOICE_MAX = 2, 8
SCORE_MIN, SCORE_MAX = 2, 5
MAX_INSTRUCTIONS_CHARS = 2000
MAX_CRITERION_CHARS = 500
MAX_STATE_CHARS = 20_000
MAX_STATE_DEPTH = 8

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")]
NonEmptyText = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def normalize_text(text: str) -> str:
    """Forma canónica para comparar descripciones: NFKC, casefold y espacios colapsados."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _require_text(value: str, limit: int, what: str) -> str:
    if not value.strip():
        raise ValueError(f"{what} vacío")
    if len(value) > limit:
        raise ValueError(f"{what} supera {limit} caracteres")
    return value


def _check_state_value(value: Any, depth: int = 0) -> None:
    if depth > MAX_STATE_DEPTH:
        raise ValueError(f"state supera la profundidad {MAX_STATE_DEPTH}")
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state contiene un número no finito")
        return
    if isinstance(value, list):
        for item in value:
            _check_state_value(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Las claves de state deben ser cadenas")
            _check_state_value(item, depth + 1)
        return
    raise ValueError(f"Tipo no permitido en state: {type(value).__name__}")


def validate_state(state: Any) -> Any:
    if isinstance(state, str):
        _require_text(state, MAX_STATE_CHARS, "state")
        return state
    if not isinstance(state, dict | list):
        raise ValueError("state debe ser texto, objeto o array JSON")
    if not state:
        raise ValueError("state vacío")
    _check_state_value(state)
    if len(json.dumps(state, ensure_ascii=False)) > MAX_STATE_CHARS:
        raise ValueError(f"state supera {MAX_STATE_CHARS} caracteres serializado")
    return state


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: NonEmptyText

    @field_validator("instructions")
    @classmethod
    def _instr(cls, v: str) -> str:
        return _require_text(v, MAX_INSTRUCTIONS_CHARS, "instructions")


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: NonEmptyText
    criteria: dict[Identifier, NonEmptyText]

    @field_validator("instructions")
    @classmethod
    def _instr(cls, v: str) -> str:
        return _require_text(v, MAX_INSTRUCTIONS_CHARS, "instructions")

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, str]) -> dict[str, str]:
        if not CHOICE_MIN <= len(v) <= CHOICE_MAX:
            raise ValueError(f"choice necesita entre {CHOICE_MIN} y {CHOICE_MAX} opciones")
        for key, desc in v.items():
            _require_text(desc, MAX_CRITERION_CHARS, f"descripción de {key}")
        normalized = [normalize_text(d) for d in v.values()]
        if len(set(normalized)) != len(normalized):
            # Filas idénticas: el modelo no podría distinguir los candidatos.
            raise ValueError("Descripciones de opciones duplicadas tras normalizar")
        return v


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: NonEmptyText
    # El orden es parte del significado: nunca se reordena.
    criteria: list[NonEmptyText]

    @field_validator("instructions")
    @classmethod
    def _instr(cls, v: str) -> str:
        return _require_text(v, MAX_INSTRUCTIONS_CHARS, "instructions")

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: list[str]) -> list[str]:
        if not SCORE_MIN <= len(v) <= SCORE_MAX:
            raise ValueError(f"score necesita entre {SCORE_MIN} y {SCORE_MAX} niveles")
        for i, desc in enumerate(v):
            _require_text(desc, MAX_CRITERION_CHARS, f"nivel {i}")
        normalized = [normalize_text(d) for d in v]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Niveles de rúbrica duplicados tras normalizar")
        return v


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class NoulTarget(_Strict):
    # int estricto (no Literal): Literal[0, 1] aceptaría ``true`` porque True == 1.
    label: int = Field(ge=0, le=1)


class ChoiceTarget(_Strict):
    class_id: Identifier


class ScoreTarget(_Strict):
    level_index: int = Field(ge=0)


class Provenance(_Strict):
    source: Literal["synthetic_rule", "human", "llm_distillation", "fixture"]
    generator_version: NonEmptyText
    label_method: NonEmptyText
    template_id: str | None = None
    seed: int | None = None
    # Para fuentes LLM (spec §5.2): modelo/proveedor, revisión, prompt y estado de revisión.
    model: str | None = None
    model_revision: str | None = None
    prompt_id: str | None = None
    review_status: Literal["unreviewed", "reviewed", "rejected"] | None = None


class Example(_Strict):
    schema_version: Literal[1]
    id: Identifier
    group_id: Identifier
    task_family: Identifier
    language: Literal["es", "en"]
    state: Any
    image_path: str | None = None
    question: Question
    target: NoulTarget | ChoiceTarget | ScoreTarget
    provenance: Provenance

    @field_validator("state", mode="before")
    @classmethod
    def _state(cls, v: Any) -> Any:
        return validate_state(v)

    @field_validator("image_path")
    @classmethod
    def _image_path(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v or v.startswith(("/", "\\")) or ":" in v or ".." in v.replace("\\", "/").split("/"):
            raise ValueError("image_path debe ser relativa a la raíz del dataset, sin '..' ni unidades")
        return v

    @model_validator(mode="after")
    def _target_matches_question(self) -> Example:
        q, t = self.question, self.target
        if isinstance(q, NoulQuestion):
            if not isinstance(t, NoulTarget):
                raise ValueError("Una pregunta noul requiere target.label")
        elif isinstance(q, ChoiceQuestion):
            if not isinstance(t, ChoiceTarget):
                raise ValueError("Una pregunta choice requiere target.class_id")
            if t.class_id not in q.criteria:
                raise ValueError(f"class_id {t.class_id!r} no está entre los criterios")
        elif isinstance(q, ScoreQuestion):
            if not isinstance(t, ScoreTarget):
                raise ValueError("Una pregunta score requiere target.level_index")
            if t.level_index >= len(q.criteria):
                raise ValueError(f"level_index {t.level_index} fuera de rango (M={len(q.criteria)})")
        return self


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Valor JSON no finito no permitido: {name}")


def parse_json_strict(line: str) -> Any:
    """Rechaza constantes no finitas y claves repetidas, incluso en objetos anidados."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Clave JSON duplicada")
            result[key] = value
        return result

    return json.loads(line, parse_constant=_reject_constant, object_pairs_hook=unique_object)
