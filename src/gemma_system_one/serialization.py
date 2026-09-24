"""Plantilla versionada: (estado, pregunta[, candidato]) -> texto de una fila del backbone.

Es la única ruta por la que el contenido llega al modelo, en entrenamiento y en
servicio. Sólo usa ``state`` y ``question``: nunca etiquetas, IDs de ejemplo, grupo,
familia, idioma, procedencia ni IDs opacos de opciones.

Cada fila termina con una instrucción fija de evaluación; la plantilla de chat del
procesador añade después el turno del modelo (sin generar nada). El contenido
variable se escapa (``& < >``) para que no pueda cerrar ni abrir delimitadores.

Imagen (fase 4, decisión 0007): una fila puede llevar la imagen del ejemplo, que va antes del
texto en el turno de usuario. El texto de la fila no cambia, así que omitir la imagen es una
ablación limpia. La huella de la fila incluye la referencia por contenido de la imagen; las
filas sin imagen conservan exactamente su huella anterior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .contracts import ChoiceQuestion, NoulQuestion, ScoreQuestion, normalize_text
from .images import IMAGE_PLACEMENT

TEMPLATE_VERSION = "gso-text-v1"

TASKS = {
    "noul": "Decide whether the answer to the question is yes, using only the state.",
    "choice": (
        "Decide whether the candidate option is the correct answer to the question, using only the state."
    ),
    "score": (
        "Decide whether the rubric level shown is the correct level for the question, using only the state."
    ),
}


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def canonical_state(state: Any) -> str:
    """Texto: tal cual (fin de línea normalizado). JSON: claves ordenadas y formato fijo."""
    if isinstance(state, str):
        return state.replace("\r\n", "\n").replace("\r", "\n")
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


def canonical_choice_order(criteria: dict[str, str]) -> list[tuple[str, str]]:
    """Orden independiente del orden del mapa: descripción normalizada y, en empate, ID."""
    return sorted(criteria.items(), key=lambda kv: (normalize_text(kv[1]), kv[0]))


@dataclass(frozen=True)
class Row:
    """Una fila del backbone. ``candidate_id`` sirve para reconstruir; no está en ``text``."""

    text: str
    question_type: str
    candidate_index: int | None = None
    candidate_id: str | None = None
    n_candidates: int = 1
    image: str | None = None  # referencia por contenido relativa al dataset (images/<sha256>.png)

    @property
    def sha256(self) -> str:
        if self.image is None:
            return hashlib.sha256(f"{TEMPLATE_VERSION}\n{self.text}".encode()).hexdigest()
        head = f"{TEMPLATE_VERSION}\n{IMAGE_PLACEMENT}\n{self.image}\n"
        return hashlib.sha256(f"{head}{self.text}".encode()).hexdigest()


def _render(state: Any, qtype: str, instructions: str, body: list[str]) -> str:
    parts = [
        f'<gso-eval version="{TEMPLATE_VERSION}">',
        "<state>",
        escape(canonical_state(state)),
        "</state>",
        f'<question type="{qtype}">',
        "<instructions>",
        escape(instructions),
        "</instructions>",
        *body,
        "</question>",
        f"<task>{TASKS[qtype]}</task>",
        "</gso-eval>",
    ]
    return "\n".join(parts)


def expand(
    state: Any, question: NoulQuestion | ChoiceQuestion | ScoreQuestion, image: str | None = None
) -> list[Row]:
    """Expande una pregunta lógica en sus filas (Noul: 1; Choice: K; Score: M), todas con la imagen."""
    if isinstance(question, NoulQuestion):
        return [Row(_render(state, "noul", question.instructions, []), "noul", image=image)]
    if isinstance(question, ChoiceQuestion):
        ordered = canonical_choice_order(question.criteria)
        options = ["<options>"]
        options += [f'<option index="{i}">{escape(d)}</option>' for i, (_, d) in enumerate(ordered)]
        options.append("</options>")
        rows = []
        for i, (cid, desc) in enumerate(ordered):
            body = [*options, f'<candidate index="{i}">{escape(desc)}</candidate>']
            text = _render(state, "choice", question.instructions, body)
            rows.append(Row(text, "choice", i, cid, len(ordered), image))
        return rows
    if isinstance(question, ScoreQuestion):
        levels = ["<rubric>"]
        levels += [f'<level index="{m}">{escape(d)}</level>' for m, d in enumerate(question.criteria)]
        levels.append("</rubric>")
        rows = []
        for m, desc in enumerate(question.criteria):
            body = [*levels, f'<evaluated-level index="{m}">{escape(desc)}</evaluated-level>']
            text = _render(state, "score", question.instructions, body)
            rows.append(Row(text, "score", m, None, len(question.criteria), image))
        return rows
    raise TypeError(f"Tipo de pregunta no soportado: {type(question).__name__}")


def expand_example(example) -> list[Row]:
    """Ruta de entrenamiento: idéntica a la de servicio porque sólo usa state, question e imagen."""
    return expand(example.state, example.question, example.image_path)
