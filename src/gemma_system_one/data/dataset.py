"""Carga y validación de un dataset JSONL v1.

Estructura en disco: ``<root>/examples.jsonl`` (+ ``dataset_manifest.json`` opcional con
su sha256). Las rutas de imagen se resuelven respecto a ``<root>`` y no pueden salir de él.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts import Example, parse_json_strict

EXAMPLES_FILE = "examples.jsonl"
MANIFEST_FILE = "dataset_manifest.json"


class DatasetError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        head = "; ".join(errors[:5])
        super().__init__(f"{len(errors)} errores en el dataset: {head}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_input_hash(example: Example) -> str:
    """Hash de lo que ve el modelo (estado + pregunta [+ imagen]), sin metadatos ni etiqueta.

    La imagen entra por su referencia por contenido (``images/<sha256>``); sin imagen el hash es el
    de fases anteriores, así que los manifiestos de split existentes no cambian."""
    payload = {"state": example.state, "question": example.question.model_dump(mode="json")}
    if example.image_path is not None:
        payload["image"] = example.image_path
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())


def serialized_input_hash(example: Example) -> str:
    """Hash de las filas serializadas: detecta duplicados semánticos que el hash crudo no ve
    (IDs de opciones renombrados, orden del mapa, finales de línea, orden de claves JSON)."""
    from ..serialization import expand_example

    rows = expand_example(example)
    if example.image_path is None:
        return sha256_bytes("\n\x00".join(r.text for r in rows).encode())
    return sha256_bytes("\n\x00".join(r.sha256 for r in rows).encode())


def target_semantic_key(example: Example) -> str:
    """Etiqueta independiente de los IDs opacos: descripción elegida para Choice."""
    from ..contracts import ChoiceQuestion

    if isinstance(example.question, ChoiceQuestion):
        return "choice:" + example.question.criteria[example.target.class_id]
    return target_key(example)


def target_key(example: Example) -> str:
    return json.dumps(example.target.model_dump(mode="json"), sort_keys=True)


def label_of(example: Example) -> Any:
    t = example.target.model_dump(mode="json")
    return next(iter(t.values()))


def descriptive_label(example: Example) -> Any:
    # No altera la representación histórica de etiquetas en manifiestos inmutables.
    if example.question.type == "choice":
        return example.question.criteria[example.target.class_id]
    return label_of(example)


@dataclass
class Dataset:
    root: Path
    examples: list[Example]
    sha256: str

    def by_id(self) -> dict[str, Example]:
        return {e.id: e for e in self.examples}


@dataclass
class ValidationReport:
    root: str
    sha256: str | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_lines(path: Path, errors: list[str]) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                errors.append(f"línea {lineno}: vacía")
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                errors.append(f"línea {lineno}: JSON inválido ({exc})")
                continue
            try:
                examples.append(Example.model_validate(raw))
            except ValidationError as exc:
                msgs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
                errors.append(f"línea {lineno}: " + " | ".join(msgs))
    return examples


def _check_images(root: Path, examples: list[Example], errors: list[str]) -> None:
    """Cada imagen se comprueba una vez: dirección por contenido, límites y decodificación."""
    from ..images import check_image_file

    checked: dict[str, list[str]] = {}
    for e in examples:
        if e.image_path is None:
            continue
        if e.image_path not in checked:
            checked[e.image_path] = check_image_file(root, e.image_path)
        errors += [f"{e.id}: {msg}" for msg in checked[e.image_path]]


def validate_examples(root: Path, examples: list[Example]) -> ValidationReport:
    report = ValidationReport(root=str(root), sha256=None)
    errors, warnings = report.errors, report.warnings

    ids = Counter(e.id for e in examples)
    errors += [f"id duplicado: {i} (x{n})" for i, n in ids.items() if n > 1]
    _check_images(root, examples, errors)

    by_input: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_input[model_input_hash(e)].append(e)
    for items in by_input.values():
        if len(items) < 2:
            continue
        idlist = [e.id for e in items]
        if len({target_key(e) for e in items}) > 1:
            errors.append(f"misma entrada con etiquetas distintas: {idlist}")
        elif len({e.group_id for e in items}) > 1:
            # Entrada idéntica en grupos distintos podría acabar en particiones distintas.
            errors.append(f"entrada duplicada en grupos distintos: {idlist}")
        else:
            warnings.append(f"entrada duplicada dentro del grupo: {idlist}")

    # Duplicados semánticos: misma serialización con distinto ID/orden/formato crudo.
    by_serialized: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_serialized[serialized_input_hash(e)].append(e)
    for items in by_serialized.values():
        if len(items) < 2 or len({model_input_hash(e) for e in items}) == 1:
            continue  # los idénticos en crudo ya se trataron arriba
        idlist = [e.id for e in items]
        if len({target_semantic_key(e) for e in items}) > 1:
            errors.append(f"misma serialización con etiquetas distintas: {idlist}")
        elif len({e.group_id for e in items}) > 1:
            errors.append(f"misma serialización en grupos distintos: {idlist}")
        else:
            warnings.append(f"serialización duplicada dentro del grupo: {idlist}")

    groups: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        groups[e.group_id].append(e)
    label_dist: dict[str, Counter] = defaultdict(Counter)
    for e in examples:
        label_dist[e.question.type][str(descriptive_label(e))] += 1
    # Grupos con varias preguntas del mismo tipo y respuestas distintas: detectan si el
    # modelo ignora las instrucciones.
    contrastive = sum(
        1
        for items in groups.values()
        if any(
            len({str(descriptive_label(e)) for e in items if e.question.type == t}) > 1
            for t in {e.question.type for e in items}
        )
    )
    report.stats = {
        "examples": len(examples),
        "groups": len(groups),
        "by_type": dict(Counter(e.question.type for e in examples)),
        "by_language": dict(Counter(e.language for e in examples)),
        "by_task_family": dict(Counter(e.task_family for e in examples)),
        "label_distribution": {k: dict(v) for k, v in label_dist.items()},
        "questions_per_group": dict(Counter(len(v) for v in groups.values())),
        "groups_with_contrasting_labels": contrastive,
        "with_image": sum(e.image_path is not None for e in examples),
        "distinct_images": len({e.image_path for e in examples if e.image_path is not None}),
    }
    for qtype, dist in label_dist.items():
        if len(dist) < 2:
            warnings.append(f"{qtype}: una sola clase observada {dict(dist)}")
    return report


def validate_dataset(root: str | Path) -> ValidationReport:
    root = Path(root)
    path = root / EXAMPLES_FILE
    if not path.is_file():
        return ValidationReport(root=str(root), sha256=None, errors=[f"No existe {path}"])
    data = path.read_bytes()
    parse_errors: list[str] = []
    examples = _parse_lines(path, parse_errors)
    report = validate_examples(root, examples)
    report.errors[:0] = parse_errors
    report.sha256 = sha256_bytes(data)
    manifest = root / MANIFEST_FILE
    if manifest.is_file():
        declared = json.loads(manifest.read_text(encoding="utf-8")).get("examples_sha256")
        if declared != report.sha256:
            report.errors.append(f"sha256 de {EXAMPLES_FILE} no coincide con {MANIFEST_FILE}")
    return report


def load_dataset(root: str | Path) -> Dataset:
    """Carga validada: cualquier error de esquema o consistencia aborta."""
    root = Path(root)
    report = validate_dataset(root)
    if not report.ok:
        raise DatasetError(report.errors)
    errors: list[str] = []
    examples = _parse_lines(root / EXAMPLES_FILE, errors)
    return Dataset(root=root, examples=examples, sha256=report.sha256 or "")
