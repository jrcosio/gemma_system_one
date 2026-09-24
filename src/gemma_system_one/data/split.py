"""Particiones por grupo: entrenamiento / validación / calibración / test (spec §5.3).

- La unidad de asignación es ``group_id``: todos los derivados de un caso van juntos.
- La asignación es determinista con la semilla y queda en un manifiesto inmutable.
- Validación elige hiperparámetros y checkpoint; calibración ajusta temperaturas; test
  sólo mide el artefacto congelado. ``load_split`` exige pedir test explícitamente.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from ..contracts import Example, parse_json_strict
from ..images import image_content_key
from ..serialization import canonical_state
from .dataset import Dataset, label_of, model_input_hash, serialized_input_hash

SPLITS = ("train", "validation", "calibration", "test")
RATIOS = {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}
NEAR_DUP_JACCARD = 0.8
SPLIT_FORMAT = 1


class SplitError(ValueError):
    pass


def _group_order_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def assign_groups(group_ids: list[str], seed: int) -> dict[str, str]:
    """Asigna cada grupo a una partición; cada partición no-train recibe ≥1 grupo."""
    groups = sorted(set(group_ids), key=lambda g: _group_order_key(seed, g))
    n = len(groups)
    if n < len(SPLITS):
        raise SplitError(f"Se necesitan al menos {len(SPLITS)} grupos; hay {n}")
    sizes = {s: max(1, round(RATIOS[s] * n)) for s in SPLITS if s != "train"}
    sizes["train"] = n - sum(sizes.values())
    if sizes["train"] < 1:
        raise SplitError("No quedan grupos para entrenamiento")
    out: dict[str, str] = {}
    i = 0
    for split in ("validation", "calibration", "test", "train"):
        for g in groups[i : i + sizes[split]]:
            out[g] = split
        i += sizes[split]
    return out


def _shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    words = text.casefold().split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


NEAR_DUP_LIMIT = 200


def near_duplicate_states(
    parts: dict[str, list[Example]], limit: int = NEAR_DUP_LIMIT
) -> tuple[list[dict[str, Any]], int]:
    """Casi duplicados entre particiones con todos los estados distintos de cada grupo
    (Jaccard de 3-gramas de palabras ≥ umbral). Devuelve como máximo ``limit`` pares y el total.

    El coste sigue siendo cuadrático en estados; ``limit`` acota el tamaño del manifiesto."""
    # Con imagen, el estado textual es sólo parte de la entrada: sólo se comparan estados con la
    # misma imagen (las imágenes idénticas entre particiones ya son un error de entrada idéntica).
    states: dict[str, list[tuple[str, str | None, set]]] = {}
    for s, xs in parts.items():
        seen: dict[tuple[str, str, str | None], set] = {}
        for e in xs:
            text = canonical_state(e.state)
            seen.setdefault((e.group_id, text, e.image_path), _shingles(text))
        states[s] = [(g, img, sh) for (g, _, img), sh in seen.items()]
    near: list[dict[str, Any]] = []
    total = 0
    for a, b in combinations(list(parts), 2):
        for ga, ia, sa in states[a]:
            for gb, ib, sb in states[b]:
                if not sa or not sb or ia != ib:
                    continue
                inter = len(sa & sb)
                if inter == 0:
                    continue
                jac = inter / len(sa | sb)
                if jac >= NEAR_DUP_JACCARD:
                    total += 1
                    if len(near) < limit:
                        near.append({"splits": [a, b], "groups": [ga, gb], "jaccard": round(jac, 3)})
    return near, total


def leakage_checks(parts: dict[str, list[Example]]) -> dict[str, Any]:
    """Comprobaciones entre particiones. Los solapamientos exactos son errores."""
    errors: list[str] = []
    group_sets = {s: {e.group_id for e in xs} for s, xs in parts.items()}
    for a, b in combinations(list(parts), 2):
        shared = group_sets[a] & group_sets[b]
        if shared:
            errors.append(f"grupos compartidos {a}/{b}: {sorted(shared)[:5]}")
    input_sets = {s: {model_input_hash(e) for e in xs} for s, xs in parts.items()}
    serial_sets = {s: {serialized_input_hash(e) for e in xs} for s, xs in parts.items()}
    for a, b in combinations(list(parts), 2):
        n = len(input_sets[a] & input_sets[b])
        if n:
            errors.append(f"{n} entradas idénticas compartidas {a}/{b}")
        n = len(serial_sets[a] & serial_sets[b]) - n
        if n > 0:
            errors.append(f"{n} entradas con igual serialización compartidas {a}/{b}")

    image_sets = {
        s: {image_content_key(e.image_path) for e in xs if e.image_path is not None}
        for s, xs in parts.items()
    }
    for a, b in combinations(list(parts), 2):
        n = len(image_sets[a] & image_sets[b])
        if n:
            errors.append(f"{n} imágenes compartidas {a}/{b}")

    near, near_total = near_duplicate_states(parts)

    templates = {
        s: {e.provenance.template_id for e in xs if e.provenance.template_id} for s, xs in parts.items()
    }
    template_overlap = {
        f"{a}/{b}": len(templates[a] & templates[b]) for a, b in combinations(list(parts), 2) if templates[a]
    }
    out = {
        "errors": errors,
        "near_duplicate_states": near,
        "near_duplicate_threshold": NEAR_DUP_JACCARD,
        # Informativo: plantillas compartidas implican que el resultado no mide transferencia.
        "shared_template_ids": template_overlap,
    }
    if near_total > len(near):
        # Sólo se añade al truncar: los manifiestos existentes (sin truncar) conservan su contenido.
        out["near_duplicate_states_total"] = near_total
    return out


def make_split(dataset: Dataset, seed: int) -> dict[str, Any]:
    assignment = assign_groups([e.group_id for e in dataset.examples], seed)
    parts: dict[str, list[Example]] = {s: [] for s in SPLITS}
    for e in dataset.examples:
        parts[assignment[e.group_id]].append(e)
    checks = leakage_checks(parts)
    if checks["errors"]:
        raise SplitError("; ".join(checks["errors"]))
    summary = {}
    for s, xs in parts.items():
        dist: dict[str, Counter] = defaultdict(Counter)
        for e in xs:
            dist[e.question.type][str(label_of(e))] += 1
        summary[s] = {
            "groups": len({e.group_id for e in xs}),
            "examples": len(xs),
            "label_distribution": {k: dict(v) for k, v in dist.items()},
        }
    return {
        "format": SPLIT_FORMAT,
        "dataset_sha256": dataset.sha256,
        "seed": seed,
        "ratios": RATIOS,
        "unit": "group_id",
        "created_utc": datetime.now(UTC).isoformat(),
        "summary": summary,
        "checks": checks,
        "splits": {
            s: {
                "groups": sorted({e.group_id for e in xs}),
                "examples": [{"id": e.id, "input_sha256": model_input_hash(e)} for e in xs],
            }
            for s, xs in parts.items()
        },
    }


def _content(manifest: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "created_utc"}


def split_path(dataset_root: Path, seed: int) -> Path:
    return Path(dataset_root) / "splits" / f"seed{seed}.json"


def write_split(manifest: dict[str, Any], path: Path) -> bool:
    """Escribe el manifiesto; es inmutable. Devuelve False si ya existía idéntico."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _content(existing) != _content(manifest):
            raise SplitError(f"{path} ya existe con otro contenido; los splits no se sobrescriben")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return True


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(
    dataset: Dataset, path: Path, splits: tuple[str, ...], *, allow_test: bool = False
) -> dict[str, list[Example]]:
    """Devuelve los ejemplos de las particiones pedidas, verificando dataset e IDs."""
    if any(s not in SPLITS for s in splits):
        raise SplitError(f"Partición desconocida: {splits}; --split all requiere un dataset externo")
    if "test" in splits and not allow_test:
        raise SplitError("La partición test sólo se usa para la medición final (allow_test=True)")
    manifest = parse_json_strict(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != SPLIT_FORMAT or set(manifest.get("splits", {})) != set(SPLITS):
        raise SplitError("Formato o particiones del manifiesto no válidos")
    if manifest.get("dataset_sha256") != dataset.sha256:
        raise SplitError("El manifiesto de split corresponde a otro contenido del dataset")
    check_planned_split(dataset.root, manifest)
    by_id = dataset.by_id()
    out: dict[str, list[Example]] = {}
    seen = set()
    for s in SPLITS:
        items = []
        for entry in manifest["splits"][s]["examples"]:
            e = by_id.get(entry["id"])
            if e is None or model_input_hash(e) != entry["input_sha256"]:
                raise SplitError(f"{s}: ejemplo {entry['id']} ausente o modificado")
            if e.id in seen:
                raise SplitError(f"ID repetido en el manifiesto: {e.id}")
            seen.add(e.id)
            items.append(e)
        declared = manifest["splits"][s]["groups"]
        actual = {e.group_id for e in items}
        if set(declared) != actual or len(declared) != len(actual):
            raise SplitError(f"{s}: grupos declarados no coinciden con los ejemplos")
        out[s] = items
    if seen != set(by_id):
        raise SplitError("El manifiesto no cubre todos los ejemplos del dataset")
    checks = leakage_checks(out)
    if checks["errors"]:
        raise SplitError("; ".join(checks["errors"]))
    return {s: out[s] for s in splits}


def check_planned_split(dataset_root: Path, manifest: dict[str, Any]) -> None:
    """Si el dataset declara un split planificado (``planned_split`` del generador visual v2),
    el reparto de ``make_split`` debe coincidir: los estilos de cada partición dependen de él."""
    path = Path(dataset_root) / "dataset_manifest.json"
    if not path.is_file():
        return
    plan = json.loads(path.read_text(encoding="utf-8")).get("planned_split")
    if not plan:
        return
    if plan.get("seed") != manifest["seed"]:
        raise SplitError("La semilla no coincide con el split planificado por el generador")
    assignment = {g: s for s in SPLITS for g in manifest["splits"][s]["groups"]}
    listing = "\n".join(f"{g}:{assignment[g]}" for g in sorted(assignment))
    digest = hashlib.sha256(listing.encode()).hexdigest()
    if digest != plan["assignment_sha256"]:
        raise SplitError("El split no coincide con el planificado por el generador (estilos por partición)")
