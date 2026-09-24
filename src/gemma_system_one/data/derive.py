"""Conjuntos derivados para la fase 6 (protocolo ``reports/phase6-protocol.md``).

- ``exclude_overlap``: holdout nuevo sin los grupos cuyo estado aparece literalmente en otro dataset
  (p. ej. el de entrenamiento). Con un repertorio finito, dos semillas pueden coincidir en algún
  estado; una entrada idéntica en train y holdout sería una fuga.
- ``widen_fault_kind``: más opciones en Choice (K hasta ``CHOICE_MAX``) sin cambiar la etiqueta.
  Sólo se añaden categorías que no pueden ser la verdadera: las del generador que no son la real
  (nunca la real en un caso ``other``) y dos categorías que el generador nunca produce como hecho.
  Las opciones originales, sus IDs y la etiqueta no cambian; así la comparación es emparejada.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contracts import CHOICE_MAX, Example
from .dataset import EXAMPLES_FILE, MANIFEST_FILE, sha256_bytes
from .generate_mixed import FAULT_KIND_OPTIONS

# Categorías que ningún estado del generador describe (comprobado: ninguna palabra clave aparece
# en pilot_v3). Tres redacciones, como en FAULT_KIND_OPTIONS.
NEVER_TRUE_FAULT_OPTIONS = {
    "es": {
        "hardware": [
            "Avería de un dispositivo físico (impresora, lector o terminal)",
            "Un equipo físico está estropeado",
            "Fallo de hardware",
        ],
        "install": [
            "Fallo al instalar o actualizar la aplicación",
            "La instalación o la actualización no se completa",
            "Problema de instalación",
        ],
    },
    "en": {
        "hardware": [
            "A physical device is broken (printer, reader or terminal)",
            "Some physical equipment is damaged",
            "Hardware failure",
        ],
        "install": [
            "The app fails to install or update",
            "Installation or update does not complete",
            "Installation problem",
        ],
    },
}


def _state_key(example: Example) -> str:
    return json.dumps(example.state, ensure_ascii=False, sort_keys=True)


def exclude_overlap(examples: list[Example], reference: list[Example]) -> tuple[list[Example], list[str]]:
    """Quita los grupos con algún estado presente en ``reference``; devuelve (conservados, excluidos)."""
    seen = {_state_key(e) for e in reference}
    excluded = sorted({e.group_id for e in examples if _state_key(e) in seen})
    drop = set(excluded)
    return [e for e in examples if e.group_id not in drop], excluded


def _category_of(text: str, lang: str) -> str:
    for cat, variants in FAULT_KIND_OPTIONS[lang].items():
        if text in variants:
            return cat
    raise ValueError(f"opción de fallo desconocida: {text!r}")


def widen_fault_kind(examples: list[Example], seed: int, target_k: int = CHOICE_MAX) -> list[Example]:
    """Preguntas ``fault_type`` con opciones añadidas hasta ``target_k``; el resto se descarta."""
    if not 2 <= target_k <= CHOICE_MAX:
        raise ValueError(f"target_k debe estar entre 2 y {CHOICE_MAX}")
    out: list[Example] = []
    for ex in examples:
        if ex.task_family != "fault_type":
            continue
        rng = random.Random(f"widen:{seed}:{ex.id}")
        lang = ex.language
        criteria = dict(ex.question.criteria)
        present = {_category_of(t, lang) for t in criteria.values()}
        label_cat = _category_of(criteria[ex.target.class_id], lang)
        # La categoría real sólo se conoce si la etiqueta no es "other"; si lo es, la real falta del
        # conjunto y añadir cualquier categoría del generador podría reintroducirla.
        addable = [] if label_cat == "other" else [c for c in FAULT_KIND_OPTIONS[lang] if c not in present]
        extra = [(lang, c, FAULT_KIND_OPTIONS[lang][c]) for c in addable]
        extra += [(lang, c, v) for c, v in NEVER_TRUE_FAULT_OPTIONS[lang].items()]
        rng.shuffle(extra)
        for _, _, variants in extra[: max(0, target_k - len(criteria))]:
            new_id = None
            while new_id is None or new_id in criteria:
                new_id = "o" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
            # Sólo las redacciones de main (la última es de transfer), como el generador.
            criteria[new_id] = rng.choice(variants[:-1])
        raw = ex.model_dump(mode="json")
        raw["question"]["criteria"] = criteria
        out.append(Example.model_validate(raw))
    return out


def write_derived(out_dir: Path, examples: list[Example], manifest: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for e in examples]
    data = ("\n".join(lines) + "\n").encode()
    target = out_dir / EXAMPLES_FILE
    if target.exists() and target.read_bytes() != data:
        raise FileExistsError(f"{target} existe con otro contenido; usa otro directorio")
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    full = {
        **manifest,
        "examples": len(examples),
        "examples_sha256": sha256_bytes(data),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    return full


def widen_fault_kind_by_facts(
    examples: list[Example], audit: list[dict[str, Any]], seed: int, target_k: int = CHOICE_MAX
) -> list[Example]:
    """v4 (revisión de la fase 6b): ampliación a ``target_k`` que no depende de la etiqueta.

    ``widen_fault_kind`` usaba la etiqueta para decidir qué podía añadir: los casos «other» sólo
    admitían dos categorías y K revelaba la respuesta. Aquí se usan los hechos del caso (``audit``
    del generador): se añade cualquier categoría de v4 ausente del conjunto salvo la verdadera, y
    en v4 siempre hay suficientes para llegar a ``target_k``, sea cual sea la etiqueta."""
    from .generate_mixed import fault_categories, true_fault_category

    if not 2 <= target_k <= CHOICE_MAX:
        raise ValueError(f"target_k debe estar entre 2 y {CHOICE_MAX}")
    facts = {a["group_id"]: a["facts"] for a in audit}
    out: list[Example] = []
    for ex in examples:
        if ex.task_family != "fault_type":
            continue
        rng = random.Random(f"widen-v4:{seed}:{ex.id}")
        cats = fault_categories(ex.language)
        text_to_cat = {t: c for c, ts in cats.items() for t in ts}
        criteria = dict(ex.question.criteria)
        present = {text_to_cat[t] for t in criteria.values()}
        true = true_fault_category(facts[ex.group_id])
        addable = [c for c in cats if c not in present and c != true]
        need = target_k - len(criteria)
        if need > len(addable):
            raise ValueError(f"{ex.id}: no hay categorías suficientes para K = {target_k}")
        for c in rng.sample(addable, need):
            new_id = None
            while new_id is None or new_id in criteria:
                new_id = "o" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
            criteria[new_id] = rng.choice(cats[c][:-1])  # redacciones de main
        raw = ex.model_dump(mode="json")
        raw["question"]["criteria"] = criteria
        out.append(Example.model_validate(raw))
    return out


# Fase 6d: cuarta categoría distractora sólo para el diagnóstico de K (no cambia el generador v4).
# Ninguna palabra clave («accesib», «letra», «contraste», «contrast») aparece en los estados de
# pilot_v3, pilot_v4 ni final13; se evita «screen», que sí aparece.
DIAG_DISTRACTOR_OPTIONS = {
    "es": {
        "accessibility": [
            "Problema de accesibilidad (tamaño de letra o contraste)",
            "La accesibilidad de la aplicación no funciona bien",
            "Fallo de accesibilidad",
        ]
    },
    "en": {
        "accessibility": [
            "Accessibility problem (text size or contrast)",
            "The app's accessibility features do not work well",
            "Accessibility failure",
        ]
    },
}
REAL_KINDS = ("network", "performance", "data", "application")


def balanced_fault_kind_pairs(
    examples: list[Example], audit: list[dict[str, Any]], seed: int, other_rate: float = 0.5
) -> tuple[list[Example], list[Example]]:
    """Revisión de la fase 6c: diagnóstico de K cuya composición no depende de la etiqueta.

    Con ``widen_fault_kind_by_facts`` (8 de 9 categorías), los casos «other» eran justo los que
    omitían una categoría real. Aquí, para toda pregunta ``fault_type``:
      - K = 4: ``none``, ``other`` y **dos** categorías reales. En un caso «other» (sorteado con
        ``other_rate`` si hay fallo) ninguna es la verdadera; si no, una es la verdadera y la otra
        al azar. En ambos casos el par de categorías es uniforme sobre los 6 pares posibles.
      - K = 8: el mismo conjunto más las **cuatro** distractoras (siempre las mismas).
    Así la estructura de las opciones es idéntica para cualquier etiqueta; los textos, IDs y el
    orden son aleatorios. Devuelve (K4, K8) con las mismas preguntas, IDs y etiquetas semánticas."""
    from .generate_mixed import FAULT_DISTRACTOR_OPTIONS, FAULT_KIND_OPTIONS, true_fault_category

    facts = {a["group_id"]: a["facts"] for a in audit}
    small, large = [], []
    for ex in examples:
        if ex.task_family != "fault_type":
            continue
        rng = random.Random(f"balanced:{seed}:{ex.id}")
        lang = ex.language
        cats = {**FAULT_KIND_OPTIONS[lang], **FAULT_DISTRACTOR_OPTIONS[lang], **DIAG_DISTRACTOR_OPTIONS[lang]}
        true = true_fault_category(facts[ex.group_id])
        if true != "none" and rng.random() < other_rate:
            label, kinds = "other", rng.sample([k for k in REAL_KINDS if k != true], 2)
        elif true != "none":
            label, kinds = true, [true, rng.choice([k for k in REAL_KINDS if k != true])]
        else:
            label, kinds = "none", rng.sample(list(REAL_KINDS), 2)
        base = ["none", "other", *kinds]
        extra = [*FAULT_DISTRACTOR_OPTIONS[lang], *DIAG_DISTRACTOR_OPTIONS[lang]]
        ids: list[str] = []
        while len(ids) < len(base) + len(extra):
            cand = "o" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
            if cand not in ids:
                ids.append(cand)
        texts = {c: rng.choice(cats[c][:-1]) for c in [*base, *extra]}  # redacciones de main
        k4 = {i: texts[c] for i, c in zip(ids, base, strict=False)}
        k8 = {**k4, **{i: texts[c] for i, c in zip(ids[len(base) :], extra, strict=True)}}
        label_id = ids[base.index(label)]
        for crit, out in ((k4, small), (k8, large)):
            raw = ex.model_dump(mode="json")
            raw["question"]["criteria"] = crit
            raw["target"] = {"class_id": label_id}
            raw["provenance"]["label_method"] = f"balanced_k_diag_v1:{label}"
            out.append(Example.model_validate(raw))
    return small, large


def swap_states(examples: list[Example], seed: int) -> list[Example]:
    """Control «sin leer el estado»: cada pregunta recibe el estado de otro grupo (derangement).

    Opciones y etiquetas no cambian, así que la accuracy que queda mide lo que se acierta sólo con
    la pregunta y las opciones. Nunca se usa para entrenar ni calibrar."""
    groups = sorted({e.group_id for e in examples})
    if len(groups) < 2:
        raise ValueError("Hacen falta al menos dos grupos")
    rng = random.Random(f"swap:{seed}")
    perm = groups[:]
    while any(a == b for a, b in zip(groups, perm, strict=True)):
        rng.shuffle(perm)
    donor = dict(zip(groups, perm, strict=True))
    state_of = {e.group_id: e.state for e in examples}
    return [e.model_copy(update={"state": state_of[donor[e.group_id]]}) for e in examples]
