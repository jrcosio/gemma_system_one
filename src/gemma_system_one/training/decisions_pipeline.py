"""``gso train`` / ``gso evaluate`` para los cabezales Noul/Choice/Score (fase 2; evaluación también fase 3).

Particiones: train ajusta pesos; validation elige época (y λ de RPS comparando runs);
calibration y test no se usan para ajustar ni seleccionar. Test sólo con ``allow_test``.
Un dataset externo (p. ej. transferencia con plantillas reservadas) se evalúa completo
como diagnóstico, tras comprobar que no comparte grupos ni entradas con el de entrenamiento.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..baselines import (
    HashedBowGroupScorer,
    HashedBowLogistic,
    LevelPriorBaseline,
    PriorBaseline,
    UniformChoiceBaseline,
)
from ..checkpoint import load_decision_heads, load_lora_decision, read_manifest, save_decision_heads
from ..config import load_config
from ..contracts import ChoiceQuestion, Example, ScoreQuestion
from ..data.dataset import load_dataset
from ..data.split import leakage_checks, load_split, manifest_sha256, split_path
from ..env import environment_manifest, git_state
from ..features import RepresentationCache, backbone_fingerprint, fingerprint_hash, get_representations
from ..hub import require_snapshot
from ..inference import reconstruct, target_index
from ..metrics import (
    binary_metrics,
    bootstrap_by_group,
    categorical_metrics,
    ordinal_metrics,
    within_group_pairwise,
)
from ..models.encoding import load_processor, processor_fingerprint
from ..models.heads import PRIMITIVES
from ..models.lora import LoraParams
from ..resources import GIB, MemoryTracker
from ..serialization import TEMPLATE_VERSION, expand
from .decisions import (
    DecisionTrainParams,
    QuestionItem,
    all_logits,
    build_items,
    flatten,
    question_nll,
    row_images,
    train_decision_heads,
)
from .lora import LORA_STAGE, LazyLoraEncoder
from .pipeline import ExtractionConfig, LazyEncoder, _write_jsonl

STAGE = "phase2_decision_heads_frozen_backbone"
OVERFIT_MAX_NLL = 0.05
RELOAD_ATOL = 1e-4


class DecisionTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["decision_heads"]
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    base_config: Path
    dataset: Path
    split_seed: int = 0
    primitives: tuple[Literal["noul", "choice", "score"], ...] = PRIMITIVES
    extraction: ExtractionConfig = ExtractionConfig()
    cache_dir: Path | None = Path("artifacts/cache/representations")
    runs_dir: Path = Path("runs")
    train: DecisionTrainParams = DecisionTrainParams()
    # "omit": control sólo texto con las mismas filas sin imagen (decisión 0007).
    images: Literal["use", "omit"] = "use"


def load_decision_config(path: Path) -> DecisionTrainConfig:
    return DecisionTrainConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _filter(examples: list[Example], primitives) -> list[Example]:
    return [e for e in examples if e.question.type in primitives]


def _load(tc: DecisionTrainConfig):
    dataset = load_dataset(tc.dataset)
    sp = split_path(tc.dataset, tc.split_seed)
    if not sp.is_file():
        raise FileNotFoundError(
            f"No existe {sp}. Ejecuta: gso split --dataset {tc.dataset} --seed {tc.split_seed}"
        )
    parts = load_split(dataset, sp, ("train", "validation"))
    train = _filter(parts["train"], tc.primitives)
    if tc.train.max_train_questions is not None:
        train = train[: tc.train.max_train_questions]
    present = {e.question.type for e in train}
    missing = set(tc.primitives) - present
    disabled = {p for p in tc.primitives if tc.train.type_weights[p] <= 0}
    if not tc.primitives or missing or disabled:
        raise ValueError(f"Primitivas sin entrenar: {sorted(missing | disabled)}; revisa subconjunto y pesos")
    val = _filter(parts["validation"], tc.primitives)
    if not val:
        raise ValueError("La partición de validación no contiene preguntas de las primitivas solicitadas")
    mode = getattr(tc, "images", "use")
    return dataset, sp, build_items(train, tc.dataset, mode), build_items(val, tc.dataset, mode)


def _extract(split: str, items: list[QuestionItem], cache, encoder):
    texts, hashes, offsets = flatten(items)
    reps, stats = get_representations(cache, hashes, texts, encoder, images=row_images(items))
    return reps, offsets, {"split": split, "questions": len(items), **stats.__dict__}


# --- métricas ----------------------------------------------------------------


def _probs(primitive: str, z: torch.Tensor) -> list[float]:
    return (
        torch.softmax(z.double(), 0).tolist()
        if primitive != "noul"
        else [float(torch.sigmoid(z[0].double()))]
    )


def evaluate_logits(
    items: list[QuestionItem], logits: list[torch.Tensor], threshold: float
) -> dict[str, Any]:
    out: dict[str, Any] = {"questions": len(items)}
    nll = [question_nll(it.primitive, z, it.target) for it, z in zip(items, logits, strict=True)]
    out["mean_nll_all"] = float(np.mean(nll))
    for prim in PRIMITIVES:
        idx = [i for i, it in enumerate(items) if it.primitive == prim]
        if not idx:
            continue
        sub_items = [items[i] for i in idx]
        if prim == "noul":
            m = binary_metrics([float(logits[i][0]) for i in idx], [it.target for it in sub_items], threshold)
            m.pop("reliability")
            m["within_group_pairwise"] = within_group_pairwise(
                [it.example.group_id for it in sub_items],
                [float(logits[i][0]) for i in idx],
                [it.target for it in sub_items],
            )
        elif prim == "choice":
            m = categorical_metrics([_probs(prim, logits[i]) for i in idx], [it.target for it in sub_items])
        else:
            m = ordinal_metrics([_probs(prim, logits[i]) for i in idx], [it.target for it in sub_items])
        by_family: dict[str, dict[str, float]] = {}
        for fam in sorted({it.example.task_family for it in sub_items}):
            fi = [i for i in idx if items[i].example.task_family == fam]
            correct = [_correct(items[i], logits[i], threshold) for i in fi]
            by_family[fam] = {
                "n": len(fi),
                "nll": float(np.mean([nll[i] for i in fi])),
                "accuracy": float(np.mean(correct)),
            }
        m["by_task_family"] = by_family
        out[prim] = m
    return out


def _correct(it: QuestionItem, z: torch.Tensor, threshold: float) -> float:
    if it.primitive == "noul":
        return float((float(torch.sigmoid(z[0])) >= threshold) == bool(it.target))
    return float(int(torch.argmax(z)) == it.target)


def _bootstrap(
    items, models: dict[str, list[torch.Tensor]], threshold: float, seed: int, reference: str = "bow"
) -> dict[str, Any]:
    """IC por bootstrap de grupos; diferencias emparejadas de cada modelo frente a ``reference``."""
    prim = np.array([it.primitive for it in items])
    per = {
        name: (
            np.array([question_nll(it.primitive, z, it.target) for it, z in zip(items, zs, strict=True)]),
            np.array([_correct(it, z, threshold) for it, z in zip(items, zs, strict=True)]),
        )
        for name, zs in models.items()
    }

    def fn(idx):
        out: dict[str, float | None] = {}
        for name, (nll, corr) in per.items():
            out[f"{name}.nll_all"] = float(nll[idx].mean())
            for p in PRIMITIVES:
                sel = idx[prim[idx] == p]
                out[f"{name}.{p}.nll"] = float(nll[sel].mean()) if len(sel) else None
                out[f"{name}.{p}.accuracy"] = float(corr[sel].mean()) if len(sel) else None
        if reference in per:
            ref_nll, ref_corr = per[reference]
            for name, (nll, corr) in per.items():
                if name == reference:
                    continue
                out[f"{name}_minus_{reference}.nll_all"] = float((nll[idx] - ref_nll[idx]).mean())
                out[f"{name}_minus_{reference}.accuracy_all"] = float((corr[idx] - ref_corr[idx]).mean())
                for p in PRIMITIVES:
                    sel = idx[prim[idx] == p]
                    out[f"{name}_minus_{reference}.{p}.nll"] = (
                        float((nll[sel] - ref_nll[sel]).mean()) if len(sel) else None
                    )
        return out

    return bootstrap_by_group([it.example.group_id for it in items], fn, reps=1000, seed=seed)


# --- baselines --------------------------------------------------------------


def baseline_logits(
    train: list[QuestionItem], splits: dict[str, list[QuestionItem]]
) -> dict[str, dict[str, list]]:
    """Prior y BoW por primitiva, ajustados sólo con train. Mismo formato que los cabezales."""
    by = {p: [it for it in train if it.primitive == p] for p in PRIMITIVES}
    prior_noul = PriorBaseline().fit([it.target for it in by["noul"]])
    prior_score = LevelPriorBaseline().fit(
        [len(it.rows) for it in by["score"]], [it.target for it in by["score"]]
    )
    bow_noul = (
        HashedBowLogistic().fit([it.rows[0].text for it in by["noul"]], [it.target for it in by["noul"]])
        if by["noul"]
        else None
    )
    bow_group = {
        p: HashedBowGroupScorer().fit(
            [[r.text for r in it.rows] for it in by[p]], [it.target for it in by[p]]
        )
        for p in ("choice", "score")
        if by[p]
    }
    out: dict[str, dict[str, list]] = {"prior": {}, "bow": {}}
    for split, items in splits.items():
        prior, bow = [], []
        for it in items:
            if it.primitive == "noul":
                prior.append(torch.tensor(prior_noul.logits(1)))
                bow.append(torch.tensor(bow_noul.logits([it.rows[0].text])) if bow_noul else torch.zeros(1))
            elif it.primitive == "choice":
                prior.append(torch.tensor(UniformChoiceBaseline().logits([len(it.rows)])[0]))
                g = bow_group.get("choice")
                bow.append(
                    torch.tensor(g.logits([[r.text for r in it.rows]])[0]) if g else torch.zeros(len(it.rows))
                )
            else:
                prior.append(torch.tensor(prior_score.logits([len(it.rows)])[0]))
                g = bow_group.get("score")
                bow.append(
                    torch.tensor(g.logits([[r.text for r in it.rows]])[0]) if g else torch.zeros(len(it.rows))
                )
        out["prior"][split], out["bow"][split] = prior, bow
    return out


def _predictions(
    items: list[QuestionItem],
    logits: list[torch.Tensor],
    threshold: float,
    temperatures: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Predicción auditable por pregunta. ``row_logits`` son crudos; con temperaturas, la
    respuesta, ``nll`` y ``correct`` usan ``z/T`` y se añade ``nll_uncalibrated``."""
    rows = []
    for it, z in zip(items, logits, strict=True):
        e = it.example
        t = 1.0 if temperatures is None else temperatures[it.primitive]
        answer = reconstruct(e.question, it.rows, z, temperature=t)
        zt = z / t
        extra = (
            {}
            if temperatures is None
            else {
                "temperature": t,
                "nll_uncalibrated": question_nll(it.primitive, z, it.target),
            }
        )
        rows.append(
            {
                "id": e.id,
                "group_id": e.group_id,
                "task_family": e.task_family,
                "language": e.language,
                "type": it.primitive,
                "target_index": it.target,
                "input_sha256": _rows_sha256(it.rows),
                "target_description": _target_description(it),
                "row_logits": [float(x) for x in z],
                "nll": question_nll(it.primitive, zt, it.target),
                "correct": bool(_correct(it, zt, threshold)),
                "answer": answer,
                **extra,
            }
        )
    return rows


def _rows_sha256(rows) -> str:
    """Huella de las filas que ve el modelo (texto y, si hay, imagen), en orden."""
    return hashlib.sha256("\n".join(r.sha256 for r in rows).encode()).hexdigest()


def _target_description(it: QuestionItem) -> Any:
    q = it.example.question
    if isinstance(q, ChoiceQuestion):
        return q.criteria[it.rows[it.target].candidate_id]
    if isinstance(q, ScoreQuestion):
        return q.criteria[it.target]
    return it.target


# --- entrenamiento ----------------------------------------------------------


def run_train_decisions(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    tc = load_decision_config(config_path)
    cfg = load_config(tc.base_config)
    tracker = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
    tracker.sample("start")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = tc.runs_dir / tc.name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, run_dir / "train_config.yaml")
    env = environment_manifest()  # incluye la copia de las fuentes al empezar (decisión 0008)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")

    dataset, sp, train, val = _load(tc)
    snapshot = require_snapshot(cfg)
    processor = load_processor(snapshot)
    fp = backbone_fingerprint(cfg, snapshot, tc.extraction.microbatch_rows)
    cache = RepresentationCache(tc.cache_dir, fp) if tc.cache_dir is not None else None
    encoder = LazyEncoder(cfg, snapshot, processor, tc.extraction.microbatch_rows, tracker)
    r_train, o_train, ext_train = _extract("train", train, cache, encoder)
    r_val, o_val, ext_val = _extract("validation", val, cache, encoder)
    if encoder.backbone is not None:
        encoder.backbone = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    result = train_decision_heads(train, r_train, o_train, tc.train, (val, r_val, o_val))
    head = {
        "train": all_logits(result.heads, train, r_train, o_train),
        "validation": all_logits(result.heads, val, r_val, o_val),
    }
    models = {"gemma_heads": head, **baseline_logits(train, {"train": train, "validation": val})}
    thr = tc.train.threshold
    metrics: dict[str, Any] = {
        "threshold": thr,
        "models": {
            name: {
                s: evaluate_logits(items, logits[s], thr)
                for s, items in (("train", train), ("validation", val))
            }
            for name, logits in models.items()
        },
    }
    metrics["validation_bootstrap"] = _bootstrap(
        val, {k: v["validation"] for k, v in models.items()}, thr, tc.train.seed
    )
    if tc.train.select_on == "train":
        tm = metrics["models"]["gemma_heads"]["train"]
        per_type = {p: {"accuracy": tm[p]["accuracy"], "nll": tm[p]["nll"]} for p in PRIMITIVES if p in tm}
        metrics["overfit_check"] = {
            "criterion": f"por primitiva: train accuracy == 1.0 y train NLL < {OVERFIT_MAX_NLL}",
            "train_questions": len(train),
            "per_type": per_type,
            "passed": bool(per_type)
            and all(v["accuracy"] == 1.0 and v["nll"] < OVERFIT_MAX_NLL for v in per_type.values()),
        }
    ckpt = save_decision_heads(
        run_dir / "checkpoint",
        result.heads,
        repo_id=cfg.model.repo_id,
        revision=cfg.model.revision,
        backbone_dtype=cfg.model.dtype,
        prompt_template=TEMPLATE_VERSION,
        extra={
            "stage": STAGE,
            "train_config": tc.name,
            "base_config": str(tc.base_config),
            "primitives": list(tc.primitives),
            "trained_primitives": sorted({it.primitive for it in train}),
            "fingerprint": fp,
            "fingerprint_sha256": fingerprint_hash(fp),
            "dataset_root": str(tc.dataset),
            "cache_dir": None if tc.cache_dir is None else str(tc.cache_dir),
            "dataset_sha256": dataset.sha256,
            "split_path": str(sp),
            "split_manifest_sha256": manifest_sha256(sp),
            "split_seed": tc.split_seed,
            "train_examples": [it.example.id for it in train],
            "threshold": thr,
            "temperatures": {p: 1.0 for p in PRIMITIVES},
            "calibrated": False,
            "train_params": tc.train.model_dump(),
            "selected_epoch": result.selected_epoch,
            "select_on": tc.train.select_on,
            "head_training_device": "cpu",
            "feature_norm": tc.train.feature_norm,
            "feature_norm_fitted_on": "train" if tc.train.feature_norm != "none" else None,
            "images": tc.images,
            "input_modality": "text+image" if any(it.image_file is not None for it in train) else "text",
            "run_start_code": env["git"],
        },
    )
    _write_jsonl(run_dir / "history.jsonl", result.history)
    for split, items in (("train", train), ("validation", val)):
        _write_jsonl(run_dir / f"predictions_{split}.jsonl", _predictions(items, head[split], thr))
    tracker.sample("end")
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "stage": STAGE,
        "selected_epoch": result.selected_epoch,
        "select_on": tc.train.select_on,
        "splits_used_for_fitting": ["train", "validation"],
        "splits_not_used_for_fitting": ["calibration", "test"],
        "counts": {"train": len(train), "validation": len(val)},
        "extraction": [ext_train, ext_val],
        "metrics": metrics,
        "memory": {k: v for k, v in tracker.summary().items() if k != "samples"},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report


# --- evaluación y robustez --------------------------------------------------


def _renamed_permuted(q: ChoiceQuestion, rng: random.Random) -> tuple[ChoiceQuestion, dict[str, str]]:
    keys = list(q.criteria)
    rng.shuffle(keys)
    mapping = {k: f"r{i:02d}x{rng.randrange(16**4):04x}" for i, k in enumerate(keys)}
    return ChoiceQuestion(
        type="choice", instructions=q.instructions, criteria={mapping[k]: q.criteria[k] for k in keys}
    ), mapping


def _robustness(items, logits, heads, encoder, cache, threshold, seed) -> dict[str, Any]:
    """(a) IDs renombrados + mapa permutado; (b) rúbrica invertida; (c) Choice sin un distractor."""
    rng = random.Random(seed)
    report: dict[str, Any] = {}
    # (a) Debe ser idéntico: las filas no contienen IDs y el orden es canónico.
    diffs, identical, n = [], 0, 0
    for it, z in zip(items, logits, strict=True):
        if it.primitive != "choice":
            continue
        n += 1
        q2, mapping = _renamed_permuted(it.example.question, rng)
        rows2 = expand(it.example.state, q2, it.rows[0].image)
        identical += [r.text for r in rows2] == [r.text for r in it.rows]
        p1 = reconstruct(it.example.question, it.rows, z)["probabilities"]
        by_text = {r.text: zi for r, zi in zip(it.rows, z, strict=True)}
        z2 = torch.stack([by_text[r.text] for r in rows2]) if all(r.text in by_text for r in rows2) else None
        if z2 is not None:
            p2 = reconstruct(q2, rows2, z2)["probabilities"]
            diffs.append(max(abs(p1[k] - p2[mapping[k]]) for k in p1))
    report["choice_id_rename_permutation"] = {
        "questions": n,
        "rows_identical": identical,
        "max_prob_diff_remapped": max(diffs) if diffs else None,
    }
    # (b) y (c) requieren filas nuevas.
    variants: list[tuple[str, QuestionItem, QuestionItem]] = []
    for it in items:
        q = it.example.question
        if it.primitive == "score":
            q2 = ScoreQuestion(type="score", instructions=q.instructions, criteria=list(reversed(q.criteria)))
            rows = expand(it.example.state, q2, it.rows[0].image)
            variants.append(
                (
                    "score_reversed",
                    it,
                    QuestionItem(it.example, rows, len(rows) - 1 - it.target, it.image_file),
                )
            )
        elif it.primitive == "choice" and len(q.criteria) >= 3:
            target_id = it.rows[it.target].candidate_id
            drop = rng.choice([k for k in q.criteria if k != target_id])
            q2 = ChoiceQuestion(
                type="choice",
                instructions=q.instructions,
                criteria={k: v for k, v in q.criteria.items() if k != drop},
            )
            rows = expand(it.example.state, q2, it.rows[0].image)
            t2 = next(i for i, r in enumerate(rows) if r.candidate_id == target_id)
            variants.append(("choice_drop_distractor", it, QuestionItem(it.example, rows, t2, it.image_file)))
    if not variants:
        return report
    v_items = [v[2] for v in variants]
    texts, hashes, offsets = flatten(v_items)
    reps, stats = get_representations(cache, hashes, texts, encoder, images=row_images(v_items))
    report["extraction"] = stats.__dict__
    v_logits = all_logits(heads, v_items, reps, offsets)
    orig = {id(it): z for it, z in zip(items, logits, strict=True)}
    for kind in ("score_reversed", "choice_drop_distractor"):
        sel = [(o, v, z) for (k, o, v), z in zip(variants, v_logits, strict=True) if k == kind]
        if not sel:
            continue
        acc_o = np.mean([_correct(o, orig[id(o)], threshold) for o, _, _ in sel])
        acc_v = np.mean([_correct(v, z, threshold) for _, v, z in sel])
        entry = {
            "questions": len(sel),
            "accuracy_original": float(acc_o),
            "accuracy_variant": float(acc_v),
            "nll_original": float(
                np.mean([question_nll(o.primitive, orig[id(o)], o.target) for o, _, _ in sel])
            ),
            "nll_variant": float(np.mean([question_nll(v.primitive, z, v.target) for _, v, z in sel])),
        }
        if kind == "choice_drop_distractor":
            same = 0
            for o, v, z in sel:
                pred_o = o.rows[int(torch.argmax(orig[id(o)]))].candidate_id
                pred_v = v.rows[int(torch.argmax(z))].candidate_id
                same += pred_o == pred_v
            entry["prediction_unchanged"] = same / len(sel)
        report[kind] = entry
    return report


def _by_primitive(pairs, threshold) -> dict[str, Any]:
    """Accuracy/NLL original frente a variante, en total y por primitiva."""
    out: dict[str, Any] = {}
    for name in ("all", *PRIMITIVES):
        sel = [(o, zo, v, zv) for o, zo, v, zv in pairs if name == "all" or o.primitive == name]
        if not sel:
            continue
        out[name] = {
            "questions": len(sel),
            "accuracy_original": float(np.mean([_correct(o, zo, threshold) for o, zo, _, _ in sel])),
            "accuracy_variant": float(np.mean([_correct(v, zv, threshold) for _, _, v, zv in sel])),
            "nll_original": float(np.mean([question_nll(o.primitive, zo, o.target) for o, zo, _, _ in sel])),
            "nll_variant": float(np.mean([question_nll(v.primitive, zv, v.target) for _, _, v, zv in sel])),
            "prediction_unchanged": float(
                np.mean(
                    [_prediction(o, zo, threshold) == _prediction(v, zv, threshold) for o, zo, v, zv in sel]
                )
            ),
        }
    return out


def _prediction(it: QuestionItem, z: torch.Tensor, threshold: float):
    if it.primitive == "noul":
        return float(torch.sigmoid(z[0])) >= threshold
    return it.rows[int(torch.argmax(z))].candidate_id or int(torch.argmax(z))


def _vision_ablation(items, logits, heads, encoder, cache, threshold, seed) -> dict[str, Any]:
    """Uso de la imagen (spec §7.C, §8): la misma pregunta con la imagen omitida o con la de otro
    grupo. El texto de las filas no cambia; si el rendimiento no cae, hay un atajo textual."""
    idx = [i for i, it in enumerate(items) if it.image_file is not None]
    if not idx:
        return {"questions_with_image": 0}
    rng = random.Random(seed)
    donors = sorted({(items[i].example.group_id, items[i].rows[0].image, items[i].image_file) for i in idx})
    variants: list[tuple[str, int, QuestionItem]] = []
    for i in idx:
        it = items[i]
        q, state = it.example.question, it.example.state
        rows = expand(state, q, None)
        variants.append(
            ("image_omitted", i, QuestionItem(it.example, rows, target_index(q, rows, it.example.target)))
        )
        pool = [d for d in donors if d[0] != it.example.group_id and d[1] != it.rows[0].image]
        if pool:
            _, rel, path = rng.choice(pool)
            rows = expand(state, q, rel)
            swapped = QuestionItem(it.example, rows, target_index(q, rows, it.example.target), path)
            variants.append(("image_swapped", i, swapped))
    v_items = [v for _, _, v in variants]
    texts, hashes, offsets = flatten(v_items)
    reps, stats = get_representations(cache, hashes, texts, encoder, images=row_images(v_items))
    v_logits = all_logits(heads, v_items, reps, offsets)
    report: dict[str, Any] = {"questions_with_image": len(idx), "seed": seed, "extraction": stats.__dict__}
    for kind in ("image_omitted", "image_swapped"):
        pairs = [
            (items[i], logits[i], v, z) for (k, i, v), z in zip(variants, v_logits, strict=True) if k == kind
        ]
        report[kind] = _by_primitive(pairs, threshold)
    report["note"] = (
        "image_swapped usa la imagen de otro grupo de la misma partición y conserva la etiqueta original: "
        "un modelo que mira la imagen debe empeorar; la etiqueta correcta para la imagen donante no se evalúa"
    )
    return report


def _external_leakage(train_dataset, examples: list[Example]) -> dict[str, Any]:
    checks = leakage_checks({"training_dataset": train_dataset.examples, "external": examples})
    total = checks.get("near_duplicate_states_total", len(checks["near_duplicate_states"]))
    return {"errors": checks["errors"], "near_duplicate_states": total}


@dataclass
class EvalContext:
    """Checkpoint verificado y listo para producir logits desde el texto (fase 2 o 3)."""

    checkpoint: Path
    manifest: dict[str, Any]
    extra: dict[str, Any]
    cfg: Any
    train_ds: Any
    split_path: Path
    heads: Any
    encoder: Any
    cache: RepresentationCache | None

    def items_for(self, split: str, *, allow_test: bool = False, dataset: Path | None = None):
        root = self.train_ds.root
        if dataset is not None:
            ext = load_dataset(dataset)
            examples, root = ext.examples, ext.root
        else:
            examples = load_split(self.train_ds, self.split_path, (split,), allow_test=allow_test)[split]
        return build_items(_filter(examples, self.extra["primitives"]), root, self.extra.get("images", "use"))

    def logits(self, split: str, items: list[QuestionItem]):
        reps, offsets, ext = _extract(split, items, self.cache, self.encoder)
        return all_logits(self.heads, items, reps, offsets), ext


@dataclass
class LoadedModel:
    """Checkpoint verificado contra la base local, sin necesidad del dataset (evaluación y servicio)."""

    checkpoint: Path
    manifest: dict[str, Any]
    extra: dict[str, Any]
    cfg: Any
    fingerprint: dict[str, Any]
    heads: Any
    encoder: Any


def load_decision_model(checkpoint: Path) -> LoadedModel:
    """Verifica etapa, primitivas entrenadas, plantilla, procesador y huella; carga cabezales (+LoRA).

    Fase 3 (LoRA): el adaptador se inyecta al cargar la base. La base se carga perezosamente.
    """
    checkpoint = Path(checkpoint)
    manifest = read_manifest(checkpoint)
    extra = manifest.get("extra", {})
    stage = extra.get("stage")
    if stage not in (STAGE, LORA_STAGE):
        raise ValueError(f"Checkpoint de otra etapa: {stage}")
    configured = set(extra.get("primitives", []))
    trained = set(extra.get("trained_primitives", []))
    if not configured or configured - trained:
        raise ValueError(f"Checkpoint con primitivas sin entrenar: {sorted(configured - trained)}")
    if manifest["prompt_template"] != TEMPLATE_VERSION:
        raise ValueError(f"Plantilla {manifest['prompt_template']} != actual {TEMPLATE_VERSION}")
    cfg = load_config(extra["base_config"])
    snapshot = require_snapshot(cfg)
    if processor_fingerprint(snapshot) != extra["fingerprint"]["processor"]:
        raise ValueError("El procesador local no coincide con el del checkpoint")
    fp = backbone_fingerprint(cfg, snapshot, extra["fingerprint"]["extraction"]["microbatch_rows"])
    if fingerprint_hash(fp) != extra["fingerprint_sha256"]:
        raise ValueError("La huella de extracción no coincide con el checkpoint")
    tracker = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
    processor = load_processor(snapshot)
    mb = fp["extraction"]["microbatch_rows"]
    kw = {
        "repo_id": cfg.model.repo_id,
        "revision": cfg.model.revision,
        "hidden_size": manifest["hidden_size"],
    }
    if stage == LORA_STAGE:
        heads, adapter_state, _ = load_lora_decision(checkpoint, **kw)
        adapter = manifest["adapter"]["config"]
        encoder = LazyLoraEncoder(
            cfg,
            snapshot,
            processor,
            mb,
            tracker,
            LoraParams(**adapter["params"]),
            adapter_state,
            adapter["targets"],
        )
    else:
        heads, _ = load_decision_heads(checkpoint, **kw)
        encoder = LazyEncoder(cfg, snapshot, processor, mb, tracker)
    return LoadedModel(checkpoint, manifest, extra, cfg, fp, heads, encoder)


def prepare_evaluation(checkpoint: Path, *, use_cache: bool = True) -> EvalContext:
    """``load_decision_model`` + dataset y split del entrenamiento sin cambios; caché sólo sin LoRA."""
    m = load_decision_model(checkpoint)
    extra = m.extra
    train_ds = load_dataset(extra["dataset_root"])
    if train_ds.sha256 != extra["dataset_sha256"]:
        raise ValueError("El dataset ha cambiado desde el entrenamiento")
    sp = Path(extra["split_path"])
    if manifest_sha256(sp) != extra["split_manifest_sha256"]:
        raise ValueError("El manifiesto de split ha cambiado desde el entrenamiento")
    cache = None
    if extra["stage"] != LORA_STAGE and use_cache and extra.get("cache_dir") is not None:
        cache = RepresentationCache(Path(extra["cache_dir"]), m.fingerprint)
    return EvalContext(m.checkpoint, m.manifest, extra, m.cfg, train_ds, sp, m.heads, m.encoder, cache)


def _reload_check(checkpoint: Path, split: str, items, logits) -> dict[str, Any] | None:
    saved = checkpoint.parent / f"predictions_{split}.jsonl"
    if not saved.is_file():
        return None
    before = {
        r["id"]: r["row_logits"] for r in map(json.loads, saved.read_text(encoding="utf-8").splitlines())
    }
    diffs = [
        float(np.max(np.abs(np.array(before[it.example.id]) - z.numpy())))
        for it, z in zip(items, logits, strict=True)
        if it.example.id in before
    ]
    return {
        "compared_questions": len(diffs),
        "max_abs_logit_diff": max(diffs) if diffs else None,
        "atol": RELOAD_ATOL,
        "ok": bool(diffs) and max(diffs) <= RELOAD_ATOL,
    }


def run_evaluate_decisions(
    checkpoint: Path,
    split: str,
    *,
    allow_test: bool = False,
    use_cache: bool = True,
    dataset: Path | None = None,
    robustness: bool = False,
    calibration: Path | None = None,
    baselines: bool = False,
    vision_ablation: bool = False,
) -> dict[str, Any]:
    from ..calibration import apply_temperatures, load_calibration

    checkpoint = Path(checkpoint)
    ctx = prepare_evaluation(checkpoint, use_cache=use_cache)
    extra = ctx.extra
    temps = None
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "stage": extra["stage"],
        "split": split,
        "use_cache": ctx.cache is not None,
        "code": git_state(snapshot=True),  # código que evalúa (puede diferir del que entrenó), con copia
    }
    if calibration is not None:
        cal = load_calibration(calibration, checkpoint)
        temps = cal["temperatures"]
        report["calibration"] = {
            "file": str(calibration),
            "temperatures": temps,
            "calibrated": cal["calibrated"],
        }
    if dataset is not None:
        if split != "all":
            raise ValueError("Con --dataset externo se evalúa completo: usa --split all")
        ext_ds = load_dataset(dataset)
        cal_ext = (cal.get("external_dataset") or {}) if calibration is not None else {}
        if cal_ext.get("sha256") == ext_ds.sha256:
            raise ValueError("No se evalúa sobre el mismo conjunto con el que se ajustaron las temperaturas")
        if cal_ext:
            cal_ds = load_dataset(cal_ext["root"])
            if cal_ds.sha256 != cal_ext["sha256"]:
                raise ValueError("El conjunto de calibración externa cambió desde el ajuste")
            cal_overlap = _external_leakage(cal_ds, ext_ds.examples)
            if cal_overlap["errors"]:
                raise ValueError(
                    f"El dataset de evaluación comparte grupos o entradas con la calibración: "
                    f"{cal_overlap['errors']}"
                )
            report["external_calibration_leakage"] = cal_overlap
        report["external_dataset"] = {"root": str(dataset), "sha256": ext_ds.sha256}
        report["external_leakage"] = _external_leakage(ctx.train_ds, ext_ds.examples)
        if report["external_leakage"]["errors"]:
            raise ValueError(
                f"El dataset externo comparte grupos o entradas: {report['external_leakage']['errors']}"
            )
        report["diagnostic_only"] = True
    else:
        report["final_test_used"] = split == "test"
    items = ctx.items_for(split, allow_test=allow_test, dataset=dataset)
    logits, ext = ctx.logits(split, items)
    thr = extra["threshold"]
    report["backbone_loaded"] = ctx.encoder.backbone is not None
    report["extraction"] = ext
    report["metrics"] = evaluate_logits(items, logits, thr)
    scored = logits
    if temps is not None:
        scored = apply_temperatures(items, logits, temps)
        report["metrics_calibrated"] = evaluate_logits(items, scored, thr)
    if baselines:
        train_items = build_items(
            _filter(load_split(ctx.train_ds, ctx.split_path, ("train",))["train"], extra["primitives"]),
            ctx.train_ds.root,
            extra.get("images", "use"),
        )
        base = baseline_logits(train_items, {split: items})
        report["baselines"] = {
            "fitted_on": "train",
            **{name: evaluate_logits(items, zs[split], thr) for name, zs in base.items()},
        }
        report["bootstrap"] = _bootstrap(
            items, {"checkpoint": scored, **{k: v[split] for k, v in base.items()}}, thr, 0
        )
    report["usage"] = {
        "questions": len(items),
        "expanded_rows": sum(len(it.rows) for it in items),
        "backbone_forwards": ext["backbone_forwards"],
        "processed_input_tokens": ext["valid_tokens"],
        "generated_tokens": 0,
    }
    if dataset is None:
        reload = _reload_check(checkpoint, split, items, logits)
        if reload is not None:
            report["reload_vs_training"] = reload
    if robustness:
        report["robustness"] = _robustness(items, logits, ctx.heads, ctx.encoder, ctx.cache, thr, seed=0)
    if vision_ablation:
        report["vision_ablation"] = _vision_ablation(
            items, logits, ctx.heads, ctx.encoder, ctx.cache, thr, seed=0
        )
    report["memory"] = {k: v for k, v in ctx.encoder.tracker.summary().items() if k != "samples"}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = checkpoint.parent / "evaluations"
    out_dir.mkdir(exist_ok=True)
    tag = f"{Path(dataset).name}-all" if dataset is not None else split
    tag += "-calibrated" if temps is not None else ""
    report["report_path"] = str(out_dir / f"{tag}-{stamp}.json")
    report["predictions_path"] = str(out_dir / f"{tag}-{stamp}-predictions.jsonl")
    Path(report["report_path"]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _write_jsonl(Path(report["predictions_path"]), _predictions(items, logits, thr, temps))
    return report
