"""Orquestación de ``gso train`` / ``gso evaluate`` / ``gso baselines`` para Noul (fase 1).

Separación de particiones: entrenamiento ajusta pesos; validación elige época y se
usa para comparar con baselines; calibración y test no se usan para ajustar ni seleccionar en esta fase
(test sólo con ``allow_test`` explícito en ``evaluate``).
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..baselines import HashedBowLogistic, PriorBaseline
from ..checkpoint import load_head, read_manifest, save_head
from ..config import ProjectConfig, load_config
from ..contracts import Example
from ..data.dataset import load_dataset
from ..data.split import load_split, manifest_sha256, split_path
from ..env import environment_manifest
from ..features import (
    RepresentationCache,
    backbone_fingerprint,
    extract_pooled,
    fingerprint_hash,
    get_representations,
)
from ..hub import require_snapshot
from ..metrics import (
    binary_metrics,
    binary_nll_from_logits,
    bootstrap_by_group,
    sigmoid,
    within_group_pairwise,
)
from ..models.backbone import load_backbone
from ..models.encoding import load_processor, processor_fingerprint
from ..resources import GIB, MemoryTracker
from ..serialization import TEMPLATE_VERSION, expand_example
from .noul import NoulTrainParams, predict_logits, train_noul_head

STAGE = "phase1_noul_frozen_backbone"
OVERFIT_MAX_NLL = 0.05  # criterio declarado de la prueba de sobreajuste (junto a accuracy = 1)
RELOAD_ATOL = 1e-4  # tolerancia de logits al recargar y recalcular desde el texto


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # 1 = representación independiente de la composición del lote (decisión 0002). Con >1,
    # en bf16/MPS el padding cambia la representación (coseno ≥ 0,9995; Δp medida ≤ 0,056).
    microbatch_rows: int = Field(default=1, ge=1)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    base_config: Path
    dataset: Path
    split_seed: int = 0
    primitive: Literal["noul"] = "noul"
    extraction: ExtractionConfig = ExtractionConfig()
    cache_dir: Path | None = Path("artifacts/cache/representations")
    runs_dir: Path = Path("runs")
    train: NoulTrainParams = NoulTrainParams()


def load_train_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _noul_only(examples: list[Example]) -> list[Example]:
    return [e for e in examples if e.question.type == "noul"]


def _rows(examples: list[Example]) -> tuple[list[str], list[str]]:
    texts, hashes = [], []
    for e in examples:
        if e.image_path is not None:
            raise ValueError("La fase 1 sólo admite texto; no se puede ignorar una imagen")
        (row,) = expand_example(e)
        texts.append(row.text)
        hashes.append(row.sha256)
    return texts, hashes


def _labels(examples: list[Example]) -> torch.Tensor:
    return torch.tensor([e.target.label for e in examples], dtype=torch.float32)


class LazyEncoder:
    """Carga el backbone sólo si la caché no tiene las representaciones."""

    def __init__(
        self, cfg: ProjectConfig, snapshot: Path, processor, microbatch_rows: int, tracker: MemoryTracker
    ):
        self.cfg, self.snapshot, self.processor = cfg, snapshot, processor
        self.microbatch_rows, self.tracker = microbatch_rows, tracker
        self.backbone = None
        self.extractions: list[dict[str, Any]] = []

    def __call__(self, texts: list[str], images: list | None = None):
        if self.backbone is None:
            self.backbone = load_backbone(self.cfg, self.snapshot)
            self.tracker.sample("backbone_loaded")
        pooled, stats = extract_pooled(
            self.backbone,
            self.processor,
            texts,
            max_length=self.cfg.runtime.max_length,
            microbatch_rows=self.microbatch_rows,
            images=images,
        )
        self.tracker.sample("extraction_done")
        return pooled, stats


def _representations(split_name, examples, cache, encoder) -> tuple[torch.Tensor, dict[str, Any]]:
    texts, hashes = _rows(examples)
    pooled, stats = get_representations(cache, hashes, texts, encoder)
    info = {"split": split_name, **stats.__dict__}
    return pooled, info


def _predictions(examples: list[Example], logits: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "id": e.id,
            "group_id": e.group_id,
            "task_family": e.task_family,
            "language": e.language,
            "label": e.target.label,
            "logit": float(z),
            "p": float(sigmoid(np.float64(z))),
        }
        for e, z in zip(examples, logits, strict=True)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k != "reliability"}


def _bootstrap(examples, models: dict[str, list[float]], seed: int) -> dict[str, Any]:
    y = np.array([e.target.label for e in examples])
    arrays = {k: np.asarray(v, np.float64) for k, v in models.items()}

    def fn(idx):
        out: dict[str, float | None] = {}
        for name, z in arrays.items():
            out[f"{name}.nll"] = binary_nll_from_logits(z[idx], y[idx])
            out[f"{name}.brier"] = float(np.mean((sigmoid(z[idx]) - y[idx]) ** 2))
            out[f"{name}.accuracy"] = float(np.mean((sigmoid(z[idx]) >= 0.5) == y[idx]))
        if "gemma_head" in arrays and "bow_logistic" in arrays:
            out["gemma_head_minus_bow.nll"] = out["gemma_head.nll"] - out["bow_logistic.nll"]
        return out

    return bootstrap_by_group([e.group_id for e in examples], fn, reps=1000, seed=seed)


def _load_splits(tc: TrainConfig):
    dataset = load_dataset(tc.dataset)
    sp = split_path(tc.dataset, tc.split_seed)
    if not sp.is_file():
        raise FileNotFoundError(
            f"No existe {sp}. Ejecuta: gso split --dataset {tc.dataset} --seed {tc.split_seed}"
        )
    parts = load_split(dataset, sp, ("train", "validation"))
    train = _noul_only(parts["train"])
    if tc.train.max_train_examples is not None:
        train = train[: tc.train.max_train_examples]
    return dataset, sp, train, _noul_only(parts["validation"])


def _baselines(train, val) -> dict[str, dict[str, list[float]]]:
    y_train = [e.target.label for e in train]
    t_train, _ = _rows(train)
    t_val, _ = _rows(val)
    prior = PriorBaseline().fit(y_train)
    bow = HashedBowLogistic().fit(t_train, y_train)
    return {
        "prior": {"train": prior.logits(len(train)), "validation": prior.logits(len(val))},
        "bow_logistic": {"train": bow.logits(t_train), "validation": bow.logits(t_val)},
    }


def run_baselines(config_path: Path) -> dict[str, Any]:
    tc = load_train_config(config_path)
    _, _, train, val = _load_splits(tc)
    base = _baselines(train, val)
    y_val = [e.target.label for e in val]
    return {
        name: _summary(binary_metrics(logits["validation"], y_val, tc.train.threshold))
        for name, logits in base.items()
    }


def run_train(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    tc = load_train_config(config_path)
    cfg = load_config(tc.base_config)
    tracker = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
    tracker.sample("start")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = tc.runs_dir / tc.name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, run_dir / "train_config.yaml")
    (run_dir / "env.json").write_text(
        json.dumps(environment_manifest(), indent=2, default=str), encoding="utf-8"
    )

    dataset, sp, train, val = _load_splits(tc)
    snapshot = require_snapshot(cfg)
    processor = load_processor(snapshot)
    fp = backbone_fingerprint(cfg, snapshot, tc.extraction.microbatch_rows)
    cache = RepresentationCache(tc.cache_dir, fp) if tc.cache_dir is not None else None
    encoder = LazyEncoder(cfg, snapshot, processor, tc.extraction.microbatch_rows, tracker)

    reps_train, ext_train = _representations("train", train, cache, encoder)
    reps_val, ext_val = _representations("validation", val, cache, encoder)
    if encoder.backbone is not None:  # liberar memoria MPS: el cabezal se entrena en CPU
        del encoder.backbone
        encoder.backbone = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    y_train, y_val = _labels(train), _labels(val)
    result = train_noul_head(reps_train, y_train, tc.train, reps_val, y_val)
    head_logits = {
        "train": predict_logits(result.head, reps_train).tolist(),
        "validation": predict_logits(result.head, reps_val).tolist(),
    }
    models = {"gemma_head": head_logits, **_baselines(train, val)}
    thr = tc.train.threshold
    metrics: dict[str, Any] = {"threshold": thr, "models": {}}
    for name, logits in models.items():
        metrics["models"][name] = {
            "train": binary_metrics(logits["train"], y_train.int().tolist(), thr),
            "validation": binary_metrics(logits["validation"], y_val.int().tolist(), thr),
            "validation_within_group_pairwise": within_group_pairwise(
                [e.group_id for e in val], logits["validation"], y_val.int().tolist()
            ),
        }
    metrics["validation_bootstrap"] = _bootstrap(
        val, {k: v["validation"] for k, v in models.items()}, seed=tc.train.seed
    )
    train_m = metrics["models"]["gemma_head"]["train"]
    if tc.train.select_on == "train":
        metrics["overfit_check"] = {
            "criterion": f"train accuracy == 1.0 y train NLL < {OVERFIT_MAX_NLL}",
            "train_examples": len(train),
            "train_accuracy": train_m["accuracy"],
            "train_nll": train_m["nll"],
            "passed": train_m["accuracy"] == 1.0 and train_m["nll"] < OVERFIT_MAX_NLL,
        }

    ckpt = save_head(
        run_dir / "checkpoint",
        result.head,
        repo_id=cfg.model.repo_id,
        revision=cfg.model.revision,
        backbone_dtype=cfg.model.dtype,
        prompt_template=TEMPLATE_VERSION,
        extra={
            "stage": STAGE,
            "train_config": tc.name,
            "base_config": str(tc.base_config),
            "fingerprint": fp,
            "fingerprint_sha256": fingerprint_hash(fp),
            "dataset_root": str(tc.dataset),
            "cache_dir": None if tc.cache_dir is None else str(tc.cache_dir),
            "dataset_sha256": dataset.sha256,
            "split_path": str(sp),
            "split_manifest_sha256": manifest_sha256(sp),
            "split_seed": tc.split_seed,
            "train_examples": [e.id for e in train],
            "threshold": thr,
            "train_params": tc.train.model_dump(),
            "selected_epoch": result.selected_epoch,
            "select_on": tc.train.select_on,
            "head_training_device": "cpu",
            "validation_metrics": _summary(metrics["models"]["gemma_head"]["validation"]),
        },
    )
    _write_jsonl(run_dir / "history.jsonl", result.history)
    for split, exs in (("train", train), ("validation", val)):
        _write_jsonl(run_dir / f"predictions_{split}.jsonl", _predictions(exs, head_logits[split]))
    tracker.sample("end")
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "stage": STAGE,
        "selected_epoch": result.selected_epoch,
        "select_on": tc.train.select_on,
        "splits_used": ["train", "validation"],
        "splits_not_used_for_fitting": ["calibration", "test"],
        "counts": {"train": len(train), "validation": len(val)},
        "extraction": [ext_train, ext_val],
        "metrics": metrics,
        "memory": {k: v for k, v in tracker.summary().items() if k != "samples"},
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_evaluate(
    checkpoint: Path, split: str, *, allow_test: bool = False, use_cache: bool = True
) -> dict[str, Any]:
    """Recarga el checkpoint y recalcula desde el texto original (backbone + cabezal)."""
    checkpoint = Path(checkpoint)
    manifest = read_manifest(checkpoint)
    extra = manifest.get("extra", {})
    if extra.get("stage") != STAGE:
        raise ValueError(f"Checkpoint de otra etapa: {extra.get('stage')}")
    if manifest["prompt_template"] != TEMPLATE_VERSION:
        raise ValueError(f"Plantilla {manifest['prompt_template']} != actual {TEMPLATE_VERSION}")
    cfg = load_config(extra["base_config"])
    snapshot = require_snapshot(cfg)
    if processor_fingerprint(snapshot) != extra["fingerprint"]["processor"]:
        raise ValueError("El procesador local no coincide con el del checkpoint")
    fp = backbone_fingerprint(cfg, snapshot, extra["fingerprint"]["extraction"]["microbatch_rows"])
    if fingerprint_hash(fp) != extra["fingerprint_sha256"]:
        raise ValueError(
            "La huella de extracción (versiones, dtype, plantilla…) no coincide con el checkpoint"
        )

    dataset = load_dataset(extra["dataset_root"])
    if dataset.sha256 != extra["dataset_sha256"]:
        raise ValueError("El dataset ha cambiado desde el entrenamiento")
    sp = Path(extra["split_path"])
    if manifest_sha256(sp) != extra["split_manifest_sha256"]:
        raise ValueError("El manifiesto de split ha cambiado desde el entrenamiento")
    examples = _noul_only(load_split(dataset, sp, (split,), allow_test=allow_test)[split])

    head, _ = load_head(
        checkpoint,
        repo_id=cfg.model.repo_id,
        revision=cfg.model.revision,
        hidden_size=manifest["hidden_size"],
    )
    tracker = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
    processor = load_processor(snapshot)
    cache_dir = extra.get("cache_dir") if use_cache else None
    cache = RepresentationCache(Path(cache_dir), fp) if cache_dir is not None else None
    encoder = LazyEncoder(cfg, snapshot, processor, fp["extraction"]["microbatch_rows"], tracker)
    reps, ext = _representations(split, examples, cache, encoder)
    logits = predict_logits(head, reps).tolist()
    labels = [e.target.label for e in examples]
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "split": split,
        "final_test_used": split == "test",
        "use_cache": use_cache,
        "backbone_loaded": encoder.backbone is not None,
        "extraction": ext,
        "metrics": _summary(binary_metrics(logits, labels, extra["threshold"])),
        "within_group_pairwise": within_group_pairwise([e.group_id for e in examples], logits, labels),
    }
    saved = checkpoint.parent / f"predictions_{split}.jsonl"
    if saved.is_file():
        before = {
            r["id"]: r["logit"] for r in map(json.loads, saved.read_text(encoding="utf-8").splitlines())
        }
        diffs = [abs(before[e.id] - z) for e, z in zip(examples, logits, strict=True) if e.id in before]
        report["reload_vs_training"] = {
            "compared": len(diffs),
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "atol": RELOAD_ATOL,
            "ok": bool(diffs) and max(diffs) <= RELOAD_ATOL,
        }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = checkpoint.parent / "evaluations"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{split}-{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_jsonl(out_dir / f"{split}-{stamp}-predictions.jsonl", _predictions(examples, logits))
    return report
