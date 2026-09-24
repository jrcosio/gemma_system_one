"""``gso train`` para la fase 3: LoRA + cabezales sobre E2B (spec §7.B, §7.D; decisión 0005).

- Arranca de un checkpoint de cabezales de fase 2 validado con el mismo dataset, split y
  huella de extracción (``init_heads_from``). Con ``lora_B = 0`` el punto de partida es
  exactamente ese modelo congelado: la época 0 de validación es su comparación emparejada.
- El estandarizador (decisión 0004) se mantiene fijo, con las estadísticas de train del
  backbone congelado; se registra su deriva por época.
- Particiones: train ajusta pesos; validation elige época; calibration y test no se leen.
- Artefactos: checkpoint de despliegue (adaptador + cabezales, sin pesos base) y directorio de
  reanudación separado (``resume/``), con optimizador, scheduler, RNG y posición del sampler.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..checkpoint import checkpoint_identity, load_decision_heads, read_manifest, save_lora_decision
from ..config import load_config
from ..env import environment_manifest
from ..features import backbone_fingerprint, fingerprint_hash
from ..hub import require_snapshot
from ..models.backbone import load_backbone
from ..models.encoding import load_processor
from ..models.heads import PRIMITIVES
from ..models.lora import LoraParams, apply_lora, enable_layer_recomputation, lora_state, lora_summary
from ..resources import GIB, MemoryTracker
from ..serialization import TEMPLATE_VERSION
from . import decisions_pipeline as dp
from .lora import LORA_STAGE, LoraTrainParams, train_lora
from .pipeline import ExtractionConfig, _write_jsonl


class LoraTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["lora_decision_heads"]
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    base_config: Path
    dataset: Path
    split_seed: int = 0
    primitives: tuple[Literal["noul", "choice", "score"], ...] = PRIMITIVES
    init_heads_from: Path
    extraction: ExtractionConfig = ExtractionConfig()
    runs_dir: Path = Path("runs")
    lora: LoraParams = LoraParams()
    train: LoraTrainParams = LoraTrainParams()
    images: Literal["use", "omit"] = "use"


def load_lora_config(path: Path) -> LoraTrainConfig:
    return LoraTrainConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _check_init(tc: LoraTrainConfig, dataset, sp: Path, fp: dict[str, Any]):
    """El punto de partida debe ser un checkpoint de fase 2 del mismo dataset, split y extracción."""
    manifest = read_manifest(tc.init_heads_from)
    extra = manifest.get("extra", {})
    problems = []
    if extra.get("stage") != dp.STAGE:
        problems.append(f"etapa {extra.get('stage')}")
    if extra.get("dataset_sha256") != dataset.sha256:
        problems.append("dataset distinto")
    if extra.get("split_manifest_sha256") != dp.manifest_sha256(sp):
        problems.append("split distinto")
    if extra.get("fingerprint_sha256") != fingerprint_hash(fp):
        problems.append("huella de extracción distinta")
    if extra.get("feature_norm") != "standardize_train":
        problems.append("sin estandarizador ajustado en train")
    if extra.get("images", "use") != tc.images:
        problems.append(f"modo de imagen distinto ({extra.get('images', 'use')} != {tc.images})")
    if not set(tc.primitives) <= set(extra.get("trained_primitives", [])):
        problems.append("primitivas sin entrenar en el punto de partida")
    if problems:
        raise ValueError(f"init_heads_from no válido ({tc.init_heads_from}): {problems}")
    heads, _ = load_decision_heads(
        tc.init_heads_from,
        repo_id=manifest["base"]["repo_id"],
        revision=manifest["base"]["revision"],
        hidden_size=manifest["hidden_size"],
    )
    if not bool(heads.standardizer.fitted):
        raise ValueError("El estandarizador del punto de partida no está ajustado")
    return heads, checkpoint_identity(tc.init_heads_from), extra


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_train_lora(
    config_path: Path, *, resume: Path | None = None, stop_after_steps: int | None = None
) -> dict[str, Any]:
    config_path = Path(config_path)
    config_text = config_path.read_text(encoding="utf-8")
    tc = load_lora_config(config_path)
    cfg = load_config(tc.base_config)
    tracker = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
    tracker.sample("start")
    if resume is not None:
        run_dir = Path(resume)
        saved = (run_dir / "train_config.yaml").read_text(encoding="utf-8")
        if saved != config_text:
            raise ValueError(f"{config_path} no es la configuración con la que empezó {run_dir}")
        if (run_dir / "checkpoint").exists():
            raise ValueError(f"{run_dir} ya terminó: tiene checkpoint")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = tc.runs_dir / tc.name / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy(config_path, run_dir / "train_config.yaml")
    env = environment_manifest()  # copia de las fuentes de esta sesión (decisión 0008)
    (run_dir / f"env-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
        json.dumps(env, indent=2, default=str), encoding="utf-8"
    )

    dataset, sp, train, val = dp._load(tc)
    snapshot = require_snapshot(cfg)
    processor = load_processor(snapshot)
    mb = tc.extraction.microbatch_rows
    fp = backbone_fingerprint(cfg, snapshot, mb)
    heads, init_identity, init_extra = _check_init(tc, dataset, sp, fp)
    backbone = load_backbone(cfg, snapshot)
    tracker.sample("backbone_loaded")
    targets = apply_lora(backbone.model, tc.lora, seed=tc.train.seed)
    summary = lora_summary(backbone.model, targets, tc.lora)
    if tc.train.recompute_layers:
        summary["recomputed_layers"] = enable_layer_recomputation(backbone.model)
    tracker.sample("lora_injected")
    eval_items = val if tc.train.select_on == "validation" else train
    meta = {
        "config_sha256": _sha(config_text),
        "dataset_sha256": dataset.sha256,
        "split_manifest_sha256": dp.manifest_sha256(sp),
        "train_ids_sha256": _sha("\n".join(it.example.id for it in train)),
        "eval_ids_sha256": _sha("\n".join(it.example.id for it in eval_items)),
        "init_heads_manifest_sha256": init_identity["manifest_sha256"],
        "fingerprint_sha256": fingerprint_hash(fp),
        "lora_targets_sha256": _sha("\n".join(targets)),
    }
    result = train_lora(
        backbone,
        processor,
        heads,
        train,
        eval_items,
        tc.train,
        max_length=cfg.runtime.max_length,
        microbatch_rows=mb,
        resume_root=run_dir / "resume",
        resume_meta=meta,
        resume=resume is not None,
        stop_after_steps=stop_after_steps,
        steps_log=run_dir / "steps.jsonl",
        tracker=tracker,
    )
    state = result.state
    _write_jsonl(run_dir / "history.jsonl", state.history)
    if result.interrupted:
        tracker.sample("interrupted")
        return {
            "run_dir": str(run_dir),
            "interrupted": True,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "next_pos": state.next_pos,
            "resume": f"gso train --config {config_path} --resume {run_dir}",
            "memory": {k: v for k, v in tracker.summary().items() if k != "samples"},
        }

    thr = tc.train.threshold
    final = [z.float() for z in result.final_eval_logits]
    initial = [torch.tensor(z, dtype=torch.float32) for z in state.initial_eval_logits]
    split = tc.train.select_on
    metrics: dict[str, Any] = {
        "threshold": thr,
        "eval_split": split,
        "models": {
            "lora_heads": {split: dp.evaluate_logits(eval_items, final, thr)},
            "initial_frozen_heads": {split: dp.evaluate_logits(eval_items, initial, thr)},
        },
    }
    metrics[f"{split}_bootstrap"] = dp._bootstrap(
        eval_items, {"lora_heads": final, "initial_frozen_heads": initial}, thr, tc.train.seed,
        reference="initial_frozen_heads",
    )  # fmt: skip
    if split == "train":
        tm = metrics["models"]["lora_heads"]["train"]
        per_type = {p: {"accuracy": tm[p]["accuracy"], "nll": tm[p]["nll"]} for p in PRIMITIVES if p in tm}
        metrics["overfit_check"] = {
            "criterion": f"por primitiva: train accuracy == 1.0 y train NLL < {dp.OVERFIT_MAX_NLL}",
            "train_questions": len(train),
            "per_type": per_type,
            "passed": bool(per_type)
            and all(v["accuracy"] == 1.0 and v["nll"] < dp.OVERFIT_MAX_NLL for v in per_type.values()),
        }
    ckpt = save_lora_decision(
        run_dir / "checkpoint",
        heads,
        lora_state(backbone.model),
        {**summary, "params": tc.lora.model_dump()},
        repo_id=cfg.model.repo_id,
        revision=cfg.model.revision,
        backbone_dtype=cfg.model.dtype,
        prompt_template=TEMPLATE_VERSION,
        extra={
            "stage": LORA_STAGE,
            "train_config": tc.name,
            "base_config": str(tc.base_config),
            "primitives": list(tc.primitives),
            "trained_primitives": sorted({it.primitive for it in train}),
            "fingerprint": fp,
            "fingerprint_sha256": fingerprint_hash(fp),
            "dataset_root": str(tc.dataset),
            "cache_dir": None,  # las representaciones dependen del adaptador: nunca se cachean
            "dataset_sha256": dataset.sha256,
            "split_path": str(sp),
            "split_manifest_sha256": dp.manifest_sha256(sp),
            "split_seed": tc.split_seed,
            "train_examples": [it.example.id for it in train],
            "threshold": thr,
            "temperatures": {p: 1.0 for p in PRIMITIVES},
            "calibrated": False,
            "train_params": tc.train.model_dump(),
            "selected_epoch": state.best_epoch,
            "select_on": tc.train.select_on,
            "global_steps": state.global_step,
            "head_training_device": str(backbone.device),
            "feature_norm": "standardize_train",
            "feature_norm_fitted_on": "train (backbone congelado; fijo durante LoRA, decisión 0005)",
            "init_heads_from": init_identity,
            "init_heads_selected_epoch": init_extra.get("selected_epoch"),
            "base_params_unchanged_probe": result.base_unchanged,
            "images": tc.images,
            "input_modality": "text+image" if any(it.image_file is not None for it in train) else "text",
            "session_code": env["git"],  # código de la última sesión (con --resume puede haber varias)
        },
    )
    _write_jsonl(run_dir / f"predictions_{split}.jsonl", dp._predictions(eval_items, final, thr))
    tracker.sample("end")
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "stage": LORA_STAGE,
        "interrupted": False,
        "resumed": resume is not None,
        "selected_epoch": state.best_epoch,
        "select_on": tc.train.select_on,
        "global_steps": state.global_step,
        "splits_used_for_fitting": ["train", "validation"],
        "splits_not_used_for_fitting": ["calibration", "test"],
        "counts": {"train": len(train), "validation": len(val)},
        "lora": {k: v for k, v in summary.items() if k != "targets"},
        "base_params_unchanged_probe": result.base_unchanged,
        "history": state.history,
        "metrics": metrics,
        "memory": {k: v for k, v in tracker.summary().items() if k != "samples"},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report
