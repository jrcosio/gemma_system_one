"""Etapa B (spec §7.B): LoRA en la atención textual + cabezales, con autograd real y sin caché.

- Cada pregunta hace forward con gradiente de sus K/M filas (``microbatch_rows`` filas por
  forward; 1 por defecto, decisión 0002) y conserva el grafo hasta la pérdida del grupo
  completo. Backward por pregunta, dividido por el número real de preguntas del paso.
- AdamW con dos grupos (LoRA y cabezales), warmup lineal, decaimiento lineal y clip global.
- Base en ``eval()``; sólo el dropout de LoRA en ``train``. Nunca ``no_grad`` en el forward
  de entrenamiento ni ``detach`` de logits.
- Evaluación por época con ``no_grad`` y la misma extracción que la fase 2 (``extract_pooled``);
  los cabezales se aplican en CPU/FP32, igual que al recargar el checkpoint.
- Reanudación: pesos entrenables, optimizador, scheduler, RNG (CPU y MPS), época, posición en
  la permutación, paso global, historial y mejor estado. Cada guardado va a un directorio
  nuevo y un puntero ``LATEST`` se sustituye de forma atómica.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from safetensors.torch import load_file, save_file

from ..features import extract_pooled, slice_batch
from ..images import load_image
from ..models.backbone import load_backbone
from ..models.encoding import build_chat_batch
from ..models.heads import PRIMITIVES, DecisionHeads
from ..models.lora import LoraParams, apply_lora, load_lora_state, lora_parameters, lora_state, set_lora_mode
from ..resources import synchronize
from .decisions import QuestionItem, all_logits, flatten, mean_nll, question_loss, row_images
from .pipeline import LazyEncoder

LORA_STAGE = "phase3_lora_decision_heads"
RESUME_FORMAT = 1
LATEST = "LATEST"


class LoraTrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=3, ge=1, le=100)
    lr_lora: float = Field(default=1e-4, gt=0)
    lr_heads: float = Field(default=5e-4, gt=0)
    weight_decay_lora: float = Field(default=0.0, ge=0)
    weight_decay_heads: float = Field(default=0.01, ge=0)
    warmup_fraction: float = Field(default=0.05, ge=0, lt=1)
    clip_grad_norm: float = Field(default=1.0, gt=0)
    questions_per_step: int = Field(default=8, ge=1)
    seed: int = 0
    threshold: float = Field(default=0.5, gt=0, lt=1)
    rps_lambda: float = Field(default=0.0, ge=0)
    type_weights: dict[str, float] = {"noul": 1.0, "choice": 1.0, "score": 1.0}
    train_heads: bool = True
    # validation: época (incluida la 0 = punto de partida) con menor NLL de validación.
    # train: prueba de sobreajuste; se evalúa en train y se queda la última época.
    select_on: Literal["validation", "train"] = "validation"
    max_train_questions: int | None = Field(default=None, ge=1)
    resume_every_steps: int | None = Field(default=50, ge=1)
    memory_sample_every_steps: int = Field(default=25, ge=1)
    # Con formas variables, el asignador MPS retiene bloques del backward: en el piloto la memoria
    # del driver llegó a ~30 GB con ~10 GB asignados y el sistema entró en presión de memoria.
    # Liberar la caché sólo devuelve bloques libres (no verificado bit a bit en MPS; decisión 0005).
    # Cada paso es caro: con 24 s/paso, un 35 % del hilo principal estaba en empty_cache. None = nunca.
    empty_mps_cache_every_steps: int | None = Field(default=10, ge=1)
    # Recomputación de activaciones por capa del decodificador (decisión 0010): mismos gradientes
    # (medido en E2B/MPS: diferencia relativa 0,0) con menos memoria y ~1,4× de tiempo por paso.
    # Necesaria para LoRA sobre E4B dentro del presupuesto de 32 GiB. Sólo actúa al entrenar.
    recompute_layers: bool = False

    @field_validator("type_weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != set(PRIMITIVES) or any(not math.isfinite(w) or w < 0 for w in v.values()):
            raise ValueError(f"type_weights debe definir pesos finitos ≥0 para {PRIMITIVES}")
        return v


def lr_factor(step: int, *, total: int, warmup: int) -> float:
    """Warmup lineal hasta 1 y decaimiento lineal hasta 0 al final del entrenamiento."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def epoch_permutation(n: int, seed: int, epoch: int) -> list[int]:
    """Permutación determinista por (semilla, época): la reanudación sólo necesita la posición."""
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100_003 + epoch)).tolist()


def group_pooled(
    backbone,
    processor,
    texts: list[str],
    *,
    max_length: int,
    microbatch_rows: int,
    image_file: Path | None = None,
) -> tuple[torch.Tensor, int]:
    """Representaciones del grupo (en el orden de ``texts``) conservando el grafo de autograd.

    Con imagen (la misma para todas las filas de la pregunta), una fila por forward."""
    if image_file is not None:
        if microbatch_rows != 1:
            raise ValueError("Con imágenes sólo se admite una fila por forward (decisiones 0002 y 0007)")
        image = load_image(image_file)
        parts, tokens = [], 0
        for text in texts:
            batch = build_chat_batch(processor, [text], max_length=max_length, images=[image])
            parts.append(backbone.encode(batch).pooled())
            tokens += int(batch["attention_mask"].sum())
        return torch.cat(parts), tokens
    batch = build_chat_batch(processor, texts, max_length=max_length)
    lengths = batch["attention_mask"].sum(dim=1)
    parts, tokens = [], 0
    for start in range(0, len(texts), microbatch_rows):
        idx = torch.arange(start, min(start + microbatch_rows, len(texts)))
        sub = slice_batch(batch, idx, int(lengths[idx].max()))
        parts.append(backbone.encode(sub).pooled())
        tokens += int(sub["attention_mask"].sum())
    return torch.cat(parts), tokens


def question_logits_with_grad(
    backbone, processor, heads: DecisionHeads, item: QuestionItem, *, max_length: int, microbatch_rows: int
) -> tuple[torch.Tensor, int]:
    pooled, tokens = group_pooled(
        backbone,
        processor,
        [r.text for r in item.rows],
        max_length=max_length,
        microbatch_rows=microbatch_rows,
        image_file=item.image_file,
    )
    return heads(item.primitive, pooled), tokens


def heads_on_cpu(heads: DecisionHeads) -> DecisionHeads:
    cpu = DecisionHeads(heads.hidden_size)
    cpu.load_state_dict({k: v.detach().cpu() for k, v in heads.state_dict().items()})
    return cpu.eval()


def eval_logits(
    backbone,
    processor,
    heads: DecisionHeads,
    items: list[QuestionItem],
    *,
    max_length: int,
    microbatch_rows: int,
) -> tuple[list[torch.Tensor], dict[str, Any], dict[str, float]]:
    """Logits sin gradiente con la extracción compartida; deriva de la estandarización fija."""
    set_lora_mode(backbone.model, training=False)
    texts, _, offsets = flatten(items)
    reps, stats = extract_pooled(
        backbone,
        processor,
        texts,
        max_length=max_length,
        microbatch_rows=microbatch_rows,
        images=row_images(items),
    )
    cpu = heads_on_cpu(heads)
    logits = all_logits(cpu, items, reps, offsets)
    with torch.no_grad():
        zs = cpu.standardizer(reps)
        drift = {
            "rows": len(reps),
            "mean_abs_dim_mean": float(zs.mean(0).abs().mean()),
            "mean_dim_std": float(zs.std(0, unbiased=False).mean()),
            "max_dim_std": float(zs.std(0, unbiased=False).max()),
        }
    return logits, stats.__dict__, drift


@dataclass
class LoopState:
    epoch: int = 1  # época en curso (1..epochs); epochs+1 = terminado
    next_pos: int = 0  # posición en la permutación de la época en curso
    global_step: int = 0
    epoch_loss_sum: float = 0.0
    epoch_steps: int = 0
    initial_done: bool = False
    best_epoch: int | None = None
    best_score: float = math.inf
    history: list[dict[str, Any]] = field(default_factory=list)
    initial_eval_logits: list[list[float]] | None = None


@dataclass
class LoraTrainResult:
    interrupted: bool
    state: LoopState
    final_eval_logits: list[torch.Tensor] | None = None
    base_unchanged: bool | None = None


def _snapshot(model, heads: DecisionHeads) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "lora": lora_state(model),
        "heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
    }


def _restore(model, heads: DecisionHeads, snap: dict[str, dict[str, torch.Tensor]]) -> None:
    load_lora_state(model, snap["lora"])
    device = next(heads.parameters()).device
    heads.load_state_dict({k: v.to(device) for k, v in snap["heads"].items()})


def save_resume(
    root: Path, *, model, heads, opt, sched, state: LoopState, best: dict | None, meta: dict[str, Any]
) -> Path:
    """Directorio nuevo por guardado + puntero ``LATEST`` atómico; conserva sólo el último."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # El mismo paso puede guardarse por periodicidad y por parada controlada.
    # Nunca reemplazar el directorio publicado antes de cambiar LATEST.
    name = f"step-{state.global_step:07d}-e{state.epoch:03d}-p{state.next_pos:06d}-{uuid4().hex}"
    tmp = Path(tempfile.mkdtemp(dir=root, prefix=f".{name}.tmp-"))
    cur = _snapshot(model, heads)
    save_file({k: v.contiguous() for k, v in cur["lora"].items()}, tmp / "lora.safetensors")
    save_file({k: v.contiguous() for k, v in cur["heads"].items()}, tmp / "heads.safetensors")
    if best is not None:
        save_file({k: v.contiguous() for k, v in best["lora"].items()}, tmp / "best_lora.safetensors")
        save_file({k: v.contiguous() for k, v in best["heads"].items()}, tmp / "best_heads.safetensors")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, tmp / "optim.pt")
    rng = {"cpu": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        rng["mps"] = torch.mps.get_rng_state()
    torch.save(rng, tmp / "rng.pt")
    (tmp / "state.json").write_text(
        json.dumps({"format": RESUME_FORMAT, "meta": meta, "state": asdict(state)}, indent=1),
        encoding="utf-8",
    )
    for f in tmp.iterdir():  # durabilidad ante corte de alimentación antes de publicar
        with f.open("rb") as fh:
            os.fsync(fh.fileno())
    final = root / name
    os.replace(tmp, final)
    fd, ptr = tempfile.mkstemp(dir=root, prefix=".latest.")
    with os.fdopen(fd, "w") as fh:
        fh.write(name)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(ptr, root / LATEST)
    dir_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    for old in root.iterdir():
        if old.is_dir() and old.name != name and not old.name.startswith("."):
            shutil.rmtree(old)
    return final


def read_resume(root: Path) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    d = root / (root / LATEST).read_text().strip()
    payload = json.loads((d / "state.json").read_text(encoding="utf-8"))
    if payload.get("format") != RESUME_FORMAT:
        raise ValueError(f"Formato de reanudación no soportado: {payload.get('format')}")
    return d, payload


def _load_resume(d: Path, payload, *, model, heads, opt, sched) -> tuple[LoopState, dict | None]:
    device = next(heads.parameters()).device
    _restore(
        model, heads, {"lora": load_file(d / "lora.safetensors"), "heads": load_file(d / "heads.safetensors")}
    )
    best = None
    if (d / "best_lora.safetensors").is_file():
        best = {
            "lora": load_file(d / "best_lora.safetensors"),
            "heads": load_file(d / "best_heads.safetensors"),
        }
    optim = torch.load(d / "optim.pt", map_location=device, weights_only=True)
    opt.load_state_dict(optim["optimizer"])
    sched.load_state_dict(optim["scheduler"])
    rng = torch.load(d / "rng.pt", weights_only=True)
    torch.set_rng_state(rng["cpu"])
    if "mps" in rng and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])
    return LoopState(**payload["state"]), best


def _discard_steps_after(steps_log: Path, step: int) -> None:
    """Al reanudar, los pasos registrados después del estado restaurado se rehacen: se apartan."""
    raw = steps_log.read_text(encoding="utf-8")
    lines = [x for x in raw.splitlines() if x.strip()]
    keep, drop = [], []
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Una interrupción puede cortar el último write. No ocultar corrupción
            # en líneas completas o anteriores; preservar el fragmento para auditoría.
            if i != len(lines) - 1 or raw.endswith("\n"):
                raise
            with steps_log.with_name("steps.truncated.txt").open("a", encoding="utf-8") as fh:
                fh.write(line)
            continue
        (keep if record["step"] <= step else drop).append(line)
    if drop:
        with steps_log.with_name("steps.discarded.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(drop) + "\n")
    fd, tmp = tempfile.mkstemp(dir=steps_log.parent, prefix=".steps.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("".join(x + "\n" for x in keep))
    os.replace(tmp, steps_log)


def _base_probe(model) -> dict[str, torch.Tensor]:
    """Copias de algunos pesos base para comprobar al final que no cambiaron."""
    names = [n for n, _ in model.named_parameters() if ".lora_" not in n]
    picks = {names[0], names[len(names) // 2], names[-1]}
    picks |= {n for n in names if n.endswith("layers.0.self_attn.q_proj.base_layer.weight")}
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in picks}


def train_lora(
    backbone,
    processor,
    heads: DecisionHeads,
    items: list[QuestionItem],
    eval_items: list[QuestionItem],
    params: LoraTrainParams,
    *,
    max_length: int,
    microbatch_rows: int,
    resume_root: Path,
    resume_meta: dict[str, Any],
    resume: bool = False,
    stop_after_steps: int | None = None,
    steps_log: Path | None = None,
    tracker=None,
) -> LoraTrainResult:
    model, device = backbone.model, backbone.device
    torch.manual_seed(params.seed)
    heads.to(device)
    for p in heads.parameters():
        p.requires_grad_(params.train_heads)
    lora = lora_parameters(model)
    groups = [
        {"name": "lora", "params": [lora[k] for k in sorted(lora)], "lr": params.lr_lora,
         "weight_decay": params.weight_decay_lora},
    ]  # fmt: skip
    if params.train_heads:
        groups.append(
            {"name": "heads", "params": [p for _, p in sorted(heads.heads.named_parameters())],
             "lr": params.lr_heads, "weight_decay": params.weight_decay_heads}
        )  # fmt: skip
    trainable = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(items) / params.questions_per_step)
    total = params.epochs * steps_per_epoch
    warmup = round(params.warmup_fraction * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_factor, total=total, warmup=warmup))
    meta = {**resume_meta, "steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warmup}
    state, best = LoopState(), None
    if resume:
        d, payload = read_resume(resume_root)
        if payload["meta"] != meta:
            diff = sorted(
                k for k in set(meta) | set(payload["meta"]) if meta.get(k) != payload["meta"].get(k)
            )
            raise ValueError(f"La reanudación no corresponde a esta configuración/datos: {diff}")
        state, best = _load_resume(d, payload, model=model, heads=heads, opt=opt, sched=sched)
        if steps_log is not None and Path(steps_log).is_file():
            _discard_steps_after(Path(steps_log), state.global_step)
    probe = _base_probe(model)
    kw = {"max_length": max_length, "microbatch_rows": microbatch_rows}

    def evaluate(epoch: int, extra: dict[str, Any]) -> list[torch.Tensor]:
        logits, stats, drift = eval_logits(backbone, processor, heads, eval_items, **kw)
        nll = mean_nll(eval_items, logits)
        record = {"epoch": epoch, "global_step": state.global_step, "eval_split": params.select_on}
        record |= {"eval_nll": nll, "feature_drift": drift, "eval_seconds": stats["seconds_synchronized"]}
        state.history.append({**record, **extra})
        return logits

    def consider(epoch: int, score: float) -> None:
        nonlocal best
        if score < state.best_score:
            state.best_score, state.best_epoch, best = score, epoch, _snapshot(model, heads)

    def checkpoint_resume() -> None:
        save_resume(
            resume_root, model=model, heads=heads, opt=opt, sched=sched, state=state, best=best, meta=meta
        )

    if not state.initial_done:
        logits = evaluate(0, {"note": "punto de partida: cabezales iniciales y LoRA con B=0"})
        state.initial_eval_logits = [z.tolist() for z in logits]
        state.initial_done = True
        h = state.history[-1]["eval_nll"]["all"]
        consider(0, h if params.select_on == "validation" else 0.0)
    while state.epoch <= params.epochs:
        perm = epoch_permutation(len(items), params.seed, state.epoch)
        while state.next_pos < len(perm):
            batch = perm[state.next_pos : state.next_pos + params.questions_per_step]
            set_lora_mode(model, training=True)
            heads.train()
            opt.zero_grad(set_to_none=True)
            synchronize(device)
            t0 = time.perf_counter()
            step_loss, rows, tokens = 0.0, 0, 0
            graph_peak = 0
            for qi in batch:
                it = items[qi]
                z, ntok = question_logits_with_grad(backbone, processor, heads, it, **kw)
                loss = params.type_weights[it.primitive] * question_loss(
                    it.primitive, z, it.target, params.rps_lambda
                )
                if device.type == "mps":
                    # Con el grafo vivo, justo antes del backward: más cerca del pico que el muestreo
                    # entre pasos, pero sigue sin ver temporales del propio backward.
                    graph_peak = max(graph_peak, torch.mps.current_allocated_memory())
                (loss / len(batch)).backward()
                step_loss += float(loss.detach())
                rows += len(it.rows)
                tokens += ntok
            # LoRA recibe gradiente en cada pregunta; un cabezal sin preguntas de su tipo en el paso, no.
            if any(lora[k].grad is None for k in lora):
                raise RuntimeError(f"LoRA sin gradiente en el paso {state.global_step + 1}")
            for p in trainable:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise RuntimeError(f"Gradiente no finito en el paso {state.global_step + 1}")
            norm = torch.nn.utils.clip_grad_norm_(trainable, params.clip_grad_norm, error_if_nonfinite=True)
            lrs = {g["name"]: g["lr"] for g in opt.param_groups}
            opt.step()
            sched.step()
            synchronize(device)
            mem = {}
            if device.type == "mps":
                every = params.empty_mps_cache_every_steps
                if every and (state.global_step + 1) % every == 0:
                    torch.mps.empty_cache()
                mem = {
                    "mps_current_before_backward_max_bytes": graph_peak,
                    "mps_current_bytes": torch.mps.current_allocated_memory(),
                    "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                }
            state.global_step += 1
            state.next_pos += len(batch)
            state.epoch_loss_sum += step_loss / len(batch)
            state.epoch_steps += 1
            if steps_log is not None:
                with Path(steps_log).open("a", encoding="utf-8") as fh:
                    rec = {"step": state.global_step, "epoch": state.epoch, "questions": len(batch)}
                    rec |= {"rows": rows, "tokens": tokens, "loss": step_loss / len(batch)}
                    rec |= {"grad_norm_before_clip": float(norm), "lr": lrs}
                    rec["seconds"] = round(time.perf_counter() - t0, 4)
                    rec |= mem
                    fh.write(json.dumps(rec) + "\n")
            if tracker is not None and state.global_step % params.memory_sample_every_steps == 0:
                tracker.sample(f"step_{state.global_step}")
            if params.resume_every_steps and state.global_step % params.resume_every_steps == 0:
                checkpoint_resume()
            if stop_after_steps is not None and state.global_step >= stop_after_steps:
                checkpoint_resume()
                return LoraTrainResult(interrupted=True, state=state)
        train_loss = state.epoch_loss_sum / max(1, state.epoch_steps)
        evaluate(state.epoch, {"train_loss_mean_steps": train_loss, "steps": state.epoch_steps})
        h = state.history[-1]["eval_nll"]["all"]
        consider(state.epoch, h if params.select_on == "validation" else -float(state.epoch))
        state.epoch += 1
        state.next_pos, state.epoch_loss_sum, state.epoch_steps = 0, 0.0, 0
        checkpoint_resume()
    unchanged = all(
        torch.equal(p.detach().cpu(), probe[n]) for n, p in model.named_parameters() if n in probe
    )
    _restore(model, heads, best)
    final, _, _ = eval_logits(backbone, processor, heads, eval_items, **kw)
    return LoraTrainResult(interrupted=False, state=state, final_eval_logits=final, base_unchanged=unchanged)


class LazyLoraEncoder(LazyEncoder):
    """Como ``LazyEncoder``, pero inyecta LoRA y carga el adaptador guardado antes de extraer."""

    def __init__(
        self,
        cfg,
        snapshot,
        processor,
        microbatch_rows,
        tracker,
        lora_params: LoraParams,
        adapter_state,
        targets,
    ):
        super().__init__(cfg, snapshot, processor, microbatch_rows, tracker)
        self.lora_params, self.adapter_state, self.targets = lora_params, adapter_state, list(targets)

    def __call__(self, texts: list[str], images: list | None = None):
        if self.backbone is None:
            backbone = load_backbone(self.cfg, self.snapshot)
            targets = apply_lora(backbone.model, self.lora_params)
            if targets != self.targets:
                raise ValueError("Los módulos objetivo del modelo no coinciden con los del checkpoint")
            load_lora_state(backbone.model, self.adapter_state)
            set_lora_mode(backbone.model, training=False)
            self.backbone = backbone
            self.tracker.sample("backbone_lora_loaded")
        return super().__call__(texts, images)
