"""``gso doctor``: compatibilidad real del backbone y recursos en este equipo.

Cada paso produce ``pass``/``fail``/``skipped`` con detalles medidos. Un fallo
guarda el tipo de excepción y el mensaje para poder reproducirlo. Los pasos que
dependen de uno fallido se marcan ``skipped`` con el motivo.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .checkpoint import save_head
from .config import ProjectConfig
from .env import environment_manifest, mask_home
from .hub import manifest_path, require_snapshot
from .models.backbone import Backbone, load_backbone, pool_last_valid, resolve_device, torch_dtype
from .models.encoding import build_chat_batch, load_processor
from .models.heads import NoulHead
from .resources import GIB, MemoryTracker, synchronize, timed

PROBE_TEMPLATE = "doctor_probe_v0"
# Frases de sonda en ES/EN de longitudes distintas para forzar padding.
PROBE_TEXTS = (
    "Estado: el cliente indica que se le ha cobrado dos veces el mismo pedido y solicita la devolución.\n"
    "Pregunta: ¿El cliente solicita una devolución?\nResponde evaluando la pregunta anterior.",
    "State: DNS does not resolve.\nQuestion: Is the fault in the network subsystem?\n"
    "Evaluate the question above.",
    "Estado: todo funciona.\nPregunta: ¿Hay incidencia?\nEvalúa la pregunta anterior.",
)
PROBE_LABELS = (1.0, 1.0, 0.0)
COSINE_TOL = 0.999  # tolerancia bf16 declarada para comparar la misma fila con/sin padding
RELOAD_ATOL = (
    1e-4  # tolerancia FP32 declarada para logits del cabezal recargado (incluye cambio de dispositivo)
)


class StepSkipped(Exception):
    pass


class Doctor:
    def __init__(self, cfg: ProjectConfig, skip_model: bool):
        self.cfg = cfg
        self.skip_model = skip_model
        self.steps: list[dict[str, Any]] = []
        self.mem = MemoryTracker(budget_bytes=int(cfg.runtime.memory_budget_gib * GIB))
        self.stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.work_dir = cfg.paths.artifacts_dir / "doctor" / self.stamp
        self.state: dict[str, Any] = {}

    # --- infraestructura -------------------------------------------------
    def step(self, name: str, fn: Callable[[], dict[str, Any]], needs: tuple[str, ...] = ()) -> None:
        failed = [n for n in needs if self._status(n) != "pass"]
        if failed:
            self.steps.append({"name": name, "status": "skipped", "reason": f"depende de {failed}"})
            print(f"[doctor] {name}: skipped (depende de {failed})")
            return
        t0 = time.perf_counter()
        entry: dict[str, Any] = {"name": name}
        try:
            self.mem.sample(f"{name}:start")
            entry["details"] = fn()
            self.mem.sample(f"{name}:end")
            entry["status"] = "pass"
        except StepSkipped as exc:
            entry.update(status="skipped", reason=str(exc))
        except Exception as exc:
            entry.update(
                status="fail",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "traceback": traceback.format_exc(limit=8)[-4000:],
                },
            )
        entry["wall_seconds"] = round(time.perf_counter() - t0, 3)
        self.steps.append(entry)
        print(f"[doctor] {name}: {entry['status']} ({entry['wall_seconds']} s)")
        if entry["status"] == "fail":
            print(f"         {entry['error']['type']}: {entry['error']['message'][:300]}")

    def _status(self, name: str) -> str | None:
        return next((s["status"] for s in self.steps if s["name"] == name), None)

    # --- pasos -----------------------------------------------------------
    def environment(self) -> dict[str, Any]:
        env = environment_manifest()
        if env["PYTORCH_ENABLE_MPS_FALLBACK"] not in (None, "0"):
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK activo: las medidas MPS no serían fiables")
        self.state["device"] = resolve_device(self.cfg)
        env["device_selected"] = str(self.state["device"])
        return env

    def mps_ops(self) -> dict[str, Any]:
        device = self.state["device"]
        if device.type != "mps":
            raise StepSkipped(f"dispositivo seleccionado {device}")
        out: dict[str, Any] = {}
        for name in ("float32", "bfloat16", "float16"):
            dtype = torch_dtype(name)
            g = torch.Generator().manual_seed(0)
            a_cpu = torch.randn(128, 256, generator=g)
            w_cpu = torch.randn(256, 64, generator=g) / 16
            results = {}
            for dev in ("cpu", "mps"):
                # detach().clone(): hojas independientes por dispositivo (a_cpu no debe requerir grad).
                a = a_cpu.to(dev, dtype).detach().clone().requires_grad_(True)
                w = w_cpu.to(dev, dtype).detach().clone().requires_grad_(True)
                y = torch.nn.functional.gelu(a @ w).softmax(-1)
                loss = (y.float() * torch.arange(64, device=dev, dtype=torch.float32)).sum()
                loss.backward()
                results[dev] = (loss.detach().cpu(), a.grad.float().cpu(), w.grad.float().cpu())
            finite = all(bool(torch.isfinite(t).all()) for t in results["mps"])
            out[name] = {
                "finite": finite,
                "loss_abs_diff_vs_cpu": float((results["mps"][0] - results["cpu"][0]).abs()),
                "grad_max_abs_diff_vs_cpu": float((results["mps"][2] - results["cpu"][2]).abs().max()),
            }
            if not finite:
                raise RuntimeError(f"Resultados no finitos en MPS con {name}")
        return out

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Identidad de checkpoint, configuración y procesador sin cargar pesos."""
        from safetensors import safe_open
        from transformers import AutoConfig

        snapshot = require_snapshot(self.cfg)
        self.state["snapshot"] = snapshot
        conf = AutoConfig.from_pretrained(snapshot)
        text = conf.get_text_config()
        proc = json.loads((snapshot / "processor_config.json").read_text())
        # Cabecera safetensors: nombres, dtypes y formas; no materializa tensores.
        counts: dict[str, int] = {}
        dtypes: set[str] = set()
        with safe_open(snapshot / "model.safetensors", "pt") as fh:
            for key in fh.keys():  # noqa: SIM118 (safe_open no es un dict)
                sl = fh.get_slice(key)
                n = 1
                for d in sl.get_shape():
                    n *= d
                parts = key.split(".")
                comp = ".".join(parts[:3]) if parts[1] == "language_model" else ".".join(parts[:2])
                counts[comp] = counts.get(comp, 0) + n
                dtypes.add(sl.get_dtype())
        dl_manifest = manifest_path(self.cfg)
        return {
            "repo_id": self.cfg.model.repo_id,
            "revision": self.cfg.model.revision,
            "snapshot_dir": mask_home(snapshot),
            "snapshot_revision_matches": snapshot.name == self.cfg.model.revision,
            "download_manifest": str(dl_manifest) if dl_manifest.exists() else None,
            "architectures": conf.architectures,
            "model_type": conf.model_type,
            "config_transformers_version": getattr(conf, "transformers_version", None),
            "text": {
                "hidden_size": text.hidden_size,
                "num_hidden_layers": text.num_hidden_layers,
                "num_kv_shared_layers": getattr(text, "num_kv_shared_layers", None),
                "vocab_size": text.vocab_size,
                "sliding_window": getattr(text, "sliding_window", None),
                "tie_word_embeddings": getattr(text, "tie_word_embeddings", None),
            },
            "vision_present": conf.vision_config is not None,
            "audio_present": getattr(conf, "audio_config", None) is not None,
            "processor_class": proc.get("processor_class"),
            "image_processor_type": proc.get("image_processor", {}).get("image_processor_type"),
            "image_seq_length": proc.get("image_seq_length"),
            "safetensors_dtypes": sorted(dtypes),
            "safetensors_params_by_component": dict(sorted(counts.items())),
            "safetensors_params_total": sum(counts.values()),
        }

    def load_model(self) -> dict[str, Any]:
        if self.skip_model:
            raise StepSkipped("--skip-model")
        device = self.state["device"]
        with timed(device) as t:
            backbone = load_backbone(self.cfg, self.state["snapshot"], device)
        self.state["backbone"] = backbone
        model = backbone.model
        comps: dict[str, int] = {}
        for name, p in model.named_parameters():
            parts = name.split(".")
            comp = ".".join(parts[:2]) if parts[0] == "language_model" else parts[0]
            comps[comp] = comps.get(comp, 0) + p.numel()
        return {
            "load_seconds_synchronized": round(t["seconds"], 3),
            "class": type(model).__name__,
            "mro": [c.__name__ for c in type(model).__mro__[:4]],
            "text_model_class": type(backbone.text_model).__name__,
            "has_lm_head": hasattr(model, "lm_head"),
            "attn_implementation": model.config._attn_implementation,
            "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "param_devices": sorted({str(p.device) for p in model.parameters()}),
            "buffer_dtypes": sorted({str(b.dtype) for b in model.buffers()}),
            "params_total": sum(comps.values()),
            "params_by_component": dict(sorted(comps.items())),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "training_mode": model.training,
            "hidden_size": backbone.hidden_size,
            "loading_info": {
                k: (v[:20] if isinstance(v, list) else v) for k, v in backbone.loading_info.items()
            },
        }

    def _processor(self):
        if "processor" not in self.state:
            self.state["processor"] = load_processor(self.state["snapshot"])
        return self.state["processor"]

    def _batch(
        self, texts: tuple[str, ...], padding_side: str, allow_over_length: bool = False
    ) -> dict[str, torch.Tensor]:
        max_length = None if allow_over_length else self.cfg.runtime.max_length
        return build_chat_batch(self._processor(), texts, max_length=max_length, padding_side=padding_side)

    def text_forward(self) -> dict[str, Any]:
        backbone: Backbone = self.state["backbone"]
        device = backbone.device
        batch = self._batch(PROBE_TEXTS, "right")
        mask = batch["attention_mask"]
        valid = mask.sum(1).tolist()
        with torch.no_grad(), timed(device) as cold:
            out = backbone.encode(batch)
        with torch.inference_mode(), timed(device) as warm:
            out_warm = backbone.encode(batch)
        h = out.last_hidden_state
        if tuple(h.shape) != (len(PROBE_TEXTS), mask.shape[1], backbone.hidden_size):
            raise RuntimeError(f"Forma inesperada {tuple(h.shape)}")
        if not bool(torch.isfinite(h).all()):
            raise RuntimeError("last_hidden_state con valores no finitos")
        pooled = out.pooled().float()
        repeat_diff = float((out_warm.pooled().float() - pooled).abs().max())

        # La misma fila sin padding debe dar la misma representación (dentro de tolerancia bf16).
        per_row: list[dict[str, float]] = []
        with torch.inference_mode():
            for i, text in enumerate(PROBE_TEXTS):
                alone = backbone.encode(self._batch((text,), "right")).pooled().float()[0]
                cached = backbone.model(
                    **{k: v.to(device) for k, v in self._batch((text,), "right").items()},
                    use_cache=True,
                    return_dict=True,
                ).last_hidden_state
                cached_last = cached[0, -1].float()
                per_row.append(
                    {
                        "valid_tokens": int(valid[i]),
                        "cos_padded_right_vs_alone": float(torch.cosine_similarity(pooled[i], alone, dim=0)),
                        "max_abs_padded_right_vs_alone": float((pooled[i] - alone).abs().max()),
                        "max_abs_use_cache_true_vs_false": float((cached_last - alone).abs().max()),
                    }
                )
            left_batch = self._batch(PROBE_TEXTS, "left")
            left = backbone.encode(left_batch)
            left_pooled = pool_last_valid(left.last_hidden_state, left.attention_mask).float()
            left_cos = torch.cosine_similarity(left_pooled, pooled, dim=1).tolist()
            # Con padding izquierdo, las posiciones por defecto no descuentan el padding: se prueban
            # también position_ids derivados de la máscara.
            lm = left_batch["attention_mask"]
            pos = (lm.cumsum(-1) - 1).clamp(min=0)
            left_pos = backbone.encode({**left_batch, "position_ids": pos}).pooled().float()
            left_pos_cos = torch.cosine_similarity(left_pos, pooled, dim=1).tolist()

        cross = torch.cosine_similarity(pooled[0], pooled[2], dim=0).item()
        self.state["probe_batch"] = batch
        self.state["pooled"] = pooled.detach()
        min_cos = min(r["cos_padded_right_vs_alone"] for r in per_row)
        if min_cos < COSINE_TOL:
            raise RuntimeError(f"Padding derecho altera la representación: cos={min_cos:.5f}")
        if cross > 0.99999:
            raise RuntimeError("Entradas distintas producen la misma representación")
        return {
            "batch_shape": list(h.shape),
            "hidden_dtype": str(h.dtype),
            "padding_tokens": int(mask.numel() - mask.sum()),
            "processed_valid_tokens": int(mask.sum()),
            "cold_forward_seconds_synchronized": round(cold["seconds"], 4),
            "warm_forward_seconds_synchronized": round(warm["seconds"], 4),
            "repeat_max_abs_diff": repeat_diff,
            "rows": per_row,
            "cos_left_padding_default_positions": left_cos,
            "cos_left_padding_mask_positions": left_pos_cos,
            "cos_between_distinct_inputs_0_2": cross,
            "tolerance": {"cosine_min": COSINE_TOL},
        }

    def head_train(self) -> dict[str, Any]:
        backbone: Backbone = self.state["backbone"]
        device = backbone.device
        torch.manual_seed(self.cfg.runtime.seed)
        backbone.freeze()
        with torch.no_grad():
            pooled = backbone.encode(self.state["probe_batch"]).pooled()
        head = NoulHead(backbone.hidden_size).to(device).train()
        watched = {
            n: p.detach().clone()
            for n, p in backbone.model.named_parameters()
            if n.endswith(
                (
                    "layers.0.self_attn.q_proj.weight",
                    "layers.34.mlp.down_proj.weight",
                    "language_model.norm.weight",
                )
            )
        }
        head_before = {n: p.detach().clone() for n, p in head.named_parameters()}
        trainable = [n for n, p in head.named_parameters() if p.requires_grad]
        trainable += [f"backbone.{n}" for n, p in backbone.model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
        y = torch.tensor(PROBE_LABELS, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        losses, grad_norms = [], []
        with timed(device) as t:
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(head(pooled), y)
                if loss.dtype != torch.float32:
                    raise RuntimeError(f"Pérdida en {loss.dtype}, se esperaba float32")
                loss.backward()
                gn = torch.sqrt(sum(p.grad.pow(2).sum() for p in head.parameters()))
                if not bool(torch.isfinite(gn)):
                    raise RuntimeError("Gradiente del cabezal no finito")
                grad_norms.append(float(gn))
                opt.step()
                losses.append(loss.item())
        head_changed = any(not torch.equal(head_before[n], p.detach()) for n, p in head.named_parameters())
        backbone_grads = [n for n, p in backbone.model.named_parameters() if p.grad is not None]
        backbone_unchanged = all(
            torch.equal(v, dict(backbone.model.named_parameters())[n].detach()) for n, v in watched.items()
        )
        head.eval()
        self.state["head"] = head
        if trainable != ["proj.weight", "proj.bias"]:
            raise RuntimeError(f"Parámetros entrenables inesperados: {trainable[:10]}")
        if backbone_grads or not backbone_unchanged or not head_changed or backbone.model.training:
            raise RuntimeError(
                f"backbone_grads={backbone_grads[:5]} backbone_unchanged={backbone_unchanged} "
                f"head_changed={head_changed} backbone_training={backbone.model.training}"
            )
        return {
            "trainable_parameters": trainable,
            "optimizer_param_count": sum(p.numel() for g in opt.param_groups for p in g["params"]),
            "losses": losses,
            "grad_norms": grad_norms,
            "loss_decreased": losses[-1] < losses[0],
            "head_weights_changed": head_changed,
            "backbone_params_with_grad": len(backbone_grads),
            "backbone_watched_unchanged": sorted(watched),
            "backbone_training_mode": backbone.model.training,
            "five_steps_seconds_synchronized": round(t["seconds"], 4),
        }

    def backbone_backward_probe(self) -> dict[str, Any]:
        """Gradiente real a través de las 35 capas en MPS (riesgo previo a LoRA, sin PEFT)."""
        backbone: Backbone = self.state["backbone"]
        device = backbone.device
        layers = backbone.text_model.layers
        probe_params = {
            "layers.0.self_attn.q_proj.weight": layers[0].self_attn.q_proj.weight,
            f"layers.{len(layers) - 1}.self_attn.q_proj.weight": layers[-1].self_attn.q_proj.weight,
        }
        head: NoulHead = self.state["head"]
        head.requires_grad_(False)
        try:
            for p in probe_params.values():
                p.requires_grad_(True)
            batch = self._batch(PROBE_TEXTS[:1], "right")
            self.mem.sample("backward_probe:before_forward")
            with timed(device) as t:
                out = backbone.encode(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    head(out.pooled()), torch.ones(1, device=device)
                )
                self.mem.sample("backward_probe:after_forward")
                loss.backward()
            details: dict[str, Any] = {"seconds_synchronized": round(t["seconds"], 4), "loss": loss.item()}
            for name, p in probe_params.items():
                g = p.grad
                if g is None:
                    raise RuntimeError(f"{name} no recibió gradiente")
                norm = float(g.float().norm())
                details[name] = {
                    "grad_dtype": str(g.dtype),
                    "grad_norm": norm,
                    "finite": bool(torch.isfinite(g).all()),
                }
                if not details[name]["finite"] or norm == 0.0:
                    raise RuntimeError(f"{name}: gradiente no finito o nulo ({norm})")
            others = [
                n
                for n, p in backbone.model.named_parameters()
                if p.grad is not None and not any(n.endswith(k) for k in probe_params)
            ]
            details["other_params_with_grad"] = others[:10]
            if others:
                raise RuntimeError(f"Gradiente en parámetros no seleccionados: {others[:5]}")
            return details
        finally:
            for p in probe_params.values():
                p.requires_grad_(False)
                p.grad = None
            head.requires_grad_(True)
            synchronize(device)
            if device.type == "mps":
                torch.mps.empty_cache()

    def long_context_probe(self) -> dict[str, Any]:
        """Medida de recursos con una fila de ``max_length`` tokens (contenido sintético repetido).

        Forward sin gradiente y forward+backward hasta la capa 0; aproxima el coste por fila
        antes de LoRA. No es un benchmark: una sola repetición tras calentar.
        """
        backbone: Backbone = self.state["backbone"]
        device = backbone.device
        n = self.cfg.runtime.max_length
        base = self._batch((" ".join(PROBE_TEXTS) * 12,), "right", allow_over_length=True)
        if base["input_ids"].shape[1] < n:
            raise RuntimeError("Texto de sonda demasiado corto para max_length")
        batch = {k: v[:, :n] for k, v in base.items()}
        head: NoulHead = self.state["head"]
        out: dict[str, Any] = {"tokens": n, "content": "sintético, recortado a max_length sólo para medir"}
        with torch.inference_mode():
            backbone.encode(batch)  # calentamiento
            with timed(device) as t:
                backbone.encode(batch)
        out["forward_inference_seconds_synchronized"] = round(t["seconds"], 4)
        out["after_forward"] = self.mem.sample("long_probe:after_inference_forward")
        q0 = backbone.text_model.layers[0].self_attn.q_proj.weight
        head.requires_grad_(False)
        try:
            q0.requires_grad_(True)
            with timed(device) as t:
                enc = backbone.encode(batch)
                loss = head(enc.pooled()).sum()
                out["with_graph"] = self.mem.sample("long_probe:graph_built")
                loss.backward()
            out["forward_backward_seconds_synchronized"] = round(t["seconds"], 4)
            out["grad_finite"] = bool(torch.isfinite(q0.grad).all())
            if not out["grad_finite"]:
                raise RuntimeError("Gradiente no finito a max_length")
        finally:
            q0.requires_grad_(False)
            q0.grad = None
            head.requires_grad_(True)
            synchronize(device)
            if device.type == "mps":
                torch.mps.empty_cache()
        keep = (
            "mps_current_allocated_bytes",
            "mps_driver_allocated_bytes",
            "process_rss_bytes",
            "swap_used_bytes",
        )
        for k in ("after_forward", "with_graph"):
            out[k] = {m: out[k].get(m) for m in keep}
        return out

    def save_reload(self) -> dict[str, Any]:
        backbone: Backbone = self.state["backbone"]
        head: NoulHead = self.state["head"].eval()
        pooled = self.state["pooled"].to(backbone.device)
        with torch.inference_mode():
            logits = head(pooled).float().cpu()
        ckpt = save_head(
            self.work_dir / "head",
            head,
            repo_id=self.cfg.model.repo_id,
            revision=self.cfg.model.revision,
            backbone_dtype=self.cfg.model.dtype,
            prompt_template=PROBE_TEMPLATE,
        )
        probe = self.work_dir / "probe.safetensors"
        save_file({"pooled": pooled.float().cpu().contiguous(), "logits": logits.contiguous()}, probe)
        runs = {}
        devices = ["cpu"] + (["mps"] if backbone.device.type == "mps" else [])
        for dev in devices:
            cmd = [
                sys.executable, "-m", "gemma_system_one.checkpoint", "verify",
                "--checkpoint", str(ckpt), "--probe", str(probe), "--device", dev, "--atol", str(RELOAD_ATOL),
            ]  # fmt: skip
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            lines = proc.stdout.strip().splitlines()
            result = json.loads(lines[-1]) if lines else {"ok": False}
            result["returncode"] = proc.returncode
            if proc.returncode != 0:
                result["stderr_tail"] = proc.stderr[-1500:]
            runs[dev] = result
        if not all(r.get("ok") for r in runs.values()):
            raise RuntimeError(f"Recarga no equivalente: {runs}")
        files = sorted(p.name for p in ckpt.iterdir())
        return {
            "checkpoint_dir": str(ckpt),
            "files": files,
            "checkpoint_bytes": sum(p.stat().st_size for p in ckpt.iterdir()),
            "fresh_process_runs": runs,
            "tolerance_atol": RELOAD_ATOL,
        }

    def vision(self) -> dict[str, Any]:
        """Repetición con una imagen (spec §2.2.8): campos del procesador, tokens visuales,
        forward sincronizado, determinismo y sensibilidad al contenido de la imagen."""
        from PIL import Image

        from .data.generate_vision import render_chart
        from .models.encoding import build_chat_batch, image_token_count

        backbone: Backbone = self.state["backbone"]
        device = backbone.device
        facts = [
            {"services": ["API", "Auth", "Sync"], "values": [82, 14, 40], "threshold": 60, "style": 0},
            {"services": ["API", "Auth", "Sync"], "values": [14, 82, 40], "threshold": 60, "style": 0},
        ]
        question = "Which service shows the most errors in the panel?"
        images = [Image.open(__import__("io").BytesIO(render_chart(f, "en"))).convert("RGB") for f in facts]
        batches = [
            build_chat_batch(self._processor(), [question], max_length=None, images=[im]) for im in images
        ]
        text_only = build_chat_batch(self._processor(), [question], max_length=None)
        self.mem.sample("vision_before")
        pooled, times = [], []
        with torch.inference_mode():
            for b in [batches[0], batches[0], batches[1], text_only]:
                with timed(device) as t:
                    out = backbone.encode(b)
                h = out.pooled().float()
                if not bool(torch.isfinite(h).all()):
                    raise RuntimeError("Representación con imagen no finita")
                pooled.append(h[0].cpu())
                times.append(round(t["seconds"], 4))
        self.mem.sample("vision_after")
        b0 = batches[0]
        cos = torch.nn.functional.cosine_similarity
        return {
            "image_size_px": list(images[0].size),
            "processor_fields": sorted(b0),
            "pixel_values_shape": list(b0["pixel_values"].shape),
            "image_tokens": image_token_count(b0),
            "row_tokens_with_image": int(b0["attention_mask"].sum()),
            "row_tokens_without_image": int(text_only["attention_mask"].sum()),
            "seconds_synchronized": {
                "cold": times[0],
                "warm_repeat": times[1],
                "other_image": times[2],
                "text_only": times[3],
            },  # fmt: skip
            "repeat_max_abs_diff": float((pooled[0] - pooled[1]).abs().max()),
            "other_image_max_abs_diff": float((pooled[0] - pooled[2]).abs().max()),
            "other_image_cosine": float(cos(pooled[0], pooled[2], dim=0)),
            "text_only_cosine": float(cos(pooled[0], pooled[3], dim=0)),
            "placement": "imagen antes del texto en el turno de usuario (decisión 0007)",
        }

    def memory(self) -> dict[str, Any]:
        summary = self.mem.summary()
        pressures = [s.get("vm_pressure_level") for s in self.mem.samples if s.get("vm_pressure_level")]
        summary["max_vm_pressure_level"] = max(pressures) if pressures else None
        if summary["over_budget"]:
            raise RuntimeError(
                f"Presupuesto de {self.cfg.runtime.memory_budget_gib} GiB superado: {summary['over_budget']}"
            )
        return {k: v for k, v in summary.items() if k != "samples"}

    # --- orquestación ----------------------------------------------------
    def run(self) -> dict[str, Any]:
        self.step("environment", self.environment)
        self.step("mps_ops", self.mps_ops, needs=("environment",))
        self.step("checkpoint_metadata", self.checkpoint_metadata, needs=("environment",))
        self.step("load_backbone", self.load_model, needs=("checkpoint_metadata",))
        self.step("text_forward", self.text_forward, needs=("load_backbone",))
        self.step("head_train", self.head_train, needs=("text_forward",))
        self.step("backbone_backward_probe", self.backbone_backward_probe, needs=("head_train",))
        self.step("long_context_probe", self.long_context_probe, needs=("head_train",))
        self.step("save_reload", self.save_reload, needs=("head_train",))
        self.step("vision", self.vision, needs=("load_backbone",))
        self.step("memory", self.memory)
        statuses = [s["status"] for s in self.steps]
        if "fail" in statuses:
            verdict = "fail"
        elif self.skip_model or any(s["status"] == "skipped" for s in self.steps):
            verdict = "partial"
        else:
            verdict = "pass"
        return {
            "tool": "gso doctor",
            "created_utc": self.stamp,
            "config": self.cfg.model_dump(mode="json"),
            "verdict": verdict,
            "verdict_scope": "texto e imagen (una por fila); audio fuera de alcance",
            "steps": self.steps,
            "memory_samples": self.mem.samples,
        }


def run_doctor(
    cfg: ProjectConfig, *, skip_model: bool = False, json_out: Path | None = None
) -> dict[str, Any]:
    report = Doctor(cfg, skip_model).run()
    out = json_out or cfg.paths.reports_dir / "doctor" / f"{report['created_utc']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[doctor] veredicto: {report['verdict']} — informe: {out}")
    return report
