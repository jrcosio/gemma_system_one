"""Checkpoint mínimo de despliegue: pesos del cabezal + manifiesto.

Nunca contiene pesos del backbone: se referencia por ``repo_id`` + ``revision``.
El guardado es atómico (directorio temporal + ``os.replace``) y los checkpoints
son inmutables: no se sobrescribe un directorio existente.

``python -m gemma_system_one.checkpoint verify ...`` recarga en un proceso nuevo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .env import git_state
from .hub import sha256_file
from .models.heads import DecisionHeads, NoulHead

FORMAT_VERSION = 1
DECISION_FORMAT_VERSION = 3  # 3: incluye el estandarizador de características (decisión 0004)
LORA_FORMAT_VERSION = 1  # adaptador LoRA + cabezales de decisión (fase 3, decisión 0005)
WEIGHTS = "head.safetensors"
ADAPTER = "adapter.safetensors"
MANIFEST = "manifest.json"


class CheckpointMismatchError(ValueError):
    pass


def _save_state(
    directory: Path,
    state: dict[str, torch.Tensor],
    *,
    kind: str,
    format_version: int,
    hidden_size: int,
    repo_id: str,
    revision: str,
    backbone_dtype: str,
    prompt_template: str,
    extra: dict[str, Any] | None,
    adapter: tuple[dict[str, torch.Tensor], dict[str, Any]] | None = None,
) -> Path:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} ya existe; los checkpoints no se sobrescriben")
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}.tmp-"))
    try:
        state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in state.items()}
        save_file(state, tmp / WEIGHTS, metadata={"format_version": str(format_version)})
        adapter_entry = None
        if adapter is not None:
            tensors, config = adapter
            tensors = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in tensors.items()}
            save_file(tensors, tmp / ADAPTER, metadata={"format_version": str(format_version)})
            adapter_entry = {
                "file": ADAPTER,
                "sha256": sha256_file(tmp / ADAPTER),
                "keys": sorted(tensors),
                "config": config,
            }
        manifest = {
            "format_version": format_version,
            "kind": kind,
            "created_utc": datetime.now(UTC).isoformat(),
            "base": {"repo_id": repo_id, "revision": revision, "dtype": backbone_dtype},
            "hidden_size": hidden_size,
            "prompt_template": prompt_template,
            "weights": {"file": WEIGHTS, "sha256": sha256_file(tmp / WEIGHTS), "keys": sorted(state)},
            **({"adapter": adapter_entry} if adapter_entry else {}),
            "git": git_state(snapshot=True),
            **({"extra": extra} if extra else {}),
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return directory


def save_head(
    directory: Path,
    head: NoulHead,
    *,
    repo_id: str,
    revision: str,
    backbone_dtype: str,
    prompt_template: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    return _save_state(
        directory,
        head.state_dict(),
        kind="noul_head",
        format_version=FORMAT_VERSION,
        hidden_size=head.hidden_size,
        repo_id=repo_id,
        revision=revision,
        backbone_dtype=backbone_dtype,
        prompt_template=prompt_template,
        extra=extra,
    )


def save_decision_heads(
    directory: Path,
    heads: DecisionHeads,
    *,
    repo_id: str,
    revision: str,
    backbone_dtype: str,
    prompt_template: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Checkpoint de los tres evaluadores (Noul/Choice/Score); tampoco incluye el backbone."""
    return _save_state(
        directory,
        heads.state_dict(),
        kind="decision_heads",
        format_version=DECISION_FORMAT_VERSION,
        hidden_size=heads.hidden_size,
        repo_id=repo_id,
        revision=revision,
        backbone_dtype=backbone_dtype,
        prompt_template=prompt_template,
        extra=extra,
    )


def save_lora_decision(
    directory: Path,
    heads: DecisionHeads,
    adapter_state: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    *,
    repo_id: str,
    revision: str,
    backbone_dtype: str,
    prompt_template: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Artefacto de despliegue de fase 3: adaptador LoRA + cabezales; referencia la base, no la copia."""
    if not adapter_state or "targets" not in adapter_config:
        raise ValueError("El adaptador necesita pesos y la lista de módulos objetivo")
    return _save_state(
        directory,
        heads.state_dict(),
        kind="lora_decision_heads",
        format_version=LORA_FORMAT_VERSION,
        hidden_size=heads.hidden_size,
        repo_id=repo_id,
        revision=revision,
        backbone_dtype=backbone_dtype,
        prompt_template=prompt_template,
        extra=extra,
        adapter=(adapter_state, adapter_config),
    )


def load_lora_decision(
    directory: Path, *, repo_id: str, revision: str, hidden_size: int
) -> tuple[DecisionHeads, dict[str, torch.Tensor], dict[str, Any]]:
    """Cabezales, estado LoRA (CPU, FP32) y manifiesto; verifica base y sha256 de ambos ficheros."""
    manifest = read_manifest(directory)
    if manifest.get("format_version") != LORA_FORMAT_VERSION or manifest.get("kind") != "lora_decision_heads":
        raise CheckpointMismatchError(
            f"Formato no soportado: {manifest.get('kind')}/{manifest.get('format_version')}"
        )
    weights = _check_base(directory, manifest, repo_id, revision, hidden_size)
    entry = manifest.get("adapter") or {}
    adapter_path = Path(directory) / entry.get("file", ADAPTER)
    if not adapter_path.is_file() or sha256_file(adapter_path) != entry.get("sha256"):
        raise CheckpointMismatchError("sha256 del adaptador no coincide con el manifiesto")
    adapter_state = load_file(adapter_path)
    if sorted(adapter_state) != entry.get("keys"):
        raise CheckpointMismatchError("Claves del adaptador distintas de las del manifiesto")
    heads = DecisionHeads(hidden_size)
    heads.load_state_dict(load_file(weights), strict=True)
    return heads.eval(), adapter_state, manifest


def checkpoint_identity(directory: Path) -> dict[str, Any]:
    """Identidad de un checkpoint para vincular artefactos derivados (calibración, evaluación)."""
    manifest = read_manifest(directory)
    return {
        "path": str(directory),
        "kind": manifest.get("kind"),
        "manifest_sha256": sha256_file(Path(directory) / MANIFEST),
        "weights_sha256": manifest["weights"]["sha256"],
        "adapter_sha256": (manifest.get("adapter") or {}).get("sha256"),
    }


def _check_base(
    directory: Path, manifest: dict[str, Any], repo_id: str, revision: str, hidden_size: int
) -> Path:
    base = manifest["base"]
    if (base["repo_id"], base["revision"]) != (repo_id, revision):
        raise CheckpointMismatchError(f"Base {base['repo_id']}@{base['revision']} != {repo_id}@{revision}")
    if manifest["hidden_size"] != hidden_size:
        raise CheckpointMismatchError(f"hidden_size {manifest['hidden_size']} != {hidden_size}")
    weights = Path(directory) / manifest["weights"]["file"]
    if sha256_file(weights) != manifest["weights"]["sha256"]:
        raise CheckpointMismatchError("sha256 del fichero de pesos no coincide con el manifiesto")
    return weights


def load_decision_heads(
    directory: Path, *, repo_id: str, revision: str, hidden_size: int
) -> tuple[DecisionHeads, dict[str, Any]]:
    manifest = read_manifest(directory)
    if manifest.get("format_version") != DECISION_FORMAT_VERSION or manifest.get("kind") != "decision_heads":
        raise CheckpointMismatchError(
            f"Formato no soportado: {manifest.get('kind')}/{manifest.get('format_version')}"
        )
    weights = _check_base(directory, manifest, repo_id, revision, hidden_size)
    heads = DecisionHeads(hidden_size)
    heads.load_state_dict(load_file(weights), strict=True)
    return heads.eval(), manifest


def read_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))


def load_head(
    directory: Path,
    *,
    repo_id: str,
    revision: str,
    hidden_size: int,
    device: str | torch.device = "cpu",
) -> tuple[NoulHead, dict[str, Any]]:
    """Recarga validando que el cabezal pertenece a este backbone y que el fichero no cambió."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") != "noul_head":
        raise CheckpointMismatchError(f"Formato no soportado: {manifest.get('format_version')}")
    base = manifest["base"]
    if (base["repo_id"], base["revision"]) != (repo_id, revision):
        raise CheckpointMismatchError(f"Base {base['repo_id']}@{base['revision']} != {repo_id}@{revision}")
    if manifest["hidden_size"] != hidden_size:
        raise CheckpointMismatchError(f"hidden_size {manifest['hidden_size']} != {hidden_size}")
    weights = directory / manifest["weights"]["file"]
    if sha256_file(weights) != manifest["weights"]["sha256"]:
        raise CheckpointMismatchError("sha256 del fichero de pesos no coincide con el manifiesto")
    head = NoulHead(hidden_size)
    head.load_state_dict(load_file(weights), strict=True)
    return head.to(device).eval(), manifest


def _verify_main(argv: list[str]) -> int:
    """Proceso nuevo: recarga el cabezal y compara logits sobre representaciones guardadas."""
    p = argparse.ArgumentParser(prog="python -m gemma_system_one.checkpoint verify")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True, help="safetensors con 'pooled' y 'logits'")
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args(argv)

    manifest = read_manifest(args.checkpoint)
    head, _ = load_head(
        args.checkpoint,
        repo_id=manifest["base"]["repo_id"],
        revision=manifest["base"]["revision"],
        hidden_size=manifest["hidden_size"],
        device=args.device,
    )
    probe = load_file(args.probe)
    with torch.inference_mode():
        logits = head(probe["pooled"].to(args.device)).cpu()
    diff = (logits - probe["logits"]).abs().max().item()
    ok = bool(torch.isfinite(logits).all()) and diff <= args.atol
    print(
        json.dumps(
            {"ok": ok, "device": args.device, "max_abs_diff": diff, "atol": args.atol, "pid": os.getpid()}
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "verify":
        print(
            "uso: python -m gemma_system_one.checkpoint verify --checkpoint DIR --probe FILE", file=sys.stderr
        )
        sys.exit(2)
    sys.exit(_verify_main(sys.argv[2:]))
