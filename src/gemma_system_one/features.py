"""Extracción de representaciones (último token válido) y caché en disco.

La caché sólo es válida para el backbone congelado: su clave incluye checkpoint, dtype,
atención, procesador, plantilla, pooling, versiones de bibliotecas, política de lotes
y el hash de cada fila en orden. Cualquier cambio produce otra clave. Con LoRA activo
no debe usarse (las representaciones dependen de los adaptadores).

Con ``microbatch_rows > 1`` la representación en bf16/MPS depende de qué filas
comparten microlote (anchura de padding): ver ``docs/decisions/0002``. Con 1 fila por
microlote es bit a bit reproducible e independiente del resto del conjunto.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import ProjectConfig
from .models.backbone import resolve_device
from .models.encoding import build_chat_batch, image_token_count, processor_fingerprint
from .resources import timed
from .serialization import TEMPLATE_VERSION

POOLING = "last_valid_token_v1"
ORDER = "length_desc"


def backbone_fingerprint(cfg: ProjectConfig, snapshot_dir: Path, microbatch_rows: int) -> dict[str, Any]:
    device = resolve_device(cfg)
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK activo: extracción MPS no verificable")
    return {
        "repo_id": cfg.model.repo_id,
        "revision": cfg.model.revision,
        "backbone_class": cfg.model.backbone_class,
        "dtype": cfg.model.dtype,
        "attn_implementation": cfg.model.attn_implementation,
        "device": str(device),
        "max_length": cfg.runtime.max_length,
        "processor": processor_fingerprint(snapshot_dir),
        "template_version": TEMPLATE_VERSION,
        "pooling": POOLING,
        "extraction": {"microbatch_rows": microbatch_rows, "order": ORDER},
        "versions": {p: metadata.version(p) for p in ("torch", "transformers")},
    }


def fingerprint_hash(fp: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


@dataclass
class ExtractionStats:
    rows: int = 0
    backbone_forwards: int = 0
    valid_tokens: int = 0
    padding_tokens: int = 0
    max_row_tokens: int = 0
    seconds_synchronized: float = 0.0
    cache_hit: bool = False
    image_rows: int = 0
    image_tokens: int = 0


def slice_batch(batch: dict[str, Any], idx: torch.Tensor, width: int) -> dict[str, Any]:
    """Filas ``idx`` de un lote: todo tensor con dimensión de lote se indexa (también
    ``pixel_values``/``image_position_ids``) y los de secuencia se recortan a ``width``."""
    n = batch["attention_mask"].shape[0]
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n:
            v = v[idx]
            if k in SEQUENCE_KEYS:
                v = v[:, :width]
        out[k] = v
    return out


SEQUENCE_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")


def extract_pooled(
    backbone,
    processor,
    texts: list[str],
    *,
    max_length: int,
    microbatch_rows: int,
    images: list[Path | None] | None = None,
) -> tuple[torch.Tensor, ExtractionStats]:
    """Representación FP32 [N, D] en CPU, en el mismo orden que ``texts``.

    Texto: se tokeniza todo junto (padding derecho), se ordena por longitud y se procesa por
    microlotes recortando el padding sobrante; el recorte es válido con padding derecho.
    Con imágenes: una fila por forward y cada imagen se carga (y verifica) al procesar su fila,
    para no materializar ``pixel_values`` de todo el conjunto.
    """
    if images is not None and any(im is not None for im in images):
        return _extract_with_images(backbone, processor, texts, images, max_length, microbatch_rows)
    batch = build_chat_batch(processor, texts, max_length=max_length)
    lengths = batch["attention_mask"].sum(dim=1)
    order = torch.argsort(lengths, descending=True, stable=True)
    out = torch.empty(len(texts), backbone.hidden_size, dtype=torch.float32)
    stats = ExtractionStats(rows=len(texts), max_row_tokens=int(lengths.max()))
    with torch.no_grad(), timed(backbone.device) as t:
        for start in range(0, len(texts), microbatch_rows):
            idx = order[start : start + microbatch_rows]
            sub = slice_batch(batch, idx, int(lengths[idx].max()))
            enc = backbone.encode(sub)
            pooled = enc.pooled().float()
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("Representación no finita")
            out[idx] = pooled.cpu()
            stats.backbone_forwards += 1
            stats.valid_tokens += int(sub["attention_mask"].sum())
            stats.padding_tokens += int(sub["attention_mask"].numel() - sub["attention_mask"].sum())
    stats.seconds_synchronized = round(t["seconds"], 4)
    return out, stats


def _extract_with_images(backbone, processor, texts, images, max_length, microbatch_rows):
    from .images import load_image

    if microbatch_rows != 1:
        raise ValueError("Con imágenes sólo se admite una fila por forward (decisiones 0002 y 0007)")
    if len(images) != len(texts):
        raise ValueError("images debe tener una entrada por fila")
    out = torch.empty(len(texts), backbone.hidden_size, dtype=torch.float32)
    stats = ExtractionStats(rows=len(texts))
    with torch.no_grad(), timed(backbone.device) as t:
        for i, (text, path) in enumerate(zip(texts, images, strict=True)):
            # Ruta del dataset (se carga y verifica) o imagen ya decodificada y validada (API).
            image = load_image(path) if isinstance(path, str | Path) else path
            batch = build_chat_batch(
                processor, [text], max_length=max_length, images=None if image is None else [image]
            )
            pooled = backbone.encode(batch).pooled().float()
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("Representación no finita")
            out[i] = pooled[0].cpu()
            n = int(batch["attention_mask"].sum())
            stats.backbone_forwards += 1
            stats.valid_tokens += n
            stats.max_row_tokens = max(stats.max_row_tokens, n)
            stats.image_rows += image is not None
            stats.image_tokens += image_token_count(batch)
    stats.seconds_synchronized = round(t["seconds"], 4)
    return out, stats


class RepresentationCache:
    def __init__(self, root: Path, fingerprint: dict[str, Any]):
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.fp_hash = fingerprint_hash(fingerprint)

    def key(self, row_hashes: list[str]) -> str:
        return hashlib.sha256((self.fp_hash + "\n" + "\n".join(row_hashes)).encode()).hexdigest()

    def path(self, row_hashes: list[str]) -> Path:
        return self.root / self.fp_hash[:16] / f"{self.key(row_hashes)}.safetensors"

    def get(self, row_hashes: list[str]) -> torch.Tensor | None:
        p = self.path(row_hashes)
        if not p.is_file():
            return None
        with safe_open(p, "pt") as fh:
            meta = fh.metadata() or {}
            if (
                meta.get("fingerprint_sha256") != self.fp_hash
                or json.loads(meta.get("rows", "[]")) != row_hashes
            ):
                return None
            return fh.get_tensor("pooled")

    def put(self, row_hashes: list[str], pooled: torch.Tensor, stats: ExtractionStats) -> Path:
        p = self.path(row_hashes)
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.parent / "fingerprint.json").write_text(json.dumps(self.fingerprint, indent=2), encoding="utf-8")
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        os.close(fd)
        meta = {
            "fingerprint_sha256": self.fp_hash,
            "rows": json.dumps(row_hashes),
            "stats": json.dumps(asdict(stats)),
        }
        save_file({"pooled": pooled.contiguous()}, tmp, metadata=meta)
        os.replace(tmp, p)
        return p


def get_representations(
    cache: RepresentationCache | None,
    row_hashes: list[str],
    texts: list[str],
    compute,
    images: list | None = None,
) -> tuple[torch.Tensor, ExtractionStats]:
    """Lee de la caché o calcula con ``compute(texts[, images])`` y guarda.

    La clave de la caché ya incluye la imagen: el hash de cada fila la contiene (decisión 0007)."""
    if cache is not None:
        hit = cache.get(row_hashes)
        if hit is not None:
            return hit, ExtractionStats(rows=len(texts), cache_hit=True)
    pooled, stats = compute(texts) if images is None else compute(texts, images)
    if cache is not None:
        cache.put(row_hashes, pooled, stats)
    return pooled, stats
