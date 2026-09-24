from __future__ import annotations

from pathlib import Path

import pytest

from gemma_system_one.config import ProjectConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
E2B_CONFIG = REPO_ROOT / "configs" / "e2b_text.yaml"

# Gemma 4 diminuto con la misma clase y ruta de código que el real (incluye capas con KV compartido).
TINY_TEXT_CONFIG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    global_head_dim=32,
    hidden_size_per_layer_input=8,
    vocab_size=512,
    vocab_size_per_layer_input=512,
    num_kv_shared_layers=2,
    sliding_window=8,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    max_position_embeddings=256,
    pad_token_id=0,
)


def tiny_gemma4_config():
    from transformers import Gemma4Config

    return Gemma4Config(text_config=dict(TINY_TEXT_CONFIG), vision_config=None, audio_config=None)


# Torre de visión diminuta con la misma ruta de código que E2B: parches de 16 px (768 valores) y
# agrupación 3×3. El procesador doble produce 36 parches → 4 tokens visuales por imagen.
TINY_VISION_CONFIG = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    global_head_dim=16,
    patch_size=16,
    pooling_kernel_size=3,
    position_embedding_size=64,
    default_output_length=4,
)
TINY_IMAGE_TOKEN, TINY_BOI, TINY_EOI, TINY_SOFT_TOKENS = 500, 498, 499, 4


def tiny_gemma4_vision_config():
    from transformers import Gemma4Config

    return Gemma4Config(
        text_config=dict(TINY_TEXT_CONFIG),
        vision_config=dict(TINY_VISION_CONFIG),
        audio_config=None,
        image_token_id=TINY_IMAGE_TOKEN,
        boi_token_id=TINY_BOI,
        eoi_token_id=TINY_EOI,
    )


@pytest.fixture
def tiny_vision_checkpoint(tmp_path: Path) -> Path:
    """Checkpoint diminuto con torre de visión (texto + imagen), aleatorio y guardado en disco."""
    import torch
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(tiny_gemma4_vision_config())
    out = tmp_path / "tiny_vision_ckpt"
    model.save_pretrained(out)
    return out


@pytest.fixture(autouse=True)
def _source_archive_in_tmp(tmp_path_factory, monkeypatch):
    """Los checkpoints de los tests guardan su copia de fuentes fuera de artifacts/ del repo."""
    monkeypatch.setenv("GSO_SOURCE_ARCHIVE_DIR", str(tmp_path_factory.getbasetemp() / "source_archive"))


@pytest.fixture
def e2b_cfg() -> ProjectConfig:
    return load_config(E2B_CONFIG)


@pytest.fixture
def cpu_cfg(e2b_cfg: ProjectConfig, tmp_path: Path) -> ProjectConfig:
    """Config real con rutas temporales, CPU y FP32 para tests sin hardware."""
    return e2b_cfg.model_copy(
        update={
            "runtime": e2b_cfg.runtime.model_copy(update={"device": "cpu"}),
            "model": e2b_cfg.model.model_copy(update={"dtype": "float32"}),
            "paths": e2b_cfg.paths.model_copy(
                update={"artifacts_dir": tmp_path / "artifacts", "reports_dir": tmp_path / "reports"}
            ),
        }
    )


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Checkpoint ``Gemma4ForConditionalGeneration`` diminuto y aleatorio guardado en disco."""
    import torch
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(tiny_gemma4_config())
    out = tmp_path / "tiny_ckpt"
    model.save_pretrained(out)
    return out


class _FakeTokenizer:
    padding_side = "left"  # como el real: el código debe fijar "right"


class FakeProcessor:
    """Doble de test del procesador: palabras -> IDs estables < 512, BOS=2, PAD=0.

    Imita la firma de ``apply_chat_template`` usada por ``build_chat_batch`` y respeta
    ``padding_side``. Sólo para tests de contratos; no sustituye al procesador real.
    """

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict] = []

    def _ids(self, text: str) -> list[int]:
        import hashlib

        words = text.split()
        return [2] + [3 + int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 500 for w in words] + [1]

    def apply_chat_template(
        self, conversations, *, tokenize, return_dict, return_tensors, processor_kwargs, **kw
    ):
        import torch

        assert tokenize and return_dict and return_tensors == "pt" and processor_kwargs == {"padding": True}
        assert kw == {"add_generation_prompt": True, "enable_thinking": False}
        self.calls.append({"n": len(conversations), "padding_side": self.tokenizer.padding_side})
        seqs, pixels = [], []
        for c in conversations:
            parts = c[0]["content"]
            text = next(p["text"] for p in parts if p["type"] == "text")
            images = [p["image"] for p in parts if p["type"] == "image"]
            ids = self._ids(text)
            if images:
                assert len(images) == 1 and parts[0]["type"] == "image"  # imagen antes del texto
                ids = [ids[0], TINY_BOI, *[TINY_IMAGE_TOKEN] * TINY_SOFT_TOKENS, TINY_EOI, *ids[1:]]
                pixels.append(self._patches(images[0]))
            seqs.append(ids)
        assert not pixels or len(pixels) == len(seqs), "el doble no mezcla filas con y sin imagen"
        width = max(map(len, seqs))
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            sl = slice(0, len(s)) if self.tokenizer.padding_side == "right" else slice(width - len(s), width)
            ids[i, sl] = torch.tensor(s)
            mask[i, sl] = 1
        out = {
            "input_ids": ids,
            "attention_mask": mask,
            "mm_token_type_ids": (ids == TINY_IMAGE_TOKEN).long(),
        }
        if pixels:
            out["pixel_values"] = torch.stack(pixels)
            grid = torch.stack(torch.meshgrid(torch.arange(6), torch.arange(6), indexing="xy"), -1).reshape(
                36, 2
            )
            out["image_position_ids"] = grid.unsqueeze(0).repeat(len(seqs), 1, 1)
        return out

    @staticmethod
    def _patches(image):
        """96×96 px → 36 parches de 16×16×3 en [0, 1]: depende de verdad del contenido de la imagen."""
        import numpy as np
        import torch

        arr = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).reshape(6, 16, 6, 16, 3).permute(0, 2, 1, 3, 4)
        return t.reshape(36, 16 * 16 * 3)


@pytest.fixture
def fake_processor() -> FakeProcessor:
    return FakeProcessor()


def make_example(**overrides):
    """Ejemplo Noul válido mínimo; ``overrides`` sustituye campos de primer nivel."""
    from gemma_system_one.contracts import Example

    raw = {
        "schema_version": 1,
        "id": "n-001",
        "group_id": "case-001",
        "task_family": "refund",
        "language": "es",
        "state": "Solicito que me devuelvan el cobro duplicado.",
        "image_path": None,
        "question": {"type": "noul", "instructions": "¿El cliente solicita una devolución?"},
        "target": {"label": 1},
        "provenance": {"source": "fixture", "generator_version": "v1", "label_method": "manual"},
    }
    raw.update(overrides)
    return Example.model_validate(raw)
