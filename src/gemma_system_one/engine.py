"""Motor de inferencia del servicio (spec §9): el mismo camino que la evaluación, sin dataset.

- Carga verificada con ``load_decision_model``: etapa, primitivas entrenadas, plantilla,
  procesador y huella de extracción. Temperaturas opcionales desde un artefacto de calibración
  vinculado al checkpoint (``load_calibration``).
- Cada pregunta se expande con ``serialization.expand`` (la serialización del entrenamiento), una fila
  por forward con ``extract_pooled`` (decisiones 0002 y 0007), cabezales en CPU/FP32 y respuesta con
  ``inference.reconstruct``. Sin ``generate()``: ``generated_tokens`` es siempre 0.
- Modalidad fija del checkpoint: los entrenados con imagen exigen imagen; los de texto la rechazan.
- ``eval()`` + ``torch.inference_mode()``; sin caché de representaciones en servicio.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .calibration import load_calibration
from .checkpoint import checkpoint_identity
from .contracts import ChoiceQuestion, NoulQuestion, ScoreQuestion
from .inference import CONFIDENCE_METHOD, reconstruct
from .models.heads import PRIMITIVES
from .serialization import expand


class RequestError(ValueError):
    """Error atribuible a la petición (HTTP 422 salvo que se indique otro código)."""

    def __init__(self, message: str, status: int = 422, code: str = "invalid_request"):
        super().__init__(message)
        self.status, self.code = status, code


@dataclass
class DecisionResult:
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    timing_ms: dict[str, float] = field(default_factory=dict)


def detect_modality(extra: dict[str, Any]) -> str:
    """``text+image`` si el checkpoint se entrenó con imágenes; ``text`` en otro caso.

    Checkpoints nuevos: ``extra.input_modality``. Anteriores: se inspecciona el dataset de
    entrenamiento declarado (sólo si existe localmente)."""
    declared = extra.get("input_modality")
    if declared in ("text", "text+image"):
        return declared
    if extra.get("images", "use") == "omit":
        return "text"
    from .data.dataset import load_dataset

    root = Path(extra.get("dataset_root", ""))
    if not (root / "examples.jsonl").is_file():
        raise RuntimeError("No se puede determinar la modalidad: declara 'modality' en la configuración")
    train_ids = set(extra.get("train_examples", []))
    uses = {e.image_path is not None for e in load_dataset(root).examples if e.id in train_ids}
    if len(uses) != 1:
        raise RuntimeError("El entrenamiento mezcla ejemplos con y sin imagen: modalidad ambigua")
    return "text+image" if uses.pop() else "text"


class DecisionEngine:
    """Carga un checkpoint de fases 2–4 y responde peticiones ya validadas (un hilo a la vez)."""

    def __init__(self, checkpoint: Path, calibration: Path | None = None, modality: str | None = None):
        from .training.decisions_pipeline import load_decision_model

        self.model = load_decision_model(checkpoint)
        self.extra = self.model.extra
        self.primitives = set(self.extra["primitives"])
        detected = modality or detect_modality(self.extra)
        if modality is not None and self.extra.get("input_modality") not in (None, modality):
            raise RuntimeError(
                f"modality={modality} contradice el checkpoint ({self.extra['input_modality']})"
            )
        self.modality = detected
        self.temperatures = {p: 1.0 for p in PRIMITIVES}
        self.calibrated = {p: False for p in PRIMITIVES}
        ident = checkpoint_identity(checkpoint)
        cal_sha = None
        if calibration is not None:
            cal = load_calibration(calibration, checkpoint)
            self.temperatures.update(cal["temperatures"])
            self.calibrated.update(cal["calibrated"])
            cal_sha = hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
        self.checkpoint_id = f"{ident['kind']}:{ident['manifest_sha256'][:16]}" + (
            f"+cal:{cal_sha[:12]}" if cal_sha else ""
        )
        self.max_length = self.model.cfg.runtime.max_length
        self.heads = self.model.heads.cpu().eval()

    @property
    def device(self) -> str:
        bb = self.model.encoder.backbone
        return "not_loaded" if bb is None else str(bb.device)

    def warmup(self) -> dict[str, Any]:
        """Petición sintética completa para el ``ready``: carga la base (y LoRA) con el mismo
        codificador perezoso que la evaluación, hace el forward y reconstruye la respuesta."""
        state = "Warmup request."
        probes = {
            "noul": NoulQuestion(type="noul", instructions="Is this a warmup request?"),
            "choice": ChoiceQuestion(
                type="choice", instructions="Select the request type.", criteria={"a": "Warmup", "b": "Other"}
            ),
            "score": ScoreQuestion(
                type="score",
                instructions="Rate the evidence for warmup.",
                criteria=["No evidence", "Explicit evidence"],
            ),
        }
        primitive = next(p for p in PRIMITIVES if p in self.primitives)
        q = {"warmup": probes[primitive]}
        image = None
        if self.modality == "text+image":
            import io

            from PIL import Image

            from .images import content_address, decode_image_bytes

            buf = io.BytesIO()
            Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
            pil, ext = decode_image_bytes(buf.getvalue(), name="warmup")
            image = (pil, content_address(buf.getvalue(), ext))
        t0 = time.perf_counter()
        res = self.decide(state, q, image)
        return {"seconds": round(time.perf_counter() - t0, 4), "usage": res.usage}

    def decide(self, state: Any, questions: dict[str, Any], image=None) -> DecisionResult:
        """``image``: None o ``(PIL RGB validada, referencia images/<sha256>.<ext>)``."""
        if self.modality == "text+image" and image is None:
            raise RequestError("Este checkpoint se entrenó con imagen: la petición debe incluir 'image'")
        if self.modality == "text" and image is not None:
            raise RequestError("Este checkpoint se entrenó sólo con texto: no admite 'image'")
        unsupported = {q.type for q in questions.values()} - self.primitives
        if unsupported:
            raise RequestError(f"Primitivas no entrenadas en este checkpoint: {sorted(unsupported)}")
        pil, ref = image if image is not None else (None, None)
        plan, texts = [], []
        for qid, q in questions.items():
            rows = expand(state, q, ref)
            plan.append((qid, q, rows, len(texts)))
            texts.extend(r.text for r in rows)
        images = [pil] * len(texts) if pil is not None else None
        from .models.encoding import InputTooLongError

        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                reps, stats = self.model.encoder(texts, images)
            except InputTooLongError as exc:
                raise RequestError(
                    f"La entrada supera el presupuesto de {self.max_length} tokens por fila"
                ) from exc
            t_forward = time.perf_counter() - t0
            answers = {}
            for qid, q, rows, start in plan:
                z = self.heads(q.type, reps[start : start + len(rows)]).float()
                answers[qid] = reconstruct(q, rows, z, temperature=self.temperatures[q.type])
        _check_answers(questions, answers)
        usage = {
            "questions": len(questions),
            "expanded_rows": len(texts),
            "backbone_forwards": stats.backbone_forwards,
            "processed_input_tokens": stats.valid_tokens,
            "image_tokens": stats.image_tokens,
            "padding_tokens": stats.padding_tokens,
            "generated_tokens": 0,
        }
        timing = {
            "forward_synchronized": round(stats.seconds_synchronized * 1000, 2),
            "engine_total": round((time.perf_counter() - t0) * 1000, 2),
            "forward_wall": round(t_forward * 1000, 2),
        }
        return DecisionResult(answers, usage, timing)

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.extra["stage"],
            "modality": self.modality,
            "primitives": sorted(self.primitives),
            "temperatures": self.temperatures,
            "calibrated": self.calibrated,
            "confidence_method": CONFIDENCE_METHOD,
            "device": self.device,
            "max_length": self.max_length,
        }


def _check_answers(questions: dict[str, Any], answers: dict[str, dict[str, Any]]) -> None:
    """Invariantes de salida (spec §9): conjunto de IDs, rangos, normalización y pertenencia."""
    if set(answers) != set(questions):
        raise RuntimeError("Respuestas no alineadas con las preguntas")
    for qid, q in questions.items():
        a = answers[qid]
        if isinstance(q, NoulQuestion):
            if not 0.0 <= a["noul"] <= 1.0:
                raise RuntimeError("noul fuera de [0, 1]")
        elif isinstance(q, ChoiceQuestion):
            if a["choice"] not in q.criteria or set(a["probabilities"]) != set(q.criteria):
                raise RuntimeError("choice fuera del conjunto de opciones")
        elif isinstance(q, ScoreQuestion) and len(a["probabilities"]) != len(q.criteria):
            raise RuntimeError("score con probabilidades desalineadas")
