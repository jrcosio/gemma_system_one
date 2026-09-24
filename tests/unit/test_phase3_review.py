"""Regresiones de revisión: interrupciones durante el guardado y escritura del log."""

import json

import pytest
import torch
from torch import nn

from gemma_system_one.models.heads import DecisionHeads
from gemma_system_one.training import lora


def test_repeated_resume_save_preserves_latest_on_publish_failure(tmp_path, monkeypatch):
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.lora_A = nn.Linear(2, 2, bias=False)
    heads = DecisionHeads(2)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    kwargs = dict(
        model=model,
        heads=heads,
        opt=opt,
        sched=sched,
        state=lora.LoopState(global_step=50),
        best=None,
        meta={},
    )
    old = lora.save_resume(tmp_path, **kwargs)
    original = lora.os.replace

    def fail_publish(src, dst):
        if str(src).split("/")[-1].startswith(".step-"):
            raise OSError("interrupción simulada antes de publicar")
        return original(src, dst)

    monkeypatch.setattr(lora.os, "replace", fail_publish)
    with pytest.raises(OSError, match="interrupción"):
        lora.save_resume(tmp_path, **kwargs)
    restored, payload = lora.read_resume(tmp_path)
    assert restored == old and payload["state"]["global_step"] == 50
    assert (restored / "lora.safetensors").is_file()


def test_resume_accepts_only_truncated_final_log_line(tmp_path):
    log = tmp_path / "steps.jsonl"
    log.write_text('{"step": 1}\n{"step": 2}\n{"step":')
    lora._discard_steps_after(log, 1)
    assert [json.loads(x)["step"] for x in log.read_text().splitlines()] == [1]
    discarded = log.with_name("steps.discarded.jsonl").read_text()
    assert json.loads(discarded.splitlines()[0])["step"] == 2
    assert log.with_name("steps.truncated.txt").read_text() == '{"step":'
    log.write_text('{broken}\n{"step": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        lora._discard_steps_after(log, 1)


def test_comparison_rejects_different_labels_for_same_ids():
    from gemma_system_one.metrics import compare_predictions

    a = [
        {
            "id": "q",
            "group_id": "g",
            "type": "noul",
            "target_index": 0,
            "target_description": 0,
            "nll": 1.0,
            "correct": False,
        }
    ]
    b = [{**a[0], "target_index": 1, "target_description": 1, "nll": 0.1, "correct": True}]
    with pytest.raises(ValueError, match="etiqueta"):
        compare_predictions(a, b, reps=10)
