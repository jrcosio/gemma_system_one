"""Fase 3 en CPU: Gemma4Model real diminuto + procesador doble (contratos, no calidad).

Cubre: arranque desde cabezales de fase 2 (época 0 idéntica), reanudación exacta tras una
interrupción, checkpoint de despliegue recargado desde el texto, calibración vinculada sólo
sobre ``calibration``, test protegido, baselines en la partición evaluada y comparación.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

import gemma_system_one.features as features_mod
import gemma_system_one.training.decisions_pipeline as dp
import gemma_system_one.training.lora_pipeline as lp
import gemma_system_one.training.pipeline as p1
from gemma_system_one.calibration import load_calibration, run_calibrate
from gemma_system_one.data.dataset import load_dataset
from gemma_system_one.data.generate_mixed import write_mixed_dataset
from gemma_system_one.data.split import SplitError, make_split, split_path, write_split
from gemma_system_one.metrics import compare_predictions

FAKE_PROC_FP = {"files_sha256": {"fake": "0"}, "chat_kwargs": {}, "padding_side": "right"}


@pytest.fixture
def setup(tmp_path, tiny_checkpoint, fake_processor, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "model": {"repo_id": "google/gemma-4-E2B-it", "revision": "a" * 40, "dtype": "float32"},
                "runtime": {"device": "cpu", "max_length": 250},
            }
        )
    )
    ds = tmp_path / "mixed"
    write_mixed_dataset(ds, 60, seed=4)
    write_split(make_split(load_dataset(ds), 0), split_path(ds, 0))
    for mod in (dp, p1, lp):
        monkeypatch.setattr(mod, "require_snapshot", lambda cfg: tiny_checkpoint, raising=False)
        monkeypatch.setattr(mod, "load_processor", lambda snap: fake_processor, raising=False)
        monkeypatch.setattr(mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP, raising=False)
    monkeypatch.setattr(features_mod, "processor_fingerprint", lambda snap: FAKE_PROC_FP)
    heads_cfg = tmp_path / "heads.yaml"
    heads_cfg.write_text(
        yaml.safe_dump(
            {
                "kind": "decision_heads",
                "name": "tiny_heads",
                "base_config": str(base),
                "dataset": str(ds),
                "cache_dir": str(tmp_path / "cache"),
                "runs_dir": str(tmp_path / "runs"),
                "train": {"epochs": 3, "lr": 0.01},
            }
        )
    )
    init = Path(dp.run_train_decisions(heads_cfg)["checkpoint"])

    def lora_cfg(name: str, **train) -> Path:
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "kind": "lora_decision_heads",
                    "name": name,
                    "base_config": str(base),
                    "dataset": str(ds),
                    "init_heads_from": str(init),
                    "runs_dir": str(tmp_path / "runs"),
                    "lora": {"r": 4, "alpha": 8, "dropout": 0.1},
                    "train": {
                        "epochs": 2,
                        "lr_lora": 5e-3,
                        "lr_heads": 5e-3,
                        "questions_per_step": 16,
                        "max_train_questions": 48,
                        "resume_every_steps": 2,
                        **train,
                    },
                }
            )  # fmt: skip
        )
        return path

    return init, lora_cfg


def _weights(ckpt: Path) -> dict[str, torch.Tensor]:
    return {**load_file(ckpt / "adapter.safetensors"), **load_file(ckpt / "head.safetensors")}


def test_lora_train_resume_reload_calibrate_and_blind_test(setup):
    init, lora_cfg = setup
    cfg = lora_cfg("tiny_lora")
    full = lp.run_train_lora(cfg)
    assert not full["interrupted"] and full["global_steps"] == 6 and full["base_params_unchanged_probe"]
    hist = full["history"]
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    # Época 0 = punto de partida exacto (B = 0): mismas NLL que los cabezales de fase 2.
    init_val = json.loads((init.parent / "metrics.json").read_text())["metrics"]["models"]["gemma_heads"]
    assert hist[0]["eval_nll"]["all"] == pytest.approx(init_val["validation"]["mean_nll_all"], abs=1e-6)
    manifest = json.loads((Path(full["checkpoint"]) / "manifest.json").read_text())
    assert manifest["kind"] == "lora_decision_heads" and manifest["extra"]["cache_dir"] is None
    assert manifest["adapter"]["config"]["targets"][0] == "language_model.layers.0.self_attn.q_proj"
    assert not any("base_layer" in k for k in manifest["adapter"]["keys"])  # sin pesos base
    assert (
        "lora_heads_minus_initial_frozen_heads.nll_all"
        in full["metrics"]["validation_bootstrap"]["intervals"]
    )

    # Interrupción + reanudación: mismos pesos finales e historial que sin interrumpir.
    part = lp.run_train_lora(cfg, stop_after_steps=3)
    assert part["interrupted"] and part["global_step"] == 3
    run_dir = Path(part["run_dir"])
    with pytest.raises(ValueError, match="configuración"):
        lp.run_train_lora(lora_cfg("tiny_lora_seed1", seed=1), resume=run_dir)
    # Un paso registrado después del último estado guardado (p. ej., proceso matado) se rehace y se aparta.
    with (run_dir / "steps.jsonl").open("a") as fh:
        fh.write(json.dumps({"step": 4, "note": "perdido"}) + "\n")
    resumed = lp.run_train_lora(cfg, resume=run_dir)
    assert json.loads((run_dir / "steps.discarded.jsonl").read_text())["note"] == "perdido"
    assert resumed["resumed"] and resumed["global_steps"] == 6
    a, b = _weights(Path(full["checkpoint"])), _weights(Path(resumed["checkpoint"]))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert [h["eval_nll"] for h in resumed["history"]] == [h["eval_nll"] for h in hist]
    steps = [json.loads(x)["step"] for x in (run_dir / "steps.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="terminó"):
        lp.run_train_lora(cfg, resume=run_dir)

    # Recarga desde el texto (proceso de evaluación sin caché): mismos logits de validación.
    ckpt = Path(full["checkpoint"])
    ev = dp.run_evaluate_decisions(ckpt, "validation", robustness=True)
    assert ev["stage"] == "phase3_lora_decision_heads" and ev["use_cache"] is False
    assert ev["reload_vs_training"]["ok"], ev["reload_vs_training"]
    rob = ev["robustness"]["choice_id_rename_permutation"]
    assert rob["rows_identical"] == rob["questions"]

    # Calibración: sólo partición calibration; artefacto vinculado al checkpoint.
    with pytest.raises(ValueError, match="calibration"):
        run_calibrate(ckpt, "validation")
    cal = run_calibrate(ckpt, "calibration")
    assert set(cal["temperatures"]) == {"noul", "choice", "score"}
    assert all(t > 0 for t in cal["temperatures"].values())
    with pytest.raises(ValueError, match="no pertenece"):
        load_calibration(cal["path"], init)

    # Test ciego: protegido sin permiso explícito; con él, calibrado y con baselines de train.
    with pytest.raises(SplitError, match="test"):
        dp.run_evaluate_decisions(ckpt, "test", calibration=Path(cal["path"]))
    t_lora = dp.run_evaluate_decisions(
        ckpt, "test", allow_test=True, calibration=Path(cal["path"]), baselines=True
    )
    assert t_lora["final_test_used"] and "metrics_calibrated" in t_lora
    assert set(t_lora["baselines"]) == {"fitted_on", "prior", "bow"}
    assert "checkpoint_minus_bow.nll_all" in t_lora["bootstrap"]["intervals"]
    cal_init = run_calibrate(init, "calibration")
    t_init = dp.run_evaluate_decisions(init, "test", allow_test=True, calibration=Path(cal_init["path"]))
    rows = [
        [json.loads(x) for x in Path(r["predictions_path"]).read_text().splitlines()]
        for r in (t_init, t_lora)
    ]
    assert all("temperature" in r and "nll_uncalibrated" in r for r in rows[1])
    cmp = compare_predictions(rows[0], rows[1], reps=50)
    assert cmp["questions"] == t_lora["usage"]["questions"]
    assert cmp["point"]["b_minus_a.nll_all"] == pytest.approx(
        t_lora["metrics_calibrated"]["mean_nll_all"] - t_init["metrics_calibrated"]["mean_nll_all"], abs=1e-9
    )
    with pytest.raises(ValueError, match="mismas preguntas"):
        compare_predictions(rows[0][1:], rows[1])


def test_init_heads_must_match_dataset_and_split(setup, tmp_path):
    init, lora_cfg = setup
    other = tmp_path / "other"
    write_mixed_dataset(other, 30, seed=5)
    write_split(make_split(load_dataset(other), 0), split_path(other, 0))
    cfg = lora_cfg("tiny_lora_bad")
    raw = yaml.safe_load(cfg.read_text())
    raw["dataset"] = str(other)
    cfg.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="dataset distinto"):
        lp.run_train_lora(cfg)


def test_lora_overfit_gate_with_frozen_heads(setup):
    """LoRA solo (cabezales fijos) reduce la pérdida de train: el adaptador aprende por sí mismo."""
    _, lora_cfg = setup
    cfg = lora_cfg("tiny_lora_only", train_heads=False, select_on="train", epochs=5, lr_lora=1e-2,
                   max_train_questions=16, questions_per_step=4, warmup_fraction=0.0)  # fmt: skip
    rep = lp.run_train_lora(cfg)
    nll = [h["eval_nll"]["all"] for h in rep["history"]]
    assert nll[-1] < 0.8 * nll[0] and rep["selected_epoch"] == 5, nll
    ckpt = Path(rep["checkpoint"])
    init_heads = load_file(Path(yaml.safe_load(cfg.read_text())["init_heads_from"]) / "head.safetensors")
    assert all(torch.equal(v, init_heads[k]) for k, v in load_file(ckpt / "head.safetensors").items())


def test_lora_training_with_layer_recomputation_matches_plain(setup):
    """Fase 6 (decisión 0010): la recomputación sólo cambia la memoria, no el entrenamiento."""
    _, lora_cfg = setup
    plain = lp.run_train_lora(lora_cfg("tiny_lora_plain", resume_every_steps=None))
    rec = lp.run_train_lora(lora_cfg("tiny_lora_recompute", resume_every_steps=None, recompute_layers=True))
    a, b = _weights(Path(plain["checkpoint"])), _weights(Path(rec["checkpoint"]))
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(b[k], a[k], rtol=1e-4, atol=1e-6)
    for hp, hr in zip(plain["history"], rec["history"], strict=True):
        assert hr["eval_nll"]["all"] == pytest.approx(hp["eval_nll"]["all"], abs=1e-5)


def test_external_calibration_set_is_bound_checked_and_never_reused_as_test(setup, tmp_path):
    """Fase 6b: temperaturas ajustadas en un conjunto externo sin fugas, nunca evaluadas sobre él."""
    init, _ = setup
    cal_ds, other = tmp_path / "cal_pool", tmp_path / "final"
    write_mixed_dataset(cal_ds, 40, seed=9)
    write_mixed_dataset(other, 20, seed=10)
    with pytest.raises(ValueError, match="--split all"):
        run_calibrate(init, "calibration", dataset=cal_ds)
    cal = run_calibrate(init, "all", dataset=cal_ds)
    assert cal["split"] == "external_calibration" and cal["questions"] == 120
    assert cal["external_dataset"]["sha256"] == load_dataset(cal_ds).sha256
    assert load_calibration(Path(cal["path"]), init)["temperatures"] == cal["temperatures"]
    # El dataset de entrenamiento comparte grupos consigo mismo: no puede ser un conjunto de calibración.
    train_root = Path(yaml.safe_load((init / "manifest.json").read_text())["extra"]["dataset_root"])
    with pytest.raises(ValueError, match="comparte"):
        run_calibrate(init, "all", dataset=train_root)
    with pytest.raises(ValueError, match="mismo conjunto"):
        dp.run_evaluate_decisions(init, "all", dataset=cal_ds, calibration=Path(cal["path"]))
    rep = dp.run_evaluate_decisions(init, "all", dataset=other, calibration=Path(cal["path"]))
    assert rep["calibration"]["temperatures"] == cal["temperatures"] and "metrics_calibrated" in rep
