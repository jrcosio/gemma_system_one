"""Entrypoint ``gso``. Cada subcomando reutiliza los módulos del paquete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _cmd_download(args: argparse.Namespace) -> int:
    from .hub import download

    cfg = load_config(args.config)
    res = download(cfg, verify_hash=args.verify_hash)
    state = "cache hit (sin descarga)" if res.cache_hit else "descargado"
    print(f"[download] {cfg.model.repo_id}@{cfg.model.revision}: {state}")
    for check in res.file_checks:
        print(f"  {json.dumps(check, ensure_ascii=False)}")
    print(f"[download] manifiesto: {res.manifest_path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    cfg = load_config(args.config)
    report = run_doctor(cfg, skip_model=args.skip_model, json_out=args.json)
    return 0 if report["verdict"] in ("pass", "partial") else 1


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _cmd_generate_data(args: argparse.Namespace) -> int:
    if args.kind == "vision":
        from .data.generate_vision import write_vision_dataset

        manifest = write_vision_dataset(
            args.out,
            args.cases,
            args.seed,
            variant=args.variant,
            questions_per_case=args.questions_per_case,
            version=args.vision_version,
            split_seed=args.split_seed,
        )
        _print_json(manifest)
        return 0
    if args.kind == "mixed":
        from .data.generate_mixed import write_mixed_dataset

        manifest = write_mixed_dataset(
            args.out,
            args.cases,
            args.seed,
            variant=args.variant,
            questions_per_case=args.questions_per_case,
            version=args.generator_version,
        )
    else:
        from .data.generate import write_dataset

        manifest = write_dataset(
            args.out, n_cases=args.cases, seed=args.seed, questions_per_case=args.questions_per_case
        )
    _print_json(manifest)
    return 0


def _cmd_validate_data(args: argparse.Namespace) -> int:
    from .data.dataset import validate_dataset

    report = validate_dataset(args.dataset)
    _print_json(
        {
            "ok": report.ok,
            "sha256": report.sha256,
            "errors": report.errors,
            "warnings": report.warnings,
            "stats": report.stats,
        }
    )
    return 0 if report.ok else 1


def _cmd_split(args: argparse.Namespace) -> int:
    from .data.dataset import load_dataset
    from .data.split import check_planned_split, make_split, split_path, write_split

    dataset = load_dataset(args.dataset)
    manifest = make_split(dataset, args.seed)
    check_planned_split(args.dataset, manifest)
    path = split_path(args.dataset, args.seed)
    created = write_split(manifest, path)
    print(f"[split] {'creado' if created else 'ya existía idéntico'}: {path}")
    _print_json({"summary": manifest["summary"], "checks": manifest["checks"]})
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    from .training.pipeline import run_baselines

    _print_json(run_baselines(args.config))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import yaml

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if kind != "lora_decision_heads" and (args.resume is not None or args.stop_after_steps is not None):
        raise SystemExit("--resume y --stop-after-steps sólo aplican a kind: lora_decision_heads")
    if kind == "lora_decision_heads":
        return _train_lora(args)
    if kind == "decision_heads":
        return _train_decisions(args.config)
    from .training.pipeline import run_train

    report = run_train(args.config)
    m = report["metrics"]["models"]
    print(
        f"[train] run: {report['run_dir']}  época elegida: {report['selected_epoch']} ({report['select_on']})"
    )
    for name, per in m.items():
        v, t = per["validation"], per["train"]
        print(
            f"  {name:13s} train nll={t['nll']:.4f} acc={t['accuracy']:.3f} | "
            f"val nll={v['nll']:.4f} brier={v['brier']:.4f} acc={v['accuracy']:.3f} (n={v['n']})"
        )
    if "overfit_check" in report["metrics"]:
        print(f"  overfit_check: {report['metrics']['overfit_check']}")
    return 0 if report["metrics"].get("overfit_check", {"passed": True})["passed"] else 1


def _train_decisions(config: Path) -> int:
    from .training.decisions_pipeline import run_train_decisions

    report = run_train_decisions(config)
    print(
        f"[train] run: {report['run_dir']}  época elegida: {report['selected_epoch']} ({report['select_on']})"
    )
    for name, per in report["metrics"]["models"].items():
        for split in ("train", "validation"):
            m = per[split]
            parts = [f"nll_all={m['mean_nll_all']:.4f}"]
            for p in ("noul", "choice", "score"):
                if p in m:
                    parts.append(f"{p}: nll={m[p]['nll']:.3f} acc={m[p]['accuracy']:.3f} (n={m[p]['n']})")
            print(f"  {name:12s} {split:10s} " + " | ".join(parts))
    if "overfit_check" in report["metrics"]:
        print(f"  overfit_check: {report['metrics']['overfit_check']}")
    return 0 if report["metrics"].get("overfit_check", {"passed": True})["passed"] else 1


def _train_lora(args: argparse.Namespace) -> int:
    from .training.lora_pipeline import run_train_lora

    report = run_train_lora(args.config, resume=args.resume, stop_after_steps=args.stop_after_steps)
    if report["interrupted"]:
        print(f"[train] interrumpido en el paso {report['global_step']}; reanudar con: {report['resume']}")
        return 0
    print(
        f"[train] run: {report['run_dir']}  época elegida: {report['selected_epoch']} ({report['select_on']})"
        f"  pasos: {report['global_steps']}  base sin cambios: {report['base_params_unchanged_probe']}"
    )
    for h in report["history"]:
        loss = h.get("train_loss_mean_steps")
        print(f"  época {h['epoch']}: eval nll_all={h['eval_nll']['all']:.4f} train_loss={loss}")
    for name, per in report["metrics"]["models"].items():
        for split, m in per.items():
            print(f"  {name:20s} {split:10s} {json.dumps(_metric_summary(m), ensure_ascii=False)}")
    if "overfit_check" in report["metrics"]:
        print(f"  overfit_check: {report['metrics']['overfit_check']}")
    return 0 if report["metrics"].get("overfit_check", {"passed": True})["passed"] else 1


_PHASE23 = ("phase2_decision_heads_frozen_backbone", "phase3_lora_decision_heads")
_BULKY = ("metrics", "metrics_calibrated", "baselines", "bootstrap", "robustness", "vision_ablation")


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .checkpoint import read_manifest

    stage = read_manifest(args.checkpoint).get("extra", {}).get("stage")
    if stage in _PHASE23:
        from .training.decisions_pipeline import run_evaluate_decisions

        report = run_evaluate_decisions(
            args.checkpoint,
            args.split,
            allow_test=args.final_test,
            use_cache=not args.no_cache,
            dataset=args.dataset,
            robustness=args.robustness,
            calibration=args.calibration,
            baselines=args.baselines,
            vision_ablation=args.vision_ablation,
        )
        summary = {k: v for k, v in report.items() if k not in _BULKY}
        summary["metrics_summary"] = _metric_summary(report["metrics"])
        if "metrics_calibrated" in report:
            summary["metrics_calibrated_summary"] = _metric_summary(report["metrics_calibrated"])
        if "baselines" in report:
            summary["baselines_summary"] = {
                k: _metric_summary(v) for k, v in report["baselines"].items() if isinstance(v, dict)
            }
        if "robustness" in report:
            summary["robustness"] = {k: v for k, v in report["robustness"].items() if k != "extraction"}
        if "vision_ablation" in report:
            va = report["vision_ablation"]
            summary["vision_ablation"] = {
                k: v.get("all", v) if isinstance(v, dict) else v for k, v in va.items() if k != "extraction"
            }
        _print_json(summary)
        reload = report.get("reload_vs_training")
        return 0 if reload is None or reload["ok"] else 1
    if (
        args.dataset is not None
        or args.robustness
        or args.calibration is not None
        or args.baselines
        or args.vision_ablation
    ):
        raise SystemExit(
            "--dataset, --robustness, --calibration, --baselines y --vision-ablation sólo aplican a fases 2–4"
        )
    from .training.pipeline import run_evaluate

    report = run_evaluate(
        args.checkpoint, args.split, allow_test=args.final_test, use_cache=not args.no_cache
    )
    _print_json(report)
    reload = report.get("reload_vs_training")
    return 0 if reload is None or reload["ok"] else 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import run_calibrate

    res = run_calibrate(args.checkpoint, args.split, use_cache=not args.no_cache, dataset=args.dataset)
    before, after = res["in_sample_metrics"]["before"], res["in_sample_metrics"]["after"]
    _print_json(
        {
            "path": res["path"],
            "temperatures": res["temperatures"],
            "calibrated": res["calibrated"],
            "fits": res["fits"],
            "in_sample_before": _metric_summary(before),
            "in_sample_after": _metric_summary(after),
        }
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .metrics import compare_predictions

    def rows(p: Path):
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    res = compare_predictions(
        rows(args.a),
        rows(args.b),
        reps=args.reps,
        seed=args.seed,
        allow_different_inputs=args.allow_different_inputs,
    )
    res = {"a": str(args.a), "b": str(args.b), **res}
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"{args.out} ya existe")
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_json({"point": res["point"], "intervals": res["bootstrap"]["intervals"]})
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .api import run_serve

    run_serve(args.config)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    report = run_benchmark(
        args.config, args.dataset, split=args.split, requests=args.requests, warmup=args.warmup, out=args.out
    )
    _print_json(
        {k: v for k, v in report.items() if k != "server_ready"} | {"server_ready": report["server_ready"]}
    )
    return 0


def _metric_summary(m: dict) -> dict:
    out = {"questions": m["questions"], "mean_nll_all": m["mean_nll_all"]}
    for p in ("noul", "choice", "score"):
        if p in m:
            out[p] = {k: m[p][k] for k in ("n", "nll", "accuracy") if k in m[p]}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gso", description="Gemma System One")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Snapshot fijado del checkpoint y manifiesto")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--verify-hash", action="store_true", help="Recalcular sha256 aunque esté en caché")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("doctor", help="Compatibilidad y recursos en este equipo")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skip-model", action="store_true", help="No cargar pesos (sólo entorno y MPS)")
    p.add_argument("--json", type=Path, default=None, help="Ruta del informe JSON")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("generate-data", help="Dataset sintético de humo por reglas (Noul)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--questions-per-case", type=int, default=3)
    p.add_argument(
        "--kind",
        choices=["noul", "mixed", "vision"],
        default="noul",
        help="noul: fase 1; mixed: Noul/Choice/Score; vision: paneles con imagen (fase 4)",
    )
    p.add_argument(
        "--variant", choices=["main", "transfer"], default="main", help="mixed/vision: plantillas reservadas"
    )
    p.add_argument("--generator-version", choices=["v1", "v2", "v3", "v4"], default="v3", help="Sólo mixed")
    p.add_argument(
        "--vision-version", choices=["v1", "v2"], default="v2", help="Sólo vision; v2: estilos por partición"
    )
    p.add_argument("--split-seed", type=int, default=0, help="Sólo vision v2: semilla del split planificado")
    p.set_defaults(func=_cmd_generate_data)

    p = sub.add_parser("validate-data", help="Errores de esquema, archivos y grupos")
    p.add_argument("--dataset", type=Path, required=True)
    p.set_defaults(func=_cmd_validate_data)

    p = sub.add_parser("split", help="Manifiesto inmutable de cuatro particiones por grupo")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("baselines", help="Prior y bolsa de palabras en validación (sin Gemma)")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=_cmd_baselines)

    p = sub.add_parser(
        "train", help="Cabezales (fases 1–2) o LoRA + cabezales (fase 3); checkpoint y registro"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None, help="Fase 3: directorio del run a reanudar")
    p.add_argument("--stop-after-steps", type=int, default=None, help="Fase 3: parar tras N pasos globales")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("evaluate", help="Recarga el checkpoint y evalúa desde el texto")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "calibration", "test", "all"], required=True)
    p.add_argument("--final-test", action="store_true", help="Autoriza leer test (sólo medición final)")
    p.add_argument("--no-cache", action="store_true", help="Recalcular representaciones con el backbone")
    p.add_argument(
        "--dataset", type=Path, default=None, help="Dataset externo de diagnóstico (con --split all)"
    )
    p.add_argument(
        "--robustness", action="store_true", help="Pruebas de renombrado, rúbrica invertida y distractor"
    )
    p.add_argument("--calibration", type=Path, default=None, help="Artefacto de gso calibrate vinculado")
    p.add_argument(
        "--baselines", action="store_true", help="Prior y BoW (ajustados en train) en esta partición"
    )
    p.add_argument(
        "--vision-ablation", action="store_true", help="Misma pregunta con la imagen omitida o intercambiada"
    )
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("calibrate", help="Temperatura por primitiva sobre la partición calibration")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["calibration", "all"], required=True)
    p.add_argument("--no-cache", action="store_true", help="Recalcular representaciones (fase 2)")
    p.add_argument(
        "--dataset", type=Path, default=None, help="Conjunto de calibración externo (con --split all)"
    )
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("compare", help="Diferencia emparejada b − a entre dos ficheros de predicciones")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--allow-different-inputs",
        action="store_true",
        help="Control deliberado (p. ej. sin imagen): mismas preguntas y etiquetas, entradas distintas",
    )
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("serve", help="API local con el modelo real (un proceso, un worker, cola acotada)")
    p.add_argument("--config", type=Path, required=True, help="Configuración kind: serve")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("benchmark", help="Arranque, warmup y latencia de la API real en otro proceso")
    p.add_argument("--config", type=Path, required=True, help="Configuración kind: serve")
    p.add_argument("--dataset", type=Path, required=True, help="Batería: casos de una partición no reservada")
    p.add_argument("--split", choices=["validation", "calibration"], default="validation")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
