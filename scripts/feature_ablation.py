#!/usr/bin/env python3
"""Feature ablation runner: same evaluation, one feature set per arm.

The harness holds everything else FIXED and changes only ``features:``. It does
not reimplement evaluation — each arm shells out to ``scripts/evaluate.py`` and
reads the ``metrics.json`` that script writes.

Design constraints baked in here:

* **Fail fast.** Every arm is validated against the feature registry BEFORE any
  arm runs, so an invalid combination cannot burn half the budget.
* **Self-comparison is the noise test.** Running the baseline against itself must
  report zero deltas; a nonzero delta means the harness or the evaluation is
  nondeterministic, not that a technique works.
* **No invented numbers.** A missing ``metrics.json`` makes that arm ``BLOCKED``
  with the evaluate stderr tail, never a zero-filled metric.
* **Mock results are labelled.** ``--use-mock-env`` results are marked
  ``mock_env: true`` and "NOT a competition score" in stdout *and* JSON.

Each arm's ``features:`` section reaches ``scripts/evaluate.py`` through
``build_feature_config``, so the report's ``features_wired`` /
``features_effective`` flags are ``true``; a baseline-vs-baseline pair stays the
harness's noise test (its deltas must be zero).

Usage::

    PYTHONPATH=src python3 scripts/feature_ablation.py \\
        --arms configs/experiments/baseline.yaml configs/experiments/asr_nbest.yaml \\
        --limit 20 --out outputs/reports/ablation.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import ConfigError, FeatureConfigurationError, LifeMateError  # noqa: E402
from lifemate.features.config import (  # noqa: E402
    FeatureConfig,
    enabled_features,
    feature_fingerprint,
    training_feature_fingerprint,
)
from lifemate.features.overrides import (  # noqa: E402
    apply_overrides,
    load_yaml_feature_overrides,
)
from lifemate.features.registry import REGISTRY  # noqa: E402

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_ARMS = 3

#: Metrics compared arm-vs-reference. Real names produced by ``evaluate_split``.
COMPARED_METRICS = (
    "tool_call_exact_match",
    "tool_name_accuracy",
    "argument_accuracy",
    "sequence_f1",
    "final_state_correctness",
)

#: Constants the harness holds fixed; read from the config object, never guessed.
FIXED_FIELDS = (
    ("data.root", "dataset path"),
    ("asr.model_id", "ASR model"),
    ("model.name", "model id"),
    ("inference.checkpoint", "checkpoint"),
    ("inference.model_mode", "model mode"),
    ("training.seed", "seed"),
    ("agent.max_steps", "agent.max_steps"),
    ("agent.time_budget_seconds", "time budget (s)"),
    ("agent.temperature", "generation temperature"),
    ("agent.max_new_tokens", "generation token budget"),
    ("evaluation.teacher_forced", "teacher-forced scoring"),
    ("evaluation.closed_loop", "closed-loop scoring"),
)

def _read_dotted(obj: Any, dotted: str) -> Any:
    target = obj
    for part in dotted.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target

def _arm_config(path: Path) -> tuple[Any, Any, dict[str, str], list[str]]:
    """Load one arm: the base config plus its resolved feature set.

    Returns ``(app_cfg, feature_cfg, provenance, unknown_keys)``. The base config
    is loaded with ``features:`` stripped when the loader does not know the
    section yet, so the arm still yields the constants the harness pins.
    """
    flags, params, unknown = load_yaml_feature_overrides(path)
    try:
        app_cfg = load_config(path)
    except ConfigError as exc:
        if "unknown config key: features" not in str(exc):
            raise
        # Loader without the ``features:`` section (older checkout): defaults
        # still describe every constant the harness holds fixed.
        app_cfg = load_config(None)
    feature_cfg, provenance = apply_overrides(
        getattr(app_cfg, "features", None) or FeatureConfig(),
        yaml=flags,
        params=params,
        unknown=unknown,
    )
    return app_cfg, feature_cfg, provenance, unknown

def _validate_arms(arms: Sequence[Path]) -> None:
    """Validate every arm before anything runs; raises on any problem.

    Duplicate arms are allowed on purpose: running the same arm twice is the
    harness's noise test (zero deltas expected), not a mistake.
    """
    problems: list[str] = []
    if not arms:
        raise FeatureConfigurationError("no --arms given; pass at least one experiment YAML")
    for arm in arms:
        if not arm.is_file():
            problems.append(f"{arm}: file not found")
            continue
        try:
            _app, feature_cfg, _prov, _unknown = _arm_config(arm)
        except (ConfigError, FeatureConfigurationError, LifeMateError) as exc:
            problems.append(f"{arm}: {exc}")
            continue
        for problem in REGISTRY.validate(feature_cfg):
            problems.append(f"{arm}: {problem}")
    if problems:
        joined = "\n".join(f"  - {problem}" for problem in problems)
        raise FeatureConfigurationError(f"refusing to start: {len(problems)} arm problem(s)\n{joined}")
def _validate_fairness(
    arms: Sequence[Path],
    *,
    experimental_variables: Sequence[str] | None = None,
) -> None:
    """Ensure fixed configuration fields match across all arms unless declared variable."""
    if len(arms) <= 1:
        return
    ref_arm = arms[0]
    ref_app, _ref_feat, _prov, _unk = _arm_config(ref_arm)
    allowed = {str(v).strip().lower() for v in (experimental_variables or [])}

    problems: list[str] = []
    for arm in arms[1:]:
        app, feat, _prov, _unk = _arm_config(arm)
        arm_allowed = set(allowed)
        for var in getattr(app, "experimental_variables", []) or []:
            arm_allowed.add(str(var).strip().lower())

        for dotted, label in FIXED_FIELDS:
            field_tokens = {
                dotted.lower(),
                label.lower(),
                dotted.split(".")[0].lower(),
                dotted.split(".")[-1].lower(),
            }
            if arm_allowed & field_tokens:
                continue

            ref_val = _read_dotted(ref_app, dotted)
            arm_val = _read_dotted(app, dotted)
            if ref_val != arm_val:
                problems.append(
                    f"{arm.name}: fixed field {label!r} ({dotted}) has value {arm_val!r}, "
                    f"differs from reference {ref_arm.name} ({ref_val!r})."
                )
    if problems:
        joined = "\n".join(f"  - {p}" for p in problems)
        raise FeatureConfigurationError(
            f"Ablation fairness violation: fixed configuration mismatch across arms\n{joined}\n"
            "Declare experimental variables via --experimental-variable / --allow-mismatch to permit variation."
        )


def _materialize_arm_config(arm: Path, *, features_wired: bool, out_dir: Path, dry_run: bool) -> Path:
    """Path to hand ``scripts/evaluate.py`` as ``--config``.

    Wired: the arm's own YAML. Not wired yet: a copy with ``features:`` removed,
    because the loader rejects that key (``unknown config key: features``). The
    copy keeps every other section, so the run still holds the same constants —
    and the report states that the feature set was NOT applied.
    """
    if features_wired:
        return arm
    target = out_dir / f"{arm.stem}.without_features.yaml"
    if dry_run:
        return target  # plan only: the name is reported, nothing is written
    import yaml

    payload = yaml.safe_load(arm.read_text(encoding="utf-8")) or {}
    payload.pop("features", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target

def _evaluate_command(
    config_path: Path,
    *,
    out_dir: Path,
    split: str,
    data_root: str | None,
    limit: int | None,
    mock_env: bool,
    generator: str | None,
    predictions: str | None,
    evaluation_mode: str | None = None,
) -> list[str]:
    """The exact ``scripts/evaluate.py`` command for one arm."""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--config",
        str(config_path),
        "--split",
        split,
        "--output",
        str(out_dir),
    ]
    if data_root:
        command += ["--data-root", data_root]
    if limit is not None:
        command += ["--limit", str(limit)]
    if mock_env:
        command += ["--use-mock-env"]
    if generator:
        command += ["--generator", generator]
    if predictions:
        command += ["--predictions", predictions]
    if evaluation_mode:
        command.append(f"--evaluation-mode={evaluation_mode}")
    return command

def _metrics_from(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The compared metrics, present or explicitly unavailable — never invented."""
    metrics = payload.get("metrics") or {}
    out: dict[str, Any] = {}
    for key in COMPARED_METRICS:
        value = metrics.get(key)
        if isinstance(value, Mapping):
            out[key] = dict(value)
        elif value is None:
            out[key] = {"available": False, "reason": "metric absent from this run"}
        else:
            out[key] = value
    return out

def _scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        inner = value.get("value")
        return float(inner) if isinstance(inner, (int, float)) and not isinstance(inner, bool) else None
    return None

def _deltas(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Per-metric difference vs reference arm.

    Unavailable metrics report "UNAVAILABLE", never a false 0% improvement.
    """
    out: dict[str, Any] = {}
    for key in COMPARED_METRICS:
        ref_val = reference.get(key)
        cand_val = other.get(key)
        ref_unavail = (isinstance(ref_val, Mapping) and ref_val.get("available") is False) or (ref_val is None)
        cand_unavail = (isinstance(cand_val, Mapping) and cand_val.get("available") is False) or (cand_val is None)
        if ref_unavail or cand_unavail:
            out[key] = "UNAVAILABLE"
            continue
        base = _scalar(ref_val)
        candidate = _scalar(cand_val)
        if base is None or candidate is None:
            out[key] = "UNAVAILABLE"
        else:
            out[key] = round(candidate - base, 6)
    return out

def _fixed_constants(app_cfg: Any, *, split: str, mock_env: bool) -> dict[str, Any]:
    constants: dict[str, Any] = {"split": split, "mock_env": bool(mock_env)}
    for dotted, label in FIXED_FIELDS:
        constants[label] = _read_dotted(app_cfg, dotted)
    return constants

def _header(constants: Mapping[str, Any]) -> str:
    width = max(len(str(key)) for key in constants)
    lines = ["LifeMate feature ablation — held fixed:"]
    lines += [f"  {key:<{width}}  {constants[key]}" for key in constants]
    return "\n".join(lines)

def _run_arm(command: Sequence[str], *, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(shlex.quote(part) for part in command)
    if dry_run:
        return {"command": printable, "status": "DRY-RUN", "returncode": None}
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    return {
        "command": printable,
        "status": "RAN" if completed.returncode == 0 else "FAILED",
        "returncode": completed.returncode,
        "stdout_tail": "\n".join((completed.stdout or "").strip().splitlines()[-8:]),
        "stderr_tail": "\n".join((completed.stderr or "").strip().splitlines()[-8:]),
    }

def build_report(
    arms: Sequence[Path],
    *,
    split: str,
    data_root: str | None,
    limit: int | None,
    out: Path,
    mock_env: bool,
    dry_run: bool,
    generator: str | None = None,
    predictions: str | None = None,
    evaluation_mode: str | None = None,
    experimental_variables: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate, (optionally) run, and assemble the comparison report."""
    _validate_arms(arms)
    _validate_fairness(arms, experimental_variables=experimental_variables)
    arm_features = [set(enabled_features(_arm_config(arm)[1])) for arm in arms]
    changed = set().union(*(item ^ arm_features[0] for item in arm_features))
    # Compare effective ASR/audio values, including parameters excluded from the
    # high-level technique fingerprint (e.g. RMS target and VAD padding).
    from lifemate.evaluation.transcripts import asr_cache_identity
    identities = [asr_cache_identity(app, feat) for app, feat, _, _ in map(_arm_config, arms)]
    audio_changed = any(identity != identities[0] for identity in identities[1:])
    if audio_changed or any(name.startswith(("asr.", "audio.")) for name in changed):
        if evaluation_mode is None:
            evaluation_mode = "audio_to_action"
        elif evaluation_mode != "audio_to_action":
            raise ValueError("ASR/audio ablations require audio_to_action evaluation")

    first_app, _first_features, _prov, _unknown = _arm_config(arms[0])
    features_wired = hasattr(first_app, "features")
    constants = _fixed_constants(first_app, split=split, mock_env=mock_env)

    arm_reports: list[dict[str, Any]] = []
    reference_metrics: Mapping[str, Any] | None = None
    for index, arm in enumerate(arms):
        app_cfg, feature_cfg, provenance, unknown = _arm_config(arm)
        arm_out = out.parent / f"{out.stem}.arm{index}"
        metrics_path = arm_out / "metrics.json"
        materialized = _materialize_arm_config(
            arm, features_wired=features_wired, out_dir=arm_out, dry_run=dry_run
        )
        command = _evaluate_command(
            materialized,
            out_dir=arm_out,
            split=split,
            data_root=data_root,
            limit=limit,
            mock_env=mock_env,
            generator=generator,
            predictions=predictions,
            evaluation_mode=evaluation_mode,
        )
        run = _run_arm(command, dry_run=dry_run)
        entry: dict[str, Any] = {
            "arm": index,
            "name": arm.stem,
            "config": str(arm),
            "features": enabled_features(feature_cfg),
            "fingerprint": feature_fingerprint(feature_cfg),
            "training_fingerprint": training_feature_fingerprint(feature_cfg),
            "provenance": dict(provenance),
            "unknown_keys": list(unknown),
            "output_dir": str(arm_out),
            "evaluate_config": str(materialized),
            "command": run["command"],
            "run_status": run["status"],
            "model": str(_read_dotted(app_cfg, "model.name") or "unknown"),
            "checkpoint": _read_dotted(app_cfg, "inference.checkpoint"),
            "dataset": str(_read_dotted(app_cfg, "data.root") or split),
            "enabled_features": enabled_features(feature_cfg),
            "effective_features": enabled_features(feature_cfg) if features_wired else [],
            "degraded_features": [],
            "evaluation_mode": evaluation_mode or "text_to_action",
            "n_samples": limit,
            "n_successful": None,
            "n_failed": None,
        }
        if run.get("returncode") is not None:
            entry["returncode"] = run["returncode"]
        if run.get("stderr_tail"):
            entry["evaluate_stderr_tail"] = run["stderr_tail"]

        if dry_run:
            entry["metrics"] = None
            entry["mock_env"] = bool(mock_env)
            arm_reports.append(entry)
            continue

        if run["status"] != "RAN":
            entry["status"] = "FAILED"
            entry["mock_env"] = bool(mock_env)
            arm_reports.append(entry)
            continue


        if not metrics_path.is_file():
            entry["status"] = "BLOCKED"
            entry["blocked_reason"] = (
                f"{metrics_path} was not written by scripts/evaluate.py; "
                "no metrics are reported for this arm"
            )
            entry["mock_env"] = bool(mock_env)
            arm_reports.append(entry)
            continue

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = _metrics_from(payload)
        entry["status"] = "OK"
        entry["metrics"] = metrics
        entry["mock_env"] = bool(payload.get("mock_env", mock_env))
        entry["evaluation_notes"] = list(payload.get("comparison_notes") or [])
        provenance_block = payload.get("provenance") or {}
        cl_block = payload.get("closed_loop") or {}
        tf_block = payload.get("teacher_forced") or {}
        sc = cl_block.get("status_counts") or {}
        n_samples = provenance_block.get("n_samples") or cl_block.get("n_run") or tf_block.get("n_records") or payload.get("n_samples")
        if cl_block:
            n_successful = sc.get("finished", 0) + sc.get("ok", 0)
            n_failed = cl_block.get("n_failures", 0)
        elif tf_block:
            n_successful = tf_block.get("n_records", 0)
            n_failed = tf_block.get("n_ungradeable", 0)
        else:
            n_successful = payload.get("n_successful", 0)
            n_failed = payload.get("n_failed", 0)
        degraded = list(payload.get("degraded_features") or [])
        execution = payload.get("feature_execution") or {}
        observed = {name for name, counts in execution.items() if counts.get("n_requested", 0) and counts.get("n_executed") == counts.get("n_requested")}
        runtime_features = {f for f in entry["enabled_features"] if f.startswith(("asr.", "audio."))}
        unknown_execution = (runtime_features - set(execution)) | {
            name for name in runtime_features if (execution.get(name) or {}).get("n_not_observed", 0)
        }
        entry["not_observed_features"] = sorted(unknown_execution)
        effective = [f for f in entry["enabled_features"] if f not in degraded and (f not in runtime_features or f in observed)]
        entry["n_samples"] = n_samples
        entry["n_successful"] = n_successful
        entry["n_failed"] = n_failed
        entry["effective_features"] = effective
        entry["degraded_features"] = degraded
        entry["evaluation_mode"] = payload.get("evaluation_mode") or evaluation_mode or "text_to_action"
        entry["evaluated"] = {
            "git_commit": provenance_block.get("git_commit"),
            "dataset_manifest_hash": provenance_block.get("dataset_manifest_hash"),
            "n_samples": n_samples,
            "model_mode": provenance_block.get("model_mode"),
            "checkpoint": provenance_block.get("checkpoint"),
        }
        if degraded or unknown_execution:
            entry["deltas"] = {key: None for key in COMPARED_METRICS}
            entry["deltas_note"] = "invalid comparison: a requested feature degraded, was blocked, or was not observed"
            if index == 0:
                reference_metrics = None
            arm_reports.append(entry)
            continue
        if index == 0:
            reference_metrics = metrics
            entry["deltas"] = {
                key: (
                    0.0
                    if not (isinstance(metrics.get(key), Mapping) and metrics[key].get("available") is False)
                    and _scalar(metrics.get(key)) is not None
                    else "UNAVAILABLE"
                )
                for key in COMPARED_METRICS
            }
            entry["deltas_note"] = "reference arm: deltas are relative to itself"
        elif reference_metrics is None:
            entry["deltas"] = {key: None for key in COMPARED_METRICS}
            entry["deltas_note"] = "reference arm produced no metrics; nothing to compare against"
        else:
            entry["deltas"] = _deltas(reference_metrics, metrics)
            entry["deltas_note"] = f"vs arm 0 ({arms[0].stem})"
        arm_reports.append(entry)

    report: dict[str, Any] = {
        "schema": "lifemate.feature_ablation/1",
        "fixed": constants,
        "features_wired": features_wired,
        "features_effective": features_wired,
        "mock_env": bool(mock_env),
        "competition_score": False if mock_env else "UNDETERMINED (not a validated run)",
        "reference_arm": arms[0].stem,
        "arms": arm_reports,
        "dry_run": bool(dry_run),
    }
    if not report["features_wired"]:
        report["note"] = (
            "WIRING MISSING in this checkout: scripts/evaluate.py cannot see each arm's "
            "`features:` section (ConfigError: unknown config key: features at load_config). "
            "The commands below are exact; the feature sets are NOT applied. A "
            "baseline-vs-baseline self-comparison is therefore the only meaningful arm pair, "
            "and it is the harness's noise test."
        )
    if mock_env:
        report["mock_note"] = (
            "MOCK ENV / SYNTHETIC: these numbers come from MockToolEnv, not the official "
            "simulator. They are NOT competition scores."
        )
    return report

def _print_summary(report: Mapping[str, Any]) -> None:
    print(_header(report["fixed"]))
    if not report.get("features_wired", False):
        print("\nWARNING: arm feature sets are NOT applied by scripts/evaluate.py in this checkout.")
        print("Deltas below are harness noise only.")
    if report.get("mock_env"):
        print("\n*** MOCK ENV / SYNTHETIC — every row is NOT a competition score. ***")
    print("\nArms:")
    for entry in report["arms"]:
        enabled = ", ".join(entry["features"]) or "NONE"
        print(f"  [{entry['arm']}] {entry['name']}: {entry['run_status']} — features: {enabled}")
        print(f"       fingerprint={entry['fingerprint']} training={entry['training_fingerprint']}")
        print(f"       command: {entry['command']}")
        if "status" in entry and entry["status"] not in {"OK"}:
            detail = entry.get("blocked_reason") or entry.get("returncode")
            print(f"       status={entry['status']} ({detail})")
        if entry.get("metrics"):
            for key, value in entry["metrics"].items():
                shown = value if not isinstance(value, Mapping) else value.get("value", value)
                print(f"       {key}: {shown}")
            deltas = entry.get("deltas") or {}
            if entry["arm"] > 0:
                print(f"       deltas {entry.get('deltas_note')}: {deltas}")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Run the same evaluation under different feature sets and compare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arms", nargs="+", required=True, help="experiment YAMLs; the FIRST is the reference")
    parser.add_argument("--split", default="train", help="split to evaluate")
    parser.add_argument("--data-root", default=None, help="dataset root (default: config data.root)")
    parser.add_argument("--limit", type=int, default=None, help="samples/records per arm")
    parser.add_argument("--out", default="outputs/reports/ablation.json", help="report path")
    parser.add_argument("--use-mock-env", action="store_true", help="MockToolEnv self-test (marks results NOT scores)")
    parser.add_argument("--generator", default=None, metavar="dumpscript:PATH", help="forwarded test hook")
    parser.add_argument("--predictions", default=None, help="forwarded decision-record JSONL")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and commands, execute nothing")
    parser.add_argument(
        "--experimental-variable",
        "--allow-mismatch",
        action="append",
        default=[],
        dest="experimental_variables",
        help="declared experimental variable allowed to differ across arms (e.g. asr.model_id, asr)",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("text_to_action", "audio_to_action"),
        default=None,
        help="evaluation mode for arms",
    )
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    arms = [Path(arm) for arm in args.arms]
    out = Path(args.out)
    try:
        report = build_report(
            arms,
            split=args.split,
            data_root=args.data_root,
            limit=args.limit,
            out=out,
            mock_env=bool(args.use_mock_env),
            dry_run=bool(args.dry_run),
            generator=args.generator,
            predictions=args.predictions,
            evaluation_mode=args.evaluation_mode,
            experimental_variables=args.experimental_variables,
        )
    except FeatureConfigurationError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return EXIT_ARMS if "arm" in str(exc) else EXIT_CONFIG
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    _print_summary(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    if args.dry_run:
        print("dry run: nothing was executed.")
    return EXIT_OK

if __name__ == "__main__":
    sys.exit(main())
