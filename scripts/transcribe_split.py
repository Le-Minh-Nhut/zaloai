#!/usr/bin/env python3
"""Transcribe one split and score the transcripts against the gold audio text.

    python3 scripts/transcribe_split.py --split train --data-root ./data

Writes ``<output>`` (predictions JSONL) and ``<output>.metrics.json``.
Offline-safe: the backend's own offline error is surfaced, never a download.
No output file is written when the dataset or the backend is unavailable.

Exit codes: ``0`` ok · ``2`` dataset/backend unavailable · ``3`` every
transcription failed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import DatasetNotFoundError, LifeMateError, ProhibitedAccessError  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DATA = 2
EXIT_FAILED = 3

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Transcribe a LifeMate split and write ASR predictions + metrics."
    )
    parser.add_argument("--split", required=True, help="split to transcribe")
    parser.add_argument("--data-root", default=None, help="dataset root (default: config data.root)")
    parser.add_argument(
        "--output",
        default=None,
        help="predictions JSONL (default: outputs/asr/<split>_predictions.jsonl)",
    )
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--limit", type=int, default=None, help="max samples")
    parser.add_argument("--device", default=None, help="backend device (cpu/cuda/auto)")
    parser.add_argument(
        "--quantization",
        choices=("int8", "fp32"),
        default=None,
        help="ONNX precision for the gipformer backend (default: config asr.quantization)",
    )
    parser.add_argument("--model-dir", default=None, help="local directory with the ONNX model files")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing predictions file")
    return parser.parse_args(argv)

def _apply_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    if args.data_root:
        cfg.data.root = args.data_root
    if args.device:
        cfg.asr.device = args.device
    if args.quantization:
        cfg.asr.quantization = args.quantization
    if args.model_dir:
        cfg.asr.model_dir = args.model_dir
    return cfg

def _reference(sample: Any) -> str:
    """Gold transcript; empty when the split carries no labels."""
    return str(getattr(sample, "audio_text", "") or "")

def _transcribe_all(backend: Any, samples: list[Any], split: str) -> list[dict[str, Any]]:
    """One row per sample. Failures keep an empty hypothesis and a reason."""
    from lifemate.asr.normalization import to_llm_text  # noqa: PLC0415
    from lifemate.data.loader import read_sample_audio  # noqa: PLC0415

    backend_name = getattr(backend, "name", "unknown")
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        row: dict[str, Any] = {
            "id": str(getattr(sample, "id", "")),
            "split": split,
            "reference": _reference(sample),
            "hypothesis": "",
            "raw_hypothesis": "",
            "latency_s": 0.0,
            "duration_s": None,
            "error": None,
        }
        try:
            audio_path = read_sample_audio(sample)  # guarded; never read sample.audio_path
        except ProhibitedAccessError:
            raise
        if audio_path is None:
            row["error"] = "audio file missing for sample"
            rows.append(row)
            continue
        try:
            result = backend.transcribe(audio_path, sample_id=row["id"])
        except Exception as exc:  # noqa: BLE001 - a failed row is data, not a crash
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            rows.append(row)
            continue
        # ``text`` is what the agent sees (backends may case-neutralise uppercase-only
        # output); ``raw_text`` is the recognizer string verbatim. Same split as
        # scripts/infer.py::_transcribe, so a scored hypothesis equals the prompt text.
        text = str(getattr(result, "text", "") or "")
        row["raw_hypothesis"] = str(getattr(result, "raw_text", None) or text)
        row["hypothesis"] = to_llm_text(text) if text else ""
        row["latency_s"] = round(float(getattr(result, "latency_s", 0.0) or 0.0), 4)
        duration = getattr(result, "duration_s", None)
        row["duration_s"] = float(duration) if isinstance(duration, (int, float)) else None
        row["error"] = getattr(result, "error", None)
        if row["error"]:
            row["hypothesis"] = ""
            row["raw_hypothesis"] = ""
        rows.append(row)
        print(f"[{index}/{len(samples)}] {row['id']}: {len(row['hypothesis'])} chars")
    return rows

def _rtf(rows: list[dict[str, Any]]) -> float | None:
    """Real-time factor: processing seconds per audio second (lower is better)."""
    total_latency = sum(float(row["latency_s"] or 0.0) for row in rows)
    total_duration = sum(float(row["duration_s"] or 0.0) for row in rows)
    if total_duration <= 0:
        return None
    return total_latency / total_duration

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = _apply_overrides(load_config(args.config), args)
    except LifeMateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA

    output = Path(args.output or f"outputs/asr/{args.split}_predictions.jsonl")
    if output.exists() and not args.overwrite:
        print(f"error: {output} exists; pass --overwrite to replace it", file=sys.stderr)
        return EXIT_USAGE

    from lifemate.asr import create_asr_backend  # noqa: PLC0415
    from lifemate.data.loader import load_samples  # noqa: PLC0415

    data_root = Path(cfg.data.root)
    try:
        samples = load_samples(data_root, args.split, limit=args.limit, with_labels=True)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DATA
    except LifeMateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA
    if not samples:
        print(f"error: split {args.split!r} has no usable samples", file=sys.stderr)
        return EXIT_DATA

    try:
        backend = create_asr_backend(cfg)
    except Exception as exc:  # noqa: BLE001 - AsrError carries the actionable install/download text
        print(f"error: ASR backend unavailable: {exc}", file=sys.stderr)
        print(
            "nothing was written; fix the backend (offline mode never downloads weights).",
            file=sys.stderr,
        )
        return EXIT_DATA
    backend_name = str(getattr(backend, "name", "unknown"))

    started = time.perf_counter()
    try:
        rows = _transcribe_all(backend, samples, args.split)
    except ProhibitedAccessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA
    wall_s = time.perf_counter() - started

    n_failed = sum(1 for row in rows if row["error"])
    if n_failed == len(rows):
        first = next((row["error"] for row in rows if row["error"]), "unknown error")
        print(
            f"error: every transcription failed ({n_failed}/{len(rows)}); first error: {first}",
            file=sys.stderr,
        )
        return EXIT_FAILED

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    good = [row for row in rows if not row["error"]]
    latencies = [float(row["latency_s"] or 0.0) for row in good]
    model_report: dict[str, Any] | None = None
    size_reporter = getattr(backend, "model_size_report", None)
    if callable(size_reporter):
        try:
            model_report = dict(size_reporter())
        except Exception as exc:  # noqa: BLE001 - size accounting is informative only
            model_report = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    print(f"backend: {backend_name}")
    print(f"n={len(rows)} n_failed={n_failed}")
    if latencies:
        print(
            f"latency_s: mean={statistics.fmean(latencies):.3f} "
            f"median={statistics.median(latencies):.3f} wall={wall_s:.1f}"
        )
    rtf = _rtf(good)
    print(f"rtf: {'n/a' if rtf is None else f'{rtf:.3f}'}")
    if model_report and model_report.get("on_disk_mb") is not None:
        print(f"model_on_disk_mb: {model_report['on_disk_mb']}")

    from lifemate.asr.metrics import (  # noqa: PLC0415
        critical_token_diagnostics,
        evaluate_asr,
        format_metrics_table,
    )

    pairs = [(row["id"], row["reference"], row["hypothesis"]) for row in good]
    raw_pairs = [(row["id"], row["reference"], row["raw_hypothesis"]) for row in good]
    metrics = evaluate_asr(pairs, backend_name, latencies=latencies)
    metrics["raw_recognizer_only"] = evaluate_asr(raw_pairs, backend_name, latencies=latencies)
    metrics["critical_tokens"] = critical_token_diagnostics(pairs)
    metrics["run"] = {
        "split": args.split,
        "data_root": str(data_root),
        "n": len(rows),
        "n_failed": n_failed,
        "wall_s": round(wall_s, 3),
        "rtf": rtf,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "quantization": getattr(backend, "quantization", None),
        "model_size": model_report,
    }
    sidecar = output.with_name(output.name + ".metrics.json")
    sidecar.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(format_metrics_table(metrics))
    print()
    print(format_metrics_table(metrics["critical_tokens"]))
    print(f"\npredictions: {output}")
    print(f"metrics: {sidecar}")
    if size_reporter and model_report and model_report.get("on_disk_bytes") == 0:
        print("note: model files were not found on disk; size accounting is empty", file=sys.stderr)
    return EXIT_OK

if __name__ == "__main__":
    raise SystemExit(main())
