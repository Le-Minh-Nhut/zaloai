#!/usr/bin/env python3
"""EDA runner for the LifeMate dataset.

    python3 scripts/run_eda.py --data-root ./data --split train --output outputs/eda

Writes ``summary.json``, ``report.md``, ``charts/*.png`` and ``tables/*.csv``.
No network, no GPU, no downloads. Matplotlib/numpy are optional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lifemate.eda import build_analysis_bundle, build_report  # noqa: E402
from lifemate.errors import DatasetNotFoundError, LifeMateError  # noqa: E402

#: Reference/hypothesis aliases tolerated in a prediction JSONL row.
_REF_KEYS = ("reference", "ref")
_HYP_KEYS = ("hypothesis", "hyp", "text")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for LifeMate (stdlib-only core)."
    )
    parser.add_argument("--data-root", default="./data", help="dataset root directory")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="split name(s); comma-separated and/or repeated "
        "(e.g. --split train,public_test or --split train --split public_test). Default: train",
    )
    parser.add_argument("--output", default="outputs/eda", help="output directory")
    parser.add_argument("--config", default=None, help="optional YAML config path")
    parser.add_argument(
        "--tokenizer", default=None, help="optional Hugging Face tokenizer id (offline-safe)"
    )
    parser.add_argument("--no-charts", action="store_true", help="skip chart generation")
    parser.add_argument(
        "--synthetic-ok",
        action="store_true",
        help="allow a fixture root; synthetic paths are detected automatically",
    )
    parser.add_argument("--limit", type=int, default=None, help="max samples per split")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (with --limit)")
    parser.add_argument(
        "--asr-predictions",
        default=None,
        metavar="FILE",
        help="JSONL predictions {id, reference, hypothesis, latency_s}; enables the ASR section",
    )
    return parser.parse_args(argv)


def _is_synthetic(root: Path) -> bool:
    """True when the data root lives under a ``tests/fixtures`` tree."""
    parts = [p for p in root.resolve().parts if p]
    for i in range(len(parts) - 1):
        if parts[i] == "tests" and parts[i + 1] == "fixtures":
            return True
    return (root / "synthetic_manifest.json").is_file() or (root / "SYNTHETIC.md").is_file()


def _load_tokenizer(name: str | None):
    """Load an HF tokenizer offline-first; ``None`` when unavailable."""
    if not name:
        return None
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - report, never crash the run
        print(
            f"warning: tokenizer {name!r} unavailable ({type(exc).__name__}: {exc}); "
            "token statistics will be skipped",
            file=sys.stderr,
        )
        return None


def _pick(row: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None

def _load_asr_predictions(path: Path) -> dict[str, Any]:
    """Parse a prediction JSONL; malformed rows are counted, never fatal."""
    pairs: list[tuple[str, str, str]] = []
    latencies: list[float] = []
    skipped: list[str] = []
    backend: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped.append(f"line {lineno}: invalid JSON")
                continue
            if not isinstance(row, dict):
                skipped.append(f"line {lineno}: row is not an object")
                continue
            ref = _pick(row, _REF_KEYS)
            hyp = _pick(row, _HYP_KEYS)
            if ref is None or hyp is None:
                skipped.append(f"line {lineno}: missing reference/hypothesis")
                continue
            uid = str(row.get("id") or f"row{lineno}")
            pairs.append((uid, str(ref), str(hyp)))
            latency = row.get("latency_s")
            if isinstance(latency, (int, float)) and not isinstance(latency, bool):
                latencies.append(float(latency))
            if backend is None and row.get("backend"):
                backend = str(row["backend"])
    return {
        "pairs": pairs,
        "latencies": latencies,
        "skipped": skipped,
        "backend": backend,
    }

def _build_asr_block(args: argparse.Namespace, samples: list[Any]) -> tuple[dict[str, Any], int]:
    """ASR section for the bundle plus an exit-code bump (2 = predictions unusable)."""
    from lifemate.asr.metrics import critical_token_diagnostics, evaluate_asr  # noqa: PLC0415

    if not args.asr_predictions:
        return (
            {
                "available": False,
                "reason": "no ASR predictions supplied to run_eda.py",
                "backend": None,
                "n": 0,
            },
            0,
        )

    path = Path(args.asr_predictions).expanduser()
    if not path.is_file():
        print(
            f"error: --asr-predictions file not found: {path} "
            "(ASR section stays unavailable; nothing was zero-filled)",
            file=sys.stderr,
        )
        return (
            {
                "available": False,
                "reason": f"ASR predictions file not found: {path}",
                "backend": None,
                "n": 0,
            },
            2,
        )

    loaded = _load_asr_predictions(path)
    for message in loaded["skipped"]:
        print(f"warning: skipped malformed ASR prediction row ({message})", file=sys.stderr)
    print(
        f"warning: ASR references were read from {path} ({len(loaded['pairs'])} row(s), "
        f"{len(loaded['skipped'])} skipped). File-supplied references are only meaningful for "
        "labeled splits; these numbers are not a benchmark result.",
        file=sys.stderr,
    )

    pairs = loaded["pairs"]
    sample_ids = {str(getattr(sample, "id", "")) for sample in samples}
    predicted_ids = {uid for uid, _, _ in pairs}
    n_missing = len(sample_ids - predicted_ids)
    n_extra = len(predicted_ids - sample_ids)
    if n_missing or n_extra:
        print(
            f"warning: ASR predictions do not match the split: {n_missing} sample(s) without "
            f"a prediction, {n_extra} prediction(s) for unknown ids",
            file=sys.stderr,
        )

    backend = loaded["backend"] or "file-supplied"
    if not pairs:
        return (
            {
                "available": False,
                "reason": f"no usable rows in {path}",
                "backend": backend,
                "n": 0,
            },
            2,
        )

    block = dict(evaluate_asr(pairs, backend, latencies=loaded["latencies"]))
    block["critical_tokens"] = critical_token_diagnostics(pairs)
    block["source"] = "file"
    block["predictions_file"] = str(path)
    block["n_skipped_rows"] = len(loaded["skipped"])
    block["n_missing"] = n_missing
    block["n_extra"] = n_extra
    return block, 0

def _print_summary(result: dict, split_bundles: dict[str, dict]) -> None:
    print(f"output: {result['out_dir']}")
    print(f"report: {result['report']}")
    print(f"summary: {result['summary']}")
    print(f"charts: {result['charts_note']}")
    print(f"tables: {len(result['tables'])} csv file(s)")
    for split, bundle in split_bundles.items():
        ds = bundle.get("dataset") or {}
        audio = bundle.get("audio") or {}
        traj = bundle.get("trajectory") or {}
        tools = bundle.get("tools") or {}
        calls = (traj.get("n_gold_calls") or {}).get("stats") or {}
        print(
            f"[{split}] samples={ds.get('n_samples', 0)} "
            f"audio_ok={audio.get('n_files_ok', 0)}/{audio.get('n_files_analyzed', 0)} "
            f"tools={tools.get('n_tools', 0) if tools.get('available') else 'n/a'} "
            f"calls_mean={calls.get('mean')} "
            f"dup_ids={ds.get('n_duplicate_ids', 0)} "
            f"malformed={(ds.get('malformed_records') or {}).get('n_malformed', 0)} "
            f"asr_n={(bundle.get('asr') or {}).get('n', 0) if (bundle.get('asr') or {}).get('available') else 'n/a'}"
        )
    print(f"synthetic: {result['synthetic']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).expanduser()
    # ``--split`` is repeatable; each value may itself be comma-separated.
    splits = [
        part.strip()
        for value in (args.split or ["train"])
        for part in value.split(",")
        if part.strip()
    ]
    out_root = Path(args.output)

    synthetic = _is_synthetic(data_root)
    if synthetic and not args.synthetic_ok:
        print(
            "note: data root looks like a synthetic fixture tree; "
            "continuing (banner will mark the report as SYNTHETIC TEST DATA)",
            file=sys.stderr,
        )

    try:
        from lifemate.config import load_config  # noqa: PLC0415

        overrides = {"data.root": str(data_root)}
        cfg = load_config(args.config, overrides=overrides)
    except LifeMateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - config problems are user-facing
        print(f"error: configuration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    tokenizer_id = args.tokenizer or getattr(cfg.model, "tokenizer_name", None)
    tokenizer = _load_tokenizer(tokenizer_id)

    from lifemate.data.discovery import discover_splits  # noqa: PLC0415
    from lifemate.data.loader import load_catalog, load_raw_split, load_samples  # noqa: PLC0415

    discovered = discover_splits(data_root)
    if not discovered:
        err = DatasetNotFoundError(splits[0] if splits else "train", str(data_root / (splits[0] if splits else "train")))
        print(str(err), file=sys.stderr)
        return 2

    split_bundles: dict[str, dict] = {}
    results: dict[str, dict] = {}
    exit_code = 0
    out_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        try:
            samples = load_samples(data_root, split, limit=args.limit)
        except DatasetNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            exit_code = max(exit_code, 2)
            continue
        except LifeMateError as exc:
            print(f"error: [{split}] {exc}", file=sys.stderr)
            exit_code = max(exit_code, 1)
            continue

        raw_records = load_raw_split(data_root, split, limit=args.limit)
        try:
            catalog = load_catalog(data_root, split)
        except LifeMateError as exc:
            print(f"warning: [{split}] catalog failed to load: {exc}", file=sys.stderr)
            catalog = None

        bundle = build_analysis_bundle(
            samples,
            raw_records=raw_records,
            catalog=catalog,
            tokenizer=tokenizer,
            synthetic=synthetic,
        )
        bundle["asr"], asr_exit = _build_asr_block(args, samples)
        exit_code = max(exit_code, asr_exit)
        bundle["meta_config"] = {
            "tokenizer_requested": tokenizer_id,
            "tokenizer_loaded": tokenizer is not None,
            "seed": args.seed,
            "limit": args.limit,
            "splits_requested": splits,
            "splits_available": sorted(discovered),
        }

        split_out = out_root if len(splits) == 1 else out_root / split
        if args.no_charts:
            bundle["_charts_disabled"] = True
        result = build_report(
            bundle,
            split=split,
            data_root=str(data_root),
            synthetic=synthetic,
            out_dir=split_out,
            charts_enabled=not args.no_charts,
        )
        split_bundles[split] = bundle
        results[split] = result
        print(f"wrote artifacts for split '{split}' to {split_out}")

    if split_bundles:
        last_split = next(reversed(split_bundles))
        last_out = out_root if len(splits) == 1 else out_root / last_split
        last_result = results[last_split]
        _print_summary(
            {
                "out_dir": last_result["out_dir"],
                "report": last_result["report"],
                "summary": last_result["summary"],
                "charts_note": last_result["charts_note"],
                "tables": last_result["tables"],
                "synthetic": synthetic,
            },
            split_bundles,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
