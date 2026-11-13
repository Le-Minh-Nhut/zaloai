#!/usr/bin/env python3
"""Build causal SFT datasets (train/val JSONL of decision points).

Usage:
    python3 scripts/prepare_dataset.py --data-root ./data --split train
    python3 scripts/prepare_dataset.py --data-root ./data --split train \
        --tokenizer Qwen/Qwen3.5-4B --max-seq-length 8192
    python3 scripts/prepare_dataset.py --data-root tests/fixtures/synthetic_lifemate \
        --split train --output /tmp/sft_synth

What it writes (``<output>/<split>/``):

* ``train.jsonl`` / ``val.jsonl`` — compact SFT records
  ``{sample_id, step_index, messages, target, meta}``; prompts are already
  rendered, so training never rebuilds them (deterministic by construction);
* ``manifest.json`` — split fingerprint (counts, seed, ratio, id/group hashes);
* ``stats.json`` — decision-point accounting, drop reasons, token-size summary.

Exit codes: 0 ok, 2 dataset not found, 3 optional tokenizer requested but
unavailable (records are still written with char/4 token estimates).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.data.split_builder import split_samples  # noqa: E402
from lifemate.data.trajectory_builder import (  # noqa: E402
    build_sft_dataset,
    to_hf_dataset_records,
    to_jsonl,
)
from lifemate.errors import DatasetNotFoundError, LifeMateError  # noqa: E402

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Build causal SFT decision points from a LifeMate split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="./data", help="dataset root directory")
    parser.add_argument("--split", default="train", help="split to convert")
    parser.add_argument("--output", default="outputs/datasets", help="output root directory")
    parser.add_argument("--config", default=None, help="optional YAML config for defaults")
    parser.add_argument("--val-ratio", type=float, default=None, help="validation fraction")
    parser.add_argument("--seed", type=int, default=None, help="split seed")
    parser.add_argument(
        "--tokenizer",
        default="none",
        help="HF tokenizer id for real token counts; 'none' => char/4 estimate only",
    )
    parser.add_argument("--limit", type=int, default=None, help="max samples to load")
    parser.add_argument("--max-seq-length", type=int, default=None, help="budget for the fit check")
    parser.add_argument("--history-limit", type=int, default=None, help="conversation turns kept in prompts")
    return parser.parse_args(argv)

def _load_tokenizer(spec: str) -> tuple[object | None, str]:
    """Best-effort tokenizer load; returns ``(tokenizer, status)``."""
    if not spec or spec.strip().lower() in {"none", "no", "false", "-"}:
        return None, "deferred to training (no tokenizer requested)"
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 - lazy, optional
    except Exception as exc:
        print(f"warning: --tokenizer {spec} ignored (transformers unavailable: {exc})")
        return None, "deferred to training (transformers unavailable)"
    try:
        return AutoTokenizer.from_pretrained(spec), f"loaded {spec}"
    except Exception as exc:  # pragma: no cover - network/weights dependent
        print(f"warning: --tokenizer {spec} could not be loaded ({exc}); using estimates")
        return None, "deferred to training (tokenizer load failed)"

def _fit_stats(records: list[dict], tokenizer: object | None, max_seq_length: int) -> dict:
    """How the sequence budget treats these examples: truncated vs dropped.

    Two different outcomes must never be conflated (a previous version reported
    only "n_fit", which read as "everything fits" while the collator was
    silently middle-truncating 75% of the records):

    * **truncated** — `kept=True` but the prompt lost its middle; the example is
      still trained on, with the request prefix intact;
    * **dropped** — `kept=False`, counted by exact reason: `target_too_long`
      (a target that alone exceeds the budget cannot be supervised at all) and
      `target_value_truncated` (a `tool_call` whose reference ID/name lived in
      the middle the cut removed; supervising it teaches an identifier the model
      was never shown). 0 of the latter at 8192, 122 at 4096.
    """
    if tokenizer is None:
        # No tokenizer: the char/4 heuristic UNDER-counts Vietnamese by ~2.5x,
        # so its numbers must not be presented as a measured answer.
        estimate = [
            int(record["meta"].get("token_estimate") or 0)
            for record in records
        ]
        over_estimate = sum(1 for value in estimate if value > max_seq_length)
        return {
            "method": "chars4_estimate",
            "caveat": (
                "no tokenizer supplied: counts are chars/4 ESTIMATES, not measurements. "
                "Vietnamese text tokenizes to roughly 2.5x more tokens than chars/4, so "
                "n_truncated_estimate is a LOWER BOUND. Pass --tokenizer for real counts."
            ),
            "max_seq_length": max_seq_length,
            "n_records": len(records),
            "n_truncated_estimate": over_estimate,
            "n_truncated": None,
            "n_dropped": None,
            "n_fit": len(records),  # records written; NOT a claim that none truncates
        }
    from lifemate.training.collator import build_supervised_example  # noqa: PLC0415

    truncated = 0
    dropped: dict[str, int] = {}
    for record in records:
        example = build_supervised_example(
            tokenizer, record["messages"], record["target"], max_seq_length
        )
        if example.get("kept"):
            # build_supervised_example marks a middle cut in `reason`
            if str(example.get("reason") or "").startswith("prompt_middle_truncated_"):
                truncated += 1
        else:
            reason = str(example.get("reason") or "unknown")
            dropped[reason] = dropped.get(reason, 0) + 1
    n_dropped = sum(dropped.values())
    return {
        "method": "tokenizer",
        "caveat": None,
        "max_seq_length": max_seq_length,
        "n_records": len(records),
        "n_truncated": truncated,
        "n_truncated_fraction": round(truncated / len(records), 6) if records else 0.0,
        "n_dropped": n_dropped,
        "n_fit": len(records) - n_dropped,
        "dropped_reasons": dict(sorted(dropped.items())),
    }

def _print_table(stats: dict, n_read: int, n_val: int, n_val_points: int) -> None:
    """Compact human-readable summary of the build (train split, then val)."""
    print("SFT dataset")
    print(f"  samples read        : {n_read} (train {stats['n_samples']}, val {n_val})")
    print(f"  samples kept        : {stats['n_samples_kept']} (train split)")
    print(f"  decision points     : {stats['n_decision_points']}")
    print(f"  dropped             : {stats['n_dropped']}")
    for reason, count in sorted((stats.get("dropped") or {}).items()):
        if count:
            print(f"    - {reason}: {count}")
    if stats.get("n_dropped_decision_points"):
        print(f"  decision points dropped: {stats['n_dropped_decision_points']}")
        for reason, count in sorted((stats.get("dropped_decision_points") or {}).items()):
            if count:
                print(f"    - {reason}: {count}")
    print(f"  visibility limited  : {stats['visibility_limited']}")
    print(f"  unobservable x-ref  : {stats['unobservable_ref']} sample(s) -> affected targets dropped")
    lengths = stats.get("trajectory_length_histogram") or {}
    print("  calls per sample    : " + ", ".join(f"{k}={v}" for k, v in sorted(lengths.items())))
    prompt = stats.get("prompt_tokens_estimate") or {}
    target = stats.get("target_tokens_estimate") or {}
    print(f"  prompt tokens (est) : min {prompt.get('min')} mean {prompt.get('mean')} max {prompt.get('max')}")
    print(f"  target tokens (est) : min {target.get('min')} mean {target.get('mean')} max {target.get('max')}")

def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = load_config(args.config) if args.config else load_config()
    except LifeMateError as exc:
        print(f"invalid config: {exc}")
        return 1

    val_ratio = args.val_ratio if args.val_ratio is not None else float(cfg.training.validation_ratio)
    seed = args.seed if args.seed is not None else int(cfg.training.seed)
    history_limit = args.history_limit if args.history_limit is not None else 10
    max_seq_length = args.max_seq_length or int(cfg.training.max_seq_length)
    limit = args.limit if args.limit is not None else cfg.data.max_samples

    try:
        samples = load_samples(args.data_root, args.split, limit=limit)
        catalog = load_catalog(args.data_root, args.split)
    except DatasetNotFoundError as exc:
        print(str(exc))
        return 2
    except LifeMateError as exc:
        print(f"failed to read split {args.split!r}: {exc}")
        return 1

    if catalog is None:
        print(
            f"warning: split {args.split!r} has no tool catalog; every gold call will be "
            "flagged catalog_missing_tool and nothing will be written"
        )
        catalog = _EmptyCatalog()
    if not samples:
        print(DatasetNotFoundError(args.split, str(Path(args.data_root) / args.split)))
        return 2

    split = split_samples(samples, ratio=val_ratio, seed=seed)
    train_points, train_stats = build_sft_dataset(split.train, catalog, history_limit=history_limit)
    val_points, val_stats = build_sft_dataset(split.val, catalog, history_limit=history_limit)

    tokenizer, tokenizer_status = _load_tokenizer(args.tokenizer)
    train_records = to_hf_dataset_records(train_points)
    val_records = to_hf_dataset_records(val_points)
    fit = _fit_stats(train_records, tokenizer, max_seq_length)
    fit_val = _fit_stats(val_records, tokenizer, max_seq_length)

    out_dir = Path(args.output) / args.split
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        to_jsonl(train_points, out_dir / "train.jsonl")
        to_jsonl(val_points, out_dir / "val.jsonl")
    except OSError as exc:
        print(f"failed to write {out_dir}: {exc}")
        return 1

    manifest = {
        **split.manifest(),
        "split": args.split,
        "data_root": str(Path(args.data_root).resolve()),
        "n_train_decision_points": len(train_points),
        "n_val_decision_points": len(val_points),
        # Points the builder refused to supervise (a gold argument the prompt
        # cannot show). Without these keys the manifest's point counts silently
        # shrink versus ``n_train_decision_points`` of an older build.
        "n_train_dropped_decision_points": train_stats["n_dropped_decision_points"],
        "n_val_dropped_decision_points": val_stats["n_dropped_decision_points"],
        "dropped_decision_points": {
            "train": train_stats["dropped_decision_points"],
            "val": val_stats["dropped_decision_points"],
        },
        "history_limit": history_limit,
        "tokenization": tokenizer_status,
        "fit": fit,
        "fit_val": fit_val,
    }
    stats = {
        "split": args.split,
        "train": train_stats,
        "val": val_stats,
        "tokenization": tokenizer_status,
        "fit": fit,
        "fit_val": fit_val,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    _print_table(train_stats, len(samples), len(split.val), len(val_points))
    print(f"  tokenization        : {tokenizer_status}")
    print(f"  max_seq_length      : {max_seq_length} (effective, from config unless --max-seq-length)")
    if fit["n_truncated"] is None:
        print(
            f"  truncation          : UNMEASURED — {fit['n_truncated_estimate']}/{fit['n_records']} "
            "over budget by the chars/4 estimate (lower bound); pass --tokenizer"
        )
    else:
        print(
            f"  truncation (train)  : {fit['n_truncated']}/{fit['n_records']} "
            f"({fit['n_truncated_fraction']*100:.2f}%) middle-truncated, "
            f"{fit['n_dropped']} dropped"
        )
        print(
            f"  truncation (val)    : {fit_val['n_truncated']}/{fit_val['n_records']} "
            f"({fit_val['n_truncated_fraction']*100:.2f}%) middle-truncated, "
            f"{fit_val['n_dropped']} dropped"
        )
    if train_stats["visibility_limited"]:
        print(
            "warning: "
            f"{train_stats['visibility_limited']} sample(s) are visibility_limited "
            "(missing final tool results) and got no terminal finish example"
        )
    if train_stats["unobservable_ref"]:
        print(
            "warning: "
            f"{train_stats['unobservable_ref']} sample(s) had reference arguments not observable from "
            f"the prompt; {train_stats['n_dropped_decision_points']} affected decision point(s) were "
            "dropped (the rest of each trajectory is kept) — see stats.json"
        )
    if train_stats["n_dropped"]:
        print(f"warning: {train_stats['n_dropped']} sample(s) dropped (see stats.json)")
    print(f"wrote {out_dir}/train.jsonl, val.jsonl, manifest.json, stats.json")
    return 0

class _EmptyCatalog:
    """Placeholder catalog so the pipeline reports rather than crashes."""

    tools: dict = {}
    apps: list = []
    specs: list = []

    def tools_for_apps(self, apps=None, include_meta: bool = True) -> list:
        return []

    def validate_call(self, name: str, args=None) -> dict:
        return {"errors": [f"unknown tool: {name}"], "warnings": []}

    def get(self, name: str):
        return None

    def to_schema_prompt(self, apps=None, include_meta: bool = True) -> str:
        return "Available tools: none"

if __name__ == "__main__":
    sys.exit(main())
