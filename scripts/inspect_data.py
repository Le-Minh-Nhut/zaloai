#!/usr/bin/env python3
"""Inspect a LifeMate data root and write machine + human readable reports.

Usage:
    python3 scripts/inspect_data.py --data-root tests/fixtures/synthetic_lifemate
    python3 scripts/inspect_data.py --data-root ./data --split train --split public_test

Exit codes: 0 success, 1 unexpected failure, 2 dataset not found.
Reports deliberately contain counts and field names only: user profile values,
contact names and message bodies are never copied into the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.data.discovery import SplitPaths, available_splits, discover_splits  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.data.validator import validate_split  # noqa: E402
from lifemate.errors import DatasetNotFoundError  # noqa: E402

PRIVATE_KEYS = {"user_profile", "contacts", "message", "messages", "body", "content", "audio_text", "transcript"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Inspect a LifeMate dataset directory.")
    parser.add_argument("--data-root", default="./data", help="dataset root (default: ./data)")
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=None,
        help="split name; repeatable, default: all discovered splits",
    )
    parser.add_argument("--output-dir", default="outputs", help="where to write reports")
    parser.add_argument("--limit", type=int, default=None, help="max samples per split")
    return parser.parse_args(argv)


def _redact(report: dict) -> dict:
    """Drop any key that could carry personal content (defensive)."""
    if isinstance(report, dict):
        return {k: _redact(v) for k, v in report.items() if k not in PRIVATE_KEYS}
    if isinstance(report, list):
        return [_redact(item) for item in report]
    return report


def inspect(data_root: str | Path, splits: list[str] | None, limit: int | None) -> dict:
    """Build the inspection report for the requested (or all) splits."""
    root = Path(data_root)
    discovered = discover_splits(root)
    if not discovered:
        # Nothing to inspect: report the canonical not-found error (exit 2)
        # instead of silently writing an empty report.
        name = ",".join(splits) if splits else "train"
        raise DatasetNotFoundError(name, str(root / name))
    wanted = splits if splits else sorted(discovered)
    reports: dict[str, dict] = {}
    for name in wanted:
        paths: SplitPaths | None = discovered.get(name)
        if paths is None:
            raise DatasetNotFoundError(name, str(root / name))
        catalog = load_catalog(root, name) if paths.catalog is not None else None
        samples = load_samples(root, name, limit=limit) if paths.jsonl is not None else []
        reports[name] = validate_split(paths, samples=samples, catalog=catalog)
    return {
        "data_root": str(root),
        "synthetic_like": any("synthetic" in key for key in _top_level_keys(root)),
        "available_splits": available_splits(root) if root.is_dir() else [],
        "inspected_splits": wanted,
        "splits": _redact(reports),
    }


def _top_level_keys(root: Path) -> set[str]:
    """Keys seen at the top level of the first split's first record (cheap)."""
    for child in sorted(root.iterdir()):
        candidate = child / f"{child.name}.jsonl"
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    first = handle.readline()
                record = json.loads(first) if first.strip() else {}
                return set(record) if isinstance(record, dict) else set()
            except (OSError, json.JSONDecodeError):
                return set()
    return set()


def render_markdown(report: dict) -> str:
    """Human-readable summary of the JSON report."""
    lines = [
        "# LifeMate data inspection",
        "",
        f"- data root: `{report['data_root']}`",
        f"- available splits: {', '.join(report['available_splits']) or '(none)'}",
        f"- inspected splits: {', '.join(report['inspected_splits']) or '(none)'}",
        "",
        "> Privacy: only counts, field names and ids of *problem* rows are reported.",
        "> User profile values, contacts and message bodies are never copied here.",
        "",
    ]
    for name, split in report["splits"].items():
        counts = split["counts"]
        lines += [
            f"## Split `{name}`",
            "",
            f"- jsonl: `{split['jsonl']}`",
            f"- catalog: {('`' + split['catalog'] + '`') if split['catalog'] else '_absent'}",
            f"- samples: {counts['samples']} (labels: {counts['with_label']}, "
            f"audio text: {counts['with_audio_text']}, history: {counts['with_history']}, "
            f"final state: {counts['with_final_state']})",
            f"- catalog: {counts['catalog_tools']} tools across {counts['catalog_apps']} apps",
            f"- environment dir present (never opened): {split['has_environment_dir']}",
            "",
        ]
        if split["categories"]:
            lines.append("- categories: " + ", ".join(f"{k}={v}" for k, v in split["categories"].items()))
            lines.append("")
        if split["field_presence"]:
            lines.append("- field presence: " + ", ".join(f"{k}={v}" for k, v in split["field_presence"].items()))
            lines.append("")
        if split["unknown_top_level_fields"]:
            lines.append(
                "- unknown top-level fields: "
                + ", ".join(f"{k}={v}" for k, v in split["unknown_top_level_fields"].items())
            )
            lines.append("")
        lines.append(f"- errors ({len(split['errors'])}):")
        lines += [f"  - {item}" for item in split["errors"]] or ["  - none"]
        lines.append(f"- warnings ({len(split['warnings'])}):")
        lines += [f"  - {item}" for item in split["warnings"]] or ["  - none"]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        report = inspect(args.data_root, args.splits, args.limit)
    except DatasetNotFoundError as exc:
        print(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - never traceback at the CLI boundary
        print(f"inspection failed: {type(exc).__name__}: {exc}")
        return 1
    out_dir = Path(args.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "data_inspection.json"
        md_path = out_dir / "data_inspection.md"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"inspection failed: cannot write report: {exc}")
        return 1
    total = sum(split["counts"]["samples"] for split in report["splits"].values())
    errors = sum(len(split["errors"]) for split in report["splits"].values())
    print(f"inspected {len(report['splits'])} split(s), {total} sample(s), {errors} error(s)")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
