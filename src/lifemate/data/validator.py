"""Split-level validation report: counts, gaps, and gold-call problems.

Nothing is raised: the report separates ``errors`` (breaks the benchmark
contract) from ``warnings`` (shape we do not recognise — expected while the
real dataset is absent). Privacy: report content never includes user profile
values, contact details, or message bodies.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import SchemaError
from .catalog import ToolCatalog
from .discovery import SplitPaths
from .loader import iter_raw_records
from .schemas import Context, Label, Sample, _as_dict

__all__ = ["parse_record_safely", "validate_split", "KNOWN_TOP_LEVEL_FIELDS"]

KNOWN_TOP_LEVEL_FIELDS = frozenset(
    {
        "id",
        "sample_id",
        "uid",
        "utt_id",
        "name",
        "audio_path",
        "audio",
        "audio_file",
        "wav",
        "wav_path",
        "audio_filename",
        "path",
        "audio_text",
        "transcript",
        "text",
        "asr_text",
        "utterance",
        "sentence",
        "context",
        "world",
        "state",
        "metadata",
        "installed_apps",
        "apps",
        "history",
        "conversation",
        "messages",
        "turns",
        "dialog",
        "label",
        "labels",
        "gold",
        "target",
        "annotation",
        "output",
        "tool_calls",
        "tool_results",
        "final_state",
        "category",
        "synthetic",
        "_synthetic",
    }
)
CONTEXT_FIELDS = ("current_time", "timezone", "current_location", "user_profile", "installed_apps")


def parse_record_safely(
    record: Mapping[str, Any], split: str, split_dir: str | Path | None = None
) -> Sample | None:
    """Parse one record into a Sample, returning ``None`` when unparsable."""
    if not isinstance(record, Mapping) or "__error__" in record:
        return None
    try:
        return Sample.from_raw(record, split, split_dir)
    except (SchemaError, ValueError, TypeError):
        return None


def _gold_call_problems(
    sample: Sample, catalog: ToolCatalog | None
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    label = sample.label
    if label is None:
        return errors, warnings
    installed = set(sample.installed_apps)
    if catalog is not None:
        for call in label.tool_calls:
            spec = catalog.get(call.name)
            if spec is None:
                errors.append(f"{sample.id}: gold tool '{call.name}' not in catalog")
                continue
            if spec.app_name and installed and spec.app_name not in installed and spec.kind != "meta":
                errors.append(
                    f"{sample.id}: gold tool '{call.name}' belongs to uninstalled app '{spec.app_name}'"
                )
            outcome = catalog.validate_call(call.name, call.arguments)
            for problem in outcome["errors"]:
                errors.append(f"{sample.id}: {problem}")
            for warning in outcome["warnings"]:
                warnings.append(f"{sample.id}: {warning}")
    if label.tool_calls and not label.tool_results:
        warnings.append(f"{sample.id}: gold tool_calls without tool_results")
    if label.tool_results and label.tool_calls and len(label.tool_results) != len(label.tool_calls):
        warnings.append(
            f"{sample.id}: gold tool_calls ({len(label.tool_calls)}) and tool_results "
            f"({len(label.tool_results)}) counts differ"
        )
    for idx, result in enumerate(label.tool_results):
        if idx < len(label.tool_calls) and result.name and result.name != label.tool_calls[idx].name:
            warnings.append(
                f"{sample.id}: tool_results[{idx}] name '{result.name}' != "
                f"tool_calls[{idx}] name '{label.tool_calls[idx].name}'"
            )
    return errors, warnings


def validate_split(
    paths: SplitPaths,
    samples: Iterable[Sample] | None = None,
    catalog: ToolCatalog | None = None,
) -> dict:
    """Build a JSON-friendly report for one split.

    ``samples`` may be pre-loaded; when ``None`` the split's JSONL is read
    through :func:`~lifemate.data.loader.iter_raw_records`.
    """
    malformed: list[dict] = []
    unknown_fields: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    duplicate_ids: list[str] = []
    missing_audio: list[str] = []
    schema_issues: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    if samples is None:
        raw_samples: list[Sample] = []
        seen_lines = 0
        if paths.jsonl is not None:
            for record in iter_raw_records(paths.jsonl):
                seen_lines += 1
                if "__error__" in record:
                    malformed.append({"line": record.get("__line__"), "error": record.get("__error__")})
                    continue
                sample = parse_record_safely(record, paths.split, paths.root)
                if sample is None:
                    schema_issues.append(f"line {seen_lines}: unparsable record")
                    continue
                raw_samples.append(sample)
        samples = raw_samples
    sample_list = list(samples)

    if catalog is None and paths.catalog is not None:
        try:
            catalog = ToolCatalog.from_file(paths.catalog)
        except Exception as exc:  # noqa: BLE001 - report, never crash a report
            warnings.append(f"catalog unreadable: {exc}")

    id_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for sample in sample_list:
        id_counts[sample.id] += 1
        for key in sample.raw:
            if key not in KNOWN_TOP_LEVEL_FIELDS:
                unknown_fields[key] += 1
        ctx = sample.context
        for name in CONTEXT_FIELDS:
            value = getattr(ctx, name, None)
            if value:
                field_presence[name] += 1
        if sample.audio_text:
            field_presence["audio_text"] += 1
        if sample.audio_path is not None:
            field_presence["audio_path"] += 1
        if sample.audio_path is not None and sample.audio_path.is_file():
            field_presence["audio_ok"] += 1
        elif sample.audio_path is not None:
            missing_audio.append(sample.id)
        if sample.history:
            field_presence["history"] += 1
        if sample.label is not None:
            field_presence["label"] += 1
            if sample.label.final_state is not None:
                field_presence["final_state"] += 1
        category = sample.raw.get("category")
        if isinstance(category, str):
            category_counts[category] += 1
        sample_errors, sample_warnings = _gold_call_problems(sample, catalog)
        errors.extend(sample_errors)
        warnings.extend(sample_warnings)

    duplicate_ids = sorted(sid for sid, count in id_counts.items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate sample ids: {duplicate_ids}")
    if missing_audio:
        errors.append(f"missing audio files: {missing_audio}")
    if malformed:
        warnings.append(f"malformed JSONL rows: {len(malformed)}")
    if schema_issues:
        warnings.append(f"unparsable records: {len(schema_issues)}")
    if catalog is None:
        warnings.append("no tool catalog found for this split")

    total = len(sample_list)
    return {
        "split": paths.split,
        "root": str(paths.root),
        "jsonl": None if paths.jsonl is None else str(paths.jsonl),
        "catalog": None if paths.catalog is None else str(paths.catalog),
        "has_environment_dir": paths.has_environment_dir,
        "counts": {
            "samples": total,
            "with_label": sum(1 for s in sample_list if s.label is not None),
            "with_audio_text": sum(1 for s in sample_list if s.audio_text),
            "with_history": sum(1 for s in sample_list if s.history),
            "with_final_state": sum(
                1 for s in sample_list if s.label is not None and s.label.final_state is not None
            ),
            "catalog_tools": 0 if catalog is None else len(catalog),
            "catalog_apps": 0 if catalog is None else len(catalog.apps),
        },
        "categories": dict(sorted(category_counts.items())),
        "duplicate_ids": duplicate_ids,
        "missing_audio": missing_audio,
        "schema_issues": schema_issues,
        "malformed_rows": malformed,
        "unknown_top_level_fields": dict(sorted(unknown_fields.items())),
        "field_presence": {k: f"{v}/{total}" for k, v in sorted(field_presence.items())},
        "errors": errors,
        "warnings": warnings,
        "gold_dependency_edges": [] if catalog is None else [list(e) for e in catalog.dependency_edges()],
    }
