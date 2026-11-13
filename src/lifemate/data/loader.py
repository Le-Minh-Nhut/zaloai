"""JSONL loading with guarded file access and no-traceback failure modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..compliance import safe_open
from ..errors import DatasetNotFoundError
from .catalog import ToolCatalog
from .discovery import SplitPaths, require_split
from .schemas import Sample

__all__ = [
    "iter_raw_records",
    "load_raw_split",
    "load_samples",
    "load_catalog",
    "read_sample_audio",
]


def iter_raw_records(path: str | Path, skip_malformed: bool = False) -> Iterator[dict]:
    """Yield one dict per JSONL line.

    Malformed lines yield ``{"__line__": n, "__error__": msg, "__raw__": text}``
    unless ``skip_malformed`` is set.
    """
    with safe_open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                if skip_malformed:
                    continue
                yield {"__line__": lineno, "__error__": f"{exc.msg} at char {exc.pos}", "__raw__": text[:500]}
                continue
            if not isinstance(record, dict):
                if skip_malformed:
                    continue
                yield {"__line__": lineno, "__error__": f"record is {type(record).__name__}, not object", "__raw__": text[:500]}
                continue
            yield record


def _require_jsonl(paths: SplitPaths) -> Path:
    if paths.jsonl is None:
        raise DatasetNotFoundError(paths.split, str(paths.root / f"{paths.split}.jsonl"))
    return paths.jsonl


def load_raw_split(
    data_root: str | Path, split: str, limit: int | None = None, skip_malformed: bool = False
) -> list[dict]:
    """Load raw records for a split; malformed rows are kept as error markers."""
    paths = require_split(data_root, split)
    jsonl = _require_jsonl(paths)
    records: list[dict] = []
    for record in iter_raw_records(jsonl, skip_malformed=skip_malformed):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def load_samples(
    data_root: str | Path,
    split: str,
    limit: int | None = None,
    with_labels: bool = True,
) -> list[Sample]:
    """Load typed samples; malformed rows are skipped, never raised."""
    from .validator import parse_record_safely  # local import avoids a cycle

    paths = require_split(data_root, split)
    jsonl = _require_jsonl(paths)
    samples: list[Sample] = []
    for record in iter_raw_records(jsonl, skip_malformed=True):
        sample = parse_record_safely(record, split, paths.root)
        if sample is None:
            continue
        if not with_labels:
            sample.label = None
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_catalog(data_root: str | Path, split: str) -> ToolCatalog | None:
    """Load the split's tool catalog, or ``None`` when absent."""
    paths = require_split(data_root, split)
    if paths.catalog is None:
        return None
    return ToolCatalog.from_file(paths.catalog)


def read_sample_audio(sample: Sample) -> Path | None:
    """Return the sample's existing audio file, or ``None``.

    Raises :class:`~lifemate.errors.ProhibitedAccessError` for prohibited
    paths; a plain missing file is reported as ``None``.
    """
    from ..compliance import assert_path_allowed

    if sample.audio_path is None:
        return None
    target = assert_path_allowed(sample.audio_path)
    return target if target.is_file() else None
