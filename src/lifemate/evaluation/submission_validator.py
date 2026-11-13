"""Structural validator for competition submission files.

Never raises on a bad file: every problem becomes an entry in ``errors`` so the
caller can print all of them at once. Checks performed:

* the file parses as JSONL (one JSON object per non-blank line),
* exactly one row per expected id, no duplicates, no missing, no extra,
* field names and types: ``id: str``, ``tool_calls: list``,
  ``state_hash: str``; each call has ``name: str`` and ``arguments: object``,
* ``state_hash`` looks like a hash (non-empty, hex-ish) when required,
* row order matches ``expected_ids`` when ``require_order`` is set.

``calls_source="env"`` records that rows are expected to come from the
environment (``env.get_tool_calls()``); the validator can only check structure,
so provenance is asserted at write time by
:func:`lifemate.evaluation.submission.assert_env_sourced`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["HEX_RE", "validate_against_sample_submission", "validate_submission"]

HEX_RE = re.compile(r"^[0-9a-fA-F][0-9a-fA-F_:-]{3,}$")
_CALL_KEYS = ("name", "arguments")


def _load_rows(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {path}: {exc}")
        return rows
    except UnicodeDecodeError as exc:
        errors.append(f"{path} is not UTF-8 text: {exc}")
        return rows
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg} at char {exc.pos})")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {lineno}: row must be a JSON object, got {type(payload).__name__}")
            continue
        rows.append({"__line__": lineno, **payload})
    return rows


def _check_row(
    row: Mapping[str, Any],
    line: int,
    errors: list[str],
    warnings: list[str],
    *,
    require_state_hash: bool,
) -> None:
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append(f"line {line}: id must be a non-empty string")

    calls = row.get("tool_calls")
    if not isinstance(calls, list):
        errors.append(f"line {line}: tool_calls must be a list, got {type(calls).__name__}")
        calls = []
    for index, call in enumerate(calls):
        where = f"line {line}: tool_calls[{index}]"
        if not isinstance(call, dict):
            errors.append(f"{where} must be an object, got {type(call).__name__}")
            continue
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{where}.name must be a non-empty string")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            errors.append(f"{where}.arguments must be an object, got {type(arguments).__name__}")
        extra = set(call) - set(_CALL_KEYS)
        if extra:
            warnings.append(f"{where} has extra keys {sorted(extra)}")
        missing = set(_CALL_KEYS) - set(call)
        if missing:
            errors.append(f"{where} is missing keys {sorted(missing)}")

    if "state_hash" not in row:
        if require_state_hash:
            errors.append(f"line {line}: missing state_hash")
        else:
            warnings.append(f"line {line}: state_hash absent")
        return
    state_hash = row.get("state_hash")
    if not isinstance(state_hash, str):
        errors.append(f"line {line}: state_hash must be a string, got {type(state_hash).__name__}")
        return
    if require_state_hash:
        if not state_hash.strip():
            errors.append(f"line {line}: state_hash is empty")
        elif not HEX_RE.match(state_hash.strip()):
            errors.append(f"line {line}: state_hash {state_hash[:24]!r} is not a hex-ish hash")


def validate_submission(
    path: str | Path,
    expected_ids: Sequence[str],
    *,
    require_state_hash: bool = True,
    calls_source: str = "env",
    require_order: bool = True,
) -> dict[str, Any]:
    """Validate ``path`` against ``expected_ids``. Returns an error report."""
    errors: list[str] = []
    warnings: list[str] = []
    file_path = Path(path)
    if calls_source not in {"env", "unknown"}:
        errors.append(f"calls_source must be 'env' or 'unknown', got {calls_source!r}")

    if not file_path.exists():
        return {
            "ok": False,
            "errors": [f"submission file not found: {file_path}"],
            "warnings": [],
            "n": 0,
            "path": str(file_path),
            "calls_source": calls_source,
        }

    rows = _load_rows(file_path, errors)
    for row in rows:
        _check_row(row, int(row["__line__"]), errors, warnings, require_state_hash=require_state_hash)

    seen: dict[str, int] = {}
    order: list[str] = []
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str):
            continue
        if sample_id in seen:
            errors.append(f"duplicate id {sample_id!r} at lines {seen[sample_id]} and {row['__line__']}")
        else:
            seen[sample_id] = int(row["__line__"])
            order.append(sample_id)

    expected = [str(item) for item in expected_ids]
    expected_set = set(expected)
    missing = [sample_id for sample_id in expected if sample_id not in seen]
    extra = [sample_id for sample_id in order if sample_id not in expected_set]
    if missing:
        errors.append(f"{len(missing)} missing id(s): {missing[:10]}")
    if extra:
        errors.append(f"{len(extra)} unexpected id(s): {extra[:10]}")
    if require_order and not missing and not extra and order != expected:
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(order, expected)) if a != b),
            0,
        )
        errors.append(
            f"row order differs from expected ids at index {first_bad} "
            f"(got {order[first_bad]!r}, expected {expected[first_bad]!r})"
        )

    if calls_source == "env" and rows and not require_state_hash:
        warnings.append("calls_source='env' but state_hash is not required; official scoring will need it")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n": len(rows),
        "n_expected": len(expected),
        "path": str(file_path),
        "calls_source": calls_source,
    }


def validate_against_sample_submission(
    path: str | Path, sample_submission_path: str | Path
) -> dict[str, Any]:
    """Compare ``path`` structurally with the official sample submission.

    Only schema/keys are compared (ids and values legitimately differ): the
    required top-level keys and the per-call keys must match. Samples are
    usually only a handful of rows, so this is a smoke check.
    """
    errors: list[str] = []
    warnings: list[str] = []
    submission = validate_submission(path, _ids_of(path, errors), require_state_hash=False)
    errors.extend(submission["errors"])
    sample_rows = _load_rows(Path(sample_submission_path), errors)
    if not sample_rows:
        errors.append(f"sample submission {sample_submission_path} has no rows")
        return {"ok": False, "errors": errors, "warnings": warnings, "n": submission["n"]}

    sample_keys = set(sample_rows[0]) - {"__line__"}
    rows = _load_rows(Path(path), [])
    if rows:
        row_keys = set(rows[0]) - {"__line__"}
        for key in sorted(sample_keys - row_keys):
            errors.append(f"submission is missing top-level key {key!r} present in the sample")
        for key in sorted(row_keys - sample_keys):
            warnings.append(f"submission has extra top-level key {key!r} not in the sample")
        sample_calls = [call for row in sample_rows for call in row.get("tool_calls") or [] if isinstance(call, dict)]
        our_calls = [call for row in rows for call in row.get("tool_calls") or [] if isinstance(call, dict)]
        if sample_calls and our_calls:
            sample_call_keys = set(sample_calls[0])
            our_call_keys = set(our_calls[0])
            for key in sorted(sample_call_keys - our_call_keys):
                errors.append(f"tool_calls entries are missing key {key!r} present in the sample")
            for key in sorted(our_call_keys - sample_call_keys):
                warnings.append(f"tool_calls entries have extra key {key!r} not in the sample")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "n": submission["n"]}


def _ids_of(path: str | Path, errors: list[str]) -> list[str]:
    """Best-effort id list for a file we are about to validate."""
    rows = _load_rows(Path(path), errors)
    ids: list[str] = []
    for row in rows:
        sample_id = row.get("id")
        if isinstance(sample_id, str):
            ids.append(sample_id)
    return ids
