"""Submission JSONL writer/reader with atomic writes and env-provenance checks.

Row shape (contract, one row per sample, input order preserved)::

    {"id": "sample-1", "tool_calls": [{"name": "alarm.list", "arguments": {}}],
     "state_hash": "ab12..."}

``tool_calls`` and ``state_hash`` MUST be copied from the environment
(``env.get_tool_calls()`` / ``env.get_state_hash()``). They are never
reconstructed, and failed calls stay in the list because the official scoring
tooling sees the whole call history.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from lifemate.errors import LifeMateError

__all__ = [
    "SubmissionError",
    "SubmissionWriter",
    "assert_env_sourced",
    "load_submission",
    "normalise_row",
    "write_submission",
]


class SubmissionError(LifeMateError):
    """Raised when a submission row or file is structurally invalid."""


def _validate_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, (list, tuple)):
        raise SubmissionError(f"row.tool_calls must be a list, got {type(tool_calls).__name__}")
    out: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, Mapping):
            name = call.get("name")
            arguments = call.get("arguments", {})
        else:
            name = getattr(call, "name", None)
            arguments = getattr(call, "arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise SubmissionError(f"row.tool_calls[{index}].name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise SubmissionError(
                f"row.tool_calls[{index}].arguments must be an object, got {type(arguments).__name__}"
            )
        out.append({"name": name, "arguments": dict(arguments)})
    return out


def _validate_hash(state_hash: Any) -> str:
    if state_hash is None:
        return ""
    if not isinstance(state_hash, str):
        raise SubmissionError(f"row.state_hash must be a string, got {type(state_hash).__name__}")
    return state_hash


def normalise_row(row: Mapping[str, Any], *, row_index: int | None = None) -> dict[str, Any]:
    """Validate and canonicalise one submission row (dict form)."""
    where = "" if row_index is None else f"row {row_index}: "
    if not isinstance(row, Mapping):
        raise SubmissionError(f"{where}row must be an object, got {type(row).__name__}")
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SubmissionError(f"{where}id must be a non-empty string")
    tool_calls = _validate_calls(row.get("tool_calls", []))
    state_hash = _validate_hash(row.get("state_hash"))
    return {"id": sample_id, "tool_calls": tool_calls, "state_hash": state_hash}


class SubmissionWriter:
    """Collect rows in input order and write them atomically.

    ``add`` validates types immediately so a bad row fails next to its sample
    instead of at the end of a long run.
    """

    def __init__(self, path: str | Path, *, extra_keys: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self.extra_keys = tuple(extra_keys)
        self.rows: list[dict[str, Any]] = []
        self._ids: set[str] = set()
        self._finalized = False

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> list[str]:
        return [row["id"] for row in self.rows]

    def add(self, sample_id: str, tool_calls: Any, state_hash: Any, **extra: Any) -> dict[str, Any]:
        """Append one row. Raises :class:`SubmissionError` on bad types/duplicates."""
        if self._finalized:
            raise SubmissionError("writer already finalized")
        row = normalise_row({"id": sample_id, "tool_calls": tool_calls, "state_hash": state_hash})
        if row["id"] in self._ids:
            raise SubmissionError(f"duplicate id in submission: {row['id']}")
        for key in extra:
            if key not in self.extra_keys:
                raise SubmissionError(f"unexpected extra key {key!r}; declare it in extra_keys")
        for key in self.extra_keys:
            if key in extra:
                row[key] = extra[key]
        self.rows.append(row)
        self._ids.add(row["id"])
        return row

    def add_result(self, sample_id: str, result: Any, env: Any = None) -> dict[str, Any]:
        """Append a row straight from an ``AgentResult`` (+ optional env check)."""
        tool_calls = getattr(result, "tool_calls", None)
        state_hash = getattr(result, "state_hash", None)
        row = self.add(sample_id, tool_calls, state_hash)
        if env is not None:
            assert_env_sourced(row, env)
        return row

    def write(self) -> Path:
        """Write ``*.tmp`` then rename (atomic on POSIX, same directory)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        )
        tmp_name = handle.name
        try:
            with handle:
                for row in self.rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return self.path

    def finalize(self) -> Path:
        """Write and freeze the writer."""
        path = self.write()
        self._finalized = True
        return path


def write_submission(rows: Iterable[Mapping[str, Any]], path: str | Path, *, extra_keys: Iterable[str] = ()) -> Path:
    """Convenience: build a writer from validated rows and finalize it."""
    writer = SubmissionWriter(path, extra_keys=extra_keys)
    for row in rows:
        extra = {k: row[k] for k in writer.extra_keys if k in row}
        writer.add(str(row.get("id", "")), row.get("tool_calls"), row.get("state_hash"), **extra)
    return writer.finalize()


def load_submission(path: str | Path) -> list[dict[str, Any]]:
    """Read a submission JSONL; raises :class:`SubmissionError` with the line number."""
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SubmissionError(f"{file_path}:{lineno}: invalid JSON ({exc.msg})") from exc
            rows.append(normalise_row(payload, row_index=lineno))
    return rows


def assert_env_sourced(row: Mapping[str, Any], env: Any) -> None:
    """Defensive provenance hook: the row must echo the env's own outputs.

    Compares against ``env.get_tool_calls()`` / ``env.get_state_hash()`` when the
    environment exposes them. Raises :class:`SubmissionError` on a mismatch so a
    reconstructed call list can never silently reach the submission file.
    """
    if not hasattr(env, "get_tool_calls") and not hasattr(env, "get_state_hash"):
        return
    if hasattr(env, "get_tool_calls"):
        env_calls = _validate_calls(env.get_tool_calls())
        row_calls = _validate_calls(row.get("tool_calls", []))
        if [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in env_calls] != [
            json.dumps(c, sort_keys=True, ensure_ascii=False) for c in row_calls
        ]:
            raise SubmissionError(
                f"row {row.get('id')!r}: tool_calls differ from env.get_tool_calls(); "
                "submission rows must be copied from the environment"
            )
    if hasattr(env, "get_state_hash"):
        env_hash = _validate_hash(env.get_state_hash())
        if str(row.get("state_hash") or "") != str(env_hash):
            raise SubmissionError(
                f"row {row.get('id')!r}: state_hash differs from env.get_state_hash(); "
                "hash rows must be copied from the environment"
            )
