"""Submission writer/validator tests: atomicity, ordering, every error class."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lifemate.evaluation.submission import (  # noqa: E402
    SubmissionError,
    SubmissionWriter,
    assert_env_sourced,
    load_submission,
    write_submission,
)
from lifemate.evaluation.submission_validator import (  # noqa: E402
    validate_against_sample_submission,
    validate_submission,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="lifemate_sub_"))


class _Env:
    """Minimal env stand-in exposing the two provenance getters."""

    def __init__(self, calls, state_hash):
        self._calls = calls
        self._hash = state_hash

    def get_tool_calls(self):
        return self._calls

    def get_state_hash(self):
        return self._hash


def test_writer_preserves_order_and_writes_atomically():
    directory = _tmpdir()
    path = directory / "submission.jsonl"
    writer = SubmissionWriter(path)
    writer.add("s3", [], HASH_B)
    writer.add("s1", [{"name": "alarm.set", "arguments": {"time": "06:30"}}], HASH_A)
    writer.add("s2", [], HASH_A)
    assert not path.exists(), "writer must not touch the target before finalize()"
    writer.finalize()
    rows = load_submission(path)
    assert [row["id"] for row in rows] == ["s3", "s1", "s2"], rows
    assert rows[1]["tool_calls"] == [{"name": "alarm.set", "arguments": {"time": "06:30"}}]
    assert not list(directory.glob("*.tmp")), "temp file was left behind"


def test_writer_rejects_duplicate_id():
    writer = SubmissionWriter(_tmpdir() / "dup.jsonl")
    writer.add("s1", [], HASH_A)
    try:
        writer.add("s1", [], HASH_B)
    except SubmissionError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate id was accepted")


def test_writer_rejects_bad_types():
    writer = SubmissionWriter(_tmpdir() / "bad.jsonl")
    cases = [
        ("s1", "not-a-list", HASH_A),
        ("s1", [{"name": "", "arguments": {}}], HASH_A),
        ("s1", [{"name": "a", "arguments": ["x"]}], HASH_A),
        ("s1", [], 17),
        (12, [], HASH_A),
    ]
    for sample_id, calls, state_hash in cases:
        try:
            writer.add(sample_id, calls, state_hash)
        except SubmissionError:
            continue
        raise AssertionError(f"bad row accepted: {sample_id!r} {calls!r} {state_hash!r}")


def test_writer_keeps_failed_calls_verbatim():
    path = _tmpdir() / "failed.jsonl"
    calls = [
        {"name": "alarm.set", "arguments": {"time": "06:30"}, "ok": False, "error": "boom"},
        {"name": "alarm.set", "arguments": {"time": "06:30"}},
    ]
    write_submission([{"id": "s1", "tool_calls": calls, "state_hash": HASH_A}], path)
    rows = load_submission(path)
    assert len(rows[0]["tool_calls"]) == 2
    assert rows[0]["tool_calls"][0]["name"] == "alarm.set"


def test_assert_env_sourced_detects_reconstruction():
    env = _Env([{"name": "alarm.set", "arguments": {"time": "06:30"}}], HASH_A)
    good = {"id": "s1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}], "state_hash": HASH_A}
    assert_env_sourced(good, env)
    for bad in (
        {"id": "s1", "tool_calls": [], "state_hash": HASH_A},
        {"id": "s1", "tool_calls": good["tool_calls"], "state_hash": HASH_B},
    ):
        try:
            assert_env_sourced(bad, env)
        except SubmissionError:
            continue
        raise AssertionError(f"env mismatch accepted: {bad}")


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_validator_accepts_a_correct_file():
    path = _write_rows(
        _tmpdir() / "ok.jsonl",
        [
            {"id": "s1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}], "state_hash": HASH_A},
            {"id": "s2", "tool_calls": [], "state_hash": HASH_B},
        ],
    )
    report = validate_submission(path, ["s1", "s2"])
    assert report["ok"], report
    assert report["n"] == 2


def test_validator_catches_missing_and_extra_rows():
    path = _write_rows(_tmpdir() / "miss.jsonl", [{"id": "s1", "tool_calls": [], "state_hash": HASH_A}])
    report = validate_submission(path, ["s1", "s2"])
    assert not report["ok"] and any("missing" in err for err in report["errors"]), report
    extra = _write_rows(
        _tmpdir() / "extra.jsonl",
        [
            {"id": "s1", "tool_calls": [], "state_hash": HASH_A},
            {"id": "s2", "tool_calls": [], "state_hash": HASH_A},
            {"id": "sX", "tool_calls": [], "state_hash": HASH_A},
        ],
    )
    report = validate_submission(extra, ["s1", "s2"])
    assert not report["ok"] and any("unexpected" in err for err in report["errors"]), report


def test_validator_catches_duplicates_and_bad_fields():
    dup = _write_rows(
        _tmpdir() / "dup2.jsonl",
        [
            {"id": "s1", "tool_calls": [], "state_hash": HASH_A},
            {"id": "s1", "tool_calls": [], "state_hash": HASH_A},
        ],
    )
    assert any("duplicate" in error for error in validate_submission(dup, ["s1"])["errors"])

    bad = _write_rows(
        _tmpdir() / "badfields.jsonl",
        [
            {"id": "s1", "tool_calls": "nope", "state_hash": HASH_A},
            {"id": "s2", "tool_calls": [{"name": "", "arguments": {}}], "state_hash": HASH_A},
            {"id": "s3", "tool_calls": [{"name": "a", "arguments": []}], "state_hash": HASH_A},
            {"id": "s4", "tool_calls": [], "state_hash": ""},
            {"id": "s5"},
            {"id": 7, "tool_calls": [], "state_hash": HASH_A},
        ],
    )
    errors = validate_submission(bad, ["s1", "s2", "s3", "s4", "s5", "7"])["errors"]
    joined = " | ".join(errors)
    for needle in ("tool_calls must be a list", "name must be a non-empty string", "arguments must be an object",
                   "state_hash is empty", "missing state_hash", "id must be a non-empty string"):
        assert needle in joined, f"{needle!r} not reported in {joined}"


def test_validator_catches_wrong_order():
    path = _write_rows(
        _tmpdir() / "order.jsonl",
        [
            {"id": "s2", "tool_calls": [], "state_hash": HASH_A},
            {"id": "s1", "tool_calls": [], "state_hash": HASH_A},
        ],
    )
    report = validate_submission(path, ["s1", "s2"])
    assert not report["ok"] and any("order" in error for error in report["errors"]), report
    relaxed = validate_submission(path, ["s1", "s2"], require_order=False)
    assert relaxed["ok"], relaxed


def test_validator_never_crashes_on_garbage():
    directory = _tmpdir()
    cases = {
        "notjson.jsonl": "{not json}\n",
        "array.jsonl": "[1,2,3]\n",
        "binary.jsonl": "\x00\x01\x02\n",
    }
    for name, text in cases.items():
        path = directory / name
        path.write_text(text, encoding="utf-8", errors="replace")
        report = validate_submission(path, ["s1"])
        assert report["ok"] is False and report["errors"], (name, report)
    missing = validate_submission(directory / "absent.jsonl", ["s1"])
    assert missing["ok"] is False and "not found" in missing["errors"][0]
    empty = directory / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert validate_submission(empty, ["s1"])["ok"] is False


def test_round_trip_and_sample_submission_comparison():
    directory = _tmpdir()
    path = directory / "submission.jsonl"
    write_submission(
        [
            {"id": "s1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}], "state_hash": HASH_A},
        ],
        path,
    )
    rows = load_submission(path)
    assert rows[0]["state_hash"] == HASH_A
    sample = directory / "sample_submission.jsonl"
    sample.write_text(json.dumps({"id": "s1", "tool_calls": [], "state_hash": ""}) + "\n", encoding="utf-8")
    comparison = validate_against_sample_submission(path, sample)
    assert comparison["ok"], comparison
