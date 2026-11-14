"""Atomic, checksummed inference progress distinct from submission output."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from lifemate.evaluation.submission import SubmissionError, normalise_row
from lifemate.compliance import assert_path_allowed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_identity(path: Any) -> str | None:
    if not path:
        return None
    target = assert_path_allowed(str(path))
    assert_path_allowed(target.resolve())
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


class ProgressJournal:
    def __init__(self, path: str | Path, identity: Mapping[str, Any], expected_ids: list[str]):
        self.path = assert_path_allowed(path)
        assert_path_allowed(self.path.resolve())
        self.identity = fingerprint(identity)
        self.expected_ids = expected_ids
        self.statuses: list[str] = []

    def read(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
            data = payload['data']
            if payload['sha256'] != fingerprint(data) or data['identity'] != self.identity:
                raise SubmissionError('progress journal checksum or run identity mismatch')
            rows = [normalise_row(row) for row in data['rows']]
            if len(rows) > len(self.expected_ids) or [row['id'] for row in rows] != self.expected_ids[:len(rows)]:
                raise SubmissionError('progress journal sample ordering mismatch')
            if any(not row['state_hash'] for row in rows):
                raise SubmissionError('progress journal contains an empty state hash')
            statuses = data.get('statuses')
            allowed = {'ok', 'asr_failed', 'history_incomplete', 'agent_failed', 'unknown'}
            if not isinstance(statuses, list) or len(statuses) != len(rows) or any(status not in allowed for status in statuses):
                raise SubmissionError('progress journal contains invalid failure accounting')
            self.statuses = statuses
            return rows
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SubmissionError(f'invalid progress journal: {exc}') from exc

    def write(self, rows: list[dict[str, Any]], statuses: list[str] | None = None) -> None:
        data = {'identity': self.identity, 'rows': rows, 'statuses': statuses if statuses is not None else ['unknown'] * len(rows)}
        payload = {'data': data, 'sha256': fingerprint(data)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=self.path.parent, prefix=self.path.name + '.', delete=False) as handle:
                name = handle.name
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
