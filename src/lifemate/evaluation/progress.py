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
        self.latencies: list[float | None] = []
        self.timeouts: list[bool | None] = []

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
            latencies = data.get('latencies', [None] * len(rows))
            timeouts = data.get('timeouts', [None] * len(rows))
            if (not isinstance(latencies, list) or len(latencies) != len(rows) or
                any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1e12) for v in latencies) or
                not isinstance(timeouts, list) or len(timeouts) != len(rows) or
                any(v is not None and not isinstance(v, bool) for v in timeouts)):
                raise SubmissionError('progress journal contains invalid timing accounting')
            self.statuses, self.latencies, self.timeouts = statuses, latencies, timeouts
            return rows
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SubmissionError(f'invalid progress journal: {exc}') from exc

    def write(self, rows: list[dict[str, Any]], statuses: list[str] | None = None,
              latencies: list[float | None] | None = None, timeouts: list[bool | None] | None = None) -> None:
        data = {'identity': self.identity, 'rows': rows, 'statuses': statuses if statuses is not None else ['unknown'] * len(rows),
                'latencies': latencies if latencies is not None else [None] * len(rows),
                'timeouts': timeouts if timeouts is not None else [None] * len(rows)}
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


def write_runtime_stats(path: str | Path, statuses: list[str], latencies: list[float | None], timeouts: list[bool | None]) -> None:
    """Persist aggregate measurements only, with missing resumed timing explicit."""
    target = assert_path_allowed(path)
    assert_path_allowed(target.resolve())
    values = sorted(float(v) for v in latencies if v is not None)
    def percentile(q: float) -> float | None:
        if not values: return None
        pos = (len(values) - 1) * q; lo = int(pos); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    data = {'n_samples':len(statuses), 'status_counts':{key:statuses.count(key) for key in sorted(set(statuses))},
            'timeout_count':sum(v is True for v in timeouts), 'timeout_unknown':sum(v is None for v in timeouts),
            'latency_seconds':{'n':len(values), 'mean':sum(values)/len(values) if values else None,
                               'p50':percentile(.5), 'p95':percentile(.95), 'max':max(values) if values else None},
            'note':'per-sample wall time; model initialization, global input hashing and journal/output writes excluded'}
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=target.parent, prefix=target.name+'.', delete=False) as handle:
            name = handle.name; json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, target)
    finally:
        if name and os.path.exists(name): os.unlink(name)
