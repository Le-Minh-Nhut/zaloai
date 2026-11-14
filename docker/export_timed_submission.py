#!/usr/bin/env python3
"""Export official timing fields from the checksummed inference progress journal."""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from lifemate.evaluation.progress import fingerprint
from lifemate.evaluation.submission import SubmissionError, normalise_row


def export(raw_path, journal_path, output_path):
    raw_path, journal_path, output_path = map(Path, (raw_path, journal_path, output_path))
    rows = [normalise_row(json.loads(line)) for line in raw_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    payload = json.loads(journal_path.read_text(encoding='utf-8'))
    data = payload['data']
    if payload.get('sha256') != fingerprint(data):
        raise SubmissionError('progress checksum mismatch')
    journal_rows = [normalise_row(row) for row in data['rows']]
    times = data['latencies']
    if rows != journal_rows or len(times) != len(rows) or not rows:
        raise SubmissionError('submission and progress do not describe the same complete rows')
    for row, seconds in zip(rows, times):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise SubmissionError('missing or invalid per-sample latency')
        row['time'] = int(seconds * 1000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output_path.parent, delete=False) as stream:
            name = stream.name
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, output_path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('usage: export_timed_submission.py RAW JOURNAL OUTPUT')
    export(*sys.argv[1:])
