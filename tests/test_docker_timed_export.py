import importlib.util
import json
import math
from pathlib import Path

import pytest

from lifemate.evaluation.progress import ProgressJournal
from lifemate.evaluation.submission import SubmissionError

spec = importlib.util.spec_from_file_location('timed_export', Path(__file__).resolve().parents[1] / 'docker/export_timed_submission.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_inputs(tmp_path, latencies=(0.1239, 1.5)):
    rows = [{'id': 'a', 'tool_calls': [], 'state_hash': 'abcd'},
            {'id': 'b', 'tool_calls': [{'name': 'x', 'arguments': {}}], 'state_hash': 'f000'}]
    raw = tmp_path / 'raw.jsonl'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    journal = ProgressJournal(tmp_path / 'progress.json', {'mode': 'test'}, ['a', 'b'])
    journal.write(rows, ['ok', 'ok'], list(latencies), [False, False])
    return raw, journal.path, rows


def test_export_preserves_actions_order_hash_and_real_timing(tmp_path):
    raw, journal, rows = write_inputs(tmp_path)
    output = tmp_path / 'submission.jsonl'
    module.export(raw, journal, output)
    exported = [json.loads(line) for line in output.read_text().splitlines()]
    assert exported == [dict(rows[0], time=123), dict(rows[1], time=1500)]


@pytest.mark.parametrize('bad', [None, -1, True, math.inf])
def test_export_rejects_missing_or_invalid_timing_without_output(tmp_path, bad):
    raw, journal, _ = write_inputs(tmp_path, (bad, 1))
    output = tmp_path / 'submission.jsonl'
    with pytest.raises(SubmissionError):
        module.export(raw, journal, output)
    assert not output.exists()


def test_export_rejects_mismatch_and_tampering(tmp_path):
    raw, journal, _ = write_inputs(tmp_path)
    raw.write_text('{"id":"different","tool_calls":[],"state_hash":"abcd"}\n')
    with pytest.raises(SubmissionError):
        module.export(raw, journal, tmp_path / 'out')
    raw, journal, _ = write_inputs(tmp_path)
    data = json.loads(journal.read_text())
    data['data']['latencies'][0] = 10
    journal.write_text(json.dumps(data))
    with pytest.raises(SubmissionError, match='checksum'):
        module.export(raw, journal, tmp_path / 'out')
