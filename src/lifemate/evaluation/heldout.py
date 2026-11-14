"""Select original source environments using a verified prepared split."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from lifemate.data.preprocessing import hash_ids, sample_hash
from lifemate.errors import LifeMateError
from lifemate.compliance import assert_path_allowed


def select_heldout(samples: Sequence[Any], dataset: str | Path, *, split: str, limit: int | None = None):
    root = assert_path_allowed(dataset)
    assert_path_allowed(root.resolve())
    for name in ('manifest.json', 'train.jsonl', 'val.jsonl'):
        assert_path_allowed((root / name).resolve())
    try:
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        ids = {}
        for name in ('train', 'val'):
            with (root / f'{name}.jsonl').open(encoding='utf-8') as handle:
                ids[name] = {str(json.loads(line)['sample_id']) for line in handle if line.strip()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LifeMateError(f'cannot read held-out dataset {root}: {exc}') from exc
    if split != 'train' or manifest.get('split') != 'train':
        raise LifeMateError('held-out dataset must refer to the original train split')
    byid = {str(sample.id): sample for sample in samples}
    if ids['train'] & ids['val'] or (ids['train'] | ids['val']) - set(byid):
        raise LifeMateError('held-out dataset has overlapping or unknown source IDs')
    for name in ('train', 'val'):
        if hash_ids(ids[name]) != manifest.get(f'{name}_ids_hash'):
            raise LifeMateError(f'{name} source IDs do not match manifest')
    if {sample_hash(byid[x]) for x in ids['train']} & {sample_hash(byid[x]) for x in ids['val']}:
        raise LifeMateError('held-out source groups overlap training groups')
    selected = [sample for sample in samples if str(sample.id) in ids['val']]
    if limit is not None:
        selected = selected[:limit]
    selected_ids = [str(sample.id) for sample in selected]
    if not selected_ids:
        raise LifeMateError('held-out selection contains no samples')
    return selected, {'ids': selected_ids, 'ids_sha256': hashlib.sha256(json.dumps(selected_ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(), 'dataset_manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()}
