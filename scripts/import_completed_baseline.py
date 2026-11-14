#!/usr/bin/env python3
"""Import a stopped, completed baseline into a fresh gated run without retraining."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

import run_overnight as r


def main():
    p=argparse.ArgumentParser(); p.add_argument('--old-run',required=True); p.add_argument('--config',required=True); p.add_argument('--stop-at',required=True); a=p.parse_args()
    old=Path(a.old_run).resolve(); old_state=json.loads((old/'state.json').read_text())
    if r.same_process(old_state.get('worker_pid'),old_state.get('worker_start')):
        raise RuntimeError('old worker is still running; do not import active artifacts')
    for j in old_state.get('jobs',{}).values():
        if j.get('status')=='RUNNING' and r.same_process(j.get('pid'),j.get('process_start')):
            raise RuntimeError('old job is still running')
    baseline=old_state.get('jobs',{}).get('baseline',{})
    if baseline.get('status')!='COMPLETED' or baseline.get('exit_code')!=0:
        raise RuntimeError('completed successful baseline required')
    cfg=json.loads(Path(a.config).read_text()); dest=Path(cfg['work_dir'])
    if dest.exists(): raise RuntimeError('new run path already exists')
    if not (cfg.get('ai_review_gate') or {}).get('enabled'): raise RuntimeError('review gate must be enabled')
    previous=old_state['config']; ignored={'work_dir','ai_review_gate'}
    if {k:v for k,v in cfg.items() if k not in ignored} != {k:v for k,v in previous.items() if k not in ignored}:
        raise RuntimeError('configuration differs from original scientific protocol')
    deadline=dt.datetime.fromisoformat(a.stop_at)
    if deadline.tzinfo is None or deadline<=r.now(): raise RuntimeError('future timezone-aware stop time required')
    selection=r.preflight(cfg)
    for key in ('ids_sha256','expected_horizon'):
        expected=old_state['expected_horizon'] if key=='expected_horizon' else old_state['heldout'][key]
        if selection[key]!=expected: raise RuntimeError(f'{key} differs from original run')
    for name, expected in old_state['dataset_sha256'].items():
        if r.sha(Path(cfg['dataset'])/name)!=expected: raise RuntimeError('dataset identity changed')
    for name, expected in old_state['model_sha256'].items():
        if r.sha(Path(cfg['model'])/name)!=expected: raise RuntimeError('model identity changed')
    candidates=sorted(p for p in (old/'candidates').glob('step-*') if r.valid_candidate(p))
    if not candidates: raise RuntimeError('no valid protected baseline candidate exists')
    dest.mkdir(parents=True)
    target=dest/'candidates'; target.mkdir()
    for source in candidates:
        identity=json.loads((source/'candidate_identity.json').read_text())
        if r.sha(source/'adapter_model.safetensors') != identity.get('adapter_sha256'):
            raise RuntimeError(f'candidate adapter hash mismatch: {source}')
        shutil.copytree(source,target/source.name)
        copied=target/source.name
        if not r.valid_candidate(copied) or r.sha(copied/'adapter_model.safetensors') != identity['adapter_sha256']:
            raise RuntimeError('copied candidate failed integrity validation')
    revision=r.git_sha(); source=r.prepare_worktree(dest/'source',revision)
    state={'status':'INTERRUPTED','created':r.now().isoformat(),'stop_at':deadline.isoformat(),'git_sha':revision,
           'dataset_sha256':old_state['dataset_sha256'],'model_sha256':old_state['model_sha256'],
           'heldout':old_state['heldout'],'expected_horizon':old_state['expected_horizon'],
           'config':cfg,'source':str(source.resolve()),'jobs':{'baseline':baseline},
           'imported_baseline':{'source':str(old),'original_git_sha':old_state['git_sha'],'no_retraining':True}}
    r.atomic(dest/'state.json',state)
    print(f'Imported completed baseline to {dest}; use runner --resume --config {a.config}. Evaluations run afresh.')


if __name__=='__main__':
    try: main()
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); sys.exit(2)
