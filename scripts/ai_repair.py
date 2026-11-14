#!/usr/bin/env python3
"""Prepare a tested repair in a new worktree; never mutate the active runner."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SUITES = {
    'training': ['tests/test_training.py', 'tests/test_training_resume.py'],
    'inference': ['tests/test_inference.py', 'tests/test_submission.py'],
    'evaluation': ['tests/test_evaluation_modes.py', 'tests/test_submission.py'],
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review', required=True, help='REJECTED gate record containing BUG_CONFIRMED')
    parser.add_argument('--worktree', required=True, help='new, non-existing isolated path')
    parser.add_argument('--suite', choices=sorted(SUITES), required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--timeout', type=int, default=600)
    args=parser.parse_args()
    if args.timeout < 1 or args.timeout > 1800: raise RuntimeError('invalid timeout')
    record=json.loads(Path(args.review).read_text())
    review=record.get('review') or {}
    if record.get('status') != 'REJECTED' or review.get('verdict') != 'BUG_CONFIRMED':
        raise RuntimeError('a rejected gate with BUG_CONFIRMED is required')
    base=record['facts']['git_sha']
    if not isinstance(base,str) or len(base)!=40 or any(c not in '0123456789abcdef' for c in base): raise RuntimeError('invalid source commit')
    target=Path(args.worktree).resolve()
    if target.exists(): raise RuntimeError('repair worktree path already exists')
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(['git','worktree','add','--detach',str(target),base],cwd=repo,check=True)
    suite=[p for p in SUITES[args.suite] if (target/p).is_file()]
    if not suite: raise RuntimeError('no focused tests exist for selected suite')
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled',PYTHONPATH='src')
    prompt=('Investigate and minimally fix the confirmed LifeMate issue in this isolated worktree. '
            'Never access hidden data, GPU, training process, external messaging or other worktrees. '
            'Preserve scientific semantics, add focused regression tests and do not commit. '
            'Evidence: '+json.dumps(review,ensure_ascii=False))
    subprocess.run(['codex','exec','--sandbox','workspace-write','--ephemeral','-C',str(target),'-'],
                   cwd=target,input=prompt,text=True,env=env,timeout=args.timeout,check=True)
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=target,text=True)
    changed=[line[3:] for line in status.splitlines()]
    if not any(path.startswith(('src/', 'scripts/', 'configs/')) for path in changed):
        raise RuntimeError('repair did not change implementation code or configuration')
    changed_tests=[line[3:] for line in status.splitlines() if line[3:].startswith('tests/test_') and line[3:].endswith('.py')]
    if not changed_tests:
        raise RuntimeError('repair must add or modify a focused regression test')
    suite=list(dict.fromkeys([*suite, *changed_tests]))
    subprocess.run([args.python,'-m','pytest','-q',*suite],cwd=target,env=env,timeout=args.timeout,check=True)
    subprocess.run(['git','diff','--check'],cwd=target,check=True)
    print(f'Repair and tests finished in {target}; review and commit there before any future use. Active runner was not changed.')


if __name__=='__main__': main()
