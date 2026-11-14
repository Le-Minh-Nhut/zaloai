#!/usr/bin/env python3
"""Durable, read-only Codex reviewer attached to the existing overnight state.

This observer never controls the active runner. Decisions are advisory: that
runner was started from immutable source and cannot be gated safely in place.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / 'configs/codex_review_schema.json'
VERDICTS = {'PASS', 'BUG_CONFIRMED', 'MODEL_UNDERPERFORMANCE', 'INCONCLUSIVE', 'BLOCKED'}
DECISIONS = {'CONTINUE', 'RUN_EVALUATION', 'START_NEXT_EXPERIMENT', 'INVESTIGATE', 'RETRY_FROM_CHECKPOINT', 'SKIP_EXPERIMENT', 'STOP'}


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as f:
            name = f.name; json.dump(data, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name): os.unlink(name)


def event(job, kind, marker, entry):
    identity = hashlib.sha256(f'{job}\0{kind}\0{marker}'.encode()).hexdigest()[:24]
    return {'id': identity, 'job': job, 'kind': kind, 'marker': str(marker), 'entry': entry}


def events_from_state(state, previous, now=None):
    now = time.time() if now is None else now
    events = []
    for name, item in state.get('jobs', {}).items():
        status = item.get('status', '')
        # A retrospective event is explicitly not a pre-training gate.
        if name == 'baseline' or name.startswith('train-'):
            if item.get('started'):
                events.append(event(name, 'BEFORE_TRAIN', item['started'], item))
            if status in {'COMPLETED', 'FAILED', 'INTERRUPTED'}:
                events.append(event(name, 'AFTER_TRAIN', item.get('ended', status), item))
            if status == 'FAILED':
                events.append(event(name, 'ON_ANOMALY', 'failed-' + str(item.get('ended', status)), item))
            telemetry = item.get('telemetry') or {}
            loss = telemetry.get('train_loss')
            if isinstance(loss, (int, float)) and not math.isfinite(loss):
                events.append(event(name, 'ON_ANOMALY', f'nonfinite-loss-{telemetry.get("step")}', item))
            stamp = telemetry.get('step_observed_at')
            if status == 'RUNNING' and stamp and now - stamp > 1800:
                # One event per observed stalled step; a later step starts a new episode.
                events.append(event(name, 'ON_ANOMALY', f'stalled-step-{telemetry.get("step")}', item))
        elif name.startswith('eval-') or name.startswith('audio-'):
            if status in {'COMPLETED', 'FAILED', 'INTERRUPTED'}:
                events.append(event(name, 'AFTER_EVAL', item.get('ended', status), item))
        elif name.startswith('submission') or name.startswith('submit-'):
            if status in {'COMPLETED', 'FAILED', 'INTERRUPTED'} and (status != 'COMPLETED' or ('output' in item and 'validation' in item)):
                events.append(event(name, 'SUBMISSION_CHECK', item.get('ended', status), item))
    return events


def validate(review, ev):
    keys = {'event_id','experiment_id','verdict','confidence','evidence','root_cause','affected_components','recommended_action','decision','retraining_necessary','checkpoint_valid'}
    if not isinstance(review, dict) or set(review) != keys: return False
    if review['event_id'] != ev['id'] or review['experiment_id'] != ev['job']: return False
    if review['verdict'] not in VERDICTS or review['decision'] not in DECISIONS or review['confidence'] not in {'LOW','MEDIUM','HIGH'}: return False
    if not isinstance(review['recommended_action'], str) or not review['recommended_action'].strip(): return False
    if review['root_cause'] is not None and not isinstance(review['root_cause'], str): return False
    if any(not isinstance(review[k], list) or any(not isinstance(v, str) for v in review[k]) for k in ('evidence','affected_components')): return False
    return all(review[k] is None or type(review[k]) is bool for k in ('retraining_necessary','checkpoint_valid'))


def enqueue(root, events):
    queue = root / 'events'; queue.mkdir(parents=True, exist_ok=True)
    for ev in events:
        path = queue / (ev['id'] + '.json')
        if not path.exists(): atomic(path, {'event':ev, 'status':'PENDING', 'attempts':0})


def process_identity(pid):
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = stat.rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def prompt(ev, state_path):
    return f'''You are a read-only independent research and QA reviewer for LifeMate.\nEvent: {json.dumps(ev, ensure_ascii=False)}\nRunner durable state: {state_path}\nInspect actual relevant evidence, logs and artifacts. BEFORE_TRAIN for an already running job is retrospective: do not claim it gated launch. Inspect provenance, held-out separation, configuration, action targets and loss semantics as relevant. For later events review checkpoints, actual evaluation identities and tool/state semantics. Treat model failures separately from infrastructure failures. Local metrics are not official scores. Never access hidden state, launch GPU jobs, modify files, restart jobs, or send messages. Do not infer evidence that is absent. Return the required JSON schema with event_id={ev['id']} and experiment_id={ev['job']}. Decisions are recommendations only. If unable to verify, use INCONCLUSIVE or BLOCKED.\n'''


def process_one(root, source, state_path, *, timeout=300, max_attempts=2, executable='codex'):
    for path in sorted((root / 'events').glob('*.json')):
        record = json.loads(path.read_text()); ev = record['event']
        if record.get('status') in {'DONE','FAILED'}: continue
        # A previously detached reviewer may survive a supervisor crash.
        # Never launch a concurrent duplicate while its PID identity matches.
        if record.get('status') == 'RUNNING' and record.get('reviewer_start'):
            if process_identity(record.get('reviewer_pid')) == record['reviewer_start']:
                continue
        if record.get('attempts',0) >= max_attempts:
            record['status'] = 'FAILED'; atomic(path,record); continue
        record['status'] = 'RUNNING'; record['attempts'] = record.get('attempts',0) + 1; record['started_at'] = time.time(); atomic(path,record)
        review_dir = root / 'reviews'; review_dir.mkdir(parents=True, exist_ok=True)
        result = review_dir / (ev['id'] + '.json'); log = review_dir / (ev['id'] + '.log')
        cmd = [executable,'exec','--sandbox','read-only','--ephemeral','--skip-git-repo-check','-C',str(source),'--output-schema',str(SCHEMA),'-o',str(result),'-']
        try:
            with log.open('w', encoding='utf-8') as output:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, text=True)
                record['reviewer_pid'] = proc.pid
                record['reviewer_start'] = process_identity(proc.pid)
                atomic(path, record)
                try:
                    proc.communicate(prompt(ev,state_path), timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.communicate()
                    record['exit_code'] = proc.returncode
                    raise
            record['exit_code'] = proc.returncode
            review = json.loads(result.read_text(encoding='utf-8')) if proc.returncode == 0 else None
            if proc.returncode != 0 or not validate(review,ev): raise ValueError('Codex failed or returned invalid structured review')
            record['status'] = 'DONE'; record['review'] = str(result)
            # Safe policy: no generated command is executed and the active runner is never controlled.
            record['decision_applied'] = False; record['policy'] = 'advisory_only_active_runner_immutable'
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            record['status'] = 'FAILED' if record['attempts'] >= max_attempts else 'PENDING'
        record['ended_at'] = time.time(); atomic(path,record)
        return True
    return False


def report(root):
    lines = ['# Đánh giá Codex tự động', '', 'Các quyết định là khuyến nghị; runner đang hoạt động không bị chặn hoặc điều khiển.', '', '| Event | Job | Trạng thái | Kết luận |', '|---|---|---|---|']
    for path in sorted((root / 'events').glob('*.json')):
        r=json.loads(path.read_text()); ev=r['event']; verdict='—'
        if r.get('review'):
            try: verdict=json.loads(Path(r['review']).read_text()).get('verdict','—')
            except (OSError,ValueError): pass
        lines.append(f"| {ev['kind']} | {ev['job']} | {r['status']} | {verdict} |")
    target=root / 'CODEX_REVIEWS.vi.md'; target.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',default='outputs/overnight'); p.add_argument('--interval',type=int,default=30); p.add_argument('--once',action='store_true'); args=p.parse_args()
    run=Path(args.run_dir).resolve(); root=run / 'codex_supervisor'; root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('supervisor already running')
        atomic(root/'process.json',{'pid':os.getpid(),'started_at':time.time()})
        while True:
            try:
                state=json.loads((run/'state.json').read_text()); source=Path(state['source'])
                enqueue(root,events_from_state(state,{})); process_one(root,source,run/'state.json'); report(root)
            except (OSError,ValueError,KeyError) as exc:
                atomic(root/'last_error.json',{'error':str(exc),'time':time.time()})
            if args.once: break
            time.sleep(max(5,args.interval))

if __name__ == '__main__': main()
