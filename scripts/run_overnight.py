#!/usr/bin/env python3
"""Explicitly launched, sequential LifeMate experiments with durable state.

No training is started by import, --dry-run, --status or --stop.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
from lifemate.data.loader import load_samples
from lifemate.evaluation.heldout import select_heldout
from lifemate.compliance import assert_path_allowed

# Fail closed until all advertised downstream stages and crash recovery have
# been implemented and verified; --dry-run and --status remain usable.
RELEASE_BLOCKERS = (
    "inference ablations, independent SFT and submission stages are not implemented",
    "unattended recovery and resource protection are not verified end to end",
)


def now():
    return dt.datetime.now().astimezone()


def sha(path):
    target = assert_path_allowed(path)
    assert_path_allowed(target.resolve())
    h = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
            name = f.name
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name): os.unlink(name)


def git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def valid_checkpoint(path):
    path = assert_path_allowed(path)
    assert_path_allowed(path.resolve())
    names = ('adapter_config.json', 'adapter_model.safetensors', 'training_state.json', 'optimizer_state.pt')
    for name in names:
        assert_path_allowed((path / name).resolve())
    if not all((path / name).is_file() for name in names):
        return False
    try:
        state = json.loads((path / 'training_state.json').read_text())
        return bool(state.get('exact_resume') and state.get('progress', {}).get('planned_total_steps') == 1614)
    except (OSError, ValueError):
        return False


def protect_checkpoint(source, destination, minimum_free_bytes):
    """Copy a complete checkpoint atomically; never replace an existing copy."""
    source, destination = assert_path_allowed(source), assert_path_allowed(destination)
    assert_path_allowed(source.resolve()); assert_path_allowed(destination.resolve())
    if not valid_checkpoint(source):
        raise RuntimeError(f'incomplete or incompatible checkpoint: {source}')
    source_hash = sha(source / 'adapter_model.safetensors')
    if destination.exists():
        if not valid_checkpoint(destination) or sha(destination / 'adapter_model.safetensors') != source_hash:
            raise RuntimeError(f'existing candidate differs from source: {destination}')
        return source_hash
    artifacts = list(source.rglob('*'))
    for artifact in artifacts:
        assert_path_allowed(artifact.resolve())
    required = sum(p.stat().st_size for p in artifacts if p.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + minimum_free_bytes:
        raise RuntimeError('not enough free disk space to protect a checkpoint')
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.exists():
        raise RuntimeError(f'incomplete candidate exists: {temporary}')
    shutil.copytree(source, temporary)
    if not valid_checkpoint(temporary) or sha(temporary / 'adapter_model.safetensors') != source_hash:
        raise RuntimeError(f'candidate copy failed verification: {temporary}')
    atomic(temporary / 'candidate_identity.json', {'source': str(source), 'adapter_sha256': source_hash, 'protected_at': now().isoformat(), 'selection': 'not yet evaluated'})
    os.replace(temporary, destination)
    return source_hash


def preflight(c):
    for p in (c['python'], c['model'] + '/config.json', c['dataset'] + '/manifest.json', c['training_config'], c['inference_config']):
        target = assert_path_allowed(p)
        assert_path_allowed(target.resolve())
        if not target.is_file(): raise RuntimeError(f'missing required file: {p}')
    weights = list(Path(c['model']).glob('model*.safetensors'))
    if not weights: raise RuntimeError('base weights missing')
    for weight in weights: assert_path_allowed(weight.resolve())
    if shutil.disk_usage('.').free < int(c['minimum_free_gib']) * 1024**3: raise RuntimeError('insufficient free disk space')
    samples = load_samples('data', 'train')
    _, selection = select_heldout(samples, c['dataset'], split='train', limit=c['validation_limit'])
    return selection


def baseline(c):
    return [c['python'], 'scripts/train.py', '--config', c['training_config'], '--dataset-dir', c['dataset'], '--model', c['model'], '--output-dir', c['work_dir'] + '/baseline/checkpoints', '--save-steps', '100']


def evaluation(c, checkpoint, out):
    return [c['python'], 'scripts/evaluate.py', '--config', c['inference_config'], '--split', 'train', '--data-root', 'data', '--model', c['model'], '--model-mode', 'lora', '--checkpoint', str(checkpoint), '--heldout-dataset', c['dataset'], '--limit', str(c['validation_limit']), '--closed-loop', '--evaluation-mode', 'text_to_action', '--output', str(out)]


def running(pid):
    try: os.kill(int(pid), 0); return True
    except (ProcessLookupError, ValueError, TypeError): return False
    except PermissionError: return True


def source_clean():
    return subprocess.run(['git', 'diff', '--quiet'], check=False).returncode == 0 and subprocess.run(['git', 'diff', '--cached', '--quiet'], check=False).returncode == 0


def gpu_free():
    try:
        q = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
        used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        return int(q[0]), used
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        raise RuntimeError('cannot verify GPU availability')


def worker(c, state, state_path, deadline):
    root = Path(c['work_dir']); lock = (root / 'worker.lock').open('a+')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another overnight worker is running')
    state['worker_pid'] = os.getpid(); state['status'] = 'RUNNING'; atomic(state_path, state)
    env = dict(os.environ, PYTHONPATH='src', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    stop_path = root / 'STOP'
    def job(name, command):
        existing = state['jobs'].get(name, {})
        if existing.get('status') == 'COMPLETED': return True
        if existing.get('status') == 'RUNNING':
            if running(existing.get('pid')):
                raise RuntimeError(f'previous job {name} is still running; refusing a second GPU job')
            if name != 'baseline':
                raise RuntimeError(f'previous {name} terminated unexpectedly; inspect artifacts before retry')
        if existing.get('status') in ('FAILED', 'INTERRUPTED') and name != 'baseline':
            raise RuntimeError(f'previous {name} did not complete; refusing automatic overwrite')
        if stop_path.exists() or now() >= deadline:
            state['status'] = 'INTERRUPTED'; atomic(state_path, state); return False
        if name == 'baseline' and deadline - now() < dt.timedelta(minutes=int(c['minimum_start_minutes'])):
            state['status'] = 'INTERRUPTED'; atomic(state_path, state); return False
        free, busy = gpu_free()
        if busy or free < 12000: raise RuntimeError(f'GPU busy or insufficient VRAM: free={free} MiB processes={busy!r}')
        log = root / 'logs' / f'{name}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        entry = {'status':'RUNNING', 'command':command, 'started':now().isoformat(), 'log':str(log), 'git_sha':state['git_sha']}
        state['jobs'][name] = entry; atomic(state_path, state)
        protected = set()
        def capture_checkpoints():
            for source in sorted((root / 'baseline/checkpoints/lora').glob('step-*')):
                if source.name not in protected and valid_checkpoint(source):
                    protect_checkpoint(source, root / 'candidates' / source.name, int(c['minimum_free_gib']) * 1024**3)
                    protected.add(source.name)
        with log.open('a', encoding='utf-8') as output:
            proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            entry['pid'] = proc.pid; atomic(state_path, state)
            stop_sent = False
            capture_error = None
            while proc.poll() is None:
                if name == 'baseline' and capture_error is None:
                    try:
                        capture_checkpoints()
                    except Exception as exc:
                        capture_error = f'{type(exc).__name__}: {exc}'
                if not stop_sent and (capture_error or stop_path.exists() or now() >= deadline):
                    proc.send_signal(signal.SIGINT); stop_sent = True
                time.sleep(2)
            if name == 'baseline' and capture_error is None:
                try:
                    capture_checkpoints()
                except Exception as exc:
                    capture_error = f'{type(exc).__name__}: {exc}'
        if capture_error:
            entry.update(exit_code=proc.returncode, ended=now().isoformat(), status='FAILED', error=capture_error)
            state['status'] = 'FAILED'; atomic(state_path, state); return False
        entry.update(exit_code=proc.returncode, ended=now().isoformat(), status='INTERRUPTED' if stop_sent else ('COMPLETED' if proc.returncode == 0 else 'FAILED'))
        atomic(state_path, state)
        if entry['status'] != 'COMPLETED': state['status'] = entry['status']; atomic(state_path, state); return False
        return True
    try:
        preflight(c)
        if state['git_sha'] != git_sha() or not source_clean(): raise RuntimeError('git HEAD changed or tracked source has uncommitted changes')
        if state['config'] != c: raise RuntimeError('runner configuration changed since run was created')
        for name, expected in state['dataset_sha256'].items():
            if sha(Path(c['dataset']) / name) != expected: raise RuntimeError(f'dataset changed: {name}')
        train_dir = root / 'baseline/checkpoints/lora'
        cmd = baseline(c)
        if state['jobs'].get('baseline', {}).get('status') in ('RUNNING', 'INTERRUPTED'):
            candidates = sorted([p for p in train_dir.glob('*-*') if valid_checkpoint(p)], key=lambda p:int(p.name.rsplit('-',1)[-1]))
            if not candidates: raise RuntimeError('interrupted baseline has no verified checkpoint; refusing restart')
            cmd += ['--resume', str(candidates[-1])]
        elif train_dir.exists() and state['jobs'].get('baseline', {}).get('status') != 'COMPLETED':
            raise RuntimeError('checkpoint directory already exists; refusing fresh start')
        if not job('baseline', cmd): return
        candidates = sorted([p for p in (root / 'candidates').glob('step-*') if valid_checkpoint(p)], key=lambda p:int(p.name.rsplit('-',1)[-1]))[-2:]
        if not candidates: raise RuntimeError('no protected complete baseline checkpoints found')
        scores = []
        for dest in candidates:
            identity = json.loads((dest / 'candidate_identity.json').read_text())
            source = identity['source']
            if sha(dest / 'adapter_model.safetensors') != identity['adapter_sha256']:
                raise RuntimeError(f'protected candidate changed: {dest}')
            out = root / 'evaluation' / dest.name
            if not job('eval-' + dest.name, evaluation(c, dest, out)): return
            result = json.loads((out / 'metrics.json').read_text())
            closed = result.get('closed_loop') or {}
            score = (closed.get('decision_metrics') or {}).get('tool_call_exact_match') or {}
            if closed.get('n_failures') or not score.get('available') or score.get('value') is None or not closed.get('heldout_selection'):
                raise RuntimeError(f'held-out evaluation incomplete or decision metric unavailable for {dest}')
            scores.append({'checkpoint':str(dest), 'source':str(source), 'sha256':sha(dest/'adapter_model.safetensors'), 'score':score['value'], 'metric':'local_tool_call_exact_match', 'evaluation':str(out), 'heldout':closed['heldout_selection']})
        scores.sort(key=lambda x:(x['score'], int(Path(x['checkpoint']).name.rsplit('-',1)[-1])), reverse=True)
        if len({x['heldout']['ids_sha256'] for x in scores}) != 1: raise RuntimeError('candidate held-out samples differ')
        state['candidates'] = scores; state['selected'] = scores[0]; atomic(state_path, state)
        # Remaining research stages are gated until execution evidence and submission
        # acceptance are established; they are never marked completed automatically.
        state['jobs']['inference_ablations'] = {'status':'BLOCKED', 'reason':'requires verified per-feature execution and checkpoint-aware ablation end-to-end'}
        state['jobs']['additional_sft'] = {'status':'BLOCKED', 'reason':'independent augmented datasets and resume safety not yet validated'}
        state['jobs']['submission'] = {'status':'BLOCKED', 'reason':'competition container and full-split runtime not validated'}
        state['status'] = 'BLOCKED'; atomic(state_path, state)
    except Exception as exc:
        state['status'] = 'FAILED'; state['error'] = f'{type(exc).__name__}: {exc}'; atomic(state_path, state)
        raise
    finally:
        state['worker_pid'] = None; atomic(state_path, state)


def main():
    p = argparse.ArgumentParser()
    action = p.add_mutually_exclusive_group(required=True)
    for name in ('dry-run','start','status','resume','stop'): action.add_argument('--'+name, action='store_true')
    action.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--config', default='configs/overnight_experiments.yaml')
    p.add_argument('--stop-at', help='deadline ISO 8601 with timezone, e.g. 2026-11-16T07:00:00+07:00')
    args = p.parse_args()
    config_path = assert_path_allowed(args.config); assert_path_allowed(config_path.resolve())
    c = json.loads(config_path.read_text()); root = assert_path_allowed(c['work_dir']); assert_path_allowed(root.resolve()); state_path = root / 'state.json'
    if args.status:
        print(state_path.read_text() if state_path.exists() else '{"status":"NOT_STARTED"}'); return 0
    if args.stop:
        if not state_path.exists(): print('no run exists', file=sys.stderr); return 2
        root.mkdir(parents=True, exist_ok=True); (root / 'STOP').write_text('stop\n'); print('stop requested'); return 0
    if args.dry_run:
        selection = preflight(c)
        print(json.dumps({'status':'DRY_RUN', 'baseline':baseline(c), 'evaluation_example':evaluation(c, '<protected-adapter>', root/'evaluation/example'), 'heldout_sha256':selection['ids_sha256'], 'heldout_count':len(selection['ids']),  'release_ready':False, 'release_blockers':RELEASE_BLOCKERS, 'blocked_stages':['inference_ablations','additional_sft','submission'], 'no_training_started':True}, indent=2, ensure_ascii=False)); return 0
    if args.worker:
        if RELEASE_BLOCKERS:
            raise RuntimeError("overnight worker is not released: " + "; ".join(RELEASE_BLOCKERS))
        state = json.loads(state_path.read_text()); worker(c, state, state_path, dt.datetime.fromisoformat(state['stop_at'])); return 0
    if args.start or args.resume:
        raise RuntimeError("overnight runner is not released: " + "; ".join(RELEASE_BLOCKERS))
    if args.start:
        if state_path.exists(): raise RuntimeError('run already exists; inspect --status, never overwrite')
        if not args.stop_at: raise RuntimeError('--start requires --stop-at with timezone')
        deadline = dt.datetime.fromisoformat(args.stop_at)
        if deadline.tzinfo is None or deadline <= now() + dt.timedelta(minutes=int(c['minimum_start_minutes'])): raise RuntimeError('deadline must include timezone and allow minimum start time')
        selection = preflight(c)
        if not source_clean(): raise RuntimeError('tracked source has uncommitted changes; run from a clean checkout')
        root.mkdir(parents=True, exist_ok=True)
        state = {'status':'PENDING','created':now().isoformat(),'stop_at':deadline.isoformat(),'git_sha':git_sha(),'dataset_sha256':{name:sha(Path(c['dataset'])/name) for name in ('manifest.json','train.jsonl','val.jsonl')},'heldout':selection,'config':c,'jobs':{}}
        atomic(state_path, state)
    else:
        if not state_path.exists(): raise RuntimeError('no state to resume')
        state = json.loads(state_path.read_text())
        if running(state.get('worker_pid')): raise RuntimeError('worker is still running')
        if state.get('status') not in ('INTERRUPTED',): raise RuntimeError(f'refusing automatic retry from {state.get("status")}')
        if args.stop_at:
            deadline=dt.datetime.fromisoformat(args.stop_at)
            if deadline.tzinfo is None or deadline <= now(): raise RuntimeError('invalid stop time')
            state['stop_at']=deadline.isoformat(); atomic(state_path,state)
        elif dt.datetime.fromisoformat(state['stop_at']) <= now(): raise RuntimeError('provide a new --stop-at to resume')
        (root/'STOP').unlink(missing_ok=True)
    log = (root/'worker.log').open('a')
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--config', args.config, '--worker'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'worker started pid={proc.pid}; status: {state_path}')
    return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); sys.exit(2)
