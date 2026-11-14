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
import math
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

# Optional research stages remain explicitly gated; baseline has its own release.


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


def expected_horizon(dataset, config):
    """Use the trainer's actual retained-example and accumulation formula."""
    from lifemate.config import load_config
    from lifemate.training.trainer import optimizer_steps_per_epoch
    manifest = json.loads((Path(dataset) / 'manifest.json').read_text())
    cfg = load_config(config)
    fit = manifest.get('fit') or {}
    if fit.get('method') != 'tokenizer' or fit.get('max_seq_length') != cfg.training.max_seq_length:
        raise RuntimeError('dataset tokenizer fit does not match training sequence length')
    n = int(fit.get('n_fit') or 0)
    if n <= 0:
        raise RuntimeError('dataset has no verified retained training examples')
    per_epoch = optimizer_steps_per_epoch(n, cfg.training.per_device_train_batch_size, cfg.training.gradient_accumulation_steps)
    return max(1, math.ceil(per_epoch * cfg.training.num_train_epochs))


def valid_checkpoint(path, horizon=None):
    path = assert_path_allowed(path)
    assert_path_allowed(path.resolve())
    names = ('adapter_config.json', 'adapter_model.safetensors', 'training_state.json', 'optimizer_state.pt')
    for name in names:
        assert_path_allowed((path / name).resolve())
    if not all((path / name).is_file() for name in names):
        return False
    try:
        state = json.loads((path / 'training_state.json').read_text())
        recorded = state.get('progress', {}).get('planned_total_steps')
        return bool(state.get('exact_resume') and isinstance(recorded, int) and recorded > 0 and (horizon is None or recorded == horizon))
    except (OSError, ValueError):
        return False


def valid_candidate(path):
    path = assert_path_allowed(path)
    assert_path_allowed(path.resolve())
    return all((path / name).is_file() for name in ('adapter_config.json', 'adapter_model.safetensors', 'candidate_identity.json'))


def protect_checkpoint(source, destination, minimum_free_bytes, horizon=None):
    """Protect inference adapter only; exact-resume checkpoints remain in training dir."""
    source, destination = assert_path_allowed(source), assert_path_allowed(destination)
    assert_path_allowed(source.resolve()); assert_path_allowed(destination.resolve())
    if not valid_checkpoint(source, horizon):
        raise RuntimeError(f'incomplete or incompatible checkpoint: {source}')
    source_hash = sha(source / 'adapter_model.safetensors')
    if destination.exists():
        if not valid_candidate(destination) or sha(destination / 'adapter_model.safetensors') != source_hash:
            raise RuntimeError(f'existing candidate differs from source: {destination}')
        return source_hash
    names = ('adapter_config.json', 'adapter_model.safetensors', 'lifemate_meta.json',
             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja')
    files = [source / name for name in names if (source / name).is_file()]
    for artifact in files: assert_path_allowed(artifact.resolve())
    required = sum(p.stat().st_size for p in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + minimum_free_bytes:
        raise RuntimeError('not enough free disk space to protect a checkpoint')
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.exists(): raise RuntimeError(f'incomplete candidate exists: {temporary}')
    temporary.mkdir()
    for artifact in files: shutil.copy2(artifact, temporary / artifact.name)
    if sha(temporary / 'adapter_model.safetensors') != source_hash:
        raise RuntimeError(f'candidate copy failed verification: {temporary}')
    atomic(temporary / 'candidate_identity.json', {'source': str(source), 'adapter_sha256': source_hash, 'protected_at': now().isoformat(), 'artifact_type':'adapter_only', 'selection':'not yet evaluated'})
    os.replace(temporary, destination)
    return source_hash


def prune_candidates(root, keep=3):
    """Only remove older runner-owned, unevaluated adapters, retaining any best."""
    root = Path(root)
    best_file = root / 'best.json'
    best = json.loads(best_file.read_text()).get('checkpoint') if best_file.exists() else None
    candidates = sorted((p for p in root.glob('step-*') if valid_candidate(p)), key=lambda p:int(p.name.rsplit('-',1)[-1]), reverse=True)
    retained = set(candidates[:max(1, keep)])
    if best: retained.add(Path(best))
    for path in candidates:
        if path not in retained and len(candidates) > 1:
            shutil.rmtree(path)




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
    selection['expected_horizon'] = expected_horizon(c['dataset'], c['training_config'])
    from lifemate.config import load_config
    from lifemate.features.config import enabled_features
    cfg = load_config(c['training_config'])
    actual = (cfg.training.tuning_mode, cfg.training.precision, cfg.lora.r, cfg.lora.alpha,
              cfg.training.learning_rate, cfg.training.max_seq_length,
              cfg.training.per_device_train_batch_size, cfg.training.gradient_accumulation_steps,
              cfg.training.num_train_epochs, cfg.training.gradient_checkpointing)
    expected = ('lora', 'bfloat16', 16, 32, 2e-4, 8192, 1, 8, 3, True)
    if actual != expected or enabled_features(cfg.features):
        raise RuntimeError('baseline config or experimental feature flags differ from validated baseline')
    try:
        mem = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    except (OSError, StopIteration, ValueError, IndexError) as exc:
        raise RuntimeError('cannot verify available host RAM') from exc
    if mem < 8 * 1024**3: raise RuntimeError('less than 8 GiB host RAM available')
    selection['host_ram_available_bytes'] = mem
    return selection


def baseline(c):
    return [c['python'], 'scripts/train.py', '--config', c['training_config'], '--dataset-dir', c['dataset'], '--model', c['model'], '--output-dir', c['work_dir'] + '/baseline/checkpoints', '--save-steps', '100']


def evaluation(c, checkpoint, out):
    return [c['python'], 'scripts/evaluate.py', '--config', c['inference_config'], '--split', 'train', '--data-root', 'data', '--model', c['model'], '--model-mode', 'lora', '--checkpoint', str(checkpoint), '--heldout-dataset', c['dataset'], '--limit', str(c['validation_limit']), '--closed-loop', '--evaluation-mode', 'text_to_action', '--output', str(out)]


def submission(c, checkpoint, output):
    return [c['python'], 'scripts/infer.py', '--config', c['inference_config'], '--split', 'public_test',
            '--data-root', 'data', '--model', c['model'], '--model-mode', 'lora',
            '--checkpoint', str(checkpoint), '--asr', 'gipformer', '--output', str(output)]


def audio_evaluation(c, checkpoint, output, feature=None):
    command = [c['python'], 'scripts/evaluate.py', '--config', c['inference_config'], '--split', 'train',
               '--data-root', 'data', '--model', c['model'], '--model-mode', 'lora', '--checkpoint', str(checkpoint),
               '--heldout-dataset', c['dataset'], '--limit', str(c['validation_limit']), '--closed-loop',
               '--evaluation-mode', 'audio_to_action', '--end-to-end-runtime', '--output', str(output)]
    if feature: command += ['--enable-feature', feature]
    return command


def public_sample_ids():
    return [str(s.id) for s in load_samples('data', 'public_test', with_labels=False)]


def verify_submission(path):
    from lifemate.evaluation.submission_validator import validate_submission
    ids = public_sample_ids()
    if not ids: raise RuntimeError('public test split is empty')
    result = validate_submission(path, ids, require_state_hash=True, calls_source='env', require_order=True)
    if not result.get('ok'): raise RuntimeError(f'submission validation failed: {result.get("errors")}')
    return result


VARIANT_FEATURES = {'slot_masking':'training.slot_masking',
                    'tool_dropout':'training.tool_dropout', 'noisy_asr':'training.noisy_asr'}


def augmentation_command(c, kind, directory):
    if kind not in VARIANT_FEATURES: raise RuntimeError(f'unsupported SFT variant {kind}')
    cmd = [c['python'], 'scripts/prepare_dataset.py', '--config', c['training_config'],
           '--data-root', 'data', '--split', 'train', '--output', str(directory),
           '--tokenizer', c['model'], '--max-seq-length', '8192', '--enable-feature', VARIANT_FEATURES[kind]]
    if kind == 'noisy_asr': cmd += ['--noisy-asr-transcripts', str(c['noisy_asr_transcripts'])]
    return cmd


def variant_training_command(c, kind, dataset, output):
    return [c['python'], 'scripts/train.py', '--config', c['training_config'], '--dataset-dir', str(dataset),
            '--model', c['model'], '--output-dir', str(output), '--save-steps', '100',
            '--enable-feature', VARIANT_FEATURES[kind]]


def verify_noisy_asset(c, train_ids):
    """Require explicit ASR provenance, hash and train-only IDs before use."""
    from lifemate.data.asr_variants import load_transcripts
    transcript = c.get('noisy_asr_transcripts'); provenance = c.get('noisy_asr_provenance')
    if not transcript or not provenance or not Path(transcript).is_file() or not Path(provenance).is_file():
        return False
    meta = json.loads(assert_path_allowed(provenance).read_text())
    if meta.get('split') != 'train' or meta.get('backend') not in ('gipformer', 'whisper') or meta.get('sha256') != sha(transcript):
        raise RuntimeError('noisy ASR provenance is invalid or uses an unverified backend')
    records = load_transcripts(transcript)
    if not records or set(records) - set(train_ids): raise RuntimeError('noisy ASR transcripts contain unknown/non-train IDs')
    if not any(record.hypotheses for record in records.values()): raise RuntimeError('noisy ASR transcripts are empty')
    return True


def effective_variant_count(dataset, kind, model, *, tokenizer=None, build=None):
    """Count augmented examples that survive the real training tokenizer fit."""
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if build is None:
        from lifemate.training.collator import build_supervised_example
        build = build_supervised_example
    root = Path(dataset)
    for raw in (root / 'val.jsonl').read_text(encoding='utf-8').splitlines():
        if raw.strip() and (json.loads(raw).get('meta') or {}).get('augmentation'):
            raise RuntimeError('validation records contain augmentation')
    observed = kept = 0
    with (root / 'train.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip(): continue
            record = json.loads(line)
            if (record.get('meta') or {}).get('augmentation') != kind: continue
            observed += 1
            example = build(tokenizer, record.get('messages') or [], str(record.get('target') or ''), 8192)
            kept += bool(example.get('kept'))
    return {'observed':observed, 'retained':kept}


def score_evaluation(payload, expected_ids, samples):
    """Local exact-call score: model stop/failure is zero; infrastructure errors abort."""
    from lifemate.evaluation.metrics import tool_call_exact_match
    closed = payload.get('closed_loop') or {}
    rows = closed.get('rows')
    selection = closed.get('heldout_selection') or {}
    if not isinstance(rows, list) or selection.get('ids') != expected_ids:
        raise RuntimeError('held-out IDs missing or inconsistent')
    if len(rows) != len(expected_ids) or [r.get('id') for r in rows] != expected_ids:
        raise RuntimeError('evaluation rows missing, reordered, or duplicated')
    byid = {str(s.id): s for s in samples}
    hits, model_failures = 0, 0
    for row in rows:
        sid = row['id']; sample = byid.get(sid)
        if sample is None or getattr(sample, 'label', None) is None:
            raise RuntimeError('missing held-out gold sample')
        status = str(row.get('status') or '')
        if status not in ('finished', 'ok', 'max_steps', 'time_budget', 'repeated_failure', 'invalid_action') or not row.get('state_hash') or not isinstance(row.get('tool_calls'), list):
            raise RuntimeError(f'infrastructure failure or corrupt result for {sid}')
        if status not in ('finished', 'ok'):
            model_failures += 1
            continue
        gold = [c.to_dict() if hasattr(c, 'to_dict') else dict(c) for c in (sample.label.tool_calls or [])]
        hits += bool(tool_call_exact_match(row['tool_calls'], gold))
    if not rows: raise RuntimeError('empty evaluation')
    return {'value':hits / len(rows), 'n':len(rows), 'model_failures':model_failures,
            'metric':'local_tool_call_exact_match', 'official_score':'UNAVAILABLE'}


def running(pid):
    try: os.kill(int(pid), 0); return True
    except (ProcessLookupError, ValueError, TypeError): return False
    except PermissionError: return True


def process_start(pid):
    """Linux process birth tick prevents PID reuse from looking like our child."""
    try:
        return int(Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def same_process(pid, start):
    if not pid:
        return False
    # If procfs was unavailable when the child was recorded, prefer refusing a
    # duplicate over assuming an unidentifiable process has exited.
    return running(pid) if start is None else process_start(pid) == start


def prepare_worktree(path, revision):
    """Checkout committed source separately; link only the existing runtime assets."""
    path = assert_path_allowed(path)
    assert_path_allowed(path.resolve())
    if path.exists(): raise RuntimeError(f'worktree already exists: {path}')
    subprocess.run(['git', 'worktree', 'add', '--detach', str(path.resolve()), revision], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for name in ('data', 'models', 'outputs'):
        target = ROOT / name
        link = path / name
        if link.exists() or link.is_symlink(): raise RuntimeError(f'worktree asset path exists: {link}')
        link.symlink_to(target.resolve(), target_is_directory=True)
    actual = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()
    if actual != revision: raise RuntimeError('worktree commit mismatch')
    return path


def source_clean():
    return subprocess.run(['git', 'diff', '--quiet'], check=False).returncode == 0 and subprocess.run(['git', 'diff', '--cached', '--quiet'], check=False).returncode == 0


def gpu_free():
    try:
        q = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
        used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        return int(q[0]), used
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        raise RuntimeError('cannot verify GPU availability')


def job_telemetry(pid, metrics_path, previous, horizon):
    """Measure only observed values; ETA is absent until two step observations."""
    result = dict(previous or {})
    try:
        lines = Path(metrics_path).read_text(encoding='utf-8').splitlines()
        records = [json.loads(line) for line in lines[-50:] if line.strip()]
        metric = next((item for item in reversed(records) if isinstance(item.get('step'), int) and 'train_loss' in item), None)
        if metric:
            stamp = time.time()
            old_step, old_time = result.get('step'), result.get('step_observed_at')
            result.update(step=metric['step'], train_loss=metric['train_loss'], learning_rate=metric.get('learning_rate'))
            if old_step is None or metric['step'] != old_step:
                if old_step is not None and metric['step'] > old_step and old_time and stamp > old_time:
                    rate = (metric['step'] - old_step) / (stamp - old_time)
                    result.update(steps_per_second=rate, estimated_seconds_remaining=max(0, horizon - metric['step']) / rate)
                result['step_observed_at'] = stamp
    except (OSError, ValueError, TypeError):
        pass
    try:
        for line in Path(f'/proc/{int(pid)}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_kib'] = int(line.split()[1]); break
    except (OSError, ValueError, IndexError):
        pass
    try:
        output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            process, memory = line.split(',')
            if int(process.strip()) == int(pid): result['gpu_memory_mib'] = int(memory.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


def write_status_report(root, state):
    """A durable, human-readable report, including incomplete/failed work."""
    root = Path(root)
    lines = ['# Kết quả chạy LifeMate qua đêm', '',
             f"Trạng thái: **{state.get('status', 'UNKNOWN')}**.",
             f"Git SHA: `{state.get('git_sha', 'UNAVAILABLE')}`.", '']
    selected = state.get('selected')
    if selected:
        lines += [f"Adapter đã chọn: `{selected.get('checkpoint')}`.",
                  f"SHA-256: `{selected.get('sha256')}`.",
                  f"Điểm diagnostic local exact-call: {selected.get('score')} (không phải điểm chính thức).", '']
    else:
        lines += ['Chưa có candidate được chọn bằng held-out.', '']
    lines += ['| Công việc | Trạng thái | Exit | Log / lý do |', '|---|---|---:|---|']
    for name, item in sorted(state.get('jobs', {}).items()):
        detail = str(item.get('log') or item.get('reason') or item.get('error') or '').replace('|', '/')
        lines.append(f"| {name} | {item.get('status', 'UNKNOWN')} | {item.get('exit_code', '—')} | {detail} |")
    lines += ['', 'Official score: UNAVAILABLE. Docker và submission chỉ hợp lệ khi có bằng chứng thực tế.', '']
    path = root / 'FINAL_RESULTS.vi.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
            name = f.name; f.write('\n'.join(lines)); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name): os.unlink(name)
    return str(path)


def worker(c, state, state_path, deadline):
    root = Path(c['work_dir']); lock = (root / 'worker.lock').open('a+')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another overnight worker is running')
    gpu_lock_path = ROOT / 'outputs' / '.lifemate_gpu.lock'
    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock = gpu_lock_path.open('a+')
    try: fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another LifeMate GPU worker is running')
    state['worker_pid'] = os.getpid(); state['worker_last_pid'] = os.getpid(); state['worker_start'] = process_start(os.getpid()); state['status'] = 'RUNNING'; atomic(state_path, state)
    horizon = int(state['expected_horizon'])
    env = {key: value for key, value in os.environ.items() if not key.startswith('LIFEMATE_FEATURE_')}
    env.update(PYTHONPATH='src', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', LIFEMATE_LOCAL_METRICS='1', WANDB_MODE='disabled')
    stop_path = root / 'STOP'
    def job(name, command, *, training=None):
        existing = state['jobs'].get(name, {})
        if existing.get('status') == 'COMPLETED': return True
        if existing.get('status') == 'RUNNING':
            if same_process(existing.get('pid'), existing.get('process_start')):
                raise RuntimeError(f'previous job {name} is still running; refusing a second GPU job')
            if training is None and not name.startswith('submission'):
                raise RuntimeError(f'previous {name} terminated unexpectedly; inspect artifacts before retry')
        if existing.get('status') == 'FAILED':
            raise RuntimeError(f'previous {name} failed; refusing automatic retry')
        if existing.get('status') == 'INTERRUPTED' and training is None and not name.startswith('submission'):
            raise RuntimeError(f'previous {name} was interrupted; refusing automatic overwrite')
        if name.startswith('submission') and existing.get('status') in ('RUNNING', 'INTERRUPTED'):
            output = Path(command[command.index('--output') + 1])
            if output.exists():
                verify_submission(output)
                existing.update(status='COMPLETED', recovered=True, ended=now().isoformat())
                atomic(state_path, state); return True
            if Path(str(output) + '.progress.json').exists(): command = [*command, '--resume']
        if stop_path.exists() or now() >= deadline:
            state['status'] = 'INTERRUPTED'; atomic(state_path, state); return False
        if training is not None and deadline - now() < dt.timedelta(minutes=int(c['minimum_start_minutes'])):
            state['status'] = 'INTERRUPTED'; atomic(state_path, state); return False
        if shutil.disk_usage(root).free < int(c['minimum_free_gib']) * 1024**3:
            raise RuntimeError('insufficient disk before job')
        free, busy = gpu_free()
        if busy or free < 12000: raise RuntimeError(f'GPU busy or insufficient VRAM: free={free} MiB processes={busy!r}')
        log = root / 'logs' / f'{name}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        attempts = list(existing.get('attempts') or [])
        if existing:
            attempts.append({k: v for k, v in existing.items() if k != 'attempts'})
        entry = {'status':'RUNNING', 'command':command, 'started':now().isoformat(), 'log':str(log), 'git_sha':state['git_sha'], 'attempts':attempts}
        state['jobs'][name] = entry; atomic(state_path, state)
        protected = set()
        def capture_checkpoints():
            if training is None: return
            for source in sorted((Path(training['directory']) / 'lora').glob('step-*')):
                if source.name not in protected and valid_checkpoint(source, training['horizon']):
                    protect_checkpoint(source, Path(training['candidates']) / source.name, int(c['minimum_free_gib']) * 1024**3, training['horizon'])
                    prune_candidates(training['candidates'])
                    protected.add(source.name)
        with log.open('a', encoding='utf-8') as output:
            try:
                proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            except OSError as exc:
                entry.update(status='FAILED', ended=now().isoformat(), error=f'{type(exc).__name__}: {exc}')
                state['status'] = 'FAILED'; atomic(state_path, state)
                raise
            entry['pid'] = proc.pid; entry['process_start'] = process_start(proc.pid); atomic(state_path, state)
            stop_sent = False
            capture_error = None
            last_monitor = 0.0
            while proc.poll() is None:
                if training is not None and time.monotonic() - last_monitor >= 10:
                    metrics = Path(training['directory']) / 'lora/metrics.jsonl'
                    entry['telemetry'] = job_telemetry(proc.pid, metrics, entry.get('telemetry'), training['horizon'])
                    entry['metrics_path'] = str(metrics)
                    entry['protected_checkpoints'] = sorted(protected)
                    atomic(state_path, state)
                    last_monitor = time.monotonic()
                if training is not None and capture_error is None:
                    try:
                        capture_checkpoints()
                    except Exception as exc:
                        capture_error = f'{type(exc).__name__}: {exc}'
                if not stop_sent and (capture_error or stop_path.exists() or now() >= deadline):
                    try:
                        proc.send_signal(signal.SIGINT); stop_sent = True
                    except ProcessLookupError:
                        pass  # The child finished between poll and signal.
                time.sleep(2)
            if training is not None and capture_error is None:
                try:
                    capture_checkpoints()
                except Exception as exc:
                    capture_error = f'{type(exc).__name__}: {exc}'
        if capture_error:
            entry.update(exit_code=proc.returncode, ended=now().isoformat(), status='FAILED', error=capture_error)
            state['status'] = 'FAILED'; atomic(state_path, state); return False
        ended = now()
        entry.update(exit_code=proc.returncode, ended=ended.isoformat(),
                     elapsed_seconds=max(0.0, (ended - dt.datetime.fromisoformat(entry['started'])).total_seconds()),
                     status='INTERRUPTED' if stop_sent else ('COMPLETED' if proc.returncode == 0 else 'FAILED'))
        atomic(state_path, state)
        if entry['status'] != 'COMPLETED': state['status'] = entry['status']; atomic(state_path, state); return False
        return True
    try:
        preflight(c)
        if state['git_sha'] != git_sha() or not source_clean(): raise RuntimeError('git HEAD changed or tracked source has uncommitted changes')
        if state['config'] != c: raise RuntimeError('runner configuration changed since run was created')
        for name, expected in state['model_sha256'].items():
            if sha(Path(c['model']) / name) != expected: raise RuntimeError(f'base model changed: {name}')
        for name, expected in state['dataset_sha256'].items():
            if sha(Path(c['dataset']) / name) != expected: raise RuntimeError(f'dataset changed: {name}')
        train_dir = root / 'baseline/checkpoints/lora'
        cmd = baseline(c)
        if state['jobs'].get('baseline', {}).get('status') in ('RUNNING', 'INTERRUPTED'):
            candidates = sorted([p for p in train_dir.glob('*-*') if valid_checkpoint(p, horizon)], key=lambda p:int(p.name.rsplit('-',1)[-1]))
            if not candidates: raise RuntimeError('interrupted baseline has no verified checkpoint; refusing restart')
            cmd += ['--resume', str(candidates[-1])]
        elif train_dir.exists() and state['jobs'].get('baseline', {}).get('status') != 'COMPLETED':
            raise RuntimeError('checkpoint directory already exists; refusing fresh start')
        baseline_training = {'directory':str(root / 'baseline/checkpoints'), 'candidates':str(root / 'candidates'), 'horizon':horizon}
        if not job('baseline', cmd, training=baseline_training): return
        candidates = sorted([p for p in (root / 'candidates').glob('step-*') if valid_candidate(p)], key=lambda p:int(p.name.rsplit('-',1)[-1]))[-2:]
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
            heldout = (result.get('closed_loop') or {}).get('heldout_selection') or {}
            score = score_evaluation(result, state['heldout']['ids'], load_samples('data','train'))
            scores.append({'checkpoint':str(dest), 'source':str(source), 'sha256':sha(dest/'adapter_model.safetensors'), 'score':score['value'], 'score_details':score, 'evaluation':str(out), 'heldout':heldout})
        scores.sort(key=lambda x:(x['score'], int(Path(x['checkpoint']).name.rsplit('-',1)[-1])), reverse=True)
        if len({x['heldout']['ids_sha256'] for x in scores}) != 1: raise RuntimeError('candidate held-out samples differ')
        state['candidates'] = scores; state['selected'] = scores[0]; atomic(root / 'candidates' / 'best.json', scores[0]); atomic(state_path, state)
        # A model-only measurement is a lower bound, never a runtime-compliance claim.
        chosen = scores[0]
        selected_payload = json.loads((Path(chosen['evaluation']) / 'metrics.json').read_text())
        latency = (((selected_payload.get('closed_loop') or {}).get('latency') or {}).get('total') or {}).get('mean')
        try:
            n_public = len(public_sample_ids())
        except Exception:
            n_public = 0
        estimated_lower_bound = float(latency) * n_public if isinstance(latency, (float, int)) and latency > 0 and n_public else None
        remaining = (deadline - now()).total_seconds()
        output = root / 'submissions' / ('candidate-' + chosen['sha256'][:12] + '.jsonl')
        if estimated_lower_bound is None:
            state['jobs']['submission'] = {'status':'SKIPPED', 'reason':'measured runtime lower bound unavailable'}
        elif remaining < estimated_lower_bound * 1.25:
            state['jobs']['submission'] = {'status':'SKIPPED', 'reason':'insufficient remaining time even for measured model-only lower bound', 'lower_bound_seconds':estimated_lower_bound}
        else:
            if not job('submission', submission(c, chosen['checkpoint'], output)): return
            state['jobs']['submission']['validation'] = verify_submission(output)
            state['jobs']['submission']['output'] = str(output)
            runtime_path = Path(str(output) + '.runtime.json')
            state['jobs']['submission']['runtime'] = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {'available':False}
        atomic(state_path, state)
        baseline_audio_score = None
        features = c.get('inference_features', ['prompt.itn', 'postprocessing.enhanced_schema_guardrail',
                                                'decoding.constrained', 'audio.vad', 'audio.rms_normalization'])
        allowed_features = {'prompt.itn', 'postprocessing.enhanced_schema_guardrail', 'decoding.constrained', 'audio.vad', 'audio.rms_normalization'}
        if not isinstance(features, list) or any(feature not in allowed_features for feature in features):
            raise RuntimeError('invalid inference feature list')
        for feature in [None, *features]:
            if stop_path.exists() or now() >= deadline:
                state['status'] = 'INTERRUPTED'; atomic(state_path, state); return
            name = 'audio-baseline' if feature is None else 'audio-' + feature.replace('.', '-')
            remaining = (deadline - now()).total_seconds()
            measured = float(latency) * len(state['heldout']['ids']) if isinstance(latency, (int, float)) and latency > 0 else None
            if measured is None or remaining < measured * 1.25:
                state['jobs'][name] = {'status':'SKIPPED', 'reason':'time budget insufficient or measured lower bound unavailable'}
                continue
            out = root / 'evaluation' / name
            if not job(name, audio_evaluation(c, chosen['checkpoint'], out, feature)): return
            payload = json.loads((out / 'metrics.json').read_text())
            score = score_evaluation(payload, state['heldout']['ids'], load_samples('data','train'))
            counts = (payload.get('feature_execution') or {}).get(feature) if feature else None
            observed = bool(counts and counts.get('n_requested') and counts.get('n_executed') == counts['n_requested'])
            state['jobs'][name].update(score=score, feature=feature, feature_execution=counts,
                                       feature_status=('BASELINE' if feature is None else 'EXECUTED' if observed else 'NOT_OBSERVED'))
            if feature is None: baseline_audio_score = score['value']
            if feature and observed and baseline_audio_score is not None:
                state['jobs'][name]['diagnostic_delta'] = score['value'] - baseline_audio_score
            atomic(state_path, state)
        variants = c.get('sft_variants', ['slot_masking', 'tool_dropout', 'noisy_asr'])
        if not isinstance(variants, list) or any(kind not in VARIANT_FEATURES for kind in variants):
            raise RuntimeError('invalid SFT variant list')
        baseline_entry = state['jobs']['baseline']
        baseline_seconds = float(baseline_entry.get('elapsed_seconds') or 0) + sum(float(a.get('elapsed_seconds') or 0) for a in baseline_entry.get('attempts', []))
        baseline_train_ids = None
        for kind in variants:
            marker = 'variant-' + kind
            if stop_path.exists() or now() >= deadline:
                state['status'] = 'INTERRUPTED'; atomic(state_path, state); return
            if kind == 'noisy_asr' and c.get('noisy_asr_transcripts') and c.get('noisy_asr_provenance'):
                with (Path(c['dataset']) / 'train.jsonl').open(encoding='utf-8') as handle:
                    baseline_train_ids = {json.loads(line)['sample_id'] for line in handle if line.strip()}
            if kind == 'noisy_asr' and not verify_noisy_asset(c, baseline_train_ids or set()):
                state['jobs'][marker] = {'status':'SKIPPED_MISSING_ASSET', 'reason':'verified train-only ASR transcripts and provenance are required'}
                atomic(state_path, state); continue
            remaining = (deadline - now()).total_seconds()
            if baseline_seconds <= 0 or remaining < baseline_seconds * 1.25:
                state['jobs'][marker] = {'status':'SKIPPED', 'reason':'remaining time below measured baseline duration plus margin'}
                atomic(state_path, state); continue
            variant_root = root / 'variants' / kind
            dataset_root = variant_root / 'dataset'
            prepared = dataset_root / 'train'
            prep_name = 'prepare-' + kind
            previous_prepare = state['jobs'].get(prep_name, {})
            if prepared.exists() and previous_prepare.get('status') != 'COMPLETED':
                raise RuntimeError(f'variant dataset already exists without a completed job: {prepared}')
            if not job(prep_name, augmentation_command(c, kind, dataset_root)): return
            manifest_path = prepared / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            augmentation = manifest.get('augmentation') or {}
            count = int((augmentation.get('n_variants_by_kind') or {}).get(kind, 0))
            effective_counts = effective_variant_count(prepared, kind, c['model']) if count > 0 else {'observed':0, 'retained':0}
            if count != effective_counts['observed']:
                raise RuntimeError(f'variant {kind} augmentation count disagrees with dataset')
            _, variant_heldout = select_heldout(load_samples('data','train'), prepared, split='train', limit=c['validation_limit'])
            if variant_heldout['ids'] != state['heldout']['ids']:
                raise RuntimeError(f'variant {kind} held-out IDs differ from baseline')
            if effective_counts['retained'] <= 0:
                state['jobs'][marker] = {'status':'SKIPPED', 'reason':'augmentation produced zero retained variants'}
                atomic(state_path, state); continue
            variant_horizon = expected_horizon(prepared, c['training_config'])
            variant_sha = {name:sha(prepared/name) for name in ('manifest.json','train.jsonl','val.jsonl')}
            old = state.get('variants', {}).get(kind)
            if old and old.get('dataset_sha256') != variant_sha: raise RuntimeError(f'variant dataset changed: {kind}')
            from lifemate.config import load_config
            from lifemate.features.config import feature_fingerprint, training_feature_fingerprint
            feature_cfg = load_config(c['training_config']).features
            flag_field = {'slot_masking':'slot_masking_enabled', 'tool_dropout':'tool_dropout_enabled', 'noisy_asr':'noisy_asr_training_enabled'}[kind]
            setattr(feature_cfg.training, flag_field, True)
            state.setdefault('variants', {})[kind] = {'dataset':str(prepared), 'dataset_sha256':variant_sha,
                'feature':VARIANT_FEATURES[kind], 'augmentation_count':count, 'retained_variants':effective_counts['retained'], 'horizon':variant_horizon,
                'feature_fingerprint':feature_fingerprint(feature_cfg), 'training_feature_fingerprint':training_feature_fingerprint(feature_cfg),
                'augmentation':{key:value for key,value in augmentation.items() if key != 'reports'},
                'seed':manifest.get('seed'), 'source_git_sha':state['git_sha']}
            atomic(state_path, state)
            estimate = baseline_seconds * variant_horizon / horizon
            if (deadline - now()).total_seconds() < estimate * 1.25:
                state['jobs'][marker] = {'status':'SKIPPED', 'reason':'remaining time below measured horizon-scaled training estimate', 'estimated_seconds':estimate}
                atomic(state_path, state); continue
            output_dir = variant_root / 'checkpoints'
            train_name = 'train-' + kind
            command = variant_training_command(c, kind, prepared, output_dir)
            prior = state['jobs'].get(train_name, {})
            if prior.get('status') in ('RUNNING','INTERRUPTED'):
                resume = sorted([p for p in (output_dir/'lora').glob('*-*') if valid_checkpoint(p, variant_horizon)], key=lambda p:int(p.name.rsplit('-',1)[-1]))
                if not resume: raise RuntimeError(f'variant {kind} has no valid checkpoint to resume')
                command += ['--resume', str(resume[-1])]
            elif (output_dir/'lora').exists() and prior.get('status') != 'COMPLETED':
                raise RuntimeError(f'variant output already exists: {output_dir}')
            variant_training = {'directory':str(output_dir), 'candidates':str(variant_root/'candidates'), 'horizon':variant_horizon}
            if not job(train_name, command, training=variant_training): return
            protected_variants = sorted([p for p in (variant_root/'candidates').glob('step-*') if valid_candidate(p)], key=lambda p:int(p.name.rsplit('-',1)[-1]))
            if not protected_variants: raise RuntimeError(f'variant {kind} has no protected candidate')
            candidate = protected_variants[-1]
            eval_out = variant_root/'evaluation'
            if not job('eval-' + kind, evaluation(c, candidate, eval_out)): return
            evaluated = json.loads((eval_out/'metrics.json').read_text())
            variant_score = score_evaluation(evaluated, state['heldout']['ids'], load_samples('data','train'))
            state['variants'][kind].update(candidate=str(candidate), adapter_sha256=sha(candidate/'adapter_model.safetensors'),
                                            score=variant_score, evaluation=str(eval_out))
            state['jobs'][marker] = {'status':'COMPLETED', 'candidate':str(candidate)}
            atomic(state_path, state)
            if variant_score['value'] > chosen['score'] and estimated_lower_bound is not None and (deadline-now()).total_seconds() >= estimated_lower_bound * 1.25:
                candidate_output = root/'submissions'/('candidate-'+sha(candidate/'adapter_model.safetensors')[:12]+'.jsonl')
                submit_name = 'submission-' + kind
                if not job(submit_name, submission(c, candidate, candidate_output)): return
                state['jobs'][submit_name]['validation'] = verify_submission(candidate_output)
                state['jobs'][submit_name]['output'] = str(candidate_output)
                runtime_path = Path(str(candidate_output) + '.runtime.json')
                state['jobs'][submit_name]['runtime'] = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {'available':False}
                atomic(state_path, state)
        if not variants:
            state['jobs']['additional_sft'] = {'status':'SKIPPED', 'reason':'no independent SFT variants configured'}
        state['status'] = 'COMPLETED_WITH_SKIPS' if any(str(j.get('status','')).startswith('SKIPPED') for j in state['jobs'].values()) else 'COMPLETED'
        atomic(state_path, state)
    except Exception as exc:
        state['status'] = 'FAILED'; state['error'] = f'{type(exc).__name__}: {exc}'; atomic(state_path, state)
        raise
    finally:
        state['worker_pid'] = None; state['worker_start'] = None
        try:
            state['final_report'] = write_status_report(root, state)
        except OSError as exc:
            state['report_error'] = str(exc)
        atomic(state_path, state)


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
        if not state_path.exists():
            print('{"status":"NOT_STARTED"}'); return 0
        state = json.loads(state_path.read_text())
        live = same_process(state.get('worker_pid'), state.get('worker_start'))
        state['worker_alive'] = live
        if state.get('status') in ('STARTING', 'RUNNING') and not live:
            child = next((v for v in state.get('jobs', {}).values() if v.get('status') == 'RUNNING'), {})
            state['observed_status'] = 'ORPHANED_CHILD' if same_process(child.get('pid'), child.get('process_start')) else 'WORKER_EXITED'
        else:
            state['observed_status'] = state.get('status')
        print(json.dumps(state, ensure_ascii=False, indent=2)); return 0
    if args.stop:
        if not state_path.exists(): print('no run exists', file=sys.stderr); return 2
        root.mkdir(parents=True, exist_ok=True); (root / 'STOP').write_text('stop\n'); print('stop requested'); return 0
    if args.dry_run:
        selection = preflight(c)
        print(json.dumps({'status':'DRY_RUN', 'baseline':baseline(c), 'evaluation_example':evaluation(c, '<protected-adapter>', root/'evaluation/example'), 'heldout_sha256':selection['ids_sha256'], 'heldout_count':len(selection['ids']), 'preflight':'PASS', 'runtime_validation':'NOT_RUN', 'real_submission_validation':'NOT_RUN', 'no_training_started':True}, indent=2, ensure_ascii=False)); return 0
    if args.worker:
        state = json.loads(state_path.read_text())
        try:
            worker(c, state, state_path, dt.datetime.fromisoformat(state['stop_at']))
        except Exception as exc:
            state['status'] = 'FAILED'; state['error'] = f'{type(exc).__name__}: {exc}'
            state['worker_pid'] = None; atomic(state_path, state)
            raise
        return 0
    root.mkdir(parents=True, exist_ok=True)
    launch_lock = (root / 'launch.lock').open('a+')
    try: fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another launch is in progress')
    if args.start:
        if state_path.exists(): raise RuntimeError('run already exists; inspect --status, never overwrite')
        if not args.stop_at: raise RuntimeError('--start requires --stop-at with timezone')
        deadline = dt.datetime.fromisoformat(args.stop_at)
        if deadline.tzinfo is None or deadline <= now() + dt.timedelta(minutes=int(c['minimum_start_minutes'])): raise RuntimeError('deadline must include timezone and allow minimum start time')
        # Only the already-existing metrics change is permitted in the caller;
        # the detached worktree always executes committed source.
        dirty = set(subprocess.check_output(['git','diff','--name-only','HEAD'], text=True).splitlines())
        if dirty - {'src/lifemate/evaluation/metrics.py'}:
            raise RuntimeError('tracked source/config changes must be committed before launch')
        selection = preflight(c)
        root.mkdir(parents=True, exist_ok=True)
        source = prepare_worktree(root / 'source', git_sha())
        model_dir = Path(c['model'])
        model_files = [model_dir / 'config.json', *sorted(model_dir.glob('model*.safetensors'))]
        state = {'status':'PENDING','created':now().isoformat(),'stop_at':deadline.isoformat(),'git_sha':git_sha(),'dataset_sha256':{name:sha(Path(c['dataset'])/name) for name in ('manifest.json','train.jsonl','val.jsonl')},'model_sha256':{p.name:sha(p) for p in model_files},'heldout':selection,'expected_horizon':selection['expected_horizon'],'config':c,'source':str(source.resolve()),'jobs':{}}
        atomic(state_path, state)
    else:
        if not state_path.exists(): raise RuntimeError('no state to resume')
        state = json.loads(state_path.read_text())
        if same_process(state.get('worker_pid'), state.get('worker_start')): raise RuntimeError('worker is still running')
        for job_info in state.get('jobs', {}).values():
            if job_info.get('status') == 'RUNNING' and same_process(job_info.get('pid'), job_info.get('process_start')):
                raise RuntimeError('a prior child is still running; refusing a duplicate job')
        if state.get('status') not in ('INTERRUPTED', 'RUNNING', 'STARTING'): raise RuntimeError(f'refusing automatic retry from {state.get("status")}')
        if args.stop_at:
            deadline=dt.datetime.fromisoformat(args.stop_at)
            if deadline.tzinfo is None or deadline <= now(): raise RuntimeError('invalid stop time')
            state['stop_at']=deadline.isoformat(); atomic(state_path,state)
        elif dt.datetime.fromisoformat(state['stop_at']) <= now(): raise RuntimeError('provide a new --stop-at to resume')
        (root/'STOP').unlink(missing_ok=True)
    source = Path(state['source'])
    if not source.is_dir(): raise RuntimeError('recorded clean worktree is missing')
    config_copy = root / 'run_config.json'
    atomic(config_copy, state['config'])
    state['status'] = 'STARTING'; atomic(state_path, state)
    log = (root/'worker.log').open('a')
    try:
        proc = subprocess.Popen([c['python'], str(source/'scripts'/'run_overnight.py'), '--config', str(config_copy.resolve()), '--worker'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=source)
    except OSError as exc:
        state['status'] = 'FAILED'; state['error'] = str(exc); atomic(state_path, state)
        raise
    # Require proof that the child entered worker(), not just that Popen worked.
    deadline_handshake = time.monotonic() + 15
    while time.monotonic() < deadline_handshake:
        current = json.loads(state_path.read_text())
        if current.get('worker_last_pid') == proc.pid:
            print(f'worker started pid={proc.pid}; status: {state_path}')
            return 0
        if proc.poll() is not None:
            if current.get('status') == 'STARTING':
                current['status'] = 'FAILED'; current['error'] = 'worker exited before startup handshake'; atomic(state_path, current)
            raise RuntimeError(f'worker exited before startup handshake ({proc.returncode}); see {root / "worker.log"}')
        time.sleep(0.05)
    raise RuntimeError(f'worker startup unconfirmed; inspect status and {root / "worker.log"} before retrying')

if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); sys.exit(2)
