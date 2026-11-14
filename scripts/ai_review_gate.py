"""Fail-closed Codex review gate for *future* experiment workers.

No model-generated shell command is ever executed. Gate records are bound to
immutable run identities, job command and job completion data.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from codex_research_supervisor import SCHEMA, atomic, validate, process_identity


class GateBlocked(RuntimeError):
    pass


def retry_rejected(directory, event_id):
    """Explicitly requeue a rejected review while retaining its audit record."""
    if len(event_id) != 64 or any(c not in '0123456789abcdef' for c in event_id):
        raise GateBlocked('invalid event id')
    root = Path(directory)
    path = root / (event_id + '.json')
    with (root / 'gate.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise GateBlocked('reviewer is running') from exc
        old = json.loads(path.read_text())
        if old.get('status') != 'REJECTED': raise GateBlocked('event was not rejected')
        archive = root / 'audit' / f'{event_id}.{time.time_ns()}.json'
        atomic(archive, old)
        atomic(path, {'status':'RETRYABLE', 'previous_record':str(archive)})


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(',', ':')).encode()).hexdigest()


def artifact_identities(command, phase):
    """Bind a review to the actual adapter and prepared dataset, not just paths."""
    artifacts = {}
    for flag, names in (('--checkpoint', ('adapter_config.json', 'adapter_model.safetensors')),
                        ('--resume', ('adapter_config.json', 'adapter_model.safetensors', 'training_state.json', 'optimizer_state.pt')),
                        ('--dataset-dir', ('manifest.json', 'train.jsonl', 'val.jsonl')),
                        ('--heldout-dataset', ('manifest.json', 'train.jsonl', 'val.jsonl'))):
        if flag not in command:
            continue
        index = command.index(flag)
        if index + 1 >= len(command): raise GateBlocked(f'{flag} lacks a path')
        directory = Path(command[index + 1])
        for name in names:
            file = directory / name
            if not file.is_file(): raise GateBlocked(f'missing required artifact: {file}')
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
            artifacts[f'{flag}/{name}'] = digest.hexdigest()
    if phase in {'AFTER_EVAL', 'SUBMISSION_CHECK'} and '--output' in command:
        index = command.index('--output')
        if index + 1 >= len(command): raise GateBlocked('--output lacks a path')
        output = Path(command[index + 1])
        candidates = [output] if output.is_file() else [output / 'metrics.json', output / 'train/manifest.json']
        for file in candidates:
            if file.is_file():
                digest = hashlib.sha256()
                with file.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
                artifacts[f'output/{file.name}'] = digest.hexdigest()
    return artifacts


def decision(review, phase, is_training, status):
    verdict = review['verdict']; choice = review['decision']
    if not review['evidence'] or review['confidence'] == 'LOW':
        raise GateBlocked('review lacks sufficient evidence or confidence')
    if verdict in {'INCONCLUSIVE', 'BLOCKED', 'BUG_CONFIRMED'}:
        raise GateBlocked(f'review requires investigation: {verdict}')
    if phase in {'BEFORE_TRAIN', 'BEFORE_JOB'} and (verdict != 'PASS' or choice != 'CONTINUE'):
        raise GateBlocked('job preflight requires PASS/CONTINUE')
    if phase in {'BEFORE_TRAIN', 'BEFORE_JOB'}:
        return 'CONTINUE'
    if status != 'COMPLETED':
        if is_training and choice == 'RETRY_FROM_CHECKPOINT' and verdict == 'PASS' and review['checkpoint_valid'] is True:
            return 'RETRY_FROM_CHECKPOINT'
        raise GateBlocked('failed job requires verified checkpoint recovery')
    allowed = {'CONTINUE'}
    if is_training: allowed.add('RUN_EVALUATION')
    if phase == 'AFTER_EVAL': allowed.add('START_NEXT_EXPERIMENT')
    if verdict not in {'PASS', 'MODEL_UNDERPERFORMANCE'} or choice not in allowed:
        raise GateBlocked(f'decision {choice} is not permitted at {phase}')
    if phase == 'AFTER_TRAIN' and review['checkpoint_valid'] is not True:
        raise GateBlocked('training checkpoint was not positively verified')
    return 'CONTINUE'


class ReviewGate:
    def __init__(self, directory, source, *, executable='codex', timeout=300, invoke=None):
        self.directory=Path(directory).resolve(); self.directory.mkdir(parents=True, exist_ok=True)
        self.source=Path(source).resolve(); self.executable=executable; self.timeout=timeout; self.invoke=invoke or self._invoke
        if timeout < 1 or timeout > 900: raise ValueError('review timeout must be between 1 and 900 seconds')

    def _invoke(self, ev, facts, output, log):
        instructions=(f"You are the scientific QA gate for LifeMate. Event id={ev['id']}, experiment id={ev['job']}. "
            "Return only the required schema. Inspect the actual evidence, dataset and artifact identities. "
            "Never run GPU jobs, change files, execute actions, access hidden state, or send messages. "
            "Examine causal targets and masking, data provenance, checkpoint and scheduler continuity, "
            "held-out IDs and comparable evaluation metrics as appropriate. Distinguish implementation bugs "
            "from model underperformance; do not call diagnostic metrics official scores. "
            "Use INCONCLUSIVE if evidence cannot establish validity. A completed training job must set "
            "checkpoint_valid true only if actually verified. You may recommend only schema decisions.\n"
            f"Facts: {json.dumps(facts, ensure_ascii=False)}")
        args=[self.executable,'exec','--sandbox','read-only','--ephemeral','--skip-git-repo-check',
              '-C',str(self.source),'--output-schema',str(SCHEMA),'-o',str(output),'-']
        with log.open('w',encoding='utf-8') as out:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=out,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            pending=json.loads(self._record.read_text())
            pending.update(reviewer_pid=proc.pid, reviewer_start=process_identity(proc.pid))
            atomic(self._record,pending)
            try: proc.communicate(instructions, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.communicate(); raise GateBlocked('reviewer timed out')
        if proc.returncode: raise GateBlocked(f'reviewer exit code {proc.returncode}')
        try: return json.loads(output.read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc: raise GateBlocked(f'invalid reviewer output: {exc}') from exc

    def check(self, phase, name, command, state, entry, *, is_training=False):
        facts={'phase':phase,'job':name,'command':command,'job_entry':entry,
               'git_sha':state['git_sha'],'dataset_sha256':state['dataset_sha256'],
               'model_sha256':state['model_sha256'],'heldout':state['heldout'],
               'source':state['source'],'config':state['config'],
               'command_artifacts':artifact_identities(command, phase)}
        event_id=identity(facts); ev={'id':event_id,'job':name,'kind':phase}
        record=self.directory/(event_id+'.json'); output=self.directory/(event_id+'.review.json'); log=self.directory/(event_id+'.log')
        with (self.directory/'gate.lock').open('a+') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise GateBlocked('another gate reviewer is running') from exc
            attempts = 0
            if record.exists():
                old=json.loads(record.read_text())
                if old.get('status') == 'APPROVED':
                    review=old.get('review')
                    if validate(review,ev): return decision(review,phase,is_training,entry.get('status'))
                    raise GateBlocked('stored review failed validation')
                if old.get('status') == 'REJECTED':
                    raise GateBlocked('previous review rejected; explicit investigation required')
                if old.get('status') == 'RUNNING':
                    if old.get('reviewer_start') and process_identity(old.get('reviewer_pid')) == old['reviewer_start']:
                        raise GateBlocked('previous reviewer is still running')
                    attempts = int(old.get('attempts', 1))
                    if attempts >= 2: raise GateBlocked('review recovery limit exhausted')
            self._record = record
            atomic(record,{'status':'RUNNING','event':ev,'facts':facts,'created_at':time.time(),'attempts':attempts+1})
            review = None
            try:
                review=self.invoke(ev,facts,output,log)
                if not validate(review,ev): raise GateBlocked('review does not match required schema/identity')
                if artifact_identities(command, phase) != facts['command_artifacts']:
                    raise GateBlocked('dataset or checkpoint changed during review')
                outcome=decision(review,phase,is_training,entry.get('status'))
                atomic(record,{'status':'APPROVED','event':ev,'facts':facts,'review':review,'outcome':outcome,'completed_at':time.time()})
                return outcome
            except (GateBlocked,OSError,ValueError) as exc:
                atomic(record,{'status':'REJECTED','event':ev,'facts':facts,'review':review,'error':str(exc),'completed_at':time.time()})
                raise GateBlocked(str(exc)) from exc
