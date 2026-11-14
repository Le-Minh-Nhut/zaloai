import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ai_review_gate import GateBlocked, ReviewGate


def review(event, *, verdict='PASS', decision='CONTINUE', checkpoint_valid=True):
    return {'event_id': event['id'], 'experiment_id': event['job'], 'verdict': verdict,
            'confidence':'HIGH', 'evidence':['verified synthetic artifact'], 'root_cause':None,
            'affected_components':[], 'recommended_action':'continue', 'decision':decision,
            'retraining_necessary':False, 'checkpoint_valid':checkpoint_valid}


def state(tmp):
    return {'git_sha':'abc', 'dataset_sha256':{'train':'hash'}, 'model_sha256':{'model':'hash'},
            'heldout':{'ids_sha256':'held'}, 'source':str(tmp), 'config':{'seed':42}}


def test_sequential_jobs_review_and_resume_without_duplicates(tmp_path):
    seen=[]
    def invoke(event, facts, output, log):
        seen.append((event['job'],event['kind']))
        return review(event, verdict='MODEL_UNDERPERFORMANCE' if event['kind']=='AFTER_EVAL' else 'PASS',
                      decision='RUN_EVALUATION' if event['kind']=='AFTER_TRAIN' else 'CONTINUE')
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke)
    st=state(tmp_path)
    artifact=tmp_path/'artifact'
    for name in ['train','eval']:
        training=name=='train'; command=[sys.executable,'-c',f"open({str(artifact)!r},'a').write({name!r}+'\\n')"]
        pre='BEFORE_TRAIN' if training else 'BEFORE_JOB'; post='AFTER_TRAIN' if training else 'AFTER_EVAL'
        gate.check(pre,name,command,st,{'status':'PENDING'},is_training=training)
        subprocess.run(command,check=True)
        gate.check(post,name,command,st,{'status':'COMPLETED','exit_code':0},is_training=training)
        gate.check(post,name,command,st,{'status':'COMPLETED','exit_code':0},is_training=training)
    assert artifact.read_text().splitlines()==['train','eval']
    assert len(seen)==4


@pytest.mark.parametrize('verdict,choice,confidence',[
    ('INCONCLUSIVE','CONTINUE','HIGH'),('BUG_CONFIRMED','CONTINUE','HIGH'),
    ('BLOCKED','CONTINUE','HIGH'),('PASS','START_NEXT_EXPERIMENT','HIGH'),
    ('PASS','CONTINUE','LOW')])
def test_bad_or_unsafe_review_blocks(tmp_path,verdict,choice,confidence):
    def invoke(event,*args):
        r=review(event,verdict=verdict,decision=choice); r['confidence']=confidence; return r
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke)
    with pytest.raises(GateBlocked): gate.check('BEFORE_TRAIN','train',[],state(tmp_path),{'status':'PENDING'},is_training=True)
    assert list((tmp_path/'gate').glob('*.json'))


def test_identity_change_forces_fresh_review_and_failure_is_not_retried_silently(tmp_path):
    seen=[]
    def invoke(event,*args): seen.append(event); return review(event)
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke); st=state(tmp_path)
    gate.check('BEFORE_JOB','eval',['one'],st,{'status':'PENDING'})
    st['dataset_sha256']['train']='different'
    gate.check('BEFORE_JOB','eval',['one'],st,{'status':'PENDING'})
    assert len(seen)==2
    gate2=ReviewGate(tmp_path/'fail',tmp_path,invoke=lambda *a: (_ for _ in ()).throw(GateBlocked('timeout')))
    for _ in range(2):
        with pytest.raises(GateBlocked): gate2.check('BEFORE_JOB','eval',[],st,{'status':'PENDING'})


def test_crash_recovery_and_retry_decision(tmp_path):
    def invoke(event,*args): return review(event,decision='RETRY_FROM_CHECKPOINT')
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke); st=state(tmp_path)
    # Interrupted review record is retried once; an approved result is reused.
    outcome=gate.check('AFTER_TRAIN','train',[],st,{'status':'FAILED'},is_training=True)
    assert outcome=='RETRY_FROM_CHECKPOINT'
    path=next(p for p in (tmp_path/'gate').glob('*.json') if not p.name.endswith('.review.json'))
    original=json.loads(path.read_text()); original['status']='RUNNING'; original['attempts']=1
    path.write_text(json.dumps(original))
    assert gate.check('AFTER_TRAIN','train',[],st,{'status':'FAILED'},is_training=True)=='RETRY_FROM_CHECKPOINT'


def test_real_subprocess_timeout_is_fail_closed(tmp_path):
    executable=tmp_path/'fake-codex'; executable.write_text('#!/bin/sh\nsleep 10\n'); executable.chmod(0o755)
    gate=ReviewGate(tmp_path/'gate',tmp_path,executable=str(executable),timeout=1)
    with pytest.raises(GateBlocked,match='timed out'):
        gate.check('BEFORE_JOB','eval',[],state(tmp_path),{'status':'PENDING'})
    records=[json.loads(p.read_text()) for p in (tmp_path/'gate').glob('*.json')]
    assert len(records)==1 and records[0]['status']=='REJECTED'


def test_modified_dataset_during_review_is_rejected(tmp_path):
    dataset=tmp_path/'dataset'; dataset.mkdir()
    for name in ('manifest.json','train.jsonl','val.jsonl'): (dataset/name).write_text('original')
    def invoke(event,*args):
        (dataset/'train.jsonl').write_text('changed')
        return review(event)
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke)
    with pytest.raises(GateBlocked,match='changed during review'):
        gate.check('BEFORE_TRAIN','train',['--dataset-dir',str(dataset)],state(tmp_path),{'status':'PENDING'},is_training=True)


def test_explicit_retry_preserves_rejected_record(tmp_path):
    from ai_review_gate import retry_rejected
    calls=[]
    def invoke(event,*args):
        calls.append(1)
        return review(event, verdict='INCONCLUSIVE' if len(calls)==1 else 'PASS')
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke); st=state(tmp_path)
    with pytest.raises(GateBlocked): gate.check('BEFORE_JOB','eval',[],st,{'status':'PENDING'})
    record=next((tmp_path/'gate').glob('*.json'))
    retry_rejected(tmp_path/'gate',record.stem)
    assert gate.check('BEFORE_JOB','eval',[],st,{'status':'PENDING'})=='CONTINUE'
    archives=list((tmp_path/'gate/audit').glob('*.json'))
    assert len(archives)==1 and json.loads(archives[0].read_text())['status']=='REJECTED'


def test_changed_evaluation_metrics_requires_new_review(tmp_path):
    out=tmp_path/'eval';out.mkdir();(out/'metrics.json').write_text('{"value":1}')
    seen=[]
    def invoke(event,*args): seen.append(event); return review(event)
    gate=ReviewGate(tmp_path/'gate',tmp_path,invoke=invoke); st=state(tmp_path)
    command=['--output',str(out)]
    gate.check('AFTER_EVAL','eval',command,st,{'status':'COMPLETED'})
    (out/'metrics.json').write_text('{"value":2}')
    gate.check('AFTER_EVAL','eval',command,st,{'status':'COMPLETED'})
    assert len(seen)==2


def test_reviewer_crash_blocks_following_job_and_records_exit(tmp_path):
    executable=tmp_path/'fake-codex'; executable.write_text('#!/bin/sh\nexit 7\n'); executable.chmod(0o755)
    gate=ReviewGate(tmp_path/'gate',tmp_path,executable=str(executable),timeout=5)
    marker=tmp_path/'job-ran'
    with pytest.raises(GateBlocked,match='exit code 7'):
        gate.check('BEFORE_JOB','eval',[],state(tmp_path),{'status':'PENDING'})
        marker.write_text('unsafe')
    assert not marker.exists()
    record=json.loads(next((tmp_path/'gate').glob('*.json')).read_text())
    assert record['status']=='REJECTED' and 'exit code 7' in record['error']
