import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('supervisor', Path(__file__).resolve().parents[1] / 'scripts/codex_research_supervisor.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def test_event_recovery_and_dedup(tmp_path):
    state={'jobs':{'baseline':{'status':'COMPLETED','started':'t1','ended':'t2'}}}
    ev=s.events_from_state(state,{})
    assert [e['kind'] for e in ev] == ['BEFORE_TRAIN','AFTER_TRAIN']
    s.enqueue(tmp_path,ev); s.enqueue(tmp_path,ev)
    assert len(list((tmp_path/'events').glob('*.json'))) == 2
    p=tmp_path/'events'/f"{ev[0]['id']}.json"; r=json.loads(p.read_text()); r.update(status='RUNNING',attempts=1); s.atomic(p,r)
    assert json.loads(p.read_text())['status']=='RUNNING'


def test_validate_rejects_wrong_ids_and_types():
    ev=s.event('baseline','AFTER_TRAIN','done',{})
    r={'event_id':ev['id'],'experiment_id':'baseline','verdict':'INCONCLUSIVE','confidence':'LOW','evidence':[], 'root_cause':None,'affected_components':[],'recommended_action':'inspect','decision':'INVESTIGATE','retraining_necessary':None,'checkpoint_valid':None}
    assert s.validate(r,ev)
    r['checkpoint_valid']='yes'; assert not s.validate(r,ev)
    r['checkpoint_valid']=True; r['event_id']='wrong'; assert not s.validate(r,ev)


def test_stall_only_after_sustained_interval():
    state={'jobs':{'baseline':{'status':'RUNNING','started':'t','telemetry':{'step':10,'step_observed_at':100}}}}
    assert len(s.events_from_state(state,{},now=1899)) == 1
    assert [e['kind'] for e in s.events_from_state(state,{},now=1901)] == ['BEFORE_TRAIN','ON_ANOMALY']


def test_process_reviews_once_and_recovery(tmp_path):
    fake=tmp_path/'fake-codex'
    fake.write_text('''#!/usr/bin/env python3
import json,re,sys
text=sys.stdin.read()
event=re.search(r'event_id=([0-9a-f]+)',text).group(1)
job=re.search(r'experiment_id=([a-z-]+)',text).group(1)
out=sys.argv[sys.argv.index('-o')+1]
json.dump(dict(event_id=event,experiment_id=job,verdict='INCONCLUSIVE',confidence='LOW',evidence=[],root_cause=None,affected_components=[],recommended_action='inspect',decision='INVESTIGATE',retraining_necessary=None,checkpoint_valid=None),open(out,'w'))
''')
    fake.chmod(0o755)
    ev=s.event('baseline','AFTER_TRAIN','done',{})
    s.enqueue(tmp_path,[ev])
    path=tmp_path/'events'/f"{ev['id']}.json"
    record=json.loads(path.read_text()); record.update(status='RUNNING',attempts=1); s.atomic(path,record)
    assert s.process_one(tmp_path,tmp_path,tmp_path/'state.json',executable=str(fake))
    record=json.loads(path.read_text())
    assert record['status']=='DONE' and record['attempts']==2 and record['decision_applied'] is False
    assert not s.process_one(tmp_path,tmp_path,tmp_path/'state.json',executable=str(fake))


def test_submission_waits_for_validation_and_training_failure_emits_anomaly():
    state={'jobs':{'submission':{'status':'COMPLETED','ended':'a'},'baseline':{'status':'FAILED','started':'b','ended':'c'}}}
    kinds=[e['kind'] for e in s.events_from_state(state,{})]
    assert kinds == ['BEFORE_TRAIN','AFTER_TRAIN','ON_ANOMALY']
    state['jobs']['submission'].update(output='path',validation={'ok':True})
    assert 'SUBMISSION_CHECK' in [e['kind'] for e in s.events_from_state(state,{})]


def test_bad_codex_result_is_bounded_and_never_applied(tmp_path):
    fake=tmp_path/'bad-codex'; fake.write_text('#!/bin/sh\nexit 2\n'); fake.chmod(0o755)
    ev=s.event('baseline','AFTER_TRAIN','done',{}); s.enqueue(tmp_path,[ev])
    path=tmp_path/'events'/f"{ev['id']}.json"
    assert s.process_one(tmp_path,tmp_path,tmp_path/'state',executable=str(fake))
    assert json.loads(path.read_text())['status']=='PENDING'
    assert s.process_one(tmp_path,tmp_path,tmp_path/'state',executable=str(fake))
    record=json.loads(path.read_text())
    assert record['status']=='FAILED' and record['attempts']==2 and record['exit_code']==2
    assert not s.process_one(tmp_path,tmp_path,tmp_path/'state',executable=str(fake))


def test_running_reviewer_is_not_duplicated(tmp_path):
    import os
    ev=s.event('baseline','AFTER_TRAIN','done',{}); s.enqueue(tmp_path,[ev])
    path=tmp_path/'events'/f"{ev['id']}.json"
    record=json.loads(path.read_text()); record.update(status='RUNNING',attempts=1,reviewer_pid=os.getpid(),reviewer_start=s.process_identity(os.getpid())); s.atomic(path,record)
    assert not s.process_one(tmp_path,tmp_path,tmp_path/'state',executable='/missing')
    assert json.loads(path.read_text())['attempts']==1
