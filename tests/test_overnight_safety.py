import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from lifemate.evaluation.progress import ProgressJournal
from lifemate.evaluation.submission import SubmissionError
from lifemate.evaluation.heldout import select_heldout
from lifemate.data.preprocessing import hash_ids


def test_journal_resume_checks_order_checksum_and_identity(tmp_path):
    path = tmp_path / 'submission.progress.json'
    journal = ProgressJournal(path, {'checkpoint': 'a'}, ['a', 'b', 'c'])
    rows = [{'id':'a', 'tool_calls':[], 'state_hash':'official-a'}, {'id':'b', 'tool_calls':[], 'state_hash':'official-b'}]
    journal.write(rows, ['asr_failed', 'ok'])
    assert journal.read() == rows
    assert journal.statuses == ['asr_failed', 'ok']
    with pytest.raises(SubmissionError):
        ProgressJournal(path, {'checkpoint':'other'}, ['a','b','c']).read()
    data=json.loads(path.read_text()); data['data']['rows'][0]['state_hash']='corrupt'; path.write_text(json.dumps(data))
    with pytest.raises(SubmissionError): journal.read()


def test_heldout_uses_original_ids_and_rejects_leak(tmp_path, monkeypatch):
    import lifemate.evaluation.heldout as module
    monkeypatch.setattr(module, 'sample_hash', lambda sample: sample.group)
    samples=[SimpleNamespace(id='a',group='g1'),SimpleNamespace(id='b',group='g2')]
    (tmp_path/'train.jsonl').write_text(json.dumps({'sample_id':'a'})+'\n')
    (tmp_path/'val.jsonl').write_text(json.dumps({'sample_id':'b'})+'\n')
    (tmp_path/'manifest.json').write_text(json.dumps({'split':'train','train_ids_hash':hash_ids(['a']),'val_ids_hash':hash_ids(['b'])}))
    chosen, info=select_heldout(samples,tmp_path,split='train')
    assert [s.id for s in chosen] == ['b'] and info['ids'] == ['b']
    samples[1].group='g1'
    with pytest.raises(Exception,match='overlap'): select_heldout(samples,tmp_path,split='train')


def test_ablation_forwards_adapter_and_heldout():
    path=Path(__file__).resolve().parents[1]/'scripts'/'feature_ablation.py'
    spec=importlib.util.spec_from_file_location('ablation_overnight',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cmd=module._evaluate_command(Path('arm.yaml'),out_dir=Path('out'),split='train',data_root='data',limit=20,mock_env=False,generator=None,predictions=None,evaluation_mode='audio_to_action',checkpoint='/models/adapter',model='/models/base',heldout_dataset='dataset')
    for flag,value in [('--checkpoint','/models/adapter'),('--model-mode','lora'),('--model','/models/base'),('--heldout-dataset','dataset')]:
        assert cmd[cmd.index(flag)+1]==value


def test_inference_interruption_resumes_from_durable_progress(tmp_path, monkeypatch):
    import copy
    from lifemate.asr.transcription import infer_module
    from lifemate.data.loader import load_samples
    from lifemate.agent.agent import ToolAgent
    root=Path(__file__).resolve().parent/'fixtures'/'synthetic_lifemate'
    original=load_samples(root,'train',limit=1,with_labels=False)[0]
    samples=[]
    for name in ('one','two','three'):
        sample=copy.deepcopy(original); sample.id=name; samples.append(sample)
    infer=infer_module()
    monkeypatch.setattr(infer,'load_samples',lambda *args,**kwargs:samples)
    monkeypatch.setattr(infer,'_build_asr',lambda *args,**kwargs:object())
    monkeypatch.setattr(infer,'_transcribe',lambda *args,**kwargs:('request',{'error':None}))
    class Generator:
        def generate(self,*args,**kwargs): return '<action>{"type":"finish"}</action>'
    monkeypatch.setattr(infer,'_load_generator',lambda *args,**kwargs:Generator())
    calls=[]; original_run=ToolAgent.run
    def track(self,sample,*args,**kwargs):
        calls.append(sample.id); return original_run(self,sample,*args,**kwargs)
    monkeypatch.setattr(ToolAgent,'run',track)
    original_write=ProgressJournal.write
    def interrupted(self,rows,statuses=None):
        original_write(self,rows,statuses)
        if len(rows)==2: raise KeyboardInterrupt()
    monkeypatch.setattr(ProgressJournal,'write',interrupted)
    out=tmp_path/'result.jsonl'
    args=['--split','train','--data-root',str(root),'--use-mock-env','--asr','gipformer','--output',str(out)]
    with pytest.raises(KeyboardInterrupt): infer.run_inference(args)
    assert not out.exists() and (tmp_path/'result.jsonl.progress.json').exists()
    monkeypatch.setattr(ProgressJournal,'write',original_write)
    assert infer.run_inference(args+['--resume'])==0
    assert calls==['one','two','three']
    assert [json.loads(line)['id'] for line in out.read_text().splitlines()]==['one','two','three']
    assert not (tmp_path/'result.jsonl.progress.json').exists()


def test_unreleased_runner_refuses_start_without_creating_state(tmp_path):
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'work_dir':str(tmp_path/'run')}))
    result=subprocess.run([sys.executable,str(script),'--config',str(config),'--start','--stop-at','2099-01-01T07:00:00+07:00'],capture_output=True,text=True)
    assert result.returncode != 0 and 'not released' in result.stderr
    assert not (tmp_path/'run'/'state.json').exists()


def test_failed_episodes_invalidate_ablation_comparison(tmp_path, monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts'/'feature_ablation.py'
    spec=importlib.util.spec_from_file_location('ablation_failed_episodes',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    arm=tmp_path/'arm.yaml'; arm.write_text('features: {}\n')
    def fake_run(command, dry_run):
        output=Path(command[command.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
        payload={'closed_loop':{'n_run':1,'n_failures':1,'status_counts':{'repeated_failure':1}}, 'metrics':{'tool_call_exact_match':{'value':0.0,'available':True}}}
        (output/'metrics.json').write_text(json.dumps(payload))
        return {'status':'RAN','command':'fake','returncode':0}
    monkeypatch.setattr(module,'_run_arm',fake_run)
    report=module.build_report([arm,arm],split='train',data_root=None,limit=1,out=tmp_path/'report.json',mock_env=True,dry_run=False)
    assert all(entry['status']=='INCOMPLETE' for entry in report['arms'])
    assert all(entry['deltas']['tool_call_exact_match'] is None for entry in report['arms'])


def test_protect_checkpoint_is_atomic_and_refuses_replacement(tmp_path):
    import sys
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    spec=importlib.util.spec_from_file_location('overnight_protection',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source=tmp_path/'step-100';source.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors','optimizer_state.pt'):
        (source/name).write_text(name)
    (source/'training_state.json').write_text(json.dumps({'exact_resume':True,'progress':{'planned_total_steps':1614}}))
    target=tmp_path/'protected'/'step-100'
    digest=module.protect_checkpoint(source,target,0)
    assert module.valid_checkpoint(target)
    assert json.loads((target/'candidate_identity.json').read_text())['adapter_sha256']==digest
    assert module.protect_checkpoint(source,target,0)==digest
    (source/'adapter_model.safetensors').write_text('different')
    with pytest.raises(RuntimeError,match='differs'):module.protect_checkpoint(source,target,0)
    assert (target/'adapter_model.safetensors').read_text()=='adapter_model.safetensors'


def test_heldout_and_progress_reject_prohibited_paths(tmp_path):
    from lifemate.errors import ProhibitedAccessError
    from lifemate.evaluation.progress import file_identity
    forbidden=tmp_path/('train'+'_env'+'ironment')/'data.json'
    forbidden.parent.mkdir(); forbidden.write_text('{}')
    with pytest.raises(ProhibitedAccessError): file_identity(forbidden)
    with pytest.raises(ProhibitedAccessError): select_heldout([],forbidden.parent,split='train')
    link=tmp_path/'alias';link.symlink_to(forbidden)
    with pytest.raises(ProhibitedAccessError): file_identity(link)
