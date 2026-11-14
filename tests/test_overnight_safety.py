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
    def interrupted(self,rows,statuses=None,latencies=None,timeouts=None):
        original_write(self,rows,statuses,latencies,timeouts)
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
    runtime=json.loads((tmp_path/'result.jsonl.runtime.json').read_text())
    assert runtime['n_samples']==3 and runtime['latency_seconds']['n']==3 and runtime['timeout_unknown']==0


def test_runner_refuses_start_without_deadline(tmp_path):
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'work_dir':str(tmp_path/'run')}))
    result=subprocess.run([sys.executable,str(script),'--config',str(config),'--start'],capture_output=True,text=True)
    assert result.returncode != 0 and 'requires --stop-at' in result.stderr
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
    assert module.valid_candidate(target)
    assert not module.valid_checkpoint(target)
    assert not (target/'optimizer_state.pt').exists()
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


def test_runner_derives_horizon_and_rejects_incompatible_checkpoint(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    spec=importlib.util.spec_from_file_location('overnight_horizon',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/'manifest.json').write_text(json.dumps({'fit':{'method':'tokenizer','n_fit':11,'max_seq_length':8192}}))
    cfg=Path(__file__).resolve().parents[1]/'configs'/'training_lora.yaml'
    assert module.expected_horizon(tmp_path,cfg)==6
    ckpt=tmp_path/'step-1';ckpt.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors','optimizer_state.pt'):
        (ckpt/name).write_text('x')
    (ckpt/'training_state.json').write_text(json.dumps({'exact_resume':True,'progress':{'planned_total_steps':6}}))
    assert module.valid_checkpoint(ckpt,6) and not module.valid_checkpoint(ckpt,1614)


def test_candidate_scoring_counts_model_failures_but_rejects_infrastructure():
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    spec=importlib.util.spec_from_file_location('overnight_scoring',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    samples=[SimpleNamespace(id=x,label=SimpleNamespace(tool_calls=[])) for x in ('a','b')]
    rows=[{'id':'a','status':'finished','state_hash':'h1','tool_calls':[]},
          {'id':'b','status':'repeated_failure','state_hash':'h2','tool_calls':[]}]
    payload={'closed_loop':{'rows':rows,'heldout_selection':{'ids':['a','b']},'n_failures':1}}
    score=module.score_evaluation(payload,['a','b'],samples)
    assert score['value']==.5 and score['model_failures']==1 and score['official_score']=='UNAVAILABLE'
    rows[1]['status']='error'
    with pytest.raises(RuntimeError,match='infrastructure'): module.score_evaluation(payload,['a','b'],samples)
    rows[1]['status']='max_steps';rows[1]['id']='a'
    with pytest.raises(RuntimeError,match='rows'): module.score_evaluation(payload,['a','b'],samples)


def test_clean_worktree_keeps_dirty_user_file_and_links_assets(tmp_path, monkeypatch):
    import subprocess
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    spec=importlib.util.spec_from_file_location('overnight_worktree',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.email','tests@example.invalid'],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.name','tests'],check=True)
    (repo/'tracked.txt').write_text('committed')
    subprocess.run(['git','-C',str(repo),'add','tracked.txt'],check=True)
    subprocess.run(['git','-C',str(repo),'commit','-qm','fixture'],check=True)
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    (repo/'tracked.txt').write_text('user dirty')
    for name in ('data','models','outputs'): (repo/name).mkdir()
    monkeypatch.setattr(module,'ROOT',repo);monkeypatch.chdir(repo)
    worktree=module.prepare_worktree(repo/'outputs'/'source',revision)
    assert (worktree/'tracked.txt').read_text()=='committed'
    assert (repo/'tracked.txt').read_text()=='user dirty'
    for name in ('data','models','outputs'):
        assert (worktree/name).resolve()==(repo/name).resolve()


def test_runner_metrics_opt_in_is_local_and_preserves_default(tmp_path, monkeypatch):
    from lifemate.config import load_config
    from lifemate.training.callbacks import JsonlTracker, NullTracker
    path=Path(__file__).resolve().parents[1]/'scripts'/'train.py'
    spec=importlib.util.spec_from_file_location('overnight_train_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=load_config();cfg.training.output_dir=str(tmp_path)
    args=SimpleNamespace(smoke=False)
    monkeypatch.delenv('LIFEMATE_LOCAL_METRICS',raising=False)
    assert isinstance(module._build_tracker(cfg,args,run_name='lora'),NullTracker)
    monkeypatch.setenv('LIFEMATE_LOCAL_METRICS','1')
    tracker=module._build_tracker(cfg,args,run_name='lora')
    assert isinstance(tracker,JsonlTracker)
    tracker.log({'step':10,'train_loss':.25,'learning_rate':.0002})
    assert json.loads((tmp_path/'lora/metrics.jsonl').read_text())['step']==10


def test_candidate_retention_preserves_best_and_recent(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    spec=importlib.util.spec_from_file_location('overnight_retention',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for step in range(1,6):
        source=tmp_path/f'source-{step}';source.mkdir()
        for name in ('adapter_config.json','adapter_model.safetensors','optimizer_state.pt'):(source/name).write_text(f'{name}-{step}')
        (source/'training_state.json').write_text(json.dumps({'exact_resume':True,'progress':{'planned_total_steps':7}}))
        module.protect_checkpoint(source,tmp_path/'candidates'/f'step-{step}',0,horizon=7)
    root=tmp_path/'candidates'
    (root/'best.json').write_text(json.dumps({'checkpoint':str(root/'step-1')}))
    module.prune_candidates(root,keep=2)
    assert {p.name for p in root.glob('step-*')}=={'step-1','step-4','step-5'}
    assert all(not (p/'optimizer_state.pt').exists() for p in root.glob('step-*'))


def test_audio_evaluation_asr_error_never_runs_agent(tmp_path):
    from lifemate.evaluation.evaluator import evaluate_closed_loop
    class Env:
        def get_tool_calls(self): return []
        def get_state_hash(self): return 'official-hash'
    class Agent:
        env=Env()
        def run(self,*args,**kwargs): raise AssertionError('must not act')
    result=evaluate_closed_loop(lambda _:Agent(),[SimpleNamespace(id='s',label=None)],split='train',data_root=tmp_path,
        evaluation_mode='audio_to_action',transcribe_sample=lambda _:('',{'error':'corrupt audio'}))
    assert result['rows'][0]['status']=='asr_failed' and result['rows'][0]['state_hash']=='official-hash'
    assert result['n_failures']==1


def test_effective_variant_count_respects_training_filter_and_validation(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_overnight.py'
    spec=importlib.util.spec_from_file_location('overnight_effective_variant',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[{'meta':{'augmentation':'slot_masking'},'messages':[],'target':'keep'},
             {'meta':{'augmentation':'slot_masking'},'messages':[],'target':'drop'},
             {'meta':{},'messages':[],'target':'base'}]
    (tmp_path/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (tmp_path/'val.jsonl').write_text(json.dumps({'meta':{}})+'\n')
    result=module.effective_variant_count(tmp_path,'slot_masking','unused',tokenizer=object(),
                                           build=lambda tok,msg,target,n:{'kept':target=='keep'})
    assert result=={'observed':2,'retained':1}
    (tmp_path/'val.jsonl').write_text(json.dumps({'meta':{'augmentation':'slot_masking'}})+'\n')
    with pytest.raises(RuntimeError,match='validation'): module.effective_variant_count(tmp_path,'slot_masking','unused',tokenizer=object(),build=lambda *a:{'kept':True})
