"""Real detached processes with tiny file-only jobs, never GPU/model training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'run_overnight.py'


def _setup(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('runner_integration_parent',SOURCE)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    run=tmp_path/'run'; dataset=tmp_path/'dataset'; model=tmp_path/'model'
    dataset.mkdir();model.mkdir()
    for name in ('manifest.json','train.jsonl','val.jsonl'):(dataset/name).write_text('{}\n')
    (model/'config.json').write_text('{}');(model/'model-1.safetensors').write_bytes(b'model')
    config={'python':sys.executable,'model':str(model),'dataset':str(dataset),'training_config':str(SOURCE.parents[1]/'configs'/'training_lora.yaml'),
            'inference_config':'unused','work_dir':str(run),'validation_limit':2,'minimum_free_gib':0,'minimum_start_minutes':0,'inference_features':['prompt.itn'],'sft_variants':[]}
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    jobs=tmp_path/'fake_jobs.py'
    jobs.write_text('''import sys,time,json,signal\nfrom pathlib import Path\nkind,root=sys.argv[1],Path(sys.argv[2])\nif kind == "baseline":\n d=root/"baseline/checkpoints/lora";d.mkdir(parents=True,exist_ok=True)\n def ckpt(step):\n  p=d/f"step-{step}";p.mkdir(exist_ok=True)\n  for n in ("adapter_config.json","adapter_model.safetensors","optimizer_state.pt"):(p/n).write_text(n+str(step))\n  (p/"training_state.json").write_text(json.dumps({"exact_resume":True,"progress":{"planned_total_steps":2}}))\n ckpt(1)\n if (root/"fail").exists(): sys.exit(7)\n if (root/"hold").exists() and "--resume" not in sys.argv:\n  def stop(*a): sys.exit(130)\n  signal.signal(signal.SIGINT,stop)\n  while True: time.sleep(.05)\n time.sleep(.3);ckpt(2)\nelif kind == "prepare":\n out=Path(sys.argv[3])/"train";out.mkdir(parents=True,exist_ok=True)\n (out/"manifest.json").write_text(json.dumps({"seed":42,"augmentation":{"n_variants_by_kind":{"slot_masking":2}}}))\n (out/"train.jsonl").write_text("{}\\n");(out/"val.jsonl").write_text("{}\\n")\nelif kind == "variant":\n out=Path(sys.argv[3])/"lora"/"step-3";out.mkdir(parents=True,exist_ok=True)\n for n in ("adapter_config.json","adapter_model.safetensors","optimizer_state.pt"):(out/n).write_text(n+"variant")\n (out/"training_state.json").write_text(json.dumps({"exact_resume":True,"progress":{"planned_total_steps":3}}))\nelif kind == "submit":\n out=Path(sys.argv[sys.argv.index("--output")+1]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("{}\\n")\nelse:\n out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)\n rows=[{"id":"a","status":"finished","state_hash":"h1","tool_calls":[]},{"id":"b","status":"repeated_failure","state_hash":"h2","tool_calls":[]}]\n payload={"closed_loop":{"rows":rows,"n_failures":1,"latency":{"total":{"mean":0.01}},"heldout_selection":{"ids":["a","b"],"ids_sha256":"fixed"}},"feature_execution":{"prompt.itn":{"n_requested":2,"n_executed":2}}}\n (out/"metrics.json").write_text(json.dumps(payload))\n''')
    def fake_worktree(path,revision):
        path.mkdir(parents=True); target=path/'scripts';target.mkdir()
        harness=target/'run_overnight.py'
        harness.write_text(f'''import importlib.util,sys\nfrom types import SimpleNamespace\nspec=importlib.util.spec_from_file_location("child_runner",{str(SOURCE)!r});m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\nm.preflight=lambda c: {{}}\nm.git_sha=lambda: "fake-commit"\nm.source_clean=lambda: True\nm.gpu_free=lambda: (16000, "")\nm.load_samples=lambda *a,**k: [SimpleNamespace(id=x,label=SimpleNamespace(tool_calls=[])) for x in ("a","b")]\nm.baseline=lambda c: [sys.executable,{str(jobs)!r},"baseline",c["work_dir"]]\nm.evaluation=lambda c,checkpoint,out: [sys.executable,{str(jobs)!r},"eval",c["work_dir"],str(out)]\nm.audio_evaluation=lambda c,checkpoint,out,feature=None: [sys.executable,{str(jobs)!r},"eval",c["work_dir"],str(out)]\nm.submission=lambda c,checkpoint,out: [sys.executable,{str(jobs)!r},"submit",c["work_dir"],"--output",str(out)]\nm.public_sample_ids=lambda: ["p"]\nm.verify_submission=lambda path: {{"ok":True}}\nm.augmentation_command=lambda c,kind,directory: [sys.executable,{str(jobs)!r},"prepare",c["work_dir"],str(directory)]\nm.variant_training_command=lambda c,kind,dataset,out: [sys.executable,{str(jobs)!r},"variant",c["work_dir"],str(out)]\nm.select_heldout=lambda *a,**k: ([],{{"ids":["a","b"],"ids_sha256":"fixed"}})\nm.expected_horizon=lambda *a,**k: 3\nm.effective_variant_count=lambda *a,**k: {{"observed":2,"retained":2}}\nsys.exit(m.main())\n''')
        return path
    monkeypatch.setattr(runner,'prepare_worktree',fake_worktree)
    monkeypatch.setattr(runner,'preflight',lambda c:{'ids':['a','b'],'ids_sha256':'fixed','expected_horizon':2})
    monkeypatch.setattr(runner,'git_sha',lambda:'fake-commit')
    original=subprocess.check_output
    def check_output(cmd,*args,**kwargs):
        if cmd[:3]==['git','diff','--name-only']:return ''
        return original(cmd,*args,**kwargs)
    monkeypatch.setattr(runner.subprocess,'check_output',check_output)
    return runner,run,config_path


def _invoke(monkeypatch,runner,config,flag,*extra):
    monkeypatch.setattr(sys,'argv',['run_overnight.py','--config',str(config),flag,*extra])
    return runner.main()


def _wait_status(run, wanted, timeout=40):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if (run/'state.json').exists():
            state=json.loads((run/'state.json').read_text())
            if state.get('status') in wanted and state.get('worker_pid') is None:return state
        time.sleep(.1)
    raise AssertionError((run/'worker.log').read_text() if (run/'worker.log').exists() else 'no worker log')


def test_detached_worker_sequential_evaluation_and_model_failure(tmp_path,monkeypatch,capsys):
    runner,run,config=_setup(tmp_path,monkeypatch)
    assert _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')==0
    capsys.readouterr()
    assert _invoke(monkeypatch,runner,config,'--status')==0
    observed=json.loads(capsys.readouterr().out)
    assert observed['worker_alive'] is True
    with pytest.raises(RuntimeError,match='already exists'):
        _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')
    state=_wait_status(run,{'COMPLETED_WITH_SKIPS','FAILED'})
    assert state['status']=='COMPLETED_WITH_SKIPS', state
    assert state['worker_last_pid'] and state['worker_pid'] is None
    assert state['jobs']['baseline']['status']=='COMPLETED'
    assert state['jobs']['eval-step-1']['status']=='COMPLETED'
    assert state['jobs']['eval-step-2']['status']=='COMPLETED'
    assert state['selected']['score']==.5 and state['selected']['score_details']['model_failures']==1
    assert 'không phải điểm chính thức' in (run/'FINAL_RESULTS.vi.md').read_text()
    assert state['jobs']['submission']['status']=='COMPLETED'
    assert state['jobs']['audio-baseline']['status']=='COMPLETED'
    assert state['jobs']['audio-prompt-itn']['feature_status']=='EXECUTED'
    assert not (run/'candidates/step-1/optimizer_state.pt').exists()
    assert not runner.same_process(state['jobs']['baseline']['pid'],state['jobs']['baseline']['process_start'])
    capsys.readouterr()
    assert _invoke(monkeypatch,runner,config,'--status')==0
    assert json.loads(capsys.readouterr().out)['worker_alive'] is False


def test_detached_stop_then_exact_resume_with_synthetic_jobs(tmp_path,monkeypatch):
    runner,run,config=_setup(tmp_path,monkeypatch)
    run.mkdir();(run/'hold').write_text('hold')
    assert _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')==0
    end=time.monotonic()+12
    while time.monotonic()<end and not (run/'baseline/checkpoints/lora/step-1/training_state.json').exists(): time.sleep(.1)
    assert (run/'baseline/checkpoints/lora/step-1/training_state.json').exists()
    assert _invoke(monkeypatch,runner,config,'--stop')==0
    first=_wait_status(run,{'INTERRUPTED','FAILED'})
    assert first['status']=='INTERRUPTED', first
    assert _invoke(monkeypatch,runner,config,'--resume')==0
    final=_wait_status(run,{'COMPLETED_WITH_SKIPS','FAILED'})
    assert final['status']=='COMPLETED_WITH_SKIPS', final
    assert '--resume' in final['jobs']['baseline']['command']
    assert final['jobs']['baseline']['status']=='COMPLETED'


def test_detached_job_failure_is_recorded_and_not_retried(tmp_path,monkeypatch):
    runner,run,config=_setup(tmp_path,monkeypatch)
    run.mkdir();(run/'fail').write_text('fail')
    assert _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')==0
    state=_wait_status(run,{'FAILED'})
    assert state['jobs']['baseline']['exit_code']==7
    assert state['jobs']['baseline']['status']=='FAILED'
    assert (run/'baseline/checkpoints/lora/step-1/training_state.json').exists()
    assert (run/'candidates/step-1/adapter_model.safetensors').exists()
    assert (run/'FINAL_RESULTS.vi.md').exists()
    with pytest.raises(RuntimeError,match='refusing automatic retry'):
        _invoke(monkeypatch,runner,config,'--resume')


def test_independent_sft_variant_uses_distinct_dataset_and_horizon(tmp_path,monkeypatch):
    runner,run,config=_setup(tmp_path,monkeypatch)
    cfg=json.loads(config.read_text());cfg['sft_variants']=['slot_masking'];config.write_text(json.dumps(cfg))
    assert _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')==0
    state=_wait_status(run,{'COMPLETED_WITH_SKIPS','COMPLETED','FAILED'},timeout=45)
    assert state['status'] in {'COMPLETED_WITH_SKIPS','COMPLETED'}, state
    variant=state['variants']['slot_masking']
    assert variant['horizon']==3 and variant['augmentation_count']==2
    assert state['jobs']['variant-slot_masking']['status']=='COMPLETED'
    assert '--resume' not in state['jobs']['train-slot_masking']['command']
    assert '/variants/slot_masking/' in variant['dataset']
    assert (run/'variants/slot_masking/candidates/step-3/adapter_model.safetensors').exists()


def test_global_gpu_lock_prevents_second_worker(tmp_path,monkeypatch):
    import fcntl
    runner,run,config=_setup(tmp_path,monkeypatch)
    lock_path=runner.ROOT/'outputs'/'.lifemate_gpu.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='before startup handshake'):
            _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')
        state=json.loads((run/'state.json').read_text())
        assert state['status']=='FAILED' and state.get('jobs')=={}
        assert not (run/'baseline').exists()


def test_gated_worker_runs_reviewed_synthetic_pipeline(tmp_path,monkeypatch):
    runner,run,config=_setup(tmp_path,monkeypatch)
    cfg=json.loads(config.read_text()); cfg['ai_review_gate']={'enabled':True,'timeout_seconds':5}; config.write_text(json.dumps(cfg))
    original=runner.prepare_worktree
    def gated_tree(path,revision):
        target=original(path,revision)
        harness=target/'scripts/run_overnight.py'
        content=harness.read_text()
        inject=f'''\nsys.path.insert(0,{str(SOURCE.parents[1]/"scripts")!r})\nimport ai_review_gate\ndef fake_review(self,event,facts,output,log):\n return {{"event_id":event["id"],"experiment_id":event["job"],"verdict":"MODEL_UNDERPERFORMANCE" if event["kind"]=="AFTER_EVAL" else "PASS","confidence":"HIGH","evidence":["synthetic job artifact"],"root_cause":None,"affected_components":[],"recommended_action":"continue","decision":"CONTINUE","retraining_necessary":False,"checkpoint_valid":True}}\nai_review_gate.ReviewGate._invoke=fake_review\n'''
        harness.write_text(content.replace('sys.exit(m.main())',inject+'\nsys.exit(m.main())'))
        return target
    monkeypatch.setattr(runner,'prepare_worktree',gated_tree)
    assert _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')==0
    state=_wait_status(run,{'COMPLETED_WITH_SKIPS','FAILED','REVIEW_BLOCKED'},timeout=45)
    assert state['status']=='COMPLETED_WITH_SKIPS',state
    records=[json.loads(p.read_text()) for p in (run/'ai_gate').glob('*.json')]
    assert records and all(r['status']=='APPROVED' for r in records)
    assert any(r['event']['kind']=='AFTER_TRAIN' for r in records)
    assert any(r['review']['verdict']=='MODEL_UNDERPERFORMANCE' for r in records)
    assert state['selected']['score']==.5


def test_gated_worker_blocks_before_training_on_inconclusive_review(tmp_path,monkeypatch):
    runner,run,config=_setup(tmp_path,monkeypatch)
    cfg=json.loads(config.read_text()); cfg['ai_review_gate']={'enabled':True,'timeout_seconds':5}; config.write_text(json.dumps(cfg))
    original=runner.prepare_worktree
    def gated_tree(path,revision):
        target=original(path,revision); harness=target/'scripts/run_overnight.py'; content=harness.read_text()
        inject=f'''\nsys.path.insert(0,{str(SOURCE.parents[1]/"scripts")!r})\nimport ai_review_gate\ndef reject(self,event,facts,output,log):\n return {{"event_id":event["id"],"experiment_id":event["job"],"verdict":"INCONCLUSIVE","confidence":"HIGH","evidence":["missing evidence"],"root_cause":None,"affected_components":[],"recommended_action":"investigate","decision":"INVESTIGATE","retraining_necessary":None,"checkpoint_valid":None}}\nai_review_gate.ReviewGate._invoke=reject\n'''
        harness.write_text(content.replace('sys.exit(m.main())',inject+'\nsys.exit(m.main())')); return target
    monkeypatch.setattr(runner,'prepare_worktree',gated_tree)
    try:
        _invoke(monkeypatch,runner,config,'--start','--stop-at','2099-01-01T07:00:00+07:00')
    except RuntimeError as exc:
        assert 'startup handshake' in str(exc)
    state=_wait_status(run,{'REVIEW_BLOCKED','FAILED'},timeout=10)
    assert state['status']=='REVIEW_BLOCKED',state
    assert not (run/'baseline').exists()
