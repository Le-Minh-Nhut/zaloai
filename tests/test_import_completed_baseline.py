import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/import_completed_baseline.py'


def test_refuses_active_worker_without_writing_destination(tmp_path):
    source=tmp_path/'source'; source.mkdir(); destination=tmp_path/'new-run'
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    (source/'state.json').write_text(json.dumps({'worker_pid':os.getpid(),'worker_start':int(stat[19])}))
    cfg=tmp_path/'config.json'; cfg.write_text(json.dumps({'work_dir':str(destination),'ai_review_gate':{'enabled':True}}))
    result=subprocess.run([sys.executable,str(SCRIPT),'--old-run',str(source),'--config',str(cfg),'--stop-at','2099-01-01T00:00:00+07:00'],capture_output=True,text=True)
    assert result.returncode != 0 and 'still running' in result.stderr
    assert not destination.exists()


def test_imports_completed_baseline_without_training(tmp_path,monkeypatch,capsys):
    import importlib.util
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec=importlib.util.spec_from_file_location('import_complete',SCRIPT)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old=tmp_path/'old'; old.mkdir(); dataset=tmp_path/'dataset'; dataset.mkdir(); model=tmp_path/'model'; model.mkdir()
    for name in ('manifest.json','train.jsonl','val.jsonl'): (dataset/name).write_text(name)
    (model/'config.json').write_text('model')
    c={'work_dir':str(tmp_path/'new'),'dataset':str(dataset),'model':str(model),'ai_review_gate':{'enabled':True}}
    previous={'work_dir':str(old),'dataset':str(dataset),'model':str(model)}
    cand=old/'candidates/step-1';cand.mkdir(parents=True)
    (cand/'adapter_config.json').write_text('{}'); (cand/'adapter_model.safetensors').write_text('adapter')
    digest=module.r.sha(cand/'adapter_model.safetensors')
    (cand/'candidate_identity.json').write_text(json.dumps({'adapter_sha256':digest}))
    original={'worker_pid':None,'jobs':{'baseline':{'status':'COMPLETED','exit_code':0}},'config':previous,
              'heldout':{'ids_sha256':'ids'},'expected_horizon':2,'git_sha':'old',
              'dataset_sha256':{n:module.r.sha(dataset/n) for n in ('manifest.json','train.jsonl','val.jsonl')},
              'model_sha256':{'config.json':module.r.sha(model/'config.json')}}
    (old/'state.json').write_text(json.dumps(original)); cfg=tmp_path/'cfg.json';cfg.write_text(json.dumps(c))
    monkeypatch.setattr(module.r,'preflight',lambda c:{'ids_sha256':'ids','expected_horizon':2})
    monkeypatch.setattr(module.r,'git_sha',lambda:'new')
    def source(path,revision): path.mkdir(); return path
    monkeypatch.setattr(module.r,'prepare_worktree',source)
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--old-run',str(old),'--config',str(cfg),'--stop-at','2099-01-01T00:00:00+07:00'])
    module.main()
    result=json.loads((Path(c['work_dir'])/'state.json').read_text())
    assert result['jobs']['baseline']['status']=='COMPLETED' and result['imported_baseline']['no_retraining']
    assert (Path(c['work_dir'])/'candidates/step-1/adapter_model.safetensors').read_text()=='adapter'
    assert json.loads((old/'state.json').read_text())==original
