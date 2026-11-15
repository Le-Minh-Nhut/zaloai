import importlib.util,json
from pathlib import Path
import pytest
from lifemate.errors import LifeMateError
P=Path(__file__).resolve().parents[1]/'scripts/transcribe_full_multi.py'
spec=importlib.util.spec_from_file_location('full_multi_test',P);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def line(row):
    return json.dumps({**row,'record_sha256':mod.digest(row)})+'\n'

def test_resume_discards_incomplete_tail_and_retains_success(tmp_path):
    p=tmp_path/'rows';row={'sample_id':'a','turn_id':'current','run_identity':'r','status':'executed'}
    p.write_text(line(row)+'{"incomplete"')
    assert mod.read_journal(p,'r')[('a','current')]==row
    assert p.read_text()==line(row)

def test_resume_rejects_corruption_or_duplicate_success(tmp_path):
    p=tmp_path/'rows';row={'sample_id':'a','turn_id':'current','run_identity':'r','status':'executed'}
    p.write_text(line(row)+line(row))
    with pytest.raises(LifeMateError,match='duplicates'):mod.read_journal(p,'r')
    p.write_text(line(row).replace('executed','failed'))
    with pytest.raises(LifeMateError,match='checksum'):mod.read_journal(p,'r')

def test_retry_can_replace_failure_but_not_success(tmp_path):
    p=tmp_path/'rows';row={'sample_id':'a','turn_id':'current','run_identity':'r','status':'failed'}
    success={**row,'status':'executed'};p.write_text(line(row)+line(success))
    assert mod.read_journal(p,'r')[('a','current')]['status']=='executed'

def test_merge_preserves_failures_and_successes(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,str(P.parent))
    from lifemate.asr.conditioned_records import produce_multi_records
    from lifemate.asr.top1_models import MODELS
    from lifemate.data.conditioned_sft import load_records
    from lifemate.asr.conditioned_records import _hash
    from lifemate.features.config import FeatureConfig
    merge_spec=importlib.util.spec_from_file_location('full_merge_test',P.parent/'merge_full_multi.py');merge=importlib.util.module_from_spec(merge_spec);merge_spec.loader.exec_module(merge)
    audio=tmp_path/'a.wav';audio.write_bytes(b'audio');sample=SimpleNamespace(id='a',audio_path=audio,history=[])
    backend=SimpleNamespace(backends={'gipformer_68m':SimpleNamespace(name='mock',model_id=MODELS['gipformer_68m'][1],revision='r')})
    def transcribe(*a,**k):
        return 'top',{'multi_asr_results':[{'model':'gipformer_68m','model_id':MODELS['gipformer_68m'][1],'status':'executed','transcript':'top'}],'conditioning':{'multi_asr':{'degraded':False}}}
    basepath=tmp_path/'base.jsonl';produce_multi_records([sample],backend,FeatureConfig(),split='train',data_root=tmp_path,output=basepath,transcribe=transcribe)
    base,bmeta=load_records(basepath);paths={('a','current'):audio};identities={n:{'model_id':MODELS[n][1],'revision':mod.REV[n],'artifacts':[]} for n in mod.REV};pmeta={'sha256':'pilot','identity':{'models':identities,'features':bmeta['identity']['features'],'decoders':{'gipformer_68m':'baseline',**mod.DEC}}}
    monkeypatch.setattr(merge,'validate_sources',lambda root,name:(base,bmeta,{},pmeta,paths,tmp_path))
    work=tmp_path/'out';work.mkdir()
    for n in mod.REV:
        d=work/n;d.mkdir(); identity={'source_artifact_sha256':bmeta['sha256'],'pilot_sha256':'pilot','model_id':MODELS[n][1],'revision':mod.REV[n],'decoder':mod.DEC[n]};(d/'identity.json').write_text(json.dumps(identity));runid=mod.digest(identity)
        failed=n=='chunkformer_rnnt';row={'sample_id':'a','turn_id':'current','audio_sha256':base[('a','current')]['audio_sha256'],'model':n,'model_id':MODELS[n][1],'revision':mod.REV[n],'decoder':mod.DEC[n],'run_identity':runid,'status':'failed' if failed else 'executed','error':'failed decode' if failed else None,'transcript':'' if failed else 'speech'}
        journal=d/'transcripts.jsonl';journal.write_text(line(row));(d/'completion.json').write_text(json.dumps({'run_identity':runid,'journal_sha256':_hash(journal),'processed_count':1}))
    merge.merge(tmp_path,work);rows,meta=load_records(work/'four_asr_train.jsonl');row=rows[('a','current')]
    assert len(row['models'])==4 and row['status']=='degraded' and row['models'][-1]['error']=='failed decode'
    assert meta['success_coverage']=={'3':1}
