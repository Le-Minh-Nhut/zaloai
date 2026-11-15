import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from lifemate.asr.conditioned_records import produce_multi_records
from lifemate.asr.interface import AsrError
from lifemate.asr.nbest import AsrHypothesis
from lifemate.asr.top1_models import MODELS, select_llm_candidates
from lifemate.data.conditioned_sft import load_records, augment_conditioned, coverage_audit
from lifemate.agent.prompt_builder import _render_request
from lifemate.errors import LifeMateError
from lifemate.features.config import FeatureConfig


class Model:
    name='mock'
    model_id='mock-id'
    revision='test'


def producer_fixture(tmp_path):
    data=tmp_path/'data'; audio=data/'train/audio/s';audio.mkdir(parents=True)
    current=audio/'s.wav';current.write_bytes(b'current')
    history=audio/'h.wav';history.write_bytes(b'history')
    sample=SimpleNamespace(id='s',audio_path=current,history=[{'role':'user','audio':'audio/s/h.wav'}])
    backend=SimpleNamespace(backends={name:Model() for name in MODELS})
    features=FeatureConfig()
    def transcribe(backend,sample,features):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':'alarm.create 07:00','raw_transcript':'alarm.create 07:00'} for name in MODELS]
        return 'alarm.create 07:00',{'multi_asr_results':models,'hypotheses':[], 'conditioning':{'multi_asr':{'degraded':False}},'latency_s':.1}
    return data,sample,backend,features,transcribe


def test_producer_writes_history_and_resume(tmp_path):
    data,sample,backend,features,transcribe=producer_fixture(tmp_path)
    output=tmp_path/'records.jsonl';calls=[]
    def interrupted(*a,**kw):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('simulated interruption')
        return transcribe(*a,**kw)
    with pytest.raises(RuntimeError):produce_multi_records([sample],backend,features,split='train',data_root=data,output=output,transcribe=interrupted)
    meta=produce_multi_records([sample],backend,features,split='train',data_root=data,output=output,transcribe=transcribe,resume=True)
    idx,_=load_records(output)
    assert meta['records']==2 and list(idx)==[('s','history:0'),('s','current')]
    assert len(idx[('s','current')]['models'])==4
    again=produce_multi_records([sample],backend,features,split='train',data_root=data,output=output,transcribe=lambda *a,**k: (_ for _ in ()).throw(AssertionError('unexpected decode')),resume=True)
    assert again['sha256']==meta['sha256']


def test_k_four_is_new_mode_only():
    candidates=[AsrHypothesis(text=f'text {i}',source=f'm{i}',provenance=(f'm{i}',)) for i in range(4)]
    with pytest.raises(AsrError): select_llm_candidates(candidates,itn_enabled=False,limit=4)
    assert len(select_llm_candidates(candidates,itn_enabled=False,limit=4,conditioned_sft=True))==4
    assert len(select_llm_candidates(candidates,itn_enabled=False,limit=2,conditioned_sft=True))==2


def prompt(request):
    return '# Yêu cầu\n'+request+'\n\n# Ngữ cảnh\nkhông có\n\n# Lịch sử hội thoại\nkhông có\n\n# Công cụ khả dụng\nalarm.create\n\n# Định dạng trả lời\n<action>'


def test_coverage_and_gold_preservation(tmp_path):
    data,sample,backend,features,transcribe=producer_fixture(tmp_path);sample.history=[]
    output=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=output,transcribe=transcribe)
    index,_=load_records(output)
    target='<action>{"type":"tool_call","name":"alarm.create","arguments":{"time":"07:00"}}</action>'
    clean={'sample_id':'s','messages':[{'role':'user','content':prompt('original')}],'target':target,'meta':{}}
    produced,counts=augment_conditioned([clean],[sample],index,k=4)
    assert counts['SUPPORTED']==1 and counts['created']==1
    assert produced[0]==clean and produced[1]['target']==target
    assert produced[1]['messages'][-1]['content'].startswith(_render_request('alarm.create 07:00',[AsrHypothesis(text='alarm.create 07:00',source='gipformer_68m',provenance=tuple(MODELS))],nbest_format=True,limit=4,conditioned_multi_sft=True))
    conflict=prompt('09:00');assert coverage_audit(target,conflict)['status']=='UNSUPPORTED'
    sample.history=[{'role':'user','content':'gold private transcript'}]
    _,report=augment_conditioned([clean],[sample],index,k=4)
    assert report['history_excluded']==1 and report['created']==0


def test_prepare_dataset_cli_end_to_end_synthetic(tmp_path):
    import os, shutil, subprocess, sys
    from lifemate.data.loader import load_samples
    root=Path(__file__).resolve().parents[1]
    data=tmp_path/'fixture';shutil.copytree(root/'tests/fixtures/synthetic_lifemate',data)
    source=data/'train/train.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    for row in rows: row['history']=[]
    source.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    samples=load_samples(data,'train')
    for sample in samples:
        sample.audio_path.parent.mkdir(parents=True,exist_ok=True);sample.audio_path.write_bytes(b'synthetic-audio-'+sample.id.encode())
    backend=SimpleNamespace(backends={name:Model() for name in MODELS})
    features=FeatureConfig()
    def transcribe(backend,sample,features):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':f'nhận dạng {i}','raw_transcript':f'nhận dạng {i}'} for i,name in enumerate(MODELS)]
        return 'nhận dạng 0',{'multi_asr_results':models,'hypotheses':[],'conditioning':{'multi_asr':{'degraded':False}}}
    artifact=tmp_path/'multi.jsonl'
    produce_multi_records(samples,backend,features,split='train',data_root=data,output=artifact,transcribe=transcribe)
    output=tmp_path/'datasets'
    env=os.environ.copy();env['PYTHONPATH']=str(root/'src');env['WANDB_MODE']='disabled'
    result=subprocess.run([sys.executable,str(root/'scripts/prepare_dataset.py'),'--data-root',str(data),'--split','train','--output',str(output),'--config',str(root/'configs/training_multi_asr_conditioned.yaml'),'--conditioned-multi-records',str(artifact),'--conditioned-unknown','include'],cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    manifest=json.loads((output/'train/manifest.json').read_text()); assert manifest['conditioned_multi_asr']['coverage']['created']>0
    train=[json.loads(x) for x in (output/'train/train.jsonl').read_text().splitlines()]
    val=[json.loads(x) for x in (output/'train/val.jsonl').read_text().splitlines()]
    assert {x['sample_id'] for x in train}.isdisjoint({x['sample_id'] for x in val})
    for variant in [x for x in train if x.get('meta',{}).get('augmentation')=='conditioned_multi_asr']:
        originals=[x for x in train if x['sample_id']==variant['sample_id'] and x['step_index']==variant['step_index'] and not x.get('meta',{}).get('augmentation')]
        assert originals and originals[0]['target']==variant['target']


def test_train_and_inference_prompt_bytes_match():
    from lifemate.agent.prompt_builder import build_decision_prompt
    from lifemate.data.augmentation import replace_first_section
    context={'current_time':'2026-01-01','installed_apps':[]}
    catalog=SimpleNamespace(specs=[])
    history=[{'role':'assistant','content':'previous'}]
    hypotheses=[AsrHypothesis(text='một',source='a',provenance=('a',)),AsrHypothesis(text='hai',source='b',provenance=('b',))]
    clean=build_decision_prompt(transcript='gold',context=context,history=history,installed_apps=[],catalog=catalog)
    section=_render_request('một',hypotheses,nbest_format=True,limit=4,conditioned_multi_sft=True)
    training=replace_first_section(clean,section)
    inference=build_decision_prompt(transcript='một',context=context,history=history,installed_apps=[],catalog=catalog,asr_hypotheses=hypotheses,nbest_format=True,nbest_limit=4,conditioned_multi_sft=True)
    assert training==inference


def test_partial_corruption_is_never_accepted(tmp_path):
    data,sample,backend,features,transcribe=producer_fixture(tmp_path)
    output=tmp_path/'records.jsonl';count=[]
    def interrupted(*a,**kw):
        count.append(1)
        if len(count)==2:raise RuntimeError('interrupt')
        return transcribe(*a,**kw)
    with pytest.raises(RuntimeError):produce_multi_records([sample],backend,features,split='train',data_root=data,output=output,transcribe=interrupted)
    partial=Path(str(output)+'.partial'); row=json.loads(partial.read_text());row['selected_text']='tampered';partial.write_text(json.dumps(row)+'\n')
    with pytest.raises(LifeMateError,match='checksum'):produce_multi_records([sample],backend,features,split='train',data_root=data,output=output,transcribe=transcribe,resume=True)


def test_unknown_excluded_by_default_without_touching_clean(tmp_path):
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    def transcribe(backend,sample,features):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':'không rõ','raw_transcript':'không rõ'} for name in MODELS]
        return 'không rõ',{'multi_asr_results':models,'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    index,_=load_records(path);clean={'sample_id':'s','messages':[{'role':'user','content':prompt('gold')}],'target':'<action>{"type":"finish"}</action>','meta':{}}
    result,stats=augment_conditioned([clean],[sample],index,k=4)
    assert result==[clean] and stats['UNKNOWN']==1 and stats['created']==0
    assert stats['decisions'][0]['audit']['reasons']


def test_context_can_supply_value_without_false_conflict():
    text=prompt('đặt báo thức 09:00').replace('# Ngữ cảnh\nkhông có','# Ngữ cảnh\nGiờ hẹn trước 07:00')
    target='<action>{"type":"tool_call","name":"alarm.create","arguments":{"time":"07:00"}}</action>'
    assert coverage_audit(target,text)['status']=='UNKNOWN'


def test_history_audio_replaces_gold_and_preserves_tool_observations(tmp_path):
    from lifemate.agent.prompt_builder import build_decision_prompt
    data,sample,backend,features,_=producer_fixture(tmp_path)
    extra=data/'train/audio/s/h2.wav';extra.write_bytes(b'history2')
    sample.history=[{'role':'user','audio':'audio/s/h.wav','audio_text':'GOLD SECRET ONE'},
                    {'role':'assistant','content':'trợ lý trước'},
                    {'role':'tool','content':'kết quả thật'},
                    {'role':'user','audio':'audio/s/h2.wav','audio_text':'GOLD SECRET TWO'}]
    def transcribe(backend,sample,features):
        text='asr '+sample.id.split(':')[-1]
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':text,'raw_transcript':text} for name in MODELS]
        return text,{'multi_asr_results':models,'conditioning':{'multi_asr':{'degraded':False}}}
    artifact=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=artifact,transcribe=transcribe)
    index,_=load_records(artifact)
    original=build_decision_prompt(transcript='gold current',context={},history=sample.history,installed_apps=[],catalog=SimpleNamespace(specs=[]))
    record={'sample_id':'s','messages':[{'role':'user','content':original}],'target':'<action>{"type":"finish"}</action>','meta':{}}
    out,counts=augment_conditioned([record],[sample],index,k=4,unknown_policy='include',data_root=data)
    assert counts['created']==1
    rendered=out[1]['messages'][-1]['content']
    assert 'GOLD SECRET' not in rendered and 'asr 0' in rendered and 'asr 3' in rendered
    assert 'trợ lý trước' in rendered and 'kết quả thật' in rendered
    assert out[1]['target']==record['target']


def test_k4_requires_conditioned_mode_and_preserves_four():
    from lifemate.asr.multi_backend import MultiAsrBackend
    from lifemate.asr.interface import TranscriptResult
    class Backend:
        name='mock'
        def __init__(self,text):self.text=text
        def transcribe(self,*a,**kw):return TranscriptResult(text=self.text,backend='mock')
    models={str(i):Backend(f'text {i}') for i in range(4)}
    with pytest.raises(AsrError):MultiAsrBackend(models['0'],enabled=True,backends=models,max_hypotheses=4)
    multi=MultiAsrBackend(models['0'],enabled=True,backends=models,max_hypotheses=4,conditioned_sft=True)
    result=multi.transcribe_multi('audio')
    assert len(result.hypotheses)==4 and [h.rank for h in result.hypotheses]==[0,1,2,3]


def test_forced_history_evaluation_removes_only_user_gold():
    from lifemate.data.history import clear_user_history_transcripts
    turns=[{'role':'user','audio':'audio/a.wav','audio_text':'GOLD','content':'GOLD'},
           {'role':'assistant','content':'previous response'}, {'role':'tool','content':'observation'}]
    sample=SimpleNamespace(history=turns)
    clear_user_history_transcripts(sample)
    assert turns[0]=={'role':'user','audio':'audio/a.wav'}
    assert turns[1]['content']=='previous response' and turns[2]['content']=='observation'


@pytest.mark.parametrize('texts,expected',[
    (['same']*4,1),(['a','b','c','d'],4),(['a','a','b','c'],3),(['wrong','wrong','wrong','right'],2),
])
def test_candidate_disagreement_cases(texts,expected):
    candidates=[AsrHypothesis(text=text,source=f'm{i}',provenance=(f'm{i}',),model_rank=1,model_priority=i) for i,text in enumerate(texts)]
    chosen=select_llm_candidates(candidates,itn_enabled=False,limit=4,conditioned_sft=True)
    assert len(chosen)==expected
    if texts.count('wrong')==3:assert chosen[0].provenance==('m0','m1','m2') and chosen[1].text=='right'


def test_all_topk_candidate_counts_and_priority():
    candidates=[AsrHypothesis(text=str(i),source=f'm{i}',provenance=(f'm{i}',)) for i in range(4)]
    for k in (1,2,3,4):
        selected=select_llm_candidates(list(reversed(candidates)),itn_enabled=False,limit=k,conditioned_sft=True)
        assert [h.text for h in selected]==list(reversed([str(i) for i in range(4)]))[:k]
        rendered=_render_request(selected[0].text,selected,nbest_format=True,limit=k,conditioned_multi_sft=True)
        if k==1:assert rendered=='# Yêu cầu\n3'
        else:assert all(h.text in rendered for h in selected)


def test_mixed_single_multi_variants_are_grouped_and_deterministic(tmp_path):
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    def transcribe(backend,sample,features):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':f'text {i}','raw_transcript':f'text {i}'} for i,name in enumerate(MODELS)]
        return 'text 0',{'multi_asr_results':models,'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    index,_=load_records(path);clean={'sample_id':'s','step_index':1,'messages':[{'role':'user','content':prompt('gold')}],'target':'<action>{"type":"finish"}</action>','meta':{}}
    out,stats=augment_conditioned([clean],[sample],index,k=4,mode='mixed',unknown_policy='include')
    repeat,_=augment_conditioned([clean],[sample],index,k=4,mode='mixed',unknown_policy='include')
    assert out==repeat and len(out)==3 and stats['created']==2
    assert [item['sample_id'] for item in out]==['s','s','s'] and all(item['target']==clean['target'] for item in out)
    assert [item['meta'].get('augmentation') for item in out]==[None,'conditioned_single_asr','conditioned_multi_asr']
    only_single,_=augment_conditioned([clean],[sample],index,k=4,mode='mixed',unknown_policy='include',multi_ratio=0)
    assert len(only_single)==2 and only_single[1]['meta']['augmentation']=='conditioned_single_asr'


def test_comparison_rejects_noncomparable_cases(monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/compare_conditioned_experiments.py'
    spec=importlib.util.spec_from_file_location('compare_conditioned',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def payload(ids):
        return {'closed_loop':{'heldout_selection':{'ids':ids},'evaluation_mode':'audio_to_action','runtime_protocol':'quality_asr_latency_excluded','history_protocol':'forced_audio_asr','rows':[{'id':i} for i in ids]}}
    monkeypatch.setattr(module,'score_evaluation',lambda p,ids,samples:{'value':0.0,'n':len(ids),'official_score':'UNAVAILABLE'})
    result=module.compare({'E0':payload(['a']),'E1':payload(['a'])},[])
    assert result['experiments']['E0']['official_score']=='UNAVAILABLE'
    with pytest.raises(ValueError,match='differ'):module.compare({'E0':payload(['a']),'E1':payload(['b'])},[])
    bad=payload(['a']);bad['closed_loop']['degraded_features']=['asr.multi_asr']
    with pytest.raises(ValueError,match='degraded'):module.compare({'E0':payload(['a']),'E1':bad},[])


def test_asr_report_uses_only_present_train_references():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/report_multi_asr.py'
    spec=importlib.util.spec_from_file_location('report_multi',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    index={('s','current'):{'models':[{'model':'a','status':'executed','transcript':'xin chào','cache_hit':True,'latency_ms':10},{'model':'b','status':'failed'}]}}
    result=module.summarize(index,[SimpleNamespace(id='s',audio_text='xin chào')])
    assert result['models']['a']['asr']['available'] and result['models']['a']['cache_hits']==1
    assert result['models']['b']['failed']==1 and result['official_score']=='UNAVAILABLE'


def test_single_model_does_not_require_other_models(tmp_path):
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    def transcribe(*args, **kwargs):
        models=[{'model':name,'model_id':MODELS[name][1], 'status':'executed' if i==1 else 'failed',
                 'transcript':'đặt báo thức 07:00' if i==1 else ''} for i,name in enumerate(MODELS)]
        return 'đặt báo thức 07:00', {'multi_asr_results':models,'conditioning':{'multi_asr':{'degraded':True}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    index,_=load_records(path)
    clean={'sample_id':'s','messages':[{'role':'user','content':prompt('gold')}], 'target':'<action>{"type":"tool_call","name":"alarm.create","arguments":{"time":"07:00"}}</action>', 'meta':{}}
    out,counts=augment_conditioned([clean],[sample],index,mode='mixed',single_model='gipformer_15')
    assert len(out)==2 and counts['created']==1 and counts['missing_or_degraded']==1
    assert out[1]['meta']['augmentation']=='conditioned_single_asr'
    out,_=augment_conditioned([clean],[sample],index,mode='single',single_model='gipformer_68m')
    assert out==[clean]


def test_semantic_coverage_counterexamples():
    target='<action>{"type":"tool_call","name":"alarm.create","arguments":{"time":"07:00"}}</action>'
    assert coverage_audit(target,prompt('đặt báo thức lúc 07:00'))['status']=='SUPPORTED'
    assert coverage_audit(target,prompt('đặt báo thức lúc 09:00'))['status']=='UNSUPPORTED'
    assert coverage_audit(target,prompt('đặt báo thức lúc 09:00').replace('không có','đã xác nhận 07:00',1))['status']!='UNSUPPORTED'
    assert coverage_audit(target,prompt('có một cuộc họp lúc 07:00'))['status']=='UNKNOWN'


def test_replay_k_and_identity(tmp_path):
    from lifemate.evaluation.artifact_replay import ArtifactReplay
    data,sample,backend,features,_=producer_fixture(tmp_path)
    for name,model in backend.backends.items(): model.model_id=MODELS[name][1]
    def transcribe(*args,**kwargs):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':f'text {i}'} for i,name in enumerate(MODELS)]
        return 'text 0',{'multi_asr_results':models,'latency_s':123,'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    for k in range(1,5):
        replay=ArtifactReplay(path,[sample],split='train',data_root=data,k=k)
        text,info=replay.current(sample)
        assert text=='text 0' and len(info['hypotheses'])==k and info['historical_asr_latency_s']==123
        replay.history(sample)
        assert sample.history[0]['content']=='text 0'
    sample.audio_path.write_bytes(b'changed')
    with pytest.raises(LifeMateError,match='identity'):ArtifactReplay(path,[sample],split='train',data_root=data,k=1)


def test_conditioned_training_manifest_checksums(tmp_path):
    import importlib.util, hashlib
    path=Path(__file__).resolve().parents[1]/'scripts/train.py'
    spec=importlib.util.spec_from_file_location('train_provenance_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train=tmp_path/'train.jsonl';val=tmp_path/'val.jsonl'
    train.write_text('{"target":"x"}\n');val.write_text('{"target":"y"}\n')
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'conditioned_multi_asr':{'train_sha256':h(train),'validation_sha256':h(val),'artifact_sha256':'abc','prompt_version':'v1'}}))
    assert module.load_records(tmp_path)[0]
    train.write_text('{"target":"changed"}\n')
    with pytest.raises(LifeMateError,match='checksum'):module.load_records(tmp_path)


def test_replay_runs_evaluator_with_history_and_hypotheses(tmp_path):
    from lifemate.evaluation.artifact_replay import ArtifactReplay
    from lifemate.evaluation.evaluator import evaluate_closed_loop
    data,sample,backend,features,_=producer_fixture(tmp_path)
    for name,model in backend.backends.items(): model.model_id=MODELS[name][1]
    def decode(*args,**kwargs):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':f'variant {i}'} for i,name in enumerate(MODELS)]
        return 'variant 0',{'multi_asr_results':models,'latency_s':99,'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=decode)
    replay=ArtifactReplay(path,[sample],split='train',data_root=data,k=4)
    text,info=replay.current(sample);seen=[]
    class Env:
        def get_tool_calls(self):return []
        def get_state_hash(self):return 'official-test-env-hash'
    class Agent:
        env=Env()
        def run(self, sample, **kwargs):
            seen.append((sample.history[0]['content'],kwargs))
            return SimpleNamespace(status='finished')
    result=evaluate_closed_loop(lambda s:Agent(),[sample],split='train',data_root=data,evaluation_mode='audio_to_action',transcripts={'s':text},asr_info={'s':info},prepare_history=replay.history)
    assert seen[0][0]=='variant 0' and len(seen[0][1]['asr_hypotheses'])==4
    assert result['rows'][0]['state_hash']=='official-test-env-hash'
    assert result['runtime_protocol']=='quality_asr_latency_excluded'
    assert result['rows'][0]['current_asr_seconds']==0


def test_semantic_itn_and_ambiguous_date():
    target='<action>{"type":"tool_call","name":"alarm.create","arguments":{"time":"07:00"}}</action>'
    assert coverage_audit(target,prompt('đặt báo thức lúc bảy giờ sáng'))['status']=='SUPPORTED'
    assert coverage_audit(target,prompt('đặt báo thức lúc bảy giờ'))['status']=='UNKNOWN'
    dated='<action>{"type":"tool_call","name":"alarm.create","arguments":{"date":"2026-11-20"}}</action>'
    assert coverage_audit(dated,prompt('đặt báo thức ngày hai mươi tháng mười một'))['status']=='UNKNOWN'


def test_conservative_reference_equivalence_rejects_numbers_and_signs():
    from lifemate.data.conditioned_sft import _equivalent_transcript
    assert _equivalent_transcript('Mở nhạc, nhé!', 'mở nhạc nhé')
    assert not _equivalent_transcript('chuyển 10.000 đồng', 'chuyển 10000 đồng')
    assert not _equivalent_transcript('trừ -5', 'trừ 5')
    assert not _equivalent_transcript('bảy giờ', 'chín giờ')


def test_exact_gold_surface_is_coverage_evidence_only(tmp_path):
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    def transcribe(*args,**kwargs):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':'mở nhạc nhé'} for name in MODELS]
        return 'mở nhạc nhé',{'multi_asr_results':models,'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    idx,_=load_records(path)
    clean={'sample_id':'s','messages':[{'role':'user','content':prompt('Mở nhạc, nhé!')}], 'target':'<action>{"type":"finish"}</action>', 'meta':{}}
    out,stats=augment_conditioned([clean],[sample],idx,mode='single')
    assert len(out)==2 and stats['SUPPORTED']==1
    assert '# Yêu cầu\nmở nhạc nhé' in out[1]['messages'][-1]['content']
    assert 'Mở nhạc, nhé!' not in out[1]['messages'][-1]['content']
    assert out[1]['target']==clean['target']


def test_replay_one_real_model_schema_and_rejects_mismatch(tmp_path):
    from lifemate.evaluation.artifact_replay import ArtifactReplay
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    backend.backends={'gipformer_68m':Model()};backend.backends['gipformer_68m'].model_id=MODELS['gipformer_68m'][1]
    def transcribe(*args,**kwargs):
        model={'model':'gipformer_68m','model_id':MODELS['gipformer_68m'][1],'status':'executed','transcript':'nhắc tôi'}
        return 'nhắc tôi',{'multi_asr_results':[model],'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'single.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    replay=ArtifactReplay(path,[sample],split='train',data_root=data,k=1)
    assert replay.current(sample)[0]=='nhắc tôi'
    sample.audio_path.write_bytes(b'tampered')
    with pytest.raises(LifeMateError,match='identity'):ArtifactReplay(path,[sample],split='train',data_root=data,k=1)


def test_merge_pilot_preserves_real_results_and_rejects_audio_mismatch(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/merge_multi_asr_pilot.py';spec=importlib.util.spec_from_file_location('merge_pilot_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    model=Model();model.model_id=MODELS['gipformer_68m'][1];backend.backends={'gipformer_68m':model}
    def transcribe(*a,**kw):
        return 'genuine a',{'multi_asr_results':[{'model':'gipformer_68m','model_id':model.model_id,'status':'executed','transcript':'genuine a'}], 'conditioning':{'multi_asr':{'degraded':False}}}
    base=tmp_path/'base.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=base,transcribe=transcribe)
    idx,_=load_records(base);sha=idx[('s','current')]['audio_sha256'];paths={};dirs={}
    for name in module.REVISIONS:
        paths[name]=tmp_path/(name+'.jsonl');dirs[name]=tmp_path/name;dirs[name].mkdir()
        paths[name].write_text(json.dumps({'sample_id':'s','turn_id':'current','audio_sha256':sha,'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':'real '+name})+'\n')
    out=tmp_path/'merged.jsonl';module.merge(base,paths,dirs,out);rows,_=load_records(out)
    assert len(rows[('s','current')]['models'])==4 and rows[('s','current')]['models'][0]['transcript']=='genuine a'
    row=json.loads(paths['gipformer_15'].read_text());row['audio_sha256']='bad';paths['gipformer_15'].write_text(json.dumps(row)+'\n')
    with pytest.raises(LifeMateError,match='audio identity'):module.merge(base,paths,dirs,tmp_path/'bad.jsonl')


def test_multi_candidate_exact_gold_is_used_only_as_evidence(tmp_path):
    data,sample,backend,features,_=producer_fixture(tmp_path);sample.history=[]
    def transcribe(*args,**kwargs):
        models=[{'model':name,'model_id':MODELS[name][1],'status':'executed','transcript':text} for name,text in zip(MODELS,['mở đèn nhé','mở nhạc nhé','mở cửa nhé','mở máy nhé'])]
        return 'mở đèn nhé',{'multi_asr_results':models,'conditioning':{'multi_asr':{'degraded':False}}}
    path=tmp_path/'records.jsonl';produce_multi_records([sample],backend,features,split='train',data_root=data,output=path,transcribe=transcribe)
    idx,_=load_records(path);clean={'sample_id':'s','messages':[{'role':'user','content':prompt('Mở nhạc, nhé!')}], 'target':'<action>{"type":"finish"}</action>', 'meta':{}}
    out,stats=augment_conditioned([clean],[sample],idx,mode='multi',k=4)
    assert stats['SUPPORTED']==1 and len(out)==2 and out[1]['target']==clean['target']
    assert 'Mở nhạc, nhé!' not in out[1]['messages'][-1]['content']
