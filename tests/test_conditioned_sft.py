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
