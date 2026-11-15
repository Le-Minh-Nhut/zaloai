import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifemate.asr.interface import AsrError, TranscriptResult
from lifemate.asr.multi_backend import MultiAsrBackend
from lifemate.asr.top1_models import MODELS, build_top1_multi, transcript_records_by_model
from lifemate.asr.top1_cache import CachedTop1
from lifemate.asr.model_specs import GIPFORMER_SPECS, artifacts_for
from lifemate.config import load_config


class Fake:
    name='fake'
    def __init__(self, text='xin chào', error=None): self.text=text; self.error=error; self.calls=0
    def transcribe(self, path, *, sample_id=None, lenient=False):
        self.calls+=1
        return TranscriptResult(text=self.text, error=self.error, backend=self.name)


def test_four_models_top1_dedup_and_provenance():
    models={'a':Fake('Xin chào'),'b':Fake('xin chào'),'c':Fake('một giờ'),'d':Fake('hai giờ')}
    backend=MultiAsrBackend(models['a'],enabled=True,backends=models,max_hypotheses=3)
    result=backend.transcribe_multi('fake.wav',sample_id='s')
    assert len(result.raw_results)==4 and [h.text for h in result.hypotheses]==['Xin chào','một giờ','hai giờ']
    assert result.hypotheses[0].provenance==('a','b')
    assert result.multi_asr and not result.is_nbest and all(m.calls==1 for m in models.values())
    assert all(h.score is None for h in result.hypotheses)
    exported=transcript_records_by_model('s',result.raw_results)
    assert set(exported)==set(models) and exported['a']['sample_id']=='s'


def test_failure_isolation_and_strict():
    models={'a':Fake(''), 'b':Fake('ok')}
    result=MultiAsrBackend(models['a'],enabled=True,backends=models).transcribe_multi('fake')
    assert result.degraded and result.hypotheses[0].text=='ok' and result.raw_results[0]['status']=='failed'
    with pytest.raises(AsrError):
        MultiAsrBackend(models['a'],enabled=True,backends=models,failure_policy='strict').transcribe_multi('fake')


def test_budget_stops_next_model_without_fabricating():
    class Slow(Fake):
        def transcribe(self,*a,**kw): time.sleep(.01); return super().transcribe(*a,**kw)
    a,b=Slow('ok'),Fake('no')
    result=MultiAsrBackend(a,enabled=True,backends={'a':a,'b':b},budget_seconds=.001).transcribe_multi('fake')
    assert result.budget_exceeded and result.degraded and b.calls==0 and result.raw_results[1]['status']=='blocked'


def test_registry_selects_four_without_loading_unselected(monkeypatch,tmp_path):
    calls=[]
    def factory(cfg,*,backend,**kwargs): calls.append((backend,kwargs)); return Fake(backend)
    monkeypatch.setattr('lifemate.asr.top1_models.create_asr_backend',factory)
    specs=[{'name':name} for name in MODELS]
    multi=build_top1_multi(None,specs,cache_dir=tmp_path)
    assert list(multi.backends)==list(MODELS) and [x[0] for x in calls]==['gipformer','gipformer','whisper','chunkformer']
    with pytest.raises(AsrError): build_top1_multi(None,[{'name':'bogus'}])
    config=load_config('configs/multi_asr_four_top1.yaml')
    assert len(config.features.asr.multi_asr_models)==4 and not config.features.asr.nbest_enabled


def test_per_model_cache_hits_and_invalidates_audio_and_artifact(tmp_path):
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    model=tmp_path/'model';model.mkdir();weights=model/'weights';weights.write_bytes(b'one')
    backend=Fake('xin');backend.model_dir=model;backend.model_id='m'; backend.revision='r'
    cached=CachedTop1(backend,tmp_path/'cache')
    assert cached.transcribe(audio,sample_id='s').text=='xin'
    assert cached.transcribe(audio,sample_id='s').text=='xin' and cached.last_cache_hit and backend.calls==1
    weights.write_bytes(b'two changed')
    cached.transcribe(audio,sample_id='s'); assert backend.calls==2
    audio.write_bytes(b'b');cached.transcribe(audio,sample_id='s');assert backend.calls==3


def test_gipformer15_upstream_artifacts_and_pin():
    spec=GIPFORMER_SPECS['gipformer1.5-68M-rnnt']
    assert spec.revision=='dd9227dcd8705c13f33bdbe59728d546ab94480f'
    assert artifacts_for(spec,'int8')['encoder']=='encoder.int8.onnx'
    assert artifacts_for(spec,'fp32')['encoder']=='encoder.onnx'


def test_conditioning_itn_prompt_and_raw_metadata(monkeypatch,tmp_path):
    from lifemate.asr.transcription import infer_module
    from lifemate.agent.prompt_builder import _render_request
    from lifemate.features.config import FeatureConfig
    import lifemate.data.loader as loader
    original=tmp_path/'original.wav';original.write_bytes(b'original')
    conditioned=tmp_path/'conditioned.wav';conditioned.write_bytes(b'conditioned')
    seen=[]
    class Seen(Fake):
        def transcribe(self,path,**kw): seen.append(Path(path)); return super().transcribe(path,**kw)
    models={'a':Seen('nhắc lúc tám giờ'),'b':Seen('nhắc lúc chín giờ')}
    backend=MultiAsrBackend(models['a'],enabled=True,backends=models)
    features=FeatureConfig();features.asr.multi_asr_enabled=True;features.prompt.itn_enabled=True;features.audio.rms_normalization_enabled=True
    infer=infer_module();monkeypatch.setattr(loader,'read_sample_audio',lambda s:original)
    monkeypatch.setattr(infer,'_condition_audio',lambda path,features,directory:(conditioned,{'rms':{'executed':True}}))
    text,info=infer._transcribe(backend,SimpleNamespace(id='s'),features=features)
    assert seen==[conditioned,conditioned] and original.read_bytes()==b'original'
    assert len(info['multi_asr_results'])==2 and info['multi_asr_results'][0]['raw_transcript']=='nhắc lúc tám giờ'
    assert all('normalized_transcript' in item for item in info['multi_asr_results'])
    prompt=_render_request(text,info['hypotheses'],nbest_format=True,limit=3)
    assert all(h.text in prompt for h in info['hypotheses'])


def test_chunkformer_uses_upstream_decode_api_and_local_artifacts(monkeypatch,tmp_path):
    import sys
    from lifemate.asr.chunkformer_backend import ChunkFormerBackend
    path=tmp_path/'model';path.mkdir()
    for name in ('config.yaml','global_cmvn','pytorch_model.pt','vocab.txt'): (path/name).write_text('x')
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    seen=[]
    class Model:
        @classmethod
        def from_pretrained(cls,p): seen.append(('load',p));return cls()
        def to(self,device): return self
        def eval(self): return self
        def endless_decode(self,**kwargs): seen.append(('decode',kwargs));return 'xin chào'
    monkeypatch.setitem(sys.modules,'chunkformer',SimpleNamespace(ChunkFormerModel=Model))
    backend=ChunkFormerBackend(path)
    assert backend.transcribe(audio).text=='xin chào' and backend.transcribe(audio).text=='xin chào'
    assert sum(x[0]=='load' for x in seen)==1 and seen[1][1]['return_timestamps'] is False


def test_evaluation_cache_preserves_four_raw_results(monkeypatch,tmp_path):
    import lifemate.data.loader as loader
    from lifemate.asr.transcription import transcribe_samples_cached
    from lifemate.features.config import FeatureConfig
    audio=tmp_path/'x.wav';audio.write_bytes(b'audio')
    sample=SimpleNamespace(id='s',audio_path=audio)
    monkeypatch.setattr(loader,'read_sample_audio',lambda s:audio)
    models={f'm{i}':Fake(f'text {i}') for i in range(4)}
    backend=MultiAsrBackend(models['m0'],enabled=True,backends=models)
    features=FeatureConfig();features.asr.multi_asr_enabled=True
    cfg=SimpleNamespace(asr=SimpleNamespace(backend='gipformer'),features=features)
    cache=tmp_path/'transcripts.json'
    _,first=transcribe_samples_cached([sample],cfg,asr=backend,features=features,cache_path=cache)
    _,second=transcribe_samples_cached([sample],cfg,asr=backend,features=features,cache_path=cache)
    assert len(first['asr_info']['s']['multi_asr_results'])==4
    assert second['cache_hit']==1 and second['asr_info']==first['asr_info']
    assert all(m.calls==1 for m in models.values())


def test_per_model_cache_invalidates_decoder_and_preserves_other_model(tmp_path):
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    a,b=Fake('a'),Fake('b')
    for model,label in ((a,'a'),(b,'b')): model.model_id=label;model.decoding_method='greedy'
    aa,bb=CachedTop1(a,tmp_path/'cache'),CachedTop1(b,tmp_path/'cache')
    for m in (aa,bb):m.transcribe(audio,sample_id='s')
    a.decoding_method='beam'
    for m in (aa,bb):m.transcribe(audio,sample_id='s')
    assert a.calls==2 and b.calls==1 and bb.last_cache_hit


def test_strict_budget_reports_overrun_even_on_last_model():
    class Slow(Fake):
        def transcribe(self,*a,**kw):time.sleep(.01);return super().transcribe(*a,**kw)
    a=Slow('ok')
    with pytest.raises(AsrError,match='budget exceeded'):
        MultiAsrBackend(a,enabled=True,backends={'a':a},budget_seconds=.001,failure_policy='strict').transcribe_multi('x')


def test_prompt_shows_combined_provenance_without_fake_nbest():
    from lifemate.agent.prompt_builder import _render_request
    models={'a':Fake('one'),'b':Fake('one'),'c':Fake('two')}
    result=MultiAsrBackend(models['a'],enabled=True,backends=models).transcribe_multi('x')
    prompt=_render_request('one',result.hypotheses,nbest_format=True,limit=3)
    assert 'a, b' in prompt and 'two' in prompt and not result.is_nbest


def test_all_failures_are_reported_to_caller(monkeypatch,tmp_path):
    from lifemate.asr.transcription import infer_module
    from lifemate.features.config import FeatureConfig
    import lifemate.data.loader as loader
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    monkeypatch.setattr(loader,'read_sample_audio',lambda sample:audio)
    models={'a':Fake(''),'b':Fake('')};backend=MultiAsrBackend(models['a'],enabled=True,backends=models)
    features=FeatureConfig();features.asr.multi_asr_enabled=True
    text,info=infer_module()._transcribe(backend,SimpleNamespace(id='s'),features=features)
    assert not text and info['error'] and len(info['multi_asr_results'])==2
    assert all(item['status']=='failed' for item in info['multi_asr_results'])


def test_itn_dedup_refills_and_prompt_has_final_provenance(monkeypatch,tmp_path):
    from lifemate.asr.transcription import infer_module
    from lifemate.agent.prompt_builder import _render_request
    from lifemate.features.config import FeatureConfig
    import lifemate.data.loader as loader
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    monkeypatch.setattr(loader,'read_sample_audio',lambda s:audio)
    models={'a':Fake('ngày hai mươi tháng mười một'),'b':Fake('20/11'),'c':Fake('chọn lịch'),'d':Fake('đặt xe')}
    backend=MultiAsrBackend(models['a'],enabled=True,backends=models,max_hypotheses=3)
    features=FeatureConfig();features.asr.multi_asr_enabled=True;features.prompt.itn_enabled=True
    text,info=infer_module()._transcribe(backend,SimpleNamespace(id='s'),features=features)
    assert [h.text for h in info['hypotheses']]==['20/11','chọn lịch','đặt xe']
    assert info['hypotheses'][0].provenance==('a','b') and len(info['raw_hypotheses'])==4
    assert len(info['multi_asr_results'])==4 and info['multi_asr_results'][0]['raw_transcript']=='ngày hai mươi tháng mười một'
    prompt=_render_request(text,info['hypotheses'],nbest_format=True,limit=3)
    assert 'a, b: 20/11' in prompt and 'd: đặt xe' in prompt and prompt.count('20/11')==1


def test_selected_model_and_elapsed_when_primary_fails(monkeypatch,tmp_path):
    from lifemate.asr.transcription import infer_module
    from lifemate.features.config import FeatureConfig
    import lifemate.data.loader as loader
    audio=tmp_path/'a.wav';audio.write_bytes(b'a');monkeypatch.setattr(loader,'read_sample_audio',lambda s:audio)
    class Slow(Fake):
        def transcribe(self,*a,**kw): time.sleep(.01); return super().transcribe(*a,**kw)
    first=Fake('');first.model_id='failed-model';second=Slow('ok');second.model_id='working-model'
    backend=MultiAsrBackend(first,enabled=True,backends={'primary':first,'secondary':second})
    features=FeatureConfig();features.asr.multi_asr_enabled=True
    text,info=infer_module()._transcribe(backend,SimpleNamespace(id='s'),features=features)
    assert text=='ok' and info['model_id']=='working-model' and info['latency_s']>=.01
    assert info['multi_asr_results'][1]['latency_ms']>=10 and info['multi_asr_results'][1]['cache_hit'] is False


@pytest.mark.parametrize('limit',[0,4,10,True])
def test_multi_candidate_limit_rejects_invalid(limit):
    with pytest.raises(AsrError): MultiAsrBackend(Fake(),enabled=True,backends={'a':Fake()},max_hypotheses=limit)


def test_config_validation_rejects_more_than_three():
    from lifemate.features.config import FeatureConfig
    from lifemate.features.registry import FeatureRegistry
    features=FeatureConfig();features.asr.multi_asr_enabled=True
    features.asr.multi_asr_models=[{'name':'gipformer_68m'}]
    features.asr.multi_asr_max_hypotheses=4
    assert any('between 1 and 3' in p for p in FeatureRegistry().validate(features))


def test_cache_hit_visible_in_per_model_result(tmp_path):
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    a=Fake('ok');a.model_id='m';wrapped=CachedTop1(a,tmp_path/'cache')
    backend=MultiAsrBackend(wrapped,enabled=True,backends={'a':wrapped})
    assert backend.transcribe_multi(audio,sample_id='s').raw_results[0]['cache_hit'] is False
    assert backend.transcribe_multi(audio,sample_id='s').raw_results[0]['cache_hit'] is True


def test_legacy_two_backend_selected_model_is_secondary(monkeypatch,tmp_path):
    from lifemate.asr.transcription import infer_module
    from lifemate.features.config import FeatureConfig
    import lifemate.data.loader as loader
    audio=tmp_path/'a.wav';audio.write_bytes(b'a');monkeypatch.setattr(loader,'read_sample_audio',lambda s:audio)
    primary=Fake('');primary.model_id='primary-id'
    secondary=Fake('ok');secondary.model_id='secondary-id'
    backend=MultiAsrBackend(primary,secondary,enabled=True,primary_topk=1)
    features=FeatureConfig();features.asr.multi_asr_enabled=True
    text,info=infer_module()._transcribe(backend,SimpleNamespace(id='s'),features=features)
    assert text=='ok' and info['model_id']=='secondary-id'


def test_invalid_topk_rejected_before_loading_models(monkeypatch):
    calls=[]
    monkeypatch.setattr('lifemate.asr.top1_models.create_asr_backend',lambda *a,**kw: calls.append(kw))
    with pytest.raises(AsrError): build_top1_multi(None,[{'name':'gipformer_68m'}],max_hypotheses=4)
    assert calls==[]


@pytest.mark.parametrize('value',[True,False,1.5,float('nan'),float('inf'),0,-1,4])
def test_candidate_count_rejects_lossy_values_before_loading(value,monkeypatch):
    from lifemate.features.config import FeatureConfig, apply_mapping
    from lifemate.errors import FeatureConfigurationError
    calls=[]
    monkeypatch.setattr('lifemate.asr.top1_models.create_asr_backend',lambda *a,**kw:calls.append(kw))
    with pytest.raises(AsrError): build_top1_multi(None,[{'name':'gipformer_68m'}],max_hypotheses=value)
    if value != 4:
        with pytest.raises(FeatureConfigurationError): apply_mapping(FeatureConfig(),{'asr':{'multi_asr_max_hypotheses':value}})
    assert not calls


def test_asr_none_rejected_when_top1_models_configured():
    from lifemate.asr.transcription import infer_module
    from lifemate.features.config import FeatureConfig
    features=FeatureConfig(); features.asr.multi_asr_enabled=True;features.asr.multi_asr_models=[{'name':'gipformer_68m'}]
    cfg=SimpleNamespace(asr=SimpleNamespace(backend='gipformer'),features=features,inference=SimpleNamespace(split='public_test'))
    with pytest.raises(AsrError,match='conflicts'):infer_module()._build_asr(SimpleNamespace(asr='none'),cfg,[])


def test_cache_leaf_symlink_to_prohibited_path_fails_without_opening_it(tmp_path):
    from lifemate.errors import ProhibitedAccessError
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    backend=Fake('ok');backend.model_id='m';cached=CachedTop1(backend,tmp_path/'cache')
    cached.transcribe(audio,sample_id='s')
    entry=next((tmp_path/'cache').glob('*.json'));entry.unlink()
    entry.symlink_to(tmp_path/'fake_environment'/'never-open.json')
    with pytest.raises(ProhibitedAccessError): cached.transcribe(audio,sample_id='s')


def test_rank_priority_and_cache_historical_latency(tmp_path):
    from lifemate.asr.top1_models import select_llm_candidates
    audio=tmp_path/'a.wav';audio.write_bytes(b'a')
    class Timed(Fake):
        def transcribe(self,*args,**kwargs):
            self.calls+=1;return TranscriptResult(text=self.text,raw_text=self.text,backend='fake',latency_s=1.2)
    a,b=Timed('one'),Timed('two');a.model_id='a';b.model_id='b'
    ca,cb=CachedTop1(a,tmp_path/'cache'),CachedTop1(b,tmp_path/'cache')
    m=MultiAsrBackend(ca,enabled=True,backends={'a':ca,'b':cb})
    m.transcribe_multi(audio,sample_id='s');result=m.transcribe_multi(audio,sample_id='s')
    selected=select_llm_candidates(result.all_hypotheses,itn_enabled=False)
    assert [h.rank for h in selected]==[0,1] and [h.model_rank for h in selected]==[1,1]
    assert [h.model_priority for h in selected]==[0,1]
    assert result.raw_results[0]['recognition_latency_ms']==0
    assert result.raw_results[0]['historical_recognition_latency_ms']==1200
    assert m.transcribe(audio,sample_id='s').model_name=='a'
