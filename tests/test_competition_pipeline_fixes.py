from types import SimpleNamespace

from lifemate.asr.interface import TranscriptResult
from lifemate.asr.multi_backend import MultiAsrBackend
from lifemate.asr.nbest import AsrHypothesis, NBestResult
from lifemate.asr.transcription import transcribe_samples_cached
from lifemate.data.history import transcribe_missing_history
from lifemate.features.config import FeatureConfig


class Backend:
    nbest_supported = True
    def __init__(self, name, texts):
        self.name, self.texts, self.calls = name, texts, 0
    def transcribe(self, path, **kwargs):
        self.calls += 1
        return TranscriptResult(text=self.texts[0], backend=self.name)
    def transcribe_nbest(self, path, **kwargs):
        self.calls += 1
        return NBestResult(hypotheses=[AsrHypothesis(text=t, rank=i, source=self.name, backend=self.name) for i, t in enumerate(self.texts)])


def test_multi_nbest_cache_round_trip_and_audio_content(tmp_path):
    audio = tmp_path / 'a.wav'
    audio.write_bytes(b'audio1')
    sample = SimpleNamespace(id='s', audio_path=audio)
    features = FeatureConfig()
    features.asr.nbest_enabled = features.asr.multi_asr_enabled = True
    primary, secondary = Backend('primary', ['một', 'hai']), Backend('secondary', ['ba'])
    backend = MultiAsrBackend(primary, secondary, enabled=True, primary_topk=3, max_hypotheses=3)
    cfg = SimpleNamespace(asr=SimpleNamespace(backend='gipformer'), features=features)
    cache = tmp_path / 'cache.json'
    first, stats = transcribe_samples_cached([sample], cfg, features=features, asr=backend, cache_path=cache)
    assert first['s'] == 'một'
    assert [h['text'] for h in stats['asr_info']['s']['hypotheses']] == ['một', 'hai', 'ba']
    calls = primary.calls
    _, again = transcribe_samples_cached([sample], cfg, features=features, asr=backend, cache_path=cache)
    assert again['asr_info'] == stats['asr_info'] and primary.calls == calls
    audio.write_bytes(b'audio2')
    transcribe_samples_cached([sample], cfg, features=features, asr=backend, cache_path=cache)
    assert primary.calls > calls


def test_history_audio_transcription_preserves_observations(tmp_path):
    audio = tmp_path / 'private_test' / 'audio' / 's' / 's_u1.wav'
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b'provided audio')
    sample = {'id': 's', 'history': [{'role': 'user', 'audio': 'audio/s/s_u1.wav'}, {'role': 'assistant', 'tool_results': [{'ok': True}]}]}
    errors = transcribe_missing_history(sample, split='private_test', data_root=tmp_path, transcribe=lambda item: ('chào', {}))
    assert errors == []
    assert sample['history'][0]['audio_text'] == 'chào'
    assert sample['history'][1]['tool_results'] == [{'ok': True}]


def test_evaluation_forwards_hypotheses(tmp_path):
    from lifemate.evaluation.evaluator import evaluate_closed_loop
    seen = {}
    class Agent:
        env = None
        def run(self, sample, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(status='finished', tool_calls=[], state_hash='', errors=[])
    hypotheses = [{'text': 'một', 'source': 'primary'}, {'text': 'hai', 'source': 'secondary'}]
    result = evaluate_closed_loop(lambda sample: Agent(), [SimpleNamespace(id='s', label=None)],
                                  split='test', data_root=tmp_path, transcripts={'s': 'một'},
                                  asr_info={'s': {'hypotheses': hypotheses}})
    assert result['n_failures'] == 0
    assert seen['transcript'] == 'một' and seen['asr_hypotheses'] == hypotheses


def test_gipformer_v15_flag_selects_id(monkeypatch):
    from lifemate.asr.transcription import infer_module
    infer = infer_module()
    captured = []
    monkeypatch.setattr(infer, 'create_asr_backend', lambda cfg, **kwargs: captured.append(kwargs) or object())
    features = FeatureConfig()
    cfg = SimpleNamespace(asr=SimpleNamespace(backend='gipformer'), inference=SimpleNamespace(split='test'), features=features)
    args = SimpleNamespace(asr=None, allow_oracle=False)
    infer._build_asr(args, cfg, [])
    assert 'model_id' not in captured[-1]
    features.asr.gipformer_v15_enabled = True
    infer._build_asr(args, cfg, [])
    assert captured[-1]['model_id'] == 'g-group-ai-lab/gipformer1.5-68M-rnnt'


def test_itn_reaches_actual_prompt_and_keeps_originals(tmp_path):
    from lifemate.agent.prompt_builder import build_decision_prompt
    from lifemate.asr.transcription import infer_module
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    sample = SimpleNamespace(id='s', audio_path=audio)
    features = FeatureConfig()
    features.asr.multi_asr_enabled = True
    features.prompt.itn_enabled = True
    backend = MultiAsrBackend(Backend('primary', ['bảy giờ ba mươi sáng']),
                              Backend('secondary', ['tám giờ tối']), enabled=True, primary_topk=1)
    text, info = infer_module()._transcribe(backend, sample, features=features)
    prompt = build_decision_prompt(transcript=text, context={}, history=[], installed_apps=[], catalog={},
                                   asr_hypotheses=info['hypotheses'], nbest_format=True)
    assert '07:30' in prompt and '20:00' in prompt
    assert info['raw_hypotheses'][0].text == 'bảy giờ ba mươi sáng'
