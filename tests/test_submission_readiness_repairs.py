import json
import time
from types import SimpleNamespace

import pytest

from lifemate.asr.nbest import AsrHypothesis
from lifemate.asr.transcription import infer_module
from lifemate.data.history import transcribe_missing_history
from lifemate.errors import LifeMateError, ProhibitedAccessError
from lifemate.evaluation.evaluator import evaluate_closed_loop
from lifemate.evaluation.transcripts import transcript_cache_key
from lifemate.features.config import FeatureConfig


def test_debug_hypotheses_are_json_safe_and_redacted(tmp_path):
    infer = infer_module()
    infer._dump_debug(tmp_path, 's', {'asr': {'hypotheses': [AsrHypothesis(text='private', rank=0, source='a')], 'nested': {'message': 'secret'}}, 'transcript': 'allowed'})
    payload = json.loads((tmp_path / 's.json').read_text())
    assert payload['asr']['hypotheses'][0]['source'] == 'a'
    assert 'message' not in payload['asr']['nested']
    assert payload['transcript'] == 'allowed'


def test_checkpoint_types_fail_before_model_load(tmp_path):
    infer = infer_module()
    args = SimpleNamespace(generator=None)
    cfg = SimpleNamespace(inference=SimpleNamespace(model_mode='lora', checkpoint=str(tmp_path)))
    with pytest.raises(LifeMateError, match='adapter_config'):
        infer._load_generator(cfg, args, lambda: '')
    (tmp_path / 'adapter_config.json').write_text('{}')
    cfg.inference.model_mode = 'full'
    with pytest.raises(LifeMateError, match='LoRA adapter'):
        infer._load_generator(cfg, args, lambda: '')
    cfg.inference.checkpoint = str(tmp_path / 'missing')
    with pytest.raises(LifeMateError, match='does not exist'):
        infer._load_generator(cfg, args, lambda: '')


def test_cache_key_changes_when_local_artifact_changes(tmp_path):
    weights = tmp_path / 'encoder.onnx'
    weights.write_bytes(b'a')
    cfg = SimpleNamespace(asr=SimpleNamespace(backend='gipformer', model_dir=str(tmp_path)))
    feature = FeatureConfig()
    before = transcript_cache_key(cfg, feature, split='val', sample_ids=['s'])
    weights.write_bytes(b'different')
    assert before != transcript_cache_key(cfg, feature, split='val', sample_ids=['s'])


def test_history_failure_is_local_but_security_is_strict(tmp_path):
    audio = tmp_path / 'test' / 'audio' / 's' / 's_u1.wav'
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b'bad')
    sample = {'id': 's', 'history': [{'role': 'user', 'audio': 'audio/s/s_u1.wav'}]}
    def broken(_):
        raise OSError('corrupt')
    assert 'corrupt' in transcribe_missing_history(sample, split='test', data_root=tmp_path, transcribe=broken)[0]
    def forbidden(_):
        raise ProhibitedAccessError('prohibited')
    with pytest.raises(ProhibitedAccessError):
        transcribe_missing_history(sample, split='test', data_root=tmp_path, transcribe=forbidden)


def test_end_to_end_asr_is_in_budget_and_reported(tmp_path):
    seen = []
    class Agent:
        env = None
        def run(self, sample, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(status='time_budget' if time.monotonic() >= kwargs['deadline'] else 'finished', errors=[])
    def asr(sample):
        time.sleep(.03)
        return 'hi', {'hypotheses': [{'text': 'hi'}]}
    result = evaluate_closed_loop(lambda _: Agent(), [SimpleNamespace(id='s', label=None)], split='test', data_root=tmp_path, deadline_seconds=.01, transcribe_sample=asr)
    assert result['runtime_protocol'] == 'end_to_end_uncached'
    assert result['rows'][0]['timed_out']
    assert result['latency']['current_asr']['mean'] >= .03
    assert seen[0]['asr_hypotheses'] == [{'text': 'hi'}]


def test_ablation_detects_rms_parameter_change(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'feature_ablation.py'
    spec = importlib.util.spec_from_file_location('lifemate_ablation_repair', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arms = []
    for value in (-23, -18):
        arm = tmp_path / f'{value}.yaml'
        arm.write_text(f'features:\n  audio:\n    rms_normalization_enabled: true\n    rms_target_dbfs: {value}\n')
        arms.append(arm)
    with pytest.raises(ValueError, match='audio_to_action'):
        module.build_report(arms, split='val', data_root=None, limit=1, out=tmp_path / 'out.json', mock_env=True, dry_run=True, evaluation_mode='text_to_action')
    report = module.build_report(arms, split='val', data_root=None, limit=1, out=tmp_path / 'out.json', mock_env=True, dry_run=True)
    assert all(arm['evaluation_mode'] == 'audio_to_action' for arm in report['arms'])


def test_one_history_failure_does_not_abort_independent_samples(tmp_path, monkeypatch):
    import copy
    import lifemate.data.history as history
    from pathlib import Path
    from lifemate.data.loader import load_samples
    infer = infer_module()
    root = Path(__file__).resolve().parent / 'fixtures' / 'synthetic_lifemate'
    original = load_samples(root, 'train', limit=1, with_labels=False)[0]
    samples = []
    for i in range(10):
        sample = copy.deepcopy(original)
        sample.id = f'sample_{i}'
        samples.append(sample)
    monkeypatch.setattr(infer, 'load_samples', lambda *args, **kwargs: samples)
    monkeypatch.setattr(infer, '_build_asr', lambda *args, **kwargs: object())
    monkeypatch.setattr(infer, '_transcribe', lambda *args, **kwargs: ('hello', {'error': None}))
    monkeypatch.setattr(history, 'transcribe_missing_history', lambda sample, **kwargs: ['corrupt audio'] if sample.id == 'sample_3' else [])
    script = tmp_path / 'script.json'
    script.write_text(json.dumps({sample.id: ['<action>{"type":"finish"}</action>'] for sample in samples}))
    out = tmp_path / 'submission.jsonl'
    rc = infer.run_inference(['--split', 'train', '--data-root', str(root), '--use-mock-env', '--generator', f'dumpscript:{script}', '--asr', 'gipformer', '--output', str(out)])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row['id'] for row in rows] == [sample.id for sample in samples]
    assert rows[3]['tool_calls'] == [] and all(row['state_hash'] for row in rows)


def test_feature_status_distinguishes_observed_and_missing():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('evaluate_repair', Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = FeatureConfig()
    features.audio.vad_enabled = features.audio.rms_normalization_enabled = True
    features.asr.gipformer_v15_enabled = features.asr.multi_asr_enabled = True
    samples = [SimpleNamespace(id='a'), SimpleNamespace(id='b')]
    info = {'a': {'model_id': 'g-group-ai-lab/gipformer1.5-68M-rnnt', 'conditioning': {'vad': {}, 'rms': {}, 'multi_asr': {'executed': False, 'degraded': True}}}}
    counts = module._feature_execution_counts(samples, features, info)
    for name in ('audio.vad', 'audio.rms_normalization', 'asr.gipformer_v15'):
        assert counts[name]['n_executed'] == 1 and counts[name]['n_not_observed'] == 1
    assert counts['asr.multi_asr']['n_degraded'] == 1
    assert counts['asr.multi_asr']['n_not_observed'] == 1


def test_runtime_backend_identity_invalidates_cached_transcript(tmp_path):
    from lifemate.asr.interface import TranscriptResult
    from lifemate.asr.transcription import transcribe_samples_cached
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    artifact = tmp_path / 'weights.onnx'
    artifact.write_bytes(b'first')
    class Backend:
        name = 'real-asr'
        model_id = 'model'
        revision = 'same'
        _resolved_paths = {'encoder': artifact}
        calls = 0
        def transcribe(self, path, **kwargs):
            self.calls += 1
            return TranscriptResult(text='ok', backend=self.name)
    backend = Backend()
    cfg = SimpleNamespace(asr=SimpleNamespace(backend='gipformer', model_dir=None), features=FeatureConfig())
    sample = SimpleNamespace(id='s', audio_path=audio)
    cache = tmp_path / 'cache.json'
    transcribe_samples_cached([sample], cfg, asr=backend, cache_path=cache)
    transcribe_samples_cached([sample], cfg, asr=backend, cache_path=cache)
    assert backend.calls == 1
    artifact.write_bytes(b'second version')
    transcribe_samples_cached([sample], cfg, asr=backend, cache_path=cache)
    assert backend.calls == 2


def test_failed_history_evaluation_preserves_environment_state(tmp_path):
    class Env:
        def get_tool_calls(self): return []
        def get_state_hash(self): return 'official-hash'
    class Agent:
        env = Env()
        def run(self, *args, **kwargs): raise AssertionError('must not act')
    def history(sample, deadline): raise OSError('bad audio')
    result = evaluate_closed_loop(lambda _: Agent(), [SimpleNamespace(id='s', label=None)], split='test', data_root=tmp_path, prepare_history=history)
    assert result['rows'][0]['status'] == 'history_incomplete'
    assert result['rows'][0]['state_hash'] == 'official-hash'
    assert result['n_failures'] == 1


def test_current_asr_failure_prevents_agent_action(tmp_path, monkeypatch):
    import copy
    from pathlib import Path
    from lifemate.data.loader import load_samples
    infer = infer_module()
    root = Path(__file__).resolve().parent / 'fixtures' / 'synthetic_lifemate'
    original = load_samples(root, 'train', limit=1, with_labels=False)[0]
    samples = [copy.deepcopy(original), copy.deepcopy(original)]
    samples[0].id, samples[1].id = 'bad', 'good'
    monkeypatch.setattr(infer, 'load_samples', lambda *args, **kwargs: samples)
    monkeypatch.setattr(infer, '_build_asr', lambda *args, **kwargs: object())
    monkeypatch.setattr(infer, '_transcribe', lambda backend, sample, **kwargs: ('', {'error': 'bad audio'}) if sample.id == 'bad' else ('ok', {'error': None}))
    from lifemate.agent.agent import ToolAgent
    called = []
    original_run = ToolAgent.run
    def track_run(self, sample, *args, **kwargs):
        called.append(sample.id)
        return original_run(self, sample, *args, **kwargs)
    monkeypatch.setattr(ToolAgent, 'run', track_run)
    script = tmp_path / 'script.json'
    script.write_text(json.dumps({'bad': ['<action>{"type":"tool_call","name":"unknown","arguments":{}}</action>'], 'good': ['<action>{"type":"finish"}</action>']}))
    out = tmp_path / 'out.jsonl'
    assert infer.run_inference(['--split', 'train', '--data-root', str(root), '--use-mock-env', '--generator', f'dumpscript:{script}', '--asr', 'gipformer', '--output', str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]['tool_calls'] == [] and rows[0]['state_hash']
    assert called == ['good']
    assert len(rows) == 2
