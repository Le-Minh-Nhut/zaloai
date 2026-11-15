"""Explicit four-model Top-1 registry; no models are downloaded here."""
from . import create_asr_backend
from .interface import AsrError, TranscriptResult
from .multi_backend import MultiAsrBackend
from lifemate.errors import ProhibitedAccessError

MODELS = {
    'gipformer_68m': ('gipformer', 'g-group-ai-lab/gipformer-68M-rnnt'),
    'gipformer_15': ('gipformer', 'g-group-ai-lab/gipformer1.5-68M-rnnt'),
    'phowhisper_small': ('whisper', 'vinai/PhoWhisper-small'),
    'chunkformer_rnnt': ('chunkformer', 'khanhld/chunkformer-rnnt-large-vie'),
}


class _Unavailable:
    name = 'unavailable'
    def __init__(self, model_id, error): self.model_id, self.error = model_id, error
    def transcribe(self, *args, **kwargs):
        return TranscriptResult(text='', backend=self.name, model_name=self.model_id, error=self.error)


def build_top1_multi(cfg, specs, *, max_hypotheses=3, failure_policy='primary_only', budget_seconds=30, cache_dir='outputs/cache/asr_top1'):
    """Construct only requested models in configuration order, never use a proxy model."""
    if not isinstance(specs, list) or not 1 <= len(specs) <= 4:
        raise AsrError('multi_asr_models requires one to four model entries')
    models = {}
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) - {'name','model_dir','device','revision'}:
            raise AsrError('each model entry accepts name, model_dir, device and revision only')
        name = spec.get('name')
        if name not in MODELS or name in models: raise AsrError(f'unknown or duplicate ASR model: {name!r}')
        backend, model_id = MODELS[name]
        directory, device, revision = spec.get('model_dir'), spec.get('device','cpu'), spec.get('revision')
        try:
            if backend == 'gipformer':
                kwargs = {'model_id':model_id,'model_dir':directory,'device':device,'offline':True}
                if revision: kwargs['revision'] = revision
            elif backend == 'whisper':
                if revision: raise AsrError('PhoWhisper revision requires a locally resolved immutable model_dir')
                kwargs = {'model_name':directory or model_id,'device':device,'offline':True}
            else:
                if revision: raise AsrError('ChunkFormer local revision provenance is not verified; use a verified artifact manifest')
                kwargs = {'model_dir':directory,'device':device}
            model = create_asr_backend(cfg, backend=backend, **kwargs)
            if revision and getattr(model, 'revision', None) != revision:
                raise AsrError(f'{name}: loaded revision differs from requested revision')
            model.model_id = model_id
            from .top1_cache import CachedTop1
            model = CachedTop1(model, cache_dir)
        except ProhibitedAccessError: raise
        except Exception as exc:
            if failure_policy == 'strict': raise AsrError(f'{name}: {exc}') from exc
            model = _Unavailable(model_id, f'{type(exc).__name__}: {exc}')
        models[name] = model
    primary = next(iter(models.values()))
    return MultiAsrBackend(primary, enabled=True, backends=models, max_hypotheses=max_hypotheses,
                           failure_policy=failure_policy, budget_seconds=budget_seconds)


def transcript_records_by_model(sample_id, results):
    """Produce separate per-model records accepted by data.asr_variants.load_transcripts."""
    records = {}
    for result in results:
        text = result.get('raw_transcript')
        if result.get('status') != 'executed' or not text:
            continue
        records[result['model']] = {'sample_id': str(sample_id), 'asr_model': result.get('model_id') or result['model'],
                                    'hypotheses': [{'text': text, 'rank': 0}],
                                    'revision': result.get('revision'), 'backend': result.get('backend')}
    return records
