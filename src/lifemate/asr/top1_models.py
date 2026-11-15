"""Explicit four-model Top-1 registry; no models are downloaded here."""
from . import create_asr_backend
from .interface import AsrError, TranscriptResult
from .multi_backend import MultiAsrBackend
from lifemate.errors import ProhibitedAccessError
from dataclasses import replace
import math
from .nbest import normalize_hypothesis_text

MODELS = {
    'gipformer_68m': ('gipformer', 'g-group-ai-lab/gipformer-68M-rnnt'),
    'gipformer_15': ('gipformer', 'g-group-ai-lab/gipformer1.5-68M-rnnt'),
    'phowhisper_small': ('whisper', 'vinai/PhoWhisper-small'),
    'chunkformer_rnnt': ('chunkformer', 'khanhld/chunkformer-rnnt-large-vie'),
}


def select_llm_candidates(hypotheses, *, itn_enabled, limit=3):
    """Normalize first, then deduplicate and take a priority-ordered Top-K."""
    from lifemate.text.itn import apply_itn_pipeline
    if isinstance(limit, bool) or not 1 <= int(limit) <= 3:
        raise AsrError('LLM candidate limit must be between 1 and 3')
    merged = {}
    for item in hypotheses:
        normalized = apply_itn_pipeline(item.text, enabled=itn_enabled)
        key = normalize_hypothesis_text(normalized)
        if not key:
            continue
        provenance = tuple(item.provenance or (item.source,))
        if key in merged:
            old = merged[key]
            merged[key] = replace(old, provenance=tuple(dict.fromkeys((*old.provenance, *provenance))))
        else:
            merged[key] = replace(item, text=normalized, provenance=provenance)
    return list(merged.values())[:int(limit)]


class _Unavailable:
    name = 'unavailable'
    def __init__(self, model_id, error): self.model_id, self.error = model_id, error
    def transcribe(self, *args, **kwargs):
        return TranscriptResult(text='', backend=self.name, model_name=self.model_id, error=self.error)


def build_top1_multi(cfg, specs, *, max_hypotheses=3, failure_policy='primary_only', budget_seconds=30, cache_dir='outputs/cache/asr_top1'):
    """Construct only requested models in configuration order, never use a proxy model."""
    if not isinstance(specs, list) or not 1 <= len(specs) <= 4:
        raise AsrError('multi_asr_models requires one to four model entries')
    if isinstance(max_hypotheses, bool) or not 1 <= int(max_hypotheses) <= 3:
        raise AsrError('max_hypotheses must be between 1 and 3')
    if failure_policy not in ('primary_only', 'strict') or not math.isfinite(float(budget_seconds)) or float(budget_seconds) <= 0:
        raise AsrError('invalid multi-ASR failure policy or budget')
    names = []
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) - {'name','model_dir','device','revision'}:
            raise AsrError('each model entry accepts name, model_dir, device and revision only')
        name = spec.get('name')
        if name not in MODELS or name in names:
            raise AsrError(f'unknown or duplicate ASR model: {name!r}')
        names.append(name)
    models = {}
    for spec in specs:
        name = spec.get('name')
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
