"""Strict replay of previously decoded train utterances; never runs an ASR model."""
from lifemate.asr.top1_models import MODELS, select_llm_candidates
from lifemate.asr.nbest import AsrHypothesis
from lifemate.asr.conditioned_records import _hash
from lifemate.data.conditioned_sft import load_records
from lifemate.data.history import _audio_path, _turn_role, _turns_of, _writeable_turn
from lifemate.data.loader import read_sample_audio
from lifemate.errors import LifeMateError


class ArtifactReplay:
    def __init__(self, path, samples, *, split, data_root, k, itn=False):
        if split != 'train':
            raise LifeMateError('replay artifact is restricted to the train split')
        self.index, self.manifest = load_records(path)
        self.k, self.itn = k, itn
        identities = self.manifest['identity'].get('models', {})
        self.expected = set(identities)
        if self.expected not in ({'gipformer_68m'}, set(MODELS)) or any(identities[name].get('model_id') != MODELS[name][1] for name in self.expected):
            raise LifeMateError('replay model identity mismatch')
        self.rows = {}
        for sample in samples:
            sid = str(sample.id)
            turns = list(_turns_of(sample))
            requested = [('current', read_sample_audio(sample))]
            for i, turn in enumerate(turns):
                if _turn_role(turn) == 'user':
                    raw = _writeable_turn(turn) or {}
                    requested.append((f'history:{i}', _audio_path(raw, split, data_root) if raw.get('audio') else None))
            for turn_id, audio in requested:
                row = self.index.get((sid, turn_id))
                if not row or not audio or not audio.is_file() or _hash(audio) != row.get('audio_sha256'):
                    raise LifeMateError(f'replay audio identity mismatch: {sid}/{turn_id}')
                models = row.get('models') or []
                if row.get('status') != 'complete' or not row.get('selected_text') or len(models) != len(self.expected) or {m.get('model') for m in models} != self.expected:
                    raise LifeMateError(f'replay incomplete: {sid}/{turn_id}')
                if any(m.get('status') != 'executed' or not m.get('transcript') or m.get('model_id') != MODELS[m['model']][1] for m in models):
                    raise LifeMateError(f'replay invalid model result: {sid}/{turn_id}')
                if any('revision' in m and m['revision'] != identities[m['model']].get('revision') for m in models):
                    raise LifeMateError(f'replay model revision mismatch: {sid}/{turn_id}')
                self.rows[(sid, turn_id)] = row

    def current(self, sample):
        row = self.rows[(str(sample.id), 'current')]
        hypotheses = [AsrHypothesis(text=m['transcript'], source=m['model'], backend=m.get('backend'),
                                   provenance=(m['model'],), model_rank=1, model_priority=i)
                      for i, m in enumerate(row['models'])]
        selected = select_llm_candidates(hypotheses, itn_enabled=self.itn, limit=self.k, conditioned_sft=True)
        return selected[0].text, {'hypotheses': selected, 'multi_asr_results': row['models'],
                                  'replay': True, 'historical_asr_latency_s': row.get('latency_s'),
                                  'conditioning': row.get('conditioning') or {}, 'itn_applied': self.itn}

    def history(self, sample, deadline=None):
        for i, turn in enumerate(_turns_of(sample)):
            if _turn_role(turn) != 'user':
                continue
            row = self.rows[(str(sample.id), f'history:{i}')]
            raw = _writeable_turn(turn)
            if raw is None:
                raise LifeMateError('history turn cannot be updated')
            for key in ('audio_text', 'transcript', 'text', 'asr_text', 'utterance'):
                raw.pop(key, None)
            raw['content'] = row['selected_text']
            if not isinstance(turn, dict):
                turn.content = row['selected_text']
