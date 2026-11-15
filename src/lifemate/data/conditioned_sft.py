"""Conservative, causal conditioned-SFT augmentation from versioned ASR records."""
from __future__ import annotations
import hashlib
import json
import re
import copy
from pathlib import Path
from lifemate.agent.protocol import parse_action
from lifemate.agent.prompt_builder import _render_request, _render_history
from lifemate.asr.conditioned_records import SCHEMA, _digest_record, _hash
from lifemate.asr.top1_models import MODELS
from lifemate.asr.nbest import AsrHypothesis
from lifemate.asr.top1_models import select_llm_candidates
from lifemate.asr.interface import candidate_count
from lifemate.compliance import assert_path_allowed
from lifemate.data.augmentation import prompt_sections, replace_first_section, replace_section_body
from lifemate.data.asr_variants import _selected, _with_prompt
from lifemate.data.history import _turn_role, _turns_of, _writeable_turn, _audio_path
from lifemate.data.loader import read_sample_audio
from lifemate.data.preprocessing import estimate_tokens
from lifemate.errors import LifeMateError


def load_records(path):
    path=assert_path_allowed(path);assert_path_allowed(path.resolve())
    manifest=path.with_name(path.name+'.manifest.json');assert_path_allowed(manifest.resolve())
    metadata=json.loads(manifest.read_text(encoding='utf-8'))
    raw=path.read_bytes()
    if metadata.get('schema')!=SCHEMA or hashlib.sha256(raw).hexdigest()!=metadata.get('sha256'):
        raise LifeMateError('multi-ASR artifact manifest or checksum mismatch')
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows)!=metadata.get('records'):raise LifeMateError('multi-ASR record count mismatch')
    index={}
    identity=metadata.get('identity') or {}
    computed=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    if computed != metadata.get('run_identity'):raise LifeMateError('multi-ASR run identity mismatch')
    inputs=identity.get('inputs')
    if not isinstance(inputs,list) or len(inputs)!=len(rows):raise LifeMateError('multi-ASR input identity count mismatch')
    requested=set((identity.get('models') or {}).keys())
    for i,row in enumerate(rows):
        key=(row.get('sample_id'),row.get('turn_id'))
        if row.get('schema')!=SCHEMA or row.get('split')!='train' or row.get('run_identity')!=metadata.get('run_identity') or row.get('record_sha256')!=_digest_record(row) or key in index:
            raise LifeMateError('invalid, duplicate or non-train multi-ASR record')
        if {k:row.get(k) for k in ('sample_id','turn_id','audio_sha256')} != inputs[i] or set(row.get('models_requested') or []) != requested:
            raise LifeMateError('multi-ASR row input or model identity mismatch')
        index[key]=row
    return index,metadata


def coverage_audit(target, visible_prompt):
    """Heuristic only: detect explicit time conflicts; never claim semantic proof."""
    action=parse_action(target)
    if action.type=='invalid':return {'status':'UNSUPPORTED','reasons':['invalid action target']}
    sections=prompt_sections(visible_prompt)
    if sections is None:return {'status':'UNKNOWN','reasons':['unrecognized prompt structure']}
    visible=' '.join(v for k,v in sections.items() if k in {'# Yêu cầu','# Các giả thuyết ASR','# Ngữ cảnh','# Lịch sử hội thoại','# Các bước đã thực hiện'})
    arguments=json.dumps(action.arguments,ensure_ascii=False)
    gold_times=re.findall(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)',arguments)
    observed=re.findall(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)',visible)
    norm=lambda t: f'{int(t.split(":")[0]):02d}:{t.split(":")[1]}'
    gold_times=[norm(t) for t in gold_times];observed=[norm(t) for t in observed]
    request=sections.get('# Các giả thuyết ASR') or sections.get('# Yêu cầu') or ''
    lines=re.findall(r'^\d+\. .+$',request,flags=re.MULTILINE) or [request]
    candidate_times=[re.findall(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)',line) for line in lines]
    if gold_times and observed and all(len(ts)==1 for ts in candidate_times) and any(t not in observed for t in gold_times):
        return {'status':'UNSUPPORTED','reasons':['explicit target time absent while other visible times occur'], 'gold_times':gold_times,'observed_times':observed}
    def leaves(value):
        if isinstance(value, dict): return [item for child in value.values() for item in leaves(child)]
        if isinstance(value, list): return [item for child in value for item in leaves(child)]
        return [value] if isinstance(value, (str, int, float)) and not isinstance(value, bool) else []
    values=leaves(action.arguments)
    if action.is_tool_call and action.name and action.name.casefold() in request.casefold() and all(str(v).casefold() in visible.casefold() for v in values):
        return {'status':'SUPPORTED','reasons':['exact tool name in request and literal arguments visible; heuristic, not semantic proof']}
    return {'status':'UNKNOWN','reasons':['heuristic cannot establish intent and argument support']}


def augment_conditioned(records, samples, index, *, k=3, ratio=1.0, seed=42, unknown_policy='exclude', itn=False, split='train', data_root=None, history_limit=10, mode='multi', single_ratio=1.0, multi_ratio=1.0):
    """Append at most one variant per decision point, preserving the exact target."""
    k=candidate_count(k,maximum=4)
    if not 0 < ratio <= 1 or not 0 <= single_ratio <= 1 or not 0 <= multi_ratio <= 1 or unknown_policy not in {'exclude','include'} or mode not in {'single','multi','mixed'}:raise LifeMateError('invalid conditioned augmentation policy')
    samples={str(s.id):s for s in samples}
    counts={'SUPPORTED':0,'UNSUPPORTED':0,'UNKNOWN':0,'history_excluded':0,'missing_or_degraded':0,'created':0}
    counts['decisions'] = []
    output=[]
    audio_hashes={}
    for record in records:
        output.append(record)
        sid=str(record['sample_id'])
        if not _selected(seed,sid,ratio):continue
        sample=samples.get(sid); row=index.get((sid,'current'))
        if sample is None or row is None or row.get('status')!='complete':counts['missing_or_degraded']+=1;continue
        if sid not in audio_hashes:
            audio=read_sample_audio(sample)
            audio_hashes[sid]=_hash(audio) if audio and audio.is_file() else None
        if not audio_hashes[sid] or audio_hashes[sid]!=row.get('audio_sha256'):
            raise LifeMateError(f'audio identity mismatch for {sid}')
        # Replace every visible user turn with its own verified ASR text.  Never
        # carry a gold history transcript into a noisy variant.
        turns=list(_turns_of(sample)); history=copy.deepcopy(turns)
        visible_start=max(0,len(turns)-history_limit) if history_limit and history_limit>0 else len(turns)
        history_missing=False
        for i,turn in enumerate(turns):
            if i<visible_start or _turn_role(turn)!='user': continue
            prior=index.get((sid,f'history:{i}'))
            raw=_writeable_turn(turn) or {}
            audio=_audio_path(raw,split,data_root) if raw.get('audio') and data_root is not None else None
            if not prior or prior.get('status')!='complete' or not prior.get('selected_text') or not audio or not audio.is_file() or _hash(audio)!=prior.get('audio_sha256'):
                history_missing=True;break
            target=history[i]; target_raw=_writeable_turn(target)
            if isinstance(target,dict):
                for key in ('audio_text','transcript','text','asr_text','utterance'):target.pop(key,None)
                target['content']=prior['selected_text']
            else:
                target.content=prior['selected_text']
                if target_raw is not None:
                    for key in ('audio_text','transcript','text','asr_text','utterance'):target_raw.pop(key,None)
                    target_raw['content']=prior['selected_text']
        if history_missing:
            counts['history_excluded']+=1;continue
        models=row.get('models') or []
        if len(models)!=4 or {m.get('model') for m in models} != set(MODELS) or any(m.get('status')!='executed' or not m.get('transcript') or m.get('model_id') != MODELS.get(m.get('model'),(None,None))[1] for m in models):
            counts['missing_or_degraded']+=1;continue
        hypotheses=[AsrHypothesis(text=m['transcript'],source=m['model'],backend=m.get('backend'),provenance=(m['model'],),model_rank=1,model_priority=i) for i,m in enumerate(models)]
        selected=select_llm_candidates(hypotheses,itn_enabled=itn,limit=k,conditioned_sft=True)
        if not selected:counts['missing_or_degraded']+=1;continue
        variants = ('single','multi') if mode == 'mixed' else (mode,)
        seen_prompts = set()
        for variant_kind in variants:
            variant_ratio = single_ratio if variant_kind == 'single' else multi_ratio
            if not _selected(f'{seed}:{variant_kind}',sid,variant_ratio):continue
            shown = selected[:1] if variant_kind == 'single' else selected
            section=_render_request(shown[0].text,shown,nbest_format=True,limit=k,conditioned_multi_sft=True)
            prompt=replace_first_section(record['messages'][-1]['content'],section)
            if prompt is None:raise LifeMateError('cannot locate request section in causal SFT prompt')
            if history:
                prompt=replace_section_body(prompt,'# Lịch sử hội thoại',_render_history(history,history_limit))
                if prompt is None:raise LifeMateError('cannot locate history section in causal SFT prompt')
            if prompt in seen_prompts:
                counts['duplicate_variants'] = counts.get('duplicate_variants', 0) + 1
                continue
            seen_prompts.add(prompt)
            audit=coverage_audit(record['target'],prompt);counts[audit['status']]+=1
            counts['decisions'].append({'sample_id':sid,'step_index':record.get('step_index'),'kind':variant_kind,'audit':audit})
            if audit['status']=='UNSUPPORTED' or (audit['status']=='UNKNOWN' and unknown_policy=='exclude'):continue
            variant=_with_prompt(record,prompt,{'augmentation':'conditioned_multi_asr' if variant_kind=='multi' else 'conditioned_single_asr','coverage_audit':audit,'candidate_k':k if variant_kind=='multi' else 1,'asr_run_identity':row['run_identity'],'token_estimate':estimate_tokens(prompt)})
            output.append(variant);counts['created']+=1
    return output,counts
