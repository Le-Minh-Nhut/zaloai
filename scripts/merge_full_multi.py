#!/usr/bin/env python3
"""Merge complete durable per-model journals in source order, retaining failures."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lifemate.asr.conditioned_records import SCHEMA,_digest_record,_hash
from lifemate.asr.nbest import AsrHypothesis
from lifemate.asr.top1_models import MODELS,select_llm_candidates
from lifemate.data.conditioned_sft import load_records
from lifemate.errors import LifeMateError
from transcribe_full_multi import REV, DEC, canonical, digest, read_journal, validate_sources, atomic

def merge(root, work):
    models=list(REV);base,bmeta,pilot,pmeta,paths,_=validate_sources(root,models[0]);journals={};identities={};journal_sha={}
    for name in models:
        _,_,_,_,other_paths,_=validate_sources(root,name)
        if list(paths)!=list(other_paths):raise LifeMateError('source ordering mismatch')
        directory=work/name;identity=json.loads((directory/'identity.json').read_text());runid=digest(identity)
        if identity['source_artifact_sha256']!=bmeta['sha256'] or identity['pilot_sha256']!=pmeta['sha256'] or identity['model_id']!=MODELS[name][1] or identity['revision']!=REV[name] or identity['decoder']!=DEC[name]:raise LifeMateError(f'incompatible identity: {name}')
        journal=directory/'transcripts.jsonl';complete=json.loads((directory/'completion.json').read_text())
        if complete['run_identity']!=runid or complete['journal_sha256']!=_hash(journal):raise LifeMateError(f'incomplete or modified journal: {name}')
        rows=read_journal(journal,runid)
        if set(rows)!=set(paths) or complete['processed_count']!=len(paths):raise LifeMateError(f'missing utterances: {name}')
        for key,row in rows.items():
            if row['audio_sha256']!=base[key]['audio_sha256'] or row['model_id']!=MODELS[name][1] or row['revision']!=REV[name] or row['decoder']!=DEC[name]:raise LifeMateError(f'row identity mismatch: {name} {key}')
            if row['status']=='executed' and (not row.get('transcript') or row.get('error')):raise LifeMateError(f'invalid success: {name} {key}')
            if row['status']!='executed' and not row.get('error'):raise LifeMateError(f'failure missing error: {name} {key}')
        journals[name]=rows;identities[name]=pmeta['identity']['models'][name];journal_sha[name]=complete['journal_sha256']
    identities={'gipformer_68m':bmeta['identity']['models']['gipformer_68m'],**identities}
    identity={'schema':SCHEMA,'split':'train','inputs':bmeta['identity']['inputs'],'models':identities,'features':pmeta['identity']['features'],'source_gipformer_sha256':bmeta['sha256'],'pilot_sha256':pmeta['sha256'],'journal_sha256':journal_sha,'decoders':pmeta['identity']['decoders'],'merge_policy':'full_independent_top1_with_failures_v1'}
    runid=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    output=work/'four_asr_train.jsonl';manifest=work/'four_asr_train.jsonl.manifest.json';partial=work/'four_asr_train.jsonl.partial';tmp=work/'four_asr_train.jsonl.manifest.json.tmp'
    if output.exists():
        _,existing=load_records(output)
        if existing.get('run_identity')!=runid:raise LifeMateError('existing final artifact differs')
        print('already complete',existing['sha256']);return
    if manifest.exists():raise LifeMateError('manifest exists without final artifact')
    keys=list(paths);prior=[]
    if partial.exists():
        for line in partial.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('record_sha256')!=_digest_record(row) or row.get('run_identity')!=runid:raise LifeMateError('partial checksum mismatch')
            prior.append(row)
        if len(prior)>len(keys) or any((row['sample_id'],row['turn_id'])!=keys[i] for i,row in enumerate(prior)):raise LifeMateError('partial ordering mismatch')
    coverage=collections.Counter()
    for row in prior:coverage[sum(m['status']=='executed' for m in row['models'])]+=1
    with partial.open('a',encoding='utf-8') as f:
        for sid,turn in keys[len(prior):]:
            key=(sid,turn);original=base[key];models68=original.get('models') or []
            if original.get('status')!='complete' or len(models68)!=1 or models68[0].get('status')!='executed':raise LifeMateError(f'invalid baseline ASR {key}')
            results=[dict(models68[0])]+[dict(journals[n][key]) for n in models]
            good=[m for m in results if m['status']=='executed'];coverage[len(good)]+=1
            hyps=[AsrHypothesis(text=m['transcript'],source=m['model'],provenance=(m['model'],),model_rank=1,model_priority=i) for i,m in enumerate(results) if m['status']=='executed']
            selected=select_llm_candidates(hyps,itn_enabled=False,limit=4,conditioned_sft=True)
            row={'schema':SCHEMA,'run_identity':runid,'split':'train','sample_id':sid,'turn_id':turn,'audio_sha256':original['audio_sha256'],'models_requested':list(MODELS),'models':results,'selected_text':selected[0].text if selected else '', 'status':'complete' if len(good)==4 else 'degraded' if good else 'failed','error':None if len(good)==4 else 'one or more ASR models failed','conditioning':{'multi_asr':{'executed':len(good)>1,'degraded':len(good)!=4}},'latency_s':None,'historical_source_sha256':bmeta['sha256']}
            row['record_sha256']=_digest_record(row);f.write(canonical(row)+'\n')
        f.flush();os.fsync(f.fileno())
    checksum=_hash(partial);m={'schema':SCHEMA,'run_identity':runid,'sha256':checksum,'records':len(keys),'identity':identity,'success_coverage':dict(sorted(coverage.items()))}
    if tmp.exists():
        if json.loads(tmp.read_text())!=json.loads(canonical(m)):raise LifeMateError('staged manifest mismatch')
    else:atomic(tmp,m)
    os.link(partial,output);os.link(tmp,manifest);partial.unlink();tmp.unlink()
    loaded,verified=load_records(output)
    if len(loaded)!=len(keys) or verified['sha256']!=checksum:raise LifeMateError('final verification failed')
    model_counts={'gipformer_68m':{'success':len(keys),'failed':0,'reused':len(keys),'new':0}}
    for name,items in journals.items():
        values=list(items.values())
        model_counts[name]={'success':sum(x['status']=='executed' for x in values),'failed':sum(x['status']!='executed' for x in values),'reused':sum(bool(x.get('cache_hit')) for x in values),'new':sum(not x.get('cache_hit') for x in values)}
    replay={'status':'NOT VERIFIED'}
    try:
        from lifemate.data.loader import load_samples
        from lifemate.evaluation.heldout import select_heldout
        from lifemate.evaluation.artifact_replay import ArtifactReplay
        samples=load_samples(root/'data','train',with_labels=True)
        samples,selection=select_heldout(samples,root/'outputs/datasets/minicpm_baseline/train',split='train',limit=12)
        counts={}
        for k in (1,2,3,4):
            reader=ArtifactReplay(output,samples,split='train',data_root=root/'data',k=k)
            counts[str(k)]=[len(reader.current(sample)[1]['hypotheses']) for sample in samples]
        replay={'status':'PASS','samples':len(samples),'ids_sha256':selection['ids_sha256'],'candidate_counts':counts,'asr_decode_executed':False}
    except Exception as exc:
        replay={'status':'BLOCKED','reason':f'{type(exc).__name__}: {exc}'[:500]}
    atomic(work/'execution_report.json',{'expected_utterances':len(keys),'current_utterances':sum(turn=='current' for _,turn in keys),'history_utterances':sum(turn!='current' for _,turn in keys),'model_results':len(keys)*4,'models':model_counts,'four_model_coverage':{str(i):coverage[i] for i in range(5)},'source_gipformer_sha256':bmeta['sha256'],'pilot_sha256':pmeta['sha256'],'output_sha256':checksum,'journal_sha256':journal_sha,'replay':replay})
    print('COMPLETE',checksum,'coverage',dict(coverage),'replay',replay['status'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();merge(a.root.resolve(),a.output_dir.resolve())
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,LifeMateError) as exc:print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(2)
