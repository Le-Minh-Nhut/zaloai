#!/usr/bin/env python3
"""Durable per-model CPU transcription, reusing a verified four-model pilot."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.asr.conditioned_records import _hash, _utterances
from lifemate.asr.top1_models import MODELS
from lifemate.data.conditioned_sft import load_records
from lifemate.data.loader import load_samples
from lifemate.evaluation.transcripts import _artifact_identity
from lifemate.text.itn import apply_itn_pipeline
from lifemate.errors import LifeMateError, ProhibitedAccessError

REV={'gipformer_15':'dd9227dcd8705c13f33bdbe59728d546ab94480f','phowhisper_small':'a86b604c346caf7148c37512eafe783a16420adb','chunkformer_rnnt':'9863592e00d6c02634c4ce2db8a4f3414f1ba7d6'}
DIR={'gipformer_15':'gipformer15','phowhisper_small':'phowhisper_small','chunkformer_rnnt':'chunkformer'}
DEC={'gipformer_15':'backend_default_cpu_int8','phowhisper_small':'backend_greedy_cpu','chunkformer_rnnt':{'chunk_size':64,'left_context_size':128,'right_context_size':128,'total_batch_duration':1800}}

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def atomic(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def validate_sources(root, model):
    base,bmeta=load_records(root/'outputs/research/e2_gipformer_20261115/train_top1.jsonl')
    pilot,pmeta=load_records(root/'outputs/research/e2_e3_pilot_20261115/four_asr_pilot.jsonl')
    spec=pmeta['identity']['models'][model];model_dir=root/'models/research_e3'/DIR[model]
    if pmeta['identity'].get('source_gipformer_sha256')!=bmeta['sha256'] or spec.get('model_id')!=MODELS[model][1] or spec.get('revision')!=REV[model] or pmeta['identity'].get('decoders',{}).get(model)!=DEC[model] or canonical(spec.get('artifacts'))!=canonical(_artifact_identity(model_dir)):
        raise LifeMateError('pilot model, revision, decoder, artifact or source identity differs')
    samples=load_samples(root/'data','train',with_labels=False)
    utterances=list(_utterances(samples,'train',root/'data'))
    if len(utterances)!=len(base) or len(base)!=5036:raise LifeMateError('source utterance count differs from verified artifact')
    paths={}
    for sid,turn,path in utterances:
        key=(sid,turn); row=base.get(key)
        if key in paths or row is None or not path or not path.is_file() or _hash(path)!=row.get('audio_sha256'):raise LifeMateError(f'base audio mismatch {key}')
        paths[key]=path
    for key,row in pilot.items():
        m=[x for x in row.get('models',[]) if x.get('model')==model]
        if key not in base or row.get('audio_sha256')!=base[key].get('audio_sha256') or len(m)!=1 or m[0].get('model_id')!=MODELS[model][1] or m[0].get('revision')!=REV[model] or m[0].get('status')!='executed' or not m[0].get('transcript'):
            raise LifeMateError(f'pilot record mismatch {key}')
    return base,bmeta,pilot,pmeta,paths,model_dir

def read_journal(path,identity):
    rows={}
    if not path.exists():return rows
    with path.open('rb+') as f:
        raw=f.read(); end=raw.rfind(b'\n')+1
        if end!=len(raw):f.seek(end);f.truncate();f.flush();os.fsync(f.fileno())
    for line in raw[:end].splitlines():
        row=json.loads(line);checksum=row.pop('record_sha256',None)
        if checksum!=digest(row) or row.get('run_identity')!=identity:raise LifeMateError('journal checksum or identity mismatch')
        key=(row['sample_id'],row['turn_id']); old=rows.get(key)
        if old and old['status']=='executed':raise LifeMateError('journal duplicates successful record')
        rows[key]=row
    return rows

def backend_for(name, directory):
    if name=='gipformer_15':
        from lifemate.asr.gipformer_backend import GipformerBackend
        return GipformerBackend(model_id=MODELS[name][1],model_dir=directory,device='cpu',offline=True)
    if name=='phowhisper_small':
        from lifemate.asr.whisper_backend import WhisperBackend
        return WhisperBackend(model_name=str(directory),device='cpu',offline=True)
    from lifemate.asr.chunkformer_backend import ChunkFormerBackend
    return ChunkFormerBackend(model_dir=directory,device='cpu')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=REV);p.add_argument('--root',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--retry-failures',action='store_true');p.add_argument('--source-commit',required=True);a=p.parse_args(); root=a.root.resolve();model=a.model
    out=a.output_dir/model;out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise LifeMateError(f'{model} worker already running')
        base,bmeta,pilot,pmeta,paths,directory=validate_sources(root,model)
        identity={'schema':'lifemate.full_model_transcripts.v1','model':model,'model_id':MODELS[model][1],'revision':REV[model],'decoder':DEC[model],'model_artifacts':pmeta['identity']['models'][model]['artifacts'],'source_artifact_sha256':bmeta['sha256'],'pilot_sha256':pmeta['sha256'],'source_commit':a.source_commit}
        runid=digest(identity);identity_path=out/'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text())!=json.loads(canonical(identity)):raise LifeMateError('run identity changed; use new directory')
        else:atomic(identity_path,identity)
        journal=out/'transcripts.jsonl';rows=read_journal(journal,runid);start=time.monotonic();backend=None
        with journal.open('a',encoding='utf-8') as f:
            for i,(key,path) in enumerate(paths.items(),1):
                prior=rows.get(key)
                if prior and (prior.get('audio_sha256')!=base[key]['audio_sha256'] or prior.get('model_id')!=MODELS[model][1] or prior.get('revision')!=REV[model]):raise LifeMateError(f'cached identity mismatch {key}')
                if prior and (prior['status']=='executed' or not a.retry_failures):continue
                sid,turn=key;item={'sample_id':sid,'turn_id':turn,'audio_sha256':base[key]['audio_sha256'],'model':model,'model_id':MODELS[model][1],'revision':REV[model],'decoder':DEC[model],'run_identity':runid}
                if key in pilot and not prior:
                    m=next(x for x in pilot[key]['models'] if x['model']==model)
                    item.update(status='executed',transcript=m['transcript'],raw_transcript=m.get('raw_transcript') or m['transcript'],latency_ms=m.get('latency_ms'),error=None,cache_hit=True,cache_source=pmeta['sha256'])
                else:
                    try:
                        if backend is None:
                            t=time.monotonic();backend=backend_for(model,directory);print('model_loaded',model,'seconds',round(time.monotonic()-t,3),flush=True)
                        t=time.monotonic();result=backend.transcribe(path,sample_id=f'{sid}:{turn}')
                        error=result.error or ('empty transcript' if not result.text else None)
                        item.update(status='failed' if error else 'executed',transcript=result.text or '',raw_transcript=result.raw_text or result.text or '',latency_ms=round((time.monotonic()-t)*1000,2),error=error,cache_hit=False)
                    except ProhibitedAccessError:raise
                    except Exception as exc:item.update(status='failed',transcript='',raw_transcript='',latency_ms=None,error=f'{type(exc).__name__}: {exc}'[:500],cache_hit=False)
                item['normalized_transcript']=apply_itn_pipeline(item['transcript'],enabled=False) if item['transcript'] else ''
                item['record_sha256']=digest(item);f.write(canonical(item)+'\n');f.flush();os.fsync(f.fileno());item.pop('record_sha256');rows[key]=item
                if i%25==0 or i==len(paths):
                    counts={'processed_count':len(rows),'successful_count':sum(x['status']=='executed' for x in rows.values()),'failed_count':sum(x['status']!='executed' for x in rows.values()),'remaining_count':len(paths)-len(rows),'reused_count':sum(bool(x.get('cache_hit')) for x in rows.values()),'elapsed_seconds_this_process':round(time.monotonic()-start,2),'last_completed_sample':sid,'last_turn':turn,'pid':os.getpid(),'model':model,'run_identity':runid,'output':str(journal)};atomic(out/'progress.json',counts);print(canonical(counts),flush=True)
        atomic(out/'completion.json',{'processed_count':len(rows),'successful_count':sum(x['status']=='executed' for x in rows.values()),'failed_count':sum(x['status']!='executed' for x in rows.values()),'run_identity':runid,'journal_sha256':_hash(journal)})
if __name__=='__main__':
    try:main()
    except (LifeMateError,OSError,ValueError) as exc:print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(2)
