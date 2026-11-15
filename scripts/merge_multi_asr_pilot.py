#!/usr/bin/env python3
"""Merge independently decoded Top-1 pilot rows with an existing verified artifact."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from lifemate.asr.conditioned_records import SCHEMA, _digest_record
from lifemate.asr.nbest import AsrHypothesis
from lifemate.asr.top1_models import MODELS, select_llm_candidates
from lifemate.data.conditioned_sft import load_records
from lifemate.evaluation.transcripts import _artifact_identity
from lifemate.errors import LifeMateError

REVISIONS = {'gipformer_15':'dd9227dcd8705c13f33bdbe59728d546ab94480f',
             'phowhisper_small':'a86b604c346caf7148c37512eafe783a16420adb',
             'chunkformer_rnnt':'9863592e00d6c02634c4ce2db8a4f3414f1ba7d6'}

def merge(base, other, model_dirs, output):
    prior, meta = load_records(base)
    source_sha=meta['sha256']; identities=meta['identity']['models']
    if set(identities)!={'gipformer_68m'} or identities['gipformer_68m']['model_id'] != MODELS['gipformer_68m'][1]:
        raise LifeMateError('input is not verified single-Gipformer artifact')
    identities=dict(identities); indices={}
    for name in REVISIONS:
        rows={}
        for line in Path(other[name]).read_text(encoding='utf-8').splitlines():
            row=json.loads(line); key=(row.get('sample_id'),row.get('turn_id'))
            if key in rows or row.get('model')!=name or row.get('model_id')!=MODELS[name][1] or row.get('status')!='executed' or not row.get('transcript'):
                raise LifeMateError(f'invalid or failed {name} row: {key}')
            if key not in prior or row.get('audio_sha256')!=prior[key].get('audio_sha256'):
                raise LifeMateError(f'audio identity mismatch: {name} {key}')
            rows[key]=row
        indices[name]=rows
        identities[name]={'model_id':MODELS[name][1], 'revision':REVISIONS[name], 'artifacts':_artifact_identity(model_dirs[name])}
    keys=list(indices['gipformer_15'])
    if not keys or any(set(indices[name])!=set(keys) for name in REVISIONS):
        raise LifeMateError('independent model sample sets differ')
    inputs=[{k:prior[key].get(k) for k in ('sample_id','turn_id','audio_sha256')} for key in keys]
    features=copy.deepcopy(meta['identity']['features'])
    features.setdefault('asr',{})['multi_asr_models'] = [{'name':name,'model_dir':str(model_dirs[name]),'device':'cpu'} for name in REVISIONS]
    features['asr']['multi_asr_models'].insert(0, {'name':'gipformer_68m','origin_artifact_sha256':source_sha})
    identity={'schema':SCHEMA,'split':'train','inputs':inputs,'models':identities,
              'features':features,'source_gipformer_sha256':source_sha,
              'decoders':{'gipformer_68m':'verified_source_artifact','gipformer_15':'backend_default_cpu_int8', 'phowhisper_small':'backend_greedy_cpu', 'chunkformer_rnnt':{'chunk_size':64,'left_context_size':128,'right_context_size':128,'total_batch_duration':1800}},
              'merge_policy':'strict_independent_top1_v1'}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    output=Path(output);manifest=output.with_name(output.name+'.manifest.json')
    if output.exists() or manifest.exists():raise LifeMateError('output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        for key in keys:
            original=prior[key]; originals=original.get('models') or []
            if original.get('status')!='complete' or len(originals)!=1 or originals[0].get('status')!='executed':
                raise LifeMateError(f'original Gipformer unavailable: {key}')
            models=[dict(originals[0])]
            for i,name in enumerate(REVISIONS,1):
                m=dict(indices[name][key]);m.update(revision=REVISIONS[name],rank=1,model_priority=i,backend=name,cache_hit=False)
                models.append(m)
            hypotheses=[AsrHypothesis(text=m['transcript'],source=m['model'],backend=m.get('backend'),provenance=(m['model'],),model_rank=1,model_priority=i) for i,m in enumerate(models)]
            selected=select_llm_candidates(hypotheses,itn_enabled=False,limit=4,conditioned_sft=True)
            row={'schema':SCHEMA,'run_identity':identity_hash,'split':'train',**{k:original[k] for k in ('sample_id','turn_id','audio_sha256')},'models_requested':list(MODELS),'models':models,'selected_text':selected[0].text,'status':'complete','error':None,'conditioning':{'multi_asr':{'executed':True,'degraded':False}},'latency_s':None,'historical_source_sha256':source_sha}
            row['record_sha256']=_digest_record(row)
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    with manifest.open('x',encoding='utf-8') as f:json.dump({'schema':SCHEMA,'run_identity':identity_hash,'sha256':checksum,'records':len(keys),'identity':identity},f,ensure_ascii=False,sort_keys=True)
    return checksum

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--input-dir',required=True);p.add_argument('--models-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    inp=Path(a.input_dir);md=Path(a.models_dir)
    print(merge(a.base,{n:inp/(n+'.jsonl') for n in REVISIONS},{'gipformer_15':md/'gipformer15','phowhisper_small':md/'phowhisper_small','chunkformer_rnnt':md/'chunkformer'},a.output))
if __name__=='__main__':main()
