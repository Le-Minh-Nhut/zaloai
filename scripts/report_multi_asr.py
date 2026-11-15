#!/usr/bin/env python3
"""Report per-model ASR diagnostics from a verified train-only artifact."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.asr.metrics import evaluate_asr
from lifemate.asr.nbest import normalize_hypothesis_text
from lifemate.compliance import assert_path_allowed
from lifemate.data.conditioned_sft import load_records
from lifemate.data.loader import load_samples


def summarize(index, samples):
    references={str(s.id):str(getattr(s,'audio_text','') or '') for s in samples}
    pairs={}; failures={}; hits={}; categories={};latencies={}
    for (sid,turn),row in index.items():
        if turn!='current':continue
        valid=[]
        for model in row.get('models') or []:
            name=model.get('model','unknown')
            if model.get('status')!='executed' or not model.get('transcript'):
                failures[name]=failures.get(name,0)+1;continue
            valid.append(normalize_hypothesis_text(model['transcript']))
            if model.get('cache_hit'):hits[name]=hits.get(name,0)+1
            if isinstance(model.get('latency_ms'),(int,float)):latencies.setdefault(name,[]).append(model['latency_ms']/1000)
            if references.get(sid):pairs.setdefault(name,[]).append((sid,references[sid],model['transcript']))
        category='no_transcript' if not valid else ('all_agree' if len(set(valid))==1 else 'disagree')
        categories[category]=categories.get(category,0)+1
    names=sorted(set(pairs)|set(failures))
    return {'models':{name:{'asr':evaluate_asr(pairs.get(name),name,latencies=latencies.get(name)),
                            'failed':failures.get(name,0),'cache_hits':hits.get(name,0)} for name in names},
            'disagreement':categories,'official_score':'UNAVAILABLE','note':'ASR diagnostics on train references, not task or leaderboard score'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--data-root',default='data');p.add_argument('--output',required=True);args=p.parse_args()
    target=assert_path_allowed(args.output);assert_path_allowed(target.resolve())
    if target.exists():raise ValueError('output already exists')
    index,meta=load_records(args.records)
    report=summarize(index,load_samples(args.data_root,'train'))
    report['artifact_sha256']=meta['sha256'];target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(f'wrote {target}')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as exc:print(f'error: {exc}',file=sys.stderr);sys.exit(2)
