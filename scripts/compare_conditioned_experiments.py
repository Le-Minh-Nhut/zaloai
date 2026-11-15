#!/usr/bin/env python3
"""Compare observed closed-loop diagnostics on the same held-out cases."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lifemate.compliance import assert_path_allowed
from lifemate.data.loader import load_samples
from run_overnight import score_evaluation


def compare(payloads, samples):
    if len(payloads)<2:raise ValueError('at least two experiments required')
    expected=None; protocol=None; replay_identity=None; results={}
    for name,payload in payloads.items():
        closed=payload.get('closed_loop') or {}
        if closed.get('mock_env') or payload.get('mock_env'):raise ValueError('mock environment cannot support this comparison')
        selection=closed.get('heldout_selection') or {}; ids=selection.get('ids')
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):raise ValueError('held-out IDs missing or duplicated')
        current=(closed.get('evaluation_mode'),closed.get('runtime_protocol'),closed.get('history_protocol'))
        if current[0]!='audio_to_action' or current[2]!='forced_audio_asr':raise ValueError('real-ASR history protocol required')
        if expected is None:
            expected=ids;protocol=current;replay_identity=closed.get('replay_identity')
        if closed.get('replay_identity') != replay_identity:
            raise ValueError('ASR replay artifact identities differ')
        if ids!=expected or current!=protocol:raise ValueError('held-out IDs or evaluation protocol differ')
        if closed.get('degraded_features'):raise ValueError('degraded ASR feature invalidates comparison')
        result=score_evaluation(payload,ids,samples)
        grouped={}
        for row in closed['rows']:
            category=row.get('asr_disagreement')
            if category:grouped.setdefault(category,[]).append(row)
        disagreement={}
        for category, group in grouped.items():
            group_ids=[row['id'] for row in group]
            disagreement[category]=score_evaluation({'closed_loop':{'rows':group,'heldout_selection':{'ids':group_ids}}},group_ids,samples)
        results[name]={'diagnostic':result,'latency':closed.get('latency',{'available':False}),
                       'status_counts':closed.get('status_counts',{}),'feature_execution':closed.get('feature_execution',{}),
                       'official_score':'UNAVAILABLE','final_state_correctness':closed.get('final_state_correctness',{'available':False})}
        results[name]['by_asr_disagreement'] = disagreement if disagreement else {'available':False}
    return {'ids':expected,'protocol':protocol,'experiments':results,'note':'local diagnostic only; not an official score'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',action='append',required=True,metavar='NAME=METRICS_JSON');p.add_argument('--data-root',default='data');p.add_argument('--output',required=True);args=p.parse_args()
    payloads={}
    for value in args.experiment:
        if '=' not in value:raise ValueError('experiment must be NAME=METRICS_JSON')
        name,path=value.split('=',1)
        if not name or name in payloads:raise ValueError('experiment names must be unique')
        path=assert_path_allowed(path);assert_path_allowed(path.resolve());payloads[name]=json.loads(path.read_text(encoding='utf-8'))
    target=assert_path_allowed(args.output);assert_path_allowed(target.resolve())
    if target.exists():raise ValueError('output already exists')
    report=compare(payloads,load_samples(args.data_root,'train'))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(f'wrote {target}')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as exc:print(f'error: {exc}',file=sys.stderr);sys.exit(2)
