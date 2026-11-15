"""Versioned multi-ASR utterance records without gold transcript substitution."""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from lifemate.compliance import assert_path_allowed
from lifemate.errors import LifeMateError
from lifemate.data.history import _audio_path, _turn_role, _turns_of, _writeable_turn
from lifemate.data.loader import read_sample_audio
from lifemate.evaluation.transcripts import _artifact_identity

SCHEMA='lifemate.multi_asr_top1.v1'


def _hash(path):
    h=hashlib.sha256()
    path=assert_path_allowed(path); assert_path_allowed(path.resolve())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _jsonable(value):
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value,dict):return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable(v) for v in value]
    return value


def _digest_record(row):
    body={k:v for k,v in row.items() if k!='record_sha256'}
    return hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def _utterances(samples, split, data_root):
    for sample in samples:
        sid=str(sample.id)
        for index,turn in enumerate(_turns_of(sample)):
            if _turn_role(turn) != 'user':continue
            raw=_writeable_turn(turn) or {}
            yield sid, f'history:{index}', _audio_path(raw,split,data_root) if raw.get('audio') else None
        yield sid, 'current', read_sample_audio(sample)


def produce_multi_records(samples, backend, features, *, split, data_root, output, transcribe, resume=False):
    """Write ordered records and a manifest; resume only a matching validated prefix."""
    output=assert_path_allowed(output); assert_path_allowed(output.resolve())
    partial=output.with_name(output.name+'.partial'); manifest=output.with_name(output.name+'.manifest.json')
    tmp=manifest.with_name(manifest.name+'.tmp')
    for p in (partial,manifest,tmp): assert_path_allowed(p.resolve())
    if any(p.is_symlink() for p in (output,partial,manifest,tmp)):
        raise LifeMateError('refusing symlinked output artifact')
    if (output.exists() or manifest.exists()) and not resume:
        raise LifeMateError('output or manifest already exists; refusing overwrite')
    if not getattr(backend,'backends',None): raise LifeMateError('explicit multi-model Top-1 backend required')
    identities={}
    for name,model in backend.backends.items():
        directory=getattr(model,'model_dir',None)
        model_name=getattr(model,'model_name',None)
        if not directory and model_name and Path(str(model_name)).is_dir(): directory=model_name
        identities[name]={'model_id':getattr(model,'model_id',None),'revision':getattr(model,'revision',None),
            'artifacts':_artifact_identity(directory),
            'resolved':{str(k):_artifact_identity(v) for k,v in (getattr(model,'_resolved_paths',{}) or {}).items()}}
    utterances=list(_utterances(samples,split,data_root))
    inputs=[{'sample_id':sid,'turn_id':turn,'audio_sha256':_hash(path) if path and path.is_file() else None} for sid,turn,path in utterances]
    identity={'schema':SCHEMA,'split':split,'inputs':inputs,'models':identities,'features':_jsonable(features)}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    if output.exists():
        source=manifest if manifest.is_file() else tmp
        if not resume or not source.is_file():raise LifeMateError('completed artifact lacks recoverable manifest')
        meta=json.loads(source.read_text(encoding='utf-8'))
        if meta.get('run_identity')!=identity_hash or meta.get('sha256')!=_hash(output) or meta.get('records')!=len(inputs):
            raise LifeMateError('completed artifact identity or checksum mismatch')
        if not manifest.exists():os.link(tmp,manifest)
        if partial.exists() and _hash(partial)==meta['sha256']:partial.unlink()
        if tmp.exists():tmp.unlink()
        return meta
    if manifest.exists():raise LifeMateError('manifest exists without completed output')
    prior=[]
    if partial.exists():
        if not resume:raise LifeMateError('partial output exists; pass resume')
        for line in partial.read_text(encoding='utf-8').splitlines():
            try:row=json.loads(line)
            except ValueError as exc:raise LifeMateError('invalid partial record') from exc
            if row.get('run_identity') != identity_hash:raise LifeMateError('partial identity mismatch')
            if row.get('record_sha256') != _digest_record(row):raise LifeMateError('partial record checksum mismatch')
            prior.append(row)
        if len(prior)>len(inputs) or any({k:r.get(k) for k in ('sample_id','turn_id','audio_sha256')} != inputs[i] for i,r in enumerate(prior)):
            raise LifeMateError('partial order or audio identity mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    flags=os.O_WRONLY|os.O_APPEND|os.O_CREAT|getattr(os,'O_NOFOLLOW',0)
    with os.fdopen(os.open(partial,flags,0o600),'a',encoding='utf-8') as f:
        for i,(sid,turn,path) in enumerate(utterances[len(prior):],start=len(prior)):
            proxy=SimpleNamespace(id=f'{sid}:{turn}',audio_path=path)
            text,info=transcribe(backend,proxy,features=features)
            row={'schema':SCHEMA,'run_identity':identity_hash,'split':split,**inputs[i],
                 'models_requested':list(backend.backends),'models':info.get('multi_asr_results') or [{'model':name,'status':'blocked','error':info.get('error') or 'not executed'} for name in backend.backends],
                 'selected_text':text,'candidates':_jsonable(info.get('hypotheses',[])),
                 'status':'failed' if info.get('error') else ('degraded' if (info.get('conditioning',{}).get('multi_asr',{}).get('degraded')) else 'complete'),
                 'error':info.get('error'),'latency_s':info.get('latency_s'),'conditioning':info.get('conditioning',{})}
            row['record_sha256']=_digest_record(row)
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    checksum=_hash(partial)
    meta={'schema':SCHEMA,'run_identity':identity_hash,'sha256':checksum,'records':len(inputs),'identity':identity}
    if tmp.exists():
        old=json.loads(tmp.read_text(encoding='utf-8'))
        if old.get('run_identity')!=identity_hash or old.get('sha256')!=checksum:
            raise LifeMateError('staged manifest does not match partial output')
    else:
        with tmp.open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,sort_keys=True,default=str);f.flush();os.fsync(f.fileno())
    # Link rather than replace: an existing artifact is never overwritten.
    os.link(partial,output);os.link(tmp,manifest);partial.unlink();tmp.unlink()
    return meta
