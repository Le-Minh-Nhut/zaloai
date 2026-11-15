"""Per-model, content-addressed raw Top-1 cache; failures are never cached."""
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from lifemate.compliance import assert_path_allowed
from .interface import TranscriptResult


class CachedTop1:
    def __init__(self, backend, directory):
        self.backend = backend
        self.directory = Path(directory)
        self.last_cache_hit = False

    def __getattr__(self, name): return getattr(self.backend, name)

    def transcribe(self, audio_path, *, sample_id=None, lenient=False):
        from lifemate.evaluation.transcripts import _artifact_identity
        path=assert_path_allowed(audio_path); assert_path_allowed(path.resolve())
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
        artifacts=getattr(self.backend,'model_dir',None)
        if not artifacts:
            model_name=getattr(self.backend,'model_name',None)
            if model_name and Path(model_name).is_dir(): artifacts=model_name
        identity={'version':1,'audio_sha256':digest.hexdigest(),'sample_id':sample_id,
                  'backend':getattr(self.backend,'name',None),'model_id':getattr(self.backend,'model_id',None),
                  'model_name':getattr(self.backend,'model_name',None),'revision':getattr(self.backend,'revision',None),
                  'artifacts':_artifact_identity(artifacts),
                  'resolved':{str(k):_artifact_identity(v) for k,v in (getattr(self.backend,'_resolved_paths',{}) or {}).items()},
                  'decoder':{k:getattr(self.backend,k,None) for k in ('quantization','decoding_method','language','device','chunk_size','left_context_size','right_context_size','total_batch_duration')}}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
        root=assert_path_allowed(self.directory); assert_path_allowed(root.resolve()); target=root/(key+'.json')
        self.last_cache_hit=False
        try:
            payload=json.loads(target.read_text(encoding='utf-8'))
            if payload.get('key')==key and payload.get('result',{}).get('text') and not payload['result'].get('error'):
                self.last_cache_hit=True
                return TranscriptResult(**payload['result'])
        except (OSError, ValueError, TypeError): pass
        result=self.backend.transcribe(path,sample_id=sample_id,lenient=lenient)
        if result.text and not result.error:
            root.mkdir(parents=True,exist_ok=True); name=None
            try:
                with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=root,delete=False) as f:
                    name=f.name;json.dump({'key':key,'result':asdict(result)},f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
                os.replace(name,target)
            finally:
                if name and os.path.exists(name): os.unlink(name)
        return result
