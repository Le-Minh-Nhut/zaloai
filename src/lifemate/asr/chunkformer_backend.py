"""Offline ChunkFormer RNNT Top-1 adapter using the upstream Python API."""
from __future__ import annotations
import time
from pathlib import Path
from lifemate.compliance import assert_path_allowed
from lifemate.errors import ProhibitedAccessError
from .interface import AsrError, TranscriptResult


class ChunkFormerBackend:
    name = 'chunkformer'
    is_oracle = False
    model_id = 'khanhld/chunkformer-rnnt-large-vie'

    def __init__(self, model_dir=None, *, revision=None, device='cpu', chunk_size=64,
                 left_context_size=128, right_context_size=128, total_batch_duration=1800):
        self.model_dir = Path(model_dir) if model_dir else None
        self.revision = revision
        self.device = device
        self.chunk_size, self.left_context_size, self.right_context_size = chunk_size, left_context_size, right_context_size
        self.total_batch_duration = total_batch_duration
        self._model = None

    def _load(self):
        if self._model is not None: return
        if self.model_dir is None: raise AsrError('ChunkFormer requires a pre-existing local model_dir; no download is attempted')
        root = assert_path_allowed(self.model_dir); assert_path_allowed(root.resolve())
        for filename in ('config.yaml', 'global_cmvn', 'pytorch_model.pt', 'vocab.txt'):
            target = assert_path_allowed(root / filename); assert_path_allowed(target.resolve())
            if not target.is_file(): raise AsrError(f'ChunkFormer artifact missing: {target}')
        try:
            from chunkformer import ChunkFormerModel
        except ImportError as exc:
            raise AsrError('ChunkFormer runtime missing: install the upstream chunkformer package') from exc
        self._model = ChunkFormerModel.from_pretrained(str(root)).to(self.device).eval()

    def transcribe(self, audio_path, *, sample_id=None, lenient=False):
        t0=time.monotonic()
        try:
            path=assert_path_allowed(audio_path); assert_path_allowed(path.resolve())
            self._load()
            text=self._model.endless_decode(audio_path=str(path), chunk_size=self.chunk_size,
                left_context_size=self.left_context_size, right_context_size=self.right_context_size,
                total_batch_duration=self.total_batch_duration, return_timestamps=False)
            if not isinstance(text,str): raise AsrError('ChunkFormer did not return decoded text; check tokenizer artifacts')
            return TranscriptResult(text=text, raw_text=text, backend=self.name, model_name=self.model_id, latency_s=time.monotonic()-t0)
        except ProhibitedAccessError: raise
        except Exception as exc:
            if not lenient: raise AsrError(f'ChunkFormer failed: {exc}') from exc
            return TranscriptResult(text='',backend=self.name,model_name=self.model_id,error=f'{type(exc).__name__}: {exc}',latency_s=time.monotonic()-t0)
