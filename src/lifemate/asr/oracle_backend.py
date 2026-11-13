"""Oracle transcript backend.

Resolves the ground-truth transcript for a given audio file. **This is an
evaluation/training shortcut, never an inference path**: feeding gold transcripts
to a submission would fabricate ASR performance.

Docstring contract (enforced by tests/test_asr.py::test_no_gold_transcript_leak_in_inference_path):
    "training/eval ONLY; MUST NOT be used for public/private test inference"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Mapping

from .interface import AsrError, TranscriptResult, wav_info

__all__ = ["OracleTranscriptBackend"]

class OracleTranscriptBackend:
    """Look up transcripts by sample id, file stem, or full file name.

    ``transcripts`` may be a mapping or a callable ``(key) -> str | None``.
    ``strict=True`` (default) raises :class:`AsrError` when the transcript is
    missing, so a silent empty transcript can never be mistaken for a prediction.
    """

    name = "oracle"
    #: Machine-checkable marker: any inference pipeline must reject this backend.
    is_oracle = True

    def __init__(
        self,
        transcripts: Mapping[str, str] | Callable[[str], str | None],
        *,
        strict: bool = True,
        model_name: str | None = "oracle-transcripts",
    ) -> None:
        if transcripts is None:
            raise AsrError("OracleTranscriptBackend requires a transcript mapping or callable")
        self._lookup = transcripts
        self.strict = bool(strict)
        self.model_name = model_name

    # -- lookup ------------------------------------------------------------
    def _candidates(self, audio_path: str | Path) -> list[str]:
        p = Path(audio_path)
        keys = [p.stem, p.name, str(p), str(p.with_suffix(""))]
        seen: set[str] = set()
        ordered = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                ordered.append(k)
        return ordered

    # -- backend API -------------------------------------------------------
    def transcribe(
        self, audio_path: str | Path, *, sample_id: str | None = None
    ) -> TranscriptResult:
        t0 = time.perf_counter()
        keys = ([sample_id] if sample_id else []) + self._candidates(audio_path)
        text: str | None = None
        for key in keys:
            try:
                if callable(self._lookup):
                    value = self._lookup(key)
                else:
                    value = self._lookup.get(key)  # type: ignore[union-attr]
            except KeyError:
                value = None
            if isinstance(value, str):
                text = value
                break
        latency = time.perf_counter() - t0
        if text is None:
            msg = f"no oracle transcript for {audio_path!r} (tried {keys[:3]})"
            if self.strict:
                raise AsrError(msg + "; strict=True forbids fabricating text")
            return TranscriptResult(
                text="",
                latency_s=latency,
                backend=self.name,
                model_name=self.model_name,
                error=msg,
            )
        info = wav_info(audio_path)
        return TranscriptResult(
            text=text,
            raw_text=text,
            language="vi",
            duration_s=info.get("duration_s") if info.get("ok") else None,
            latency_s=latency,
            backend=self.name,
            model_name=self.model_name,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"OracleTranscriptBackend(n={len(self._lookup) if not callable(self._lookup) else 'callable'}, strict={self.strict})"
