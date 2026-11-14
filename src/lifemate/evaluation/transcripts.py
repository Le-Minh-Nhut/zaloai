"""ASR transcript cache for the audio-to-action evaluation mode.

A transcript depends on the ASR configuration, the split, the exact sample set
and the feature fingerprint (VAD/RMS/ITN/N-best change the text). The cache key
is a hash of exactly those four things, so a run can never reuse a transcript
produced by a different recogniser or a different feature set.

Stdlib only; no ASR import (the caller owns the backend).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["transcript_cache_key", "load_cache", "save_cache"]

#: ASR fields that change the recognised text (everything else is run-local).
_KEYED_ASR_FIELDS = (
    "backend",
    "model_id",
    "revision",
    "model_dir",
    "runtime",
    "quantization",
    "decoding_method",
    "language",
    "sample_rate",
)

def _asr_signature(cfg: Any) -> dict[str, Any]:
    asr = getattr(cfg, "asr", None)
    return {name: getattr(asr, name, None) for name in _KEYED_ASR_FIELDS}

def transcript_cache_key(
    cfg: Any,
    features: Any,
    *,
    split: str,
    sample_ids: Iterable[str],
) -> str:
    """Stable key over (ASR config, split, sample ids, feature fingerprint)."""
    try:
        from lifemate.features.config import feature_fingerprint  # noqa: PLC0415
    except ImportError:  # pragma: no cover - features slice absent
        fingerprint = ""
    else:
        fingerprint = feature_fingerprint(features) if features is not None else ""
    payload = {
        "asr": _asr_signature(cfg),
        "split": str(split),
        "sample_ids": sorted(str(item) for item in sample_ids),
        "feature_fingerprint": fingerprint,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def load_cache(path: str | Path) -> dict[str, dict[str, str]]:
    """``key -> {sample_id: transcript}``; a missing or broken file is empty."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    cache: dict[str, dict[str, str]] = {}
    for key, entry in payload.items():
        transcripts = entry.get("transcripts") if isinstance(entry, Mapping) else None
        if isinstance(transcripts, Mapping):
            cache[str(key)] = {str(sid): str(text) for sid, text in transcripts.items()}
    return cache

def save_cache(path: str | Path, key: str, transcripts: Mapping[str, str], **meta: Any) -> Path:
    """Merge ``transcripts`` under ``key`` into the cache file (atomic write)."""
    target = Path(path)
    payload = {
        existing_key: {"transcripts": existing}
        for existing_key, existing in load_cache(target).items()
    }
    payload[key] = {"transcripts": dict(transcripts), **meta}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
