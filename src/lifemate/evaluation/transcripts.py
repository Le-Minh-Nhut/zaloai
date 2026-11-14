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

from lifemate.errors import ProhibitedAccessError

__all__ = [
    "transcript_cache_key",
    "load_cache",
    "save_cache",
    "load_cache_entries",
    "asr_cache_identity",
    "audio_identities",
    "CACHE_VERSION",
]

#: Cache schema version. Bumped to 2 when per-sample audio identity and the
#: full ASR-affecting parameter set joined the key; v1 entries never hit.
CACHE_VERSION = 2

#: ASR fields that change the recognised text (everything else is run-local).
_KEYED_ASR_FIELDS = (
    "backend",
    "model_id",
    "model_name",
    "revision",
    "model_dir",
    "runtime",
    "quantization",
    "decoding_method",
    "language",
    "sample_rate",
)


def _flag(features: Any, section: str, name: str) -> bool:
    return bool(getattr(getattr(features, section, None), name, False))


def _param(features: Any, section: str, name: str, default: Any) -> Any:
    value = getattr(getattr(features, section, None), name, default)
    return default if value is None else value


def _asr_signature(cfg: Any) -> dict[str, Any]:
    asr = getattr(cfg, "asr", None)
    return {name: getattr(asr, name, None) for name in _KEYED_ASR_FIELDS}


def asr_cache_identity(cfg: Any, features: Any) -> dict[str, Any]:
    """Every ASR-affecting knob that must invalidate the cache.

    ``feature_fingerprint`` semantics are untouched; this is the extra,
    cache-local identity covering VAD/RMS values, decoder params and the
    N-best/multi-ASR/v15 switches. Plain JSON values only.
    """
    asr = getattr(cfg, "asr", None)
    identity: dict[str, Any] = {"asr": _asr_signature(cfg)}
    identity["decoding_method"] = getattr(asr, "decoding_method", None)
    identity["nbest_size"] = getattr(asr, "nbest_size", None)
    identity["audio"] = {
        "vad_enabled": _flag(features, "audio", "vad_enabled"),
        "vad_min_silence_ms": _param(features, "audio", "vad_min_silence_ms", 300),
        "vad_padding_ms": _param(features, "audio", "vad_padding_ms", 200),
        "rms_normalization_enabled": _flag(features, "audio", "rms_normalization_enabled"),
        "rms_target_dbfs": _param(features, "audio", "rms_target_dbfs", -23.0),
    }
    identity["asr_features"] = {
        "gipformer_v15_enabled": _flag(features, "asr", "gipformer_v15_enabled"),
        "nbest_enabled": _flag(features, "asr", "nbest_enabled"),
        "nbest_size": _param(features, "asr", "nbest_size", 3),
        "multi_asr_enabled": _flag(features, "asr", "multi_asr_enabled"),
        "secondary_model_id": _param(features, "asr", "secondary_model_id", ""),
        "secondary_backend": _param(features, "asr", "secondary_backend", "whisper"),
        "multi_asr_policy": _param(features, "asr", "multi_asr_policy", "primary_topk_secondary_top1"),
        "multi_asr_max_hypotheses": _param(features, "asr", "multi_asr_max_hypotheses", 3),
        "multi_asr_failure_policy": _param(features, "asr", "multi_asr_failure_policy", "primary_only"),
    }
    identity["itn_enabled"] = _flag(features, "prompt", "itn_enabled")
    return identity


def audio_identities(samples: Iterable[Any]) -> dict[str, str]:
    """Per-sample SHA-256 of the existing audio file.

    Missing/unreadable audio maps to
    ``""`` so the key stays stable and the miss is handled downstream.
    """
    try:
        from lifemate.data.loader import read_sample_audio  # noqa: PLC0415
    except ImportError:  # pragma: no cover - loader always present here
        read_sample_audio = None  # type: ignore[assignment]
    identities: dict[str, str] = {}
    for sample in samples or ():
        sid = str(getattr(sample, "id", "") or "")
        tag = ""
        try:
            path = read_sample_audio(sample) if read_sample_audio is not None else None
            if path is not None:
                with Path(path).open("rb") as handle:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                tag = digest.hexdigest()
        except ProhibitedAccessError:
            raise
        except OSError:
            tag = ""
        identities[sid] = tag
    return identities


def transcript_cache_key(
    cfg: Any,
    features: Any,
    *,
    split: str,
    sample_ids: Iterable[str],
    audio_identities: Mapping[str, str] | None = None,
) -> str:
    """Stable key over (ASR config, split, sample ids, feature fingerprint).

    v2 adds the full ASR-affecting identity plus per-sample audio identity,
    so v1 entries never hit. ``audio_identities`` maps sample id to the cheap
    content digest from :func:`audio_identities`; ``None`` means audio unknown
    (still a v2 key, just without per-file tags).
    """
    try:
        from lifemate.features.config import feature_fingerprint  # noqa: PLC0415
    except ImportError:  # pragma: no cover - features slice absent
        fingerprint = ""
    else:
        fingerprint = feature_fingerprint(features) if features is not None else ""
    ids = sorted(str(item) for item in sample_ids)
    audio = {str(k): str(v) for k, v in (audio_identities or {}).items() if str(k) in set(ids)}
    payload = {
        "v": CACHE_VERSION,
        "asr": _asr_signature(cfg),
        "identity": asr_cache_identity(cfg, features),
        "split": str(split),
        "sample_ids": ids,
        "audio": audio,
        "feature_fingerprint": fingerprint,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def load_cache_entries(path: str | Path) -> dict[str, dict[str, Any]]:
    """Full cache entries (``key -> entry dict``); broken file reads as empty."""
    target = Path(path)
    if not target.is_file():
        return {}
    payload = _read_payload(target)
    entries: dict[str, dict[str, Any]] = {}
    for key, entry in payload.items():
        if isinstance(entry, Mapping):
            entries[str(key)] = dict(entry)
    return entries


def load_cache(path: str | Path) -> dict[str, dict[str, str]]:
    """``key -> {sample_id: transcript}``; a missing or broken file is empty."""
    entries = load_cache_entries(path)
    cache: dict[str, dict[str, str]] = {}
    for key, entry in entries.items():
        transcripts = entry.get("transcripts") if isinstance(entry, Mapping) else None
        if isinstance(transcripts, Mapping):
            cache[str(key)] = {str(sid): str(text) for sid, text in transcripts.items()}
    return cache


def load_hypotheses(path: str | Path, key: str) -> dict[str, list[dict[str, Any]]]:
    """Hypothesis dicts stored under ``key`` (``{}`` when absent or legacy)."""
    entries = load_cache_entries(path)
    entry = entries.get(str(key), {})
    raw = entry.get("hypotheses") if isinstance(entry, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, hyps in raw.items():
        if isinstance(hyps, list):
            out[str(sid)] = [dict(h) for h in hyps if isinstance(h, Mapping)]
    return out


def save_cache(
    path: str | Path,
    key: str,
    transcripts: Mapping[str, str],
    hypotheses: Mapping[str, list[dict[str, Any]]] | None = None,
    **meta: Any,
) -> Path:
    """Merge ``transcripts`` (plus optional hypothesis dicts) under ``key``."""
    target = Path(path)
    payload = {k: dict(v) for k, v in load_cache_entries(target).items()}
    entry: dict[str, Any] = {"transcripts": {str(k): str(v) for k, v in dict(transcripts).items()}}
    if hypotheses is not None:
        entry["hypotheses"] = {
            str(k): [dict(h) for h in v] for k, v in dict(hypotheses).items()
        }
    entry.update(meta)
    payload[str(key)] = entry
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
