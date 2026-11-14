"""Shared transcription helpers and transcript caching."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

from lifemate.errors import LifeMateError
from lifemate.evaluation.transcripts import load_cache, save_cache, transcript_cache_key

__all__ = [
    "infer_module",
    "transcribe_samples_cached",
]


def infer_module() -> Any:
    """Import ``scripts/infer.py`` by path (scripts/ is not a package)."""
    infer_path = Path(__file__).resolve().parents[3] / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location("lifemate_infer_script", infer_path)
    if spec is None or spec.loader is None:
        raise LifeMateError(f"cannot load {infer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transcribe_samples_cached(
    samples: Sequence[Any],
    cfg: Any,
    *,
    features: Any = None,
    asr: Any = None,
    split: str = "val",
    cache_path: str | Path | None = None,
    args: Any = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Transcribe samples with caching keyed by ASR config + split + sample IDs + feature fingerprint.

    Returns (transcripts_dict, stats_dict).
    """
    features = features if features is not None else getattr(cfg, "features", None)
    sample_ids = [str(getattr(s, "id", "") or "") for s in samples]
    key = transcript_cache_key(cfg, features, split=split, sample_ids=sample_ids)

    target_cache_path = Path(cache_path) if cache_path else Path("outputs/cache/transcripts.json")
    loaded = load_cache(target_cache_path)
    cached_transcripts = dict(loaded.get(key, {}))

    missing = [s for s in samples if str(getattr(s, "id", "") or "") not in cached_transcripts]
    stats = {"cache_hit": len(cached_transcripts), "cache_miss": len(missing), "cache_key": key}
    if missing:
        backend = asr
        infer = infer_module()
        if backend is None:
            backend = infer._build_asr(args, cfg, samples)
        for s in missing:
            sid = str(getattr(s, "id", "") or "")
            text, _info = infer._transcribe(backend, s, features=features)
            cached_transcripts[sid] = text
        save_cache(target_cache_path, key, cached_transcripts)

    return cached_transcripts, stats
