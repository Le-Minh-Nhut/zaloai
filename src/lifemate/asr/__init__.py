"""ASR backends: oracle (eval), whisper (real inference), factory."""

from __future__ import annotations

from typing import Any, Mapping

from .interface import (
    ASRBackend,
    AsrError,
    TranscriptResult,
    load_audio_16k_mono,
    resample_linear,
    wav_info,
)
from .metrics import evaluate_asr, save_asr_metrics, unavailable_metrics
from .normalization import (
    cer,
    fold_diacritics,
    levenshtein,
    normalize_vi,
    normalization_report,
    tokenize_words,
    wer,
)
from .oracle_backend import OracleTranscriptBackend

__all__ = [
    "ASRBackend",
    "AsrError",
    "TranscriptResult",
    "GipformerBackend",
    "OracleTranscriptBackend",
    "WhisperBackend",
    "load_audio_16k_mono",
    "resample_linear",
    "wav_info",
    "normalize_vi",
    "fold_diacritics",
    "levenshtein",
    "tokenize_words",
    "normalization_report",
    "cer",
    "wer",
    "evaluate_asr",
    "save_asr_metrics",
    "unavailable_metrics",
]


def __getattr__(name: str) -> Any:
    # Heavy backends are exported lazily: importing lifemate.asr must not require torch/sherpa.
    if name == "WhisperBackend":
        from .whisper_backend import WhisperBackend  # noqa: PLC0415

        return WhisperBackend
    if name == "GipformerBackend":
        from .gipformer_backend import GipformerBackend  # noqa: PLC0415

        return GipformerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_asr_backend(
    cfg: Any = None,
    transcripts: Mapping[str, str] | Any | None = None,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> ASRBackend:
    """Build a backend from ``AppConfig.asr`` (or explicit overrides).

    ``backend`` values: ``gipformer`` (default) | ``whisper`` | ``oracle`` | ``none``.
    ``none`` is a valid *configured* value but constructing it raises a clear
    error, so a misconfiguration fails loudly rather than transcribing nothing.
    """
    asr_cfg = getattr(cfg, "asr", None)

    def _get(field: str, default: Any) -> Any:
        if asr_cfg is None:
            return default
        if isinstance(asr_cfg, Mapping):
            return asr_cfg.get(field, default)
        return getattr(asr_cfg, field, default)

    name = (backend or _get("backend", "gipformer") or "gipformer").strip().lower()

    if name == "gipformer":
        from .gipformer_backend import GipformerBackend  # noqa: PLC0415

        opts = {
            "model_dir": _get("model_dir", None),
            "model_id": _get("model_id", "g-group-ai-lab/gipformer-68M-rnnt"),
            "revision": _get("revision", "318322b009200f8e77a3751fa64638e5cc85a7dc"),
            "quantization": _get("quantization", "int8"),
            "device": _get("device", "cpu"),
            "num_threads": _get("num_threads", 4),
            "decoding_method": _get("decoding_method", "modified_beam_search"),
            "offline": _get("offline", True),
            "max_audio_seconds": _get("max_audio_seconds", 60.0),
        }
        opts.update(kwargs)
        return GipformerBackend(**opts)

    if name == "oracle":
        if transcripts is None:
            raise AsrError(
                "backend='oracle' requires transcripts (mapping or callable); "
                "oracle ASR is for training/evaluation only"
            )
        return OracleTranscriptBackend(transcripts, **kwargs)

    if name == "whisper":
        from .whisper_backend import WhisperBackend  # noqa: PLC0415

        opts = {
            "model_name": _get("model_name", "vinai/PhoWhisper-small"),
            "device": _get("device", "auto"),
            "language": _get("language", "vi"),
            "max_audio_seconds": _get("max_audio_seconds", 600.0),
        }
        opts.update(kwargs)
        return WhisperBackend(**opts)  # type: ignore[arg-type]

    if name == "none":
        raise AsrError(
            "ASR backend 'none' is configured but cannot transcribe. "
            "Choose 'gipformer' for real inference, 'whisper' for comparison, "
            "or 'oracle' for evaluation."
        )

    raise AsrError(
        f"ASR backend {name!r} is not usable. Choose 'gipformer' (default), "
        "'whisper' for comparison, or 'oracle' for transcript-based evaluation."
    )
