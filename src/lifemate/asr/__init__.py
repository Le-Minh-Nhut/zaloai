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
from .transcription import (
    infer_module,
    transcribe_samples_cached,
)
from .model_specs import (
    GIPFORMER_SPECS,
    GipformerSpec,
    KNOWN_MODEL_IDS,
    require_spec_artifacts,
    resolve_gipformer_spec,
    validate_artifacts,
)
from .multi_backend import MultiAsrBackend
from .nbest import (
    NBEST_MAX,
    NBEST_MIN,
    NBestResult,
    NBestUnsupportedError,
    AsrHypothesis,
    require_nbest_support,
    rescore_configurations,
    runtime_nbest_capability,
    validate_nbest_size,
)
from .oracle_backend import OracleTranscriptBackend

__all__ = [
    "ASRBackend",
    "AsrError",
    "TranscriptResult",
    "GipformerBackend",
    "OracleTranscriptBackend",
    "WhisperBackend",
    "MultiAsrBackend",
    "AsrHypothesis",
    "NBestResult",
    "NBestUnsupportedError",
    "require_nbest_support",
    "rescore_configurations",
    "runtime_nbest_capability",
    "validate_nbest_size",
    "NBEST_MIN",
    "NBEST_MAX",
    "GipformerSpec",
    "GIPFORMER_SPECS",
    "KNOWN_MODEL_IDS",
    "resolve_gipformer_spec",
    "validate_artifacts",
    "require_spec_artifacts",
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
    "infer_module",
    "transcribe_samples_cached",
]

def __getattr__(name: str) -> Any:
    # Heavy backends are exported lazily: importing lifemate.asr must not require torch/sherpa.
    if name == "WhisperBackend":
        from .whisper_backend import WhisperBackend  # noqa: PLC0415

        return WhisperBackend
    if name == "GipformerBackend":
        from .gipformer_backend import GipformerBackend  # noqa: PLC0415

        return GipformerBackend
    if name == "create_asr_backend":
        return create_asr_backend
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
            "nbest_size": _get("nbest_size", 0),
        }
        opts.update(kwargs)
        return GipformerBackend(**opts)

    if name == "multi":
        # Composed backend: the caller supplies the real components, so nothing is
        # silently substituted here (no "secondary" that is secretly the primary).
        opts = {
            "enabled": _get("multi_asr_enabled", False),
            "policy": _get("multi_asr_policy", "primary_topk_secondary_top1"),
            "max_hypotheses": _get("multi_asr_max_hypotheses", 3),
            "failure_policy": _get("multi_asr_failure_policy", "primary_only"),
            "primary_topk": _get("multi_asr_primary_topk", 3),
            "budget_seconds": _get("multi_asr_budget_seconds", None),
        }
        opts.update(kwargs)
        if "primary" not in opts:
            raise AsrError(
                "backend='multi' requires primary=<ASRBackend> (and optionally secondary=<ASRBackend>); "
                "the factory will not invent either one"
            )
        return MultiAsrBackend(**opts)

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
        "'whisper' for comparison, 'multi' for a composed primary+secondary run, "
        "or 'oracle' for transcript-based evaluation."
    )
