"""ASR backend contract, result type, and stdlib audio loading/resampling.

Import-safe without torch/transformers/numpy.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lifemate.errors import LifeMateError
__all__ = [
    "AsrError",
    "TranscriptResult",
    "ASRBackend",
    "TARGET_SAMPLE_RATE",
    "DEFAULT_MAX_AUDIO_SECONDS",
    "load_audio_16k_mono",
    "resample_linear",
    "wav_info",
]

#: Whisper-family models and the Zalo fixtures are 16 kHz mono.
TARGET_SAMPLE_RATE = 16000

#: Hard guard: refuse to load more than this many seconds of audio.
#: Overridable per call (``max_seconds``) from ``cfg.asr.max_audio_seconds``.
DEFAULT_MAX_AUDIO_SECONDS = 600.0


class AsrError(LifeMateError, RuntimeError):
    """Raised for unrecoverable ASR problems (missing package, bad audio, no model)."""


def candidate_count(value: Any, *, maximum: int = 3) -> int:
    """Reject lossy conversion, booleans and non-finite values before loading."""
    import math
    if isinstance(value, bool):
        raise AsrError('candidate count must be an integer')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AsrError('candidate count must be an integer') from exc
    if not math.isfinite(number) or not number.is_integer() or not 1 <= number <= maximum:
        raise AsrError(f'candidate count must be an integer between 1 and {maximum}')
    return int(number)


@dataclass
class TranscriptResult:
    """One transcription attempt. ``error`` set and ``text == ""`` on failure."""

    text: str
    raw_text: str | None = None
    language: str | None = None
    duration_s: float | None = None
    latency_s: float = 0.0
    backend: str = "unknown"
    model_name: str | None = None
    error: str | None = None
    #: Sum of the recognizer's per-token log probabilities (``ys_log_probs``).
    #: Only comparable between decodes of the SAME model; ``None`` when the
    #: backend exposes no such field. Never a confidence for another model.
    acoustic_logprob: float | None = None
    #: Emitted token count backing ``acoustic_logprob`` (``None`` when unknown).
    n_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "language": self.language,
            "duration_s": self.duration_s,
            "latency_s": self.latency_s,
            "backend": self.backend,
            "model_name": self.model_name,
            "error": self.error,
            "acoustic_logprob": self.acoustic_logprob,
            "n_tokens": self.n_tokens,
        }

    def __post_init__(self) -> None:
        # An errored result must never carry text: no fabricated transcripts.
        if self.error is not None and self.text:
            self.text = ""


@runtime_checkable
class ASRBackend(Protocol):
    """Structural protocol every backend satisfies."""

    name: str
    is_oracle: bool

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        sample_id: str | None = None,
        lenient: bool = False,
    ) -> TranscriptResult:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Audio loading (stdlib only)
# ---------------------------------------------------------------------------


def wav_info(path: str | Path) -> dict[str, Any]:
    """Cheap header read: rate/channels/width/frames/duration."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            return {
                "ok": True,
                "sample_rate": rate,
                "channels": wf.getnchannels(),
                "sample_width_bits": wf.getsampwidth() * 8,
                "n_frames": frames,
                "duration_s": (frames / rate) if rate else 0.0,
            }
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _decode_frame(raw: bytes, sample_width: int) -> list[int]:
    """Decode one interleaved frame of PCM bytes into signed ints."""
    if sample_width == 1:  # 8-bit WAV is unsigned offset-binary
        return [b - 128 for b in raw]
    if sample_width == 2:
        return [
            int.from_bytes(raw[i : i + 2], "little", signed=True)
            for i in range(0, len(raw), 2)
        ]
    if sample_width == 3:
        return [
            int.from_bytes(raw[i : i + 3], "little", signed=True)
            for i in range(0, len(raw), 3)
        ]
    if sample_width == 4:
        return [
            int.from_bytes(raw[i : i + 4], "little", signed=True)
            for i in range(0, len(raw), 4)
        ]
    raise AsrError(f"unsupported sample width: {sample_width * 8} bits")


def resample_linear(
    samples: list[float], src_rate: int, dst_rate: int = TARGET_SAMPLE_RATE
) -> list[float]:
    """Linear-interpolation resampler. Identity when rates match."""
    if src_rate <= 0 or dst_rate <= 0:
        raise AsrError(f"invalid sample rate: src={src_rate} dst={dst_rate}")
    if src_rate == dst_rate or not samples:
        return list(samples)
    n_out = int(round(len(samples) * dst_rate / src_rate))
    if n_out <= 0:
        return []
    ratio = src_rate / dst_rate
    last = len(samples) - 1
    out: list[float] = []
    for i in range(n_out):
        pos = i * ratio
        left = int(pos)
        frac = pos - left
        if left >= last:
            out.append(samples[last])
        else:
            out.append(samples[left] * (1.0 - frac) + samples[left + 1] * frac)
    return out


def load_audio_16k_mono(
    path: str | Path,
    target_rate: int = TARGET_SAMPLE_RATE,
    max_seconds: float | None = DEFAULT_MAX_AUDIO_SECONDS,
) -> tuple[list[float], int]:
    """Load a PCM WAV as mono float samples in [-1, 1) at ``target_rate``.

    * 8/16/24/32-bit PCM supported; float and compressed WAVs raise AsrError.
    * Multi-channel audio is averaged (mono mixdown).
    * Samples are only scaled by ``1 / 2**(bits-1)``; no peak normalization.
    * ``max_seconds`` truncates the read (never loads more than the guard).
    """
    path = Path(path)
    from ..compliance import assert_path_allowed

    path = assert_path_allowed(path)          # *_environment paths are never opened
    # A symlink inside an allowed directory can point INTO a *_environment tree;
    # the lexical guard above cannot see that, so check the resolved target too.
    assert_path_allowed(path.resolve())
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            total = wf.getnframes()
            if width not in (1, 2, 3, 4):
                raise AsrError(
                    f"unsupported WAV sample width {width * 8} bits in {path.name}; "
                    "expected PCM 8/16/24/32-bit (float or compressed WAV not supported)"
                )
            want = total
            if max_seconds is not None and rate > 0:
                want = min(total, int(max_seconds * rate))
            raw = wf.readframes(want)
    except wave.Error as exc:
        raise AsrError(f"cannot read WAV {path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise AsrError(f"audio file not found: {path}") from exc
    except OSError as exc:
        raise AsrError(f"cannot open audio {path}: {exc}") from exc

    if not raw:
        return [], target_rate

    ints = _decode_frame(raw, width)
    scale = float(1 << (width * 8 - 1))
    if channels > 1:
        n_frames = len(ints) // channels
        mono = [
            sum(ints[f * channels : (f + 1) * channels]) / channels
            for f in range(n_frames)
        ]
    else:
        mono = ints
    floats = [v / scale for v in mono]
    if rate != target_rate:
        floats = resample_linear(floats, rate, target_rate)
    return floats, target_rate


