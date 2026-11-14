"""Optional audio conditioning: energy VAD trim + RMS levelling.

Pure Python, operating on ``list[float]`` samples in [-1, 1] — the same shape
``lifemate.asr.interface.load_audio_16k_mono`` returns — so it imports and runs
under the stdlib interpreter (no numpy, no torch, no silero). Nothing here
resamples: the caller's ``sample_rate`` is only used to size frames, so 8 kHz and
44.1 kHz inputs work unchanged.

Channel convention: samples are treated as a FLAT INTERLEAVED buffer. This
module never mixes down and never guesses a channel count from the length, so
``trim_silence(..., channels=2)`` is how a stereo buffer stays aligned: frame and
hop sizes are whole numbers of interleaved samples, the kept span is snapped to
the interleave, and each 20 ms frame is scored by RMS over all of its samples
(an average across channels, which is channel-order-insensitive). Pass the
channel count you already know; the default 1 means "mono".

Both functions are disabled by default and return the INPUT OBJECT UNCHANGED
(``changed=False``, no copy) when ``enabled=False``, so a disabled call cannot
alter the baseline path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["AudioResult", "normalize_rms", "trim_silence"]

#: VAD frame length and hop (seconds). 20 ms frames on a 10 ms hop.
_FRAME_S = 0.020
_HOP_S = 0.010
#: A trimmed span shorter than this is treated as "nothing to trim".
_MIN_KEEP_S = 0.100

def _dbfs(amplitude: float) -> float:
    """20*log10 of an RMS/peak amplitude; digital silence is ``-inf``."""
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)

def _rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(float(x) * float(x) for x in samples) / len(samples))

def _peak(samples: Sequence[float]) -> float:
    return max((abs(float(x)) for x in samples), default=0.0)

def _stats(samples_in: Sequence[float], samples_out: Sequence[float], gain_db: float) -> dict:
    """The always-present measurement block (before/after, no invented values)."""
    rms_in, rms_out = _rms(samples_in), _rms(samples_out)
    return {
        "rms_dbfs_before": _dbfs(rms_in),
        "rms_dbfs_after": _dbfs(rms_out),
        "peak_before": _peak(samples_in),
        "peak_after": _peak(samples_out),
        "n_samples_in": len(samples_in),
        "n_samples_out": len(samples_out),
        "gain_db": gain_db,
    }

@dataclass
class AudioResult:
    """Processed samples plus the statistics that justify the decision."""

    samples: list[float]
    enabled: bool
    changed: bool
    stats: dict = field(default_factory=dict)

def _frame_bounds(total: int, frame_len: int, hop: int) -> list[tuple[int, int]]:
    """Half-open frame windows; the final one is clipped to the buffer end."""
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        bounds.append((start, min(start + frame_len, total)))
        start += hop
    return bounds

def trim_silence(
    samples: list[float],
    *,
    sample_rate: int = 16000,
    enabled: bool = False,
    min_silence_ms: int = 300,
    padding_ms: int = 200,
    threshold_dbfs: float = -40.0,
    channels: int = 1,
) -> AudioResult:
    """Trim leading/trailing silence, keeping interior pauses and padding.

    Energy-based: 20 ms frames on a 10 ms hop, a frame counts as speech when its
    RMS exceeds ``threshold_dbfs``. The kept span runs from the first to the last
    speech frame plus ``padding_ms`` on both sides — interior silence is never
    removed. Returns the input unchanged when every frame is silent
    (``stats["all_silent"]=True``) or when the surviving span would be shorter
    than ~100 ms. ``min_silence_ms`` is the run of silence that must be present
    at an edge for the trim to be attempted at all.
    """
    if not enabled:
        return AudioResult(
            samples=samples, enabled=False, changed=False, stats=_stats(samples, samples, 0.0)
        )

    total = len(samples)
    channels = max(1, int(channels))
    # Lengths are in interleaved samples: one audio second holds
    # ``sample_rate * channels`` of them.
    frame_len = max(channels, int(round(sample_rate * _FRAME_S)) * channels)
    hop = max(channels, int(round(sample_rate * _HOP_S)) * channels)
    if total == 0:
        stats = _stats(samples, samples, 0.0)
        stats.update({"all_silent": True, "trimmed": False})
        return AudioResult(samples=samples, enabled=True, changed=False, stats=stats)

    bounds = _frame_bounds(total, frame_len, hop)
    speech = [
        index
        for index, (start, end) in enumerate(bounds)
        if _dbfs(_rms(samples[start:end])) > threshold_dbfs
    ]
    if not speech:
        stats = _stats(samples, samples, 0.0)
        stats.update({"all_silent": True, "trimmed": False})
        return AudioResult(samples=samples, enabled=True, changed=False, stats=stats)

    leading = bounds[speech[0]][0]
    trailing = total - bounds[speech[-1]][1]
    if max(leading, trailing) < int(sample_rate * min_silence_ms / 1000):
        # Neither edge carries a long enough silence run: leave the clip alone.
        stats = _stats(samples, samples, 0.0)
        stats.update({"all_silent": False, "trimmed": False, "leading_silence": leading,
                      "trailing_silence": trailing})
        return AudioResult(samples=samples, enabled=True, changed=False, stats=stats)

    pad = max(0, int(round(sample_rate * padding_ms / 1000)))
    start = max(0, bounds[speech[0]][0] - pad)
    end = min(total, bounds[speech[-1]][1] + pad)
    start -= start % channels  # keep interleaved frames aligned
    end -= end % channels
    if end - start < int(sample_rate * _MIN_KEEP_S) or end <= start:
        stats = _stats(samples, samples, 0.0)
        stats.update({"all_silent": False, "trimmed": False})
        return AudioResult(samples=samples, enabled=True, changed=False, stats=stats)

    kept = list(samples[start:end])
    stats = _stats(samples, kept, 0.0)
    stats.update({"all_silent": False, "trimmed": end - start < total, "trim_start": start, "trim_end": end})
    return AudioResult(samples=kept, enabled=True, changed=end - start < total, stats=stats)

def normalize_rms(
    samples: list[float],
    *,
    enabled: bool = False,
    target_dbfs: float = -23.0,
    peak_ceiling: float = 0.99,
) -> AudioResult:
    """Scale toward ``target_dbfs``, capping the peak at ``peak_ceiling``.

    When the gain needed to reach the target would push the peak past the
    ceiling, a smaller gain is applied and ``stats["limited"]=True`` records it:
    levelling never clips. Silence (RMS 0) is returned unchanged — no NaN, no
    division — and a gain of exactly 1.0 is a no-op rather than a copy.
    """
    rms_before = _rms(samples)
    if not enabled or rms_before <= 0.0 or not samples:
        stats = _stats(samples, samples, 0.0)
        stats["limited"] = False
        return AudioResult(samples=samples, enabled=enabled, changed=False, stats=stats)

    peak_before = _peak(samples)
    gain_db = target_dbfs - _dbfs(rms_before)
    gain = 10.0 ** (gain_db / 20.0)
    limited = False
    if peak_before > 0.0 and peak_before * gain > peak_ceiling:
        gain = peak_ceiling / peak_before
        limited = True
    if abs(gain - 1.0) < 1e-12:
        stats = _stats(samples, samples, 0.0)
        stats["limited"] = limited
        return AudioResult(samples=samples, enabled=True, changed=False, stats=stats)

    ceiling = abs(peak_ceiling)
    out = [max(-ceiling, min(ceiling, float(x) * gain)) for x in samples]
    applied_db = 20.0 * math.log10(gain) if gain > 0 else 0.0
    stats = _stats(samples, out, applied_db)
    stats["limited"] = limited
    return AudioResult(samples=out, enabled=True, changed=True, stats=stats)
