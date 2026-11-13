"""Audio EDA with the stdlib :mod:`wave` module only.

All amplitude statistics are reported **on the int16 scale** (range roughly
[-32768, 32767]) regardless of the file's own bit depth, so a single documented
threshold (``silence_threshold = 500``) means the same thing everywhere.
Scaling is ``sample * 2**(15 - (bits-1))`` for integer PCM.

Prohibited simulator-directory paths are refused via
:func:`lifemate.compliance.assert_path_allowed` before any I/O.
"""

from __future__ import annotations

import math
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..compliance import assert_path_allowed

__all__ = [
    "SILENCE_THRESHOLD",
    "EXPECTED_SAMPLE_RATE",
    "analyze_wav",
    "analyze_split_audio",
    "pearson",
    "flag_counts",
]

#: |sample| <= this on the int16 scale counts as silence (≈ -36 dBFS).
SILENCE_THRESHOLD = 500
#: Competition audio is expected at 16 kHz.
EXPECTED_SAMPLE_RATE = 16000
#: Flag boundaries.
VERY_SHORT_SECONDS = 0.3
VERY_LONG_SECONDS = 30.0
#: rms below this (int16 scale) is treated as digital silence.
SILENT_RMS = 1.0
#: silence_ratio above this marks the clip silent.
SILENT_RATIO = 0.95
#: Cap on per-file rows kept in memory / serialised to summary.json.
PER_FILE_CAP = 5000

_SUPPORTED_WIDTHS = (1, 2, 3, 4)
_CHUNK_FRAMES = 1 << 16


def _decode(raw: bytes, width: int) -> list[int]:
    if width == 1:
        return [b - 128 for b in raw]
    return [
        int.from_bytes(raw[i : i + width], "little", signed=True)
        for i in range(0, len(raw), width)
    ]


def _to_int16_scale(value: int, width: int) -> float:
    """Map an integer PCM sample of any supported width onto the int16 scale."""
    if width == 2:
        return float(value)
    shift = 15 - (width * 8 - 1)  # 1 byte -> x256, 3 bytes -> /256, 4 bytes -> /65536
    return value * float(2**shift)


def _empty_result(path: str | Path, error: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "ok": False,
        "error": error,
        "duration_s": None,
        "sample_rate": None,
        "channels": None,
        "sample_width_bits": None,
        "n_frames": None,
        "peak_amplitude": None,
        "rms": None,
        "silence_ratio": None,
        "silence_threshold": SILENCE_THRESHOLD,
        "clipping_ratio": None,
        "flags": ["corrupt"],
    }


def analyze_wav(path: str | Path) -> dict[str, Any]:
    """Analyze one WAV file. Never raises; problems land in ``error`` + ``flags``."""
    try:
        allowed = assert_path_allowed(path)
    except Exception as exc:  # noqa: BLE001 - prohibition is a finding, not a crash
        return _empty_result(path, f"prohibited path: {exc}")

    if not allowed.is_file():
        return _empty_result(allowed, "file not found")

    try:
        wf_handle = wave.open(str(allowed), "rb")
    except (wave.Error, EOFError, OSError) as exc:
        # wave.Error covers float/compressed WAV (formats other than PCM).
        return _empty_result(
            allowed,
            f"unsupported or corrupt WAV: {type(exc).__name__}: {exc}",
        )

    with wf_handle as wf:
        try:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
        except (wave.Error, EOFError) as exc:
            return _empty_result(allowed, f"corrupt WAV header: {exc}")

        if width not in _SUPPORTED_WIDTHS:
            return _empty_result(
                allowed,
                f"unsupported sample width {width * 8} bits (PCM 8/16/24/32 only)",
            )

        flags: list[str] = []
        if n_frames <= 0 or rate <= 0:
            result = _empty_result(allowed, "empty audio")
            result.update(
                {
                    "sample_rate": rate or None,
                    "channels": channels,
                    "sample_width_bits": width * 8,
                    "n_frames": n_frames,
                }
            )
            result["flags"] = ["empty"]
            return result

        n_values = 0
        silent_values = 0
        clipped_values = 0
        peak = 0.0
        sum_sq = 0.0
        max_pos = float(2 ** (width * 8 - 1) - 1)
        min_neg = float(-(2 ** (width * 8 - 1)))
        # int16-scale conversion factor and the equivalent raw-sample threshold.
        scale = float(2 ** (15 - (width * 8 - 1)))
        silence_raw = SILENCE_THRESHOLD / scale
        try:
            while True:
                raw = wf.readframes(_CHUNK_FRAMES)
                if not raw:
                    break
                values = _decode(raw, width)
                for v in values:
                    scaled = v * scale
                    n_values += 1
                    if -silence_raw <= v <= silence_raw:
                        silent_values += 1
                    if v >= max_pos or v <= min_neg:
                        clipped_values += 1
                    av = -scaled if scaled < 0 else scaled
                    if av > peak:
                        peak = av
                    sum_sq += scaled * scaled
        except (wave.Error, EOFError, OSError) as exc:
            return _empty_result(allowed, f"truncated/corrupt data: {exc}")

    if n_values == 0:
        result = _empty_result(allowed, "empty audio")
        result["flags"] = ["empty"]
        return result

    duration = n_frames / rate
    rms = math.sqrt(sum_sq / n_values)
    silence_ratio = silent_values / n_values
    clipping_ratio = clipped_values / n_values

    if channels <= 0 or rate <= 0:
        flags.append("corrupt")
    if rate != EXPECTED_SAMPLE_RATE:
        flags.append("unexpected_sample_rate")
    if duration < VERY_SHORT_SECONDS:
        flags.append("very_short")
    if duration > VERY_LONG_SECONDS:
        flags.append("very_long")
    if rms < SILENT_RMS or silence_ratio > SILENT_RATIO:
        flags.append("silent")
    if clipping_ratio > 0:
        flags.append("clipping")

    return {
        "path": str(allowed),
        "ok": True,
        "error": None,
        "duration_s": duration,
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bits": width * 8,
        "n_frames": n_frames,
        "peak_amplitude": peak,
        "rms": rms,
        "silence_ratio": silence_ratio,
        "silence_threshold": SILENCE_THRESHOLD,
        "clipping_ratio": clipping_ratio,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Split-level aggregation
# ---------------------------------------------------------------------------
def pearson(xs: Iterable[float], ys: Iterable[float]) -> dict[str, Any]:
    """Pearson correlation with stdlib math; ``{"available": False}`` if n < 3."""
    pairs = [
        (float(x), float(y))
        for x, y in zip(list(xs), list(ys))
        if isinstance(x, (int, float)) and isinstance(y, (int, float))
        and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    n = len(pairs)
    if n < 3:
        return {"available": False, "reason": "n < 3", "n": n}
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return {"available": False, "reason": "zero variance", "n": n}
    return {
        "available": True,
        "n": n,
        "r": sxy / math.sqrt(sxx * syy),
        "r_squared": (sxy * sxy) / (sxx * syy),
        "note": "descriptive only; correlation is not causation",
    }


def flag_counts(per_file: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in per_file:
        for flag in row.get("flags", []) or []:
            counts[flag] += 1
    return dict(sorted(counts.items()))


def _stats(values: list[float]) -> dict[str, Any]:
    from .dataset_analysis import describe  # local import keeps modules decoupled

    return describe(values)


def analyze_split_audio(
    samples: Iterable[Any],
    max_files: int | None = None,
    *,
    synthetic: bool = False,
    max_rows: int = PER_FILE_CAP,
) -> dict[str, Any]:
    """Analyze up to ``max_files`` audio files referenced by ``samples``.

    One unreadable file is recorded as a flagged row; the run continues.
    ``"synthetic"`` is a passthrough marker propagated to the report banner.
    """
    samples = list(samples or [])
    seen: set[str] = set()
    per_file: list[dict[str, Any]] = []
    durations: list[float] = []
    durations_for_corr: list[float] = []
    word_counts_for_corr: list[int] = []
    char_counts_for_corr: list[int] = []
    rates: Counter[str] = Counter()
    channels_dist: Counter[str] = Counter()
    widths: Counter[str] = Counter()
    missing_audio = 0
    truncated = False

    candidates = []
    for sample in samples:
        audio = getattr(sample, "audio_path", None)
        if audio is None:
            missing_audio += 1
            continue
        key = str(audio)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((sample, audio))

    if max_files is not None and len(candidates) > max_files:
        truncated = True
        candidates = candidates[:max_files]

    for sample, audio in candidates:
        row = analyze_wav(audio)
        row["sample_id"] = str(getattr(sample, "id", ""))
        per_file.append(row)
        if not row.get("ok"):
            continue
        duration = row.get("duration_s")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
            text = getattr(sample, "audio_text", "") or ""
            durations_for_corr.append(float(duration))
            word_counts_for_corr.append(len(text.split()) if text.strip() else 0)
            char_counts_for_corr.append(len(text))
        rates[str(row.get("sample_rate"))] += 1
        channels_dist[str(row.get("channels"))] += 1
        widths[str(row.get("sample_width_bits"))] += 1

    total_seconds = math.fsum(durations)
    by_rate = {k: v for k, v in sorted(rates.items(), key=lambda kv: -kv[1])}
    unexpected = sum(
        v for k, v in by_rate.items() if k not in (str(EXPECTED_SAMPLE_RATE),)
    )

    return {
        "synthetic": bool(synthetic),
        "n_samples": len(samples),
        "n_files_analyzed": len(per_file),
        "n_files_ok": sum(1 for r in per_file if r.get("ok")),
        "n_files_failed": sum(1 for r in per_file if not r.get("ok")),
        "n_missing_audio_refs": missing_audio,
        "truncated_by_max_files": truncated,
        "max_files": max_files,
        "duration_stats": _stats(durations),
        "total_duration_hours": total_seconds / 3600.0,
        "total_duration_seconds": total_seconds,
        "sample_rate_distribution": by_rate,
        "n_unexpected_sample_rate": unexpected,
        "channels_distribution": dict(sorted(channels_dist.items())),
        "bit_depth_distribution": dict(sorted(widths.items())),
        "flag_counts": flag_counts(per_file),
        "silence_threshold": SILENCE_THRESHOLD,
        "duration_vs_transcript_words": pearson(durations_for_corr, word_counts_for_corr),
        "duration_vs_transcript_chars": pearson(durations_for_corr, char_counts_for_corr),
        "per_file": per_file[:max_rows],
        "per_file_rows_returned": min(len(per_file), max_rows),
        "per_file_rows_total": len(per_file),
        "per_file_truncated": len(per_file) > max_rows,
    }
