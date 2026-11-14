"""Audio-op tests: VAD trim and RMS levelling on plain float lists. Stdlib only.

Fixtures are synthetic tones/silence generated with ``math`` so this module runs
under ``python3`` (no numpy, no torch, no silero).

Run with: python3 tests/run_tests.py test_audio_ops.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lifemate.asr.audio_ops import AudioResult, normalize_rms, trim_silence  # noqa: E402

REQUIRED_STATS = (
    "rms_dbfs_before",
    "rms_dbfs_after",
    "peak_before",
    "peak_after",
    "n_samples_in",
    "n_samples_out",
    "gain_db",
)

def _tone(sample_rate: int, seconds: float, amplitude: float, freq: float = 440.0) -> list[float]:
    n = int(round(sample_rate * seconds))
    return [amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate) for i in range(n)]

def _silence(sample_rate: int, seconds: float) -> list[float]:
    return [0.0] * int(round(sample_rate * seconds))

def _padded(sample_rate: int = 16000, front: float = 0.5, body: float = 0.5, back: float = 0.5):
    """Half a second of silence, half a second of tone, half a second of silence."""
    return (
        _silence(sample_rate, front)
        + _tone(sample_rate, body, 0.3)
        + _silence(sample_rate, back)
    )

def _assert_stats(result: AudioResult) -> None:
    for key in REQUIRED_STATS:
        assert key in result.stats, f"stats missing {key!r}: {sorted(result.stats)}"

def _assert_disabled_is_identity(func, samples: list[float], **kwargs) -> None:
    result = func(samples, **kwargs)
    assert result.enabled is False
    assert result.changed is False
    assert result.samples is samples, "a disabled call must return the identical object"
    assert result.stats["n_samples_in"] == len(samples)
    assert result.stats["n_samples_out"] == len(samples)
    _assert_stats(result)

# ------------------------------------------------------------------- VAD trim

def test_trim_removes_edges_and_keeps_padding():
    sample_rate = 16000
    samples = _padded(sample_rate)
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True)
    _assert_stats(result)
    assert result.enabled is True
    assert result.changed is True
    kept_seconds = len(result.samples) / sample_rate
    assert 0.88 <= kept_seconds <= 0.96, kept_seconds  # 0.5 s tone + 2 x 0.2 s padding
    assert len(result.samples) < len(samples)
    assert result.stats["n_samples_in"] == len(samples)
    assert result.stats["n_samples_out"] == len(result.samples)
    # The tone itself survives untouched: nothing is rescaled here.
    assert max(abs(x) for x in result.samples) > 0.29
    assert result.stats["peak_after"] == result.stats["peak_before"]

def test_trim_keeps_interior_pauses():
    sample_rate = 16000
    samples = (
        _silence(sample_rate, 0.5)
        + _tone(sample_rate, 0.3, 0.3)
        + _silence(sample_rate, 0.6)
        + _tone(sample_rate, 0.3, 0.3)
        + _silence(sample_rate, 0.5)
    )
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True, padding_ms=0)
    assert result.changed is True
    kept_seconds = len(result.samples) / sample_rate
    assert 1.15 <= kept_seconds <= 1.30, kept_seconds  # tone + pause + tone
    # The 0.6 s hole is still there: count near-zero runs in the middle.
    middle = result.samples[len(result.samples) // 2 - 100 : len(result.samples) // 2 + 100]
    assert max(abs(x) for x in middle) < 0.01, "the interior pause was removed"

def test_trim_silence_only_returns_input_unchanged():
    sample_rate = 16000
    samples = _silence(sample_rate, 1.0)
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True)
    _assert_stats(result)
    assert result.samples is samples, "never return empty audio"
    assert result.changed is False
    assert result.stats["all_silent"] is True
    assert result.stats["n_samples_out"] == sample_rate
    _assert_disabled_is_identity(trim_silence, samples, sample_rate=sample_rate)

def test_trim_short_utterance_is_kept():
    """A surviving span under ~100 ms is kept as-is instead of micro-trimmed."""
    sample_rate = 16000
    samples = _silence(sample_rate, 0.2) + _tone(sample_rate, 0.04, 0.3) + _silence(sample_rate, 0.2)
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True, padding_ms=0)
    _assert_stats(result)
    assert result.changed is False, result.stats
    assert result.stats["n_samples_out"] == len(samples)
    _assert_disabled_is_identity(trim_silence, samples, sample_rate=sample_rate, padding_ms=0)

def test_trim_respects_short_min_silence_run():
    """With ``min_silence_ms`` above the real silence, nothing is trimmed."""
    sample_rate = 16000
    samples = _silence(sample_rate, 0.1) + _tone(sample_rate, 0.5, 0.3) + _silence(sample_rate, 0.1)
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True, min_silence_ms=300)
    assert result.changed is False
    assert result.stats["n_samples_out"] == len(samples)

def test_trim_at_8k():
    sample_rate = 8000
    samples = _padded(sample_rate)
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True)
    _assert_stats(result)
    assert result.changed is True
    kept_seconds = len(result.samples) / sample_rate
    assert 0.88 <= kept_seconds <= 0.96, kept_seconds
    assert result.stats["n_samples_in"] == len(samples)
    _assert_disabled_is_identity(trim_silence, samples, sample_rate=sample_rate)

def test_trim_at_44k1():
    sample_rate = 44100
    samples = _padded(sample_rate)
    result = trim_silence(samples, sample_rate=sample_rate, enabled=True)
    _assert_stats(result)
    assert result.changed is True
    kept_seconds = len(result.samples) / sample_rate
    assert 0.88 <= kept_seconds <= 0.96, kept_seconds
    assert result.stats["n_samples_in"] == len(samples) == 66150
    _assert_disabled_is_identity(trim_silence, samples, sample_rate=sample_rate)

def test_trim_stereo_interleaved_stays_aligned():
    """Stereo is a FLAT interleaved buffer: never mixed down, always aligned.

    ``channels=2`` keeps the trimmed span on an interleaved boundary so the
    caller's L/R pairing survives; the default ``channels=1`` still trims but may
    land mid-pair, which is why callers pass the count they know.
    """
    sample_rate = 16000
    mono = _padded(sample_rate)
    interleaved = [value for value in mono for _ in (0, 1)]
    assert len(interleaved) == 2 * len(mono)

    result = trim_silence(interleaved, sample_rate=sample_rate, enabled=True, channels=2)
    _assert_stats(result)
    assert result.changed is True
    assert result.stats["n_samples_out"] % 2 == 0
    assert result.stats["trim_start"] % 2 == 0 and result.stats["trim_end"] % 2 == 0
    assert result.stats["n_samples_in"] == len(interleaved)
    # No mixdown: consecutive pairs still carry the same sample value.
    assert result.samples[0] == result.samples[1]
    assert result.samples[100] == result.samples[101]
    _assert_disabled_is_identity(trim_silence, interleaved, sample_rate=sample_rate, channels=2)

def test_trim_empty_list():
    result = trim_silence([], enabled=True)
    _assert_stats(result)
    assert result.samples == []
    assert result.changed is False
    assert result.stats["all_silent"] is True
    assert result.stats["n_samples_in"] == 0
    _assert_disabled_is_identity(trim_silence, [])

def test_trim_one_sample():
    samples = [0.5]
    result = trim_silence(samples, enabled=True)
    _assert_stats(result)
    assert result.changed is False
    assert result.stats["n_samples_out"] == 1
    _assert_disabled_is_identity(trim_silence, samples)

# ---------------------------------------------------------------- RMS levelling

def test_normalize_rms_quiet_input_reaches_target():
    samples = _tone(16000, 0.5, 0.01)
    result = normalize_rms(samples, enabled=True, target_dbfs=-23.0)
    _assert_stats(result)
    assert result.changed is True
    assert abs(result.stats["rms_dbfs_after"] - (-23.0)) < 1e-6, result.stats
    assert result.stats["gain_db"] > 0
    assert result.stats["n_samples_out"] == len(samples)
    _assert_disabled_is_identity(normalize_rms, samples)

def test_normalize_rms_loud_input_is_attenuated():
    samples = _tone(16000, 0.5, 0.95)
    result = normalize_rms(samples, enabled=True)
    _assert_stats(result)
    assert result.changed is True
    assert result.stats["gain_db"] < 0
    assert abs(result.stats["rms_dbfs_after"] - (-23.0)) < 1e-6
    assert result.stats["peak_after"] <= 0.99
    _assert_disabled_is_identity(normalize_rms, samples)

def test_normalize_rms_clipping_is_limited_not_clipped():
    """A peaky buffer cannot reach the target without clipping: a smaller gain wins."""
    samples = [0.001] * 15999 + [0.98]  # low RMS, near-full-scale peak
    result = normalize_rms(samples, enabled=True, target_dbfs=-23.0, peak_ceiling=0.99)
    _assert_stats(result)
    assert result.stats["limited"] is True
    assert result.stats["peak_after"] <= 0.99
    assert result.stats["gain_db"] < (-23.0 - result.stats["rms_dbfs_before"])
    assert max(abs(x) for x in result.samples) <= 0.99
    _assert_disabled_is_identity(normalize_rms, samples)

def test_normalize_rms_silence_is_never_nan():
    samples = [0.0] * 160
    result = normalize_rms(samples, enabled=True)
    _assert_stats(result)
    assert result.samples is samples
    assert result.changed is False
    assert result.stats["gain_db"] == 0.0
    assert math.isinf(result.stats["rms_dbfs_before"])
    assert not any(math.isnan(x) for x in result.samples)
    _assert_disabled_is_identity(normalize_rms, samples)

def test_normalize_rms_one_sample_and_empty():
    one = [0.5]
    result = normalize_rms(one, enabled=True)
    _assert_stats(result)
    assert result.stats["n_samples_out"] == 1
    assert result.changed is True
    # One sample: RMS == peak == 0.5, so reaching -23 dBFS is still under the
    # ceiling and no limiting kicks in.
    assert abs(result.stats["rms_dbfs_after"] - (-23.0)) < 1e-9, result.stats
    assert abs(abs(result.samples[0]) - 0.07079457843841379) < 1e-12, result.samples
    assert result.stats["limited"] is False
    _assert_disabled_is_identity(normalize_rms, one)

    empty: list[float] = []
    empty_result = normalize_rms(empty, enabled=True)
    _assert_stats(empty_result)
    assert empty_result.samples == []
    assert empty_result.changed is False
    _assert_disabled_is_identity(normalize_rms, empty)

def test_normalize_rms_varies_with_sample_rate_only_by_fixture_length():
    """No resampling: the output length equals the input length at any rate."""
    for sample_rate in (8000, 44100):
        samples = _tone(sample_rate, 0.5, 0.02)
        result = normalize_rms(samples, enabled=True)
        _assert_stats(result)
        assert result.stats["n_samples_out"] == len(samples) == sample_rate // 2
        assert abs(result.stats["rms_dbfs_after"] - (-23.0)) < 1e-6
        _assert_disabled_is_identity(normalize_rms, samples)

# ------------------------------------------------------------------- contract

def test_results_are_plain_lists_of_floats():
    samples = _padded()
    trimmed = trim_silence(samples, enabled=True)
    levelled = normalize_rms(samples, enabled=True)
    for result in (trimmed, levelled):
        assert isinstance(result.samples, list)
        assert result.samples and all(isinstance(x, float) for x in result.samples)
        assert isinstance(result.stats, dict)

def test_disabled_does_not_copy_and_stats_are_zero_gain():
    samples = _padded()
    for func in (trim_silence, normalize_rms):
        result = func(samples)
        assert result.samples is samples
        assert result.stats["gain_db"] == 0.0
        assert result.stats["rms_dbfs_before"] == result.stats["rms_dbfs_after"]
