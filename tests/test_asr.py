"""ASR tests: normalization, WER/CER, backends, metrics, leak guard. Stdlib only."""

from __future__ import annotations

import math
import tempfile
import wave
from pathlib import Path

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, expected, rel=1e-5, abs=1e-5):
            self.expected = expected
            self.rel = rel
            self.abs = abs
        def __eq__(self, actual):
            return math.isclose(actual, self.expected, rel_tol=self.rel, abs_tol=self.abs)
        def __repr__(self):
            return f"approx({self.expected})"
    class _Raises:
        def __init__(self, exc_type):
            self.exc_type = exc_type
            self.value = None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.exc_type}")
            if issubclass(exc_type, self.exc_type):
                self.value = exc_val
                return True
            return False
    class _PytestShim:
        @staticmethod
        def approx(expected, rel=1e-5, abs=1e-5):
            return _Approx(expected, rel=rel, abs=abs)
        @staticmethod
        def raises(exc_type):
            return _Raises(exc_type)
        @staticmethod
        def skip(reason=""):
            return None
    pytest = _PytestShim()  # type: ignore

from lifemate.asr import (
    AsrError,
    OracleTranscriptBackend,
    cer,
    create_asr_backend,
    evaluate_asr,
    fold_diacritics,
    levenshtein,
    load_audio_16k_mono,
    normalize_vi,
    normalization_report,
    resample_linear,
    save_asr_metrics,
    tokenize_words,
    unavailable_metrics,
    wer,
)
from lifemate.asr.metrics import percentile
from lifemate.asr.interface import TranscriptResult, wav_info
from lifemate.asr.normalization import for_wer, to_llm_text, int_to_vietnamese_words
from lifemate.asr.metrics import critical_token_diagnostics, format_metrics_table

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wav(path: Path, values, rate=16000, channels=1, width=2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        frames = bytearray()
        for v in values:
            for _ in range(channels):
                frames += int(v).to_bytes(width, "little", signed=True)
        wf.writeframes(bytes(frames))
    return path


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
def test_normalize_preserves_vietnamese_diacritics():
    assert normalize_vi("Đặt báo thức") == "đặt báo thức"
    assert normalize_vi("đặt báo thức") == "đặt báo thức"
    # Six meaning-bearing variants stay distinct (no accidental folding).
    assert len({normalize_vi(w) for w in ("ma", "má", "mà", "mả", "mã", "mạ")}) == 6


def test_normalize_strips_punctuation_and_collapses_whitespace():
    assert normalize_vi("  Xin CHÀO,   bạn!  ") == "xin chào bạn"
    assert normalize_vi("a\n\tb   c") == "a b c"
    # Updated for two-path split: for_wer/normalize_vi expands 20.000 to spoken form; to_llm_text preserves punctuation/digits.
    assert normalize_vi("Giá: 20.000đ?") == "giá hai mươi nghìn đ"
    assert to_llm_text("Giá: 20.000đ?") == "Giá: 20.000đ?"
    assert normalize_vi(None) == ""
    assert normalize_vi("") == ""


def test_normalize_nfc_equivalence():
    # 'e' + combining acute and precomposed 'é' must normalize identically.
    assert normalize_vi("e\u0301") == normalize_vi("é") == "é"


def test_normalize_expands_numbers_and_documented_abbreviations():
    # Updated for two-path split: for_wer converts numbers to words; to_llm_text keeps digits unchanged.
    assert normalize_vi("Đặt báo thức lúc 6 giờ 30 sáng mai") == "đặt báo thức lúc sáu giờ ba mươi sáng mai"
    assert to_llm_text("Đặt báo thức lúc 6 giờ 30 sáng mai") == "Đặt báo thức lúc 6 giờ 30 sáng mai"
    assert normalize_vi("ko dc vs anh A") == "không được với anh a"
    assert normalize_vi("5 kg gạo") == "năm ki lô gam gạo"
    # Abbreviation expansion is whole-token only: 'k' inside numbers is untouched.
    assert normalize_vi("10k") == "10k"


def test_fold_diacritics():
    assert fold_diacritics("Đặt báo thức") == "dat bao thuc"
    assert fold_diacritics("") == ""


def test_tokenize_words():
    assert tokenize_words("xin chào bạn") == ["xin", "chào", "bạn"]
    assert tokenize_words("") == []


def test_normalization_report_lists_rules():
    report = normalization_report()
    assert report["diacritics_preserved"] is True
    # Updated for two-path split: number verbalization is now active in for_wer.
    assert report["number_verbalization"] is True
    assert any("unicode_nfc" in r for r in report["rules"])
    assert len(report["rules"]) == len(set(report["rules"]))


# ---------------------------------------------------------------------------
# cer / wer with hand-checked numbers
# ---------------------------------------------------------------------------
def test_levenshtein_known_cases():
    assert levenshtein("kitten", "sitting") == 3  # k->s, e->i, +g
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("bán", "ban") == 1


def test_cer_diacritic_difference_is_counted():
    # ref "xin chào" = 8 chars, hyp "xin chao" differs at position 5 only:
    # 1 substitution / 8 reference chars = 0.125.
    assert cer("xin chào", "xin chao") == pytest.approx(0.125)
    # Normalization does NOT fold diacritics, so the normalized CER is unchanged.
    assert cer(normalize_vi("xin chào"), normalize_vi("xin chao")) == pytest.approx(0.125)
    # The ASCII-folded view (supplementary metric) hides it completely.
    assert cer(fold_diacritics("xin chào"), fold_diacritics("xin chao")) == 0.0


def test_wer_known_cases():
    # ref has 2 words, 1 wrong -> 0.5
    assert wer("xin chào", "xin chao") == pytest.approx(0.5)
    # 1 deletion of 3 words -> 1/3
    assert wer("dat bao thuc", "dat bao") == pytest.approx(1 / 3)
    # identical -> 0
    assert wer("a b c", "a b c") == 0.0
    # empty reference: 0 when hypothesis empty, 1.0 otherwise
    assert wer("", "") == 0.0
    assert wer("", "x") == 1.0
    # extra hypothesis words count as insertions: 2/2 = 1.0
    assert wer("a b", "a b c d") == pytest.approx(1.0)


def test_cer_wer_are_normalized_by_reference_length_not_hypothesis():
    # Sanity: a longer hypothesis cannot lower the rate.
    short = wer("a b c", "a b")
    long = wer("a b c", "a b c d e f")
    assert long > short


# ---------------------------------------------------------------------------
# Audio loading / resampling
# ---------------------------------------------------------------------------
def test_resample_linear_identity_and_length_ratio():
    signal = [0.0, 0.5, -0.5, 1.0, 0.0]
    assert resample_linear(signal, 16000, 16000) == signal  # identity at same rate
    assert resample_linear(signal, 16000, 16000) is not signal  # returns a copy

    down = resample_linear(signal, 32000, 16000)
    assert len(down) == round(len(signal) * 16000 / 32000)

    long_signal = [math.sin(i / 10.0) for i in range(44100)]
    up = resample_linear(long_signal, 16000, 32000)
    assert len(up) == 88200  # exact 2x length ratio
    assert resample_linear([], 8000, 16000) == []


def test_resample_preserves_ramp_endpoints():
    ramp = [float(i) for i in range(100)]
    out = resample_linear(ramp, 8000, 4000)
    assert out[0] == 0.0
    # Output positions are i * (src/dst) = 0, 2, ..., 98, so the final input
    # sample (99) is not emitted. Documented edge behaviour, not a bug.
    assert out[-1] == 98.0
    # The interpolated ramp stays monotone and evenly spaced by 2.
    assert all(b - a == pytest.approx(2.0) for a, b in zip(out, out[1:]))


def test_load_audio_mono_mixdown():
    with tempfile.TemporaryDirectory() as tmp:
        # Stereo: left = +1000, right = -1000 -> mono average 0.
        path = tmp and Path(tmp) / "stereo.wav"
        samples = [1000, -1000] * 160
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            frames = bytearray()
            for v in samples:
                frames += int(v).to_bytes(2, "little", signed=True)
            wf.writeframes(bytes(frames))
        data, rate = load_audio_16k_mono(path)
        assert rate == 16000
        assert len(data) == 160  # 320 interleaved samples -> 160 frames
        assert all(abs(v) < 1e-9 for v in data)

        # Mono 16-bit -32768 -> exactly -1.0 (no extra normalization).
        mono = _wav(Path(tmp) / "mono.wav", [-32768, 32767])
        data2, _ = load_audio_16k_mono(mono)
        assert data2[0] == -1.0
        assert data2[1] == pytest.approx(32767 / 32768)


def test_load_audio_resamples_and_truncates():
    with tempfile.TemporaryDirectory() as tmp:
        path = _wav(Path(tmp) / "8k.wav", [1000] * 8000, rate=8000)
        data, rate = load_audio_16k_mono(path)
        assert rate == 16000
        assert len(data) == 16000  # 1 s at 8 kHz -> 1 s at 16 kHz

        truncated, _ = load_audio_16k_mono(path, max_seconds=0.25)
        assert len(truncated) == 4000


def test_load_audio_rejects_unsupported_width():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "float.wav"
        # Build a 32-bit float WAV by hand (wave module refuses to write floats).
        import struct

        payload = struct.pack("<4f", 0.0, 0.1, -0.1, 0.0)
        fmt = struct.pack("<HHIIHH", 3, 1, 16000, 16000 * 4, 4, 32)  # format 3 = IEEE float
        riff = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(payload)) + b"WAVE"
        riff += b"fmt " + struct.pack("<I", len(fmt)) + fmt
        riff += b"data" + struct.pack("<I", len(payload)) + payload
        path.write_bytes(riff)
        with pytest.raises(AsrError):
            load_audio_16k_mono(path)


def test_load_audio_missing_file_raises_asr_error():
    with pytest.raises(AsrError):
        load_audio_16k_mono("/nonexistent/nope.wav")


def test_wav_info_on_fixture():
    fixture = REPO_ROOT / "tests/fixtures/synthetic_lifemate/train/audio/train_single_tool/train_single_tool.wav"
    info = wav_info(fixture)
    assert info["ok"] is True
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["duration_s"] > 0


# ---------------------------------------------------------------------------
# Oracle backend
# ---------------------------------------------------------------------------
def test_oracle_returns_exact_text():
    fixture = REPO_ROOT / "tests/fixtures/synthetic_lifemate/train/audio/train_single_tool/train_single_tool.wav"
    backend = OracleTranscriptBackend({"train_single_tool": "Dat bao thuc luc 6 gio 30 sang mai."})
    result = backend.transcribe(fixture)
    assert result.text == "Dat bao thuc luc 6 gio 30 sang mai."
    assert result.backend == "oracle"
    assert result.error is None
    assert result.ok is True
    assert result.duration_s and result.duration_s > 0
    assert backend.is_oracle is True


def test_oracle_strict_missing_raises_never_fabricates():
    backend = OracleTranscriptBackend({}, strict=True)
    with pytest.raises(AsrError) as excinfo:
        backend.transcribe("unknown_sample.wav")
    assert "strict" in str(excinfo.value)


def test_oracle_non_strict_returns_error_result_without_text():
    backend = OracleTranscriptBackend({}, strict=False)
    result = backend.transcribe("unknown_sample.wav")
    assert result.error is not None
    assert result.text == ""


def test_oracle_accepts_callable_and_resolves_by_filename():
    def lookup(key: str) -> str | None:
        return "xin chào" if key == "clip" else None

    backend = OracleTranscriptBackend(lookup)
    result = backend.transcribe("/some/dir/clip.wav")
    assert result.text == "xin chào"


def test_oracle_requires_transcripts():
    with pytest.raises(AsrError):
        OracleTranscriptBackend(None)


def test_transcript_result_error_clears_text():
    result = TranscriptResult(text="leaked", error="failed")
    assert result.text == ""
    assert result.ok is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_oracle_and_none():
    backend = create_asr_backend(None, transcripts={"a": "b"}, backend="oracle")
    assert backend.is_oracle is True
    with pytest.raises(AsrError) as excinfo:
        create_asr_backend(None, backend="none")
    err_str = str(excinfo.value).lower()
    assert "not usable" in err_str or "cannot transcribe" in err_str


def test_factory_reads_config_object():
    from lifemate.config import load_config

    cfg = load_config(overrides={"asr.backend": "oracle"})
    backend = create_asr_backend(cfg, transcripts={"a": "b"})
    assert backend.name == "oracle"


def test_default_asr_backend_is_gipformer():
    from lifemate.config import load_config

    cfg = load_config()
    assert cfg.asr.backend == "gipformer"
    assert cfg.asr.model_id == "g-group-ai-lab/gipformer-68M-rnnt"
    assert cfg.asr.revision == "318322b009200f8e77a3751fa64638e5cc85a7dc"
    assert cfg.asr.quantization == "int8"
    assert cfg.asr.runtime == "onnx"

def test_factory_oracle_without_transcripts_raises():
    with pytest.raises(AsrError) as excinfo:
        create_asr_backend(None, backend="oracle")
    assert "training/evaluation only" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Whisper backend: actionable error, no network
# ---------------------------------------------------------------------------
def test_whisper_backend_without_torch_is_actionable():
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        has_torch = False
    else:
        has_torch = True
    if has_torch:
        pytest.skip("torch installed; the missing-package path cannot be exercised")

    from lifemate.asr import WhisperBackend

    with pytest.raises(AsrError) as excinfo:
        WhisperBackend(model_name="vinai/PhoWhisper-small", device="cpu", offline=True)
    message = str(excinfo.value)
    assert "torch" in message
    assert "transformers" in message
    assert "No network access was attempted" in message
    assert "backend='oracle'" in message  # points at the stdlib alternative


def test_whisper_backend_import_is_lazy():
    """Importing the module must not import torch/transformers."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import lifemate.asr.whisper_backend as w;"
        "assert 'torch' not in sys.modules, 'torch imported eagerly';"
        "assert 'transformers' not in sys.modules, 'transformers imported eagerly';"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_evaluate_asr_available_structure():
    pairs = [
        ("s1", "xin chào", "xin chào"),  # exact after normalization
        ("s2", "đặt báo thức", "dat bao thuc"),  # all diacritics lost
        ("s3", "tắt đèn", ""),  # empty hypothesis
    ]
    result = evaluate_asr(pairs, backend_name="whisper", latencies=[0.5, 1.0, 1.5])
    assert result["available"] is True
    assert result["n"] == 3
    assert result["backend"] == "whisper"
    assert result["sentence_exact_match"] == pytest.approx(1 / 3)
    assert result["empty_hypotheses"] == 1
    assert result["raw"]["wer_mean"] > result["normalized"]["wer_mean"] or math.isclose(
        result["raw"]["wer_mean"], result["normalized"]["wer_mean"]
    )
    # Exact-match sample contributes zero error.
    assert result["per_utterance"][0]["cer_normalized"] == 0.0
    # Diacritic loss is penalised in the normalized view (no folding).
    assert result["per_utterance"][1]["cer_normalized"] > 0
    assert result["latency_s"]["p50"] == pytest.approx(1.0)
    assert result["latency_s"]["p95"] == pytest.approx(1.45)


def test_evaluate_asr_unavailable_structure():
    for empty in (None, []):
        result = evaluate_asr(empty, backend_name="none")
        assert result["available"] is False
        assert result["n"] == 0
        assert result["raw"] is None and result["normalized"] is None
        assert result["reason"]
    assert unavailable_metrics("why")["available"] is False


def test_save_asr_metrics_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nested" / "asr_metrics.json"
        data = evaluate_asr([("s1", "a", "b")], backend_name="oracle")
        written = save_asr_metrics(target, data)
        assert written == target and target.is_file()
        import json

        assert json.loads(target.read_text())["n"] == 1


# ---------------------------------------------------------------------------
# Leak guard
# ---------------------------------------------------------------------------
def test_no_gold_transcript_leak_in_inference_path():
    """Oracle ASR may only be reachable behind an explicit evaluation flag."""
    asr_dir = REPO_ROOT / "src" / "lifemate" / "asr"
    # Only the backend module, the package factory and the protocol may mention
    # the oracle at all. Other modules must not import or construct it.
    allowed = {"oracle_backend.py", "__init__.py", "interface.py"}
    # Single-quoted mentions are allowed (whisper_backend suggests 'oracle' in a
    # hint string); construction/import in a non-eval module is not.
    forbidden = ("OracleTranscriptBackend", 'backend="oracle"', "import oracle")
    oracle_mentions = []
    for path in sorted(asr_dir.glob("*.py")):
        if path.name in allowed:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(token in line for token in forbidden):
                oracle_mentions.append(f"{path.name}:{lineno}: {line.strip()}")
    assert oracle_mentions == [], oracle_mentions

    # oracle_backend.py must state its restriction.
    oracle_src = (asr_dir / "oracle_backend.py").read_text(encoding="utf-8")
    assert "training/eval ONLY" in oracle_src
    assert "MUST NOT be used for public/private test inference" in oracle_src
    assert "is_oracle = True" in oracle_src

    infer = REPO_ROOT / "scripts" / "infer.py"
    if not infer.is_file():
        pytest.skip("scripts/infer.py not present yet")
    lines = infer.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines)
    # An explicit eval-only flag must exist and gate oracle use.
    assert "--allow-oracle" in text
    assert "LABELED_SPLITS" in text

    gate_line = next(
        (i for i, line in enumerate(lines, 1) if "--allow-oracle" in line and "add_argument" in line),
        None,
    )
    assert gate_line is not None
    # The oracle backend is only constructed after the split/flag guards.
    build_lines = [
        i for i, line in enumerate(lines, 1) if 'backend="oracle"' in line
    ]
    oracle_error_line = next(
        (i for i, line in enumerate(lines, 1) if "forbidden for split" in line), None
    )
    assert build_lines, "expected the guarded oracle construction"
    assert oracle_error_line is not None
    for line_no in build_lines:
        assert line_no > oracle_error_line, "oracle backend constructed before the split guard"
        assert line_no > gate_line, "oracle backend constructed before the eval-only flag guard"


# ---------------------------------------------------------------------------
# Two-path normalization (for_wer vs to_llm_text) and diagnostic metrics
# ---------------------------------------------------------------------------

def test_two_path_divergence_table_12_inputs():
    """Exact divergence table for 12 tricky inputs between for_wer and to_llm_text."""
    # Input -> (expected for_wer, expected to_llm_text)
    expectations: dict[str, tuple[str, str]] = {
        # 1. Spoken money phrase: for_wer keeps lowercase words; to_llm_text preserves case/punctuation.
        "chuyển năm trăm nghìn": (
            "chuyển năm trăm nghìn",
            "chuyển năm trăm nghìn",
        ),
        # 2. Numeric currency: for_wer verbalizes magnitude (500.000 -> năm trăm nghìn đ); to_llm_text keeps digits.
        "500.000đ": (
            "năm trăm nghìn đ",
            "500.000đ",
        ),
        # 3. Time + date: for_wer leaves unexpandable time/date as digits; to_llm_text keeps full format.
        "15h30 ngày 20/11": (
            "15h30 ngày 20 11",
            "15h30 ngày 20/11",
        ),
        # 4. Kinship + entity name: for_wer lowercases; to_llm_text preserves capitalization for entity recognition.
        "gọi cho anh Nam": (
            "gọi cho anh nam",
            "gọi cho anh Nam",
        ),
        # 5. Device command: both preserve diacritics; for_wer lowercases.
        "bật đèn phòng khách": (
            "bật đèn phòng khách",
            "bật đèn phòng khách",
        ),
        # 6. English loanword: preserved in both paths.
        "book vé máy bay": (
            "book vé máy bay",
            "book vé máy bay",
        ),
        # 7. Vietnamese address: for_wer expands street/district numbers; to_llm_text preserves commas, case, digits.
        "Số 12 đường Lê Lợi, Quận 1": (
            "số mười hai đường lê lợi quận một",
            "Số 12 đường Lê Lợi, Quận 1",
        ),
        # 8. International phone number: for_wer strips '+' and keeps digits; to_llm_text preserves '+' and digits.
        "+84900000000": (
            "84900000000",
            "+84900000000",
        ),
        # 9. Decimal magnitude: for_wer expands '1,5' to 'một phẩy năm'; to_llm_text preserves '1,5 triệu'.
        "1,5 triệu": (
            "một phẩy năm triệu",
            "1,5 triệu",
        ),
        # 10. Number words: identical in both paths.
        "hai mươi mốt": (
            "hai mươi mốt",
            "hai mươi mốt",
        ),
        # 11. Location abbreviation: for_wer lowercases and separates; to_llm_text keeps dot-joined acronym.
        "TP.HCM": (
            "tp hcm",
            "TP.HCM",
        ),
        # 12. English loanword / casual shop command: lowercase in for_wer; exact case in to_llm_text.
        "ok shop": (
            "ok shop",
            "ok shop",
        ),
    }

    assert len(expectations) == 12
    for inp, (exp_wer, exp_llm) in expectations.items():
        actual_wer = for_wer(inp)
        actual_llm = to_llm_text(inp)
        assert actual_wer == exp_wer, f"for_wer mismatch on '{inp}': got '{actual_wer}', expected '{exp_wer}'"
        assert actual_llm == exp_llm, f"to_llm_text mismatch on '{inp}': got '{actual_llm}', expected '{exp_llm}'"


def test_to_llm_text_idempotence_and_properties():
    """to_llm_text must be idempotent, preserve digits, diacritics, and symbols."""
    inputs = [
        "  Xin chào , tôi cần chuyển 500.000đ  lúc 15h30 ngày 20/11 . ",
        "+84900000000 gọi anh Nam tại TP.HCM",
        "book phòng 1,5 triệu!",
        "Số 12 đường Lê Lợi, Quận 1",
    ]
    for inp in inputs:
        once = to_llm_text(inp)
        twice = to_llm_text(once)
        assert once == twice, f"idempotence violated: {once!r} != {twice!r}"

    # Preserves digits, case, currency
    assert to_llm_text("500.000đ") == "500.000đ"
    assert to_llm_text("anh Nam") == "anh Nam"
    assert to_llm_text("+84900000000") == "+84900000000"


def test_for_wer_fold_diacritics_only_in_wer_path():
    """Diacritics preserved by default in for_wer; fold_diacritics folds them."""
    assert for_wer("xin chào") == "xin chào"
    assert for_wer("xin chào", fold_diacritics=True) == "xin chao"
    # to_llm_text NEVER folds diacritics
    assert to_llm_text("xin chào") == "xin chào"


def test_vietnamese_number_magnitude_round_trip():
    """Round-trip magnitude assertions for canonical Vietnamese words."""
    cases = [
        (1500000, "một triệu năm trăm nghìn"),
        (500000, "năm trăm nghìn"),
        (21, "hai mươi mốt"),
        (15, "mười lăm"),
        (105, "một trăm linh năm"),
        (1000000000, "một tỷ"),
    ]
    for val, expected_words in cases:
        actual_words = int_to_vietnamese_words(val)
        assert actual_words == expected_words, f"{val} -> {actual_words} != {expected_words}"
        # Assert for_wer expands formatted or plain number to the canonical words
        if val >= 1000:
            formatted = f"{val:,}".replace(",", ".")
            assert expected_words in for_wer(formatted)
        else:
            assert expected_words == for_wer(str(val))


def test_critical_token_diagnostics():
    """Test critical_token_diagnostics across all token classes and magnitude errors."""
    # Empty pairs -> available: False
    assert critical_token_diagnostics([])["available"] is False
    assert critical_token_diagnostics(None)["available"] is False

    # Case a: magnitude error ("chuyển năm trăm nghìn" vs "chuyển năm triệu") -> >= 1
    mag_pairs = [("a", "chuyển năm trăm nghìn", "chuyển năm triệu")]
    diag_mag = critical_token_diagnostics(mag_pairs)
    assert diag_mag["available"] is True
    assert diag_mag["classes"]["numbers"]["magnitude_errors"] >= 1
    assert diag_mag["classes"]["money"]["magnitude_errors"] >= 1

    # Case b: identical numbers -> 0
    ok_num_pairs = [("b", "chuyển năm trăm nghìn", "chuyển năm trăm nghìn")]
    diag_ok = critical_token_diagnostics(ok_num_pairs)
    assert diag_ok["classes"]["numbers"]["magnitude_errors"] == 0
    assert diag_ok["classes"]["money"]["magnitude_errors"] == 0

    # Case c: dotted magnitude change ("500.000đ" vs "5.000.000đ") -> >= 1
    c_pairs = [("c", "mua 2 cân gạo giá 500.000đ", "mua 2 cân gạo giá 5.000.000đ")]
    diag_c = critical_token_diagnostics(c_pairs)
    assert diag_c["classes"]["numbers"]["magnitude_errors"] >= 1
    assert diag_c["classes"]["money"]["magnitude_errors"] >= 1

    # Case d: small count mismatch ("gọi 2 cuộc" vs "gọi 3 cuộc") -> 0 magnitude errors
    d_pairs = [("d", "gọi 2 cuộc", "gọi 3 cuộc")]
    diag_d = critical_token_diagnostics(d_pairs)
    assert diag_d["classes"]["numbers"]["magnitude_errors"] == 0
    assert diag_d["classes"]["money"]["magnitude_errors"] == 0

    # Case multi: multiple magnitudes in one sentence ("5 triệu và 500 nghìn" vs "5 triệu và 5 triệu")
    multi_pairs = [("m", "mua 5 triệu với 500 nghìn", "mua 5 triệu với 5 triệu")]
    diag_multi = critical_token_diagnostics(multi_pairs)
    assert diag_multi["classes"]["numbers"]["magnitude_errors"] >= 1

    # Case 3: money case
    money_pairs = [("u3", "giá 20.000đ", "giá 20.000đ")]
    diag_money = critical_token_diagnostics(money_pairs)
    assert diag_money["classes"]["money"]["n_ref"] >= 1
    assert diag_money["classes"]["money"]["exact_set_match_rate"] == 1.0

    # Case 4: time case
    time_pairs = [("u4", "hẹn lúc 15h30", "hẹn lúc 15h30")]
    diag_time = critical_token_diagnostics(time_pairs)
    assert diag_time["classes"]["times"]["n_ref"] >= 1
    assert diag_time["classes"]["times"]["exact_set_match_rate"] == 1.0

    # Case 5: entity with diacritics
    entity_pairs = [("u5", "gặp anh Nam tại Hà Nội", "gặp anh Nam tại Hà Nội")]
    diag_entity = critical_token_diagnostics(entity_pairs)
    assert diag_entity["classes"]["entities"]["n_ref"] >= 1
    assert diag_entity["classes"]["entities"]["exact_set_match_rate"] == 1.0

    # Case 6: English loanwords
    loan_pairs = [("u6", "book vé máy bay trên app", "book vé máy bay trên app")]
    diag_loan = critical_token_diagnostics(loan_pairs)
    assert diag_loan["classes"]["english_loanwords"]["n_ref"] >= 1
    assert "book" in diag_loan["classes"]["english_loanwords"] or diag_loan["classes"]["english_loanwords"]["n_ref"] >= 1

    # Format table produces non-empty string
    table_str = format_metrics_table(diag_mag)
    assert "Critical Token Diagnostics" in table_str
    assert "| Class |" in table_str


def test_static_regression_no_forbidden_cross_calls():
    """Static AST/text scan: to_llm_text/for_wer must not be improperly imported or called."""
    forbidden_dirs = [
        REPO_ROOT / "src" / "lifemate" / "agent",
        REPO_ROOT / "src" / "lifemate" / "training",
        REPO_ROOT / "src" / "lifemate" / "data",
    ]
    forbidden_scripts = [
        REPO_ROOT / "scripts" / "prepare_dataset.py",
        REPO_ROOT / "scripts" / "train.py",
    ]

    for fdir in forbidden_dirs:
        if fdir.is_dir():
            for pyfile in fdir.rglob("*.py"):
                content = pyfile.read_text(encoding="utf-8")
                assert "to_llm_text" not in content, f"to_llm_text found in {pyfile}"

    for sfile in forbidden_scripts:
        if sfile.is_file():
            content = sfile.read_text(encoding="utf-8")
            assert "to_llm_text" not in content, f"to_llm_text found in {sfile}"

    # In scripts/infer.py, for_wer must not appear in transcript construction
    infer_file = REPO_ROOT / "scripts" / "infer.py"
    if infer_file.is_file():
        content = infer_file.read_text(encoding="utf-8")
        assert "for_wer" not in content, "for_wer found in scripts/infer.py"

def test_default_decoding_method_is_modified_beam_search():
    """Published WER was produced with modified_beam_search; both configs agree."""
    from lifemate.config import ASR_DECODINGS, ConfigError, load_config

    assert load_config().asr.decoding_method == "modified_beam_search"
    for name in ("base.yaml", "inference.yaml"):
        path = REPO_ROOT / "configs" / name
        assert load_config(path).asr.decoding_method == "modified_beam_search", name
    assert set(ASR_DECODINGS) == {"greedy_search", "modified_beam_search"}
    with pytest.raises(ConfigError) as excinfo:
        load_config(None, overrides={"asr.decoding_method": "bogus"})
    assert "modified_beam_search" in str(excinfo.value)


def test_protocol_signature_matches_backend_implementations():
    """ASRBackend.transcribe must declare what every caller and backend uses."""
    import inspect

    from lifemate.asr import GipformerBackend, WhisperBackend
    from lifemate.asr.interface import ASRBackend

    protocol_params = inspect.signature(ASRBackend.transcribe).parameters
    assert "sample_id" in protocol_params and protocol_params["sample_id"].default is None
    assert "lenient" in protocol_params and protocol_params["lenient"].default is False
    assert protocol_params["sample_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert protocol_params["lenient"].kind is inspect.Parameter.KEYWORD_ONLY

    # Every shipped backend accepts the protocol's call shape, so a backend
    # written against the Protocol cannot TypeError at the call site.
    for backend_cls in (GipformerBackend, WhisperBackend, OracleTranscriptBackend):
        call = inspect.signature(backend_cls.transcribe).bind
        call(object(), "a.wav", sample_id="x")  # the shape scripts/infer.py uses
        if backend_cls is not OracleTranscriptBackend:
            call(object(), "a.wav", sample_id="x", lenient=True)

def test_magnitude_diagnostics_recognise_ngan_spelling():
    """Regression: 'ngàn' (southern spelling, present in real train transcripts)
    was absent from the magnitude/money patterns, so 'năm trăm ngàn' produced no
    scale at all and a true 'ngàn' -> 'triệu' drift scored 0 magnitude errors.
    """
    from lifemate.asr.metrics import _extract_phrase_magnitudes, critical_token_diagnostics

    assert _extract_phrase_magnitudes("năm trăm ngàn") == [10**3]
    assert _extract_phrase_magnitudes("hai triệu tám") == [10**6]

    drift = critical_token_diagnostics([("a", "chuyển hai trăm ngàn", "chuyển hai trăm triệu")])
    assert drift["classes"]["numbers"]["magnitude_errors"] >= 1

    same = critical_token_diagnostics([("b", "chuyển hai trăm ngàn", "chuyển hai trăm ngàn")])
    assert same["classes"]["numbers"]["magnitude_errors"] == 0
    assert same["classes"]["money"]["n_ref"] >= 1

def test_raw_metric_view_is_case_insensitive():
    """Regression: Gipformer emits ALL-UPPERCASE, so a case-sensitive raw view
    scored CER 0.7848 on a 0.1113-CER system — 100% of that gap was case.
    The raw diagnostic must stay case-insensitive (still unnormalised otherwise).
    """
    from lifemate.asr.metrics import evaluate_asr

    pairs = [("a", "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng",
              "NẾU MAI MƯA THÌ GIỜI CHẠY BỘ MUỘN MỘT CHUYẾN")]
    result = evaluate_asr(pairs, "gipformer")
    # Only real word errors remain: "dời"->"giời", "1"->"một", "tiếng"->"chuyến".
    assert result["raw"]["cer_mean"] <= 0.25, result["raw"]
    assert result["raw"]["wer_mean"] < 0.60, result["raw"]

    identical = evaluate_asr([("b", "Xin chào", "XIN CHÀO")], "gipformer")
    assert identical["raw"]["cer_mean"] == 0.0
    assert identical["raw"]["wer_mean"] == 0.0
    # Raw stays raw: punctuation and digits are NOT normalised away.
    punct = evaluate_asr([("c", "giá 500.000đ, nhé!", "giá 500.000đ nhé")], "gipformer")
    assert punct["raw"]["cer_mean"] > 0.0

def test_transcribe_split_scores_the_agent_facing_text():
    """Regression: transcribe_split read `raw_text` (verbatim recognizer string)
    for BOTH fields, so a scored hypothesis could differ from what the agent sees.
    scripts/infer.py::_transcribe uses `text`; the scorer must match it.
    """
    import json

    import lifemate.asr as asr_pkg
    from lifemate.asr.interface import TranscriptResult

    class _Stub:
        name = "stub"
        is_oracle = False
        quantization = "int8"

        def transcribe(self, audio_path, *, sample_id=None, lenient=False):
            return TranscriptResult(
                text="nếu mai mưa thì thôi",            # agent-facing
                raw_text="NẾU MAI MƯA THÌ THÔI",        # recognizer verbatim
                backend="stub",
                latency_s=0.01,
                duration_s=1.0,
            )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ts_under_test", REPO_ROOT / "scripts" / "transcribe_split.py"
    )
    transcribe_split = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transcribe_split)  # type: ignore[union-attr]

    FIXTURES = REPO_ROOT / "tests" / "fixtures" / "synthetic_lifemate"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "preds.jsonl"
        original = asr_pkg.create_asr_backend
        asr_pkg.create_asr_backend = lambda cfg, backend=None, **kw: _Stub()
        try:
            rc = transcribe_split.main([
                "--split", "train", "--data-root", str(FIXTURES),
                "--output", str(out), "--overwrite",
            ])
        finally:
            asr_pkg.create_asr_backend = original
        assert rc == 0
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows and all(r["hypothesis"] == "nếu mai mưa thì thôi" for r in rows)
    assert all(r["raw_hypothesis"] == "NẾU MAI MƯA THÌ THÔI" for r in rows)
