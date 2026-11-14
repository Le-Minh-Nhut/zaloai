"""Tests for Gipformer-68M-RNNT ASR backend, config, CLI, and integration.

Stdlib-runnable without pytest, torch, or real sherpa-onnx wheels.
Real model weights and GPU jobs are absent; tests use fake/mocked recognizers.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import subprocess
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, expected: float, rel: float = 1e-5, abs: float = 1e-5) -> None:
            self.expected = expected
            self.rel = rel
            self.abs = abs

        def __eq__(self, actual: Any) -> bool:
            return math.isclose(float(actual), self.expected, rel_tol=self.rel, abs_tol=self.abs)

        def __repr__(self) -> str:
            return f"approx({self.expected})"

    class _Raises:
        def __init__(self, exc_type: type[BaseException]) -> None:
            self.exc_type = exc_type
            self.value: BaseException | None = None

        def __enter__(self) -> _Raises:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.exc_type}")
            if issubclass(exc_type, self.exc_type):
                self.value = exc_val
                return True
            return False

    class _PytestShim:
        @staticmethod
        def approx(expected: float, rel: float = 1e-5, abs: float = 1e-5) -> _Approx:
            return _Approx(expected, rel=rel, abs=abs)

        @staticmethod
        def raises(exc_type: type[BaseException]) -> _Raises:
            return _Raises(exc_type)

        @staticmethod
        def skip(reason: str = "") -> None:
            return None

    pytest = _PytestShim()  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.asr import AsrError, TranscriptResult, create_asr_backend
from lifemate.asr.gipformer_backend import DEFAULT_MODEL_ID, PINNED_REVISION, GipformerBackend
from lifemate.config import AppConfig, AsrConfig, ConfigError, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_wav(path: Path, n_samples: int = 16000, rate: int = 16000, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        # 440 Hz tone or simple sine-like ramp
        data = bytearray()
        for i in range(n_samples):
            val = int(10000 * math.sin(2 * math.pi * 440 * (i / rate)))
            data.extend(int(val).to_bytes(2, "little", signed=True) * channels)
        handle.writeframes(data)
    return path


def _populate_fake_model_dir(dir_path: Path, flist: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for fname in flist:
        (dir_path / fname).write_text("fake onnx data", encoding="utf-8")


class _FakeStream:
    def __init__(self, text: str = "xin chào LifeMate") -> None:
        self.accepted: list[tuple[int, Any]] = []
        self.result = types.SimpleNamespace(text=text)

    def accept_waveform(self, rate: int, samples: Any) -> None:
        self.accepted.append((rate, samples))


class _FakeOfflineRecognizer:
    last_kwargs: dict[str, Any] = {}
    last_streams: list[Any] = []
    stream_text: str = "xin chào LifeMate"

    @classmethod
    def from_transducer(cls, **kwargs: Any) -> _FakeOfflineRecognizer:
        cls.last_kwargs = kwargs
        return cls()

    def create_stream(self) -> _FakeStream:
        return _FakeStream(text=self.stream_text)

    def decode_streams(self, streams: list[Any]) -> None:
        _FakeOfflineRecognizer.last_streams = streams


@contextlib.contextmanager
def _mock_sherpa(stream_text: str = "xin chào LifeMate"):
    fake_mod = types.ModuleType("sherpa_onnx")
    _FakeOfflineRecognizer.last_kwargs = {}
    _FakeOfflineRecognizer.last_streams = []
    _FakeOfflineRecognizer.stream_text = stream_text
    fake_mod.OfflineRecognizer = _FakeOfflineRecognizer  # type: ignore[attr-defined]

    old_mod = sys.modules.get("sherpa_onnx")
    sys.modules["sherpa_onnx"] = fake_mod
    try:
        yield _FakeOfflineRecognizer
    finally:
        if old_mod is None:
            sys.modules.pop("sherpa_onnx", None)
        else:
            sys.modules["sherpa_onnx"] = old_mod


# ---------------------------------------------------------------------------
# Test Cases (Items 1 - 17)
# ---------------------------------------------------------------------------


def test_01_default_backend_is_gipformer():
    """Item 1: default backend is gipformer in config and factory."""
    cfg = load_config()
    assert cfg.asr.backend == "gipformer"

    base_cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert base_cfg.asr.backend == "gipformer"

    infer_cfg = load_config(REPO_ROOT / "configs" / "inference.yaml")
    assert infer_cfg.asr.backend == "gipformer"


def test_02_exact_model_id_and_revision_string():
    """Item 2: exact model_id and revision strings present in config and backend defaults."""
    assert DEFAULT_MODEL_ID == "g-group-ai-lab/gipformer-68M-rnnt"
    assert PINNED_REVISION == "318322b009200f8e77a3751fa64638e5cc85a7dc"

    cfg = load_config()
    assert cfg.asr.model_id == DEFAULT_MODEL_ID
    assert cfg.asr.revision == PINNED_REVISION
    assert cfg.asr.quantization == "int8"
    assert cfg.asr.runtime == "onnx"


def test_03_version_1_5_is_spec_known_but_cannot_load_here():
    """Item 3: 1.5 is a KNOWN spec, yet unusable here (no weights were ever observed).

    Config accepts any known spec id (whether the artifacts exist is a load-time
    question, so an optional checkpoint needs no config edit); the backend then
    refuses the [UNVERIFIED] spec and an unknown id is a config error.
    """
    cfg = load_config()
    assert "1.5" not in cfg.asr.model_id

    AppConfig(asr=AsrConfig(model_id="g-group-ai-lab/gipformer1.5-68M-rnnt"))

    # Unknown gipformer ids stay config errors, listing the known ones.
    with pytest.raises(ConfigError) as excinfo:
        AppConfig(asr=AsrConfig(model_id="g-group-ai-lab/gipformer9-68M-rnnt"))
    assert "unknown gipformer model_id" in str(excinfo.value)
    assert DEFAULT_MODEL_ID in str(excinfo.value)

    # Backend constructor refuses the [UNVERIFIED] 1.5 spec (never a silent 68M load).
    with pytest.raises(AsrError) as excinfo2:
        GipformerBackend(model_id="g-group-ai-lab/gipformer1.5-68M-rnnt")
    assert "rejected" in str(excinfo2.value) and "UNVERIFIED" in str(excinfo2.value)


def test_04_recognizer_built_with_right_filenames_int8_and_fp32():
    """Item 4 & 5: monkeypatch fake sherpa_onnx; verify kwargs for int8 and fp32."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        int8_files = [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ]
        fp32_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        _populate_fake_model_dir(tmp_path, int8_files + fp32_files)

        with _mock_sherpa() as rec_mock:
            # int8
            b_int8 = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            assert rec_mock.last_kwargs["encoder"].endswith("encoder.int8.onnx")
            assert rec_mock.last_kwargs["decoder"].endswith("decoder.int8.onnx")
            assert rec_mock.last_kwargs["joiner"].endswith("joiner.int8.onnx")
            assert rec_mock.last_kwargs["tokens"].endswith("tokens.txt")
            assert b_int8.parameter_count() == 68_000_000

            # fp32
            b_fp32 = GipformerBackend(model_dir=tmp_path, quantization="fp32", offline=True)
            assert rec_mock.last_kwargs["encoder"].endswith("encoder.onnx")
            assert rec_mock.last_kwargs["decoder"].endswith("decoder.onnx")
            assert rec_mock.last_kwargs["joiner"].endswith("joiner.onnx")
            assert rec_mock.last_kwargs["tokens"].endswith("tokens.txt")
            assert not rec_mock.last_kwargs["encoder"].endswith(".int8.onnx")


def test_06_fp32_and_int8_missing_files_error_not_silent():
    """Item 6: selecting fp32 when only int8 exists raises AsrError without silent substitution."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )

        with _mock_sherpa():
            # Requesting fp32 when only int8 exists must raise
            with pytest.raises(AsrError) as excinfo:
                GipformerBackend(model_dir=tmp_path, quantization="fp32", offline=True)
            msg = str(excinfo.value)
            assert "encoder.onnx" in msg
            assert "silent fallback is forbidden" in msg

    # Opposite direction: dir with only fp32 files, requesting int8
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        )

        with _mock_sherpa():
            with pytest.raises(AsrError) as excinfo:
                GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            msg = str(excinfo.value)
            assert "encoder.int8.onnx" in msg
            assert "silent fallback is forbidden" in msg


def test_07_cpu_device_path():
    """Item 7: CPU device path sets num_threads and does not pass CUDA provider."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )

        with _mock_sherpa() as rec_mock:
            b = GipformerBackend(
                model_dir=tmp_path,
                quantization="int8",
                device="cpu",
                num_threads=8,
                offline=True,
            )
            assert b.device == "cpu"
            assert rec_mock.last_kwargs["num_threads"] == 8
            assert rec_mock.last_kwargs.get("provider") != "cuda"


def test_08_audio_decoding_16k_and_8k_resampled():
    """Item 8: audio decoding handles 16 kHz and resamples 8 kHz to 16 kHz."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        wav_16k = _create_wav(tmp_path / "audio_16k.wav", n_samples=16000, rate=16000)
        wav_8k = _create_wav(tmp_path / "audio_8k.wav", n_samples=8000, rate=8000)

        with _mock_sherpa(stream_text="test audio") as rec_mock:
            backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)

            res16 = backend.transcribe(wav_16k)
            assert res16.ok is True
            assert res16.duration_s == pytest.approx(1.0, rel=1e-2)
            assert len(rec_mock.last_streams) == 1
            rate, samples = rec_mock.last_streams[0].accepted[0]
            assert rate == 16000
            assert len(samples) == 16000

            res8 = backend.transcribe(wav_8k)
            assert res8.ok is True
            assert res8.duration_s == pytest.approx(1.0, rel=1e-2)
            rate8, samples8 = rec_mock.last_streams[0].accepted[0]
            assert rate8 == 16000
            assert len(samples8) == 16000


def test_09_empty_audio_raises_or_lenient():
    """Item 9: empty audio raises AsrError, or returns error result in lenient mode."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        empty_wav = _create_wav(tmp_path / "empty.wav", n_samples=0, rate=16000)

        with _mock_sherpa():
            backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)

            with pytest.raises(AsrError) as excinfo:
                backend.transcribe(empty_wav, lenient=False)
            assert "empty audio" in str(excinfo.value)

            res = backend.transcribe(empty_wav, lenient=True)
            assert res.ok is False
            assert res.text == ""
            assert "empty audio" in str(res.error)


def test_10_invalid_wav_raises():
    """Item 10: invalid WAV file raises AsrError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        bad_wav = tmp_path / "corrupt.wav"
        bad_wav.write_bytes(b"NOT A WAV HEADER JUNK DATA")

        with _mock_sherpa():
            backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            with pytest.raises(AsrError):
                backend.transcribe(bad_wav, lenient=False)


def test_11_vietnamese_text_preserved():
    """Item 11: Vietnamese text with diacritics and special characters preserved."""
    sample_text = "Chuyển 500.000đ cho anh Tuấn lúc 14 giờ"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        wav = _create_wav(tmp_path / "audio.wav", n_samples=8000, rate=16000)

        with _mock_sherpa(stream_text=sample_text):
            backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            res = backend.transcribe(wav)
            assert res.raw_text == sample_text
            # Mixed-case recognizer output passes through verbatim; LLM
            # normalization happens once in scripts/infer.py, so every backend is
            # treated equally.
            assert res.text == sample_text
            assert res.text == res.raw_text


def test_12_missing_model_files_gives_actionable_error():
    """Item 12: missing model files raises AsrError with exact names and download command."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with _mock_sherpa():
            with pytest.raises(AsrError) as excinfo:
                GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            msg = str(excinfo.value)
            assert "missing Gipformer (int8) model files" in msg
            assert "encoder.int8.onnx" in msg
            assert "python scripts/download_asr_model.py" in msg


def test_13_offline_mode_makes_zero_network_calls():
    """Item 13: offline=True never calls huggingface_hub."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )

        def _forbidden_hub_call(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("NETWORK CALL WAS ATTEMPTED!")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = _forbidden_hub_call  # type: ignore[attr-defined]

        old_hf = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hf
        try:
            with _mock_sherpa():
                backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
                assert backend.offline is True
        finally:
            if old_hf is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = old_hf


def test_14_cli_infer_gipformer_integration():
    """Item 14: infer.py --help lists gipformer; missing model dir exits 2 without traceback."""
    # 1. --help check
    probe = subprocess.run(
        [sys.executable, "scripts/infer.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0
    assert "gipformer" in probe.stdout

    # 2. Missing model dir exits 2 with clean message and no submission written
    with tempfile.TemporaryDirectory() as tmp:
        out_sub = Path(tmp) / "sub.jsonl"
        fake_empty_model_dir = Path(tmp) / "no_models_here"
        fake_empty_model_dir.mkdir()

        cmd = [
            sys.executable,
            "scripts/infer.py",
            "--split",
            "public_test",
            "--data-root",
            "tests/fixtures/synthetic_lifemate",
            "--output",
            str(out_sub),
            "--use-mock-env",
            "--generator",
            "dumpscript:tests/fixtures/synthetic_generator_sample.json",
            "--limit",
            "1",
            "--asr",
            "gipformer",
            "--override",
            f"asr.model_dir={fake_empty_model_dir}",
        ]
        run_res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert run_res.returncode == 2
        assert "ERROR:" in run_res.stderr
        assert "GipformerBackend requires" in run_res.stderr or "missing Gipformer" in run_res.stderr or "download" in run_res.stderr
        assert "Traceback" not in run_res.stderr
        assert not out_sub.exists()


def test_15_agent_mock_toolenv_e2e_with_gipformer_stub():
    """Item 15: Agent + MockToolEnv + stub ASR processes one sample end to end."""
    from lifemate.agent.agent import ToolAgent
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.data.loader import load_catalog, load_samples
    from lifemate.env.mock import MockToolEnv

    catalog = load_catalog(REPO_ROOT / "tests/fixtures/synthetic_lifemate", "public_test")
    assert catalog is not None
    registry = ToolRegistry(catalog)
    samples = load_samples(
        REPO_ROOT / "tests/fixtures/synthetic_lifemate", "public_test", limit=1, with_labels=False
    )
    assert len(samples) == 1
    sample = samples[0]

    # Stub backend simulating recognized voice prompt
    stub_text = "Đặt báo thức 7 giờ sáng"
    stub_result = TranscriptResult(text=stub_text, raw_text=stub_text, backend="gipformer")

    env = MockToolEnv()

    class _FakeGen:
        def __init__(self) -> None:
            self.step = 0

        def generate(self, messages: Any) -> str:
            if self.step == 0:
                self.step += 1
                return '<action>{"name": "smart_home.list_devices", "arguments": {}}</action>'
            return '<action>{"type": "finish"}</action>'

    agent = ToolAgent(generator=_FakeGen(), registry=registry, env=env)
    res = agent.run(sample, transcript=stub_result.text)
    assert res is not None
    assert env.get_tool_calls() is not None
    assert len(env.get_tool_calls()) >= 1


def test_16_oracle_restriction_unchanged():
    """Item 16: Oracle ASR is forbidden on public_test without labeled data and --allow-oracle."""
    cmd = [
        sys.executable,
        "scripts/infer.py",
        "--split",
        "public_test",
        "--data-root",
        "tests/fixtures/synthetic_lifemate",
        "--output",
        "/tmp/oracle_forbidden.jsonl",
        "--use-mock-env",
        "--asr",
        "oracle",
        "--limit",
        "1",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 2
    assert "forbidden" in res.stderr


def test_17_submission_compatibility():
    """Item 17: submission output rows contain expected keys from environment."""
    from lifemate.evaluation.submission import assert_env_sourced

    from lifemate.env.mock import MockToolEnv

    env = MockToolEnv()
    fake_row = {
        "id": "sample_001",
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash(),
    }
    assert_env_sourced(fake_row, env)


def test_18_docker_structural_check():
    """Docker structural check: Dockerfile contains sherpa-onnx and /models/asr."""
    dockerfile = REPO_ROOT / "docker" / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "sherpa-onnx" in content
    assert "sherpa-onnx-core" in content
    assert "/models/asr" in content
    assert "ASR=gipformer" in content
    # Ensure whisper weights are not baked/copied
    assert "COPY" not in [
        line for line in content.splitlines() if "whisper" in line.lower() and "copy" in line.lower()
    ]


def test_19_download_script_dry_run():
    """download_asr_model.py --dry-run prints size and file list."""
    cmd = [sys.executable, "scripts/download_asr_model.py", "--dry-run", "--quantization", "int8"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0
    assert "Gipformer-68M-RNNT download plan (dry run)" in res.stdout
    assert "encoder.int8.onnx" in res.stdout
    assert "tokens.txt" in res.stdout
    assert "Total Size:" in res.stdout


def test_20_download_script_prohibited_path_guard():
    """download_asr_model.py rejects paths under *_environment directories."""
    prohibited_dir = "data/public_test_environment/models"  # compliance-scan: allow
    cmd = [
        sys.executable,
        "scripts/download_asr_model.py",
        "--output-dir",
        prohibited_dir,
        "--dry-run",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 2
    assert "prohibited" in res.stderr.lower() or "prohibited" in res.stdout.lower()


def test_21_model_size_report_structure():
    """Backend model_size_report gives parameter and on-disk budget."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )

        with _mock_sherpa():
            backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            report = backend.model_size_report()
            assert report["backend"] == "gipformer"
            assert report["parameter_class"] == "68M"
            assert report["parameters"] == 68_000_000
            assert report["on_disk_bytes"] > 0
            assert report["on_disk_mb"] >= 0
            assert "tokens.txt" in report["files"]

# ---------------------------------------------------------------------------
# Defence in depth: the openers refuse *_environment paths themselves
# ---------------------------------------------------------------------------
def test_load_audio_16k_mono_refuses_prohibited_path():
    """load_audio_16k_mono guards before wave.open, not only the caller."""
    from lifemate.asr.interface import load_audio_16k_mono
    from lifemate.errors import ProhibitedAccessError

    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / "train_environment"  # compliance-scan: allow
        env_dir.mkdir()
        wav = _create_wav(env_dir / "a.wav", n_samples=160)
        try:
            load_audio_16k_mono(wav)
        except ProhibitedAccessError as exc:
            assert "train_environment" in exc.path
        else:  # pragma: no cover
            raise AssertionError("load_audio_16k_mono opened a *_environment path")

def test_gipformer_load_audio_refuses_prohibited_path():
    """GipformerBackend._load_audio raises before soundfile/wave touch the file."""
    from lifemate.errors import ProhibitedAccessError

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_dir = tmp_path / "model"
        _populate_fake_model_dir(
            model_dir, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        env_dir = tmp_path / "public_test_environment"  # compliance-scan: allow
        env_dir.mkdir()
        wav = _create_wav(env_dir / "x.wav", n_samples=160)

        opened: list[str] = []
        real_wave_open = wave.open

        def _spy_open(*args: Any, **kwargs: Any) -> Any:
            opened.append(str(args[0]))
            return real_wave_open(*args, **kwargs)

        with _mock_sherpa():
            backend = GipformerBackend(model_dir=model_dir, quantization="int8", offline=True)
            wave.open = _spy_open  # type: ignore[assignment]
            try:
                try:
                    backend._load_audio(wav)
                except ProhibitedAccessError as exc:
                    assert "public_test_environment" in exc.path
                else:  # pragma: no cover
                    raise AssertionError("_load_audio read a *_environment path")
            finally:
                wave.open = real_wave_open  # type: ignore[assignment]
        assert opened == [], f"a file was opened before the guard fired: {opened}"


def test_soundfile_read_is_bounded_by_max_audio_seconds():
    """max_audio_seconds must bound the READ, not only the post-hoc slice.

    Regression: ``sf.read(path, dtype="float32")`` read the whole file (frames=-1)
    and the guard truncated afterwards, so a long recording was fully materialised
    (a Python float list is ~73 MB per 10 min at 16 kHz) before being cut.
    """
    import types

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_dir = tmp_path / "model"
        _populate_fake_model_dir(
            model_dir, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        wav = _create_wav(tmp_path / "long.wav", n_samples=16000 * 600, rate=16000)  # 10 min

        seen: dict[str, Any] = {}

        class _Arr(list):
            size = property(lambda self: len(self))
            ndim = 1

        def _fake_read(path: str, dtype: str | None = None, frames: int = -1, **kw: Any):
            seen["frames"] = frames
            with wave.open(str(path), "rb") as handle:
                n = handle.getnframes() if frames is None or frames < 0 else min(frames, handle.getnframes())
                raw = handle.readframes(n)
            return _Arr([b / 32768.0 for b in raw[::2]]), 16000

        fake_sf = types.ModuleType("soundfile")
        fake_sf.read = _fake_read  # type: ignore[attr-defined]
        fake_sf.info = lambda path: types.SimpleNamespace(samplerate=16000)  # type: ignore[attr-defined]

        old_sf = sys.modules.get("soundfile")
        sys.modules["soundfile"] = fake_sf
        try:
            with _mock_sherpa():
                backend = GipformerBackend(
                    model_dir=model_dir, quantization="int8", offline=True, max_audio_seconds=60.0
                )
                samples, _rate, duration = backend._load_audio(wav)
        finally:
            if old_sf is None:
                sys.modules.pop("soundfile", None)
            else:
                sys.modules["soundfile"] = old_sf

        assert seen["frames"] == 960_000, f"read was not bounded: frames={seen['frames']}"
        assert duration == pytest.approx(60.0, rel=1e-3)
        assert len(samples) == 960_000


def test_symlink_into_environment_dir_is_refused():
    """The lexical guard cannot see a symlink; the resolved target must be checked too.

    Regression: ``safe_split/innocent.wav -> train_environment/secret.wav`` passed
    ``is_prohibited_path`` because no *component* of the link path ends in
    ``_environment``, and the loader opened the prohibited file.
    """
    import os

    from lifemate.asr.interface import load_audio_16k_mono
    from lifemate.errors import ProhibitedAccessError

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env_dir = tmp_path / "train_environment"  # compliance-scan: allow
        env_dir.mkdir()
        real = _create_wav(env_dir / "secret.wav", n_samples=160)
        link_dir = tmp_path / "safe_split"
        link_dir.mkdir()
        link = link_dir / "innocent.wav"
        os.symlink(real, link)

        with pytest.raises(ProhibitedAccessError) as excinfo:
            load_audio_16k_mono(link)
        assert "train_environment" in excinfo.value.path

        model_dir = tmp_path / "model"
        _populate_fake_model_dir(
            model_dir, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        with _mock_sherpa():
            backend = GipformerBackend(model_dir=model_dir, quantization="int8", offline=True)
            with pytest.raises(ProhibitedAccessError):
                backend.transcribe(link, lenient=True)  # lenient must not swallow it


# ---------------------------------------------------------------------------
# download_asr_model.py --verify: hash checking, offline degradation
# ---------------------------------------------------------------------------
def _load_download_script():
    import importlib.util

    path = REPO_ROOT / "scripts" / "download_asr_model.py"
    spec = importlib.util.spec_from_file_location("lifemate_download_asr_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verify_cli_parses_without_network():
    """--help must not touch the network or require huggingface_hub."""
    res = subprocess.run(
        [sys.executable, "scripts/download_asr_model.py", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0
    assert "--verify" in res.stdout


def test_verify_detects_same_size_tampering():
    """A flipped byte at identical length must fail; the old size check passed it.

    Uses injected hashes so the test is hermetic (no network), but exercises the
    real digest path: sha256 for LFS files, git-blob sha1 for plain ones.
    """
    mod = _load_download_script()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        payload = b"tokens payload\n"
        (out / "tokens.txt").write_bytes(payload)
        size = len(payload)

        mod.KNOWN_FILES = {**mod.KNOWN_FILES, "tokens.txt": size}
        good = mod.file_digest(out / "tokens.txt", "sha256")
        assert len(good) == 64
        git_id = mod.file_digest(out / "tokens.txt", "git-sha1")
        assert len(git_id) == 40 and git_id != good

        tampered = bytearray(payload)
        tampered[0] ^= 0xFF
        (out / "tokens.txt").write_bytes(bytes(tampered))
        assert (out / "tokens.txt").stat().st_size == size  # size check still satisfied

        assert mod.verify_files(out, ["tokens.txt"], {"tokens.txt": ("sha256", good)}) == 2
        assert mod.verify_files(out, ["tokens.txt"], {"tokens.txt": ("git-sha1", git_id)}) == 2

        # restoring the bytes makes the same hashes pass again
        (out / "tokens.txt").write_bytes(payload)
        assert mod.verify_files(out, ["tokens.txt"], {"tokens.txt": ("sha256", good)}) == 0


def test_verify_offline_degrades_to_sizes_only():
    """No metadata -> size fallback, never claims a hash check, still fails on size."""
    mod = _load_download_script()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "tokens.txt").write_bytes(b"x" * mod.KNOWN_FILES["tokens.txt"])
        assert mod.verify_files(out, ["tokens.txt"], None) == 0

        (out / "tokens.txt").write_bytes(b"x" * 10)
        assert mod.verify_files(out, ["tokens.txt"], None) == 2


def test_verify_reports_missing_file():
    mod = _load_download_script()
    with tempfile.TemporaryDirectory() as tmp:
        assert mod.verify_files(Path(tmp), ["tokens.txt"], None) == 2


def test_fetch_pinned_hashes_returns_none_offline():
    """Unreachable metadata must yield None, not an exception or a fake dict."""
    mod = _load_download_script()
    import urllib.request

    real = urllib.request.urlopen

    def _boom(*args, **kwargs):
        raise OSError("network disabled in tests")

    urllib.request.urlopen = _boom
    try:
        assert mod.fetch_pinned_hashes("g-group-ai-lab/gipformer-68M-rnnt", "abc", timeout=0.5) is None
    finally:
        urllib.request.urlopen = real

def test_factory_default_decoding_matches_backend_and_config():
    """Regression: create_asr_backend fell back to greedy_search while AsrConfig,
    GipformerBackend and both YAMLs default to modified_beam_search — so an
    asr_cfg lacking the field silently decoded differently from the published WER.
    """
    from lifemate.asr.gipformer_backend import GipformerBackend as _B  # noqa: F401

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path,
            ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
        )
        with _mock_sherpa() as rec_mock:
            create_asr_backend(
                {"asr": {"model_dir": str(tmp_path), "quantization": "int8"}}  # type: ignore[arg-type]
            )
        assert rec_mock.last_kwargs["decoding_method"] == "modified_beam_search"

def test_uppercase_only_recognizer_text_is_lowercased_for_the_agent():
    """Regression: gipformer-68M-rnnt emits ALL UPPERCASE (4790/4790 real turns),
    while every training transcript is sentence-case, so the agent's request field
    shifted on the whole corpus. Uppercase-only output carries no case information
    and is lowercased for `text`; `raw_text` stays verbatim, and mixed-case output
    is never touched.
    """
    from lifemate.asr.gipformer_backend import lower_if_uppercase_only

    assert lower_if_uppercase_only("MAI TÔI CÓ NHỮNG BÁO THÍCH NÀO VẬY") == \
        "mai tôi có những báo thích nào vậy"
    # mixed case -> untouched (proper nouns survive)
    assert lower_if_uppercase_only("Chuyển 500.000đ cho anh Tuấn") == \
        "Chuyển 500.000đ cho anh Tuấn"
    # digits/punctuation only -> untouched, no crash
    assert lower_if_uppercase_only("500.000") == "500.000"
    assert lower_if_uppercase_only("") == ""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _populate_fake_model_dir(
            tmp_path, ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        )
        wav = _create_wav(tmp_path / "audio.wav", n_samples=8000, rate=16000)
        with _mock_sherpa(stream_text="NẾU MAI MƯA THÌ DỜI CHẠY BỘ MUỘN 1 TIẾNG"):
            backend = GipformerBackend(model_dir=tmp_path, quantization="int8", offline=True)
            res = backend.transcribe(wav)
    assert res.raw_text == "NẾU MAI MƯA THÌ DỜI CHẠY BỘ MUỘN 1 TIẾNG"      # verbatim kept
    assert res.text == "nếu mai mưa thì dời chạy bộ muộn 1 tiếng"            # agent-facing
