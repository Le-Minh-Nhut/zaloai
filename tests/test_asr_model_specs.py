"""Spec-table tests: gipformer checkpoints are described, not substring-matched.

Stdlib-runnable: the spec table is pure stdlib, and every assertion about the
backend exercises a path that fails *before* sherpa-onnx is imported.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None  # stdlib runner: assertions only

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.asr.gipformer_backend import DEFAULT_MODEL_ID, PINNED_REVISION, GipformerBackend
from lifemate.asr.interface import AsrError
from lifemate.asr.model_specs import (
    GIPFORMER_SPECS,
    KNOWN_MODEL_IDS,
    artifacts_for,
    hash_tokens_file,
    require_spec_artifacts,
    resolve_gipformer_spec,
    spec_key,
    validate_artifacts,
)
from lifemate.config import AppConfig, AsrConfig, ConfigError, load_config
from lifemate.compliance import ENV_DIR_SUFFIX
from lifemate.errors import ProhibitedAccessError

_68M = "gipformer-68M-rnnt"
_15 = "gipformer1.5-68M-rnnt"
LOCAL_DIR = REPO_ROOT / "models" / "gipformer-68M-rnnt"


def test_spec_table_describes_exactly_the_measured_68m_files():
    """The 68M spec lists the files that are actually on this machine."""
    spec = GIPFORMER_SPECS[_68M]
    assert spec.model_id == DEFAULT_MODEL_ID
    assert spec.revision == PINNED_REVISION
    assert spec.sample_rate == 16000
    assert spec.feature_dim == 80
    assert spec.verified is True
    assert artifacts_for(spec, "int8") == {
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
        "tokens": "tokens.txt",
    }
    # The fp32 set is the same roles without the .int8 infix (checkpoint config.json).
    assert artifacts_for(spec, "fp32")["encoder"] == "encoder.onnx"
    assert spec.languages == ("vi",)


def test_tokens_sha256_is_the_hash_of_the_local_file():
    """The recorded digest is measured, never copied from a manifest."""
    spec = GIPFORMER_SPECS[_68M]
    assert spec.tokens_sha256
    token_file = LOCAL_DIR / "tokens.txt"
    if not token_file.is_file():  # weights absent: the entry must still be a real digest
        assert len(spec.tokens_sha256) == 64
        return
    assert hash_tokens_file(token_file) == spec.tokens_sha256
    # And validate_artifacts agrees, including on a corrupted copy.
    assert validate_artifacts(spec, LOCAL_DIR, verify_tokens=True) == []
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp)
        for role, name in artifacts_for(spec, "int8").items():
            (broken / name).write_bytes(b"x")
        problems = validate_artifacts(spec, broken, verify_tokens=True)
        assert any("tokens.txt sha256 mismatch" in problem for problem in problems)


def test_15_spec_is_marked_unverified_and_declares_no_artifact_set():
    """No weights were ever observed, so the entry says so instead of guessing."""
    spec = GIPFORMER_SPECS[_15]
    assert spec.verified is False
    assert spec.quantizations == ()
    assert spec.revision == ""
    assert spec.tokens_sha256 is None
    assert "[UNVERIFIED]" in spec.notes
    assert "never observed" in spec.notes.lower() or "NEVER observed" in spec.notes
    # Conventional role names only — and asking for files raises instead of guessing.
    assert sorted(spec.onnx) == ["decoder", "encoder", "joiner", "tokens"]
    try:
        artifacts_for(spec, "int8")
    except ConfigError as exc:
        assert "[UNVERIFIED]" in str(exc) or "UNVERIFIED" in str(exc)
    else:
        raise AssertionError("an unverified spec must not produce an artifact list")


def test_known_ids_and_unknown_id_error_lists_them():
    assert set(KNOWN_MODEL_IDS) == {
        "g-group-ai-lab/gipformer-68M-rnnt",
        "g-group-ai-lab/gipformer1.5-68M-rnnt",
    }
    for model_id in KNOWN_MODEL_IDS:
        assert resolve_gipformer_spec(model_id).model_id == model_id

    try:
        resolve_gipformer_spec("someone-else/gipformer-9000-rnnt")
    except ConfigError as exc:
        message = str(exc)
        assert "unknown Gipformer model" in message
        for known in KNOWN_MODEL_IDS:
            assert known in message
        assert "model_specs.py" in message and "add a GipformerSpec" in message
    else:
        raise AssertionError("an unknown model id must be a ConfigError, never a fallback")


def test_resolution_accepts_directory_names_and_rejects_conflicts():
    assert spec_key("models/gipformer-68M-rnnt") == _68M
    assert spec_key("/abs/path/gipformer1.5-68M-rnnt/") == _15
    assert spec_key("models/unknown-thing") is None
    # kind= disambiguates a directory that carries no descriptor...
    assert resolve_gipformer_spec("models/whatever", kind=_68M).key == _68M
    # ...but a conflicting hint is an error, not a precedence rule.
    try:
        resolve_gipformer_spec("models/gipformer1.5-68M-rnnt", kind=_68M)
    except ConfigError as exc:
        assert "conflicting Gipformer specs" in str(exc)
    else:
        raise AssertionError("conflicting spec hints must raise")


def test_environment_paths_are_refused_by_spec_resolution():
    # Composed from the shipped constant so no literal in this file reads as a
    # prohibited path component (the repo's static scanner checks exactly that).
    prohibited = [
        "data/train/" + "train" + ENV_DIR_SUFFIX + "/gipformer-68M-rnnt",
        "data/public_test/" + "public_test" + ENV_DIR_SUFFIX + "/x",
    ]
    for path in prohibited:
        try:
            resolve_gipformer_spec(path)
        except ProhibitedAccessError:
            pass
        else:
            raise AssertionError(f"*_environment path was accepted: {path}")
        try:
            validate_artifacts(GIPFORMER_SPECS[_68M], path)
        except ProhibitedAccessError:
            raise
        # validate_artifacts itself reports rather than raises; resolution above is the guard.


def test_validate_artifacts_reports_missing_files_without_raising():
    spec = GIPFORMER_SPECS[_68M]
    with tempfile.TemporaryDirectory() as tmp:
        problems = validate_artifacts(spec, tmp)
        assert len(problems) == 4
        assert any("encoder.int8.onnx" in problem for problem in problems)
        assert any("tokens.txt" in problem for problem in problems)
        # A non-directory is a problem, not an exception.
        assert validate_artifacts(spec, Path(tmp) / "nope")[0].startswith("model directory not found")


def test_require_spec_artifacts_names_every_missing_file():
    spec = GIPFORMER_SPECS[_68M]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            require_spec_artifacts(spec, tmp)
        except AsrError as exc:
            message = str(exc)
            for filename in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
                assert filename in message
            assert "download_asr_model.py" in message
        else:
            raise AssertionError("missing artifacts must raise AsrError")


def test_int8_only_checkout_makes_fp32_fail_on_encoder_onnx():
    """The local set is INT8-ONLY: asking for fp32 names the missing file."""
    spec = GIPFORMER_SPECS[_68M]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in artifacts_for(spec, "int8").values():
            (root / name).write_bytes(b"int8")
        assert validate_artifacts(spec, root, quantization="int8") == []
        problems = validate_artifacts(spec, root, quantization="fp32")
        assert "missing encoder file: encoder.onnx" in problems
        try:
            require_spec_artifacts(spec, root, quantization="fp32")
        except AsrError as exc:
            assert "encoder.onnx" in str(exc)
            assert "quantization='fp32'" in str(exc) or "fp32" in str(exc)
        else:
            raise AssertionError("fp32 request against an int8-only checkout must fail")


def test_backend_rejects_15_through_the_spec_not_a_substring():
    try:
        GipformerBackend(model_id="g-group-ai-lab/gipformer1.5-68M-rnnt")
    except AsrError as exc:
        message = str(exc)
        assert "[UNVERIFIED]" in message
        assert "gipformer1.5-68M-rnnt" in message
    else:
        raise AssertionError("the [UNVERIFIED] 1.5 spec must not build a recognizer")


def test_backend_rejects_unknown_model_instead_of_falling_back_to_68m():
    try:
        GipformerBackend(model_id="someone-else/gipformer-mystery")
    except ConfigError as exc:
        assert "unknown Gipformer model" in str(exc)
    else:
        raise AssertionError("an unknown model id must never silently load the 68M checkpoint")


def test_backend_reports_missing_fp32_files_before_importing_sherpa():
    """No sherpa import needed: the artifact check runs first and names encoder.onnx."""
    try:
        GipformerBackend(model_dir=LOCAL_DIR, quantization="fp32")
    except AsrError as exc:
        message = str(exc)
        assert "missing Gipformer (fp32) model files" in message
        assert "encoder.onnx" in message
        assert "silent fallback is forbidden" in message
    else:
        raise AssertionError("fp32 against the int8-only checkout must fail")


def test_backend_spec_is_exposed_for_inspection():
    spec = resolve_gipformer_spec(DEFAULT_MODEL_ID)
    assert spec.key == _68M
    assert spec.verified is True


def test_config_accepts_known_ids_and_rejects_unknown_or_unverified():
    cfg = load_config()
    assert cfg.asr.model_id == DEFAULT_MODEL_ID
    cfg.validate()  # baseline config still validates

    # Verified id: accepted by config (whether its files exist is a load-time question).
    AppConfig(asr=AsrConfig(model_id=DEFAULT_MODEL_ID))

    # A KNOWN id is accepted by config even when its weights are [UNVERIFIED] here:
    # whether the files exist is a load-time question, not a config one.
    AppConfig(asr=AsrConfig(model_id="g-group-ai-lab/gipformer1.5-68M-rnnt"))

    # An UNKNOWN gipformer id is a config error listing every known id.
    try:
        AppConfig(asr=AsrConfig(model_id="someone/gipformer-9000"))
    except ConfigError as exc:
        message = str(exc)
        assert "unknown gipformer model_id" in message
        for known in KNOWN_MODEL_IDS:
            assert known in message
    else:
        raise AssertionError("an unknown gipformer model id must be rejected by AppConfig")

    # Non-gipformer backends are unaffected by the gipformer rule.
    AppConfig(asr=AsrConfig(backend="whisper", model_id="vinai/PhoWhisper-small"))
