"""Compliance guard tests: prohibited path detection and static scan."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate import compliance  # noqa: E402
from lifemate.compliance import (  # noqa: E402
    assert_no_environment_access,
    assert_path_allowed,
    is_prohibited_path,
    read_json,
    scan_source_for_prohibited_access,
)
from lifemate.errors import ProhibitedAccessError  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"


def test_env_dir_suffix_constant():
    assert compliance.ENV_DIR_SUFFIX == "_environment"


def test_is_prohibited_path_three_official_forms():
    assert is_prohibited_path("/data/train_environment")
    assert is_prohibited_path("/data/public_test_environment/")
    assert is_prohibited_path("/data/private_test_environment")


def test_is_prohibited_path_nested_and_relative():
    assert is_prohibited_path("data/train/train_environment/x.json")
    assert is_prohibited_path(Path("a/b/public_test_environment/c.jsonl"))
    assert is_prohibited_path("train_environment")
    assert is_prohibited_path("data/*_environment/*.json")
    assert is_prohibited_path("data/*_environment")


def test_is_prohibited_path_allows_normal_paths():
    for path in [
        "/data/train/train.jsonl",
        "/data/public_test/audio/x.wav",
        "mock_state/train_multi_step.json",
        "tests/fixtures/synthetic_lifemate/train",
        "/data/environmental_science/x.json",
        "src/lifemate/env/mock.py",
    ]:
        assert not is_prohibited_path(path), path


def test_assert_path_allowed_raises_with_attributes():
    try:
        assert_path_allowed("/data/train_environment/sample.jsonl")
    except ProhibitedAccessError as exc:
        assert exc.path == "/data/train_environment/sample.jsonl"
        assert "prohibited" in exc.reason
    else:  # pragma: no cover
        raise AssertionError("ProhibitedAccessError not raised")


def test_safe_open_refuses_without_reading():
    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / "train_environment"
        env_dir.mkdir()
        secret = env_dir / "state.json"
        secret.write_text('{"secret": true}', encoding="utf-8")
        try:
            with compliance.safe_open(secret, "r", encoding="utf-8") as handle:
                handle.read()  # pragma: no cover
        except ProhibitedAccessError as exc:
            assert "train_environment" in exc.path
        else:  # pragma: no cover
            raise AssertionError("safe_open did not raise")

        try:
            read_json(secret)
        except ProhibitedAccessError:
            pass
        else:  # pragma: no cover
            raise AssertionError("read_json did not raise")

        # A permitted path in the same temp tree still works.
        allowed = Path(tmp) / "ok.json"
        allowed.write_text('{"a": 1}', encoding="utf-8")
        assert read_json(allowed) == {"a": 1}


def test_safe_open_reads_synthetic_catalog():
    path = FIXTURES / "train" / "train_tools_catalog.json"
    assert read_json(path)["_synthetic"] is True


def test_assert_no_environment_access():
    assert_no_environment_access(["data/train/train.jsonl", "data/public_test/x.wav"])
    try:
        assert_no_environment_access(["data/train/train.jsonl", "data/train_environment/a"])
    except ProhibitedAccessError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ProhibitedAccessError")


def test_scan_source_is_clean_on_repo():
    findings = scan_source_for_prohibited_access(ROOT)
    assert findings == [], findings


def test_scan_source_detects_violation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "bad.py").write_text(
            'PATH = "data/train_environment/state.json"\n'
            'GLOB = "data/*_environment/*.json"\n'
            "# a comment mentioning train_environment is ignored\n",
            encoding="utf-8",
        )
        findings = scan_source_for_prohibited_access(root)
        assert len(findings) == 2, findings
        assert all(f["file"] == "src/bad.py" for f in findings)
        assert any("train_environment" in f["text"] for f in findings)
        assert {f["line"] for f in findings} == {1, 2}


def test_scan_source_missing_dirs_ok():
    with tempfile.TemporaryDirectory() as tmp:
        assert scan_source_for_prohibited_access(Path(tmp) / "nope") == []


def test_scan_detects_bare_literal_join_patterns():
    """Bare `joinpath("…_environment")` / `Path(base) / "…_environment"` count.

    Regression: the scan previously required a "/" inside the literal, so the
    exact shape an accidental violation takes was invisible.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "leaky.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "def a(base):\n"
            '    return Path(base) / "train_environment"\n'
            "\n"
            "def b(base):\n"
            '    return base.joinpath("public_test_environment")\n'
            "\n"
            "def c(base):\n"
            '    return os.path.join(base, "private_test_environment")\n'
            "\n"
            "def d(base):\n"
            '    return base / "x" / "train_environment"\n',
            encoding="utf-8",
        )
        findings = scan_source_for_prohibited_access(root)
        lines = sorted(f["line"] for f in findings)
        assert lines == [4, 7, 10, 13], findings
        assert all(f["file"] == "src/leaky.py" for f in findings)
        assert len(findings) == 4


def test_scan_ignores_benign_identifiers():
    """Names that merely mention "environment" are not path violations."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "fine.py").write_text(
            'NAME = "environment"\n'
            'APP = "training_env"\n'
            'VAR = "env_dir"\n'
            'SEP = "environmental_science"\n'
            'def create_environment(x):\n'
            '    return {"create_environment": x}\n'
            'SUFFIX = "_environment"\n'
            'ALL = ["create_environment", "describe_environment"]\n'
            'CALL = create_environment("environment")\n',
            encoding="utf-8",
        )
        findings = scan_source_for_prohibited_access(root)
        assert findings == [], findings


def test_scan_flags_suffixed_names_but_not_bare_suffix():
    """`x_environment` in path position is flagged; the bare suffix alone is not."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "mixed.py").write_text(
            'from pathlib import Path\n'
            'SUFFIX = "_environment"\n'
            'BAD = Path("/data") / ("split" + SUFFIX)\n'
            'DIRECT = Path("cache_environment")\n',
            encoding="utf-8",
        )
        findings = scan_source_for_prohibited_access(root)
        assert {f["line"] for f in findings} == {4}, findings
