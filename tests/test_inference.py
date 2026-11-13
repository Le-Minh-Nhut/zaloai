"""End-to-end inference tests: mock env + scripted generator -> submission.

Covers the three mock barriers (CLI flag, factory argument, banner/output guard),
the oracle-ASR refusal, env-sourced rows, failed-call preservation and one real
subprocess run of ``scripts/infer.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lifemate.agent.agent import ToolAgent  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.config import load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, read_sample_audio  # noqa: E402
from lifemate.env.factory import create_environment  # noqa: E402
from lifemate.env.mock import MockToolEnv  # noqa: E402
from lifemate.errors import ProhibitedAccessError  # noqa: E402
from lifemate.evaluation.submission import SubmissionWriter, load_submission  # noqa: E402
from lifemate.evaluation.submission_validator import validate_submission  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"


class ScriptedGenerator:
    """Test-local deterministic generator: one raw output per call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.max_new_tokens = 64

    def generate(self, messages):
        if not self.outputs:
            return '<action>{"type": "finish"}</action>'
        return self.outputs.pop(0)


def _mock_env(sample):
    state = None
    state_path = FIXTURES / "mock_state" / f"{sample.id}.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    return create_environment(
        sample.split,
        sample.id,
        FIXTURES,
        use_mock=True,
        catalog=load_catalog(FIXTURES, sample.split),
        mock_state=state,
        apps=sample.installed_apps,
    )


def test_agent_to_submission_end_to_end():
    samples = load_samples(FIXTURES, "train", limit=3, with_labels=False)
    catalog = load_catalog(FIXTURES, "train")
    gen = ScriptedGenerator(
        [
            '<action>{"type": "tool_call", "name": "alarm.set", '
            '"arguments": {"time": "06:30", "label": "morning", "repeat": "once"}}</action>',
            '<action>{"type": "finish"}</action>',
        ]
    )
    directory = Path(tempfile.mkdtemp(prefix="lifemate_e2e_"))
    output = directory / "submission.jsonl"
    writer = SubmissionWriter(output)
    expected_ids = []
    for sample in samples:
        expected_ids.append(sample.id)
        env = _mock_env(sample)
        agent = ToolAgent(generator=gen, registry=ToolRegistry(catalog).for_sample(sample), env=env)
        result = agent.run(sample)
        # Rows must echo the env, not the agent's own notes.
        writer.add(sample.id, env.get_tool_calls(), env.get_state_hash())
        assert result.state_hash == env.get_state_hash()
    writer.finalize()

    rows = load_submission(output)
    assert [row["id"] for row in rows] == expected_ids
    assert rows[0]["tool_calls"][0]["name"] == "alarm.set"
    assert len(rows[0]["state_hash"]) == 64
    report = validate_submission(output, expected_ids)
    assert report["ok"], report


def test_failed_tool_call_is_preserved():
    """A tool the env rejects must appear in the submission with its error."""
    samples = {sample.id: sample for sample in load_samples(FIXTURES, "train", with_labels=False)}
    sample = samples["train_failed_tool_result"]
    catalog = load_catalog(FIXTURES, "train")
    gen = ScriptedGenerator(
        [
            '<action>{"type": "tool_call", "name": "messaging.send", '
            '"arguments": {"contact_id": "c1", "body": "hello"}}</action>',
            '<action>{"type": "finish"}</action>',
        ]
    )
    # A schema-valid call that the environment itself rejects: exactly the case
    # that must survive into the submission with its error intact.
    state_path = FIXTURES / "mock_state" / f"{sample.id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    env = MockToolEnv(
        catalog=catalog,
        initial_state=state,
        apps=sample.installed_apps,
        sample_id=sample.id,
        fail_tools={"messaging.send": 1},
    )
    agent = ToolAgent(generator=gen, registry=ToolRegistry(catalog).for_sample(sample), env=env)
    agent.run(sample)
    calls = env.get_tool_calls()
    assert calls and calls[0]["name"] == "messaging.send"
    assert calls[0].get("ok") is False and calls[0].get("error")
    row = {"id": sample.id, "tool_calls": calls, "state_hash": env.get_state_hash()}
    assert row["tool_calls"][0]["error"], "failed call lost its error"


def test_mock_requires_explicit_flag_and_output_guard():
    """CLI-level barriers: mock needs a flag; it cannot use the official path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_infer", ROOT / "scripts" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    sys.modules["lifemate_infer"] = infer
    spec.loader.exec_module(infer)

    from lifemate.errors import LifeMateError

    args = infer.parse_args(["--use-mock-env"])
    try:
        infer._refuse_mock_on_official_output(args, Path("outputs/submission.jsonl"))
    except LifeMateError as exc:
        assert "official output path" in str(exc)
    else:
        raise AssertionError("mock run was allowed to write the official output path")

    # explicit --output is accepted
    args2 = infer.parse_args(["--use-mock-env", "--output", "/tmp/ok.jsonl"])
    infer._refuse_mock_on_official_output(args2, Path("/tmp/ok.jsonl"))

    # factory default is the official env
    import inspect

    default = inspect.signature(create_environment).parameters["use_mock"].default
    assert default is False, "create_environment must default to the official simulator"


def test_oracle_asr_refused_for_public_test():
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_infer2", ROOT / "scripts" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    sys.modules["lifemate_infer2"] = infer
    spec.loader.exec_module(infer)
    from lifemate.errors import LifeMateError

    cfg = load_config()
    cfg.inference.split = "public_test"
    args = infer.parse_args(["--asr", "oracle", "--allow-oracle"])
    samples = load_samples(FIXTURES, "public_test", limit=1, with_labels=False)
    try:
        infer._build_asr(args, cfg, samples)
    except LifeMateError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("oracle ASR was accepted on public_test")

    # and without --allow-oracle even a labeled split is refused
    cfg.inference.split = "train"
    args = infer.parse_args(["--asr", "oracle"])
    try:
        infer._build_asr(args, cfg, samples)
    except LifeMateError as exc:
        assert "allow-oracle" in str(exc)
    else:
        raise AssertionError("oracle ASR was accepted without --allow-oracle")


def test_generator_flag_guard():
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_infer3", ROOT / "scripts" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    sys.modules["lifemate_infer3"] = infer
    spec.loader.exec_module(infer)
    from lifemate.errors import LifeMateError

    official = infer.parse_args(["--generator", "dumpscript:/tmp/x.json"])
    try:
        infer._validate_generator_flag(official)
    except LifeMateError as exc:
        assert "TEST HOOK" in str(exc)
    else:
        raise AssertionError("--generator was accepted without --use-mock-env")
    infer._validate_generator_flag(infer.parse_args(["--generator", "dumpscript:/tmp/x.json", "--use-mock-env"]))


def test_infer_subprocess_mock_limit_two():
    """Real CLI run: exit 0, rows valid, mock banner printed, oracle refused."""
    directory = Path(tempfile.mkdtemp(prefix="lifemate_cli_"))
    output = directory / "submission.jsonl"
    script = directory / "dumpscript.json"
    samples = load_samples(FIXTURES, "public_test", limit=2, with_labels=False)
    payload = {
        samples[0].id: [
            '<action>{"type": "tool_call", "name": "smart_home.device_set", '
            '"arguments": {"device_id": "light1", "on": false}}</action>',
            '<action>{"type": "finish"}</action>',
        ],
        samples[1].id: ['<action>{"type": "finish"}</action>'],
    }
    script.write_text(json.dumps(payload), encoding="utf-8")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--split",
        "public_test",
        "--data-root",
        str(FIXTURES),
        "--use-mock-env",
        "--generator",
        f"dumpscript:{script}",
        "--limit",
        "2",
        "--output",
        str(output),
        "--asr",
        "none",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert "MOCK ENVIRONMENT" in completed.stderr

    rows = load_submission(output)
    assert len(rows) == 2
    ids = [sample.id for sample in samples]
    report = validate_submission(output, ids)
    assert report["ok"], report

    refused = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer.py"),
            "--split",
            "public_test",
            "--data-root",
            str(FIXTURES),
            "--use-mock-env",
            "--generator",
            f"dumpscript:{script}",
            "--output",
            str(directory / "second.jsonl"),
            "--asr",
            "oracle",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert refused.returncode == 2, (refused.returncode, refused.stderr)
    assert "forbidden" in refused.stderr


def test_missing_dataset_exit_code_two():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer.py"),
            "--split",
            "public_test",
            "--data-root",
            str(Path(tempfile.mkdtemp(prefix="lifemate_empty_"))),
            "--output",
            str(Path(tempfile.mkdtemp(prefix="lifemate_out_")) / "s.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert completed.returncode == 2
    assert "Dataset not found." in completed.stderr
    assert "Traceback" not in completed.stderr


def test_official_env_preflight_exits_two_before_model_load():
    """Without zaic_tools the run must fail in seconds with exit 2 and a clear message."""
    import importlib.util

    if importlib.util.find_spec("zaic_tools") is not None:
        return  # the real SDK is installed here: this gate is about its absence
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer.py"),
            "--split",
            "public_test",
            "--data-root",
            str(FIXTURES),
            "--asr",
            "none",
            "--limit",
            "1",
            "--output",
            str(Path(tempfile.mkdtemp(prefix="lifemate_pre_")) / "real.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(ROOT),
    )
    assert completed.returncode == 2, (completed.returncode, completed.stderr)
    assert "zaic_tools" in completed.stderr
    assert "torch" not in completed.stderr, "model loading must not be attempted before the env preflight"

def test_infer_override_preserves_config_and_handles_keys():
    """D1: --override applies in-place, preserves config/cli settings, unknown key exits 2 with ConfigError."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("infer_d1", ROOT / "scripts" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

    # (a) keeps config's split/output and sets max_steps 4
    args_a = infer.parse_args(["--config", "configs/inference.yaml", "--override", "agent.max_steps=4"])
    cfg_a = infer._build_config(args_a)
    assert cfg_a.agent.max_steps == 4
    assert cfg_a.inference.split == "public_test"
    assert cfg_a.inference.output == "outputs/submission.jsonl"

    # (b) --output /tmp/x.jsonl --override training.seed=1 still writes /tmp/x.jsonl
    args_b = infer.parse_args(["--output", "/tmp/x.jsonl", "--override", "training.seed=1"])
    cfg_b = infer._build_config(args_b)
    assert cfg_b.inference.output == "/tmp/x.jsonl"
    assert cfg_b.training.seed == 1

    # (c) unknown key exits 2 with ConfigError message, no traceback
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "infer.py"), "--override", "unknown.section.key=1"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 2
    assert "ConfigError" in proc.stderr or "unknown override" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_infer_mock_barrier_on_resolved_config():
    """D2: mock barrier guards resolved config, refusing default output without explicit --output."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("infer_d2", ROOT / "scripts" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "mock_cfg.yaml"
        cfg_path.write_text("inference:\n  use_mock_env: true\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "infer.py"), "--config", str(cfg_path)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert proc.returncode == 4, proc.stderr
        assert "refuses to write the default official output path" in proc.stderr

        script_file = Path(tmp) / "script.json"
        script_file.write_text('{"test_single_tool": ["<action>{\\"type\\": \\"finish\\"}</action>"]}', encoding="utf-8")
        out_file = Path(tmp) / "out.jsonl"
        proc2 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "infer.py"),
                "--config",
                str(cfg_path),
                "--data-root",
                str(FIXTURES),
                "--generator",
                f"dumpscript:{script_file}",
                "--asr",
                "none",
                "--limit",
                "1",
                "--output",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert proc2.returncode == 0, proc2.stderr
        assert "MOCK ENVIRONMENT" in proc2.stderr
        assert out_file.exists()


def test_infer_assert_env_sourced_provenance_guard():
    """D3: assert_env_sourced catches divergent env output and exits 4 with provenance error."""
    with tempfile.TemporaryDirectory() as tmp:
        script_file = Path(tmp) / "script.json"
        script_file.write_text('{"test_single_tool": ["<action>{\\"type\\": \\"finish\\"}</action>"]}', encoding="utf-8")
        out_file = Path(tmp) / "out.jsonl"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "infer.py"),
            "--split",
            "public_test",
            "--data-root",
            str(FIXTURES),
            "--use-mock-env",
            "--generator",
            f"dumpscript:{script_file}",
            "--asr",
            "none",
            "--limit",
            "1",
            "--output",
            str(out_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        assert proc.returncode == 0, proc.stderr

        mismatch_code = (
            "import sys\n"
            "sys.path.insert(0, 'src')\n"
            "from lifemate.evaluation.submission import SubmissionWriter\n"
            "orig_add = SubmissionWriter.add\n"
            "def bad_add(self, sample_id, tool_calls, state_hash, **extra):\n"
            "    return orig_add(self, sample_id, [{'name': 'fake.call', 'arguments': {}}], state_hash, **extra)\n"
            "SubmissionWriter.add = bad_add\n"
            "import scripts.infer as infer\n"
            "sys.exit(infer.main(sys.argv[1:]))\n"
        )
        proc_bad = subprocess.run(
            [sys.executable, "-c", mismatch_code, *cmd[2:]],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert proc_bad.returncode == 4, proc_bad.stderr
        assert "tool_calls differ from env.get_tool_calls()" in proc_bad.stderr


def test_infer_mock_output_guard_blocks_explicit_official_path():
    """Mock cannot write outputs/submission.jsonl even if passed explicitly."""
    with tempfile.TemporaryDirectory() as tmp:
        script_file = Path(tmp) / "script.json"
        script_file.write_text('{"train_single_tool": ["<action>{\\"type\\": \\"finish\\"}</action>"]}', encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "infer.py"),
                "--use-mock-env",
                "--split",
                "train",
                "--data-root",
                str(FIXTURES),
                "--generator",
                f"dumpscript:{script_file}",
                "--asr",
                "none",
                "--limit",
                "1",
                "--output",
                "outputs/submission.jsonl",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert proc.returncode == 4, proc.stderr
        assert "refuses to write the default official output path" in proc.stderr


def test_infer_resume_preserves_dataset_order():
    """--resume loads existing prefix and appends remaining in dataset order."""
    with tempfile.TemporaryDirectory() as tmp:
        script_file = Path(tmp) / "script.json"
        script_file.write_text(
            json.dumps(
                {
                    "train_single_tool": ['<action>{"type": "finish"}</action>'],
                    "train_multi_step": ['<action>{"type": "finish"}</action>'],
                }
            ),
            encoding="utf-8",
        )
        out_file = Path(tmp) / "sub.jsonl"
        # Run 1: limit 1
        proc1 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "infer.py"),
                "--use-mock-env",
                "--split",
                "train",
                "--data-root",
                str(FIXTURES),
                "--generator",
                f"dumpscript:{script_file}",
                "--asr",
                "none",
                "--limit",
                "1",
                "--output",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert proc1.returncode == 0, proc1.stderr

        # Run 2: resume limit 2
        proc2 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "infer.py"),
                "--use-mock-env",
                "--split",
                "train",
                "--data-root",
                str(FIXTURES),
                "--generator",
                f"dumpscript:{script_file}",
                "--asr",
                "none",
                "--limit",
                "2",
                "--output",
                str(out_file),
                "--resume",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert proc2.returncode == 0, proc2.stderr
        assert "resume: 1 row(s) already present" in proc2.stderr
        # Verify final file order
        rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert rows[0]["id"] == "train_single_tool"
        assert rows[1]["id"] == "train_multi_step"


# ---------------------------------------------------------------------------
# ASR wiring: loud failures and guarded audio
# ---------------------------------------------------------------------------
def _load_infer(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "infer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def test_unknown_asr_backend_raises_instead_of_returning_none():
    """A typo in --asr must never yield a submission of empty transcripts."""
    import argparse

    from lifemate.asr import AsrError

    infer = _load_infer("lifemate_infer_bogus")
    cfg = load_config()
    cfg.inference.split = "train"
    args = argparse.Namespace(asr="bogus", allow_oracle=False, transcripts=None)
    try:
        backend = infer._build_asr(args, cfg, [])
    except AsrError as exc:
        message = str(exc)
        assert "bogus" in message
        assert "gipformer" in message and "whisper" in message
    else:  # pragma: no cover
        raise AssertionError(f"unknown backend returned {backend!r} instead of raising")
    # The CLI rejects the same typo before any work starts.
    import contextlib
    import io

    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            infer.parse_args(["--asr", "bogus"])
    except SystemExit as exc:
        assert exc.code != 0
        assert "invalid choice" in stderr.getvalue()
    else:  # pragma: no cover
        raise AssertionError("CLI accepted --asr bogus")

def test_transcribe_propagates_prohibited_audio_access():
    """A *_environment audio path is a compliance violation, not a soft error."""
    from lifemate.data.schemas import Context, Sample
    from lifemate.errors import ProhibitedAccessError

    class AsrStub:
        name = "stub"

        def transcribe(self, path, sample_id=""):  # pragma: no cover - must not run
            raise AssertionError("transcribe must not be called on a prohibited path")

    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / "train_environment"  # compliance-scan: allow
        env_dir.mkdir()
        wav = env_dir / "a.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 160)
        sample = Sample(
            id="guarded",
            split="train",
            audio_path=wav,  # compliance-scan: allow
            audio_text="",
            context=Context(),
        )
        try:
            read_sample_audio(sample)
        except ProhibitedAccessError:
            pass
        else:  # pragma: no cover
            raise AssertionError("read_sample_audio accepted a *_environment path")

        infer = _load_infer("lifemate_infer_guarded")
        try:
            infer._transcribe(AsrStub(), sample)
        except ProhibitedAccessError as exc:
            assert "train_environment" in exc.path
        else:  # pragma: no cover
            raise AssertionError("_transcribe downgraded a ProhibitedAccessError")

def test_transcribe_propagates_backend_prohibited_access():
    """A backend that reaches a *_environment path must abort the run too.

    The audio-path guard is not enough: an ASR backend can open other banned
    files (model files, reference transcripts). ``_transcribe``'s generic
    ``except Exception`` used to turn that into an empty transcript — a
    compliance violation silently downgraded to a bad prediction.
    """
    from lifemate.data.schemas import Context, Sample
    from lifemate.errors import ProhibitedAccessError

    class AsrReachesBannedPath:
        name = "stub"

        def transcribe(self, path, sample_id=""):
            raise ProhibitedAccessError("data/train_environment/model.onnx")  # compliance-scan: allow

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 160)
        sample = Sample(
            id="backend-guarded", split="train", audio_path=wav, audio_text="", context=Context()
        )
        infer = _load_infer("lifemate_infer_backend_guarded")
        try:
            infer._transcribe(AsrReachesBannedPath(), sample)
        except ProhibitedAccessError as exc:
            assert "train_environment" in exc.path
        else:  # pragma: no cover
            raise AssertionError("_transcribe downgraded a backend ProhibitedAccessError")

def test_transcribe_missing_audio_reports_reason():
    from lifemate.data.schemas import Context, Sample

    class AsrStub:
        name = "stub"

    sample = Sample(
        id="no-audio", split="train", audio_path=None, audio_text="", context=Context()
    )
    infer = _load_infer("lifemate_infer_missing")
    text, info = infer._transcribe(AsrStub(), sample)
    assert text == ""
    assert info["error"] == "audio file missing for sample"
    assert info["backend"] == "stub"

# ---------------------------------------------------------------------------
# ASR failure accounting: a broken backend must not write a green submission
# ---------------------------------------------------------------------------
class _FailingAsr:
    """Raises on every call, like a backend with missing ONNX files."""

    name = "gipformer"

    def transcribe(self, audio_path, *, sample_id=None, lenient=False):
        raise RuntimeError(f"recognizer exploded for {sample_id}")


class _PartialAsr:
    """Fails for exactly one sample id, succeeds with text for the rest."""

    name = "gipformer"

    def __init__(self, failing_id):
        self.failing_id = failing_id

    def transcribe(self, audio_path, *, sample_id=None, lenient=False):
        from lifemate.asr.interface import TranscriptResult

        if sample_id == self.failing_id:
            raise RuntimeError("decoder produced no tokens")
        return TranscriptResult(text="Bật đèn phòng khách.", raw_text="Bật đèn phòng khách.", backend=self.name)


class _EmptyAsr:
    """Returns an empty string and no error: the silent-failure backend."""

    name = "gipformer"

    def transcribe(self, audio_path, *, sample_id=None, lenient=False):
        from lifemate.asr.interface import TranscriptResult

        return TranscriptResult(text="", raw_text="", backend=self.name)


def _run_infer_with_asr(backend, *, limit, output, extra_args=()):
    """Run ``run_inference`` with ``_build_asr`` replaced by ``backend``."""
    import contextlib
    import io

    infer = _load_infer(f"lifemate_infer_stub_{abs(hash((id(backend), str(output))))}")
    script = Path(tempfile.mkdtemp(prefix="lifemate_asr_script_")) / "script.json"
    payload = {}
    for sample in load_samples(FIXTURES, "train", limit=limit, with_labels=False):
        payload[sample.id] = ['<action>{"type": "finish"}</action>']
    script.write_text(json.dumps(payload), encoding="utf-8")

    infer._build_asr = lambda args, cfg, samples: backend  # type: ignore[assignment]
    argv = [
        "--split",
        "train",
        "--data-root",
        str(FIXTURES),
        "--use-mock-env",
        "--generator",
        f"dumpscript:{script}",
        "--asr",
        "gipformer",
        "--limit",
        str(limit),
        "--output",
        str(output),
        *extra_args,
    ]
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = infer.run_inference(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def test_asr_failing_every_sample_exits_three_without_submission():
    """Every sample failing ASR is a model problem: exit 3, no output file."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "submission.jsonl"
        code, _stdout, stderr = _run_infer_with_asr(_FailingAsr(), limit=3, output=output)
        assert code == 3, stderr
        assert not output.exists(), "a failing ASR backend must not leave a submission behind"
        assert "refusing to write a submission" in stderr
        assert "asr failed:" in stderr
        assert stderr.count("WARNING:") >= 3


def test_asr_failing_one_of_three_still_writes_and_reports():
    """A minority of ASR failures is visible in the summary, not fatal."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "submission.jsonl"
        backend = _PartialAsr("train_multi_step")
        code, stdout, stderr = _run_infer_with_asr(backend, limit=3, output=output)
        assert code == 0, stderr
        assert "1 ASR failure" in stdout
        rows = load_submission(output)
        assert len(rows) == 3
        report = validate_submission(output, [row["id"] for row in rows])
        assert report["ok"], report


def test_asr_empty_transcripts_without_error_exits_three():
    """Empty text with no error object is still a total ASR failure."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "submission.jsonl"
        code, _stdout, stderr = _run_infer_with_asr(_EmptyAsr(), limit=3, output=output)
        assert code == 3, stderr
        assert not output.exists()
        assert "no usable transcript" in stderr


# ---------------------------------------------------------------------------
# to_llm_text lives in the pipeline, applied once for every backend
# ---------------------------------------------------------------------------
def test_transcribe_normalizes_once_and_keeps_raw():
    from lifemate.asr.normalization import to_llm_text

    raw = "  Gọi   cho anh  Nam, 500.000đ "

    class AsrStub:
        name = "stub"

        def transcribe(self, audio_path, *, sample_id=None, lenient=False):
            from lifemate.asr.interface import TranscriptResult

            return TranscriptResult(text=raw, raw_text=raw, backend=self.name)

    sample = load_samples(FIXTURES, "train", limit=1, with_labels=False)[0]
    infer = _load_infer("lifemate_infer_normalize")
    text, info = infer._transcribe(AsrStub(), sample)
    assert text == to_llm_text(raw)
    assert text == to_llm_text(to_llm_text(raw))  # idempotent: applied exactly once
    assert info["raw_transcript"] == raw


def test_oracle_transcript_goes_through_the_same_normalization():
    from lifemate.asr.normalization import to_llm_text
    from lifemate.asr.oracle_backend import OracleTranscriptBackend

    raw = "  Đặt   báo thức  lúc 6 giờ 30  sáng mai. "
    sample = load_samples(FIXTURES, "train", limit=1, with_labels=False)[0]
    oracle = OracleTranscriptBackend({sample.id: raw})
    infer = _load_infer("lifemate_infer_oracle_norm")
    text, info = infer._transcribe(oracle, sample)
    assert text == to_llm_text(raw)
    assert info["raw_transcript"] == raw

# --------------------------------------------------------------------------- #
# official ToolEnv: per-split import, root convention, per-split app sets (Phase D)
# --------------------------------------------------------------------------- #
DATA_ROOT = ROOT / "data"
SPLITS_AVAILABLE = (DATA_ROOT / "train" / "zaic_tools").is_dir() and (DATA_ROOT / "public_test" / "zaic_tools").is_dir()

def test_official_env_imports_the_requested_splits_copy():
    """Each split ships its own zaic_tools with different apps: never share one."""
    if not SPLITS_AVAILABLE:
        return  # dataset not extracted for this checkout
    import sys as _sys

    def live_apps() -> set:
        """Apps of the module the env just imported (a stale `import` binding is the trap)."""
        return set(_sys.modules["zaic_tools"].APPS)

    env = create_environment("train", "train_0001", DATA_ROOT)
    assert Path(_sys.modules["zaic_tools"].__file__).resolve().parent == (
        DATA_ROOT / "train" / "zaic_tools"
    ).resolve()
    assert "robot_vacuum" in live_apps() and "ecommerce" not in live_apps()
    assert "sys.ask_missing_info" in env.available_tools()

    create_environment("public_test", "public_0002", DATA_ROOT)
    assert Path(_sys.modules["zaic_tools"].__file__).resolve().parent == (
        DATA_ROOT / "public_test" / "zaic_tools"
    ).resolve()
    assert "ecommerce" in live_apps() and "robot_vacuum" not in live_apps()

    # Going back must re-select the train copy (Python caches the first import).
    back = create_environment("train", "train_0001", DATA_ROOT)
    assert Path(_sys.modules["zaic_tools"].__file__).resolve().parent == (
        DATA_ROOT / "train" / "zaic_tools"
    ).resolve()
    assert "robot_vacuum" in live_apps()
    assert back.get_state_hash() == env.get_state_hash()

def test_official_env_root_must_contain_the_split_dir():
    """`data_root` is the dir CONTAINING <split>/; the split dir itself is wrong."""
    if not SPLITS_AVAILABLE:
        return
    from lifemate.errors import LifeMateError

    for wrong in (DATA_ROOT / "public_test", FIXTURES, "/nonexistent"):
        try:
            create_environment("train", "train_0001", wrong)
        except LifeMateError as exc:
            assert "data_root must be the" in str(exc) or "zaic_tools" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError(f"wrong root {wrong} was accepted")

def test_official_env_call_rejections_are_logged_and_matchn_gold():
    """A rejected call stays in get_tool_calls(); gold replay reproduces the transcript."""
    if not SPLITS_AVAILABLE:
        return
    env = create_environment("train", "train_0002", DATA_ROOT)
    rejected = env.call("sms.send", {})                       # missing required args
    assert rejected["ok"] is False and rejected.get("error")
    assert env.get_tool_calls() == [{"name": "sms.send", "arguments": {}}], "rejected calls must be logged"
    # the official env prefers the env's own validator: agreement, not a local guess
    assert env.validate("sms.send", {}).ok is False
    assert env.validate("sms.send", {"phone_number": "0900000000", "content": "hi"}).ok is True

def test_scripted_official_opt_in_and_official_output_barrier():
    """The pipeline-test opt-in must stay explicit and never reach the official path."""
    import importlib.util

    from lifemate.errors import LifeMateError

    spec = importlib.util.spec_from_file_location("lifemate_infer_optin", ROOT / "scripts" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    sys.modules["lifemate_infer_optin"] = infer
    spec.loader.exec_module(infer)

    opts = infer.parse_args(["--generator", "dumpscript:/tmp/x.json", "--allow-scripted-official"])
    infer._validate_generator_flag(opts)   # allowed only with the explicit flag
    try:
        infer._validate_generator_flag(infer.parse_args(["--generator", "dumpscript:/tmp/x.json"]))
    except LifeMateError as exc:
        assert "TEST HOOK" in str(exc)
    else:
        raise AssertionError("--generator accepted without any opt-in")

    occluded = directory = Path(tempfile.mkdtemp(prefix="lifemate_optin_"))
    code = infer.run_inference([
        "--split", "public_test", "--data-root", str(FIXTURES), "--asr", "none",
        "--generator", "dumpscript:/tmp/x.json", "--allow-scripted-official",
        "--limit", "1", "--output", "outputs/submission.jsonl",
    ])
    assert code == 4, "a scripted run must never write the official submission path"
    assert not Path("outputs/submission.jsonl").exists()

def test_official_preflight_creates_no_session():
    """ToolEnv may be built once per sample: the preflight imports the SDK only."""
    data_root = ROOT / "data"
    if not (data_root / "train" / "zaic_tools").is_dir():
        return
    from lifemate.env.official import OfficialToolEnv, check_official_sdk

    check_official_sdk("train", data_root)          # must not raise
    import zaic_tools

    built: list[str] = []
    session_cls = zaic_tools.ToolEnv

    def counting(split, sample_id, root=None):
        built.append(str(sample_id))
        return session_cls(split, sample_id, root=root)

    zaic_tools.ToolEnv = counting
    try:
        check_official_sdk("train", data_root)
        assert built == [], "preflight built a session"
        env = create_environment("train", "train_0001", data_root)
    finally:
        zaic_tools.ToolEnv = session_cls
    assert built == ["train_0001"] and isinstance(env, OfficialToolEnv)

def test_scripted_run_on_real_split_is_accepted_by_the_official_validator():
    """Acceptance: our pipeline's row (real env, real catalog) passes the organizers' CLI."""
    data_root = ROOT / "data"
    if not (data_root / "public_test" / "zaic_tools").is_dir():
        return
    from lifemate.agent.protocol import FINISH_ACTION, make_tool_call, render_action
    from lifemate.data.loader import load_catalog, load_samples

    catalog = load_catalog(data_root, "public_test")
    samples = load_samples(data_root, "public_test", limit=3)
    script = {}
    for sample in samples:
        installed = set(sample.context.installed_apps)
        read = next(
            (n for n, sp in sorted(catalog.tools.items())
             if sp.kind == "read" and sp.app_name in installed and not sp.required),
            None,
        )
        outs = ([render_action(make_tool_call(read))] if read else []) + [render_action(FINISH_ACTION)]
        script[sample.id] = outs

    directory = Path(tempfile.mkdtemp(prefix="lifemate_real_"))
    dump = directory / "dumps.json"
    dump.write_text(json.dumps(script), encoding="utf-8")
    output = directory / "sub.jsonl"
    infer = _load_infer("lifemate_infer_real")
    code = infer.run_inference([
        "--split", "public_test", "--data-root", str(data_root), "--asr", "none",
        "--generator", f"dumpscript:{dump}", "--allow-scripted-official",
        "--limit", "3", "--output", str(output), "--quiet",
    ])
    assert code == 0, code
    rows = load_submission(output)
    assert [r["id"] for r in rows] == [s.id for s in samples], "rows out of dataset order"
    assert all(len(r["state_hash"]) == 64 for r in rows)

    completed = subprocess.run(
        [sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", str(data_root),
         "--validate", str(output)],
        capture_output=True, text=True, timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(data_root / "public_test")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "0 schema errors" in completed.stdout, completed.stdout

# --------------------------------------------------------------------------- #
# ASR history transcripts (history audio -> text for a test split)
# --------------------------------------------------------------------------- #
def _history_fixture(tmpdir: Path) -> Path:
    """A tiny history-transcript JSONL valid for FIXTURES' train split."""
    path = tmpdir / "history_transcripts.jsonl"
    rows = [
        {"split": "train", "id": "train_single_tool", "turn_index": 1,
         "audio": "audio/train_single_tool/train_single_tool_u1.wav",
         "text": "MAI TÔI CÓ BÁO THỨC NÀO", "error": None},
        {"split": "train", "id": "train_single_tool", "turn_index": 2,
         "audio": "audio/train_single_tool/train_single_tool_u2.wav",
         "text": "xóa cái sớm nhất", "error": None},
        {"split": "train", "id": "train_missing_turn", "turn_index": 1,
         "audio": "audio/train_missing_turn/train_missing_turn_u1.wav",
         "text": "chỉ lượt này có", "error": None},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path

def _turn(role, **fields):
    from lifemate.data.schemas import HistoryTurn

    return HistoryTurn.from_raw({"role": role, **fields})

def test_history_texts_fill_a_blank_user_turn():
    """A user turn that has audio only gets the ASR text, verbatim (real shape)."""
    from lifemate.data.history import apply_history_texts, load_history_texts

    directory = Path(tempfile.mkdtemp(prefix="lifemate_hist_"))
    texts = load_history_texts(_history_fixture(directory), data_root=FIXTURES)
    turn = _turn("user", audio="audio/train_single_tool/train_single_tool_u1.wav")
    sample = type("S", (), {})()
    sample.id = "train_single_tool"
    sample.history = [turn]
    stats = apply_history_texts([sample], texts, "train")
    assert stats["filled"] == 1 and stats["missing"] == 0, stats
    assert turn.raw["audio_text"] == "MAI TÔI CÓ BÁO THỨC NÀO"
    assert turn.content == "MAI TÔI CÓ BÁO THỨC NÀO"

    from lifemate.agent.prompt_builder import build_decision_prompt

    prompt = build_decision_prompt(
        transcript="x", context=None, history=[turn], installed_apps=[], catalog=None, prior_steps=[]
    )
    assert "Người dùng: MAI TÔI CÓ BÁO THỨC NÀO" in prompt
    assert "Người dùng: \n" not in prompt

def test_history_texts_never_overwrite_gold_audio_text():
    """On train the gold `audio_text` wins; nothing is overwritten."""
    from lifemate.data.history import apply_history_texts, load_history_texts

    directory = Path(tempfile.mkdtemp(prefix="lifemate_hist2_"))
    texts = load_history_texts(_history_fixture(directory), data_root=FIXTURES)
    turn = _turn("user", audio="audio/train_single_tool/train_single_tool_u1.wav",
                 audio_text="Bản vàng của BTC")
    sample = type("S", (), {})()
    sample.id = "train_single_tool"
    sample.history = [turn]
    stats = apply_history_texts([sample], texts, "train")
    assert stats["filled"] == 0 and stats["existing"] == 1, stats
    assert turn.raw["audio_text"] == "Bản vàng của BTC"
    assert turn.content == "Bản vàng của BTC"

def test_history_texts_missing_turn_stays_blank():
    """A turn absent from the file renders blank: no invented text, counted as missing."""
    from lifemate.data.history import apply_history_texts, load_history_texts

    directory = Path(tempfile.mkdtemp(prefix="lifemate_hist3_"))
    texts = load_history_texts(_history_fixture(directory), data_root=FIXTURES)
    turns = [_turn("user", audio="audio/train_single_tool/train_single_tool_u1.wav"),
             _turn("user", audio="audio/train_single_tool/train_single_tool_u2.wav"),
             _turn("user", audio="audio/train_single_tool/train_single_tool_u3.wav")]
    sample = type("S", (), {})()
    sample.id = "train_single_tool"
    sample.history = turns
    stats = apply_history_texts([sample], texts, "train")
    assert stats["filled"] == 2 and stats["missing"] == 1, stats
    assert turns[2].raw.get("audio_text") is None
    assert turns[2].content == ""

def test_history_texts_audio_outside_the_split_audio_dir_is_refused():
    """A row pointing anywhere but <split>/audio/ is a data-integrity failure."""
    from lifemate.data.history import HistoryTranscriptError, load_history_texts

    directory = Path(tempfile.mkdtemp(prefix="lifemate_hist4_"))
    for bad in (
        # Built at runtime: a literal *_environment component is banned in source.
        "audio/../../" + "public_test/pub" + "lic_test_" + "environment/public_0002/state.json",
        "/etc/passwd",
        "other_split/audio/x.wav",
    ):
        path = directory / "bad.jsonl"
        path.write_text(json.dumps({"split": "train", "id": "train_single_tool", "turn_index": 1,
                                    "audio": bad, "text": "x"}) + "\n", encoding="utf-8")
        try:
            load_history_texts(path, data_root=FIXTURES)
        except (HistoryTranscriptError, ProhibitedAccessError) as exc:
            assert bad.split("/")[-1] in str(exc) or "audio" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError(f"audio path {bad!r} was accepted")

def test_history_transcripts_flag_absent_changes_nothing():
    """Without the flag the run behaves exactly as before, and says so loudly."""
    infer = _load_infer("lifemate_infer_nohist")
    args = infer.parse_args([])
    assert args.history_transcripts is None
    cfg = infer._build_config(args)
    assert cfg.inference.history_transcripts is None
    args = infer.parse_args(["--history-transcripts", "outputs/asr/history_transcripts.jsonl"])
    assert infer._build_config(args).inference.history_transcripts == "outputs/asr/history_transcripts.jsonl"

def test_history_texts_are_not_renormalised():
    """The file's text is already normalised: uppercase must survive untouched."""
    from lifemate.agent.prompt_builder import build_decision_prompt
    from lifemate.data.history import apply_history_texts, load_history_texts

    directory = Path(tempfile.mkdtemp(prefix="lifemate_hist5_"))
    texts = load_history_texts(_history_fixture(directory), data_root=FIXTURES)
    turn = _turn("user", audio="audio/train_single_tool/train_single_tool_u1.wav")
    sample = type("S", (), {})()
    sample.id = "train_single_tool"
    sample.history = [turn]
    apply_history_texts([sample], texts, "train")
    assert turn.content == "MAI TÔI CÓ BÁO THỨC NÀO", "text was normalised a second time"
    prompt = build_decision_prompt(
        transcript="x", context=None, history=[turn], installed_apps=[], catalog=None, prior_steps=[]
    )
    assert "MAI TÔI CÓ BÁO THỨC NÀO" in prompt

# --------------------------------------------------------------------------- #
# Blank history is refused by default (the container seam must not degrade)
# --------------------------------------------------------------------------- #
def _blank_history_split(tmpdir: Path) -> Path:
    """A data root whose ``public_test`` split has a history user turn with no text.

    Mirrors the real split: the test split ships history AUDIO only, so the turn
    parses with an empty ``content`` until a transcript fills it.
    """
    root = tmpdir / "data"
    split_dir = root / "public_test"
    (split_dir / "audio" / "blank_0001").mkdir(parents=True)
    (split_dir / "audio" / "blank_0001" / "blank_0001.wav").write_bytes(b"RIFF")
    record = {
        "id": "blank_0001",
        "category": "single_tool",
        "audio_path": "audio/blank_0001/blank_0001.wav",
        "context": {"installed_apps": ["smart_home"], "current_time": "2026-11-07T09:30:00+07:00"},
        "history": [
            {"role": "user", "audio": "audio/blank_0001/blank_0001_u1.wav"},
            {"role": "assistant", "content": "Dạ, em nghe."},
        ],
    }
    (split_dir / "public_test.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return root

def test_blank_history_fails_unless_explicitly_allowed():
    """No transcripts file + blank history turns => exit 2 before any model work.

    Silently blank history is the defect this guards: on the real split 419 of 634
    `public_test` samples lose their prior turns. Only ``--allow-missing-history``
    (a diagnostic opt-out) may continue.
    """
    directory = Path(tempfile.mkdtemp(prefix="lifemate_blankhist_"))
    data_root = _blank_history_split(directory)
    output = directory / "submission.jsonl"
    infer = _load_infer("lifemate_infer_blankhist")

    refused = infer.run_inference([
        "--split", "public_test", "--data-root", str(data_root), "--asr", "none",
        "--use-mock-env", "--limit", "1", "--output", str(output),
    ])
    assert refused == 2, refused
    assert not output.exists(), "a refused run must not leave a submission behind"

    script = directory / "dumpscript.json"
    script.write_text(json.dumps({"blank_0001": ['<action>{"type": "finish"}</action>']}), encoding="utf-8")
    allowed = infer.run_inference([
        "--split", "public_test", "--data-root", str(data_root), "--asr", "none",
        "--use-mock-env", "--generator", f"dumpscript:{script}",
        "--limit", "1", "--output", str(output),
        "--allow-missing-history",
    ])
    assert allowed == 0, allowed
    assert output.exists()

def test_uncovered_history_turns_from_a_partial_file_are_refused():
    """A transcripts file that misses a turn is partial application, not success."""
    directory = Path(tempfile.mkdtemp(prefix="lifemate_parthist_"))
    data_root = _blank_history_split(directory)
    partial = directory / "partial.jsonl"
    partial.write_text(
        json.dumps({"split": "train", "id": "train_single_tool", "turn_index": 1, "text": "x"}) + "\n",
        encoding="utf-8",
    )
    output = directory / "submission.jsonl"
    infer = _load_infer("lifemate_infer_parthist")

    refused = infer.run_inference([
        "--split", "public_test", "--data-root", str(data_root), "--asr", "none",
        "--use-mock-env", "--limit", "1", "--output", str(output),
        "--history-transcripts", str(partial),
    ])
    assert refused == 2, refused
    assert not output.exists()
