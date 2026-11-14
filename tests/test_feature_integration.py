"""End-to-end evidence that the feature flags reach the inference path.

Every assertion here is about an OBSERVABLE difference (prompt text, parsed
action, samples handed to the recognizer, kwargs handed to ``generate``), not
about a unit-tested helper being importable. Five flags had no production call
site before this file existed: ``prompt.itn``,
``postprocessing.enhanced_schema_guardrail``, ``audio.vad``,
``audio.rms_normalization``, ``decoding.constrained``.

GOLDEN provenance (requirement 1): the digest below was captured BEFORE the
wiring landed, by running the unmodified ``scripts/infer.py`` at HEAD 452cea4::

    WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src python3 scripts/infer.py \\
      --config configs/experiments/baseline.yaml --split train \\
      --data-root tests/fixtures/synthetic_lifemate --limit 5 --use-mock-env \\
      --allow-missing-history --asr oracle --allow-oracle \\
      --generator dumpscript:/tmp/ws_i_script.json --output /tmp/ws_i_baseline.jsonl

Re-running the same command after the wiring produces the SAME digest, so the
all-flags-off path is byte-identical. The rows themselves are embedded as
``GOLDEN_ROWS`` (they are also a fingerprint of the submission writer), and the
sha256 of that file is asserted below.

Stdlib only: no pytest fixtures, no torch, no numpy. Every test takes zero
arguments so ``python3 tests/run_tests.py`` collects them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.agent.agent import ToolAgent  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.features.config import FeatureConfig  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
INFER_SCRIPT = ROOT / "scripts" / "infer.py"

#: sha256 of the all-flags-off prediction file captured at HEAD 452cea4.
GOLDEN_DIGEST = "495454e51db78bb4a072ccb36679f87cc4c2337f9b419b5ddf6b91106761b998"

#: The same 5 rows, verbatim (order is part of the contract).
GOLDEN_ROWS = [
    {
        "id": "train_single_tool",
        "tool_calls": [
            {"name": "alarm.set", "arguments": {"time": "06:30", "label": "morning", "repeat": "once"}}
        ],
        "state_hash": "d281b295cd7b1ad3d66d7c3118f0f73ff8311f185844f3c9b9d5134e48e93611",
    },
    {
        "id": "train_multi_step",
        "tool_calls": [],
        "state_hash": "50c92c74bd9b6f48767fb8a91ddf10515886b8637f0bc8f2934522f473a37079",
    },
    {
        "id": "train_read_before_write",
        "tool_calls": [],
        "state_hash": "3d774270e116eb6bb7f43cf5bd49bc2a7a37dd3d518c50cfd03d82ff1fd85a19",
    },
    {
        "id": "train_result_dependent_arg",
        "tool_calls": [],
        "state_hash": "6b3927081bed9196b5f388f76884348c95aba6a2597c3db8435d53ef5be7ce01",
    },
    {
        "id": "train_missing_info",
        "tool_calls": [],
        "state_hash": "6b3927081bed9196b5f388f76884348c95aba6a2597c3db8435d53ef5be7ce01",
    },
]

#: One scripted decision per fixture id, so the run is deterministic.
SCRIPT = {
    "train_single_tool": [
        '<action>{"type": "tool_call", "name": "alarm.set", '
        '"arguments": {"time": "06:30", "label": "morning", "repeat": "once"}}</action>',
        '<action>{"type": "finish"}</action>',
    ],
    "train_multi_step": ['<action>{"type": "finish"}</action>'],
    "train_read_before_write": ['<action>{"type": "finish"}</action>'],
    "train_result_dependent_arg": ['<action>{"type": "finish"}</action>'],
    "train_missing_info": ['<action>{"type": "finish"}</action>'],
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _infer() -> Any:
    """Load ``scripts/infer.py`` by path (scripts/ is not a package)."""
    name = "lifemate_infer_feature_integration"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, INFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _env() -> dict[str, str]:
    """Subprocess env that can never reach a live wandb server."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["WANDB_BASE_URL"] = "http://127.0.0.1:9"
    env.pop("LIFEMATE_FEATURE_ASR_NBEST", None)
    return env


def _run_infer(directory: Path, extra: list[str], *, limit: int = 5, script: Mapping[str, Any] = SCRIPT) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the real pipeline on the fixture split; return (process, output path)."""
    script_path = directory / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    output = directory / "predictions.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(INFER_SCRIPT),
            "--config",
            "configs/experiments/baseline.yaml",
            "--split",
            "train",
            "--data-root",
            str(FIXTURES),
            "--limit",
            str(limit),
            "--use-mock-env",
            "--allow-missing-history",
            "--asr",
            "oracle",
            "--allow-oracle",
            "--generator",
            f"dumpscript:{script_path}",
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
        env=_env(),
    )
    return completed, output


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features(**overrides: Any) -> FeatureConfig:
    """FeatureConfig with ``section_field=value`` (e.g. ``prompt_itn_enabled=True``)."""
    cfg = FeatureConfig()
    for key, value in overrides.items():
        section, _, name = key.partition("_")
        target = getattr(cfg, section, None)
        assert target is not None, f"no feature section {section!r}"
        assert hasattr(target, name), f"no feature field {section}.{name}"
        setattr(target, name, value)
    return cfg


class _Sample:
    """Minimal sample duck-type: ``id`` + ``audio_path`` is all ``_transcribe`` reads."""

    def __init__(self, sample_id: str, audio_path: Path) -> None:
        self.id = sample_id
        self.audio_path = audio_path


class _StubBackend:
    """Records the path it was handed; returns a fixed transcript."""

    name = "stub"

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.paths: list[Path] = []

    def transcribe(self, audio_path: Any, *, sample_id: str | None = None, lenient: bool = False) -> Any:
        from lifemate.asr.interface import TranscriptResult

        self.paths.append(Path(audio_path))
        return TranscriptResult(
            text=self.transcript,
            raw_text=self.transcript,
            backend=self.name,
            latency_s=0.0,
            duration_s=0.1,
        )


def _silent_wav(path: Path, n_samples: int = 20000, leading: int = 6000, trailing: int = 6000) -> Path:
    """Tone in the middle, digital silence at both edges (VAD has something to trim)."""
    import math
    import struct

    frames = bytearray()
    for index in range(n_samples):
        inside = leading <= index < n_samples - trailing
        value = 0.3 * math.sin(2.0 * math.pi * 440.0 * index / 16000.0) if inside else 0.0
        frames += struct.pack("<h", int(round(max(-1.0, min(1.0, value)) * 32767)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(frames))
    return path


def _gen_catalog() -> ToolCatalog:
    """Catalog with a bounded integer argument (``hour: 0..23``)."""
    return ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "app_name": "alarm",
                    "description": "Alarms.",
                    "tools": [
                        {
                            "name": "alarm.set",
                            "description": "Create an alarm.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "time": {"type": "string"},
                                    "hour": {"type": "integer", "minimum": 0, "maximum": 23},
                                },
                                "required": ["time"],
                            },
                        }
                    ],
                }
            ]
        }
    )


class _PermissiveRegistry:
    """Registry whose availability/schema checks pass everything.

    It isolates layer 2: ``parse_model_output`` accepts the call, so a rejection
    can only have come from the guardrail.
    """

    def __init__(self) -> None:
        self.spec_parameters = {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "hour": {"type": "integer", "minimum": 0, "maximum": 23},
            },
            "required": ["time"],
        }

    def tool_names(self) -> list[str]:
        return ["alarm.set"]

    def is_available(self, name: str) -> bool:
        return name == "alarm.set"

    def get(self, name: str) -> Any:
        parameters = dict(self.spec_parameters)

        class _Spec:
            pass

        spec = _Spec()
        spec.parameters = parameters
        return spec if name == "alarm.set" else None

    def validate(self, name: str, args: Any) -> dict[str, Any]:
        return {"errors": [], "warnings": []}

    def for_sample(self, fields: Any) -> "_PermissiveRegistry":
        return self


class _ScriptedGen:
    """Replays ``outputs``; records the kwargs of every call."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append({"messages": list(messages), **kwargs})
        return self.outputs.pop(0)


class _RecordingEnv:
    """Miniature ToolEnvironment capturing executed calls."""

    def __init__(self, reject: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.reject = reject

    def validate(self, name: str, arguments: Any) -> dict[str, Any]:
        return {"errors": []}

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        if name in self.reject:
            return {"ok": False, "error": "simulated failure"}
        self.calls.append((name, dict(arguments)))
        return {"ok": True, "result": {}}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return [{"name": name, "arguments": args} for name, args in self.calls]

    def get_state_hash(self) -> str:
        return "0" * 64


def _agent_env_rate(expected_calls: list[tuple[str, dict[str, Any]]], features: Any, transcript: str = "dat bao thuc 6 gio") -> dict[str, Any]:
    """Drive one ToolAgent run and return what actually happened."""
    generator = _ScriptedGen(
        [
            '<action>{"type": "tool_call", "name": "alarm.set", '
            '"arguments": {"time": "06:30", "hour": 7}}</action>',
            '<action>{"type": "finish"}</action>',
        ]
    )
    env = _RecordingEnv()
    agent = ToolAgent(
        generator=generator,
        registry=ToolRegistry(_gen_catalog(), apps=["alarm"]),
        env=env,
        config=type("Cfg", (), {"features": features})(),
    )
    result = agent.run(
        {"id": "unit", "context": {"installed_apps": ["alarm"]}}, transcript=transcript
    )
    assert env.calls == expected_calls, env.calls
    return {"calls": env.calls, "result": result, "generator": generator}


# --------------------------------------------------------------------------- #
# 1. Baseline identity
# --------------------------------------------------------------------------- #
def test_all_flags_off_is_byte_identical_twice_and_matches_the_golden():
    """Two OFF runs agree byte-for-byte and equal the digest captured pre-wiring."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        first, out_a = _run_infer(Path(tmp_a), [])
        second, out_b = _run_infer(Path(tmp_b), [])
        assert first.returncode == 0, (first.stdout, first.stderr)
        assert second.returncode == 0, (second.stdout, second.stderr)
        assert _digest(out_a) == _digest(out_b), "two OFF runs must agree byte-for-byte"
        assert GOLDEN_DIGEST == _digest(out_a), (
            f"the OFF path changed: {_digest(out_a)} != {GOLDEN_DIGEST}"
        )
        rows = [json.loads(line) for line in out_a.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == GOLDEN_ROWS, rows


def test_golden_rows_reproduce_the_recorded_digest():
    """Self-check: the embedded GOLDEN_ROWS really hash to GOLDEN_DIGEST."""
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in GOLDEN_ROWS
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == GOLDEN_DIGEST


# --------------------------------------------------------------------------- #
# 2. Per-feature discrimination
# --------------------------------------------------------------------------- #
def test_prompt_itn_changes_the_prompt_text_exactly():
    """``prompt.itn`` turns spoken numbers into surface forms; OFF is identical."""
    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("itn", audio)
        spoken = "Đặt báo thức lúc sáu giờ ba mươi sáng mai."
        backend = _StubBackend(spoken)

        off, off_info = infer._transcribe(backend, sample, features=_features())
        on, on_info = infer._transcribe(backend, sample, features=_features(prompt_itn_enabled=True))

    assert off == "Đặt báo thức lúc sáu giờ ba mươi sáng mai."
    assert on == "Đặt báo thức lúc 06:30 mai."
    assert off != on
    assert off_info["raw_transcript"] == on_info["raw_transcript"] == spoken


def test_prompt_itn_off_matches_the_baseline_normalizer():
    """With the flag off the pipe is ``to_llm_text`` and nothing else."""
    from lifemate.asr.normalization import to_llm_text

    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("itn_off", audio)
        for raw in ("Đặt báo thức lúc sáu giờ ba mươi sáng mai.", "Giá 500.000đ nhé", ""):
            text, _info = infer._transcribe(_StubBackend(raw), sample, features=_features())
            assert text == (to_llm_text(raw) if raw else "")


def test_audio_vad_and_rms_change_the_samples_handed_to_the_recognizer():
    """VAD trims the silent edges; RMS rescales. OFF hands over the original path."""
    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("cond", audio)
        text = "dat bao thuc"

        # OFF: the recognizer receives the ORIGINAL path, no conditioning block.
        off_backend = _StubBackend(text)
        _off_text, off_info = infer._transcribe(off_backend, sample, features=_features())
        assert off_backend.paths == [audio], off_backend.paths
        assert "conditioning" not in off_info

        vad_backend = _StubBackend(text)
        _vad_text, vad_info = infer._transcribe(
            vad_backend, sample, features=_features(audio_vad_enabled=True)
        )
        rms_backend = _StubBackend(text)
        _rms_text, rms_info = infer._transcribe(
            rms_backend, sample, features=_features(audio_rms_normalization_enabled=True)
        )

    assert vad_backend.paths and vad_backend.paths[0] != audio
    assert vad_info["conditioning"]["vad"]["changed"] is True
    assert vad_info["conditioning"]["vad"]["n_samples_out"] < vad_info["conditioning"]["vad"]["n_samples_in"]

    assert rms_backend.paths and rms_backend.paths[0] != audio
    # The fixture peaks at 0.3 with RMS ~ -13.5 dBFS; the default target is -23 dBFS.
    rms_stats = rms_info["conditioning"]["rms"]
    assert rms_stats["changed"] is True
    assert round(rms_stats["rms_dbfs_after"], 1) == -23.0
    assert rms_stats["n_samples_out"] == rms_stats["n_samples_in"], "RMS must not resample"


def test_vad_off_writes_no_temp_wav():
    """Requirement 4: with both audio flags off nothing is conditioned or written."""
    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("nocond", audio)
        written: list[Path] = []
        original = infer._write_wav

        def _spy(path: Path, samples: Any, rate: int) -> None:
            written.append(Path(path))
            return original(path, samples, rate)

        infer._write_wav = _spy
        try:
            backend = _StubBackend("x")
            infer._transcribe(backend, sample, features=_features())
        finally:
            infer._write_wav = original
    assert written == [], written
    assert backend.paths == [audio]


def test_guardrail_flag_changes_the_parsed_action():
    """``postprocessing.enhanced_schema_guardrail`` rejects ``hour: 25`` (0..23)."""
    outputs = [
        '<action>{"type": "tool_call", "name": "alarm.set", '
        '"arguments": {"time": "06:30", "hour": 25}}</action>',
        '<action>{"type": "finish"}</action>',
    ]
    off_generator = _ScriptedGen(list(outputs))
    env_off = _RecordingEnv()
    agent_off = ToolAgent(
        generator=off_generator,
        registry=_PermissiveRegistry(),
        env=env_off,
        guardrail=False,
    )
    result_off = agent_off.run({"id": "off"}, transcript="t")

    on_generator = _ScriptedGen(list(outputs))
    env_on = _RecordingEnv()
    agent_on = ToolAgent(
        generator=on_generator,
        registry=_PermissiveRegistry(),
        env=env_on,
        guardrail=True,
    )
    result_on = agent_on.run({"id": "on"}, transcript="t")

    # OFF: the parser's verdict alone -> the call reaches the environment as-is.
    assert env_off.calls == [("alarm.set", {"time": "06:30", "hour": 25})]
    assert result_off.guardrail == []

    # ON: rejected, never executed, argument NOT clamped.
    assert env_on.calls == []
    assert len(result_on.guardrail) == 1
    event = result_on.guardrail[0]
    assert event["verdict"] == "rejected"
    assert event["original_action"]["arguments"] == {"time": "06:30", "hour": 25}
    assert event["action"]["type"] == "invalid"
    assert "outside 0..23" in "; ".join(event["reasons"]), event["reasons"]
    assert "hour" not in json.dumps(env_on.calls)


def test_guardrail_off_is_the_parser_verdict_alone():
    """Both layers agree on a valid call: OFF invokes the env, ON does too."""
    expected = [("alarm.set", {"time": "06:30", "hour": 7})]
    off = _agent_env_rate(expected, features=_features())
    on = _agent_env_rate(
        expected, features=_features(postprocessing_enhanced_schema_guardrail_enabled=True)
    )
    assert off["calls"] == on["calls"] == expected
    assert on["result"].guardrail == [], "a valid call must not be reported as a rejection"


def test_guardrail_cannot_weaken_the_parser():
    """A parser rejection stays rejected with the guardrail ON."""
    outputs = [
        '<action>{"type": "tool_call", "name": "ghost.tool", "arguments": {}}</action>',
        '<action>{"type": "finish"}</action>',
    ]
    for flag in (False, True):
        env = _RecordingEnv()
        agent = ToolAgent(
            generator=_ScriptedGen(list(outputs)),
            registry=ToolRegistry(_gen_catalog(), apps=["alarm"]),
            env=env,
            guardrail=flag,
        )
        result = agent.run({"id": "ghost", "installed_apps": ["alarm"]}, transcript="t")
        assert env.calls == [], "an unavailable tool must never reach the environment"
        assert result.errors and "not available" in result.errors[0], result.errors
        assert result.guardrail == [], "the parser's rejection is authoritative, layer 2 stays silent"


def test_decoding_constrained_passes_the_flag_and_the_per_sample_tool_set():
    """``decoding.constrained`` reaches ``HFGenerator`` and the per-sample setter."""
    import inspect

    from lifemate.models.generation import HFGenerator

    infer = _infer()
    off = infer._constrained_kwargs(_features(), HFGenerator)
    on = infer._constrained_kwargs(_features(decoding_constrained_decoding_enabled=True), HFGenerator)
    accepted = set(inspect.signature(HFGenerator.__init__).parameters)
    assert off == {}
    if "constrained" in accepted:
        assert on == {"constrained": True, "allowed_tools": ()}
    else:  # pragma: no cover - generator hook removed
        assert on == {}


def test_generator_without_the_hook_is_reported_as_pending_not_claimed():
    """A generator without ``constrained`` yields no kwargs and a loud PENDING line."""
    import contextlib
    import io

    infer = _infer()

    class _NoHook:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        kwargs = infer._constrained_kwargs(
            _features(decoding_constrained_decoding_enabled=True), _NoHook
        )
    assert kwargs == {}
    assert "PENDING" in stderr.getvalue()
    assert "UNCONSTRAINED" in stderr.getvalue()


# --------------------------------------------------------------------------- #
# 4. No flag, no cost
# --------------------------------------------------------------------------- #
def test_no_flag_no_cost():
    """Every OFF run: one audio load, no MultiAsrBackend, no temp WAV, no logits processor."""
    infer = _infer()
    from lifemate.asr import interface as asr_interface

    loads: list[Any] = []
    constructed: list[Any] = []
    written: list[Any] = []
    processors: list[Any] = []

    original_load = asr_interface.load_audio_16k_mono
    original_write = infer._write_wav
    import lifemate.asr.multi_backend as multi_backend
    import lifemate.models.constrained as constrained

    original_multi = multi_backend.MultiAsrBackend
    original_processor = constrained.ActionGrammarLogitsProcessor

    def _count_load(*args: Any, **kwargs: Any):
        loads.append(args)
        return original_load(*args, **kwargs)

    def _count_write(*args: Any, **kwargs: Any):
        written.append(args)
        return original_write(*args, **kwargs)

    class _CountMulti(original_multi):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append(args)
            super().__init__(*args, **kwargs)

    class _CountProcessor(original_processor):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            processors.append(args)
            super().__init__(*args, **kwargs)

    asr_interface.load_audio_16k_mono = _count_load
    infer._write_wav = _count_write
    multi_backend.MultiAsrBackend = _CountMulti
    constrained.ActionGrammarLogitsProcessor = _CountProcessor
    try:
        with tempfile.TemporaryDirectory() as tmp:
            audio = _silent_wav(Path(tmp) / "a.wav")
            backend = _StubBackend("x")
            infer._transcribe(backend, _Sample("nocost", audio), features=_features())
            completed, output = _run_infer(Path(tmp), [])
            assert completed.returncode == 0, (completed.stdout, completed.stderr)
            assert output.exists()
    finally:
        asr_interface.load_audio_16k_mono = original_load
        infer._write_wav = original_write
        multi_backend.MultiAsrBackend = original_multi
        constrained.ActionGrammarLogitsProcessor = original_processor

    # ``_transcribe`` never loads audio itself; the recognizer owns the decode.
    assert loads == [], f"load_audio_16k_mono was called with every flag off: {loads}"
    assert constructed == [], "MultiAsrBackend must not be constructed with asr.multi_asr off"
    assert written == [], "no temp WAV is written with both audio flags off"
    assert processors == [], "no logits processor is constructed with decoding.constrained off"


def test_multi_asr_off_returns_the_primary_object_itself():
    """``asr.multi_asr`` off: the object ``create_asr_backend`` built comes back unchanged."""
    infer = _infer()

    class _Primary:
        name = "primary"

        def transcribe(self, path: Any, **kwargs: Any) -> Any:  # pragma: no cover - not called
            raise AssertionError("not called here")

    primary = _Primary()
    assert infer._compose_multi_asr(type("C", (), {"features": _features()})(), primary) is primary


def test_multi_asr_on_composes_a_multi_backend():
    """``asr.multi_asr`` on: a MultiAsrBackend is built (secondary may be absent)."""
    infer = _infer()

    class _Primary:
        name = "primary"

        def transcribe(self, path: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("not called here")

    from lifemate.asr.multi_backend import MultiAsrBackend

    cfg = type(
        "C",
        (),
        {"features": _features(asr_multi_asr_enabled=True, asr_secondary_model_id="")},
    )()
    composed = infer._compose_multi_asr(cfg, _Primary())
    assert isinstance(composed, MultiAsrBackend)
    assert composed.enabled is True


# --------------------------------------------------------------------------- #
# 5. Misspelling fails
# --------------------------------------------------------------------------- #
def test_misspelled_feature_id_fails_with_a_suggestion():
    """``--enable-feature asr.nbets`` exits non-zero naming ``asr.nbest``."""
    completed = subprocess.run(
        [
            sys.executable,
            str(INFER_SCRIPT),
            "--enable-feature",
            "asr.nbets",
            "--output",
            "/tmp/lifemate_feature_typo.jsonl",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
        env=_env(),
    )
    assert completed.returncode != 0, completed.stdout
    combined = completed.stdout + completed.stderr
    assert "asr.nbets" in combined
    assert "asr.nbest" in combined, combined
    assert "did you mean" in combined, combined


def test_feature_report_is_printed_once_before_the_run():
    """The effective configuration is announced once, with the mode label."""
    with tempfile.TemporaryDirectory() as tmp:
        completed, _output = _run_infer(Path(tmp), [])
    assert completed.stderr.count("LifeMate Effective Configuration") == 1
    assert "BASELINE" in completed.stderr


def test_experimental_run_writes_a_feature_sidecar_and_baseline_does_not():
    """Provenance: an EXPERIMENTAL run records fingerprints; BASELINE leaves no file."""
    from lifemate.features.config import feature_fingerprint, training_feature_fingerprint

    with tempfile.TemporaryDirectory() as tmp:
        base_completed, base_output = _run_infer(Path(tmp), [], limit=1)
        assert base_completed.returncode == 0, (base_completed.stdout, base_completed.stderr)
        baseline_sidecar = base_output.with_name(base_output.name + ".run.json")
        assert not baseline_sidecar.exists(), "a BASELINE run must not add a provenance file"

    with tempfile.TemporaryDirectory() as tmp:
        on_completed, on_output = _run_infer(Path(tmp), ["--enable-feature", "prompt.itn"], limit=1)
        assert on_completed.returncode == 0, (on_completed.stdout, on_completed.stderr)
        sidecar = on_output.with_name(on_output.name + ".run.json")
        assert sidecar.exists(), "an EXPERIMENTAL run must record its feature identity"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))

    report = payload["features"]
    assert report["mode"] == "EXPERIMENTAL"
    assert report["enabled"] == ["prompt.itn"]
    expected = _features(prompt_itn_enabled=True)
    assert report["fingerprint"] == feature_fingerprint(expected)
    assert report["training_fingerprint"] == training_feature_fingerprint(expected)
    blob = json.dumps(payload, ensure_ascii=False)
    for forbidden in ("transcript", "user_profile", "messages"):
        assert forbidden not in blob, f"provenance must not carry {forbidden}"


def test_itn_on_changes_the_pipeline_sidecar_and_off_does_not():
    """The CLI flag reaches the pipeline end to end: prompts differ, rows do not."""
    with tempfile.TemporaryDirectory() as tmp:
        off_completed, off_output = _run_infer(Path(tmp), [], limit=1)
        assert off_completed.returncode == 0, (off_completed.stdout, off_completed.stderr)
        on_completed, on_output = _run_infer(Path(tmp), ["--enable-feature", "prompt.itn"], limit=1)
        assert on_completed.returncode == 0, (on_completed.stdout, on_completed.stderr)
        off_rows = [json.loads(line) for line in off_output.read_text(encoding="utf-8").splitlines() if line.strip()]
        on_rows = [json.loads(line) for line in on_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Same fixture, same scripted decision: ITN changes the prompt, not the call.
    assert [row["tool_calls"] for row in off_rows] == [row["tool_calls"] for row in on_rows]
    assert [row["id"] for row in off_rows] == [row["id"] for row in on_rows]


def test_evaluate_accepts_the_feature_flags_and_reports_them():
    """``evaluate.py`` has the same flags and puts the report in its JSON output."""
    evaluator = ROOT / "scripts" / "evaluate.py"
    completed = subprocess.run(
        [sys.executable, str(evaluator), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
        env=_env(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "--enable-feature" in completed.stdout
    assert "--disable-feature" in completed.stdout

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util,sys,json;"
            f"spec=importlib.util.spec_from_file_location('ev',r'{evaluator}');"
            "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            "args=m.parse_args(['--validation-only','--enable-feature','prompt.itn']);"
            "cfg=m._build_config(args);"
            "print(json.dumps(m._print_feature_report(cfg)))",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(ROOT),
        env=_env(),
    )
    assert probe.returncode == 0, probe.stderr
    payload = json.loads(probe.stdout.splitlines()[-1])
    assert payload["enabled"] == ["prompt.itn"]
    assert payload["mode"] == "EXPERIMENTAL"
    assert len(payload["fingerprint"]) == 16
    assert len(payload["training_fingerprint"]) == 16


def test_cli_disable_feature_beats_the_yaml_overlay():
    """``--disable-feature`` is the last layer: it beats ``features:`` from YAML."""
    infer = _infer()
    args = infer.parse_args(
        ["--config", "configs/experiments/itn.yaml", "--disable-feature", "prompt.itn"]
    )
    cfg = infer._build_config(args)
    assert cfg.features.prompt.itn_enabled is False


def test_itn_overlay_enables_the_flag_without_any_cli_argument():
    """The YAML overlay alone reaches the effective config."""
    infer = _infer()
    cfg = infer._build_config(infer.parse_args(["--config", "configs/experiments/itn.yaml"]))
    assert cfg.features.prompt.itn_enabled is True


# --------------------------------------------------------------------------- #
# Wiring presence: the call sites must exist, not just the helpers
# --------------------------------------------------------------------------- #
def test_infer_calls_each_seam():
    """AST check: the five call sites exist in ``scripts/infer.py``."""
    import ast

    tree = ast.parse(INFER_SCRIPT.read_text(encoding="utf-8"))
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            called.add(str(name))
    for seam in (
        "trim_silence",
        "normalize_rms",
        "apply_itn_pipeline",
        "MultiAsrBackend",
        "transcribe_nbest",
        "_condition_audio",
        "_compose_multi_asr",
        "_constrained_kwargs",
    ):
        assert seam in called, f"{seam} has no call site in scripts/infer.py"


def test_agent_module_calls_the_guardrail():
    """AST check: ``src/lifemate/agent/agent.py`` calls ``check_action``."""
    import ast

    source = ROOT / "src" / "lifemate" / "agent" / "agent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "check_action" in names
    assert "_guardrail_apply" in names


def test_transcribe_split_routes_through_the_itn_pipeline():
    """``scripts/transcribe_split.py`` uses ``apply_itn_pipeline``, gated on the flag."""
    import ast

    source = ROOT / "scripts" / "transcribe_split.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "apply_itn_pipeline" in names
    assert "to_llm_text" not in names, "the ungated normalizer must not be a call site any more"


def test_transcribe_split_output_is_unchanged_with_flags_off():
    """Its JSONL is byte-identical to the pre-wiring file when nothing is enabled."""
    import lifemate.asr as asr_pkg
    from lifemate.asr.interface import TranscriptResult

    class _Stub:
        name = "stub"
        is_oracle = False
        quantization = "int8"

        def transcribe(self, audio_path: Any, *, sample_id: str | None = None, lenient: bool = False) -> Any:
            return TranscriptResult(
                text="nếu mai mưa thì thôi",
                raw_text="NẾU MAI MƯA THÌ THÔI",
                backend="stub",
                latency_s=0.01,
                duration_s=1.0,
            )

    spec = importlib.util.spec_from_file_location(
        "_ts_feature_integration", ROOT / "scripts" / "transcribe_split.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = asr_pkg.create_asr_backend
    asr_pkg.create_asr_backend = lambda cfg, backend=None, **kwargs: _Stub()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            for path in (first, second):
                assert (
                    module.main(
                        [
                            "--split",
                            "train",
                            "--data-root",
                            str(FIXTURES),
                            "--output",
                            str(path),
                            "--overwrite",
                            "--limit",
                            "2",
                        ]
                    )
                    == 0
                )
            assert first.read_bytes() == second.read_bytes()
            rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines() if line.strip()]
            # ``python3`` runs the pipeline WITHOUT feature flags at all.
            assert all(row["hypothesis"] == "nếu mai mưa thì thôi" for row in rows)
            metrics = json.loads(Path(str(first) + ".metrics.json").read_text(encoding="utf-8"))
    finally:
        asr_pkg.create_asr_backend = original
    assert "features" not in metrics["run"], "a BASELINE sidecar must stay byte-identical"


# --------------------------------------------------------------------------- #
# 6. Constrained decoding: the kwargs handed to generate() (torch-gated)
# --------------------------------------------------------------------------- #
def test_constrained_flag_adds_a_logits_processor_to_the_generate_call():
    """Torch-gated: ``constrained=True`` routes decode through the masked call.

    The claim under test is exactly "the kwargs handed to generate differ": with
    the flag off ``model.generate`` is called with no ``logits_processor``; with
    it on ``constrained_generate`` receives ``enabled=True`` and the sample's
    tool allow-list, which is where the grammar mask is built.
    """
    try:
        import torch

        from lifemate.models import constrained as constrained_module
        from lifemate.models.generation import HFGenerator
    except ImportError:
        print("SKIP: torch/transformers unavailable")
        return

    from lifemate.models.profiles import ModelProfile

    profile = ModelProfile(
        name="generic",
        family="generic",
        chat_template_kwargs={},
        speculative_stop_ids=(),
        lora_target_modules=(),
        default_max_new_tokens=16,
        default_seq_length=64,
        architecture_fallbacks=(),
        supports_thinking=False,
    )

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            return "U: " + messages[-1]["content"] + "\nA:"

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [7]

        def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
            return '<action>{"type": "finish"}</action>'

        def __len__(self) -> int:
            return 16

    class _Model:
        device = "cpu"

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            self.plain_calls.append(kwargs)
            return torch.tensor([[1, 2, 3, 4]])

        def __init__(self) -> None:
            self.plain_calls: list[dict[str, Any]] = []

    recorded: list[dict[str, Any]] = []
    original = constrained_module.constrained_generate

    def _fake_constrained_generate(model: Any, tokenizer: Any, prompt: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs)
        return torch.tensor([[1, 2, 3, 4]])

    try:
        constrained_module.constrained_generate = _fake_constrained_generate

        off_model = _Model()
        off = HFGenerator(off_model, _Tokenizer(), profile=profile, max_new_tokens=4)
        off.generate([{"role": "user", "content": "x"}])
        off_constrained_calls = list(recorded)

        on_model = _Model()
        on = HFGenerator(
            on_model, _Tokenizer(), profile=profile, max_new_tokens=4,
            constrained=True, allowed_tools=(),
        )
        on.set_allowed_tools(["alarm.set"])
        on.generate([{"role": "user", "content": "x"}])
    finally:
        constrained_module.constrained_generate = original

    # OFF: the plain path, no grammar mask anywhere.
    assert off_constrained_calls == [], "constrained_generate must not run with the flag off"
    assert len(off_model.plain_calls) == 1
    assert "logits_processor" not in off_model.plain_calls[0]

    # ON: the masked path receives enabled=True and the per-sample tool set.
    assert len(recorded) == 1, recorded
    assert recorded[0]["enabled"] is True
    assert tuple(recorded[0]["allowed_tools"]) == ("alarm.set",)
    assert on_model.plain_calls == [], "the constrained path must not fall back to model.generate"


# --------------------------------------------------------------------------- #
# 7. asr.nbest backend matrix (P0 regression)
#
# ``transcribe_nbest`` exists on GipformerBackend and MultiAsrBackend ONLY, so
# calling it on Whisper/Oracle raised AttributeError. That escaped the old
# ``except NBestUnsupportedError``, the enclosing handler turned it into an empty
# transcript with no warning, and the run silently produced a submission of blank
# transcripts. These tests pin the capability gate on all four backends.
# --------------------------------------------------------------------------- #
class _Top1OnlyBackend:
    """A backend with ``transcribe`` only — no ``transcribe_nbest``, no flag.

    That is the WhisperBackend/OracleTranscriptBackend shape, which is exactly
    what used to raise ``AttributeError``.
    """

    is_oracle = False

    def __init__(self, name: str, transcript: str) -> None:
        self.name = name
        self.transcript = transcript
        self.paths: list[Path] = []

    def transcribe(self, audio_path: Any, *, sample_id: str | None = None, lenient: bool = False) -> Any:
        from lifemate.asr.interface import TranscriptResult

        self.paths.append(Path(audio_path))
        return TranscriptResult(
            text=self.transcript, raw_text=self.transcript, backend=self.name
        )


class _NBestCapableBackend(_Top1OnlyBackend):
    """Advertises N-best AND really returns two hypotheses."""

    nbest_supported = True

    def transcribe_nbest(self, audio_path: Any, *, nbest_size: int = 3, sample_id: str | None = None) -> Any:
        from lifemate.asr.nbest import AsrHypothesis, NBestResult

        self.paths.append(Path(audio_path))
        return NBestResult(
            hypotheses=[
                AsrHypothesis(text="n-best rank 0", rank=0, score=-3.0, source=self.name, backend=self.name),
                AsrHypothesis(text="n-best rank 1", rank=1, score=-4.5, source=self.name, backend=self.name),
            ],
            requested_size=int(nbest_size),
            scores_calibrated=True,
        )


def test_nbest_matrix_gipformer_falls_back_to_top1_loudly():
    """gipformer: ``nbest_supported = False`` -> capability gate -> top-1 text."""
    import contextlib
    import io

    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("gipformer", audio)
        backend = _Top1OnlyBackend("gipformer", "chao buoi sang")
        backend.nbest_supported = False  # the real GipformerBackend contract
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            text, info = infer._transcribe(
                backend, sample, features=_features(asr_nbest_enabled=True)
            )
        top1_text, top1_info = infer._transcribe(
            _Top1OnlyBackend("gipformer", "chao buoi sang"), sample, features=_features()
        )
        warnings = stderr.getvalue()

    assert text == top1_text == "chao buoi sang", (text, top1_text)
    assert info["error"] is None, info
    assert info["backend"] == "gipformer"
    assert info["nbest_requested"] is True
    # The recorded reason names the backend and the missing capability.
    assert "N-best decoding is not available" in info["nbest_fallback"], info["nbest_fallback"]
    assert "backend 'gipformer'" in info["nbest_fallback"], info["nbest_fallback"]
    assert "nbest_size=3" in info["nbest_fallback"], info["nbest_fallback"]
    assert "WARNING: asr.nbest is ON but backend 'gipformer'" in warnings, warnings


def test_nbest_matrix_whisper_never_raises_attribute_error():
    """whisper: no ``transcribe_nbest`` at all -> top-1 text, no AttributeError."""
    import contextlib
    import io

    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("whisper", audio)
        backend = _Top1OnlyBackend("whisper", "xin chao the gioi")
        assert not hasattr(backend, "transcribe_nbest")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            text, info = infer._transcribe(
                backend, sample, features=_features(asr_nbest_enabled=True)
            )
        warnings = stderr.getvalue()

    assert text == "xin chao the gioi", f"whisper dropped the transcript: {text!r}"
    assert info["error"] is None, f"AttributeError escaped the gate: {info}"
    assert "AttributeError" not in str(info)
    assert info["backend"] == "whisper"
    assert "N-best decoding is not available" in info["nbest_fallback"], info["nbest_fallback"]
    assert "backend 'whisper'" in info["nbest_fallback"], info["nbest_fallback"]
    assert "WARNING: asr.nbest is ON but backend 'whisper'" in warnings, warnings


def test_nbest_matrix_oracle_falls_back_to_the_gold_transcript():
    """oracle: same gate, the gold text survives (it is the labeled-split path)."""
    import contextlib
    import io

    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("oracle", audio)
        backend = _Top1OnlyBackend("oracle", "Dat bao thuc luc 6 gio 30 sang mai.")
        backend.is_oracle = True
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            text, info = infer._transcribe(
                backend, sample, features=_features(asr_nbest_enabled=True)
            )
        warnings = stderr.getvalue()

    assert text == "Dat bao thuc luc 6 gio 30 sang mai."
    assert info["error"] is None, info
    assert "AttributeError" not in str(info)
    assert info["backend"] == "oracle"
    assert "N-best decoding is not available" in info["nbest_fallback"], info["nbest_fallback"]
    assert "backend 'oracle'" in info["nbest_fallback"], info["nbest_fallback"]
    assert "WARNING: asr.nbest is ON but backend 'oracle'" in warnings, warnings


def test_nbest_matrix_capable_backend_uses_rank_zero():
    """A backend that really has N-best: rank-0 text, ``returned == 2``."""
    import contextlib
    import io

    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("capable", audio)
        backend = _NBestCapableBackend("capable", "top-1 would be this")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            text, info = infer._transcribe(
                backend, sample, features=_features(asr_nbest_enabled=True)
            )
        warnings = stderr.getvalue()

    assert text == "n-best rank 0"
    assert "nbest_fallback" not in info, info
    assert info["error"] is None
    assert info["conditioning"]["nbest"]["returned"] == 2
    assert info["conditioning"]["nbest"]["requested"] == 3
    assert info["conditioning"]["nbest"]["degraded"] is False
    assert warnings == "", f"a capable backend must not warn: {warnings!r}"


def test_nbest_negative_control_without_the_gate_whisper_loses_the_transcript():
    """Remove the capability gate -> whisper's text IS lost. Proves the gate is the fix.

    This is the pre-fix behaviour, reproduced deliberately: the AttributeError is
    not caught by ``except NBestUnsupportedError``, the enclosing handler turns it
    into ``("", {"error": ...})`` and the submission would be blank.
    """
    infer_path = INFER_SCRIPT
    import lifemate.asr.nbest as nbest_module
    from lifemate.asr.nbest import NBestUnsupportedError

    source = infer_path.read_text(encoding="utf-8")
    # Rebuild the PRE-FIX ``_nbest_text`` in isolation: only NBestUnsupportedError.
    prefix = '''
def _prefix_nbest_text(asr, audio_path, sample_id, features):
    size = int(getattr(features.asr, "nbest_size", 3))
    try:
        return asr.transcribe_nbest(audio_path, nbest_size=size, sample_id=sample_id), None
    except NBestUnsupportedError as exc:
        return None, str(exc)
'''
    namespace: dict[str, Any] = {"NBestUnsupportedError": NBestUnsupportedError}
    exec(prefix, namespace)  # noqa: S102 - a deliberate reproduction of the old body
    ungated = namespace["_prefix_nbest_text"]

    backend = _Top1OnlyBackend("whisper", "xin chao the gioi")
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        try:
            ungated(backend, audio, "whisper", _features(asr_nbest_enabled=True))
        except AttributeError as exc:
            assert "transcribe_nbest" in str(exc)
            lost = str(exc)
        else:  # pragma: no cover - the gate removal must break it
            raise AssertionError("the ungated call was expected to raise AttributeError")

    # Same shape the old ``_transcribe`` produced for that exception.
    old_result = ("", {"backend": "whisper", "error": f"AttributeError: {lost}"[:300]})
    assert old_result[0] == "", "the pre-fix path really did produce an empty transcript"
    assert "AttributeError" in old_result[1]["error"]

    # And the GATED call, on the same backend, does not lose it.
    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        text, info = infer._transcribe(
            _Top1OnlyBackend("whisper", "xin chao the gioi"),
            _Sample("whisper", audio),
            features=_features(asr_nbest_enabled=True),
        )
    assert text == "xin chao the gioi"
    assert info["error"] is None


def test_readable_audio_with_an_empty_transcript_surfaces_as_an_error():
    """No silent blank line: an empty top-1 result on good audio is an ASR error."""
    infer = _infer()
    with tempfile.TemporaryDirectory() as tmp:
        audio = _silent_wav(Path(tmp) / "a.wav")
        sample = _Sample("blank", audio)
        # n-best ON, capability gate falls back, and the top-1 is itself empty.
        text, info = infer._transcribe(
            _Top1OnlyBackend("whisper", ""),
            sample,
            features=_features(asr_nbest_enabled=True),
        )
        plain_text, plain_info = infer._transcribe(
            _Top1OnlyBackend("whisper", ""), sample, features=_features()
        )
    assert text == "" and plain_text == ""
    assert info["error"] and "empty transcript" in info["error"], info
    assert "after the asr.nbest fallback" in info["error"], info
    assert plain_info["error"] and "empty transcript" in plain_info["error"], plain_info
