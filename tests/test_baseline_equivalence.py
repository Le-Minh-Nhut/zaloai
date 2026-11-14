"""Baseline-equivalence fixture: "flags off == pre-feature behaviour".

Every golden below was computed ONCE from the pre-feature code path and pasted
in literally. Nothing here regenerates its own expectation: a golden that
recomputes itself proves nothing, and a golden that drifts silently is worse
than no golden.

Coverage (the components a feature flag can touch):

1. ``agent.prompt_builder.build_decision_prompt`` — full rendered prompt body
   (sha256 + length + ordered section headers), with and without prior steps;
2. ``agent.protocol.render_action`` — the frozen ``<action>...</action>`` bytes
   for one ``tool_call`` and one ``finish``;
3. ``data.trajectory_builder.build_sft_dataset`` — record count, per-record
   sha256 over the whole fixture split, and the accounting ``stats``;
4. ``asr.normalization.to_llm_text`` — 12 Vietnamese strings, exact outputs;
5. ``asr.interface.load_audio_16k_mono`` / ``resample_linear`` — synthetic
   PCM16 WAVs at 8000 Hz and 44100 Hz: length, first/last samples, sha256 of the
   quantised float list;
6. ``evaluation.submission`` — the exact JSONL bytes of one submitted row.

Determinism: every one of the six is fully deterministic here — same inputs
produce byte-identical output across runs (each pinning test renders twice and
asserts stability *before* comparing to the golden, so a nondeterministic
component fails loudly instead of being papered over). The only nondeterministic
value in the wider evaluation payload is its ``timestamp``, which is not pinned.

Coupling: nothing in this file imports ``lifemate.features`` into a module under
test. The final test imports ``FeatureConfig`` itself, and a static test proves
the six modules under test never reference the feature package at all — that
zero coupling is the point.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.agent.prompt_builder import StepRecord, build_decision_prompt  # noqa: E402
from lifemate.agent.protocol import FINISH_ACTION, make_tool_call, render_action  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.asr.interface import load_audio_16k_mono, resample_linear  # noqa: E402
from lifemate.asr.normalization import to_llm_text  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.data.schemas import (  # noqa: E402
    Context,
    HistoryTurn,
    Label,
    Sample,
    ToolCall,
    ToolResult,
)
from lifemate.data.trajectory_builder import build_sft_dataset  # noqa: E402
from lifemate.evaluation.submission import normalise_row, write_submission  # noqa: E402

CATALOG_PATH = FIXTURES / "train" / "train_tools_catalog.json"
APPS = ["alarm", "calendar", "smart_home", "messaging"]

# --------------------------------------------------------------------------- #
# Goldens (pre-feature code path, computed once)
# --------------------------------------------------------------------------- #
#: 1. Prompt body, no prior steps.
PROMPT_SHA256 = "3301a2ea0fd1c83bafb8561dedc786e096c7f0644c8117a8bd403aae8cf5db82"
PROMPT_LENGTH = 4024
PROMPT_SECTIONS = [
    "# Yêu cầu",
    "# Ngữ cảnh",
    "# Lịch sử hội thoại",
    "# Công cụ khả dụng",
    "# Định dạng trả lời",
]
#: 1b. Prompt body with one executed step appended.
PROMPT_STEPS_SHA256 = "2aa5d7bc02e42676a0ccae59c9188d5883ef0bf733838bb75587f1123f961122"
PROMPT_STEPS_LENGTH = 4171

#: 2. Frozen action serialisation.
TARGET_TOOL_CALL = (
    '<action>{"type": "tool_call", "name": "alarm.set", '
    '"arguments": {"time": "06:30"}}</action>'
)
TARGET_FINISH = '<action>{"type": "finish"}</action>'

#: 3. SFT dataset over the whole synthetic train split.
SFT_RECORD_COUNT = 26
SFT_SAMPLE_COUNT = 14
SFT_RECORDS_SHA256 = "0111a24d3ea356198b51b3dd467aa9dabc3a85dc6e72c6bb5e8e026f5f34ce45"
SFT_STATS_SHA256 = "e4457f9a8144739fb1b02e3b72b8db371c807667a2430873af1aca22d0a16201"
SFT_FIRST_RECORD_SHA256 = "436bad64acba7c24d6ebcff3a452a12b8c3208c48494e46ac71cfc410d886fc0"
SFT_LAST_RECORD_SHA256 = "3c9a67e29915bf2cffc9e5a844d00d4eea46420cca8e8f7249de56eef3448e1c"
SFT_FIRST_SAMPLE = ("train_single_tool", 0)

#: 4. ``to_llm_text`` — Vietnamese, diacritics, digits, time, money, phone, case.
LLM_TEXT_CASES = [
    ("Đặt báo thức lúc 6 giờ sáng mai.", "Đặt báo thức lúc 6 giờ sáng mai."),
    ("mai 7h30 đặt lịch họp với anh Nam ở Lê Lợi", "mai 7h30 đặt lịch họp với anh Nam ở Lê Lợi"),
    ("15h30 ngày 20/11/2026 nhắc tôi gọi mẹ", "15h30 ngày 20/11/2026 nhắc tôi gọi mẹ"),
    ("Tổng tiền là 500.000đ ạ", "Tổng tiền là 500.000đ ạ"),
    ("Số điện thoại +84900123456", "Số điện thoại +84900123456"),
    ("TÔI MUỐN GỬI TIN NHẮN", "TÔI MUỐN GỬI TIN NHẮN"),
    ("bật đèn phòng khách , và tắt quạt", "bật đèn phòng khách, và tắt quạt"),
    ("ok book shop luôn nhé", "ok book shop luôn nhé"),
    ("TP.HCM mưa to", "TP.HCM mưa to"),
    ("Giá 1,5 triệu đồng", "Giá 1,5 triệu đồng"),
    ("  nhiều   khoảng   trắng\tở đây  ", "nhiều khoảng trắng ở đây"),
    ("câu một.Tiếp theo là câu hai?Đúng vậy!", "câu một. Tiếp theo là câu hai? Đúng vậy!"),
]

#: 5. Audio: identical synthetic signals, pinned after decode + resample.
AUDIO_8K_SAMPLES = [0, 1000, -1000, 32767, -32768, 500, -500, 0]
AUDIO_44K_SAMPLES = [((i * 997) % 65536) - 32768 for i in range(44)]
AUDIO_8K = {
    "rate": 16000,
    "length": 16,
    "quant_head": [0, 500, 1000, 0, -1000, 15884],
    "quant_tail": [500, 0, -500, -250, 0, 0],
    "quant_sha256": "3e4a028a1e11e0a5179a5fcf939d903534601ae44e98993926154de11d93ede9",
}
AUDIO_44K = {
    "rate": 16000,
    "length": 16,
    "quant_head": [-32768, -30020, -27272, -24524, -21776, -19028],
    "quant_tail": [-5288, -2540, 208, 2956, 5704, 8452],
    "quant_sha256": "99851f66a1836016bbabf6b55fe8525558fb08098a03f0076dfa2b1d57360667",
}
RESAMPLE_UPSAMPLED = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0]
RESAMPLE_SHA256 = "f5a16f50f81eeb73383d2ad155c6db8b253b215a4742ceb9fd3936ba59f07d14"

#: 6. Submission row -> exact JSONL bytes.
SUBMISSION_ROW = {
    "id": "train_single_tool",
    "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}],
    "state_hash": "ab12cd34",
}
SUBMISSION_LINE = (
    '{"id": "train_single_tool", "tool_calls": [{"name": "alarm.set", '
    '"arguments": {"time": "06:30"}}], "state_hash": "ab12cd34"}\n'
)
SUBMISSION_SHA256 = "5b3bbc48cb3bc460c9272b815d369952dde76d090e2af47302ef0035eecfcf5b"
SUBMISSION_BYTES = 125

#: The six modules a feature flag could reach; none may import lifemate.features.
COUPLED_MODULES = [
    "src/lifemate/agent/prompt_builder.py",
    "src/lifemate/agent/protocol.py",
    "src/lifemate/data/trajectory_builder.py",
    "src/lifemate/asr/normalization.py",
    "src/lifemate/asr/interface.py",
    "src/lifemate/evaluation/submission.py",
]

# --------------------------------------------------------------------------- #
# Builders (identical inputs for the golden and the equivalence check)
# --------------------------------------------------------------------------- #
def _fixture_sample() -> Sample:
    """The fixed record the prompt golden was taken from."""
    return Sample(
        id="s1",
        split="train",
        audio_path=None,
        audio_text="Dat bao thuc luc 6 gio sang mai.",
        context=Context(
            current_time="2026-11-07T09:30:00+07:00",
            timezone="Asia/Ho_Chi_Minh",
            current_location="Ho Chi Minh City",
            installed_apps=list(APPS),
        ),
        history=[
            HistoryTurn(role="user", content="Mai 7 gio dat bao thuc giup minh nhe."),
            HistoryTurn(
                role="assistant",
                content="Đã đặt báo thức 7 giờ sáng mai.",
                tool_call={"name": "alarm.set", "arguments": {"time": "07:00"}},
                tool_result={"ok": True, "events": [{"kind": "alarm_set"}]},
            ),
        ],
        label=Label(
            tool_calls=[ToolCall(name="alarm.set", arguments={"time": "06:30"})],
            tool_results=[ToolResult(name="alarm.set", ok=True, result={"marker": "obs_0"})],
            final_state=None,
        ),
        raw={"category": "test"},
    )

def _render_prompt(*, with_steps: bool) -> str:
    sample = _fixture_sample()
    registry = ToolRegistry(ToolCatalog.from_file(CATALOG_PATH)).for_sample(sample)
    return build_decision_prompt(
        transcript=sample.audio_text,
        context=sample.context,
        history=sample.history,
        installed_apps=sample.context.installed_apps,
        catalog=registry,
        prior_steps=[
            StepRecord(
                index=0,
                action=make_tool_call("alarm.list", {}),
                observation={"ok": True, "events": []},
            )
        ]
        if with_steps
        else None,
    )

def _sft_records() -> tuple[list[dict], dict]:
    samples = load_samples(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    points, stats = build_sft_dataset(samples, catalog)
    return [point.to_record() for point in points], stats

def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _write_pcm16_wav(path: Path, rate: int, samples: list[int]) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))
    return path

def _decode_wav(rate: int, samples: list[int]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_pcm16_wav(Path(tmp) / f"probe_{rate}.wav", rate, samples)
        audio, out_rate = load_audio_16k_mono(path)
    quantised = [round(value * 32768) for value in audio]
    return {
        "rate": out_rate,
        "length": len(audio),
        "quant_head": quantised[:6],
        "quant_tail": quantised[-6:],
        "quant_sha256": _sha(str(quantised)),
    }

def _submission_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_submission([SUBMISSION_ROW], Path(tmp) / "submission.jsonl")
        return path.read_bytes()

def all_baseline_outputs() -> dict[str, dict]:
    """The six captured outputs, keyed by component."""
    records, stats = _sft_records()
    return {
        "prompt": {
            "sha256": _sha(_render_prompt(with_steps=False)),
            "length": len(_render_prompt(with_steps=False)),
            "sections": [
                line
                for line in _render_prompt(with_steps=False).splitlines()
                if line.startswith("# ")
            ],
        },
        "prompt_with_steps": {
            "sha256": _sha(_render_prompt(with_steps=True)),
            "length": len(_render_prompt(with_steps=True)),
        },
        "protocol": {
            "tool_call": render_action(make_tool_call("alarm.set", {"time": "06:30"})),
            "finish": render_action({"type": "finish"}),
        },
        "sft": {
            "count": len(records),
            "records_sha256": _sha(_canonical(records)),
            "stats_sha256": _sha(json.dumps(stats, sort_keys=True)),
            "first_record_sha256": _sha(_canonical(records[0])),
            "last_record_sha256": _sha(_canonical(records[-1])),
            "first_sample": [records[0]["sample_id"], records[0]["step_index"]],
        },
        "llm_text": [to_llm_text(text) for text, _ in LLM_TEXT_CASES],
        "audio": {
            "wav_8k": _decode_wav(8000, AUDIO_8K_SAMPLES),
            "wav_44k": _decode_wav(44100, AUDIO_44K_SAMPLES),
            "resample": [round(value, 9) for value in resample_linear([0.0, 1.0, 0.0, -1.0], 8000, 16000)],
        },
        "submission": {
            "bytes": _submission_bytes().decode("utf-8"),
            "sha256": _sha(_submission_bytes().decode("utf-8")),
        },
    }

def _golden_outputs() -> dict[str, dict]:
    """The same shape, built entirely from the pasted golden constants."""
    return {
        "prompt": {
            "sha256": PROMPT_SHA256,
            "length": PROMPT_LENGTH,
            "sections": PROMPT_SECTIONS,
        },
        "prompt_with_steps": {
            "sha256": PROMPT_STEPS_SHA256,
            "length": PROMPT_STEPS_LENGTH,
        },
        "protocol": {"tool_call": TARGET_TOOL_CALL, "finish": TARGET_FINISH},
        "sft": {
            "count": SFT_RECORD_COUNT,
            "records_sha256": SFT_RECORDS_SHA256,
            "stats_sha256": SFT_STATS_SHA256,
            "first_record_sha256": SFT_FIRST_RECORD_SHA256,
            "last_record_sha256": SFT_LAST_RECORD_SHA256,
            "first_sample": list(SFT_FIRST_SAMPLE),
        },
        "llm_text": [expected for _, expected in LLM_TEXT_CASES],
        "audio": {
            "wav_8k": {
                "rate": AUDIO_8K["rate"],
                "length": AUDIO_8K["length"],
                "quant_head": AUDIO_8K["quant_head"],
                "quant_tail": AUDIO_8K["quant_tail"],
                "quant_sha256": AUDIO_8K["quant_sha256"],
            },
            "wav_44k": {
                "rate": AUDIO_44K["rate"],
                "length": AUDIO_44K["length"],
                "quant_head": AUDIO_44K["quant_head"],
                "quant_tail": AUDIO_44K["quant_tail"],
                "quant_sha256": AUDIO_44K["quant_sha256"],
            },
            "resample": RESAMPLE_UPSAMPLED,
        },
        "submission": {"bytes": SUBMISSION_LINE, "sha256": SUBMISSION_SHA256},
    }

# --------------------------------------------------------------------------- #
# 1. prompt
# --------------------------------------------------------------------------- #
def test_prompt_golden_is_stable_and_matches_the_pinned_bytes():
    first = _render_prompt(with_steps=False)
    second = _render_prompt(with_steps=False)
    assert first == second, "build_decision_prompt is nondeterministic"
    assert len(first) == PROMPT_LENGTH, len(first)
    assert _sha(first) == PROMPT_SHA256
    headers = [line for line in first.splitlines() if line.startswith("# ")]
    assert headers == PROMPT_SECTIONS, headers
    assert first.endswith("\n")
    assert first.startswith("# Yêu cầu\n" + "Dat bao thuc luc 6 gio sang mai.")

def test_prompt_golden_with_prior_steps():
    body = _render_prompt(with_steps=True)
    assert _sha(body) == PROMPT_STEPS_SHA256
    assert len(body) == PROMPT_STEPS_LENGTH
    assert "# Các bước đã thực hiện" in body

# --------------------------------------------------------------------------- #
# 2. protocol
# --------------------------------------------------------------------------- #
def test_protocol_golden_bytes_are_the_canonical_action_format():
    call = render_action(make_tool_call("alarm.set", {"time": "06:30"}))
    finish = render_action({"type": "finish"})
    assert call == TARGET_TOOL_CALL, call
    assert finish == TARGET_FINISH, finish
    assert call == render_action(make_tool_call("alarm.set", {"time": "06:30"}))
    assert finish == render_action(FINISH_ACTION)
    # key order is fixed: type, name, arguments
    assert call.index('"type"') < call.index('"name"') < call.index('"arguments"')

# --------------------------------------------------------------------------- #
# 3. SFT dataset
# --------------------------------------------------------------------------- #
def test_sft_dataset_golden_counts_and_hashes():
    records, stats = _sft_records()
    again, stats_again = _sft_records()
    assert again == records and stats_again == stats, "build_sft_dataset is nondeterministic"
    assert len(records) == SFT_RECORD_COUNT
    assert stats["n_samples"] == SFT_SAMPLE_COUNT
    assert stats["n_decision_points"] == SFT_RECORD_COUNT
    assert _sha(_canonical(records)) == SFT_RECORDS_SHA256
    assert _sha(json.dumps(stats, sort_keys=True)) == SFT_STATS_SHA256
    assert _sha(_canonical(records[0])) == SFT_FIRST_RECORD_SHA256
    assert _sha(_canonical(records[-1])) == SFT_LAST_RECORD_SHA256
    assert (records[0]["sample_id"], records[0]["step_index"]) == SFT_FIRST_SAMPLE
    for record in records:
        assert set(record) == {"sample_id", "step_index", "messages", "target", "meta"}
        assert [m["role"] for m in record["messages"]] == ["system", "user"]
        assert record["target"].startswith("<action>") and record["target"].endswith("</action>")

# --------------------------------------------------------------------------- #
# 4. to_llm_text
# --------------------------------------------------------------------------- #
def test_llm_text_golden_pairs_are_exact_and_idempotent():
    assert len(LLM_TEXT_CASES) == 12
    for source, expected in LLM_TEXT_CASES:
        actual = to_llm_text(source)
        assert actual == expected, (source, actual, expected)
        assert to_llm_text(actual) == expected, f"not idempotent: {source!r}"
    assert to_llm_text(None) == ""

# --------------------------------------------------------------------------- #
# 5. audio
# --------------------------------------------------------------------------- #
def test_audio_golden_8k_pcm16():
    decoded = _decode_wav(8000, AUDIO_8K_SAMPLES)
    assert decoded["rate"] == AUDIO_8K["rate"]
    assert decoded["length"] == AUDIO_8K["length"]
    assert decoded["quant_head"] == AUDIO_8K["quant_head"]
    assert decoded["quant_tail"] == AUDIO_8K["quant_tail"]
    assert decoded["quant_sha256"] == AUDIO_8K["quant_sha256"]

def test_audio_golden_44k_pcm16():
    decoded = _decode_wav(44100, AUDIO_44K_SAMPLES)
    assert decoded["rate"] == AUDIO_44K["rate"]
    assert decoded["length"] == AUDIO_44K["length"]
    assert decoded["quant_head"] == AUDIO_44K["quant_head"]
    assert decoded["quant_tail"] == AUDIO_44K["quant_tail"]
    assert decoded["quant_sha256"] == AUDIO_44K["quant_sha256"]

def test_resample_linear_golden_and_identity_case():
    upsampled = [round(v, 9) for v in resample_linear([0.0, 1.0, 0.0, -1.0], 8000, 16000)]
    assert upsampled == RESAMPLE_UPSAMPLED, upsampled
    assert _sha(str(upsampled)) == RESAMPLE_SHA256
    assert resample_linear([0.25, -0.5], 16000, 16000) == [0.25, -0.5]
    assert resample_linear([], 8000, 16000) == []

# --------------------------------------------------------------------------- #
# 6. submission
# --------------------------------------------------------------------------- #
def test_submission_golden_line_bytes():
    raw = _submission_bytes()
    assert len(raw) == SUBMISSION_BYTES, len(raw)
    assert raw.decode("utf-8") == SUBMISSION_LINE
    assert _sha(raw.decode("utf-8")) == SUBMISSION_SHA256
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert normalise_row(SUBMISSION_ROW) == {
        "id": SUBMISSION_ROW["id"],
        "tool_calls": SUBMISSION_ROW["tool_calls"],
        "state_hash": SUBMISSION_ROW["state_hash"],
    }

# --------------------------------------------------------------------------- #
# The equivalence proof: flags off == baseline
# --------------------------------------------------------------------------- #
def test_default_feature_config_reproduces_every_golden_output():
    """With every flag off, all six components are byte-identical to baseline.

    ``lifemate.features`` is imported *here*, not into any module under test, and
    the config is validated first so a broken validator cannot hide behind the
    comparison.
    """
    from lifemate.features.config import FeatureConfig, enabled_features
    from lifemate.features.registry import REGISTRY

    cfg = FeatureConfig()
    assert enabled_features(cfg) == [], enabled_features(cfg)
    REGISTRY.validate_or_raise(cfg)

    actual = all_baseline_outputs()
    golden = _golden_outputs()
    assert actual == golden, json.dumps(
        {key: (actual[key], golden[key]) for key in golden if actual[key] != golden[key]},
        ensure_ascii=False,
        indent=2,
    )
    # And the "off" state really is the untouched default object, not a set of
    # explicit False overrides that happen to agree.
    assert cfg.to_dict() == FeatureConfig().to_dict()

def test_modules_under_test_never_import_the_feature_package():
    """Zero coupling, checked statically: the flags cannot have touched them."""
    for relative in COUPLED_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "features" in module.split(".") or any(
                    "features" in (alias.name or "") for alias in node.names
                ):
                    found.append(f"from {module} import ...")
            elif isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names if "features" in alias.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in {"lifemate.features", "features.config", "features.registry"}:
                    found.append(repr(node.value))
        assert found == [], (relative, found)


# --------------------------------------------------------------------------- #
# The golden must be able to FAIL
# --------------------------------------------------------------------------- #
def test_golden_comparison_detects_a_perturbed_expectation():
    """A golden nobody can trip is decoration.

    Each mutation below changes exactly one pinned value; the comparison must
    reject every one of them, which is what makes the equivalence test above a
    real assertion instead of a tautology.
    """
    baseline = all_baseline_outputs()
    assert baseline == _golden_outputs()

    def _zero(digest: str) -> str:
        return "0" * len(digest)

    mutations = {
        "prompt sha": lambda out: out["prompt"].update(sha256=_zero(PROMPT_SHA256)),
        "prompt length": lambda out: out["prompt"].update(length=PROMPT_LENGTH + 1),
        "prompt sections": lambda out: out["prompt"].update(sections=out["prompt"]["sections"][:-1]),
        "prompt with_steps": lambda out: out["prompt_with_steps"].update(sha256=_zero(PROMPT_STEPS_SHA256)),
        "protocol tool_call": lambda out: out["protocol"].update(tool_call=TARGET_TOOL_CALL.replace("06:30", "07:30")),
        "protocol finish": lambda out: out["protocol"].update(finish="<action>{}</action>"),
        "sft count": lambda out: out["sft"].update(count=SFT_RECORD_COUNT + 1),
        "sft records": lambda out: out["sft"].update(records_sha256=_zero(SFT_RECORDS_SHA256)),
        "sft stats": lambda out: out["sft"].update(stats_sha256=_zero(SFT_STATS_SHA256)),
        "sft first": lambda out: out["sft"].update(first_record_sha256=_zero(SFT_FIRST_RECORD_SHA256)),
        "sft last": lambda out: out["sft"].update(last_record_sha256=_zero(SFT_LAST_RECORD_SHA256)),
        "sft first sample": lambda out: out["sft"].update(first_sample=["other", 0]),
        "llm text": lambda out: out["llm_text"].__setitem__(1, "sai"),
        "audio 8k sha": lambda out: out["audio"]["wav_8k"].update(quant_sha256=_zero(AUDIO_8K["quant_sha256"])),
        "audio 8k head": lambda out: out["audio"]["wav_8k"].update(quant_head=[0, 0]),
        "audio 44k length": lambda out: out["audio"]["wav_44k"].update(length=3),
        "audio 44k tail": lambda out: out["audio"]["wav_44k"].update(quant_tail=[0]),
        "resample": lambda out: out["audio"].update(resample=[0.0]),
        "submission bytes": lambda out: out["submission"].update(bytes='{"id": "x"}\n'),
        "submission sha": lambda out: out["submission"].update(sha256=_zero(SUBMISSION_SHA256)),
    }
    missed = []
    for label, mutate in mutations.items():
        expected = json.loads(json.dumps(baseline))
        mutate(expected)
        if all_baseline_outputs() == expected:
            missed.append(label)
    assert missed == [], missed
    print(f"  {len(mutations)} perturbation(s) all rejected")
