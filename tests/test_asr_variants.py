"""Noisy-ASR / N-best variants: rendering, selection and inertness tests.

Runs under the stdlib runner (``python3 tests/run_tests.py``) with no fixtures on
disk: every transcript file is written to a temporary directory.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate.agent.prompt_builder import build_decision_prompt  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.data.asr_variants import (  # noqa: E402
    NBEST_HEADER,
    REQUEST_HEADER,
    TranscriptHypothesis,
    TranscriptRecord,
    apply_nbest_format,
    apply_noisy_asr,
    load_transcripts,
    render_request_section,
)
from lifemate.data.augmentation import prompt_section  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.data.trajectory_builder import DecisionPoint, to_hf_dataset_records  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402

MODULE = ROOT / "src" / "lifemate" / "data" / "asr_variants.py"
GOLD = "Đặt báo thức lúc 6 giờ 30 sáng mai."
CATALOG = ToolCatalog.from_dict(
    {
        "apps": [
            {
                "app_name": "alarm",
                "tools": [
                    {
                        "name": "alarm.set",
                        "parameters": {
                            "type": "object",
                            "properties": {"time": {"type": "string"}},
                            "required": ["time"],
                        },
                    }
                ],
            },
            {"app_name": "sys", "tools": [{"name": "sys.ask_missing_info", "kind": "meta", "parameters": {}}]},
        ]
    }
)

def _prompt(transcript: str = GOLD, history: list | None = None) -> str:
    return build_decision_prompt(
        transcript=transcript,
        context={
            "installed_apps": ["alarm"],
            "current_time": "2026-11-07T09:30:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
        },
        history=history or [],
        installed_apps=["alarm"],
        catalog=ToolRegistry(CATALOG, apps=["alarm"]),
    )

def _records(n: int = 4, transcript: str = GOLD) -> list[dict]:
    points = [
        DecisionPoint(
            sample_id=f"s{index}",
            step_index=0,
            prompt_body=_prompt(transcript),
            target_action={"type": "tool_call", "name": "alarm.set", "arguments": {"time": "06:30"}},
        )
        for index in range(n)
    ]
    return to_hf_dataset_records(points)

def _record(sample_id: str, hypotheses: list[str], *, model: str = "gipformer-68M") -> TranscriptRecord:
    return TranscriptRecord(
        sample_id=sample_id,
        hypotheses=tuple(
            TranscriptHypothesis(text=text, score=None, rank=index, source="asr")
            for index, text in enumerate(hypotheses)
        ),
        asr_model=model,
        asr_revision="rev1",
        asr_config={"backend": "gipformer"},
    )

def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path

# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_single_form_matches_the_prompt_builder_bytes():
    record = _record("s0", ["báo thức sáu rưỡi sáng mai"])
    section = render_request_section(record)
    assert section == REQUEST_HEADER + "\n" + "báo thức sáu rưỡi sáng mai", repr(section)
    # The baseline builder writes exactly this section into its prompt.
    assert prompt_section(_prompt("báo thức sáu rưỡi sáng mai"), REQUEST_HEADER) == (
        "báo thức sáu rưỡi sáng mai"
    )
    assert section.endswith("\n".join(["báo thức sáu rưỡi sáng mai"]))

def test_nbest_form_is_the_numbered_list_with_one_blank_line():
    record = _record("s0", ["một", "hai", "ba"])
    section = render_request_section(record, nbest_format=True, k=3)
    assert section == NBEST_HEADER + "\n\n1. một\n2. hai\n3. ba", repr(section)
    assert NBEST_HEADER == "# Các giả thuyết ASR"

def test_nbest_k_caps_the_list():
    record = _record("s0", ["một", "hai", "ba", "bốn"])
    assert render_request_section(record, nbest_format=True, k=2) == (
        NBEST_HEADER + "\n\n1. một\n2. hai"
    )

def test_duplicate_hypothesis_texts_are_deduped_keeping_best_rank():
    record = _record("s0", ["giống", "khác", "giống"])
    assert [hyp.text for hyp in record.hypotheses] == ["giống", "khác"]
    assert render_request_section(record, nbest_format=True) == NBEST_HEADER + "\n\n1. giống\n2. khác"

def test_single_hypothesis_falls_back_to_the_single_form():
    record = _record("s0", ["chỉ một"])
    assert render_request_section(record, nbest_format=True, k=3) == REQUEST_HEADER + "\nchỉ một"

def test_noisy_variant_replaces_only_the_request_section():
    (record,) = _records(1)
    transcripts = {"s0": _record("s0", ["báo thức sáu rưỡi sáng mai", "báo thức sáu giờ sáng mai"])}
    produced, report = apply_noisy_asr([record], transcripts, enabled=True, ratio=1.0, seed=7)
    assert report["n_variants_created"] == 1
    original_prompt = record["messages"][-1]["content"]
    variant_prompt = produced[1]["messages"][-1]["content"]
    assert prompt_section(variant_prompt, REQUEST_HEADER) == "báo thức sáu rưỡi sáng mai"
    # Every other byte is untouched (anchor on the next section header).
    tail = original_prompt.index("\n\n# Ngữ cảnh")
    assert variant_prompt.endswith(original_prompt[tail:])
    assert variant_prompt.startswith(REQUEST_HEADER)

def test_noisy_variant_keeps_the_target_verbatim():
    (record,) = _records(1)
    transcripts = {"s0": _record("s0", ["hỏng", "vẫn hỏng"])}
    produced, _report = apply_noisy_asr([record], transcripts, enabled=True, ratio=1.0, seed=7)
    assert produced[1]["target"] == record["target"]
    assert produced[1]["target"] == '<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": "06:30"}}</action>'
    assert produced[1]["step_index"] == record["step_index"]
    assert produced[1]["sample_id"] == record["sample_id"]
    assert produced[1]["messages"][0] == record["messages"][0]

def test_provenance_meta_is_written_on_both_sides():
    (record,) = _records(1)
    transcripts = {"s0": _record("s0", ["a", "b"], model="gipformer-68M")}
    produced, _report = apply_noisy_asr([record], transcripts, enabled=True, ratio=1.0, seed=11)
    variant_meta = produced[1]["meta"]
    assert variant_meta["augmentation"] == "noisy_asr"
    assert variant_meta["asr_model"] == "gipformer-68M"
    assert variant_meta["asr_revision"] == "rev1"
    assert variant_meta["asr_rank"] == 1
    assert variant_meta["noisy_asr_ratio"] == 1.0
    assert variant_meta["augmentation_seed"] == 11
    assert variant_meta["gold_transcript_replaced"] is True
    # The original keeps its own meta untouched (no augmentation key added).
    assert "augmentation" not in produced[0]["meta"]
    assert produced[0] is record

# --------------------------------------------------------------------------- #
# Selection / inertness
# --------------------------------------------------------------------------- #
def test_ratio_zero_with_the_flag_off_is_inert_identity():
    records = _records(4)
    produced, report = apply_noisy_asr(records, None, enabled=False, ratio=0.0)
    assert produced is records
    assert report == {}

def test_ratio_above_zero_with_the_flag_off_is_inert():
    records = _records(4)
    produced, report = apply_noisy_asr(records, {}, enabled=False, ratio=0.5)
    assert produced is records
    assert report == {}

def test_ratio_out_of_range_with_the_flag_on_raises():
    records = _records(2)
    for ratio in (0.0, -0.5, 1.5):
        try:
            apply_noisy_asr(records, {"s0": _record("s0", ["a", "b"])}, enabled=True, ratio=ratio)
        except LifeMateError as exc:
            assert "ratio" in str(exc)
        else:
            raise AssertionError(f"ratio {ratio} must raise when enabled")

def test_missing_transcripts_are_counted_not_invented():
    records = _records(2)
    transcripts = {"s0": _record("s0", ["a", "b"])}
    produced, report = apply_noisy_asr(records, transcripts, enabled=True, ratio=1.0, seed=3)
    assert report["n_records_in"] == 2
    assert report["n_missing_transcript"] == 1
    assert report["n_variants_created"] == 1
    assert len(produced) == 3  # no variant for the sample without a transcript
    assert {record["sample_id"] for record in produced} == {"s0", "s1"}
    assert all(record["meta"].get("augmentation") != "noisy_asr" for record in produced[2:])

def test_selection_is_deterministic_under_reordering():
    records = _records(12)
    transcripts = {record["sample_id"]: _record(record["sample_id"], ["x", "y"]) for record in records}
    first, report_first = apply_noisy_asr(records, transcripts, enabled=True, ratio=0.5, seed=42)
    shuffled = list(reversed(records))
    _second, report_second = apply_noisy_asr(shuffled, transcripts, enabled=True, ratio=0.5, seed=42)
    assert report_first["sample_ids_selected_hash"] == report_second["sample_ids_selected_hash"]
    assert report_first["n_variants_created"] == report_second["n_variants_created"]
    assert 0 < report_first["n_variants_created"] < len(records)
    selected = {
        variant["sample_id"] for variant in first if variant["meta"].get("augmentation") == "noisy_asr"
    }
    assert selected == {
        variant["sample_id"]
        for variant in apply_noisy_asr(shuffled, transcripts, enabled=True, ratio=0.5, seed=42)[0]
        if variant["meta"].get("augmentation") == "noisy_asr"
    }

def test_a_different_seed_changes_the_selection():
    records = _records(20)
    transcripts = {record["sample_id"]: _record(record["sample_id"], ["x", "y"]) for record in records}
    a = apply_noisy_asr(records, transcripts, enabled=True, ratio=0.5, seed=1)[1]["sample_ids_selected_hash"]
    b = apply_noisy_asr(records, transcripts, enabled=True, ratio=0.5, seed=2)[1]["sample_ids_selected_hash"]
    assert a != b

def test_nbest_format_variant_and_its_single_hypothesis_skips():
    records = _records(2)
    transcripts = {
        "s0": _record("s0", ["một", "hai", "ba", "bốn"]),
        "s1": _record("s1", ["chỉ một"]),
    }
    produced, report = apply_nbest_format(records, transcripts, enabled=True, k=3, seed=42)
    assert report["n_variants_created"] == 1
    assert report["n_single_hypothesis"] == 1
    variant = produced[1]  # appended right after its own original
    # Body keeps the blank line the N-best form puts after its header.
    assert prompt_section(variant["messages"][-1]["content"], NBEST_HEADER) == "\n1. một\n2. hai\n3. ba"
    assert prompt_section(variant["messages"][-1]["content"], REQUEST_HEADER) is None
    assert variant["target"] == records[0]["target"]
    assert variant["meta"]["n_hypotheses_shown"] == 3

def test_nbest_format_disabled_is_identity():
    records = _records(2)
    produced, report = apply_nbest_format(records, None, enabled=False)
    assert produced is records
    assert report == {}

# --------------------------------------------------------------------------- #
# Transcript files
# --------------------------------------------------------------------------- #
def test_load_transcripts_reads_the_producer_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            Path(tmp) / "t.jsonl",
            [
                {
                    "sample_id": "s0",
                    "hypotheses": [
                        {"text": "một", "rank": 0},
                        {"text": "hai", "rank": 1},
                    ],
                    "asr_model": "gipformer-68M",
                    "asr_revision": "rev1",
                    "asr_config": {"quantization": "int8"},
                }
            ],
        )
        loaded = load_transcripts(path)
        assert list(loaded) == ["s0"]
        assert [hyp.text for hyp in loaded["s0"].hypotheses] == ["một", "hai"]
        assert loaded["s0"].asr_model == "gipformer-68M"
        assert loaded["s0"].asr_revision == "rev1"
        assert loaded["s0"].asr_config == {"quantization": "int8"}

def test_missing_transcript_file_raises_with_the_producing_command():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "absent.jsonl"
        try:
            load_transcripts(missing)
        except LifeMateError as exc:
            message = str(exc)
            assert "scripts/transcribe_split.py" in message
            assert "--transcript-records" in message
        else:
            raise AssertionError("a missing transcript file must raise")

def test_require_multi_refuses_a_one_best_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "t.jsonl", [{"sample_id": "s0", "hypotheses": [{"text": "một", "rank": 0}]}])
        assert load_transcripts(path)["s0"].hypotheses
        try:
            load_transcripts(path, require_multi=True)
        except LifeMateError as exc:
            assert "2 distinct hypotheses" in str(exc)
        else:
            raise AssertionError("require_multi must refuse a 1-best file")

def test_transcript_paths_are_compliance_guarded():
    from lifemate.errors import ProhibitedAccessError

    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / ("train" + "_environment")  # compliance-scan: allow
        env_dir.mkdir()
        target = env_dir / "t.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        try:
            load_transcripts(target)
        except ProhibitedAccessError:
            return
        raise AssertionError("a *_environment transcript path must be refused")

# --------------------------------------------------------------------------- #
# Static: no ASR model, no network
# --------------------------------------------------------------------------- #
def test_module_never_imports_an_asr_stack_or_a_network_client():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("transformers", "torch", "requests", "urllib", "http", "sherpa_onnx", "huggingface_hub"):
        assert forbidden not in imported, f"{forbidden} must not be imported by asr_variants"
    text = MODULE.read_text(encoding="utf-8")
    for forbidden in ("hf_hub_download", "from_pretrained", "urllib.request", "requests."):
        assert forbidden not in text, f"{forbidden} must not appear in asr_variants"

def test_all_flags_off_returns_the_same_objects_and_bytes():
    records = _records(3)
    before = [(id(record), json.dumps(record, sort_keys=True)) for record in records]
    noisy, noisy_report = apply_noisy_asr(records, None, enabled=False, ratio=0.0, seed=42)
    nbest, nbest_report = apply_nbest_format(records, None, enabled=False)
    assert noisy is records and nbest is records
    assert noisy_report == {} and nbest_report == {}
    assert [(id(record), json.dumps(record, sort_keys=True)) for record in records] == before

def test_noisy_asr_replace_mode_maintains_record_count_and_ratio():
    records = _records(10)
    transcripts = {record["sample_id"]: _record(record["sample_id"], ["noisy hyp"]) for record in records}
    produced, report = apply_noisy_asr(records, transcripts, enabled=True, ratio=0.5, seed=42, mode="replace")
    assert len(produced) == 10
    assert report["n_original_records"] == 10
    assert report["n_total_records"] == 10
    assert report["augmentation_mode"] == "replace"
    assert report["requested_noisy_ratio"] == 0.5
    assert report["effective_noisy_ratio"] == round(report["n_noisy_records"] / 10, 6)
    assert report["n_clean_records"] + report["n_noisy_records"] == 10
    for key in (
        "n_original_records",
        "n_clean_records",
        "n_noisy_records",
        "n_total_records",
        "requested_noisy_ratio",
        "effective_noisy_ratio",
        "n_missing_transcripts",
        "n_skipped",
        "augmentation_mode",
        "seed",
        "dataset_fingerprint",
    ):
        assert key in report, f"missing {key} in report"
    assert len(report["dataset_fingerprint"]) == 16

def test_noisy_asr_augment_mode_preserves_clean_and_appends():
    records = _records(10)
    transcripts = {record["sample_id"]: _record(record["sample_id"], ["noisy hyp"]) for record in records}
    produced, report = apply_noisy_asr(records, transcripts, enabled=True, ratio=0.5, seed=42, mode="augment")
    assert report["augmentation_mode"] == "augment"
    assert report["n_original_records"] == 10
    assert report["n_clean_records"] == 10
    assert report["n_total_records"] == 10 + report["n_noisy_records"]
    assert len(produced) == report["n_total_records"]

def test_noisy_asr_missing_transcripts_skipped_in_replace_mode():
    records = _records(10)
    # Only s0 has transcript; s1..s9 missing
    transcripts = {"s0": _record("s0", ["noisy hyp"])}
    produced, report = apply_noisy_asr(records, transcripts, enabled=True, ratio=1.0, seed=42, mode="replace")
    assert len(produced) == 10
    assert report["n_missing_transcripts"] == 9
    assert report["n_skipped"] == 9
    assert report["n_noisy_records"] == 1
    assert report["n_clean_records"] == 9
    assert produced[0]["meta"].get("augmentation") == "noisy_asr"
    assert all(r["meta"].get("augmentation") != "noisy_asr" for r in produced[1:])

def test_noisy_asr_sample_grouping_preserved():
    records = _records(5)
    # Duplicate sample_ids to simulate multi-turn steps for same sample
    multi_turn = []
    for r in records:
        step0 = dict(r)
        step1 = dict(r)
        step1["step_index"] = 1
        multi_turn.extend([step0, step1])
    transcripts = {r["sample_id"]: _record(r["sample_id"], ["noisy"]) for r in records}
    produced, _report = apply_noisy_asr(multi_turn, transcripts, enabled=True, ratio=0.6, seed=42, mode="replace")
    # For each sample_id, either all steps are noisy or none are
    by_sid: dict[str, list[bool]] = {}
    for item in produced:
        sid = item["sample_id"]
        is_noisy = (item.get("meta") or {}).get("augmentation") == "noisy_asr"
        by_sid.setdefault(sid, []).append(is_noisy)
    for sid, statuses in by_sid.items():
        assert len(statuses) == 2
        assert statuses[0] == statuses[1], f"sample {sid} steps must have identical noisy status"
