"""Evaluation tests: hand-computed metrics, availability flags, failure buckets,
W&B redaction, Dockerfile references, documented commands, verify_all schema."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lifemate.evaluation import error_analysis, metrics  # noqa: E402
from lifemate.evaluation.submission import SubmissionWriter  # noqa: E402
from lifemate.tracking.wandb_tracker import (  # noqa: E402
    REDACTED,
    WandbTracker,
    describe_environment,
    sanitize,
    wandb_available,
)


# --------------------------------------------------------------------------- #
# Metrics: hand-computed values
# --------------------------------------------------------------------------- #
def test_exact_match_canonicalisation():
    assert metrics.tool_call_exact_match(
        [{"name": "alarm.set", "arguments": {"repeat": "once", "time": "06:30"}}],
        [{"name": "alarm.set", "arguments": {"time": " 06:30 ", "repeat": "once"}}],
    ), "key order / whitespace must not matter"
    assert metrics.tool_call_exact_match(
        [{"name": "t", "arguments": {"n": 1}}], [{"name": "t", "arguments": {"n": 1.0}}]
    ), "1 and 1.0 must compare equal"
    assert not metrics.tool_call_exact_match(
        [{"name": "t", "arguments": {"n": True}}], [{"name": "t", "arguments": {"n": 1}}]
    ), "bool is not an int"
    assert not metrics.tool_call_exact_match(
        [{"name": "t", "arguments": {"xs": [1, 2]}}], [{"name": "t", "arguments": {"xs": [2, 1]}}]
    ), "argument lists stay order sensitive"
    assert metrics.tool_call_exact_match([], [])
    assert not metrics.tool_call_exact_match([], [{"name": "t", "arguments": {}}])


def test_lcs_f1_hand_computed():
    gold = [
        {"name": "a", "arguments": {}},
        {"name": "b", "arguments": {}},
        {"name": "c", "arguments": {}},
    ]
    pred = [
        {"name": "a", "arguments": {}},
        {"name": "x", "arguments": {}},
        {"name": "c", "arguments": {}},
    ]
    block = metrics.sequence_precision_recall_f1(pred, gold)
    # LCS = [a, c] = 2. precision 2/3, recall 2/3, f1 2/3.
    assert block["lcs_len"] == 2, block
    assert abs(block["precision"] - 2 / 3) < 1e-9
    assert abs(block["recall"] - 2 / 3) < 1e-9
    assert abs(block["f1"] - 2 / 3) < 1e-9

    reordered = [
        {"name": "b", "arguments": {}},
        {"name": "a", "arguments": {}},
    ]
    assert metrics.sequence_precision_recall_f1(reordered, gold)["lcs_len"] == 1
    empty = metrics.sequence_precision_recall_f1([], [])
    assert empty["f1"] == 1.0 and empty["lcs_len"] == 0


def test_missing_info_precision_recall():
    preds = [
        {"id": "1", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {}}]},
        {"id": "2", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {}}]},
        {"id": "3", "tool_calls": []},
        {"id": "4", "tool_calls": []},
    ]
    golds = [
        {"id": "1", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {}}]},
        {"id": "2", "tool_calls": []},
        {"id": "3", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {}}]},
        {"id": "4", "tool_calls": []},
    ]
    block = metrics.missing_info_detection(preds, golds)
    assert (block["tp"], block["fp"], block["fn"], block["tn"]) == (1, 1, 1, 1), block
    assert abs(block["precision"] - 0.5) < 1e-9
    assert abs(block["recall"] - 0.5) < 1e-9
    assert abs(block["f1"] - 0.5) < 1e-9


def test_out_of_scope_accuracy():
    preds = [{"tool_calls": []}, {"tool_calls": [{"name": "a", "arguments": {}}]}, {"tool_calls": []}]
    golds = [{"tool_calls": []}, {"tool_calls": []}, {"tool_calls": [{"name": "a", "arguments": {}}]}]
    block = metrics.out_of_scope_accuracy(preds, golds)
    assert block["n"] == 2 and block["correct"] == 1, block
    assert abs(block["value"] - 0.5) < 1e-9
    assert metrics.out_of_scope_accuracy([{"tool_calls": []}], [{"tool_calls": [{"name": "a"}]}])["value"] is None


def test_tool_selection_and_stop_accuracy():
    gold = [{"name": "a", "arguments": {}}]
    assert metrics.tool_selection_accuracy(gold, gold)["value"] == 1.0
    assert metrics.tool_selection_accuracy([{"name": "b", "arguments": {}}], gold)["value"] == 0.0
    assert metrics.stop_accuracy({"tool_calls": [{"arguments": {}, "name": "a"}], "finish": True},
                                 {"tool_calls": [{"arguments": {}, "name": "a"}], "finish": True})["value"] == 1.0
    assert metrics.stop_accuracy({"tool_calls": [], "finish": True}, {"tool_calls": [{}], "finish": True})["value"] == 0.0
    assert metrics.call_count_delta([{}, {}], [{}]) == 1


def test_next_action_accuracy_per_step():
    gold = [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
    pred = [{"name": "a", "arguments": {}}]
    block = metrics.next_action_accuracy(pred, gold)
    assert block["n"] == 2 and block["correct"] == 1
    assert abs(block["value"] - 0.5) < 1e-9


def test_aggregate_reports_unavailable_blocks():
    preds = [{"id": "1", "tool_calls": [{"name": "a", "arguments": {"x": 1}}]}]
    golds = [{"id": "1", "tool_calls": [{"name": "a", "arguments": {"x": 1}}]}]
    payload = metrics.aggregate_teacher_forced(preds, golds)
    assert payload["tool_call_exact_match"]["available"] is True
    assert payload["next_action_accuracy"]["available"] is False, "no per-step data -> undefinable"
    assert payload["out_of_scope_accuracy"]["available"] is False, "no zero-call gold sample"
    assert payload["missing_info_detection"]["precision"] is None, "no positive predictions"
    assert payload["n_records"] == 1
    assert "LCS" in payload["definitions"]["sequence_f1"]

    empty = metrics.aggregate_teacher_forced([], [])
    assert empty["n_records"] == 0
    assert empty["tool_call_exact_match"]["available"] is False


def test_metrics_have_printed_definitions():
    for name in (
        "tool_call_exact_match",
        "sequence_f1",
        "argument_exact_match",
        "tool_selection_accuracy",
        "next_action_accuracy",
        "stop_accuracy",
        "missing_info_detection",
        "out_of_scope_accuracy",
        "call_count_delta",
    ):
        assert name in metrics.METRIC_DEFINITIONS, name


# --------------------------------------------------------------------------- #
# Error analysis: one rule per bucket
# --------------------------------------------------------------------------- #
def test_failure_classification_rules():
    gold = {"id": "1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}]}
    ok_pred = {"id": "1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}]}

    assert error_analysis.classify_failure({"status": "timeout", "tool_calls": []}, gold)["bucket"] == "runtime"
    wrong_pred = {"id": "1", "tool_calls": [], "raw_output": ""}
    assert (
        error_analysis.classify_failure(wrong_pred, gold, asr_result={"text": "", "error": None})["bucket"] == "ASR"
    ), "an empty transcript must be blamed on ASR before any model-side bucket"
    assert (
        error_analysis.classify_failure(ok_pred, gold, asr_result={"text": "", "error": None})["bucket"] == "matched"
    ), "a correct prediction is not a failure even if the transcript was empty"
    assert (
        error_analysis.classify_failure({**wrong_pred, "prompt_truncated": True}, gold)["bucket"]
        == "prompt_construction"
    )
    assert (
        error_analysis.classify_failure({**ok_pred, "prompt_truncated": True}, gold)["bucket"] == "matched"
    ), "a truncated prompt that still produced the gold sequence is not a failure"
    assert (
        error_analysis.classify_failure({"id": "1", "tool_calls": [], "invalid_actions": 2}, gold)["bucket"]
        == "llm_planning"
    )
    assert (
        error_analysis.classify_failure(
            {"id": "1", "tool_calls": [{"name": "nope.nope", "arguments": {}}]}, gold
        )["bucket"]
        == "tool_selection"
    )
    assert (
        error_analysis.classify_failure(
            {"id": "1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "07:00"}}]}, gold
        )["bucket"]
        == "argument_generation"
    )
    assert (
        error_analysis.classify_failure(
            {**ok_pred, "tool_results": [{"name": "alarm.set", "ok": False, "error": "nope"}]}, gold
        )["bucket"]
        == "tool_execution"
    )
    assert (
        error_analysis.classify_failure({"id": "1", "tool_calls": []}, gold)["bucket"] == "stopping"
    )
    assert error_analysis.classify_failure(ok_pred, gold)["bucket"] == "matched"


def test_analyze_aggregates_without_pii():
    golds = [
        {"id": "s1", "tool_calls": [{"name": "a", "arguments": {}}], "user_profile": {"name": "X"}},
        {"id": "s2", "tool_calls": [{"name": "a", "arguments": {}}]},
    ]
    preds = [
        {"id": "s1", "tool_calls": []},
        {"id": "s2", "tool_calls": [{"name": "a", "arguments": {}}]},
    ]
    report = error_analysis.analyze(preds, golds)
    assert report["n_total"] == 2 and report["n_failed"] == 1
    assert report["counts"]["matched"] == 1
    assert report["dominant_bucket"] == "stopping"
    assert report["examples"]["stopping"] == ["s1"]
    assert report["recommendations"], "dominant bucket should yield text guidance"
    serialized = json.dumps(report)
    assert "X" not in serialized.replace("s1", ""), "profile content leaked into the report"


def test_summarize_asr_errors():
    assert error_analysis.summarize_asr_errors([])["available"] is False
    summary = error_analysis.summarize_asr_errors([("xin chao", "xin chao"), ("tat den", "bat den")])
    assert summary["available"] is True and summary["n"] == 2
    # Worker C's evaluate_asr reports nested raw/normalized blocks; both shapes
    # are accepted, but a degraded transcript must raise *some* error rate.
    flat = summary.get("wer")
    nested = (summary.get("normalized") or {}).get("wer_mean") or (summary.get("raw") or {}).get("wer_mean")
    assert (flat if flat is not None else nested) > 0, summary


# --------------------------------------------------------------------------- #
# W&B tracker: redaction + no-op behaviour
# --------------------------------------------------------------------------- #
def test_sanitize_redacts_secrets_and_pii():
    payload = {
        "api_key": "abc123",
        "WANDB_TOKEN": "t",
        "password": "hunter2",
        "nested": {"secret_value": "s", "learning_rate": 1e-5, "user_profile": {"name": "X"}},
        "audio_text": "xin chao",
        "messages": [{"role": "user", "content": "hi"}],
        "count": 3,
    }
    clean = sanitize(payload)
    assert clean["api_key"] == REDACTED
    assert clean["WANDB_TOKEN"] == REDACTED
    assert clean["password"] == REDACTED
    assert clean["nested"]["secret_value"] == REDACTED
    assert clean["nested"]["learning_rate"] == 1e-5
    assert clean["nested"]["user_profile"] == "[DROPPED_PII]"
    assert clean["audio_text"] == "[DROPPED_PII]"
    assert clean["count"] == 3
    allowed = sanitize({"user_profile": {"name": "X"}}, allow_pii=True)
    assert allowed["user_profile"] == {"name": "X"}


def test_tracker_disabled_is_a_safe_noop():
    tracker = WandbTracker(None)
    assert tracker.enabled is False
    assert tracker.url is None
    tracker.log({"loss": 1.0})
    tracker.log_metrics({"acc": 0.5}, step=1)
    tracker.log_summary({"final": 2.0})
    tracker.log_config({"api_key": "x", "lr": 1e-4})
    tracker.finish()
    assert tracker.summary["loss"] == 1.0
    assert "api_key" not in tracker.summary or tracker.summary["api_key"] == REDACTED
    env = describe_environment()
    for key in ("git_commit", "python", "torch", "gpu"):
        assert key in env
    assert isinstance(wandb_available(), bool)


# --------------------------------------------------------------------------- #
# Repo-level consistency checks
# --------------------------------------------------------------------------- #
def test_dockerfile_references_only_existing_paths():
    dockerfile = ROOT / "docker" / "Dockerfile"
    assert dockerfile.is_file(), "docker/Dockerfile is missing"
    text = dockerfile.read_text(encoding="utf-8")
    copy_lines = re.findall(r"^COPY\s+(.+)$", text, flags=re.MULTILINE)
    assert copy_lines, "Dockerfile has no COPY lines"
    checked = 0
    for line in copy_lines:
        if "--from=" in line:  # COPY --from=build /opt/venv /opt/venv
            continue
        parts = [part for part in line.split() if not part.startswith("--")]
        source = parts[0]
        assert (ROOT / source).exists(), f"Dockerfile copies a path that does not exist: {source}"
        checked += 1
    assert checked >= 4, f"expected the repo copies (src/scripts/configs/docker), saw {checked}"
    assert "entrypoint.sh" in text
    assert (ROOT / "docker" / "entrypoint.sh").is_file()
    # The image must actually start the entrypoint: without ENTRYPOINT/CMD a
    # `docker run` (and the cloud job's `command`) started nothing.
    assert re.search(r"^ENTRYPOINT\s*\[", text, flags=re.MULTILINE), "Dockerfile declares no ENTRYPOINT"
    assert "/app/docker/entrypoint.sh" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "USER lifemate" in text
    directives = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert not any(line.strip().upper().startswith("HEALTHCHECK") for line in directives), (
        "offline batch image must not declare a healthcheck"
    )


def test_documented_scripts_exist():
    pattern = re.compile(r"python3?\s+scripts/([\w./-]+\.py)")
    checked = 0
    for doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "docker").glob("*.md"))]:
        if not doc.is_file():
            continue
        for match in pattern.finditer(doc.read_text(encoding="utf-8", errors="replace")):
            script = ROOT / "scripts" / match.group(1).split("/")[-1]
            assert script.is_file(), f"{doc.relative_to(ROOT)} references missing {script}"
            checked += 1
    assert checked > 0, "no documented script commands found"

def test_gate_n_discriminates_unknown_flags():
    """Negative self-test: gate N flag scanner rejects an undocumented flag."""
    probe = subprocess.run(
        [sys.executable, "scripts/verify_all.py", "--self-test"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert "discriminates unknown flag" in probe.stdout


def test_verify_all_json_schema_parses():
    generated = ROOT / "outputs" / "verification" / "verify_all.json"
    if not generated.is_file():
        # Not generated yet: validate the shape with a synthetic payload.
        payload = {
            "meta": {"timestamp": "x", "python": "3.12", "platform": "p", "hostname": "h", "git_commit": "c", "deps": {}},
            "results": [{"gate": "A", "title": "t", "status": "PASS", "detail": "d"}],
            "totals": {"PASS": 1, "FAIL": 0, "SKIP": 0, "BLOCKED": 0},
        }
    else:
        payload = json.loads(generated.read_text(encoding="utf-8"))
    assert set(payload) >= {"meta", "results", "totals"}
    for item in payload["results"]:
        assert item["status"] in {"PASS", "FAIL", "SKIP", "BLOCKED"}
        assert {"gate", "title", "detail"} <= set(item)


def test_submission_writer_used_by_infer_keeps_ids_in_order():
    from lifemate.evaluation import evaluator

    paths = evaluator.artifact_paths(tempfile.mkdtemp(prefix="lifemate_art_"))
    assert paths["metrics_json"].name == "metrics.json"
    assert paths["comparisons_md"].name == "comparisons.md"

    writer = SubmissionWriter(Path(tempfile.mkdtemp(prefix="lifemate_w_")) / "s.jsonl")
    writer.add("b", [], "c" * 64)
    writer.add("a", [], "d" * 64)
    assert writer.ids == ["b", "a"]


# --------------------------------------------------------------------------- #
# Evaluator: teacher-forced with a test double, artifact writing
# --------------------------------------------------------------------------- #
def test_teacher_forced_with_test_double():
    from lifemate.evaluation import evaluator

    class StubGenerator:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.max_new_tokens = 8

        def generate(self, messages):
            assert messages, "the stored prompt must be passed to the generator"
            return self.outputs.pop(0)

    records = [
        {
            "id": "s1",
            "messages": [{"role": "user", "content": "dat bao thuc 6h30"}],
            "target": '<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": "06:30"}}</action>',
            "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}],
        },
        {
            "id": "s2",
            "prompt": "lich hop voi An",
            "target": '<action>{"type": "tool_call", "name": "calendar.create_event", "arguments": {"title": "An"}}</action>',
            "tool_calls": [{"name": "calendar.create_event", "arguments": {"title": "An"}}],
        },
    ]
    generator = StubGenerator(
        [
            '<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": " 06:30 "}}</action>',
            "no action block at all",
        ]
    )
    payload = evaluator.evaluate_teacher_forced(generator, records, max_new_tokens=16)
    assert payload["n_records"] == 2
    assert payload["metrics"]["tool_call_exact_match"]["value"] == 0.5
    by_id = {row["id"]: row for row in payload["records"]}
    assert by_id["s1"]["exact_match"] is True
    assert by_id["s2"]["parse_error"], "unparseable output must be recorded, not raised"
    assert payload["metrics"]["error_analysis"]["counts"]["llm_planning"] == 1

    empty = evaluator.evaluate_teacher_forced(generator, [], max_new_tokens=8)
    assert empty["metrics"]["available"] is False and empty["metrics"]["n_records"] == 0


def test_evaluate_split_writes_all_artifacts():
    from lifemate.config import load_config
    from lifemate.evaluation import evaluator

    directory = Path(tempfile.mkdtemp(prefix="lifemate_report_"))
    payload = evaluator.evaluate_split(
        cfg=load_config(),
        split="train",
        output_dir=directory,
        teacher_forced={
            "n_records": 1,
            "metrics": metrics.aggregate_teacher_forced(
                [{"id": "s1", "tool_calls": [{"name": "a", "arguments": {}}]}],
                [{"id": "s1", "tool_calls": [{"name": "a", "arguments": {}}]}],
            ),
        },
        predictions=[{"id": "s1", "tool_calls": [{"name": "a", "arguments": {}}]}],
        data_root=ROOT / "tests" / "fixtures" / "synthetic_lifemate",
        mock_env=True,
    )
    for name in ("metrics.json", "metrics.md", "predictions.jsonl", "error_analysis.json", "comparisons.md"):
        assert (directory / name).is_file(), name
    written = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert written["provenance"]["split"] == "train"
    assert written["provenance"]["config_hash"]
    assert written["environment"]["python"]
    assert "environment" in written and "gpu" in written["environment"]
    assert written["mock_env"] is True
    assert "MOCK ENV / SYNTHETIC" in (directory / "metrics.md").read_text(encoding="utf-8")
    assert payload["artifacts"]["metrics_json"].endswith("metrics.json")


def test_closed_loop_records_missing_hidden_state_metric():
    from lifemate.agent.agent import ToolAgent
    from lifemate.data.loader import load_catalog, load_samples
    from lifemate.env.factory import create_environment
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.evaluation import evaluator

    fixtures = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
    samples = load_samples(fixtures, "train", limit=2, with_labels=True)
    catalog = load_catalog(fixtures, "train")

    class StubGenerator:
        def generate(self, messages):
            return '<action>{"type": "finish"}</action>'

    def factory(sample):
        env = create_environment(
            sample.split, sample.id, fixtures, use_mock=True, catalog=catalog, apps=sample.installed_apps
        )
        return ToolAgent(generator=StubGenerator(), registry=ToolRegistry(catalog).for_sample(sample), env=env)

    summary = evaluator.evaluate_closed_loop(
        factory, samples, split="train", data_root=fixtures, max_samples=2, deadline_seconds=30.0
    )
    assert summary["n_run"] == 2
    assert summary["final_state_correctness"]["available"] is False
    assert "official scoring tooling" in summary["final_state_correctness"]["reason"]
    assert summary["rows"][0]["state_hash"], "rows must carry the env state hash"


def test_teacher_forced_accepts_worker_d_records_when_present():
    """Contract check against the real SFT export shape (skipped when absent)."""
    from lifemate.evaluation import evaluator

    records_path = ROOT / "outputs" / "datasets" / "train" / "val.jsonl"
    if not records_path.is_file():
        return  # SFT export not built in this workspace: nothing to check
    rows = evaluator.load_decision_records(records_path, limit=1)
    assert rows, "val.jsonl parsed to zero records"
    target = str(rows[0].get("target") or "")
    assert target, "SFT record has no target"

    class Replay:
        max_new_tokens = 8

        def __init__(self, output):
            self.output = output

        def generate(self, messages):
            assert messages[-1]["role"] == "user", messages
            return self.output

    payload = evaluator.evaluate_teacher_forced(Replay(target), rows, max_new_tokens=8)
    assert payload["n_records"] == 1
    assert payload["metrics"]["tool_call_exact_match"]["value"] == 1.0, payload["records"]

def test_teacher_forced_never_scores_gold_against_itself():
    """Neither the gold target nor a promptless row may produce a score."""
    from lifemate.evaluation import evaluator

    target = '<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": "06:30"}}</action>'
    calls = [{"name": "alarm.set", "arguments": {"time": "06:30"}}]

    class Echo:
        max_new_tokens = 8
        seen: list = []

        def generate(self, messages):
            Echo.seen.append(messages)
            return messages[-1]["content"]  # a model that merely repeats its context

    out = evaluator.evaluate_teacher_forced(Echo(), [{"id": "s1", "prompt": "dat bao thuc 6h30", "target": target}])
    assert [m["content"] for m in Echo.seen[0]] == ["dat bao thuc 6h30"], "gold target leaked into model input"
    assert out["metrics"]["tool_call_exact_match"]["value"] == 0.0, "gold-in-prompt must not score"

    # predictions.jsonl / submission.jsonl rows carry `tool_calls` (the prediction),
    # never gold: reading it as gold would self-score a perfect run.
    assert not evaluator._is_gradeable({"id": "s1", "tool_calls": calls}), "`tool_calls` is not gold"
    # ... and even with gold attached, a promptless row is scored on an empty context.
    assert not evaluator._is_gradeable({"id": "s1", "tool_calls": calls, "gold_tool_calls": calls}), "needs a prompt"
    for row in ({"id": "s1", "tool_calls": calls}, {"id": "s1", "gold_tool_calls": calls}):
        payload = evaluator.evaluate_teacher_forced(Echo(), [row])
        assert payload["n_records"] == 0 and payload["n_ungradeable"] == 1, payload
        assert payload["metrics"]["available"] is False, payload["metrics"]
        assert "prompt" in payload["metrics"]["reason"] and "gold" in payload["metrics"]["reason"]

    # A crashed / unparseable generation produced no decision: it must count as a
    # miss, never as a correct "do nothing" against a bare `finish` gold.
    finish_row = {"id": "s1", "messages": [{"role": "user", "content": "q"}], "target": '<action>{"type": "finish"}</action>'}

    class Broken:
        max_new_tokens = 8

        def generate(self, messages):
            raise RuntimeError("CUDA out of memory")

    broken = evaluator.evaluate_teacher_forced(Broken(), [finish_row])
    assert broken["metrics"]["n_unmeasured"] == 1, broken["metrics"]
    assert broken["metrics"]["tool_call_exact_match"]["value"] == 0.0, broken["metrics"]
    assert broken["metrics"]["stop_accuracy"]["value"] == 0.0, broken["metrics"]

    silent = evaluator.evaluate_teacher_forced(Echo(), [dict(finish_row, messages=[{"role": "user", "content": "q"}])])
    assert silent["metrics"]["tool_call_exact_match"]["value"] == 0.0, "unparseable output must not score"

    # The per-record view is persisted verbatim into metrics.json, so it must not
    # contradict the aggregate on the same run.
    for payload in (broken, silent):
        assert payload["records"][0]["exact_match"] is False, payload["records"]
        assert payload["metrics"]["tool_call_exact_match"]["value"] == 0.0
    correct = evaluator.evaluate_teacher_forced(
        type("Ok", (), {"max_new_tokens": 8, "generate": lambda self, m: '<action>{"type": "finish"}</action>'})(),
        [finish_row],
    )
    assert correct["records"][0]["exact_match"] is True and correct["metrics"]["tool_call_exact_match"]["value"] == 1.0

# --------------------------------------------------------------------------- #
# evaluator self-check on the real dataset (Phase I)
# --------------------------------------------------------------------------- #
def _real_split():
    root = Path(__file__).resolve().parents[1] / "data" / "train"
    return root if (root / "train.jsonl").is_file() and (root / "zaic_tools").is_dir() else None

def test_closed_loop_on_real_labels_is_env_sourced_and_scores_one():
    """Gold replay through the official ToolEnv: exact match 1.0, rows from the env."""
    root = _real_split()
    if root is None:
        return  # dataset not extracted
    from lifemate.agent.agent import ToolAgent
    from lifemate.agent.protocol import FINISH_ACTION, render_action
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.data.loader import load_catalog, load_samples
    from lifemate.env.factory import create_environment
    from lifemate.evaluation import evaluator

    data_root = root.parent
    catalog = load_catalog(data_root, "train")
    samples = load_samples(data_root, "train", limit=4, with_labels=True)
    assert samples and all(s.label is not None for s in samples)
    envs: dict = {}

    class GoldReplay:
        max_new_tokens = 8

        def __init__(self):
            self.queue: list[str] = []

        def generate(self, messages):
            return self.queue.pop(0) if self.queue else render_action(FINISH_ACTION)

    generator = GoldReplay()

    def factory(sample):
        env = create_environment("train", str(sample.id), data_root, catalog=catalog)
        envs[str(sample.id)] = env                       # one env per sample, recorded
        calls = [c.to_dict() for c in (sample.label.tool_calls or [])]
        generator.queue = [render_action({"type": "tool_call", **c}) for c in calls] + [render_action(FINISH_ACTION)]
        return ToolAgent(generator=generator, registry=ToolRegistry(catalog).for_sample(sample), env=env)

    summary = evaluator.evaluate_closed_loop(factory, samples, split="train", data_root=data_root)
    assert summary["n_run"] == len(samples)
    assert len(envs) == len(samples) and len({id(e) for e in envs.values()}) == len(samples), "env reused across samples"
    assert summary["status_counts"] == {"finished": len(samples)}, summary["status_counts"]
    assert summary["decision_metrics"]["tool_call_exact_match"]["value"] == 1.0
    assert summary["final_state_correctness"]["available"] is False
    assert all(len(row["state_hash"]) == 64 and row["tool_calls"] for row in summary["rows"])
    # rows must equal what the env logged (never the agent's own bookkeeping)
    for row, sample in zip(summary["rows"], samples):
        gold = [c.to_dict() for c in (sample.label.tool_calls or [])]
        assert row["tool_calls"] == gold, row["id"]
        assert row["state_hash"] == envs[str(sample.id)].get_state_hash()

def test_gold_labels_replay_reproduces_official_transcript_and_state():
    """Replay gold `tool_calls` through ONE ToolEnv: transcript, results and final_state match."""
    root = _real_split()
    if root is None:
        return
    import json as _json

    from lifemate.env.factory import create_environment

    data_root = root.parent
    rows = [_json.loads(line) for line in (root / "train.jsonl").open(encoding="utf-8")][:12]
    checked_fs = 0
    for row in rows:
        env = create_environment("train", row["id"], data_root)     # exactly ONE env
        calls = row["label"]["tool_calls"]
        results = [env.call(c["name"], c.get("arguments") or {}) for c in calls]
        assert env.get_tool_calls() == calls, row["id"]
        assert results == row["label"]["tool_results"], row["id"]
        final_state = row["label"]["final_state"] or {}
        if final_state:
            state = env.get_state()
            assert {app: state.get(app) for app in final_state} == final_state, row["id"]
            checked_fs += 1
    assert checked_fs, "no sampled train row had a final_state label"

def test_evaluator_self_check_on_gold_input_marks_crashes_unmeasured():
    """ORACLE-on-gold self-check: 1.0; corrupted file < 1.0; crash -> n_unmeasured."""
    root = _real_split()
    if root is None:
        return
    from lifemate.agent.prompt_builder import StepRecord, build_decision_prompt, to_messages
    from lifemate.agent.protocol import FINISH_ACTION, render_action
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.data.loader import load_catalog, load_samples
    from lifemate.evaluation import evaluator

    data_root = root.parent
    catalog = load_catalog(data_root, "train")
    samples = load_samples(data_root, "train", limit=6, with_labels=True)
    records = []
    for sample in samples:
        registry = ToolRegistry(catalog).for_sample(sample)
        calls = [c.to_dict() for c in (sample.label.tool_calls or [])]
        results = list(sample.label.tool_results or [])
        steps = []
        for index, call in enumerate(calls):
            body = build_decision_prompt(
                transcript=sample.audio_text, context=sample.context, history=sample.history,
                installed_apps=sample.context.installed_apps, catalog=registry, prior_steps=steps,
            )
            observation = results[index].to_dict() if hasattr(results[index], "to_dict") else (results[index] or {})
            records.append({
                "id": f"{sample.id}#{index}", "messages": to_messages(body),
                "target": render_action({"type": "tool_call", **call}), "gold_tool_calls": [call],
            })
            steps.append(StepRecord(index=index, action={"type": "tool_call", **call}, observation=observation))
        body = build_decision_prompt(
            transcript=sample.audio_text, context=sample.context, history=sample.history,
            installed_apps=sample.context.installed_apps, catalog=registry, prior_steps=steps,
        )
        records.append({"id": f"{sample.id}#finish", "messages": to_messages(body),
                        "target": render_action(FINISH_ACTION), "gold_tool_calls": []})
    assert records

    class Queue:
        max_new_tokens = 8

        def __init__(self, outputs):
            self.outputs = list(outputs)

        def generate(self, messages):
            return self.outputs.pop(0)

    class Crasher:
        max_new_tokens = 8

        def generate(self, messages):
            raise RuntimeError("simulated crash")

    oracle = evaluator.evaluate_teacher_forced(Queue([r["target"] for r in records]), records)
    assert oracle["metrics"]["tool_call_exact_match"]["value"] == 1.0
    assert oracle["metrics"]["n_unmeasured"] == 0

    corrupted = [render_action({"type": "tool_call", "name": "alarm.list", "arguments": {}})
                 if r["gold_tool_calls"] else r["target"] for r in records]
    bad = evaluator.evaluate_teacher_forced(Queue(corrupted), records)
    assert 0.0 <= bad["metrics"]["tool_call_exact_match"]["value"] < 1.0
    assert all(r["exact_match"] is False for r in bad["records"] if r["gold"])

    crash = evaluator.evaluate_teacher_forced(Crasher(), records)
    assert crash["metrics"]["tool_call_exact_match"]["value"] == 0.0
    assert crash["metrics"]["n_unmeasured"] == len(records)
    assert all(r["exact_match"] is False for r in crash["records"])

    # A predictions/submission-shaped file is NOT gold.
    import tempfile as _tempfile
    path = Path(_tempfile.mkdtemp(prefix="lifemate_pred_")) / "pred.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({"id": sample.id, "tool_calls": [], "state_hash": "0" * 64}) + "\n")
    refused = evaluator.evaluate_teacher_forced(Queue([]), path)
    assert refused["metrics"]["available"] is False and refused["n_ungradeable"] == len(samples)

def test_evaluate_scripted_opt_in_requires_mock_or_explicit_flag():
    """The scripted generator is a TEST HOOK: mock, or the explicit pipeline-test flag."""
    import importlib.util

    from lifemate.config import load_config
    from lifemate.errors import LifeMateError

    spec = importlib.util.spec_from_file_location("lifemate_eval_script", ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    cfg = load_config()

    args = ev.parse_args(["--generator", "dumpscript:/tmp/x.json", "--validation-only"])
    try:
        ev._generator(cfg, args)
    except LifeMateError as exc:
        assert "TEST HOOK" in str(exc), exc
    else:
        raise AssertionError("--generator accepted without any opt-in")

    args = ev.parse_args(["--generator", "dumpscript:/tmp/x.json", "--allow-scripted-official"])
    assert args.allow_scripted_official is True

def test_evaluate_closed_loop_builds_one_env_per_sample():
    """ToolEnv may be created once per sample; the evaluator's preflight must not add one."""
    data_root = ROOT / "data"
    if not (data_root / "train" / "zaic_tools").is_dir():
        return
    import importlib.util

    import lifemate.env.factory as factory_mod

    spec = importlib.util.spec_from_file_location("lifemate_eval_envs", ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    counts: dict[str, int] = {}
    real = factory_mod.create_environment

    def counting(split, sample_id, data_root, **kwargs):
        counts[str(sample_id)] = counts.get(str(sample_id), 0) + 1
        return real(split, sample_id, data_root, **kwargs)

    directory = Path(tempfile.mkdtemp(prefix="lifemate_eval_env_"))
    dump = directory / "dumps.json"
    from lifemate.agent.protocol import FINISH_ACTION, render_action

    dump.write_text(json.dumps({"train_0001": [render_action(FINISH_ACTION)],
                                "train_0002": [render_action(FINISH_ACTION)]}), encoding="utf-8")
    factory_mod.create_environment = counting
    argv = ["--closed-loop", "--split", "train", "--data-root", str(data_root),
            "--generator", f"dumpscript:{dump}", "--allow-scripted-official",
            "--limit", "2", "--output", str(directory / "out"), "--model-mode", "base"]
    try:
        code = ev.main(argv)
    finally:
        factory_mod.create_environment = real
    assert code == 0, code
    assert counts == {"train_0001": 1, "train_0002": 1}, counts

def _load_verify_all_module():
    """Import ``scripts/verify_all.py`` without running it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_verify_all_probe", ROOT / "scripts" / "verify_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_q_probes_the_real_environment_and_never_passes():
    """Gate Q must be decided by probes, and must never report PASS.

    Two defects covered. (a) "real dataset" was appended unconditionally, so the
    gate claimed the dataset was missing after it had been extracted (1545 train /
    634 public_test rows, all exercised this session). (b) the ToolEnv probe was a
    bare top-level ``import zaic_tools``, which can never succeed because the SDK
    ships per split at ``<data_root>/<split>/zaic_tools``; the official adapter's
    ``check_official_sdk(split, root)`` is the correct session-free probe.
    """
    verify_all = _load_verify_all_module()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty = verify_all.probe_real_environment(str(root))
        assert "real dataset" in empty["blocked"], empty["blocked"]
        assert "real dataset" not in empty["verified"], empty["verified"]
        assert empty["probes"]["real_dataset"]["ok"] is False
        assert empty["probes"]["real_dataset"]["splits"] == []

        # A split directory alone (catalog only, no jsonl) is still a discovered
        # split, so the item must flip to verified without any SDK present.
        split = root / "train"
        split.mkdir()
        (split / "train_tools_catalog.json").write_text("{}", encoding="utf-8")
        present = verify_all.probe_real_environment(str(root))
        assert "real dataset" in present["verified"], present
        assert "real dataset" not in present["blocked"], present["blocked"]
        assert present["probes"]["real_dataset"]["splits"] == ["train"]
        # No per-split SDK under this root -> that item is blocked, with a reason.
        assert any("official ToolEnv" in item for item in present["blocked"]), present["blocked"]
        assert present["probes"]["official_toolenv"]["ok"] is False

        # A per-split SDK tree makes the ToolEnv probe succeed.
        sdk = split / "zaic_tools"
        sdk.mkdir()
        (sdk / "__init__.py").write_text("", encoding="utf-8")
        with_sdk = verify_all.probe_real_environment(str(root))
        assert with_sdk["probes"]["official_toolenv"]["ok"] is True, with_sdk["probes"]["official_toolenv"]
        assert with_sdk["probes"]["official_toolenv"]["verified_splits"] == ["train"]
        assert not any("official ToolEnv" in item for item in with_sdk["blocked"]), with_sdk["blocked"]

        # The official final-state scorer is always reported as an own item.
        assert any("final-state scorer" in item for item in with_sdk["blocked"]), with_sdk["blocked"]
        assert with_sdk["detail"].startswith("cannot verify here: ")
        assert "Verified here: " in with_sdk["detail"]

    result = verify_all.gate_q_blocked_real()
    assert result["status"] == "BLOCKED", result["status"]
    assert result["status"] != "PASS"
    assert isinstance(result["blocked"], list) and result["blocked"]
    assert "real dataset" in result["blocked"] or "real dataset" in result["verified"]
    print(f"  gate Q -> always BLOCKED, probes decide each item; blocked={result['blocked']}")

def test_footer_verdict_is_derived_from_gate_q_probes():
    """The footer verdict must not be a second, hand-written claim.

    It was the literal string ``REAL DATASET TESTED: no / ...`` while Gate Q's
    probe reported the dataset present, so the artifact contradicted itself. The
    footer now reads gate Q's own ``probes`` — one probe call, one truth — and
    this test is the anti-drift guard: a fake root with a split must produce
    ``yes (train ...)``, an empty root must produce ``no``.
    """
    verify_all = _load_verify_all_module()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        split = root / "train"
        split.mkdir()
        (split / "train.jsonl").write_text(
            json.dumps({"id": "s1", "context": {}, "history": [], "label": {"tool_calls": []}}) + "\n",
            encoding="utf-8",
        )
        present = verify_all.probe_real_environment(str(root))
        gate_q = {"gate": "Q", "status": "BLOCKED", "detail": present["detail"],
                  "blocked": present["blocked"], "verified": present["verified"],
                  "probes": present["probes"]}
        footer = verify_all.real_environment_summary([gate_q])
        assert "REAL DATASET TESTED: yes (train" in footer, footer
        assert "REAL DATASET TESTED: no" not in footer, footer
        assert "OFFICIAL SDK: no" in footer, footer  # no zaic_tools under this root

        empty = verify_all.probe_real_environment(str(root / "nope"))
        footer_empty = verify_all.real_environment_summary(
            [{"gate": "Q", "status": "BLOCKED", "probes": empty["probes"],
              "blocked": empty["blocked"], "verified": empty["verified"]}]
        )
        assert "REAL DATASET TESTED: no" in footer_empty, footer_empty
        assert "yes (train" not in footer_empty, footer_empty

    # A run without gate Q (e.g. --gates M) must say so rather than invent a claim.
    assert "not probed in this run" in verify_all.real_environment_summary(
        [{"gate": "M", "status": "PASS", "probes": {}}]
    )
    # And the live footer agrees with the live gate.
    live = verify_all.gate_q_blocked_real()
    live_footer = verify_all.real_environment_summary([live])
    assert ("REAL DATASET TESTED: yes" in live_footer) == bool(
        live["probes"]["real_dataset"]["ok"]
    ), live_footer
    print(f"  footer -> {live_footer}")

def test_entrypoint_wires_history_transcripts() -> None:
    """`HISTORY_TRANSCRIPTS` must reach `scripts/infer.py`, and fail loudly when unusable.

    The seam existed in `scripts/infer.py` but no deployment path passed it, so a
    container run fell back to `history transcripts: NONE configured` and 419 of
    634 `public_test` samples lost their prior turns. The entrypoint is run with a
    no-op `python` first on PATH so the built command can be inspected instead of
    executing inference.
    """
    entrypoint = ROOT / "docker" / "entrypoint.sh"
    assert entrypoint.is_file()
    assert subprocess.run(["bash", "-n", str(entrypoint)], capture_output=True).returncode == 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shim = tmp_path / "shim"
        shim.mkdir()
        fake_python = shim / "python"
        fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
        fake_python.chmod(0o755)
        env = {**os.environ, "PATH": f"{shim}:{os.environ.get('PATH', '')}"}

        transcripts = tmp_path / "history_transcripts.jsonl"
        transcripts.write_text('{"split": "public_test", "id": "x", "turn_index": 1, "text": "a"}\n',
                               encoding="utf-8")

        run = subprocess.run(
            ["bash", str(entrypoint)],
            capture_output=True, text=True, env={**env, "HISTORY_TRANSCRIPTS": str(transcripts)},
        )
        assert run.returncode == 0, run.stderr
        assert "--history-transcripts" in run.stdout, run.stdout
        assert str(transcripts) in run.stdout, run.stdout
        # The flag and its value must be adjacent (argv order matters to argparse).
        args = run.stdout.split("\n")
        assert args[args.index("--history-transcripts") + 1] == str(transcripts), args

        unset = subprocess.run(["bash", str(entrypoint)], capture_output=True, text=True, env=env)
        assert unset.returncode == 0, unset.stderr
        assert "--history-transcripts" not in unset.stdout, unset.stdout

        missing = subprocess.run(
            ["bash", str(entrypoint)],
            capture_output=True, text=True,
            env={**env, "HISTORY_TRANSCRIPTS": str(tmp_path / "absent.jsonl")},
        )
        assert missing.returncode == 2, (missing.returncode, missing.stdout, missing.stderr)
        assert "HISTORY_TRANSCRIPTS set but not readable" in missing.stderr, missing.stderr
        assert "--history-transcripts" not in missing.stdout, missing.stdout
    print("  entrypoint -> --history-transcripts passed when readable; exit 2 when unreadable")
