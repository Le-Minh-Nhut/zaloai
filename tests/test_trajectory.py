"""Causal SFT construction tests: alignment, causality, leakage, splits, determinism.

Stdlib only (no torch/transformers/pytest fixtures). Every test is a zero-arg
function so ``python3 tests/run_tests.py`` finds it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.agent.prompt_builder import StepRecord, build_decision_prompt  # noqa: E402
from lifemate.agent.protocol import FINISH_ACTION, render_action  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.data.schemas import Context, HistoryTurn, Label, Sample, ToolCall, ToolResult  # noqa: E402
from lifemate.data.split_builder import split_samples  # noqa: E402
from lifemate.data.trajectory_builder import (  # noqa: E402
    DecisionPoint,
    align_calls_and_results,
    assert_no_label_leak,
    build_decision_points,
    build_sft_dataset,
    from_jsonl,
    to_hf_dataset_records,
    to_jsonl,
    tool_result_to_observation,
)
from lifemate.training.collator import build_supervised_example  # noqa: E402

CATALOG_PATH = FIXTURES / "train" / "train_tools_catalog.json"
APPS = ["alarm", "calendar", "smart_home", "messaging"]

# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _catalog() -> ToolCatalog:
    return ToolCatalog.from_file(CATALOG_PATH)

def _sample(
    sample_id: str = "s1",
    calls: list[tuple[str, dict]] | None = None,
    results: list[dict | None] | None = None,
    apps: list[str] | None = None,
    transcript: str = "Dat bao thuc luc 6 gio sang mai.",
    final_state: dict | None = None,
    history: list | None = None,
) -> Sample:
    """Build a synthetic Sample with a gold label of the requested shape."""
    call_objs = [ToolCall(name=name, arguments=dict(args)) for name, args in (calls or [])]
    result_objs = [
        ToolResult(name=call_objs[i].name, ok=True, result={"marker": f"obs_{i}"})
        if entry is None
        else ToolResult.from_raw(entry)
        for i, entry in enumerate(results or [])
    ]
    label = Label(
        tool_calls=call_objs,
        tool_results=[r for r in result_objs if r is not None],
        final_state=final_state,
    )
    context = Context(
        current_time="2026-11-07T09:30:00+07:00",
        timezone="Asia/Ho_Chi_Minh",
        current_location="Ho Chi Minh City",
        installed_apps=list(APPS if apps is None else apps),
    )
    return Sample(
        id=sample_id,
        split="train",
        audio_path=None,
        audio_text=transcript,
        context=context,
        history=list(history or []),
        label=label,
        raw={"category": "test"},
    )

def _labelled_results(calls: list[tuple[str, dict]], ok: bool = True) -> list[dict]:
    return [
        {"name": name, "ok": ok, "result": {"marker": f"obs_{index}"}}
        for index, (name, _args) in enumerate(calls)
    ]

# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def test_alignment_equal_counts() -> None:
    """Equal call/result counts pair 1:1 with no issues."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"})]
    alignment = align_calls_and_results(_sample(calls=calls, results=_labelled_results(calls)))
    assert alignment.n_steps == 2
    assert alignment.truncated is False
    assert alignment.unaligned_results == []
    assert all(result is not None for _call, result in alignment.steps)
    assert alignment.issues == [], alignment.issues
    print("  equal counts -> 1:1, no issues")

def test_alignment_truncated_results_never_borrow() -> None:
    """Fewer results than calls: prefix pairs, tail is MISSING, nothing is borrowed."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"}), ("alarm.cancel", {"id": "a1"})]
    results = [{"name": "alarm.list", "ok": True, "result": {"marker": "obs_0"}}]
    alignment = align_calls_and_results(_sample(calls=calls, results=results))
    assert alignment.truncated is True
    assert alignment.n_steps == 3
    assert alignment.steps[0][1] is not None
    assert alignment.steps[1][1] is None, "step 1 must be MISSING, not borrow a later result"
    assert alignment.steps[2][1] is None
    observation = tool_result_to_observation(alignment.steps[1][1])
    assert observation == {"ok": False, "error": "missing_tool_result"}, observation
    print("  truncated results -> prefix aligned, tail MISSING, no borrowing")

def test_alignment_extra_results_are_ignored() -> None:
    """More results than calls: extras are listed but never attached."""
    calls = [("alarm.list", {})]
    results = [
        {"name": "alarm.list", "ok": True, "result": {"marker": "obs_0"}},
        {"name": "alarm.set", "ok": True, "result": {"marker": "obs_EXTRA"}},
    ]
    alignment = align_calls_and_results(_sample(calls=calls, results=results))
    assert alignment.n_steps == 1
    assert len(alignment.unaligned_results) == 1
    assert "obs_EXTRA" not in json.dumps(
        tool_result_to_observation(alignment.steps[0][1]), ensure_ascii=False
    )
    assert any("extra_results" in issue for issue in alignment.issues), alignment.issues
    print("  extra results -> unaligned + ignored, never attached")

def test_alignment_name_mismatch_flagged() -> None:
    """A result whose name differs from the call at that position is an issue."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"})]
    results = [
        {"name": "alarm.set", "ok": True, "result": {}},
        {"name": "alarm.list", "ok": True, "result": {}},
    ]
    alignment = align_calls_and_results(_sample(calls=calls, results=results))
    mismatches = [issue for issue in alignment.issues if "name_mismatch" in issue]
    assert len(mismatches) == 2, alignment.issues
    # pairing stays positional on purpose: remapping would be a guess
    assert alignment.steps[0][1].name == "alarm.set"
    print("  name mismatch -> flagged twice, positional pairing preserved")

def test_alignment_out_of_order_and_unknown_ok() -> None:
    """Explicit indices that disagree, and a missing ``ok``, are both reported."""
    sample = _sample(
        calls=[("alarm.list", {}), ("alarm.set", {"time": "06:30"})],
        results=[
            {"name": "alarm.list", "ok": True, "result": {}},
            {"name": "alarm.set", "ok": True, "result": {}},
        ],
    )
    sample.label.raw = {
        "tool_results": [
            {"name": "alarm.list", "ok": True, "result": {}, "index": 0},
            {"name": "alarm.set", "ok": True, "result": {}, "index": 5},
        ]
    }
    alignment = align_calls_and_results(sample)
    assert any("result_out_of_order" in issue for issue in alignment.issues), alignment.issues

    # ToolResult.from_raw infers ok=True when no error field is present, so an
    # unknown ok only happens when a record genuinely omits the flag *and* the
    # error field: build that shape directly.
    unknown = _sample(calls=[("alarm.list", {})], results=[])
    unknown.label.tool_results = [ToolResult(name="alarm.list", ok=None, result={"marker": "obs_0"})]
    unknown_alignment = align_calls_and_results(unknown)
    assert any("ok_unknown" in issue for issue in unknown_alignment.issues), unknown_alignment.issues
    # ok unknown + no error field => treated as success (documented rule)
    assert tool_result_to_observation(unknown_alignment.steps[0][1])["ok"] is True

    unknown_error = _sample(calls=[("alarm.list", {})], results=[])
    unknown_error.label.tool_results = [ToolResult(name="alarm.list", ok=None, error="nope")]
    assert tool_result_to_observation(unknown_error.label.tool_results[0])["ok"] is False
    print("  out-of-order index -> flagged; unknown ok -> flagged (+ defaults)")

def test_tool_result_to_observation_rules() -> None:
    """Every documented branch of the observation converter."""
    assert tool_result_to_observation(None) == {"ok": False, "error": "missing_tool_result"}
    assert tool_result_to_observation(ToolResult(ok=True, result={"a": 1})) == {"ok": True, "result": {"a": 1}}
    assert tool_result_to_observation(ToolResult(ok=False, error="boom")) == {"ok": False, "error": "boom"}
    assert tool_result_to_observation(ToolResult(ok=False)) == {"ok": False, "error": "tool_error"}
    assert tool_result_to_observation(ToolResult(ok=None, result={"a": 1})) == {"ok": True, "result": {"a": 1}}
    assert tool_result_to_observation(ToolResult(ok=None, error="nope")) == {"ok": False, "error": "nope"}
    print("  observation converter -> all 6 documented branches")

# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
def test_no_future_observation_in_prompt() -> None:
    """Prompt k contains observations of steps < k only."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"}), ("alarm.cancel", {"id": "a_9"})]
    sample = _sample(calls=calls, results=_labelled_results(calls))
    points = build_decision_points(sample, _catalog())
    assert [p.step_index for p in points] == [0, 1, 2, 3], [p.step_index for p in points]

    for point in points:
        for index in range(len(calls)):
            marker = f"obs_{index}"
            if index < point.step_index:
                assert marker in point.prompt_body, f"DP{point.step_index} lost obs_{index}"
            else:
                assert marker not in point.prompt_body, (
                    f"LEAK: DP{point.step_index} shows future observation obs_{index}"
                )
    print("  3-step trajectory -> prompt k has obs < k only (no future leakage)")

def test_causal_ordering_preserved() -> None:
    """Observations appear in step order inside the prompt."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"}), ("alarm.cancel", {"id": "a_9"})]
    sample = _sample(calls=calls, results=_labelled_results(calls))
    prompt = build_decision_points(sample, _catalog())[-1].prompt_body
    positions = [prompt.index(f"obs_{index}") for index in range(len(calls))]
    assert positions == sorted(positions), positions
    print("  final prompt -> observations in causal order")

def test_terminal_finish_point() -> None:
    """A 2-call trajectory yields 2 call points + 1 finish point."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"})]
    points = build_decision_points(_sample(calls=calls, results=_labelled_results(calls)), _catalog())
    assert points[-1].target_action == FINISH_ACTION, points[-1].target_action
    assert points[-1].step_index == 2
    assert points[-1].meta["is_terminal"] is True
    for point in points[:-1]:
        assert point.target_action["type"] == "tool_call"
        assert point.meta["is_terminal"] is False
    print("  2 calls -> targets [call, call, finish]")

def test_ask_missing_info_is_terminal() -> None:
    """A trajectory ending in ask_missing_info gets NO finish point."""
    calls = [("sys.ask_missing_info", {"missing_fields": ["start_time"]})]
    results = [{"name": "sys.ask_missing_info", "ok": True, "result": {"question": "Luc may gio?"}}]
    points = build_decision_points(_sample(calls=calls, results=results), _catalog())
    assert len(points) == 1, [p.target_action for p in points]
    assert points[0].target_action["name"] == "sys.ask_missing_info"
    assert all(p.target_action != FINISH_ACTION for p in points)
    print("  ends with ask_missing_info -> [ask_missing_info], no finish")

def test_ask_missing_info_midtrajectory_is_ordinary() -> None:
    """ask_missing_info as a non-final call is just another step."""
    calls = [("alarm.list", {}), ("sys.ask_missing_info", {"missing_fields": ["time"]})]
    results = [
        {"name": "alarm.list", "ok": True, "result": {"marker": "obs_0"}},
        {"name": "sys.ask_missing_info", "ok": True, "result": {"question": "May gio?"}},
    ]
    points = build_decision_points(_sample(calls=calls, results=results), _catalog())
    assert len(points) == 2, [p.step_index for p in points]
    assert points[1].target_action["name"] == "sys.ask_missing_info"
    print("  ask_missing_info mid-trajectory -> ordinary step, still terminal rule applied")

def test_ask_missing_alias_is_terminal() -> None:
    """The meta tool stays terminal when the catalog spells it without ``sys.``.

    The catalog normalises ``sys.ask_missing_info`` and a ``sys``-app
    ``ask_missing_info`` to ``kind == "meta"``. Checking the literal name instead
    of the resolved spec appended a ``finish`` point right after a question, which
    teaches the model to hang up instead of waiting for the user's answer.
    """
    catalog = ToolCatalog.from_dict(
        {
            "apps": [
                {"app_name": "alarm", "tools": [{"name": "alarm.list", "description": "List alarms."}]},
                {
                    "app_name": "sys",
                    "tools": [
                        {
                            "name": "ask_missing_info",
                            "description": "Ask the user for missing information.",
                        }
                    ],
                },
            ]
        }
    )
    sample = _sample(calls=[("ask_missing_info", {"missing_fields": ["time"]})],
                     results=[{"name": "ask_missing_info", "ok": True, "result": {"question": "May gio?"}}])
    points = build_decision_points(sample, catalog, registry=ToolRegistry(catalog).for_sample(sample))
    assert len(points) == 1, [(p.step_index, p.target_action) for p in points]
    assert points[0].target_action["name"] == "ask_missing_info"

    # A normal read tool of the same trajectory still earns its finish point.
    plain = _sample(calls=[("alarm.list", {})], results=[{"name": "alarm.list", "ok": True, "result": {}}])
    assert len(build_decision_points(plain, catalog, registry=ToolRegistry(catalog).for_sample(plain))) == 2
    print("  ask_missing_info alias -> terminal (no finish point); normal read unchanged")

def test_zero_call_sample_single_finish() -> None:
    """An out-of-scope request yields exactly one finish decision point."""
    points = build_decision_points(
        _sample(sample_id="oos", calls=[], results=[], transcript="Dat ve may bay di Ha Noi."),
        _catalog(),
    )
    assert len(points) == 1
    assert points[0].target_action == FINISH_ACTION
    assert points[0].prompt_body.count("# Các bước đã thực hiện") == 0
    assert points[0].meta["n_steps_total"] == 0
    print("  zero calls -> single finish target, no step section")

def test_missing_last_result_skips_finish() -> None:
    """A trajectory whose last observation is missing gets no finish point."""
    calls = [("alarm.list", {}), ("alarm.set", {"time": "06:30"})]
    results = [{"name": "alarm.list", "ok": True, "result": {"marker": "obs_0"}}]
    points = build_decision_points(_sample(calls=calls, results=results), _catalog())
    assert [p.step_index for p in points] == [0, 1], [p.step_index for p in points]
    assert all(p.target_action != FINISH_ACTION for p in points)
    assert all(p.meta["visibility_limited"] for p in points)
    print("  missing final result -> no finish point, visibility_limited")

def test_failed_result_is_observed_as_error() -> None:
    """A failed result reaches later prompts as {ok: false, error: ...}."""
    calls = [("alarm.set", {"time": "06:30"}), ("alarm.list", {})]
    results = [
        {"name": "alarm.set", "ok": False, "error": "malformed time"},
        {"name": "alarm.list", "ok": True, "result": {"marker": "obs_1"}},
    ]
    points = build_decision_points(_sample(calls=calls, results=results), _catalog())
    assert '"ok": false' in points[1].prompt_body
    assert "malformed time" in points[1].prompt_body
    print("  failed call -> error observation visible in the next prompt")

def test_unobservable_ref_arg_flagged() -> None:
    """A write argument not present anywhere visible is flagged, never injected."""
    calls = [("messaging.send", {"contact_id": "c_secret_42", "body": "hi"})]
    results = [{"name": "messaging.send", "ok": True, "result": {"sent": True}}]
    sample = _sample(calls=calls, results=results)
    points = build_decision_points(sample, _catalog())
    flagged = points[0].meta.get("unobservable_ref_args") or []
    assert any("c_secret_42" in item for item in flagged), flagged
    assert "unobservable_ref" in points[0].meta["sample_flags"]
    # the gold value is in the *target* (that is the label) but the prompt must
    # not gain it from anywhere else: only the target block carries it
    body = points[0].prompt_body
    assert "c_secret_42" not in body
    print("  unobservable x-ref arg -> meta flag, prompt untouched")

def test_unobservable_ref_arg_x_match_without_id_suffix() -> None:
    """A reference-annotated property lacking an ``_id`` suffix is still a reference.

    ``smart_home.device_set.device_id`` carries ``x-ref`` *and* ``x-match``; the
    ``x-ref`` is what makes it a reference (``x-match`` is free-form text — see
    :func:`lifemate.data.trajectory_builder._ref_keys`). The point of the test is
    that the annotation, not the name suffix, decides.
    """
    catalog = _catalog()
    calls = [("smart_home.device_set", {"device_id": "dev_unseen_99", "on": True})]
    results = [{"name": "smart_home.device_set", "ok": True, "result": {"updated": True}}]
    points = build_decision_points(_sample(calls=calls, results=results, apps=["smart_home"]), catalog)
    flagged = points[0].meta.get("unobservable_ref_args") or []
    assert any("dev_unseen_99" in item for item in flagged), flagged
    assert "unobservable_ref" in points[0].meta["sample_flags"]
    assert "dev_unseen_99" not in points[0].prompt_body

def test_observable_ref_arg_not_flagged() -> None:
    """A value that a previous observation exposed is not flagged."""
    calls = [("messaging.search_contacts", {"query": "Lan"}), ("messaging.send", {"contact_id": "c_lan", "body": "hi"})]
    results = [
        {"name": "messaging.search_contacts", "ok": True, "result": {"contacts": [{"id": "c_lan"}]}},
        {"name": "messaging.send", "ok": True, "result": {"sent": True}},
    ]
    points = build_decision_points(_sample(calls=calls, results=results), _catalog())
    assert not points[1].meta.get("unobservable_ref_args"), points[1].meta.get("unobservable_ref_args")
    assert "unobservable_ref" not in points[1].meta["sample_flags"]
    print("  observable x-ref arg -> not flagged")

def test_ref_arg_only_in_dropped_history_is_unobservable() -> None:
    """A value living beyond the rendered history window must be reported.

    Regression: ``_unobservable_ref_args`` searched *every* history turn while
    :func:`build_decision_prompt` renders only the last ``history_limit`` turns,
    so a gold ID that only existed in an earlier turn was declared observable
    and supervised. Real train data: 13 samples / 19 arguments (e.g.
    ``train_1597 email.archive.email_id=mail_5310``, the value produced in turn 3
    of a 20-turn history).
    """
    old_turn = HistoryTurn(
        role="assistant",
        content="Dạ danh sách liên hệ: Lan.",
        tool_call={"tool_calls": [{"name": "messaging.search_contacts", "arguments": {}}]},
        tool_result={"tool_results": [{"ok": True, "contacts": [{"id": "c_old"}]}]},
    )
    padding = [HistoryTurn(role="user", content=f"lượt {i}") for i in range(12)]
    calls = [("messaging.send", {"contact_id": "c_old", "body": "hi"})]
    results = [{"name": "messaging.send", "ok": True, "result": {"sent": True}}]
    sample = _sample(calls=calls, results=results, history=[old_turn] + padding)

    points = build_decision_points(sample, _catalog(), history_limit=10)
    assert "c_old" not in points[0].prompt_body, "the dropped turn must not be rendered"
    assert points[0].meta.get("unobservable_ref_args") == ["step0:messaging.send.contact_id=c_old"]

    kept, stats = build_sft_dataset([sample], _catalog(), history_limit=10)
    assert [p.target_action for p in kept] == [{"type": "finish"}], [p.target_action for p in kept]
    assert stats["dropped_decision_points"] == {"unobservable_ref_target": 1}, stats["dropped_decision_points"]
    assert stats["n_dropped"] == 0, stats["dropped"]

    # Control: the same value inside the window IS observable -> kept, no drop.
    visible = _sample(calls=calls, results=results, history=[old_turn])
    points = build_decision_points(visible, _catalog(), history_limit=10)
    assert not points[0].meta.get("unobservable_ref_args")
    kept_visible, ok_stats = build_sft_dataset([visible], _catalog(), history_limit=10)
    assert len(kept_visible) == 2 and ok_stats["n_dropped_decision_points"] == 0
    print("  ref value beyond the history window -> target dropped; inside -> kept")

def test_label_final_state_never_in_prompt() -> None:
    """final_state (and its structure) never reaches a prompt."""
    final_state = {"apps": {"alarm": {"alarms": [{"id": "a_77", "time": "06:30"}], "count": 1}}}
    calls = [("alarm.set", {"time": "06:30"})]
    results = [{"name": "alarm.set", "ok": True, "result": {"alarm_id": "a_77"}}]
    sample = _sample(calls=calls, results=results, final_state=final_state)
    points = build_decision_points(sample, _catalog())
    for point in points:
        assert "final_state" not in point.prompt_body
        assert '"count": 1' not in point.prompt_body
        assert "a_77" not in point.prompt_body or "alarm_id" in point.prompt_body
    print("  final_state -> absent from every prompt")

def test_assert_no_label_leak_catches_injection() -> None:
    """The guard hard-fails on a label-only key or a copied final_state."""
    final_state = {"apps": {"alarm": {"alarms": [{"id": "zz_leak"}], "count": 1}}, "notes": {"n": 2}}
    canonical = json.dumps(final_state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    raisers = 0
    for body in (
        'State: {"final_state": {}}',
        canonical,
        'Bước 2: kết quả {"alarms":[{"id":"zz_leak"}],"count":1}',  # a copied state slice
    ):
        try:
            assert_no_label_leak(body, final_state)
        except Exception as exc:
            assert "leak" in str(exc).lower(), str(exc)
            raisers += 1
    assert raisers == 3, raisers

    # a legitimate prompt passes
    assert_no_label_leak("# Yêu cầu\nDat bao thuc\n\nKết quả: {\"ok\": true, \"result\": {}}", final_state)
    print("  assert_no_label_leak -> 3 injections caught, clean prompt passes")

# --------------------------------------------------------------------------- #
# dataset assembly / stats
# --------------------------------------------------------------------------- #
def test_stats_accounting_on_fixture() -> None:
    """Fixture split: every sample accounted for, drop reasons exact."""
    samples = load_samples(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    points, stats = build_sft_dataset(samples, catalog)
    assert stats["n_samples"] == len(samples), stats["n_samples"]
    assert stats["n_dropped"] == 0, stats["dropped"]
    assert stats["n_decision_points"] == len(points)
    assert stats["n_samples_kept"] + stats["n_dropped"] == stats["n_samples"]
    assert sum(stats["trajectory_length_histogram"].values()) == stats["n_samples"]
    assert stats["target_tokens_estimate"]["max"] > 0
    # 13 fixture samples: 11 with 1-2 calls (+1 finish each) + 2 zero-call finish
    # points = 30 built, minus 4 targets whose reference value the prompt cannot
    # show (fixture ids that exist nowhere: `c_404`, `c_an`, `light_meeting`, and
    # `ac_1` at step 0 — step 1 of that sample is fine because step 0's own
    # action already displays `ac_1`).
    assert stats["n_decision_points"] == 26, stats["n_decision_points"]
    assert stats["n_dropped_decision_points"] == 4, stats["dropped_decision_points"]
    assert stats["dropped_decision_points"] == {"unobservable_ref_target": 4}, stats["dropped_decision_points"]
    print(f"  fixture stats -> {stats['n_decision_points']} points, 0 dropped samples, 4 dropped targets")

def test_stats_drop_reasons() -> None:
    """no_label / zero_calls_no_finish_possible / catalog_missing_tool / empty_prompt."""
    catalog = _catalog()
    no_label = _sample(sample_id="nolabel", calls=[("alarm.list", {})])
    no_label.label = None

    stray_results = _sample(sample_id="stray", calls=[], results=[{"ok": True, "result": {}}])

    missing_tool = _sample(
        sample_id="missing", calls=[("no_such_app.do", {})], results=[{"ok": True, "result": {}}]
    )

    uninstalled = _sample(
        sample_id="uninstalled",
        calls=[("alarm.list", {})],
        results=[{"ok": True, "result": {}}],
        apps=["messaging"],
    )

    # zero calls + no request text + no context fields => nothing to condition on
    empty = _sample(sample_id="empty", calls=[], results=[], transcript="")
    empty.context = Context()

    _points, stats = build_sft_dataset([no_label, stray_results, missing_tool, uninstalled, empty], catalog)
    dropped = stats["dropped"]
    assert dropped["no_label"] == 1, dropped
    assert dropped["zero_calls_no_finish_possible"] == 1, dropped
    assert dropped["catalog_missing_tool"] == 2, dropped  # unknown tool + uninstalled app
    assert dropped["empty_prompt"] == 1, dropped
    assert stats["n_dropped"] == 5
    print("  drop reasons -> no_label/zero_calls/catalog_missing_tool/empty_prompt counted")

def test_unexecutable_gold_target_is_dropped() -> None:
    """A gold call missing a required argument never becomes a supervised target.

    The supervised target is what inference must emit; the agent loop rejects an
    action that fails catalog validation before it reaches the env, so training on
    it would teach an action that can never execute. The missing-information case
    belongs to ``sys.ask_missing_info``, not to a fabricated argument.
    """
    calls = [("alarm.set", {})]  # alarm.set requires "time"
    sample = _sample(calls=calls, results=_labelled_results(calls))
    points, stats = build_sft_dataset([sample], _catalog())
    assert points == [], [p.target_text() for p in points]
    assert stats["dropped"]["unexecutable_gold_target"] == 1, stats["dropped"]

    # A valid call of the same tool is kept: the drop is about executability only.
    ok_calls = [("alarm.set", {"time": "06:30"})]
    good = _sample(calls=ok_calls, results=_labelled_results(ok_calls))
    kept, ok_stats = build_sft_dataset([good], _catalog())
    assert len(kept) == 2 and ok_stats["dropped"]["unexecutable_gold_target"] == 0
    print("  unexecutable gold target -> dropped; the executable twin is kept")

def test_misaligned_results_emit_no_decision_points() -> None:
    """A result provably belonging to another call step is never used as a past.

    With equal call/result counts, positional pairing is normally safe. An
    explicit ``index`` that disagrees turns step k's observation into a *future*
    observation as soon as the order is reversed, so no prompt built from this
    record can be trusted.
    """
    calls = [("alarm.set", {"time": "06:00"}), ("alarm.set", {"time": "07:00"})]
    sample = _sample(calls=calls, results=_labelled_results(calls))
    sample.label.raw = {
        "tool_results": [
            {"index": 1, "name": "alarm.set", "ok": True, "result": {"marker": "obs_0"}},
            {"index": 0, "name": "alarm.set", "ok": True, "result": {"marker": "obs_1"}},
        ]
    }
    alignment = align_calls_and_results(sample)
    assert alignment.misaligned is True, alignment.issues
    assert build_decision_points(sample, _catalog()) == []
    points, stats = build_sft_dataset([sample], _catalog())
    assert points == []
    assert stats["dropped"]["misaligned_results"] == 1, stats["dropped"]

    # Control: a *name* difference alone must NOT discard the sample. Result
    # names are free-form in real data ("list" vs "alarm.list"), so a mismatch is
    # a quality flag, not proof that the observation belongs to another call.
    named = _sample(calls=calls, results=_labelled_results(calls))
    named.label.tool_results[0].name = "list"
    named_points = build_decision_points(named, _catalog())
    assert len(named_points) == 3, [p.target_action for p in named_points]
    assert "name_mismatch" in named_points[0].meta["sample_flags"]
    assert align_calls_and_results(named).misaligned is False
    kept_named, named_stats = build_sft_dataset([named], _catalog())
    assert len(kept_named) == 3 and named_stats["dropped"]["misaligned_results"] == 0
    print("  misaligned results -> zero decision points; a name mismatch alone is only flagged")

def test_jsonl_round_trip_and_determinism() -> None:
    """Two builds are byte-identical; to_jsonl→from_jsonl round-trips."""
    samples = load_samples(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    first, _ = build_sft_dataset(samples, catalog)
    second, _ = build_sft_dataset(samples, catalog)

    with tempfile.TemporaryDirectory() as tmp:
        a = to_jsonl(first, Path(tmp) / "a.jsonl")
        b = to_jsonl(second, Path(tmp) / "b.jsonl")
        assert a.read_bytes() == b.read_bytes(), "trajectory build is not deterministic"
        restored = from_jsonl(a)
        assert len(restored) == len(first)
        for left, right in zip(first, restored):
            assert left.sample_id == right.sample_id
            assert left.step_index == right.step_index
            assert left.prompt_body == right.prompt_body
            assert left.target_action == right.target_action
            assert left.meta == right.meta
        records = to_hf_dataset_records(restored)
        assert set(records[0]) == {"sample_id", "step_index", "messages", "target", "meta"}
    print(f"  determinism -> identical bytes; round-trip of {len(first)} points exact")

def test_records_are_self_contained() -> None:
    """A record's messages/target reproduce the decision point without rebuilding."""
    samples = load_samples(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    points, _ = build_sft_dataset(samples, catalog)
    record = points[0].to_record()
    assert [m["role"] for m in record["messages"]] == ["system", "user"]
    assert record["messages"][1]["content"] == points[0].prompt_body
    assert record["target"] == render_action(points[0].target_action)
    assert record["meta"]["token_estimate"] > 0
    assert record["meta"]["source"] == "train"
    print("  record -> self-contained messages+target, meta carries estimates")

def test_decision_point_meta_fields() -> None:
    """Every decision point carries source/category/estimates/counters."""
    samples = load_samples(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    points, _ = build_sft_dataset(samples, catalog)
    for point in points:
        for key in ("token_estimate", "n_steps_total", "source", "category", "n_calls_bucket", "sample_flags"):
            assert key in point.meta, (point.sample_id, key)
    print("  meta -> required keys present on all points")

# --------------------------------------------------------------------------- #
# split builder
# --------------------------------------------------------------------------- #
def test_split_deterministic_and_disjoint() -> None:
    """Same seed → same split; train and val share no duplicate group."""
    samples = load_samples(FIXTURES, "train")
    first = split_samples(samples, ratio=0.2, seed=7)
    second = split_samples(samples, ratio=0.2, seed=7)
    assert [s.id for s in first.train] == [s.id for s in second.train]
    assert [s.id for s in first.val] == [s.id for s in second.val]
    first.assert_no_overlap()
    manifest = first.manifest()
    assert manifest["n_train"] + manifest["n_val"] == len(samples), manifest
    assert manifest["seed"] == 7 and manifest["ratio"] == 0.2
    assert len(manifest["train_ids_hash"]) == 64
    assert first.val, "ratio 0.2 must produce a non-empty val set"
    print(f"  split -> deterministic, {manifest['n_train']} train / {manifest['n_val']} val, no overlap")

def test_split_keeps_duplicates_together() -> None:
    """Identical normalised requests land in the same split (leakage guard)."""
    duplicates = [
        _sample(
            sample_id=f"dup_{index}",
            calls=[("alarm.list", {})],
            results=[{"name": "alarm.list", "ok": True, "result": {"marker": f"m{index}"}}],
            transcript="\u0110\u1eb7t b\u00e1o th\u1ee9c l\u00fac 6 gi\u1edd s\u00e1ng mai.",
        )
        for index in range(6)
    ]
    # same request, diacritics stripped: normalises to the same hash
    duplicates.append(
        _sample(
            sample_id="dup_unaccented",
            calls=[("alarm.list", {})],
            results=[{"name": "alarm.list", "ok": True, "result": {"marker": "m0"}}],
            transcript="dat bao thuc luc 6 gio sang mai.",
        )
    )
    distinct = [
        _sample(
            sample_id=f"distinct_{index}",
            calls=[("alarm.list", {})],
            results=[{"name": "alarm.list", "ok": True, "result": {"marker": "x"}}],
            transcript=f"Yeu cau khac nhau so {index}.",
        )
        for index in range(10)
    ]
    result = split_samples(duplicates + distinct, ratio=0.3, seed=3)
    sides = {
        split_name
        for split_name, group in (("train", result.train), ("val", result.val))
        for sample in group
        if sample.id.startswith("dup_")
    }
    assert len(sides) == 1, f"duplicate group spans both splits: {sides}"
    assert sum(1 for s in result.train + result.val if s.id.startswith("dup_")) == 7
    result.assert_no_overlap()
    print(f"  duplicates -> all 7 in one split ({sides.pop()}), no leakage")

def test_split_stratified_by_call_bucket() -> None:
    """Buckets are represented in train; small buckets never disappear."""
    zero = [
        _sample(sample_id=f"zero_{i}", calls=[], results=[], transcript=f"Ranh {i}.")
        for i in range(6)
    ]
    one = [
        _sample(
            sample_id=f"one_{i}",
            calls=[("alarm.list", {})],
            results=[{"name": "alarm.list", "ok": True, "result": {}}],
            transcript=f"Liet ke bao thuc {i}.",
        )
        for i in range(6)
    ]
    three = [
        _sample(
            sample_id=f"three_{i}",
            calls=[("alarm.list", {})] * 3,
            results=[{"name": "alarm.list", "ok": True, "result": {}}] * 3,
            transcript=f"Ba buoc {i}.",
        )
        for i in range(6)
    ]
    result = split_samples(zero + one + three, ratio=0.34, seed=11)
    buckets = {"zero_": 0, "one_": 0, "three_": 0}
    for sample in result.train:
        for prefix in buckets:
            if sample.id.startswith(prefix):
                buckets[prefix] += 1
    assert all(count >= 3 for count in buckets.values()), buckets
    assert result.val, "val must not be empty"
    print(f"  stratification -> every call bucket kept in train {buckets}")

def test_split_ratio_zero_keeps_everything() -> None:
    """ratio=0 puts every sample in train."""
    samples = load_samples(FIXTURES, "train")
    result = split_samples(samples, ratio=0.0, seed=1)
    assert result.val == []
    assert len(result.train) == len(samples)
    print("  ratio 0.0 -> everything in train")

# --------------------------------------------------------------------------- #
# prompt fidelity
# --------------------------------------------------------------------------- #
def test_prompt_matches_inference_renderer() -> None:
    """DP prompts equal a direct build_decision_prompt call (train == inference)."""
    sample = load_samples(FIXTURES, "train")[0]
    catalog = load_catalog(FIXTURES, "train")
    registry = ToolRegistry(catalog).for_sample(sample)
    points = build_decision_points(sample, catalog, registry=registry)
    direct = build_decision_prompt(
        transcript=sample.audio_text,
        context=sample.context,
        history=sample.history,
        installed_apps=sample.installed_apps,
        catalog=registry,
        prior_steps=[],
    )
    assert points[0].prompt_body == direct, "train prompt drifted from inference renderer"
    print("  prompt bytes identical to inference renderer")

def test_installed_app_filtering() -> None:
    """Only tools of installed apps (+ meta) reach the prompt."""
    catalog = _catalog()
    sample = _sample(calls=[("alarm.list", {})], results=[{"name": "alarm.list", "ok": True, "result": {}}], apps=["alarm"])
    prompt = build_decision_points(sample, catalog)[0].prompt_body
    assert "alarm.list" in prompt
    assert "calendar.create_event" not in prompt, "uninstalled app tool leaked into the prompt"
    print("  installed apps -> uninstalled tools absent from the prompt")

# --------------------------------------------------------------------------- #
# collator hand-off (collator lives in training, tested here once)
# --------------------------------------------------------------------------- #
class _CharTokenizer:
    """Deterministic char-level tokenizer stub."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [2 + (ord(ch) % 90) for ch in text]

def test_collator_masks_prompt_and_keeps_target() -> None:
    """labels[-100] prefix length equals the prompt token count; target intact."""
    tokenizer = _CharTokenizer()
    point = build_decision_points(
        _sample(calls=[("alarm.list", {})], results=[{"name": "alarm.list", "ok": True, "result": {}}]),
        _catalog(),
    )[0]
    example = build_supervised_example(tokenizer, point.messages(), point.target_text(), 512)
    assert example["kept"] is True, example
    prompt_len = example["n_prompt_tokens"]
    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert example["labels"][prompt_len:] == example["input_ids"][prompt_len:]
    assert example["labels"][-1] == tokenizer.eos_token_id
    assert len(example["input_ids"]) == len(example["labels"])
    print(f"  collator -> prompt {prompt_len} masked, target {example['n_target_tokens']} supervised")

def test_fixture_trajectories_strict_causality_and_no_leak() -> None:
    """Every fixture decision point obeys causal ordering and never leaks label.final_state."""
    samples = load_samples(FIXTURES, "train", with_labels=True)
    catalog = load_catalog(FIXTURES, "train")
    registry = ToolRegistry(catalog)
    assert len(samples) > 0

    for sample in samples:
        points = build_decision_points(sample, catalog, registry=registry)
        for i, point in enumerate(points):
            # 1. Terminal finish points come last
            if point.target_action.get("type") == "finish":
                assert i == len(points) - 1, f"finish at {i} not last ({len(points)}) in {sample.id}"

            # 2. Step k prompt contains no observation of step >= k; results never borrowed forward
            for later_step_num in range(i + 1, len(points) + 2):
                assert f"Bước {later_step_num}:" not in point.prompt_body, (
                    f"step {i} prompt has future Bước {later_step_num} in {sample.id}"
                )
            for later_idx in range(i, len(points)):
                later_obs = points[later_idx].observation_after
                if later_obs and isinstance(later_obs, dict):
                    res = later_obs.get("result")
                    if isinstance(res, dict) and "marker" in res:
                        assert res["marker"] not in point.prompt_body, (
                            f"step {i} prompt has marker from step {later_idx} in {sample.id}"
                        )
            # 3. label.final_state never appears in any prompt
            prompt_text = point.prompt_body
            if sample.label and sample.label.final_state:
                final_state_str = json.dumps(sample.label.final_state, ensure_ascii=False)
                assert final_state_str not in prompt_text, f"final_state leaked in {sample.id} step {i}"
                for k_sub, v_sub in sample.label.final_state.items():
                    if isinstance(v_sub, (dict, list)) and len(json.dumps(v_sub)) > 15:
                        assert json.dumps(v_sub, ensure_ascii=False) not in prompt_text, f"final_state sub leaked"

def main() -> int:
    """Run every test in this module (stdlib runner parity)."""
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - test harness
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
