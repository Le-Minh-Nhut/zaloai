"""Slot-masking and tool-dropout: preconditions, identity, and registry safety.

Runs under the stdlib runner (``python3 tests/run_tests.py``). The catalog used
here is a real-looking fixture: ``sms.send`` requires ``contact_id`` *and*
``body``, so a slot can be dropped from a request that states one of them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate.agent.prompt_builder import build_decision_prompt  # noqa: E402
from lifemate.agent.tool_registry import META_TOOL_NAME, ToolRegistry  # noqa: E402
from lifemate.agent.protocol import parse_action  # noqa: E402
from lifemate.data.augmentation import (  # noqa: E402
    apply_slot_masking,
    apply_tool_dropout,
    prompt_section,
    TOOLS_HEADER,
)
from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.data.trajectory_builder import DecisionPoint, StepRecord, to_hf_dataset_records  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402

_CATALOG_RAW = {
    "apps": [
        {
            "app_name": "sms",
            "tools": [
                {
                    "name": "sms.send",
                    "description": "Send a text message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contact_id": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["contact_id", "body"],
                    },
                },
                {
                    "name": "sms.list_threads",
                    "description": "List message threads.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            ],
        },
        {
            "app_name": "alarm",
            "tools": [
                {
                    "name": "alarm.set",
                    "description": "Create an alarm.",
                    "parameters": {
                        "type": "object",
                        "properties": {"time": {"type": "string"}},
                        "required": ["time"],
                    },
                }
            ],
        },
        {
            "app_name": "sys",
            "tools": [
                {
                    "name": "sys.ask_missing_info",
                    "kind": "meta",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "params": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["tool_name", "params"],
                    },
                }
            ],
        },
    ]
}

CATALOG = ToolCatalog.from_dict(_CATALOG_RAW)
APPS = ["alarm", "sms"]
TRANSCRIPT = "Gửi tin nhắn cho An. Nội dung là mai họp lúc 9 giờ."

def _prompt(
    transcript: str = TRANSCRIPT,
    history: list | None = None,
    prior_steps: list | None = None,
    catalog: ToolCatalog | None = None,
) -> str:
    view = catalog if catalog is not None else CATALOG
    return build_decision_prompt(
        transcript=transcript,
        context={
            "installed_apps": APPS,
            "current_time": "2026-11-07T09:30:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
        },
        history=history or [],
        installed_apps=APPS,
        catalog=ToolRegistry(view, apps=APPS),
        prior_steps=prior_steps or [],
    )

def _record(
    sample_id: str = "s1",
    *,
    transcript: str = TRANSCRIPT,
    history: list | None = None,
    prior_steps: list | None = None,
    name: str = "sms.send",
    arguments: dict | None = None,
    catalog: ToolCatalog | None = None,
    step_index: int = 0,
) -> dict:
    point = DecisionPoint(
        sample_id=sample_id,
        step_index=step_index,
        prompt_body=_prompt(transcript, history, prior_steps, catalog),
        target_action={
            "type": "tool_call",
            "name": name,
            "arguments": {"contact_id": "An", "body": "mai họp lúc 9 giờ"} if arguments is None else arguments,
        },
    )
    return to_hf_dataset_records([point])[0]

def _only(outcome: list[dict], records: list[dict]) -> dict:
    assert len(outcome) == len(records) + 1, f"expected exactly one variant, got {len(outcome)}"
    return outcome[-1]

# --------------------------------------------------------------------------- #
# Disabled path
# --------------------------------------------------------------------------- #
def test_slot_masking_disabled_is_identity():
    records = [_record()]
    outcome, report = apply_slot_masking(records, enabled=False, catalog=CATALOG)
    assert outcome is records
    assert report == {}

def test_tool_dropout_disabled_is_identity():
    records = [_record()]
    outcome, report = apply_tool_dropout(records, enabled=False, catalog=CATALOG)
    assert outcome is records
    assert report == {}

def test_all_flags_off_keeps_the_same_objects_and_bytes():
    records = [_record("s1"), _record("s2")]
    before = [(id(record), record["messages"][-1]["content"], record["target"]) for record in records]
    slot, slot_report = apply_slot_masking(records, enabled=False, catalog=CATALOG)
    drop, drop_report = apply_tool_dropout(records, enabled=False, catalog=CATALOG)
    assert slot is records and drop is records
    assert slot_report == {} and drop_report == {}
    assert [(id(r), r["messages"][-1]["content"], r["target"]) for r in records] == before

def test_enabled_slot_masking_without_a_catalog_raises():
    try:
        apply_slot_masking([_record()], enabled=True)
    except LifeMateError as exc:
        assert "catalog" in str(exc)
    else:
        raise AssertionError("slot masking must require a catalog")

# --------------------------------------------------------------------------- #
# Slot masking: accepting case
# --------------------------------------------------------------------------- #
def test_slot_masking_accepts_and_targets_the_meta_tool():
    records = [_record()]
    outcome, report = apply_slot_masking(records, enabled=True, seed=5, catalog=CATALOG)
    variant = _only(outcome, records)
    assert report["n_variants_created"] == 1
    # Only "An" is stated as a literal in the request; "mai họp lúc 9 giờ" also is,
    # so the first *sorted* visible required key is the one dropped.
    assert report["per_record"] == []
    action = parse_action(variant["target"])
    assert action.name == META_TOOL_NAME
    assert set(action.arguments) == {"tool_name", "params"}
    assert action.arguments["tool_name"] == "sms.send"
    # Required keys are tried in sorted order; "body" is the first whose value
    # ("mai họp lúc 9 giờ") is a literal in the request.
    assert action.arguments["params"] == ["body"]
    assert CATALOG.validate_call(META_TOOL_NAME, action.arguments)["errors"] == []

def test_slot_masking_removes_the_value_from_the_request_only():
    records = [_record()]
    outcome, _report = apply_slot_masking(records, enabled=True, catalog=CATALOG)
    variant = _only(outcome, records)
    request = prompt_section(variant["messages"][-1]["content"], "# Yêu cầu")
    assert "mai họp lúc 9 giờ" not in request
    assert request.strip() == "Gửi tin nhắn cho An."
    # Everything after the request section is byte-identical to the original.
    original = records[0]["messages"][-1]["content"]
    tail = original.index("\n\n# Ngữ cảnh")
    assert variant["messages"][-1]["content"].endswith(original[tail:])
    assert variant["meta"]["augmentation"] == "slot_masking"
    assert variant["meta"]["slot_masked_argument"] == "body"
    assert variant["meta"]["slot_masked_value"] == "mai họp lúc 9 giờ"
    assert variant["meta"]["gold_target_replaced"] is True
    assert variant["sample_id"] == records[0]["sample_id"]
    assert variant["step_index"] == records[0]["step_index"]

def test_slot_masking_marks_the_original_and_keeps_it_first():
    records = [_record()]
    outcome, _report = apply_slot_masking(records, enabled=True, catalog=CATALOG)
    assert outcome[0] is records[0]
    assert "slot_masked_argument" not in outcome[0]["meta"]

# --------------------------------------------------------------------------- #
# Slot masking: one rejecting case per rule
# --------------------------------------------------------------------------- #
def test_rule1_finish_target_is_skipped():
    record = _record()
    record["target"] = '<action>{"type": "finish"}</action>'
    outcome, report = apply_slot_masking([record], enabled=True, catalog=CATALOG)
    assert len(outcome) == 1
    assert report["skips_by_rule"] == {"1": 1}
    assert report["per_record"][0]["reason"] == "1:target_is_not_a_tool_call"

def test_rule2_argument_not_required_is_skipped():
    record = _record(name="sms.list_threads", arguments={})
    outcome, report = apply_slot_masking([record], enabled=True, catalog=CATALOG)
    assert len(outcome) == 1
    assert report["per_record"][0]["reason"] == "2:tool_declares_no_required_argument"

def test_rule3_value_absent_from_the_request_is_skipped():
    record = _record(arguments={"contact_id": "Zzz", "body": "Zzz khác"})
    outcome, report = apply_slot_masking([record], enabled=True, catalog=CATALOG)
    assert len(outcome) == 1
    assert report["per_record"][0]["reason"] == "3:no_required_value_is_a_literal_in_request"

def test_rule3_single_clause_request_is_skipped_not_gutted():
    """Removing the only clause would leave no request at all: skip, never gut."""
    record = _record(transcript="Nhắn An.")
    outcome, report = apply_slot_masking([record], enabled=True, catalog=CATALOG)
    assert len(outcome) == 1
    assert report["per_record"][0]["rule"] == 3
    assert report["per_record"][0]["reason"] in {
        "3:clause_not_cleanly_removable",
        "3:no_required_value_is_a_literal_in_request",
    }

def test_rule4_skips_only_the_value_in_history_and_uses_the_other():
    """Per-candidate rule 4: a reused value is skipped, another one is dropped."""
    history = [{"role": "user", "content": "Gửi cho An nhé."}]
    record = _record(history=history)
    outcome, report = apply_slot_masking([record], enabled=True, catalog=CATALOG)
    assert report["n_variants_created"] == 1
    variant = outcome[-1]
    assert variant["meta"]["slot_masked_argument"] == "body"
    assert variant["meta"]["slot_masked_value"] == "mai họp lúc 9 giờ"

def test_rule4_value_already_in_history_is_skipped():
    """Every visible required value is in the history, so none may be dropped."""
    history = [{"role": "user", "content": "Gửi cho An, nội dung mai họp lúc 9 giờ."}]
    record = _record(history=history)
    outcome, report = apply_slot_masking([record], enabled=True, catalog=CATALOG)
    assert len(outcome) == 1
    assert report["per_record"][0]["reason"] == "4:value_already_in_history"

def test_rule5_meta_tool_missing_from_the_catalog_is_skipped():
    no_meta = ToolCatalog.from_dict(
        {"apps": [app for app in _CATALOG_RAW["apps"] if app["app_name"] != "sys"]}
    )
    record = _record(catalog=no_meta, arguments={"contact_id": "An", "body": "body"})
    outcome, report = apply_slot_masking([record], enabled=True, catalog=no_meta)
    assert len(outcome) == 1
    assert report["per_record"][0]["reason"] == "5:meta_tool_not_shown_in_prompt"
    # ... and with the meta tool absent from the catalog entirely.
    store_only = ToolCatalog.from_dict(
        {
            "apps": [
                _CATALOG_RAW["apps"][0],
                {"app_name": "sys", "tools": [{"name": "sys.echo", "kind": "read", "parameters": {}}]},
            ]
        }
    )
    record = _record(catalog=store_only, arguments={"contact_id": "An", "body": "body"})
    outcome, report = apply_slot_masking([record], enabled=True, catalog=store_only)
    assert len(outcome) == 1
    assert report["per_record"][0]["reason"] in {
        "5:meta_tool_not_shown_in_prompt",
        "5:meta_tool_absent_from_catalog",
    }

def test_rule5_unfillable_meta_schema_is_skipped_not_invented():
    """A meta schema whose arrays are not strings-of-params cannot be filled safely."""
    weird = ToolCatalog.from_dict(
        {
            "apps": [
                _CATALOG_RAW["apps"][0],
                {
                    "app_name": "sys",
                    "tools": [
                        {
                            "name": "sys.ask_missing_info",
                            "kind": "meta",
                            "parameters": {
                                "type": "object",
                                "properties": {"question": {"type": "object"}},
                                "required": ["question"],
                            },
                        }
                    ],
                },
            ]
        }
    )
    record = _record(catalog=weird)
    outcome, report = apply_slot_masking([record], enabled=True, catalog=weird)
    assert len(outcome) == 1
    assert report["per_record"][0]["reason"] == "5:meta_schema_unfillable_without_invention"

def test_every_skip_names_its_rule_number():
    records = [
        _record("a"),
        _record("b", name="sms.list_threads", arguments={}),
        _record("c", arguments={"contact_id": "Zzz", "body": "Zzz"}),
    ]
    _outcome, report = apply_slot_masking(records, enabled=True, catalog=CATALOG)
    assert report["n_records_in"] == 3
    for skip in report["per_record"]:
        assert skip["rule"] in {1, 2, 3, 4, 5}
        assert skip["reason"].startswith(f"{skip['rule']}:")
    assert sum(report["skips_by_rule"].values()) == report["n_skipped"]

# --------------------------------------------------------------------------- #
# Tool dropout
# --------------------------------------------------------------------------- #
def test_tool_dropout_removes_only_unused_tools():
    records = [_record()]
    outcome, report = apply_tool_dropout(records, enabled=True, seed=3, catalog=CATALOG)
    variant = _only(outcome, records)
    assert report["n_tools_dropped"] == 1
    dropped = variant["meta"]["tools_dropped"]
    assert len(dropped) == 1
    assert dropped[0] != "sms.send"  # the target's own tool
    assert dropped[0] != META_TOOL_NAME
    tools = prompt_section(variant["messages"][-1]["content"], TOOLS_HEADER)
    assert f"### {dropped[0]}" not in tools
    assert "### sms.send" in tools
    assert variant["target"] == records[0]["target"]

def test_tool_dropout_never_removes_the_gold_target_tool():
    """Negative guard: ``sms.send`` is a droppable candidate except for the guard.

    ``max_tools=2`` makes the target tool reachable for removal (its app has two
    tools, it is not the meta tool), so this assertion fails the moment the
    protected-name check is deleted from the transform.
    """
    record = _record(name="sms.send")
    outcome, report = apply_tool_dropout([record], enabled=True, max_tools=2, catalog=CATALOG)
    variant = outcome[-1]
    assert report["n_variants_created"] == 1
    assert "sms.send" not in variant["meta"]["tools_dropped"]
    assert variant["meta"]["tools_dropped"] == ["sms.list_threads"]
    assert variant["target"] == record["target"]
    assert "### sms.send" in prompt_section(variant["messages"][-1]["content"], TOOLS_HEADER)

def test_tool_dropout_protects_tools_used_by_prior_steps():
    step = StepRecord(
        index=0,
        action={"type": "tool_call", "name": "sms.list_threads", "arguments": {}},
        observation={"ok": True},
    )
    record = _record(prior_steps=[step], name="alarm.set", arguments={"time": "06:30"})
    outcome, _report = apply_tool_dropout([record], enabled=True, catalog=CATALOG)
    variant = _only(outcome, [record])
    tools = prompt_section(variant["messages"][-1]["content"], TOOLS_HEADER)
    assert "### sms.list_threads" in tools  # shown in the steps section as well
    assert "### alarm.set" in tools
    assert "sms.list_threads" not in variant["meta"]["tools_dropped"]

def test_tool_dropout_keeps_installed_apps_consistent():
    """``alarm`` has one tool: dropping it while the app stays installed is inconsistent."""
    record = _record(name="sms.send")
    for seed in range(30):
        outcome, _report = apply_tool_dropout([record], enabled=True, seed=seed, catalog=CATALOG)
        if len(outcome) == 1:
            continue
        variant = outcome[-1]
        assert "alarm.set" not in variant["meta"]["tools_dropped"]
        context = prompt_section(variant["messages"][-1]["content"], "# Ngữ cảnh")
        assert "installed_apps: alarm, sms" in context

def test_tool_dropout_respects_max_tools():
    records = [_record()]
    one, report_one = apply_tool_dropout(records, enabled=True, max_tools=1, catalog=CATALOG)
    assert report_one["n_tools_dropped"] == 1
    for max_tools in (0, -1):
        try:
            apply_tool_dropout(records, enabled=True, max_tools=max_tools, catalog=CATALOG)
        except LifeMateError as exc:
            assert "max_tools" in str(exc)
        else:
            raise AssertionError("max_tools < 1 must raise")

def test_tool_dropout_choice_is_deterministic_and_order_independent():
    records = [_record("a"), _record("b"), _record("c")]
    first, _report = apply_tool_dropout(records, enabled=True, seed=9, catalog=CATALOG)
    second, _report2 = apply_tool_dropout(list(reversed(records)), enabled=True, seed=9, catalog=CATALOG)
    by_id_first = {
        record["sample_id"]: record["meta"]["tools_dropped"]
        for record in first
        if record["meta"].get("augmentation") == "tool_dropout"
    }
    by_id_second = {
        record["sample_id"]: record["meta"]["tools_dropped"]
        for record in second
        if record["meta"].get("augmentation") == "tool_dropout"
    }
    assert by_id_first == by_id_second

def test_tool_dropout_skips_a_prompt_it_cannot_reproduce():
    """A prompt rendered from another catalog state is skipped, never half-rewritten."""
    wide = ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "app_name": "sms",
                    "tools": _CATALOG_RAW["apps"][0]["tools"]
                    + [{"name": "sms.extra", "parameters": {"type": "object"}}],
                },
                _CATALOG_RAW["apps"][1],
                _CATALOG_RAW["apps"][2],
            ]
        }
    )
    record = _record(catalog=wide)  # prompt lists sms.extra, which CATALOG does not know
    outcome, report = apply_tool_dropout([record], enabled=True, catalog=CATALOG)
    assert len(outcome) == 1
    assert report["skips_by_reason"] == {"tools_section_not_reproducible_from_catalog": 1}

def test_tool_dropout_never_touches_the_inference_registry():
    """The transform edits prompt bytes only; the registry the agent uses is intact."""
    before = ToolRegistry(CATALOG, apps=APPS)
    before_names = before.tool_names()
    before_render = before.render()
    records = [_record()]
    outcome, _report = apply_tool_dropout(records, enabled=True, catalog=CATALOG)
    assert len(outcome) == 2  # a variant was produced from the same catalog
    after = ToolRegistry(CATALOG, apps=APPS)
    assert after.tool_names() == before_names
    assert after.render() == before_render
    assert set(before_names) == {"sms.send", "sms.list_threads", "alarm.set", META_TOOL_NAME}
    # The same fresh registry still renders every tool into a new prompt.
    fresh = _prompt()
    for name in before_names:
        assert f"### {name}" in fresh

def test_dropout_variant_target_bytes_are_verbatim():
    record = _record()
    outcome, _report = apply_tool_dropout([record], enabled=True, catalog=CATALOG)
    variant = outcome[-1]
    assert variant["target"] == record["target"]
    assert json.dumps(variant["target"]) == json.dumps(record["target"])
    assert variant["meta"]["augmentation"] == "tool_dropout"
