"""Tests for the per-sample tool registry (stdlib only).

Run with: python -m pytest tests/test_tool_registry.py -q
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent.prompt_builder import SYSTEM_PROMPT, build_decision_prompt, to_messages  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402

NEEDS_GPU: list[str] = []


def _catalog() -> ToolCatalog:
    return ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "app_name": "alarm",
                    "description": "Alarms.",
                    "tools": [
                        {
                            "name": "alarm.list",
                            "description": "List existing alarms.",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                        {
                            "name": "alarm.set",
                            "description": "Create an alarm.",
                            "annotations": {"readOnlyHint": False},
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "time": {"type": "string"},
                                    "repeat": {"type": "string", "enum": ["once", "daily"]},
                                },
                                "required": ["time"],
                            },
                        },
                    ],
                },
                {
                    "app_name": "messaging",
                    "description": "Messages.",
                    "tools": [
                        {
                            "name": "messaging.send",
                            "description": "Send a message.",
                            "parameters": {
                                "type": "object",
                                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                                "required": ["body"],
                            },
                        }
                    ],
                },
                {
                    "app_name": "sys",
                    "description": "System.",
                    "tools": [
                        {
                            "name": "sys.ask_missing_info",
                            "description": "Ask the user for missing info.",
                            "parameters": {
                                "type": "object",
                                "properties": {"question": {"type": "string"}},
                                "required": ["question"],
                            },
                        }
                    ],
                },
            ]
        }
    )


class _Ctx:
    def __init__(self, apps):
        self.installed_apps = list(apps)
        self.current_time = "2026-11-07 08:00"
        self.timezone = "Asia/Ho_Chi_Minh"
        self.current_location = "Hà Nội"
        self.user_profile = {"name": "An"}


class _Sample:
    def __init__(self, apps):
        self.id = "s1"
        self.audio_text = "đặt báo thức 7 giờ"
        self.context = _Ctx(apps)
        self.history = []


def test_only_installed_app_tools_visible():
    registry = ToolRegistry(_catalog()).for_sample(_Sample(["alarm"]))
    names = registry.tool_names()
    assert "alarm.set" in names and "alarm.list" in names
    assert "messaging.send" not in names, "uninstalled app tool leaked into the registry"


def test_meta_tool_always_present():
    registry = ToolRegistry(_catalog()).for_sample(_Sample(["messaging"]))
    assert "sys.ask_missing_info" in registry.tool_names()


def test_unknown_tool_rejected():
    registry = ToolRegistry(_catalog(), apps=["alarm"])
    assert registry.is_available("nope.nothing") is False
    report = registry.validate("nope.nothing", {})
    assert report["errors"], "unknown tool must produce an error"


def test_required_and_enum_validation_surfaced():
    registry = ToolRegistry(_catalog(), apps=["alarm"])
    missing = registry.validate("alarm.set", {})
    assert any("time" in e for e in missing["errors"])
    bad_enum = registry.validate("alarm.set", {"time": "07:00", "repeat": "hourly"})
    assert any("enum" in e for e in bad_enum["errors"])
    good = registry.validate("alarm.set", {"time": "07:00", "repeat": "daily"})
    assert good["errors"] == []


def test_rendered_prompt_has_no_uninstalled_tools():
    registry = ToolRegistry(_catalog(), apps=["alarm"])
    assert "messaging.send" not in registry.render()

    sample = _Sample(["alarm"])
    prompt = build_decision_prompt(
        transcript=sample.audio_text,
        context=sample.context,
        history=[],
        installed_apps=sample.context.installed_apps,
        catalog=registry,
        prior_steps=[],
    )
    assert "alarm.set" in prompt
    assert "messaging.send" not in prompt, "prompt leaked an uninstalled app tool"
    assert "sys.ask_missing_info" in prompt


def test_documents_and_len():
    registry = ToolRegistry(_catalog(), apps=["alarm"])
    docs = registry.documents
    assert all(isinstance(d, dict) and d.get("name") for d in docs)
    assert len(registry) == len(registry.tool_names())


def test_sample_as_dict_also_filters():
    registry = ToolRegistry(_catalog()).for_sample({"context": {"installed_apps": ["messaging"]}})
    assert registry.tool_names() == ["messaging.send", "sys.ask_missing_info"]


def test_to_messages_shape():
    messages = to_messages("body")
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "body"},
    ]


def _unseen_catalog() -> ToolCatalog:
    """Hand-written split-B catalog: ``pharmacy`` exists in no other catalog here.

    Mirrors the competition's real gap (11 of 36 ``public_test`` apps never appear
    in ``train``): nothing registers this app on the training side, so selection
    and rendering must work from the catalog alone.
    """
    return ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "app_name": "pharmacy",
                    "description": "Nhà thuốc.",
                    "tools": [
                        {
                            "name": "pharmacy.order",
                            "description": "Đặt thuốc.",
                            "kind": "write",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "item_id": {"type": "string"},
                                    "delivery": {"type": "string", "enum": ["pickup", "courier"]},
                                    "note": {"type": "string"},
                                },
                                "required": ["item_id"],
                                "x-one-of": [["note"], ["delivery"]],
                            },
                        }
                    ],
                },
                {
                    "app_name": "sys",
                    "description": "System.",
                    "tools": [
                        {
                            "name": "sys.ask_missing_info",
                            "description": "Ask the user for missing info.",
                            "parameters": {
                                "type": "object",
                                "properties": {"question": {"type": "string"}},
                                "required": ["question"],
                            },
                        }
                    ],
                },
            ]
        }
    )

def test_unseen_app_tool_renders_full_schema_and_is_selectable():
    """A tool known only to the current split's catalog is fully usable, unregistered."""
    from lifemate.agent.tool_parser import parse_model_output

    registry = ToolRegistry(_unseen_catalog()).for_sample(_Sample(["pharmacy"]))
    assert registry.is_available("pharmacy.order")

    prompt = build_decision_prompt(
        transcript="mua thuốc",
        context=_Ctx(["pharmacy"]),
        history=[],
        installed_apps=["pharmacy"],
        catalog=registry,
        prior_steps=[],
    )
    assert "### pharmacy.order (kind=write)" in prompt, "declared kind must be rendered"
    block = prompt.split("### pharmacy.order (kind=write)")[1].split("\n## ")[0]
    assert "required: item_id" in block, "required flag must be rendered"
    assert '"enum": ["pickup", "courier"]' in block, "enum must survive rendering"
    assert '"x-one-of": [["note"], ["delivery"]]' in block, "x-one-of must survive rendering"
    assert '"item_id": {"type": "string"}' in block, "argument type must be rendered"

    # Selection and argument validation both come from the same catalog.
    accepted = parse_model_output(
        '<action>{"type": "tool_call", "name": "pharmacy.order", '
        '"arguments": {"item_id": "d1", "delivery": "courier"}}</action>',
        registry,
    )
    assert accepted.type == "tool_call", accepted.error
    missing_group = parse_model_output(
        '<action>{"type": "tool_call", "name": "pharmacy.order", "arguments": {"item_id": "d1"}}</action>',
        registry,
    )
    assert missing_group.type == "invalid", "x-one-of group must be enforced"

def test_registry_and_parser_follow_the_current_splits_catalog():
    """Two catalogs in one process: no train-side registration, no cached view."""
    from lifemate.agent.tool_parser import parse_model_output

    call = '<action>{"type": "tool_call", "name": "pharmacy.order", "arguments": {"item_id": "d1", "note": "gấp"}}</action>'
    test_side = ToolRegistry(_unseen_catalog(), apps=["pharmacy"])
    train_side = ToolRegistry(_catalog(), apps=["pharmacy"])
    assert test_side.is_available("pharmacy.order")
    assert not train_side.is_available("pharmacy.order")
    assert parse_model_output(call, test_side).type == "tool_call"
    rejected = parse_model_output(call, train_side)
    assert rejected.type == "invalid" and "pharmacy.order" in rejected.error

def test_needs_gpu_report():
    assert NEEDS_GPU == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("SKIP: none - ran all registry tests inline")
