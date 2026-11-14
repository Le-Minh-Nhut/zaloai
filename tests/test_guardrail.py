"""Guardrail tests: enum/range/type/unknown/placeholder + env conflict. Stdlib only.

Run with: python3 tests/run_tests.py test_guardrail.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent.guardrail import GuardrailOutcome, check_action  # noqa: E402
from lifemate.agent.tool_parser import parse_model_output  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402

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
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "time": {"type": "string"},
                                    "repeat": {"type": "string", "enum": ["once", "daily"]},
                                    "hour": {"type": "integer", "minimum": 0, "maximum": 23},
                                    "volume": {
                                        "type": "integer",
                                        "exclusiveMinimum": 0,
                                        "exclusiveMaximum": 11,
                                    },
                                    "enabled": {"type": "boolean"},
                                    "label": {"type": "string"},
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
            ]
        }
    )

def _registry(apps=("alarm",)) -> ToolRegistry:
    return ToolRegistry(_catalog(), apps=list(apps))

def _call(name: str = "alarm.set", **arguments) -> dict:
    return {"type": "tool_call", "name": name, "arguments": arguments}

class _Env:
    """Stand-in for ``zaic_tools.ToolEnv`` exposing only ``validate``."""

    def __init__(self, errors: list[str] | None = None) -> None:
        self.errors = list(errors or [])
        self.calls: list[tuple[str, dict]] = []

    def validate(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return {"errors": list(self.errors)}

# ------------------------------------------------------------- disabled path

def test_disabled_returns_the_same_object_unchanged():
    registry = _registry()
    action = _call(hour=25)
    outcome = check_action(action, registry)
    assert outcome.verdict == "unchanged"
    assert outcome.reasons == ()
    assert outcome.action is action, "a disabled call must not copy or touch the action"
    assert outcome.original_action is None
    assert outcome.env_said_ok is False

def test_disabled_finish_is_also_unchanged():
    outcome = check_action({"type": "finish"}, _registry())
    assert outcome.verdict == "unchanged"
    assert outcome.action == {"type": "finish"}

# ---------------------------------------------------------------- acceptances

def test_accepting_case_per_check():
    registry = _registry()
    cases = [
        _call(time="07:00", repeat="daily"),          # enum ok
        _call(time="07:00", hour=0),                  # inclusive minimum
        _call(time="07:00", hour=23),                 # inclusive maximum
        _call(time="07:00", volume=1),                # exclusive minimum, above
        _call(time="07:00", volume=10),               # exclusive maximum, below
        _call(time="07:00", enabled=True),            # boolean type
        _call(time="07:00", label="báo thức"),        # declared optional string
    ]
    for action in cases:
        outcome = check_action(action, registry, enabled=True)
        assert outcome.verdict == "accepted", (action, outcome.reasons)
        assert outcome.reasons == ()
        assert outcome.action is action

def test_finish_is_accepted():
    action = {"type": "finish"}
    outcome = check_action(action, _registry(), enabled=True)
    assert outcome.verdict == "accepted"
    assert outcome.action is action

# ------------------------------------------------------------------ rejections

def test_enum_violation_is_rejected_and_names_the_alternatives():
    outcome = check_action(_call(time="07:00", repeat="hourly"), _registry(), enabled=True)
    assert outcome.verdict == "rejected"
    assert len(outcome.reasons) == 1, outcome.reasons
    reason = outcome.reasons[0]
    assert reason.startswith("enum: repeat"), reason
    assert "once" in reason and "daily" in reason, reason
    assert outcome.original_action == _call(time="07:00", repeat="hourly")

def test_range_violation_is_rejected_and_never_remapped():
    outcome = check_action(_call(time="07:00", hour=25), _registry(), enabled=True)
    assert outcome.verdict == "rejected"
    assert outcome.reasons == ("range: hour=25 outside 0..23",), outcome.reasons
    assert "hour=25" in outcome.reasons[0], "the offending value must be reported as-is"
    assert outcome.action["arguments"]["hour"] == 25, "25 must NOT be rewritten to 5"
    assert outcome.original_action["arguments"]["hour"] == 25

def test_exclusive_bounds_are_named():
    registry = _registry()
    low = check_action(_call(time="07:00", volume=0), registry, enabled=True)
    assert low.verdict == "rejected"
    assert low.reasons == ("range: volume=0 outside the exclusive minimum 0",), low.reasons
    high = check_action(_call(time="07:00", volume=11), registry, enabled=True)
    assert high.reasons == ("range: volume=11 outside the exclusive maximum 11",), high.reasons

def test_type_mismatch_is_rejected_without_coercion():
    registry = _registry()
    cases = (
        (_call(time="07:00", hour="25"), "type: hour expected integer, got str"),
        (_call(time="07:00", enabled="yes"), "type: enabled expected boolean, got str"),
        (_call(time="07:00", hour=False), "type: hour expected integer, got bool"),
        (_call(time=7), "type: time expected string, got int"),
    )
    for action, expected in cases:
        outcome = check_action(action, registry, enabled=True)
        assert outcome.verdict == "rejected", action
        assert expected in outcome.reasons, (action, outcome.reasons)
        assert outcome.action is action, "a rejected action is never rewritten"

def test_schema_permitted_coercion_is_normalized_not_rejected():
    """A whole JSON number where an integer is declared: coerced, and recorded."""
    action = _call(time="07:00", hour=7.0)
    outcome = check_action(action, _registry(), enabled=True)
    assert outcome.verdict == "normalized", outcome.reasons
    assert outcome.reasons == ("normalized: hour=7.0 -> 7 (integer declared)",)
    assert outcome.action["arguments"]["hour"] == 7
    assert isinstance(outcome.action["arguments"]["hour"], int)
    assert outcome.original_action["arguments"]["hour"] == 7.0
    assert action["arguments"]["hour"] == 7.0, "the caller's action is not mutated"
    fractional = check_action(_call(time="07:00", hour=7.5), _registry(), enabled=True)
    assert fractional.verdict == "rejected", "7.5 is not a whole number: no coercion"

def test_unsupported_argument_is_rejected():
    outcome = check_action(_call(time="07:00", sneaky="x"), _registry(), enabled=True)
    assert outcome.verdict == "rejected"
    assert outcome.reasons == ("unsupported: unknown argument 'sneaky'",), outcome.reasons

def test_placeholders_are_rejected_for_required_arguments():
    registry = _registry()
    for value in ("<time>", "<...>", "...", "TODO", "None", "null", ""):
        outcome = check_action(_call(time=value), registry, enabled=True)
        assert outcome.verdict == "rejected", value
        assert outcome.reasons and "unrecognized placeholder" in outcome.reasons[0], outcome.reasons
        assert "time" in outcome.reasons[0]

def test_placeholder_inside_a_list_is_rejected():
    catalog = ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "app_name": "notes",
                    "description": "Notes.",
                    "tools": [
                        {
                            "name": "notes.add",
                            "description": "Add a note.",
                            "parameters": {
                                "type": "object",
                                "properties": {"tags": {"type": "array"}},
                                "required": ["tags"],
                            },
                        }
                    ],
                }
            ]
        }
    )
    registry = ToolRegistry(catalog, apps=["notes"])
    outcome = check_action(
        {"type": "tool_call", "name": "notes.add", "arguments": {"tags": ["ok", "<tag>"]}},
        registry,
        enabled=True,
    )
    assert outcome.verdict == "rejected", outcome.reasons
    assert "placeholder" in outcome.reasons[0]

def test_optional_placeholder_is_not_rejected():
    """The official validator reads an empty optional value as 'not given'."""
    outcome = check_action(_call(time="07:00", label=""), _registry(), enabled=True)
    assert outcome.verdict == "accepted", outcome.reasons

def test_reasons_are_sorted_by_argument_name():
    outcome = check_action(
        _call(repeat="hourly", time="07:00", hour=25), _registry(), enabled=True
    )
    assert outcome.verdict == "rejected"
    assert outcome.reasons == (
        "enum: repeat='hourly' not in ['once', 'daily']",
        "range: hour=25 outside 0..23",
    ), outcome.reasons

# --------------------------------------------------------------- tool names

def test_unavailable_tool_is_rejected_and_the_parser_agrees():
    registry = _registry(apps=["alarm"])
    action = _call(name="messaging.send", body="hi")
    guard = check_action(action, registry, enabled=True)
    assert guard.verdict == "rejected"
    assert guard.reasons == (
        "tool: 'messaging.send' is not available for this sample; available: alarm.list, alarm.set",
    ), guard.reasons

    rendered = '<action>{"type": "tool_call", "name": "messaging.send", "arguments": {"body": "hi"}}</action>'
    parsed = parse_model_output(rendered, registry)
    assert parsed.type == "invalid", "the existing parser already rejects this tool"
    assert "not available" in (parsed.error or "")
    assert len(guard.reasons) == 1, "the guardrail reports the tool once, not twice"

def test_guardrail_does_not_double_report_parser_rejections():
    """A protocol-level failure stays a single reason, prefixed ``protocol:``."""
    outcome = check_action({"type": "invalid", "error": "bad json"}, _registry(), enabled=True)
    assert outcome.verdict == "rejected"
    assert outcome.reasons == ("protocol: bad json",), outcome.reasons

# ------------------------------------------------------------------- env

def test_env_acceptance_conflict_is_flagged_not_silenced():
    env = _Env()  # says ok
    action = _call(time="07:00", hour=25)
    outcome = check_action(action, _registry(), enabled=True, env=env)
    assert outcome.verdict == "rejected", "the guardrail may only ADD rejections"
    assert outcome.env_said_ok is True
    assert all(reason.startswith("conflict:") for reason in outcome.reasons), outcome.reasons
    assert "env accepted this call" in outcome.reasons[0]
    assert "hour=25" in outcome.reasons[0]
    assert outcome.action is action, "no silent rewrite over an env acceptance"
    assert outcome.original_action["arguments"]["hour"] == 25
    assert env.calls == [("alarm.set", {"time": "07:00", "hour": 25})]

def test_env_rejection_is_not_weakened():
    """An env rejection stays a rejection; it is repeated, never dropped."""
    env = _Env(errors=["hour must be 0..23"])
    outcome = check_action(_call(time="07:00", hour=5), _registry(), enabled=True, env=env)
    assert outcome.verdict == "rejected"
    assert outcome.env_said_ok is False
    assert outcome.reasons == ("env: hour must be 0..23",), outcome.reasons
    assert outcome.original_action["arguments"]["hour"] == 5

def test_env_is_ignored_when_disabled():
    env = _Env()
    action = _call(time="07:00")
    outcome = check_action(action, _registry(), env=env)
    assert outcome.verdict == "unchanged"
    assert env.calls == [], "a disabled guardrail must not call the env"

# ------------------------------------------------------------------ outcome

def test_outcome_shape_and_flags():
    outcome = check_action(_call(time="07:00"), _registry(), enabled=True)
    assert isinstance(outcome, GuardrailOutcome)
    assert outcome.accepted is True
    rejected = check_action(_call(time="07:00", hour=25), _registry(), enabled=True)
    assert rejected.accepted is False
    assert isinstance(rejected.reasons, tuple)
    assert isinstance(rejected.original_action, dict)
