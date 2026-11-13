"""Mock environment tests: log, hashing, validation, determinism."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate.data.discovery import require_split  # noqa: E402
from lifemate.data.loader import load_catalog  # noqa: E402
from lifemate.env.base import ToolEnvironment, ValidationOutcome  # noqa: E402
from lifemate.env.factory import create_environment  # noqa: E402
from lifemate.env.mock import MockToolEnv  # noqa: E402
from lifemate.env.official import MISSING_SDK_MESSAGE  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
CATALOG = load_catalog(FIXTURES, "train")
APPS = ["alarm", "smart_home", "messaging"]

STATE = {
    "apps": {
        "smart_home": {"light_living": {"id": "light_living", "on": False, "brightness": 40}},
        "alarm": {"06:30": {"id": "06:30", "enabled": True}},
    }
}


def _env(**kwargs) -> MockToolEnv:
    kwargs.setdefault("catalog", CATALOG)
    kwargs.setdefault("apps", APPS)
    kwargs.setdefault("initial_state", json.loads(json.dumps(STATE)))
    return MockToolEnv(**kwargs)


def test_mock_is_tool_environment_protocol():
    env = _env()
    assert isinstance(env, ToolEnvironment)
    assert env.name == "mock"


def test_available_tools_filtered_by_apps():
    env = _env()
    tools = set(env.available_tools())
    assert "alarm.set" in tools
    assert "sys.ask_missing_info" in tools
    assert "calendar.create_event" not in tools


def test_validate_does_not_touch_state_or_log():
    env = _env()
    before = env.get_state_hash()
    outcome = env.validate("alarm.set", {})
    assert isinstance(outcome, ValidationOutcome)
    assert outcome.ok is False
    assert any("missing required argument" in e for e in outcome.errors)
    assert env.get_state_hash() == before
    assert env.get_tool_calls() == []
    assert env.validate("alarm.set", {"time": "07:00"}).ok is True


def test_call_read_returns_initial_state():
    env = _env()
    result = env.call("smart_home.list_devices", {"room": "living_room"})
    assert result["ok"] is True
    assert result["result"]["light_living"]["on"] is False


def test_write_changes_state_hash_only_on_success():
    env = _env(fail_tools={"smart_home.device_set": 1})
    initial = env.get_state_hash()

    read = env.call("smart_home.list_devices", {})
    assert read["ok"] is True
    assert env.get_state_hash() == initial

    failed = env.call("smart_home.device_set", {"device_id": "light_living", "on": True})
    assert failed["ok"] is False
    assert env.get_state_hash() == initial  # simulated failure leaves state untouched

    write = env.call("smart_home.device_set", {"device_id": "light_living", "on": True})
    assert write["ok"] is True
    after_write = env.get_state_hash()
    assert after_write != initial

    invalid = env.call("smart_home.device_set", {"on": True})
    assert invalid["ok"] is False
    assert env.get_state_hash() == after_write


def test_failed_call_appears_in_log():
    env = _env()
    result = env.call("alarm.set", {})
    assert result["ok"] is False
    log = env.get_tool_calls()
    assert len(log) == 1
    entry = log[0]
    assert entry["name"] == "alarm.set"
    assert entry["arguments"] == {}
    assert entry["ok"] is False
    assert "error" in entry and "result" not in entry


def test_log_is_verbatim_and_ordered():
    env = _env()
    env.call("alarm.list", {})
    env.call("smart_home.device_set", {"device_id": "light_living", "on": True})
    log = env.get_tool_calls()
    assert [e["name"] for e in log] == ["alarm.list", "smart_home.device_set"]
    assert log[1]["arguments"] == {"device_id": "light_living", "on": True}
    assert log[1]["result"] == {"ok": True, "app": "smart_home", "key": "light_living", "arguments": {"device_id": "light_living", "on": True}}
    log[0]["name"] = "mutated"
    assert env.get_tool_calls()[0]["name"] == "alarm.list"


def test_unknown_tool():
    env = _env()
    result = env.call("nope.nope", {})
    assert result == {"name": "nope.nope", "arguments": {}, "ok": False, "error": "unknown tool: nope.nope"}


def test_invalid_arguments_error_string():
    env = _env()
    result = env.call("alarm.set", {"time": "06:30", "repeat": "hourly"})
    assert result["ok"] is False
    assert "not in enum" in result["error"]


def test_fail_tools_counter():
    env = _env(fail_tools={"alarm.list": 2})
    first = env.call("alarm.list", {})
    second = env.call("alarm.list", {})
    third = env.call("alarm.list", {})
    assert first["ok"] is False and "simulated failure" in first["error"]
    assert second["ok"] is False
    assert third["ok"] is True
    assert len(env.get_tool_calls()) == 3


def test_ask_missing_info_meta():
    env = _env()
    result = env.call("sys.ask_missing_info", {"missing_fields": ["time"]})
    assert result["ok"] is True
    assert "time" in result["result"]["question"]


def test_deterministic_across_instances():
    a = _env()
    b = _env()
    sequence = [
        ("alarm.list", {}),
        ("smart_home.device_set", {"device_id": "light_living", "on": True}),
        ("alarm.set", {}),
        ("sys.ask_missing_info", {"missing_fields": ["time"]}),
    ]
    for name, args in sequence:
        assert a.call(name, args) == b.call(name, args)
    assert a.get_state_hash() == b.get_state_hash()
    assert a.get_tool_calls() == b.get_tool_calls()


def test_state_hash_is_sha256_of_canonical_json():
    import hashlib

    env = _env()
    payload = {k: v for k, v in env.state.items() if k != "writes"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert env.get_state_hash() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_mock_never_reads_fixture_env_dirs():
    # Mock state comes only from the constructor argument.
    env = MockToolEnv(catalog=CATALOG, initial_state={"apps": {"alarm": {"a1": {"id": "a1"}}}}, apps=["alarm"])
    assert env.state["apps"]["alarm"] == {"a1": {"id": "a1"}}
    assert env.call("alarm.list", {})["result"] == {"a1": {"id": "a1"}}


def test_factory_requires_explicit_mock():
    mock = create_environment(
        "train", "train_single_tool", FIXTURES, use_mock=True, catalog=CATALOG, mock_state={"apps": {}}
    )
    assert isinstance(mock, MockToolEnv)

    try:
        create_environment("train", "train_single_tool", FIXTURES)
    except LifeMateError as exc:
        assert "zaic_tools not installed" in str(exc)
        assert str(exc) == MISSING_SDK_MESSAGE
    else:  # pragma: no cover - only when the SDK happens to be installed
        raise AssertionError("expected LifeMateError without zaic_tools")


def test_official_env_missing_sdk_message():
    from lifemate.env.official import OfficialToolEnv

    try:
        OfficialToolEnv("train", "x", FIXTURES)
    except LifeMateError as exc:
        assert "zaic_tools not installed" in str(exc)
        assert "use_mock=True" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected LifeMateError")


def test_mock_without_catalog_is_permissive():
    env = MockToolEnv()
    assert env.available_tools() == []
    result = env.call("anything.goes", {"x": 1})
    assert result["ok"] is True
    assert env.get_state_hash() != ""


def test_split_has_no_environment_dir():
    paths = require_split(FIXTURES, "train")
    assert paths.has_environment_dir is False
