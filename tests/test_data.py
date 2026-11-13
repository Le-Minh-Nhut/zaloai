"""Data layer tests: discovery, loading, schemas, catalog, validation, CLI."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.data.discovery import (  # noqa: E402
    available_splits,
    discover_splits,
    require_split,
)
from lifemate.data.loader import (  # noqa: E402
    iter_raw_records,
    load_catalog,
    load_raw_split,
    load_samples,
    read_sample_audio,
)
from lifemate.data.validator import validate_split  # noqa: E402
from lifemate.errors import DatasetNotFoundError, ProhibitedAccessError  # noqa: E402
from lifemate.env.mock import MockToolEnv  # noqa: E402

TRAIN = FIXTURES / "train"


def _cli():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import inspect_data

    return inspect_data


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discovery_finds_train_and_public_test():
    splits = discover_splits(FIXTURES)
    assert "train" in splits
    assert "public_test" in splits
    assert splits["train"].jsonl == TRAIN / "train.jsonl"
    assert splits["train"].catalog == TRAIN / "train_tools_catalog.json"
    assert splits["train"].audio_dir == TRAIN / "audio"
    assert splits["train"].sample_submission == TRAIN / "sample_submission.jsonl"
    assert splits["train"].has_environment_dir is False


def test_discovery_does_not_invent_private_test():
    names = available_splits(FIXTURES)
    assert "private_test" not in names
    assert names == sorted(names)


def test_discovery_skips_environment_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train").mkdir()
        (root / "train" / "train.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
        env = root / "train_environment"  # compliance-scan: allow
        env.mkdir()
        (env / "secret.json").write_text("{}", encoding="utf-8")
        splits = discover_splits(root)
        assert "train" in splits
        assert all(not name.endswith("_environment") for name in splits)
        assert splits["train"].has_environment_dir is True


def test_discovery_finds_nested_environment_dir():
    """Official layout nests ``<name>_environment`` inside the split directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train").mkdir()
        (root / "train" / "train.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
        env = root / "train" / "train_environment"  # compliance-scan: allow
        env.mkdir()
        splits = discover_splits(root)
        assert splits["train"].has_environment_dir is True
        assert splits["train"].root == root / "train"

def test_discovery_handles_flat_jsonl_and_catalog_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "flat.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
        (root / "cat_only").mkdir()
        (root / "cat_only" / "cat_only_tools_catalog.json").write_text('{"apps": []}', encoding="utf-8")
        splits = discover_splits(root)
        assert splits["flat"].jsonl == root / "flat.jsonl"
        assert "cat_only" in splits
        assert splits["cat_only"].jsonl is None


def test_require_split_missing_raises_expected_error():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            require_split(tmp, "train")
        except DatasetNotFoundError as exc:
            assert exc.split == "train"
            assert exc.expected_path == str(Path(tmp) / "train")
        else:  # pragma: no cover
            raise AssertionError("DatasetNotFoundError not raised")


def test_dataset_not_found_message_is_exact():
    exc = DatasetNotFoundError("train", "/data/train")
    expected = (
        "Dataset not found.\n"
        "\n"
        "Expected directory:\n"
        "/data/train\n"
        "\n"
        "Please extract the official LifeMate dataset\n"
        "into the configured data directory.\n"
        "\n"
        "Implementation is available.\n"
        "Real-data validation has not yet been performed."
    )
    assert str(exc) == expected


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_loader_parses_all_synthetic_train_samples():
    samples = load_samples(FIXTURES, "train")
    assert len(samples) == 14
    ids = [sample.id for sample in samples]
    assert len(set(ids)) == len(ids)
    assert all(sample.audio_text for sample in samples)
    assert all(sample.history for sample in samples)
    assert all(sample.installed_apps for sample in samples)
    assert sum(1 for s in samples if s.label and s.label.final_state) == 2


def test_loader_limit_and_no_labels():
    samples = load_samples(FIXTURES, "train", limit=3, with_labels=False)
    assert len(samples) == 3
    assert all(sample.label is None for sample in samples)


def test_loader_missing_split_raises_not_found():
    try:
        load_samples(FIXTURES, "private_test")
    except DatasetNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DatasetNotFoundError")


def test_iter_raw_records_malformed_modes():
    path = FIXTURES / "malformed" / "malformed.jsonl"
    strict = list(iter_raw_records(path))
    # 5 source lines: 1 valid, 2 broken JSON, 1 blank (skipped), 1 non-object,
    # 1 object without an id (valid JSON, rejected later by the schema layer).
    assert len(strict) == 4
    errors = [row for row in strict if "__error__" in row]
    assert len(errors) == 2
    assert [row["__line__"] for row in errors] == [2, 4]
    assert "record is list, not object" in errors[1]["__error__"]

    clean = list(iter_raw_records(path, skip_malformed=True))
    assert len(clean) == 2
    assert clean[0]["id"] == "malformed_ok"
    assert "id" not in clean[1]


def test_load_raw_split_keeps_error_markers():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train").mkdir()
        (root / "train" / "train.jsonl").write_text('{"id": 1}\nnot json\n', encoding="utf-8")
        rows = load_raw_split(root, "train")
        assert rows[0]["id"] == 1
        assert "__error__" in rows[1]


def test_all_train_audio_files_exist():
    samples = load_samples(FIXTURES, "train")
    for sample in samples:
        found = read_sample_audio(sample)
        assert found is not None, sample.id
        assert found.is_file()
        assert found.suffix == ".wav"


def test_public_test_missing_audio_reports_none():
    samples = {s.id: s for s in load_samples(FIXTURES, "public_test")}
    assert read_sample_audio(samples["test_single_tool"]) is not None
    assert read_sample_audio(samples["test_missing_audio"]) is None


def test_read_sample_audio_rejects_prohibited_path():
    from lifemate.data.schemas import Context, Sample

    sample = Sample(
        id="x",
        split="train",
        audio_path=Path("/data/train_environment/a/x.wav"),  # compliance-scan: allow
        audio_text="",
        context=Context(),
    )
    try:
        read_sample_audio(sample)
    except ProhibitedAccessError as exc:
        assert "train_environment" in exc.path
    else:  # pragma: no cover
        raise AssertionError("ProhibitedAccessError not raised")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def test_sample_to_model_input_has_no_label_leakage():
    samples = load_samples(FIXTURES, "train")
    payload = samples[0].to_model_input()
    serialized = json.dumps(payload)
    for forbidden in ("label", "final_state", "tool_calls", "tool_results", "gold", "target"):
        assert forbidden not in payload, forbidden
        assert forbidden not in serialized, forbidden


def test_tolerant_parsers_accept_alternate_shapes():
    from lifemate.data.schemas import (
        Label,
        Sample,
        ToolCall,
        ToolResult,
        parse_history,
    )

    call = ToolCall.from_raw({"function": {"name": "alarm.set", "arguments": '{"time": "07:00"}'}})
    assert call.name == "alarm.set"
    assert call.arguments == {"time": "07:00"}
    assert call.to_dict() == {"name": "alarm.set", "arguments": {"time": "07:00"}}

    result = ToolResult.from_raw({"success": False, "output": None, "error": "boom"})
    assert result.ok is False and result.error == "boom"

    turns = parse_history(
        [
            {"speaker": "user", "utterance": "hi"},
            {"role": "assistant", "tool_call": {"name": "alarm.list", "arguments": {}}},
            "plain string turn",
        ]
    )
    assert [t.role for t in turns][0] == "user"
    assert turns[0].content == "hi"
    assert turns[1].tool_call["name"] == "alarm.list"
    assert turns[2].content == "plain string turn"

    label = Label.from_raw({"tool_calls": [{"name": "alarm.list", "arguments": {}}]})
    assert label.final_state is None
    assert label.is_model_input is False
    assert len(label.tool_calls) == 1

    sample = Sample.from_raw(
        {
            "id": "s1",
            "audio": "s1.wav",
            "transcript": "hello",
            "installed_apps": ["alarm"],
            "messages": [{"role": "user", "content": "hi"}],
        },
        "train",
        TRAIN,
    )
    assert sample.installed_apps == ["alarm"]
    assert sample.audio_text == "hello"
    assert sample.label is None


def test_unknown_fields_tolerated():
    samples = {s.id: s for s in load_samples(FIXTURES, "train")}
    sample = samples["train_unknown_fields"]
    assert sample.raw["extra_top_level"] == {"note": "tolerated unknown field"}
    assert sample.context.user_profile.get("name") == "Test User"
    assert "wifi_ssid" in sample.raw["context"]


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def test_catalog_parses_and_infers_kinds():
    catalog = load_catalog(FIXTURES, "train")
    assert catalog is not None
    assert "sys" in catalog.apps
    assert set(catalog.apps) == {"alarm", "calendar", "messaging", "smart_home", "sys"}
    assert len(catalog) == 13
    assert catalog.get("alarm.set").kind == "write"
    assert catalog.get("alarm.list").kind == "read"
    assert catalog.get("sys.ask_missing_info").kind == "meta"
    assert catalog.get("smart_home.list_devices").app_name == "smart_home"


def test_tools_for_apps_filters_uninstalled():
    catalog = load_catalog(FIXTURES, "train")
    names = {spec.name for spec in catalog.tools_for_apps(["alarm"])}
    assert "alarm.set" in names
    assert "sys.ask_missing_info" in names  # meta always included
    assert "messaging.send" not in names
    no_meta = {spec.name for spec in catalog.tools_for_apps(["alarm"], include_meta=False)}
    assert "sys.ask_missing_info" not in no_meta


def test_catalog_prompt_and_dependency_edges():
    catalog = load_catalog(FIXTURES, "train")
    prompt = catalog.to_schema_prompt(["smart_home"])
    assert "smart_home.device_set" in prompt
    assert "sys.ask_missing_info" in prompt
    edges = catalog.dependency_edges()
    assert ("smart_home.device_set", "#/apps/smart_home/tools/0", "device_id") in edges


def test_validate_call_catches_problems():
    catalog = load_catalog(FIXTURES, "train")

    unknown = catalog.validate_call("nope.nope", {})
    assert unknown["errors"] and "unknown tool" in unknown["errors"][0]

    missing = catalog.validate_call("alarm.set", {})
    assert any("missing required argument 'time'" in e for e in missing["errors"])

    mistyped = catalog.validate_call("alarm.set", {"time": 630})
    assert any("expected type string" in e for e in mistyped["errors"])

    enum_bad = catalog.validate_call("alarm.set", {"time": "06:30", "repeat": "hourly"})
    assert any("not in enum" in e for e in enum_bad["errors"])

    # official rule (`zaic_tools.validate.schema_error`): reject only when NO
    # x-one-of group is fully present; a flat list is a single group.
    one_of_bad = catalog.validate_call("messaging.send", {"body": "hi"})
    assert any("cần một trong" in e for e in one_of_bad["errors"]), one_of_bad

    one_of_ok = catalog.validate_call("messaging.send", {"body": "hi", "contact_id": "c1"})
    assert one_of_ok["errors"] == []

    # both groups present is accepted too (the official env allows it)
    one_of_both = catalog.validate_call("messaging.send", {"body": "hi", "contact_id": "c1", "group_id": "g1"})
    assert one_of_both["errors"] == [], one_of_both

    # the official EMPTY rule: "" / [] / {} count as absent
    one_of_empty = catalog.validate_call("messaging.send", {"body": "hi", "contact_id": "", "group_id": "g1"})
    assert one_of_empty["errors"] == [], one_of_empty
    one_of_all_empty = catalog.validate_call("messaging.send", {"body": "hi", "contact_id": "", "group_id": ""})
    assert any("cần một trong" in e for e in one_of_all_empty["errors"]), one_of_all_empty

    bool_as_int = catalog.validate_call(
        "smart_home.device_set", {"device_id": "light", "on": True, "brightness": True}
    )
    assert any("expected type integer" in e for e in bool_as_int["errors"])

    warnings_only = catalog.validate_call("alarm.set", {"time": "06:30", "extra": 1})
    assert warnings_only["errors"] == []
    assert any("unknown argument 'extra'" in w for w in warnings_only["warnings"])

    good = catalog.validate_call("smart_home.device_set", {"device_id": "light_living", "on": False})
    assert good == {"errors": [], "warnings": []}


def test_catalog_unresolvable_ref_is_warning():
    catalog = ToolCatalog.from_dict(
        {
            "tools": [
                {
                    "name": "x.set",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string", "x-ref": "#/nope/missing"}},
                        "required": ["id"],
                    },
                }
            ]
        }
    )
    report = catalog.validate_call("x.set", {"id": "abc"})
    assert report["errors"] == []
    assert any("unresolvable x-ref" in w for w in report["warnings"])


def test_catalog_accepts_alternate_shapes():
    flat = {"tools": {"alarm.list": {"description": "list"}}}
    catalog = ToolCatalog.from_dict(flat)
    assert "alarm.list" in catalog.tools
    assert catalog.tools["alarm.list"].kind == "read"

    openai = {"tools": [{"type": "function", "function": {"name": "calendar.get_event", "description": "one", "parameters": {"properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}}}]}
    catalog2 = ToolCatalog.from_dict(openai)
    assert catalog2.get("calendar.get_event").required == ["event_id"]

def test_tool_result_payload_survives_and_matches_env_shape():
    """Real records nest no payload under ``result``: every other key is the payload.

    Shape taken verbatim from ``data/train/train.jsonl`` (``train_0001``,
    ``train_0025``): the official env returns ``{"ok": True, **payload}``.
    """
    from lifemate.data.schemas import ToolResult, parse_tool_results
    from lifemate.data.trajectory_builder import tool_result_to_observation

    forecast = [{"location": "Đà Nẵng", "date": "2026-11-29", "rain": True}]
    parsed = ToolResult.from_raw({"ok": True, "forecast": forecast})
    assert parsed.ok is True
    assert parsed.payload == {"forecast": forecast}, parsed.payload
    assert parsed.result == {"forecast": forecast}
    assert tool_result_to_observation(parsed) == {"ok": True, "forecast": forecast}

    # ask_missing_info: payload inline, nothing lost
    ask = parse_tool_results([{"ok": True, "waiting_for_user": True}])[0]
    assert tool_result_to_observation(ask) == {"ok": True, "waiting_for_user": True}

    # a genuinely empty success stays empty (no phantom "result": null)
    assert tool_result_to_observation(ToolResult.from_raw({"ok": True})) == {"ok": True}

    # transport-style {"ok", "result"} records still work
    assert tool_result_to_observation(ToolResult.from_raw({"ok": True, "result": {"a": 1}})) == {
        "ok": True,
        "result": {"a": 1},
    }
    # failures keep the error only
    assert tool_result_to_observation(ToolResult.from_raw({"ok": False, "error": "boom"})) == {
        "ok": False,
        "error": "boom",
    }

def test_history_user_turn_reads_audio_text():
    """Real history user turns carry ``audio_text`` (the assistant carries ``text``)."""
    from lifemate.data.schemas import parse_history

    turns = parse_history(
        [
            {
                "role": "user",
                "audio": "audio/train_0008/train_0008_u1.wav",
                "audio_text": "Mai tôi có những báo thức nào vậy?",
            },
            {"role": "assistant", "text": "Bạn có 2 báo thức.", "tool_calls": [], "tool_results": []},
        ]
    )
    assert turns[0].content == "Mai tôi có những báo thức nào vậy?", repr(turns[0].content)
    assert turns[0].raw["audio_text"] == "Mai tôi có những báo thức nào vậy?"
    assert turns[1].content == "Bạn có 2 báo thức."

def test_one_of_nested_groups_match_official_rule():
    """Nested x-one-of (real catalog shape) accepts any single complete group."""
    catalog = ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "id": "music",
                    "tools": [
                        {
                            "name": "music.play",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "track_id": {"type": "string"},
                                    "playlist_id": {"type": "string"},
                                },
                                "required": [],
                                "x-one-of": [["track_id"], ["playlist_id"]],
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert catalog.validate_call("music.play", {"track_id": "trk_861"})["errors"] == []
    assert catalog.validate_call("music.play", {"playlist_id": "pl_2047"})["errors"] == []
    # both groups present: the official env accepts this (train_0925 shape)
    assert catalog.validate_call("music.play", {"track_id": "trk_861", "playlist_id": "pl_2047"})["errors"] == []
    # no group complete -> rejected
    assert catalog.validate_call("music.play", {})["errors"], "empty arguments must be rejected"
    assert catalog.validate_call("music.play", {"track_id": ""})["errors"], 'a "" value counts as absent'

def test_required_argument_empty_values_count_as_missing():
    """Official EMPTY rule: ``""``/``[]``/``{}`` are missing, not "provided"."""
    catalog = load_catalog(FIXTURES, "train")
    for empty in ("", [], {}):
        report = catalog.validate_call("alarm.set", {"time": empty})
        assert any("missing required argument 'time'" in e for e in report["errors"]), (empty, report)

def test_catalog_enforces_bounds_formats_and_array_items():
    """The catalog validator applies every value rule the official scorer applies.

    Uses the shipped catalog's own declarations: `fitness.set_goal.steps` and
    `home_energy.set_alert.threshold_kwh` carry `exclusiveMinimum: 0`,
    `package_tracking.schedule_pickup.date` carries `format: date`,
    `tasks.add.due` `format: date-time`, `email.send.to` `items.type: string`.
    """
    catalog = ToolCatalog.from_dict(
        {
            "apps": [
                {"id": "fitness", "tools": [{"name": "fitness.set_goal", "parameters": {
                    "type": "object",
                    "properties": {"steps": {"type": "integer", "exclusiveMinimum": 0}},
                    "required": ["steps"]}}]},
                {"id": "home_energy", "tools": [{"name": "home_energy.set_alert", "parameters": {
                    "type": "object",
                    "properties": {"threshold_kwh": {"type": "integer", "exclusiveMinimum": 0}},
                    "required": ["threshold_kwh"]}}]},
                {"id": "package_tracking", "tools": [{"name": "package_tracking.schedule_pickup", "parameters": {
                    "type": "object",
                    "properties": {"address": {"type": "string"}, "date": {"type": "string", "format": "date"}},
                    "required": ["address", "date"]}}]},
                {"id": "tasks", "tools": [{"name": "tasks.add", "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "due": {"type": "string", "format": "date-time"}},
                    "required": ["title"]}}]},
                {"id": "email", "tools": [{"name": "email.send", "parameters": {
                    "type": "object",
                    "properties": {"to": {"type": "array", "items": {"type": "string"}},
                                   "subject": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "subject", "body"]}}]},
                {"id": "appliances", "tools": [{"name": "appliances.start_cycle", "parameters": {
                    "type": "object",
                    "properties": {"delay_minutes": {"type": "integer", "minimum": 0}},
                    "required": ["delay_minutes"]}}]},
            ]
        }
    )

    # exclusiveMinimum is strict: 0 is rejected, 1 is accepted
    assert catalog.validate_call("fitness.set_goal", {"steps": 0})["errors"]
    assert catalog.validate_call("fitness.set_goal", {"steps": 1})["errors"] == []
    assert catalog.validate_call("home_energy.set_alert", {"threshold_kwh": 0})["errors"]

    # minimum is inclusive: 0 passes
    assert catalog.validate_call("appliances.start_cycle", {"delay_minutes": 0})["errors"] == []
    assert catalog.validate_call("appliances.start_cycle", {"delay_minutes": -1})["errors"]

    # `format: date` needs a real calendar date, not just the right shape
    base = {"address": "25 Nguyễn Du"}
    assert catalog.validate_call("package_tracking.schedule_pickup", {**base, "date": "2026-02-30"})["errors"], "Feb 30 is not a date"
    assert catalog.validate_call("package_tracking.schedule_pickup", {**base, "date": "2026-11-29"})["errors"] == []
    assert catalog.validate_call("package_tracking.schedule_pickup", {**base, "date": "29/11/2026"})["errors"]

    # `format: date-time` accepts the offset form the docs use, and range-checks hh:mm
    assert catalog.validate_call("tasks.add", {"title": "x", "due": "2026-11-29T05:30:00+07:00"})["errors"] == []
    assert catalog.validate_call("tasks.add", {"title": "x", "due": "2026-11-29T25:61:00+07:00"})["errors"]
    assert catalog.validate_call("tasks.add", {"title": "x", "due": "not-a-date"})["errors"]

    # items.type: a non-string element in a string array is rejected
    good = {"to": ["lan.nguyen@example.test"], "subject": "s", "body": "b"}
    assert catalog.validate_call("email.send", good)["errors"] == []
    assert catalog.validate_call("email.send", {**good, "to": [123]})["errors"]

    # an empty optional value is "not given", so it is never validated
    optional = {"to": ["a@b.c"], "subject": "s", "body": "b", "cc": []}
    assert catalog.validate_call("email.send", optional)["errors"] == []
    # ...but an empty required value still counts as missing
    assert catalog.validate_call("fitness.set_goal", {"steps": ""})["errors"]
    # unknown arguments stay WARNINGS, exactly as the official validator ignores them
    warned = catalog.validate_call("fitness.set_goal", {"steps": 5, "extra": 1})
    assert warned["errors"] == [] and any("unknown argument" in w for w in warned["warnings"])

    # non-finite numbers are rejected rather than slipping through a bound compare
    assert catalog.validate_call("fitness.set_goal", {"steps": float("inf")})["errors"]

def test_declared_kind_beats_verb_inference():
    """The official catalog declares ``kind``; inference disagreed on 11/147 train tools.

    Real examples: ``calculator.calculate``/``tv.guide`` are declared ``read``
    but end in a verb that is not in ``READ_VERBS``; ``flights.check_in`` is
    declared ``write`` but named ``check``.
    """
    catalog = ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "id": "tv",
                    "tools": [
                        {"name": "tv.guide", "kind": "read", "parameters": {"type": "object", "properties": {}, "required": []}},
                        {"name": "tv.record", "kind": "write", "parameters": {"type": "object", "properties": {}, "required": []}},
                    ],
                }
            ]
        }
    )
    assert catalog.get("tv.guide").kind == "read"
    assert catalog.get("tv.record").kind == "write"

    # a catalog without a declared kind still falls back to inference
    inferred = ToolCatalog.from_dict({"tools": [{"name": "alarm.list", "description": "list"}]})
    assert inferred.get("alarm.list").kind == "read"



# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def test_validate_split_report_on_synthetic():
    paths = require_split(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    report = validate_split(paths, catalog=catalog)
    assert report["counts"]["samples"] == 14
    assert report["counts"]["catalog_tools"] == 13
    assert report["duplicate_ids"] == []
    assert report["missing_audio"] == []
    assert report["errors"] == []
    assert "extra_top_level" in report["unknown_top_level_fields"]
    assert report["field_presence"]["timezone"] == "13/14"
    assert "label" not in report["unknown_top_level_fields"]


def test_validate_split_flags_bad_gold_calls():
    paths = require_split(FIXTURES, "train")
    catalog = load_catalog(FIXTURES, "train")
    samples = load_samples(FIXTURES, "train")
    for sample in samples:
        if sample.id == "train_single_tool":
            sample.label.tool_calls[0].name = "ghost.tool"
        if sample.id == "train_multi_step":
            sample.label.tool_calls[0].arguments = {"title": "x"}
    report = validate_split(paths, samples=samples, catalog=catalog)
    assert any("ghost.tool" in e for e in report["errors"])
    assert any("missing required argument 'start'" in e for e in report["errors"])


# --------------------------------------------------------------------------- #
# Scripts
# --------------------------------------------------------------------------- #
def test_inspect_data_main_synthetic():
    cli = _cli()
    out = Path(tempfile.mkdtemp(prefix="inspect_ok_"))
    code = cli.main(["--data-root", str(FIXTURES), "--output-dir", str(out)])
    assert code == 0
    payload = json.loads((out / "data_inspection.json").read_text(encoding="utf-8"))
    assert payload["inspected_splits"] == ["malformed", "public_test", "train"]
    markdown = (out / "data_inspection.md").read_text(encoding="utf-8")
    assert "Privacy" in markdown
    assert "+84900000000" not in markdown
    assert "Cam on ca nha" not in markdown
    assert "Test User" not in markdown


def test_inspect_data_main_empty_root_exits_2_without_traceback():
    cli = _cli()
    empty = Path(tempfile.mkdtemp(prefix="inspect_empty_"))
    out = Path(tempfile.mkdtemp(prefix="inspect_empty_out_"))
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli.main(["--data-root", str(empty), "--output-dir", str(out)])
    assert code == 2
    printed = buffer.getvalue()
    assert "Dataset not found." in printed
    assert "Real-data validation has not yet been performed." in printed
    assert "Traceback" not in printed
    assert str(empty / "train") in printed


def test_inspect_data_subprocess_empty_root():
    import subprocess

    empty = Path(tempfile.mkdtemp(prefix="inspect_sub_"))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_data.py"), "--data-root", str(empty)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    assert proc.returncode == 2
    assert "Dataset not found." in proc.stdout
    assert proc.stderr.strip() == ""


# --------------------------------------------------------------------------- #
# Config (owned by this module so the stdlib runner sees it with the data suite)
# --------------------------------------------------------------------------- #
def test_load_config_defaults_without_files():
    from lifemate.config import load_config

    cfg = load_config()
    assert cfg.training.tuning_mode == "full"
    assert cfg.data.root == "./data"
    assert cfg.model.name == "Qwen/Qwen3.5-4B"
    assert cfg.asr.backend == "gipformer"  # gipformer peer cut over default ASR backend
    assert cfg.inference.use_mock_env is False
    assert cfg.wandb.enabled is False
    assert cfg.training.report_to == []
    assert isinstance(cfg.to_dict(), dict)


def test_load_config_overrides_coerce_types():
    from lifemate.config import load_config

    cfg = load_config(
        overrides={
            "training.tuning_mode": "lora",
            "model.name": "local/model",
            "training.num_train_epochs": "3",
            "training.gradient_checkpointing": "false",
            "training.report_to": "wandb,tensorboard",
            "inference.limit": 5,
            "model.tokenizer_name": "tok",
        }
    )
    assert cfg.training.tuning_mode == "lora"
    assert cfg.model.name == "local/model"
    assert cfg.training.num_train_epochs == 3 and isinstance(cfg.training.num_train_epochs, int)
    assert cfg.training.gradient_checkpointing is False
    assert cfg.training.report_to == ["wandb", "tensorboard"]
    assert cfg.inference.limit == 5
    assert cfg.model.tokenizer_name == "tok"


def test_load_config_rejects_bad_tuning_mode_and_unknown_key():
    from lifemate.config import load_config
    from lifemate.errors import ConfigError

    for overrides in ({"training.tuning_mode": "qlora"}, {"training.nope": 1}, {"nope.key": 1}):
        try:
            load_config(overrides=overrides)
        except ConfigError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"ConfigError expected for {overrides}")


def test_dump_and_reload_roundtrip():
    from lifemate.config import dump_config, load_config

    cfg = load_config(overrides={"training.tuning_mode": "lora", "training.learning_rate": 0.0002})
    out = Path(tempfile.mkdtemp(prefix="cfg_")) / "snapshot.yaml"
    dump_config(cfg, out)
    assert out.is_file()
    reloaded = load_config(out)
    assert reloaded.to_dict() == cfg.to_dict()


def test_load_config_reads_plain_yaml_subset():
    from lifemate.config import load_config

    out = Path(tempfile.mkdtemp(prefix="cfg_yaml_")) / "mini.yaml"
    out.write_text(
        "training:\n"
        "  tuning_mode: lora\n"
        "  report_to: [wandb]\n"
        "asr:\n"
        "  backend: whisper\n"
        "  sample_rate: 8000\n",
        encoding="utf-8",
    )
    cfg = load_config(out)
    assert cfg.training.tuning_mode == "lora"
    assert cfg.training.report_to == ["wandb"]
    assert cfg.asr.backend == "whisper"
    assert cfg.asr.sample_rate == 8000


def test_broken_wav_is_zero_length_and_undecodable():
    import wave

    broken = FIXTURES / "broken" / "zero_length.wav"
    assert broken.is_file()
    assert broken.stat().st_size == 0
    try:
        with wave.open(str(broken), "rb"):
            pass
    except (wave.Error, EOFError):
        pass
    else:  # pragma: no cover
        raise AssertionError("zero-length wav should not decode")


def test_train_split_is_clean_and_has_no_environment_dir():
    paths = require_split(FIXTURES, "train")
    rows = load_raw_split(FIXTURES, "train")
    assert all("__error__" not in row for row in rows)
    assert len(rows) == 14
    children = {child.name for child in FIXTURES.iterdir()}
    assert not any(name.endswith("_environment") for name in children)
    assert "private_test" not in children
    assert paths.jsonl is not None and paths.jsonl.is_file()


def test_synthetic_fixtures_are_marked():
    manifest = json.loads((FIXTURES / "synthetic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["_synthetic"] is True
    assert manifest["_warning"] == "SYNTHETIC TEST DATA / NOT OFFICIAL COMPETITION DATA"
    assert set(manifest["splits"]) == {"train", "public_test"}
    assert len(manifest["categories"]) == 17
    note = (FIXTURES / "SYNTHETIC.md").read_text(encoding="utf-8")
    assert "SYNTHETIC TEST DATA / NOT OFFICIAL COMPETITION DATA" in note
    for row in load_raw_split(FIXTURES, "train"):
        assert row["_synthetic"] is True
