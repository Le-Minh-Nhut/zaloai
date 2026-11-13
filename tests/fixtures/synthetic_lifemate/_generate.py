"""Build the synthetic fixture dataset used by the test suite.

Not official competition data. Everything here is generated deterministically
with the stdlib so the fixtures are reproducible and inspectable.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNTHETIC_MARKER = "_synthetic\": true"
CATALOG_MARKER = "_synthetic"


def write_wav(path: Path, seconds: float, freq: float | None, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        data = bytearray()
        for idx in range(frames):
            value = 0.0 if freq is None else 0.3 * math.sin(2 * math.pi * freq * idx / sample_rate)
            data += struct.pack("<h", int(value * 32767))
        handle.writeframes(bytes(data))


def tone_name(idx: int) -> float:
    return 220.0 + 40.0 * idx


def catalog() -> dict:
    def obj(props: dict, required: list[str], **extra) -> dict:
        return {"type": "object", "properties": props, "required": required, **extra}

    return {
        "_synthetic": True,
        "_note": "SYNTHETIC TEST DATA / NOT OFFICIAL COMPETITION DATA",
        "apps": [
            {
                "app_name": "alarm",
                "description": "Alarms and timers.",
                "tools": [
                    {
                        "name": "alarm.set",
                        "description": "Create an alarm at a clock time.",
                        "annotations": {"readOnlyHint": False},
                        "parameters": obj(
                            {
                                "time": {"type": "string", "description": "HH:MM"},
                                "label": {"type": "string"},
                                "repeat": {"type": "string", "enum": ["once", "daily", "weekdays"]},
                                "days": {"type": "array", "items": {"type": "string"}},
                            },
                            ["time"],
                        ),
                    },
                    {
                        "name": "alarm.list",
                        "description": "List existing alarms.",
                        "parameters": obj({}, []),
                    },
                    {
                        "name": "alarm.cancel",
                        "description": "Cancel an alarm by id.",
                        "parameters": obj({"alarm_id": {"type": "string"}}, ["alarm_id"]),
                    },
                ],
            },
            {
                "app_name": "calendar",
                "description": "Calendar events.",
                "tools": [
                    {
                        "name": "calendar.search_events",
                        "description": "Search events by title.",
                        "parameters": obj(
                            {"query": {"type": "string"}, "date": {"type": "string"}}, ["query"]
                        ),
                    },
                    {
                        "name": "calendar.create_event",
                        "description": "Create a calendar event.",
                        "parameters": obj(
                            {
                                "title": {"type": "string"},
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                                "attendees": {"type": "array", "items": {"type": "string"}},
                            },
                            ["title", "start"],
                        ),
                    },
                    {
                        "name": "calendar.get_event",
                        "description": "Get one event by id.",
                        "parameters": obj({"event_id": {"type": "string"}}, ["event_id"]),
                    },
                ],
            },
            {
                "app_name": "smart_home",
                "description": "Home devices.",
                "tools": [
                    {
                        "name": "smart_home.list_devices",
                        "description": "List devices in the home.",
                        "parameters": obj(
                            {"room": {"type": "string"}},
                            [],
                        ),
                    },
                    {
                        "name": "smart_home.device_set",
                        "description": "Set a device state; device_id comes from device_list.",
                        "parameters": obj(
                            {
                                "device_id": {
                                    "type": "string",
                                    "x-ref": "#/apps/smart_home/tools/0",
                                    "x-match": "id",
                                },
                                "on": {"type": "boolean"},
                                "brightness": {"type": "integer"},
                            },
                            ["device_id", "on"],
                        ),
                    },
                    {
                        "name": "smart_home.device_status",
                        "description": "Read a device status.",
                        "parameters": obj({"device_id": {"type": "string"}}, ["device_id"]),
                    },
                ],
            },
            {
                "app_name": "messaging",
                "description": "Chat messages.",
                "tools": [
                    {
                        "name": "messaging.search_contacts",
                        "description": "Search the contact list.",
                        "parameters": obj({"query": {"type": "string"}}, ["query"]),
                    },
                    {
                        "name": "messaging.send",
                        "description": "Send a message to a contact or a group.",
                        "parameters": obj(
                            {
                                "recipient": {"type": "string", "x-one-of": ["contact_id", "group_id"]},
                                "contact_id": {"type": "string"},
                                "group_id": {"type": "string"},
                                "body": {"type": "string"},
                                "urgent": {"type": "boolean"},
                            },
                            ["body"],
                            **{"x-one-of": ["contact_id", "group_id"]},
                        ),
                    },
                    {
                        "name": "messaging.read_thread",
                        "description": "Read a message thread.",
                        "parameters": obj({"thread_id": {"type": "string"}}, ["thread_id"]),
                    },
                ],
            },
            {
                "app_name": "sys",
                "description": "Meta tools.",
                "tools": [
                    {
                        "name": "sys.ask_missing_info",
                        "description": "Ask the user for missing information.",
                        "annotations": {"readOnlyHint": True},
                        "parameters": obj(
                            {"missing_fields": {"type": "array", "items": {"type": "string"}}}, []
                        ),
                    }
                ],
            },
        ],
    }


def context(apps: list[str], full: bool = True) -> dict:
    ctx: dict = {"installed_apps": apps}
    if full:
        ctx.update(
            {
                "current_time": "2026-11-07T09:30:00+07:00",
                "timezone": "Asia/Ho_Chi_Minh",
                "current_location": "Ho Chi Minh City",
                "user_profile": {"name": "Test User", "phone": "+84900000000"},
            }
        )
    return ctx


def rec(
    sid: str,
    category: str,
    ctx: dict,
    history: list[dict],
    calls: list[dict],
    results: list[dict],
    final_state: dict | None = None,
) -> dict:
    label: dict = {"tool_calls": calls, "tool_results": results}
    if final_state is not None:
        label["final_state"] = final_state
    return {
        "_synthetic": True,
        "id": sid,
        "category": category,
        "audio_path": f"{sid}/{sid}.wav",
        "audio_text": "",
        "context": ctx,
        "history": history,
        "label": label,
    }


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def asst(text: str, calls: list[dict] | None = None) -> dict:
    turn: dict = {"role": "assistant", "content": text}
    if calls:
        turn["tool_calls"] = calls
    return turn


def tool(name: str, result: dict) -> dict:
    return {"role": "tool", "name": name, "result": result}


def train_records() -> list[dict]:
    four = ["alarm", "calendar", "smart_home", "messaging"]
    rows: list[dict] = []

    rows.append(
        rec(
            "train_single_tool",
            "single_tool",
            context(four),
            [user("Dat bao thuc luc 6 gio 30 sang mai.")],
            [{"name": "alarm.set", "arguments": {"time": "06:30", "label": "morning", "repeat": "once"}}],
            [{"name": "alarm.set", "ok": True, "result": {"alarm_id": "alarm_1", "time": "06:30"}}],
            {"apps": {"alarm": {"alarm_1": {"alarm_id": "alarm_1", "time": "06:30", "enabled": True}}}},
        )
    )
    rows[-1]["audio_text"] = "Dat bao thuc luc 6 gio 30 sang mai."

    rows.append(
        rec(
            "train_multi_step",
            "multi_step",
            context(["alarm", "calendar"]),
            [user("Tao lich hop voi An 3 gio chieu nay va dat nhac truoc 10 phut.")],
            [
                {"name": "calendar.create_event", "arguments": {"title": "Hop voi An", "start": "2026-11-07T15:00:00+07:00"}},
                {"name": "alarm.set", "arguments": {"time": "14:50", "label": "meeting"}},
            ],
            [
                {"name": "calendar.create_event", "ok": True, "result": {"event_id": "evt_9"}},
                {"name": "alarm.set", "ok": True, "result": {"alarm_id": "alarm_2"}},
            ],
        )
    )
    rows[-1]["audio_text"] = "Tao lich hop voi An 3 gio chieu nay va dat nhac truoc 10 phut."

    rows.append(
        rec(
            "train_read_before_write",
            "read_before_write",
            context(["smart_home"]),
            [
                user("Bat den phong khach."),
                asst("", [{"name": "smart_home.list_devices", "arguments": {"room": "living_room"}}]),
                tool("smart_home.list_devices", {"ok": True, "result": {"devices": [{"id": "light_living", "on": False}]}}),
            ],
            [{"name": "smart_home.device_set", "arguments": {"device_id": "light_living", "on": True}}],
            [{"name": "smart_home.device_set", "ok": True, "result": {"device_id": "light_living", "on": True}}],
            {"apps": {"smart_home": {"light_living": {"id": "light_living", "on": True}}}},
        )
    )
    rows[-1]["audio_text"] = "Bat den phong khach."

    rows.append(
        rec(
            "train_result_dependent_arg",
            "result_dependent_arg",
            context(["calendar", "messaging"]),
            [user("Nhan vao nhom Cong ty roi gui loi cam on.")],
            [
                {"name": "messaging.search_contacts", "arguments": {"query": "Cong ty"}},
                {"name": "messaging.send", "arguments": {"group_id": "grp_42", "body": "Cam on ca nha"}},
            ],
            [
                {"name": "messaging.search_contacts", "ok": True, "result": {"contacts": [{"id": "grp_42", "kind": "group"}]}},
                {"name": "messaging.send", "ok": True, "result": {"message_id": "msg_1"}},
            ],
        )
    )
    rows[-1]["audio_text"] = "Nhan vao nhom Cong ty roi gui loi cam on."

    rows.append(
        rec(
            "train_missing_info",
            "missing_info",
            context(["calendar"]),
            [user("Dat lich hop voi sep.")],
            [{"name": "sys.ask_missing_info", "arguments": {"missing_fields": ["start_time"]}}],
            [{"name": "sys.ask_missing_info", "ok": True, "result": {"question": "Luc may gio?"}}],
        )
    )
    rows[-1]["audio_text"] = "Dat lich hop voi sep."

    rows.append(
        rec(
            "train_out_of_scope",
            "out_of_scope",
            context(["alarm"]),
            [user("Dat ve may bay di Ha Noi.")],
            [],
            [],
        )
    )
    rows[-1]["audio_text"] = "Dat ve may bay di Ha Noi."

    rows.append(
        rec(
            "train_invalid_arguments",
            "invalid_arguments",
            context(["alarm"]),
            [user("Dat bao thuc luc 25 gio.")],
            [{"name": "alarm.set", "arguments": {"time": "25:00", "repeat": "daily"}}],
            [{"name": "alarm.set", "ok": False, "error": "invalid time format: hour must be 00-23"}],
        )
    )
    rows[-1]["audio_text"] = "Dat bao thuc luc 25 gio."

    rows.append(
        rec(
            "train_failed_tool_result",
            "failed_tool_result",
            context(["messaging"]),
            [user("Gui tin nhan cho Lan.")],
            [
                {"name": "messaging.search_contacts", "arguments": {"query": "Lan"}},
                {"name": "messaging.send", "arguments": {"contact_id": "c_404", "body": "Chao Lan"}},
            ],
            [
                {"name": "messaging.search_contacts", "ok": True, "result": {"contacts": []}},
                {"name": "messaging.send", "ok": False, "error": "contact not found"},
            ],
        )
    )
    rows[-1]["audio_text"] = "Gui tin nhan cho Lan."

    rows.append(
        rec(
            "train_repeated_failed_calls",
            "repeated_failed_calls",
            context(["smart_home"]),
            [user("Tat dieu hoa.")],
            [
                {"name": "smart_home.device_set", "arguments": {"device_id": "ac_1", "on": False}},
                {"name": "smart_home.device_set", "arguments": {"device_id": "ac_1", "on": False}},
            ],
            [
                {"name": "smart_home.device_set", "ok": False, "error": "device offline"},
                {"name": "smart_home.device_set", "ok": False, "error": "device offline"},
            ],
        )
    )
    rows[-1]["audio_text"] = "Tat dieu hoa."

    rows.append(
        rec(
            "train_empty_tool_calls",
            "empty_tool_calls",
            context(["alarm"]),
            [user("Bao thuc 6 gio da co chua?")],
            [],
            [],
        )
    )
    rows[-1]["audio_text"] = "Bao thuc 6 gio da co chua?"

    rows.append(
        rec(
            "train_multiple_installed_apps",
            "multiple_installed_apps",
            context(["alarm", "calendar", "messaging", "smart_home"]),
            [user("Nhac An hop luc 9 gio va bat den phong hop.")],
            [
                {"name": "messaging.send", "arguments": {"contact_id": "c_an", "body": "9h hop nhe"}},
                {"name": "smart_home.device_set", "arguments": {"device_id": "light_meeting", "on": True}},
            ],
            [
                {"name": "messaging.send", "ok": True, "result": {"message_id": "msg_2"}},
                {"name": "smart_home.device_set", "ok": True, "result": {"device_id": "light_meeting", "on": True}},
            ],
        )
    )
    rows[-1]["audio_text"] = "Nhac An hop luc 9 gio va bat den phong hop."

    long_history = [user("Xin chao.")]
    for idx in range(6):
        long_history.append(asst(f"Ban can gi them khong? (luot {idx})"))
        long_history.append(user(f"Toi muon dat bao thuc, lan {idx}."))
    long_history.append(user("Duoc roi, dat 7 gio sang mai."))
    rows.append(
        rec(
            "train_long_history",
            "long_history",
            context(["alarm"]),
            long_history,
            [{"name": "alarm.set", "arguments": {"time": "07:00", "repeat": "once"}}],
            [{"name": "alarm.set", "ok": True, "result": {"alarm_id": "alarm_7"}}],
        )
    )
    rows[-1]["audio_text"] = "Duoc roi, dat 7 gio sang mai."

    rows.append(
        rec(
            "train_missing_optional_fields",
            "missing_optional_fields",
            context(["alarm"], full=False),
            [user("Tat tat ca bao thuc.")],
            [{"name": "alarm.list", "arguments": {}}],
            [{"name": "alarm.list", "ok": True, "result": {"alarms": []}}],
        )
    )
    rows[-1]["audio_text"] = "Tat tat ca bao thuc."

    rows.append(
        rec(
            "train_unknown_fields",
            "unknown_fields",
            {**context(["calendar"]), "wifi_ssid": "test-net", "model_hint": "ignored"},
            [{"role": "user", "content": "Xem lich tuan nay.", "lang": "vi"}],
            [{"name": "calendar.search_events", "arguments": {"query": "tuan nay"}}],
            [{"name": "calendar.search_events", "ok": True, "result": {"events": []}}],
        )
    )
    rows[-1]["audio_text"] = "Xem lich tuan nay."
    rows[-1]["extra_top_level"] = {"note": "tolerated unknown field"}
    rows[-1]["audio_text_normalized"] = "Xem lich tuan nay."
    return rows


def public_test_records() -> list[dict]:
    rows = [
        {
            "_synthetic": True,
            "id": "test_single_tool",
            "category": "single_tool",
            "audio_path": "test_single_tool/test_single_tool.wav",
            "audio_text": "Het gio, tat den.",
            "context": context(["smart_home"]),
            "history": [user("Het gio, tat den.")],
        },
        {
            "_synthetic": True,
            "id": "test_missing_info",
            "category": "missing_info",
            "audio_path": "test_missing_info/test_missing_info.wav",
            "audio_text": "Dat lich hop.",
            "context": context(["calendar"]),
            "history": [user("Dat lich hop.")],
        },
        {
            "_synthetic": True,
            "id": "test_missing_audio",
            "category": "missing_audio",
            "audio_path": "test_missing_audio/test_missing_audio.wav",
            "audio_text": "Gui tin nhan chao.",
            "context": context(["messaging"]),
            "history": [user("Gui tin nhan chao.")],
        },
    ]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build() -> None:
    (ROOT / "train" / "audio").mkdir(parents=True, exist_ok=True)
    (ROOT / "public_test" / "audio").mkdir(parents=True, exist_ok=True)
    (ROOT / "mock_state").mkdir(parents=True, exist_ok=True)
    (ROOT / "malformed").mkdir(parents=True, exist_ok=True)
    (ROOT / "broken").mkdir(parents=True, exist_ok=True)

    rows = train_records()
    write_jsonl(ROOT / "train" / "train.jsonl", rows)
    for idx, row in enumerate(rows):
        sid = row["id"]
        write_wav(ROOT / "train" / "audio" / sid / f"{sid}.wav", 0.5 + 0.125 * (idx % 20), tone_name(idx))

    test_rows = public_test_records()
    write_jsonl(ROOT / "public_test" / "public_test.jsonl", test_rows)
    for idx, row in enumerate(test_rows[:-1]):
        sid = row["id"]
        write_wav(ROOT / "public_test" / "audio" / sid / f"{sid}.wav", 1.0, tone_name(idx + 3))

    (ROOT / "train" / "train_tools_catalog.json").write_text(
        json.dumps(catalog(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "public_test" / "public_test_tools_catalog.json").write_text(
        json.dumps(catalog(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    write_jsonl(
        ROOT / "train" / "sample_submission.jsonl",
        [{"_synthetic": True, "id": row["id"], "tool_calls": [], "tool_results": []} for row in rows],
    )

    states = {
        "train_read_before_write": {
            "_synthetic": True,
            "apps": {
                "smart_home": {
                    "light_living": {"id": "light_living", "room": "living_room", "on": False, "brightness": 40}
                }
            },
        },
        "train_multi_step": {"_synthetic": True, "apps": {"alarm": {}, "calendar": {}}},
        "train_failed_tool_result": {"_synthetic": True, "apps": {"messaging": {"contacts": []}}},
    }
    for sid, state in states.items():
        (ROOT / "mock_state" / f"{sid}.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "_synthetic": True,
        "_warning": "SYNTHETIC TEST DATA / NOT OFFICIAL COMPETITION DATA",
        "categories": {},
        "splits": {},
    }
    for row in rows:
        manifest["splits"].setdefault("train", []).append(row["id"])
        manifest["categories"][row["id"]] = row["category"]
    for row in test_rows:
        manifest["splits"].setdefault("public_test", []).append(row["id"])
        manifest["categories"][row["id"]] = row["category"]
    (ROOT / "synthetic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    malformed_lines = [
        json.dumps({"id": "malformed_ok", "audio_text": "fine", "context": context(["alarm"]), "history": []}),
        '{"id": "broken_json", "audio_text": "unterminated',
        "",
        "[1, 2, 3]",
        json.dumps({"audio_text": "no id here"}),
    ]
    (ROOT / "malformed" / "malformed.jsonl").write_text(
        "\n".join(malformed_lines) + "\n", encoding="utf-8"
    )

    (ROOT / "broken" / "zero_length.wav").write_bytes(b"")

    (ROOT / "SYNTHETIC.md").write_text(
        "# SYNTHETIC TEST DATA / NOT OFFICIAL COMPETITION DATA\n\n"
        "Every file under this directory is generated by `tests/build_fixtures.py`.\n"
        "It exists only to exercise the loader, validator, catalog and mock\n"
        "environment on CPU with stdlib-only dependencies. Shapes are *plausible\n"
        "guesses*, not the organizer's schema. Do not report numbers from this\n"
        "data as results.\n\n"
        "No `*_environment` directory exists here on purpose: the mock environment\n"
        "is seeded from `mock_state/<id>.json`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
    print(f"synthetic fixtures written to {ROOT}")
