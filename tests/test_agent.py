"""Closed-loop agent tests (stdlib only).

Run with: python -m pytest tests/test_agent.py -q
Uses a hand-written fake environment implementing the ``ToolEnvironment``
protocol with the same semantics as ``lifemate.env.mock.MockToolEnv``:
every call (successful or not) is appended to the log, observations are
returned as dicts, and the state hash covers mutable state only.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent.agent import AgentResult, ToolAgent  # noqa: E402
from lifemate.agent.protocol import (  # noqa: E402
    FINISH_ACTION,
    make_tool_call,
    parse_action,
    render_action,
)
from lifemate.agent.prompt_builder import (  # noqa: E402
    StepRecord,
    build_decision_prompt,
    compact_history,
    estimate_prompt_tokens,
)
from lifemate.agent.stopping_policy import StoppingPolicy, signature  # noqa: E402
from lifemate.agent.tool_parser import parse_model_output  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.data.catalog import ToolCatalog  # noqa: E402
from lifemate.env.base import ValidationOutcome  # noqa: E402

NEEDS_GPU: list[str] = []

GOLD_SENTINEL = "GOLD_FINAL_STATE_SENTINEL"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def catalog() -> ToolCatalog:
    def obj(props: dict, required: list[str]) -> dict:
        return {"type": "object", "properties": props, "required": required}

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
                            "parameters": obj({}, []),
                        },
                        {
                            "name": "alarm.set",
                            "description": "Create an alarm.",
                            "annotations": {"readOnlyHint": False},
                            "parameters": obj(
                                {
                                    "time": {"type": "string"},
                                    "label": {"type": "string"},
                                    "repeat": {"type": "string", "enum": ["once", "daily"]},
                                },
                                ["time"],
                            ),
                        },
                    ],
                },
                {
                    "app_name": "messaging",
                    "description": "Messages.",
                    "tools": [
                        {
                            "name": "messaging.search_contacts",
                            "description": "Find a contact.",
                            "parameters": obj({"query": {"type": "string"}}, ["query"]),
                        },
                        {
                            "name": "messaging.send",
                            "description": "Send a message.",
                            "annotations": {"readOnlyHint": False},
                            "parameters": obj(
                                {"contact_id": {"type": "string"}, "body": {"type": "string"}},
                                ["contact_id", "body"],
                            ),
                        },
                    ],
                },
                {
                    "app_name": "sys",
                    "description": "System tools.",
                    "tools": [
                        {
                            "name": "sys.ask_missing_info",
                            "description": "Ask the user for missing information.",
                            "parameters": obj({"question": {"type": "string"}}, []),
                        }
                    ],
                },
            ]
        }
    )


class Ctx:
    def __init__(self, apps=("alarm", "messaging", "sys")):
        self.current_time = "2026-11-07T08:00:00+07:00"
        self.timezone = "Asia/Ho_Chi_Minh"
        self.current_location = "Hà Nội"
        self.user_profile = {"name": "An", "phone": "+84900000000"}
        self.installed_apps = list(apps)


class HistoryTurn:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.tool_call = None
        self.tool_result = None
        self.raw = {}


class Sample:
    def __init__(self, apps=("alarm", "messaging", "sys"), history=None, text="đặt báo thức 7 giờ sáng"):
        self.id = "sample-1"
        self.split = "train"
        self.audio_path = None
        self.audio_text = text
        self.context = Ctx(apps)
        self.history = list(history or [])
        self.label = {"final_state": GOLD_SENTINEL, "tool_calls": [{"name": "GOLD_TOOL_SENTINEL"}]}


class FakeEnv:
    """Same semantics as MockToolEnv, hand-written to keep tests self-contained."""

    name = "fake"

    def __init__(self, cat=None, apps=("alarm", "messaging", "sys"), state=None, fail_counts=None):
        self.catalog = cat or catalog()
        self.apps = list(apps)
        self.state = {"apps": json.loads(json.dumps(state or {}))}
        self.writes: list[dict] = []
        self.log: list[dict] = []
        self.fail_counts = dict(fail_counts or {})

    def available_tools(self) -> list[str]:
        return [s.name for s in self.catalog.tools_for_apps(self.apps)]

    def validate(self, tool_name: str, arguments: dict) -> ValidationOutcome:
        rules = self.catalog.validate_call(tool_name, arguments)
        return ValidationOutcome.from_errors(list(rules["errors"]))

    def call(self, tool_name: str, arguments: dict) -> dict:
        args = dict(arguments or {})
        spec = self.catalog.get(tool_name)
        if spec is None or tool_name not in self.available_tools():
            return self._record(tool_name, args, False, error=f"unknown tool: {tool_name}")
        report = self.catalog.validate_call(tool_name, args)
        if report["errors"]:
            return self._record(tool_name, args, False, error="; ".join(report["errors"]))
        if self.fail_counts.get(tool_name, 0) > 0:
            self.fail_counts[tool_name] -= 1
            return self._record(tool_name, args, False, error=f"simulated failure for {tool_name}")
        if spec.kind == "meta":
            return self._record(tool_name, args, True, result={"question": args.get("question", "")})
        if spec.kind == "read":
            bucket = self.state["apps"].get(spec.app_name, {})
            return self._record(tool_name, args, True, result=bucket)
        key = args.get("alarm_id") or args.get("contact_id") or tool_name.split(".")[-1]
        self.state["apps"].setdefault(spec.app_name, {})[str(key)] = args
        self.writes.append({"tool": tool_name, "key": str(key), "arguments": args})
        return self._record(tool_name, args, True, result={"ok": True, "key": str(key), "arguments": args})

    def get_tool_calls(self) -> list[dict]:
        return self.log  # live object on purpose: identity is contractual

    def get_state_hash(self) -> str:
        blob = json.dumps(self.state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _record(self, name, args, ok, result=None, error=None) -> dict:
        entry = {"name": name, "arguments": dict(args), "ok": ok}
        if ok:
            entry["result"] = result
        else:
            entry["error"] = error
        self.log.append(entry)
        return entry


class FakeGenerator:
    """Scripted generator: pops one output per call, records the prompts."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append([dict(m) for m in messages])
        if not self.outputs:
            return render_action(FINISH_ACTION)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def prompts(self) -> list[str]:
        return [calls[1]["content"] for calls in self.calls]


class DeterministicScriptedGenerator:
    """Minimal scripted generator used by tests (never shipped in the package).

    Same duck type as ``HFGenerator``: ``.generate(messages) -> str``.
    """

    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append([dict(m) for m in messages])
        if not self.outputs:
            raise AssertionError("DeterministicScriptedGenerator ran out of scripted outputs")
        return self.outputs.pop(0)


def tool(name, **arguments) -> str:
    return render_action(make_tool_call(name, arguments))


FINISH = render_action(FINISH_ACTION)


def build_agent(outputs, env=None, sample=None, policy=None, apps=("alarm", "messaging", "sys")):
    cat = catalog()
    registry = ToolRegistry(cat, apps=list(apps))
    env = env or FakeEnv(cat, apps=apps)
    generator = FakeGenerator(outputs)
    agent = ToolAgent(generator, registry, env, policy=policy)
    return agent, generator, env, sample or Sample(apps=apps)


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
def test_render_and_parse_roundtrip():
    text = render_action(make_tool_call("alarm.set", {"time": "07:00"}))
    assert text == '<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": "07:00"}}</action>'
    action = parse_action(text)
    assert action.is_tool_call and action.name == "alarm.set"


def test_parse_tolerates_prose_and_fences():
    assert parse_action("```json\n<action>{\"type\":\"finish\"}</action>\n```").is_finish
    assert parse_action("suy nghĩ... <action>{\"type\": \"finish\"}</action> xong").is_finish


def test_parse_rejects_bad_tool_calls():
    assert parse_action('<action>{"type": "tool_call"}</action>').type == "invalid"
    bad_args = parse_action('<action>{"type": "tool_call", "name": "alarm.list", "arguments": 3}</action>')
    assert bad_args.type == "invalid" and "arguments" in bad_args.error


def test_parse_garbage_never_raises():
    for text in ("", "no tags", "<action>not json</action>", "</action>", "<action>{}</action>"):
        assert parse_action(text).type == "invalid"


def test_parse_legacy_raw_json_name():
    # Documented best-effort tolerance: bare JSON with a "name" key.
    bare = parse_action('{"name": "alarm.list", "arguments": {}}')
    assert bare.is_tool_call and bare.name == "alarm.list"
    legacy = parse_action('<action>{"name": "alarm.list", "arguments": {}}</action>')
    assert legacy.is_tool_call and legacy.name == "alarm.list"
    assert parse_action('{"unknown": 1}').type == "invalid"


def test_parser_marks_unknown_and_unavailable():
    registry = ToolRegistry(catalog(), apps=["alarm"])
    unknown = parse_model_output(tool("nope.nope"), registry)
    assert unknown.type == "invalid" and "nope.nope" in unknown.error
    uninstalled = parse_model_output(tool("messaging.send", contact_id="c1", body="hi"), registry)
    assert uninstalled.type == "invalid" and "messaging.send" in uninstalled.error


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def test_prompt_deterministic_and_label_free():
    sample = Sample(history=[HistoryTurn("user", "chào"), HistoryTurn("assistant", "chào bạn")])
    registry = ToolRegistry(catalog(), apps=sample.context.installed_apps)
    kwargs = dict(
        transcript=sample.audio_text,
        context=sample.context,
        history=sample.history,
        installed_apps=sample.context.installed_apps,
        catalog=registry,
        prior_steps=[],
    )
    first = build_decision_prompt(**kwargs)
    second = build_decision_prompt(**kwargs)
    assert first == second, "prompt must be byte-identical for identical inputs"
    assert GOLD_SENTINEL not in first and "GOLD_TOOL_SENTINEL" not in first
    for header in ("# Yêu cầu", "# Ngữ cảnh", "# Lịch sử hội thoại", "# Công cụ khả dụng", "# Định dạng trả lời"):
        assert header in first
    assert "alarm.set" in first and "required: time" in first


def test_prompt_omits_steps_section_when_empty():
    registry = ToolRegistry(catalog(), apps=["alarm"])
    prompt = build_decision_prompt(
        transcript="x", context=Ctx(), history=[], installed_apps=["alarm"], catalog=registry, prior_steps=[]
    )
    assert "# Các bước đã thực hiện" not in prompt


def test_compact_history_and_token_estimate():
    turns = [HistoryTurn("user", f"turn{i}") for i in range(30)]
    kept = compact_history(turns, 10)
    assert len(kept) == 10 and kept[-1].content == "turn29"
    assert compact_history(turns, 0) == []
    assert estimate_prompt_tokens("a" * 400) == 100
    assert compact_history(None, 5) == []


def test_step_record_from_call_log():
    steps = StepRecord.from_call_log(
        [
            {"name": "alarm.list", "arguments": {}, "ok": True, "result": {"x": 1}},
            {"name": "alarm.set", "arguments": {"time": "07:00"}, "ok": False, "error": "boom"},
        ]
    )
    assert [s.index for s in steps] == [0, 1]
    assert steps[0].action["name"] == "alarm.list"
    assert steps[1].observation == {"ok": False, "error": "boom"}


# --------------------------------------------------------------------------- #
# loop behaviour
# --------------------------------------------------------------------------- #
def test_single_tool_then_finish():
    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH])
    result = agent.run(sample)
    assert result.status == "finished" and result.finish_reason == "finish"
    assert [c["name"] for c in result.tool_calls] == ["alarm.list"]
    assert result.n_llm_calls == 2
    assert len(result.steps) == 1 and result.steps[0].action["name"] == "alarm.list"


def test_multi_step_two_calls():
    agent, gen, env, sample = build_agent(
        [tool("alarm.list"), tool("alarm.set", time="07:00"), FINISH]
    )
    result = agent.run(sample)
    assert [s.action["name"] for s in result.steps] == ["alarm.list", "alarm.set"]
    assert result.status == "finished"


def test_result_dependent_next_args_visible_in_prompt():
    env = FakeEnv(state={"alarm": {"alarm_1": {"time": "06:30"}}})
    agent, gen, env, sample = build_agent(
        [tool("alarm.list"), tool("alarm.set", time="06:30"), FINISH], env=env
    )
    result = agent.run(sample)
    assert result.status == "finished"
    second_prompt = gen.prompts[1]
    assert "alarm_1" in second_prompt, "step-2 prompt must contain the step-1 observation"
    assert "# Các bước đã thực hiện" in second_prompt


def test_read_before_write_ordering():
    env = FakeEnv(state={"messaging": {"contact_9": {"name": "Bình"}}})
    agent, gen, env, sample = build_agent(
        [
            tool("messaging.search_contacts", query="Bình"),
            tool("messaging.send", contact_id="contact_9", body="chào"),
            FINISH,
        ],
        env=env,
    )
    result = agent.run(sample)
    assert [c["name"] for c in result.tool_calls] == [
        "messaging.search_contacts",
        "messaging.send",
    ]
    assert "contact_9" in gen.prompts[1]


def test_tool_failure_observed_then_recovery():
    env = FakeEnv(fail_counts={"messaging.search_contacts": 1})
    agent, gen, env, sample = build_agent(
        [
            tool("messaging.search_contacts", query="Bình"),
            tool("messaging.search_contacts", query="Binh"),
            FINISH,
        ],
        env=env,
        policy=StoppingPolicy(max_steps=6, max_consecutive_failures=2),
    )
    result = agent.run(sample)
    assert result.status == "finished"
    assert result.tool_calls[0]["ok"] is False
    assert result.steps[0].observation["ok"] is False
    assert "simulated failure" in result.steps[0].observation["error"]


def test_repeated_identical_failure_stops():
    env = FakeEnv(fail_counts={"alarm.set": 10})
    same = tool("alarm.set", time="25:99")
    agent, gen, env, sample = build_agent([same, same, same, same], env=env)
    result = agent.run(sample)
    assert result.status == "repeated_failure"
    assert result.finish_reason == "repeated_failure"
    assert len(result.steps) == 2


def test_env_error_then_finish():
    env = FakeEnv(fail_counts={"alarm.list": 1})
    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH], env=env)
    result = agent.run(sample)
    assert result.status == "finished"
    assert result.tool_calls[0]["ok"] is False
    assert len(result.tool_calls) == 1


def test_finish_immediately_is_out_of_scope_no_calls():
    agent, gen, env, sample = build_agent([FINISH])
    result = agent.run(sample)
    assert result.status == "finished"
    assert result.tool_calls == []
    assert env.get_tool_calls() == []
    assert result.state_hash == env.get_state_hash()
    assert "alarm.list" in gen.prompts[0]  # tools still offered


def test_missing_info_via_meta_tool():
    agent, gen, env, sample = build_agent(
        [tool("sys.ask_missing_info", question="Bạn muốn đặt báo thức lúc mấy giờ?"), FINISH]
    )
    result = agent.run(sample)
    assert result.status == "finished"
    assert result.tool_calls[0]["name"] == "sys.ask_missing_info"
    assert "mấy giờ" in result.tool_calls[0]["result"]["question"]


def test_invalid_json_output_is_fed_back():
    agent, gen, env, sample = build_agent(["tôi sẽ đặt báo thức", FINISH])
    result = agent.run(sample)
    assert result.status == "finished"
    assert result.tool_calls == [], "invalid action must never reach the environment"
    assert result.steps[0].observation["note"] == "invalid action, not executed"
    assert result.errors


def test_unknown_tool_not_executed():
    agent, gen, env, sample = build_agent([tool("ghost.tool", x=1), FINISH])
    result = agent.run(sample)
    assert result.tool_calls == []
    assert "ghost.tool" in result.steps[0].observation["error"]


def test_uninstalled_tool_not_executed():
    agent, gen, env, sample = build_agent(
        [tool("messaging.send", contact_id="c", body="x"), FINISH], apps=("alarm",)
    )
    result = agent.run(sample)
    assert result.tool_calls == []
    assert "messaging.send" in result.steps[0].observation["error"]
    assert "messaging.send" not in gen.prompts[0]


def test_missing_required_arg_rejected_before_env():
    agent, gen, env, sample = build_agent([tool("alarm.set"), FINISH])
    result = agent.run(sample)
    assert result.tool_calls == []
    assert "time" in result.steps[0].observation["error"]


def test_invalid_arg_type_rejected():
    agent, gen, env, sample = build_agent([tool("alarm.set", time=7), FINISH])
    result = agent.run(sample)
    assert result.tool_calls == []
    assert "expected type string" in result.steps[0].observation["error"]


def test_invalid_enum_rejected():
    agent, gen, env, sample = build_agent(
        [tool("alarm.set", time="07:00", repeat="hourly"), FINISH]
    )
    result = agent.run(sample)
    assert result.tool_calls == []
    assert "enum" in result.steps[0].observation["error"]


def test_max_steps_reached():
    agent, gen, env, sample = build_agent(
        [tool("alarm.list")] * 5,
        policy=StoppingPolicy(max_steps=2, max_repeated_identical_calls=99),
    )
    result = agent.run(sample)
    assert result.status == "max_steps" and result.finish_reason == "max_steps"
    assert len(result.steps) == 2


def test_time_budget_reached_with_injected_clock():
    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH])
    ticks = iter([0.0, 0.0, 500.0, 500.0])
    agent.clock = lambda: next(ticks, 500.0)
    result = agent.run(sample)
    assert result.status == "time_budget" and result.finish_reason == "time_budget"
    assert result.elapsed_seconds >= 120.0


def test_deadline_argument_stops_run():
    agent, gen, env, sample = build_agent([FINISH])
    agent.clock = lambda: 1000.0
    result = agent.run(sample, deadline=999.0)
    assert result.status == "time_budget"
    assert result.n_llm_calls == 0


def test_empty_history_prompt():
    agent, gen, env, sample = build_agent([FINISH], sample=Sample(history=[]))
    agent.run(sample)
    assert "(không có)" in gen.prompts[0]


def test_long_history_is_compacted():
    turns = [HistoryTurn("user", f"lượt {i} " + "x" * 50) for i in range(60)]
    agent, gen, env, sample = build_agent([FINISH], sample=Sample(history=turns))
    agent.history_limit = 10
    agent.run(sample)
    prompt = gen.prompts[0]
    assert "lượt 59" in prompt and "lượt 50" in prompt
    assert "lượt 49 " not in prompt and "lượt 10 " not in prompt
    assert len(prompt) < 20000, f"prompt not bounded: {len(prompt)} chars"


def test_state_hash_and_tool_calls_come_from_env():
    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH])
    result = agent.run(sample)
    assert result.tool_calls is env.log, "tool_calls must be the env log object verbatim"
    assert result.state_hash == env.get_state_hash()
    assert result is not None


def test_generator_exception_becomes_error_status():
    agent, gen, env, sample = build_agent([RuntimeError("model server exploded")])
    result = agent.run(sample)
    assert result.status == "error" and result.finish_reason == "generator_error"
    assert any("model server exploded" in e for e in result.errors)
    assert result.tool_calls == []


def test_env_exception_is_captured_not_raised():
    class ExplodingEnv(FakeEnv):
        def call(self, tool_name, arguments):
            raise RuntimeError("env down")

    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH], env=ExplodingEnv())
    result = agent.run(sample)
    assert result.status == "finished"
    assert result.steps[0].observation["ok"] is False
    assert "env down" in result.steps[0].observation["error"]


def test_prohibited_access_is_never_captured_as_an_observation():
    """A compliance violation must stop the run, not become model-visible data.

    A ``*_environment`` read is a competition-rule violation. Turning it into
    ``{"ok": false, "error": ...}`` keeps the pipeline alive, lets the model
    "recover" from it, and produces a submission built on a banned read.
    """
    from lifemate.errors import ProhibitedAccessError

    class ProhibitedEnv(FakeEnv):
        def call(self, tool_name, arguments):
            raise ProhibitedAccessError("data/train_environment/a.jsonl")  # compliance-scan: allow

    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH], env=ProhibitedEnv())
    try:
        agent.run(sample)
    except ProhibitedAccessError as exc:
        assert "train_environment" in exc.path
    else:  # pragma: no cover
        raise AssertionError("agent downgraded a ProhibitedAccessError to an observation")


def test_sample_as_dict_and_transcript_override():
    agent, gen, env, sample = build_agent([FINISH])
    fields = {
        "id": "d1",
        "context": sample.context,
        "history": [],
        "installed_apps": ["alarm"],
    }
    result = agent.run(fields, transcript="mở danh sách báo thức")
    assert result.status == "finished" and result.sample_id == "d1"
    assert "mở danh sách báo thức" in gen.prompts[0]


def test_no_gold_labels_in_any_prompt():
    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH])
    agent.run(sample)
    for prompt in gen.prompts:
        assert GOLD_SENTINEL not in prompt
        assert "GOLD_TOOL_SENTINEL" not in prompt
        assert "final_state" not in prompt


def test_run_batch():
    agent, gen, env, sample = build_agent([FINISH, FINISH])
    results = agent.run_batch([sample, sample])
    assert [r.status for r in results] == ["finished", "finished"]
    try:
        agent.run_batch([sample], transcripts=["a", "b"])
    except ValueError as exc:
        assert "length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("run_batch must reject mismatched transcripts")


def test_agent_requires_generator():
    try:
        ToolAgent(None, ToolRegistry(catalog(), apps=["alarm"]), FakeEnv())
    except ValueError as exc:
        assert "generator" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ToolAgent must reject a missing generator")


def test_signature_stability_and_policy_reasons():
    assert signature(make_tool_call("alarm.list", {})) == signature(
        parse_action(tool("alarm.list")).to_dict()
    )
    policy = StoppingPolicy(max_steps=1)
    assert policy.should_stop(0, None, 0.0) == (False, "")
    assert policy.should_stop(1, None, 0.0) == (True, "max_steps")
    assert StoppingPolicy().should_stop(0, parse_action(FINISH)) == (True, "finish")
    assert StoppingPolicy(time_budget_seconds=1.0).should_stop(0, None, 5.0) == (True, "time_budget")


def test_scripted_generator_double_and_config_policy():
    from lifemate.config import AgentConfig

    class _Cfg:
        agent = AgentConfig(max_steps=2, time_budget_seconds=30.0, stop_on_finish=True)

    gen = DeterministicScriptedGenerator([tool("alarm.list"), FINISH])
    env, sample = FakeEnv(), Sample()
    agent = ToolAgent(gen, ToolRegistry(catalog(), apps=sample.context.installed_apps), env, config=_Cfg())
    assert agent.policy.max_steps == 2 and agent.policy.time_budget_seconds == 30.0
    result = agent.run(sample)
    assert result.status == "finished" and result.n_llm_calls == 2


def test_executor_counts_skipped_invalid_actions_without_env_calls():
    from lifemate.agent.tool_executor import ToolExecutor

    env = FakeEnv()
    executor = ToolExecutor(env, ToolRegistry(catalog(), apps=["alarm"]))
    observation = executor.execute_validated(parse_model_output(tool("ghost.tool"), ToolRegistry(catalog(), apps=["alarm"])))
    assert observation["ok"] is False and observation["skipped"] is True
    assert executor.failure_streak == 1
    assert env.log == [], "skipped calls must not reach the environment log"
    executor.reset()
    assert executor.failure_streak == 0


def test_agent_result_to_dict_is_serialisable():
    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH])
    payload = json.loads(json.dumps(agent.run(sample).to_dict(), ensure_ascii=False))
    assert payload["status"] == "finished" and payload["n_steps"] == 1
    assert isinstance(AgentResult(status="finished").tool_calls, list)


def test_multi_turn_history_with_tool_turns():
    turn = HistoryTurn("assistant", "gọi alarm.list")
    turn.tool_call = make_tool_call("alarm.list", {})
    turn.tool_result = {"ok": True, "result": {"alarm_1": {"time": "06:30"}}}
    agent, gen, env, sample = build_agent([FINISH], sample=Sample(history=[turn]))
    result = agent.run(sample)
    assert result.status == "finished"
    prompt = gen.prompts[0]
    assert "Trợ lý: gọi alarm.list" in prompt
    assert "Người dùng:" not in prompt  # no user turn invented
    assert "alarm_1" in prompt


def test_catalog_specs_empty_installed_apps_only_has_meta():
    from lifemate.agent.prompt_builder import _catalog_specs, build_decision_prompt
    cat = catalog()
    specs = _catalog_specs(cat, installed_apps=[])
    assert len(specs) == 1 and specs[0].name == "sys.ask_missing_info"
    prompt = build_decision_prompt(
        transcript="test",
        context=Ctx(),
        history=[],
        installed_apps=[],
        catalog=cat,
        prior_steps=[],
    )
    assert "sys.ask_missing_info" in prompt
    assert "alarm.set" not in prompt


def test_agent_records_env_exception_into_errors():
    class ExplodingEnv(FakeEnv):
        def call(self, tool_name: str, arguments: dict) -> dict:
            raise RuntimeError("SDK exploded during execution")

    agent, gen, env, sample = build_agent([tool("alarm.list"), FINISH], env=ExplodingEnv())
    result = agent.run(sample)
    assert any("environment raised RuntimeError: SDK exploded during execution" in e for e in result.errors)

def test_needs_gpu_report():
    assert NEEDS_GPU == []

# --------------------------------------------------------------------------- #
# validation authority + malformed output (Phase D/H)
# --------------------------------------------------------------------------- #
class StrictEnv(FakeEnv):
    """Env whose validator enforces a rule the local catalog does not (empty value)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validated: list[tuple[str, dict]] = []

    def validate(self, tool_name, arguments):
        self.validated.append((tool_name, dict(arguments)))
        if tool_name == "alarm.set" and not str(arguments.get("time") or "").strip():
            return ValidationOutcome(ok=False, errors=["time phải có dạng date-time"])
        return ValidationOutcome()

def test_env_validator_is_authoritative_over_catalog():
    """A call the official env accepts must never be rejected locally, and vice versa."""
    env = FakeEnv(catalog(), apps=("alarm", "messaging", "sys"))
    registry = ToolRegistry(catalog(), apps=["alarm", "messaging", "sys"])
    # The local catalog has no rule for empty required values; the env does.
    strict = StrictEnv(catalog(), apps=("alarm", "messaging", "sys"))
    accepted_by_env = parse_model_output(tool("alarm.set", time=" "), registry, strict)
    assert accepted_by_env.type == "invalid" and "time" in accepted_by_env.error
    assert strict.validated, "the env validator must actually be consulted"
    # With no env validator the registry stays the gate (backwards compatible).
    assert parse_model_output(tool("alarm.set", time=" "), registry).is_tool_call

def test_malformed_output_never_counts_as_success():
    """No malformed output may reach the env or be recorded as a success."""
    for bad in (
        "không có action nào",
        "<action>not json</action>",
        '<action>{"type": "tool_call", "name": "alarm.list", "arguments": 7}</action>',
        '<action>{"type": "tool_call", "name": "ghost.tool", "arguments": {}}</action>',
        tool("alarm.set"),           # missing required `time`
        tool("alarm.set", time=7),   # wrong arg type
        tool("messaging.send", contact_id="c1", body="x"),  # installed? no (see apps below)
    ):
        # `messaging` is deliberately NOT installed, so an uninstalled-app call is
        # part of the malformed set: the registry, not the env, must reject it.
        agent, gen, env, sample = build_agent([bad, FINISH], apps=("alarm", "sys"))
        result = agent.run(sample)
        assert result.tool_calls == [], f"malformed action reached env: {bad!r}"
        assert len(result.steps) == 1 and result.steps[0].observation["ok"] is False, bad
        assert result.steps[0].observation.get("skipped") is True, bad
        assert result.errors, f"failure not recorded for {bad!r}"

def test_consecutive_malformed_outputs_stop_the_loop():
    """Two invalid actions in a row hit the failure streak: stop, never execute."""
    agent, gen, env, sample = build_agent(["garbage one", "garbage two", FINISH])
    result = agent.run(sample)
    assert result.tool_calls == []
    assert result.finish_reason == "repeated_failure"
    assert result.n_llm_calls == 2

def test_env_validate_false_with_no_error_list_is_still_a_rejection():
    """The official validate() returns {"ok": False, "error": ...} and no `errors` key."""
    from lifemate.env.official import OfficialToolEnv

    class Sdk:
        def __init__(self): self.allowed = {"alarm.set"}

        def validate(self, name, args):
            return {"ok": False, "error": "thiếu tham số bắt buộc: time"}

        def get_tool_calls(self): return []

        def get_state_hash(self): return "hash"

    env = OfficialToolEnv.__new__(OfficialToolEnv)
    env.split, env.sample_id, env.data_root, env.catalog = "train", "s1", ".", None
    env._env = Sdk()
    outcome = env.validate("alarm.set", {})
    assert outcome.ok is False
    assert outcome.errors and "thiếu tham số" in outcome.errors[0], outcome.errors


def test_prompt_mentions_no_tool_of_an_uninstalled_app():
    """Even the format example must not name a tool the sample cannot call."""
    from lifemate.agent.prompt_builder import build_decision_prompt
    cat = catalog()
    prompt = build_decision_prompt(
        transcript="x", context=Ctx(apps=("messaging",)), history=[], installed_apps=["messaging"],
        catalog=ToolRegistry(cat, apps=["messaging"]), prior_steps=[],
    )
    assert "messaging.send" in prompt
    assert "alarm." not in prompt, "uninstalled app named in the prompt"

if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:  # pragma: no cover - inline runner
                failed += 1
                print(f"FAIL {name}: {exc}")
    print("failed:", failed, "| NEEDS_GPU:", NEEDS_GPU or "none")

# --------------------------------------------------------------------------- #
# history rendering fidelity (D-1 / D-2)
# --------------------------------------------------------------------------- #
class RawTurn:
    """A history turn shaped like the dataset's: parsed fields + untouched raw."""

    def __init__(self, role, content="", tool_call=None, tool_result=None, raw=None):
        self.role = role
        self.content = content
        self.tool_call = tool_call
        self.tool_result = tool_result
        self.raw = dict(raw or {})

def test_history_user_turn_with_audio_text_is_rendered():
    """A user turn carrying only `audio_text` must not render blank."""
    turn = RawTurn("user", raw={"role": "user", "audio": "audio/x/x_u1.wav", "audio_text": "Mai tôi có báo thức nào?"})
    prompt = build_decision_prompt(
        transcript="dời báo thức", context=Ctx(), history=[turn],
        installed_apps=["alarm"], catalog=catalog(), prior_steps=[],
    )
    assert "Người dùng: Mai tôi có báo thức nào?" in prompt
    assert "Người dùng: \n" not in prompt, "blank user turn"

def test_history_turn_with_only_audio_renders_nothing_extra():
    """public_test history has audio but no text: render the label alone, never a fake."""
    turn = RawTurn("user", raw={"role": "user", "audio": "audio/y/y_u1.wav"})
    prompt = build_decision_prompt(
        transcript="dời báo thức", context=Ctx(), history=[turn],
        installed_apps=["alarm"], catalog=catalog(), prior_steps=[],
    )
    section = prompt.split("# Lịch sử hội thoại")[1].split("# Công cụ khả dụng")[0]
    assert section.strip() == "Người dùng:"
    assert "u1.wav" not in section and "audio" not in section

def test_history_tool_result_payload_is_rendered_from_the_raw_record():
    """`{name, ok, result: null}` wrappers must not erase the real payload."""
    raw_results = [{"ok": True, "alarms": [{"id": "alm_01", "time": "2026-11-14T06:15:00+07:00"}]}]
    turn = RawTurn(
        "assistant", "Mai có 2 báo thức.",
        tool_call={"tool_calls": [{"name": "alarm.list", "arguments": {}}]},
        tool_result={"tool_results": [{"name": None, "ok": True, "result": None}]},
        raw={"role": "assistant", "text": "Mai có 2 báo thức.",
             "tool_calls": [{"name": "alarm.list", "arguments": {}}], "tool_results": raw_results},
    )
    prompt = build_decision_prompt(
        transcript="x", context=Ctx(), history=[turn],
        installed_apps=["alarm"], catalog=catalog(), prior_steps=[],
    )
    assert "alm_01" in prompt, "payload lost in the rendered history"
    assert '"name": null' not in prompt, "wrapper shape leaked into the prompt"
    assert '<action>{"type": "tool_call", "name": "alarm.list", "arguments": {}}</action>' in prompt

def test_history_tool_calls_and_results_render_one_pair_per_call():
    turn = RawTurn(
        "assistant", "đã kiểm tra",
        tool_call={"tool_calls": [{"name": "alarm.list", "arguments": {}},
                                  {"name": "weather.get_forecast", "arguments": {"date": "2026-11-21"}}]},
        tool_result={"tool_results": [{"ok": True, "alarms": []}, {"ok": True, "forecast": []}]},
        raw={},
    )
    prompt = build_decision_prompt(
        transcript="x", context=Ctx(), history=[turn],
        installed_apps=["alarm"], catalog=catalog(), prior_steps=[],
    )
    section = prompt.split("# Lịch sử hội thoại")[1].split("# Công cụ khả dụng")[0]
    assert section.count("  gọi: ") == 2 and section.count("  kết quả: ") == 2
    assert '"alarms"' in section and '"forecast"' in section

def test_call_log_entry_without_result_never_fabricates_ok():
    """The official call log stores only {name, arguments}: no invented `ok: true`."""
    steps = StepRecord.from_call_log([{"name": "alarm.list", "arguments": {}}])
    assert steps[0].observation == {"name": "alarm.list", "arguments": {}}
    assert "ok" not in steps[0].observation
    # entries that DO carry an outcome keep it
    ok_step = StepRecord.from_call_log([{"name": "alarm.list", "arguments": {}, "ok": True, "result": {"x": 1}}])
    assert ok_step[0].observation == {"ok": True, "x": 1}
