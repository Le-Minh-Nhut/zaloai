"""Deterministic prompt construction for the LifeMate agent.

This module is a shared text contract: training-data builders and inference MUST
produce identical prompt bytes for identical inputs, otherwise the fine-tuned
model sees a different distribution at test time.

Importable without torch/transformers (stdlib only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from lifemate.agent.protocol import ACTION_CLOSE, ACTION_OPEN, FINISH_ACTION, render_action

#: Default number of conversation turns kept in the prompt.
DEFAULT_HISTORY_LIMIT = 10

#: Request header used when the prompt lists several ASR hypotheses.
#: Same bytes as ``lifemate.data.augmentation.NBEST_HEADER`` (training format).
NBEST_HEADER = "# Các giả thuyết ASR"

SYSTEM_PROMPT = """\
You are a Vietnamese voice assistant that fulfils a user's spoken request by
calling the available tools one step at a time.

Rules:
1. Emit EXACTLY ONE action block per turn and nothing else:
   <action>{"type": "tool_call", "name": "<tool.name>", "arguments": {...}}</action>
   or the stop marker:
   <action>{"type": "finish"}</action>
2. Only the tools listed under "Công cụ khả dụng" may be used. Never invent a
   tool name, an application, an ID or a value.
3. Use read tools first to discover IDs/values (contacts, alarms, notes, files,
   menu items ...) before calling write tools that consume them.
4. When an argument required by the next tool cannot be obtained from the user
   request, the context, the conversation history or a previous tool result,
   call sys.ask_missing_info with a short question written in Vietnamese.
5. If no installed application can serve the request, emit
   <action>{"type": "finish"}</action> without calling any tool.
6. When the user's goal is achieved, emit <action>{"type": "finish"}</action>.
7. Tool results may contain errors. Read them, recover with a different tool or
   different arguments, and never repeat an identical failing call. If the goal
   cannot be reached, finish.
8. Output JSON only inside the action block: ASCII keys, double quotes, no
   comments, no markdown fences, no text before or after the block.
"""

_RESPONSE_FORMAT_SPEC = f"""\
Mỗi lượt chỉ xuất đúng MỘT khối hành động, không kèm văn bản khác:
{ACTION_OPEN}{{"type": "tool_call", "name": "<tên_công_cụ>", "arguments": {{...}}}}{ACTION_CLOSE}
Khi đã hoàn thành mục tiêu:
{ACTION_OPEN}{{"type": "finish"}}{ACTION_CLOSE}
Ví dụ (tên công cụ luôn lấy từ danh sách "Công cụ khả dụng" ở trên):
{ACTION_OPEN}{{"type": "tool_call", "name": "<tên_công_cụ>", "arguments": {{}}}}{ACTION_CLOSE}
"""


@dataclass
class StepRecord:
    """One executed (or attempted) step of an agent trajectory."""

    index: int
    action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None

    @classmethod
    def from_call_log(cls, call_log: Sequence[dict[str, Any]]) -> list["StepRecord"]:
        """Convert env call-log entries into step records.

        Entries look like ``{name, arguments, ok, result|error}``; extra keys
        (``step``, ``skipped`` ...) are ignored, and the observation is
        normalised to ``{"ok": bool, ...}``.
        """
        steps: list[StepRecord] = []
        for index, entry in enumerate(call_log or []):
            if not isinstance(entry, dict):
                continue
            action, observation = _call_log_entry_to_step(entry)
            steps.append(StepRecord(index=index, action=action, observation=observation))
        return steps

    def to_dict(self) -> dict[str, Any]:
        """Plain dict form (action/observation stay verbatim, no copying)."""
        return {"index": self.index, "action": self.action, "observation": self.observation}


def _call_log_entry_to_step(entry: dict[str, Any]) -> tuple[dict, dict]:
    """Split one env call-log entry into ``(action, observation)``.

    An entry with ``result``/``error`` (mock env, tests) yields that observation.
    The OFFICIAL ``get_tool_calls()`` only stores ``{"name", "arguments"}``: the
    submission format has no result field, so its payload cannot be recovered
    here. Such an entry must NOT be turned into ``{"ok": True}`` — a fabricated
    success is worse than a visibly missing observation. Use the agent loop's
    ``StateTracker`` (it keeps the live return value) when payload matters.
    """
    name = entry.get("name") or entry.get("tool") or ""
    arguments = entry.get("arguments")
    if arguments is None:
        arguments = entry.get("args") or {}
    action = {"type": "tool_call", "name": name, "arguments": dict(arguments or {})}
    if "result" in entry and "error" not in entry:
        result = entry.get("result")
        observation = result if isinstance(result, dict) else {"ok": True, "result": result}
    elif "error" in entry:
        observation = {"ok": False, "error": entry.get("error")}
    else:
        observation = dict(entry)
    if "ok" in entry:
        observation.setdefault("ok", bool(entry["ok"]))
    if "skipped" not in observation and entry.get("skipped"):
        observation["skipped"] = entry["skipped"]
    return action, observation


# --------------------------------------------------------------------- helpers
def _json(value: Any) -> str:
    """Stable JSON rendering (sorted keys); falls back on unsortable mappings."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:  # e.g. mixed-type mapping keys in a free-form profile
        return json.dumps(value, ensure_ascii=False, default=str)


def _turn_fields(turn: Any) -> dict[str, Any]:
    """Attribute access that also accepts plain dicts (HistoryTurn or raw)."""
    if isinstance(turn, dict):
        return turn
    return {
        "role": getattr(turn, "role", None),
        "content": getattr(turn, "content", None),
        "tool_call": getattr(turn, "tool_call", None),
        "tool_result": getattr(turn, "tool_result", None),
        "raw": getattr(turn, "raw", None),
    }


def compact_history(history: Sequence[Any] | None, limit: int = DEFAULT_HISTORY_LIMIT) -> list[Any]:
    """Keep only the last ``limit`` turns, oldest→newest (input order preserved)."""
    turns = list(history or [])
    if limit is None or limit <= 0:
        return []
    return turns[-limit:]


def estimate_prompt_tokens(text: str, tokenizer: Any = None) -> int | None:
    """Best-effort token count for context budgeting.

    Uses the real tokenizer when one is supplied, else the usual ~4 chars/token
    heuristic. Never raises: budgeting must not break a run.
    """
    if not isinstance(text, str) or not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # pragma: no cover - tokenizer quirks fall back to heuristic
            return len(text) // 4
    return len(text) // 4


_TURN_TEXT_KEYS = ("content", "audio_text", "transcript", "text", "asr_text", "utterance")

def _turn_text(fields: dict[str, Any]) -> str:
    """Visible text of a turn: `content`, else the raw utterance fields.

    A history user turn carries `{"audio": ..., "audio_text": ...}` (train) or
    only `{"audio": ...}` (public_test). The text is rendered from whatever the
    record holds — never invented — so a train prompt and an inference prompt
    built from the same record are byte-identical.
    """
    raw = fields.get("raw") if isinstance(fields.get("raw"), dict) else {}
    for source in (fields, raw):
        for key in _TURN_TEXT_KEYS:
            value = source.get(key)
            if value not in (None, ""):
                return " ".join(str(value).split())
    return ""

def _tool_calls_of(tool_call: Any) -> list[dict]:
    """Normalise a turn's tool-call field to a list of ``{name, arguments}``."""
    if not tool_call:
        return []
    if isinstance(tool_call, dict) and isinstance(tool_call.get("tool_calls"), list):
        items = tool_call["tool_calls"]
    elif isinstance(tool_call, (list, tuple)):
        items = list(tool_call)
    else:
        items = [tool_call]
    calls: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool") or item.get("tool_name")
        if name:
            calls.append({"name": str(name), "arguments": dict(item.get("arguments") or item.get("args") or {})})
        elif item.get("type"):
            calls.append(dict(item))  # already a rendered action dict
    return calls

def _tool_results_of(tool_result: Any) -> list[Any]:
    """Normalise a turn's tool-result field to one entry per executed call."""
    if tool_result is None:
        return []
    if isinstance(tool_result, dict) and isinstance(tool_result.get("tool_results"), list):
        return list(tool_result["tool_results"])
    if isinstance(tool_result, (list, tuple)):
        return list(tool_result)
    return [tool_result]

def _render_history(history: Sequence[Any] | None, limit: int) -> str:
    """Conversation section, oldest→newest, last ``limit`` turns only.

    Tool calls and their results are rendered exactly like the agent's own prior
    steps (one ``gọi:``/``kết quả:`` pair per call), so a history turn and a live
    step are the same text for the same data.
    """
    turns = compact_history(history, limit)
    if not turns:
        return "(không có)"
    labels = {"user": "Người dùng", "assistant": "Trợ lý", "system": "Hệ thống", "tool": "Công cụ"}
    lines: list[str] = []
    for turn in turns:
        fields = _turn_fields(turn)
        role = str(fields.get("role") or "user")
        label = labels.get(role, role)
        lines.append(f"{label}: {_turn_text(fields)}")
        calls = _tool_calls_of(fields.get("tool_call")) or _tool_calls_of(fields.get("tool_calls"))
        results = _tool_results_of(fields.get("tool_result"))
        if not results:
            results = _tool_results_of(fields.get("tool_results"))
        if not _has_payload(results):
            # A parsed wrapper can carry only {name, ok, result: None}. The
            # untouched record holds the env's real return, so prefer it then.
            raw = fields.get("raw") if isinstance(fields.get("raw"), dict) else {}
            results = _tool_results_of(raw.get("tool_results")) or results
        for index, call in enumerate(calls):
            lines.append(f"  gọi: {_render_action_safe({'type': 'tool_call', **call})}")
            if index < len(results):
                lines.append(f"  kết quả: {_json(_observation_of(results[index]))}")
        for extra in results[len(calls):]:  # results without a matching call entry
            lines.append(f"  kết quả: {_json(_observation_of(extra))}")
    return "\n".join(lines)

def _has_payload(results: list[Any]) -> bool:
    """True when at least one parsed result still carries something beyond ok/name."""
    for item in results:
        if not isinstance(item, dict):
            return True
        if set(item) - {"name", "ok", "result", "error"}:
            return True
        if item.get("result") is not None or item.get("error") is not None:
            return True
    return False


def _observation_of(result: Any) -> Any:
    """The payload a tool returned, in the same shape ``env.call()`` uses.

    A raw history entry is the env's own return (``{"ok": true, "alarms": [...]}``);
    a parsed ``ToolResult.to_dict()`` is ``{"name", "ok", "result"|"error"}``.
    The wrapper keys are dropped so both render as the env's return value.
    """
    if not isinstance(result, dict):
        return result
    if set(result) <= {"name", "ok", "result", "error"}:
        if "error" in result:
            return {"ok": result.get("ok", False), "error": result["error"]}
        if result.get("result") is not None:
            payload = result["result"]
            if isinstance(payload, dict) and "ok" in payload:
                return payload
            return {"ok": result.get("ok", True), **({"result": payload} if not isinstance(payload, dict) else payload)}
        return {"ok": result.get("ok", True)}
    return result


def _render_action_safe(action: Any) -> str:
    """``render_action`` that never raises on odd legacy records."""
    try:
        return render_action(action)
    except (ValueError, TypeError):
        return ACTION_OPEN + _json(action) + ACTION_CLOSE


def _hypothesis_items(asr_hypotheses: Sequence[Any] | None) -> list[tuple[str, str]]:
    """``(text, source)`` pairs: input order, deduped, empty texts dropped."""
    from lifemate.asr.nbest import normalize_hypothesis_text  # stdlib-only module

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in asr_hypotheses or ():
        if isinstance(raw, Mapping):
            text, source = raw.get("text"), raw.get("source")
        elif isinstance(raw, str):
            text, source = raw, ""
        else:
            text, source = getattr(raw, "text", ""), getattr(raw, "source", "")
        text = " ".join(str(text or "").split())
        key = normalize_hypothesis_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        items.append((text, str(source or "")))
    return items

def _render_request(
    transcript: str,
    asr_hypotheses: Sequence[Any] | None = None,
    *,
    nbest_format: bool = False,
    limit: int = 3,
) -> str:
    """Request section: the numbered hypothesis list when it carries signal.

    A single surviving hypothesis (or an empty list) keeps the baseline section
    byte-for-byte: a one-item list would move the prompt distribution for no
    signal, and the training format makes the same choice.
    """
    items = _hypothesis_items(asr_hypotheses) if nbest_format else []
    if len(items) > 1:
        shown = items[: max(1, int(limit))]
        sources = {source for _text, source in shown if source}
        labelled = len(sources) > 1
        lines = [
            f"{index}. {source + ': ' if labelled and source else ''}{text}"
            for index, (text, source) in enumerate(shown, 1)
        ]
        return NBEST_HEADER + "\n\n" + "\n".join(lines)
    return "# Yêu cầu\n" + (transcript or "").strip()

def _render_context(context: Any, installed_apps: Iterable[str] | None = None) -> str:
    """Context section. ``user_profile`` keeps only keys that exist in the sample;
    values are the model-visible ones already present in the dataset context."""
    if context is None:
        return "(không có)"
    get = context.get if isinstance(context, dict) else lambda k, d=None: getattr(context, k, d)
    lines: list[str] = []
    for key, label in (
        ("current_time", "current_time"),
        ("timezone", "timezone"),
        ("current_location", "current_location"),
    ):
        value = get(key)
        lines.append(f"{label}: {value if value not in (None, '') else '(không có)'}")

    profile = get("user_profile") or {}
    if isinstance(profile, dict) and profile:
        lines.append("user_profile:")
        for key in sorted(profile, key=str):  # stable ordering
            lines.append(f"  {key}: {_json(profile[key])}")
    else:
        lines.append("user_profile: (không có)")

    apps = sorted(get("installed_apps") or installed_apps or [], key=str)
    apps_list = ", ".join(str(a) for a in apps) if apps else "(không có)"
    lines.append(f"installed_apps: {apps_list}")
    return "\n".join(lines)


def _required_args(parameters: dict[str, Any] | None) -> list[str]:
    """Sorted JSON-Schema ``required`` names."""
    if not isinstance(parameters, dict):
        return []
    required = parameters.get("required")
    return sorted(str(r) for r in required) if isinstance(required, list) else []


def _spec_fields(spec: Any) -> tuple[str, str, str, dict[str, Any], str]:
    """Normalise a ``ToolSpec`` or raw dict into ``(name, app, desc, params, kind)``."""
    if isinstance(spec, dict):
        return (
            str(spec.get("name") or ""),
            str(spec.get("app_name") or ""),
            str(spec.get("description") or ""),
            spec.get("parameters") or {},
            str(spec.get("kind") or "unknown"),
        )
    return (
        str(getattr(spec, "name", "") or ""),
        str(getattr(spec, "app_name", "") or ""),
        str(getattr(spec, "description", "") or ""),
        getattr(spec, "parameters", {}) or {},
        str(getattr(spec, "kind", "unknown") or "unknown"),
    )


def _catalog_specs(catalog: Any, installed_apps: Iterable[str] | None) -> list[Any]:
    """Specs to render: registry view if present, else the catalog narrowed to apps."""
    specs = list(getattr(catalog, "specs", []) or [])
    if specs:
        return specs
    apps = list(installed_apps) if installed_apps is not None else []
    tools_for_apps = getattr(catalog, "tools_for_apps", None)
    if callable(tools_for_apps) and installed_apps is not None:
        return list(tools_for_apps(apps, include_meta=True))
    return list((getattr(catalog, "tools", {}) or {}).values())


def _render_tools(catalog: Any, installed_apps: Iterable[str] | None = None) -> str:
    """Tools grouped by app: kind, description, required args, JSON schema."""
    specs = _catalog_specs(catalog, installed_apps)
    if not specs:
        return "(không có)"
    grouped: dict[str, list[Any]] = {}
    for spec in specs:
        name, app_name, _desc, _params, _kind = _spec_fields(spec)
        grouped.setdefault(app_name or "other", []).append(spec)
    lines: list[str] = []
    for app in sorted(grouped):
        lines.append(f"## {app}")
        for spec in sorted(grouped[app], key=lambda s: _spec_fields(s)[0]):
            name, _app, desc, params, kind = _spec_fields(spec)
            required = _required_args(params)
            lines.append(f"### {name} (kind={kind})")
            if desc:
                lines.append(" ".join(desc.split()))
            lines.append(f"required: {', '.join(required) if required else '(none)'}")
            lines.append(f"parameters: {_json(params)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_steps(prior_steps: Sequence[StepRecord]) -> str:
    """Executed-steps section: rendered action then its observation JSON."""
    lines: list[str] = []
    for step in prior_steps:
        action = step.action if step.action is not None else FINISH_ACTION
        lines.append(f"Bước {step.index + 1}:")
        lines.append(_render_action_safe(action))
        observation = step.observation if step.observation is not None else {}
        lines.append(f"Kết quả: {_json(observation)}")
    return "\n".join(lines)


def build_decision_prompt(
    *,
    transcript: str,
    context: Any,
    history: Sequence[Any] | None,
    installed_apps: Iterable[str] | None,
    catalog: Any,
    prior_steps: Sequence[StepRecord] | None = None,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    asr_hypotheses: Sequence[Any] | None = None,
    nbest_format: bool = False,
    nbest_limit: int = 3,
) -> str:
    """Render the full decision prompt for one agent step.

    Deterministic for identical inputs: fixed section order, sorted tool and
    profile keys, no timestamps of the moment of rendering. Contains no gold
    labels, no future observations and no tools of uninstalled apps (``catalog``
    is expected to be the per-sample catalog/registry view).

    ``asr_hypotheses`` (strings, ``AsrHypothesis`` or ``{"text", "source"}``
    mappings) is only rendered when ``nbest_format`` is on AND 2+ distinct texts
    survive deduplication; otherwise the section is the baseline
    ``"# Yêu cầu\\n" + transcript`` byte-for-byte.
    """
    steps = list(prior_steps or [])

    sections: list[str] = []
    sections.append(
        _render_request(
            transcript, asr_hypotheses, nbest_format=nbest_format, limit=nbest_limit
        )
    )
    # ``context`` already carries installed_apps; the registry view supplies the tools.
    sections.append("# Ngữ cảnh\n" + _render_context(context, installed_apps))
    sections.append("# Lịch sử hội thoại\n" + _render_history(history, history_limit))
    sections.append("# Công cụ khả dụng\n" + _render_tools(catalog, installed_apps))
    if steps:
        sections.append("# Các bước đã thực hiện\n" + _render_steps(steps))
    sections.append("# Định dạng trả lời\n" + _RESPONSE_FORMAT_SPEC)
    return "\n\n".join(sections).strip() + "\n"


def to_messages(body: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Wrap a rendered prompt body in chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": body},
    ]


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "NBEST_HEADER",
    "SYSTEM_PROMPT",
    "StepRecord",
    "build_decision_prompt",
    "compact_history",
    "estimate_prompt_tokens",
    "to_messages",
]
