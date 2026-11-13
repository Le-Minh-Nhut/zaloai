"""Tolerant record schemas for the LifeMate dataset.

Nothing here validates against an official spec: the real dataset is not
present yet, so every parser accepts several plausible shapes and keeps
unknown fields in ``raw``. Strictness lives in
:mod:`lifemate.data.validator`, which reports problems instead of raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import SchemaError

__all__ = [
    "Context",
    "HistoryTurn",
    "ToolCall",
    "ToolResult",
    "Label",
    "Sample",
    "parse_context",
    "parse_history",
    "parse_tool_calls",
    "parse_tool_results",
]

# Keys that may carry the model-visible transcript.
HISTORY_KEYS = ("history", "conversation", "messages", "turns", "dialog")
# Keys that may carry gold labels.
LABEL_KEYS = ("label", "labels", "gold", "target", "annotation", "output")
# Keys that may carry the audio reference.
AUDIO_KEYS = ("audio_path", "audio", "audio_file", "wav", "wav_path", "audio_filename", "path")
# Keys that may carry the ground-truth transcript.
TEXT_KEYS = ("audio_text", "transcript", "text", "asr_text", "utterance", "sentence")

_MISSING = object()


def _first(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _as_dict(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(item) for item in value)
    return str(value)


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    """Per-sample world state visible to the model (no labels inside)."""

    current_time: str | None = None
    timezone: str | None = None
    current_location: str | None = None
    user_profile: dict = field(default_factory=dict)
    installed_apps: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any) -> "Context":
        """Build a Context from a loosely shaped mapping."""
        data = _as_dict(raw)
        apps = _first(data, ("installed_apps", "apps", "available_apps", "app_list"), [])
        if isinstance(apps, str):
            apps = [part.strip() for part in apps.split(",") if part.strip()]
        elif isinstance(apps, Mapping):
            apps = [str(key) for key in apps]
        elif not isinstance(apps, (list, tuple)):
            apps = []
        profile = _first(data, ("user_profile", "profile", "user", "user_info"), {})
        if not isinstance(profile, Mapping):
            profile = {"value": profile} if profile is not None else {}
        return cls(
            current_time=_first(data, ("current_time", "now", "timestamp", "time")),
            timezone=_first(data, ("timezone", "tz", "time_zone")),
            current_location=_first(data, ("current_location", "location", "place", "city")),
            user_profile=dict(profile),
            installed_apps=[str(app) for app in apps],
        )

    def to_dict(self) -> dict:
        """Plain dict, model-visible only."""
        return {
            "current_time": self.current_time,
            "timezone": self.timezone,
            "current_location": self.current_location,
            "user_profile": dict(self.user_profile),
            "installed_apps": list(self.installed_apps),
        }


# --------------------------------------------------------------------------- #
# Tool call / result
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One tool invocation request."""

    name: str = ""
    arguments: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolCall":
        """Accept ``{"name", "arguments"}``, OpenAI-style, or ``{"tool": ...}``."""
        data = _as_dict(raw)
        if not data and isinstance(raw, str):
            return cls(name=raw.strip(), arguments={})
        if "function" in data and isinstance(data["function"], Mapping):
            fn = dict(data["function"])
            return cls(
                name=str(fn.get("name", "")),
                arguments=_as_dict(fn.get("arguments", {})),
            )
        name = _first(data, ("name", "tool", "tool_name", "action", "api"), "")
        args = _first(data, ("arguments", "args", "parameters", "params", "input"), {})
        return cls(name=str(name), arguments=_as_dict(args))

    def to_dict(self) -> dict:
        """Canonical form used by logs, prompts and scoring."""
        return {"name": self.name, "arguments": dict(self.arguments)}


#: Record keys that are NOT part of the payload ``env.call()`` returns.
#: The official env returns ``{"ok": True, **payload}`` (payload inline, never
#: nested under a ``result`` key); ``name``/``tool`` are our own bookkeeping.
#: Deliberately minimal: ``exchange_rate.convert`` really does return a payload
#: key called ``result``, so it must not be filtered out.
_RESULT_CONTROL_KEYS = frozenset(
    {"ok", "success", "error", "error_message", "message_if_error", "name", "tool", "tool_name", "api"}
)

@dataclass
class ToolResult:
    """One environment response to a :class:`ToolCall`.

    ``payload`` is everything the env returned besides the control keys — the
    shape ``env.call()`` actually produces (``{"ok": True, "events": [...]}``).
    ``result`` is the legacy single-value view (explicit ``result``-ish key, else
    the payload) kept for callers that only want one value.
    """

    name: str | None = None
    ok: bool | None = None
    result: Any = None
    error: str | None = None
    payload: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolResult":
        """Accept ``{"ok", <payload...>}``, ``{"ok", "result"}``, ``{"error"}``, or a bare value."""
        if not isinstance(raw, Mapping):
            return cls(result=raw, ok=True, payload={} if raw is None else {"result": raw})
        data = dict(raw)
        ok: bool | None = None
        if "ok" in data and isinstance(data["ok"], bool):
            ok = data["ok"]
        elif "success" in data and isinstance(data["success"], bool):
            ok = data["success"]
        error = _first(data, ("error", "error_message", "message_if_error"))
        if error is not None and not isinstance(error, str):
            error = str(error)
        if ok is None:
            ok = error is None
        payload = {key: value for key, value in data.items() if key not in _RESULT_CONTROL_KEYS}
        result = _first(data, ("result", "output", "data", "response", "value"))
        if result is None and payload:
            result = payload
        name = _first(data, ("name", "tool", "tool_name", "api"))
        return cls(
            name=None if name is None else str(name), ok=ok, result=result, error=error, payload=payload
        )

    def to_dict(self) -> dict:
        """Plain dict in the env's shape: ``{"ok", **payload}`` or ``{"ok", "error"}``."""
        out: dict[str, Any] = {"ok": self.ok}
        if self.name is not None:
            out["name"] = self.name
        if self.error is not None:
            out["error"] = self.error
        elif self.payload:
            out.update(self.payload)
        else:
            out["result"] = self.result
        return out


def _collect_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return [value]


def parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Parse any supported tool-call container into a list."""
    calls: list[ToolCall] = []
    for item in _collect_list(raw):
        call = ToolCall.from_raw(item)
        if call.name:
            calls.append(call)
    return calls


def parse_tool_results(raw: Any) -> list[ToolResult]:
    """Parse any supported tool-result container into a list."""
    return [ToolResult.from_raw(item) for item in _collect_list(raw)]


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
@dataclass
class HistoryTurn:
    """One conversation turn. ``raw`` keeps the original record untouched."""

    role: str = "user"
    content: str = ""
    tool_call: dict | None = None
    tool_result: dict | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "HistoryTurn":
        """Parse one turn; unknown extra keys survive in ``raw``."""
        if isinstance(raw, str):
            return cls(role="user", content=raw, raw={"content": raw})
        data = _as_dict(raw)
        role = str(_first(data, ("role", "speaker", "from", "author"), "user")).lower()
        # ``audio_text`` is the real dataset's key for a user turn's words;
        # ``text`` is the assistant's. Both must be readable as ``content``.
        content = _as_text(_first(data, ("content", "utterance", "audio_text", "text", "message"), ""))
        tool_call_raw = _first(data, ("tool_call", "tool_calls", "call"), _MISSING)
        tool_result_raw = _first(data, ("tool_result", "tool_results", "result", "observation"), _MISSING)
        tool_call = None
        if tool_call_raw is not _MISSING:
            if isinstance(tool_call_raw, (list, tuple)):
                tool_call = {"tool_calls": [ToolCall.from_raw(item).to_dict() for item in tool_call_raw]}
            else:
                tool_call = ToolCall.from_raw(tool_call_raw).to_dict()
        tool_result = None
        if tool_result_raw is not _MISSING:
            if isinstance(tool_result_raw, (list, tuple)):
                tool_result = {"tool_results": [ToolResult.from_raw(item).to_dict() for item in tool_result_raw]}
            else:
                tool_result = ToolResult.from_raw(tool_result_raw).to_dict()
        if not content and data.get("name") and data.get("arguments") is not None:
            # Bare {name, arguments} assistant turn.
            tool_call = ToolCall.from_raw(data).to_dict()
        return cls(role=role, content=content, tool_call=tool_call, tool_result=tool_result, raw=data)

    def to_dict(self) -> dict:
        """Model-visible form of the turn."""
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call:
            out["tool_call"] = dict(self.tool_call)
        if self.tool_result:
            out["tool_result"] = dict(self.tool_result)
        return out


def parse_history(raw: Any) -> list[HistoryTurn]:
    """Parse the visible conversation from any supported container."""
    turns: list[HistoryTurn] = []
    for item in _collect_list(raw):
        if isinstance(item, Mapping) and isinstance(item.get("history"), (list, tuple)):
            turns.extend(HistoryTurn.from_raw(turn) for turn in item["history"])
        else:
            turns.append(HistoryTurn.from_raw(item))
    return turns


# --------------------------------------------------------------------------- #
# Label
# --------------------------------------------------------------------------- #
@dataclass
class Label:
    """Gold annotation.

    ``final_state`` is parsed for scoring convenience only and is explicitly
    marked ``is_model_input = False``: it must never be shown to the model.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    final_state: dict | None = None
    is_model_input: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "Label":
        """Parse a label block tolerantly."""
        data = _as_dict(raw)
        calls_raw = _first(data, ("tool_calls", "calls", "actions", "tool_call"), [])
        results_raw = _first(data, ("tool_results", "results", "observations"), [])
        final_state = _first(data, ("final_state", "state", "end_state", "final_app_state"))
        return cls(
            tool_calls=parse_tool_calls(calls_raw),
            tool_results=parse_tool_results(results_raw),
            final_state=dict(final_state) if isinstance(final_state, Mapping) else None,
            is_model_input=False,
            raw=data,
        )

    def to_dict(self) -> dict:
        """Plain dict, labels included (only for scoring/reporting)."""
        return {
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [res.to_dict() for res in self.tool_results],
            "final_state": self.final_state,
            "is_model_input": self.is_model_input,
        }


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One benchmark item: audio + context + history (+ gold label)."""

    id: str
    split: str
    audio_path: Path | None
    audio_text: str
    context: Context
    history: list[HistoryTurn] = field(default_factory=list)
    label: Label | None = None
    raw: dict = field(default_factory=dict)

    @property
    def installed_apps(self) -> list[str]:
        """Apps available in this sample's context."""
        return list(self.context.installed_apps)

    @property
    def has_label(self) -> bool:
        """True when a gold label block was provided."""
        return self.label is not None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], split: str, data_root: str | Path | None = None) -> "Sample":
        """Build a Sample from one JSONL record.

        ``data_root`` is the *split directory* (``.../train``). Audio
        references are resolved against it when they are bare filenames.
        """
        if not isinstance(raw, Mapping):
            raise SchemaError(f"sample record must be a mapping, got {type(raw).__name__}")
        data = dict(raw)
        sample_id = _first(data, ("id", "sample_id", "uid", "utt_id", "name"), "")
        if sample_id in ("", None):
            raise SchemaError("sample record has no id field")
        context_raw = _first(data, ("context", "world", "state", "metadata"), {})
        context = Context.from_raw(context_raw)
        # Apps sometimes live at top level rather than inside context.
        if not context.installed_apps:
            top_apps = _first(data, ("installed_apps", "apps"), [])
            if top_apps:
                context = Context.from_raw({**context.to_dict(), "installed_apps": top_apps})
        history_raw = _first(data, HISTORY_KEYS, [])
        label_raw = _first(data, LABEL_KEYS, _MISSING)
        if label_raw is _MISSING and ("tool_calls" in data or "tool_results" in data):
            label_raw = {k: data[k] for k in ("tool_calls", "tool_results", "final_state") if k in data}
        label = None if label_raw is _MISSING or label_raw is None else Label.from_raw(label_raw)
        audio_ref = _first(data, AUDIO_KEYS)
        audio_path = _resolve_audio(audio_ref, Path(data_root) if data_root else None, sample_id)
        return cls(
            id=str(sample_id),
            split=split,
            audio_path=audio_path,
            audio_text=_as_text(_first(data, TEXT_KEYS, "")),
            context=context,
            history=parse_history(history_raw),
            label=label,
            raw=data,
        )

    def to_model_input(self) -> dict:
        """Everything the model may see: no label, state, or tool calls."""
        return {
            "id": self.id,
            "split": self.split,
            "audio_path": None if self.audio_path is None else str(self.audio_path),
            "audio_text": self.audio_text,
            "context": self.context.to_dict(),
            "installed_apps": self.installed_apps,
            "history": [turn.to_dict() for turn in self.history],
        }


def _resolve_audio(ref: Any, split_dir: Path | None, sample_id: str) -> Path | None:
    """Resolve an audio reference against conventional dataset layouts."""
    if ref is None:
        return None
    text = str(ref)
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if split_dir is None:
        return candidate
    for pattern in (
        split_dir / candidate,
        split_dir / "audio" / candidate,
    ):
        if pattern.is_file():
            return pattern
    conventional = split_dir / "audio" / sample_id / f"{sample_id}.wav"
    if conventional.is_file():
        return conventional
    # Report the most plausible path even when missing (validator flags it).
    return split_dir / "audio" / candidate


def parse_context(raw: Any) -> Context:
    """Module-level alias for :meth:`Context.from_raw`."""
    return Context.from_raw(raw)
