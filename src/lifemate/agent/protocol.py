"""Frozen action protocol for the LifeMate agent.

An assistant decision is exactly one block::

    <action>{"type": "tool_call", "name": "alarm.list", "arguments": {}}</action>
    <action>{"type": "finish"}</action>

This module is the single source of truth for that format; training data
builders, inference and evaluation MUST import it instead of re-implementing
serialisation/parsing.

Pure stdlib: importable without torch/transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

ACTION_OPEN = "<action>"
ACTION_CLOSE = "</action>"
FINISH_ACTION: dict[str, Any] = {"type": "finish"}

ActionType = Literal["tool_call", "finish", "invalid"]


def make_tool_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a canonical ``tool_call`` action dict."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tool call name must be a non-empty string, got {name!r}")
    return {"type": "tool_call", "name": name, "arguments": dict(arguments or {})}


def render_action(action: dict[str, Any]) -> str:
    """Serialise one action dict into the frozen ``<action>...</action>`` block.

    Key order is fixed to ``type, name, arguments`` so the rendered bytes are
    stable for identical inputs (regression tests assert prompt determinism).
    """
    if not isinstance(action, dict):
        raise ValueError(f"action must be a dict, got {type(action).__name__}")
    kind = str(action.get("type") or "").strip()
    payload: dict[str, Any] = {"type": kind}
    if kind == "tool_call":
        payload["name"] = action.get("name")
        payload["arguments"] = action.get("arguments") or {}
    elif kind != "finish":
        raise ValueError(f"cannot render action of type {kind!r}; expected 'tool_call' or 'finish'")
    return ACTION_OPEN + json.dumps(payload, ensure_ascii=False) + ACTION_CLOSE


@dataclass
class Action:
    """A parsed assistant decision."""

    type: ActionType
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None
    error: str | None = None

    @property
    def is_finish(self) -> bool:
        """True when this action is the ``finish`` stop marker."""
        return self.type == "finish"

    @property
    def is_tool_call(self) -> bool:
        """True when this action is an executable ``tool_call``."""
        return self.type == "tool_call"

    def to_dict(self) -> dict[str, Any]:
        """Canonical dict form (``invalid`` keeps its error text)."""
        if self.is_tool_call:
            return make_tool_call(self.name or "", self.arguments)
        if self.is_finish:
            return dict(FINISH_ACTION)
        return {
            "type": "invalid",
            "name": self.name,
            "arguments": self.arguments,
            "error": self.error,
        }


def _strip_fences(payload: str) -> str:
    """Drop markdown code fences some models wrap around the JSON payload."""
    text = payload.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _invalid(raw: str | None, error: str) -> Action:
    """Build an ``invalid`` action carrying a model-feedable error message."""
    return Action(type="invalid", raw=raw, error=error)


def _parse_payload(payload: str, raw: str | None) -> Action:
    """Turn one JSON payload into an ``Action`` (or an invalid one)."""
    text = _strip_fences(payload)
    if not text:
        return _invalid(raw, "empty <action> block: expected a JSON object")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return _invalid(raw, f"action payload is not valid JSON ({exc.msg} at char {exc.pos})")
    if not isinstance(obj, dict):
        return _invalid(raw, f"action payload must be a JSON object, got {type(obj).__name__}")

    kind = obj.get("type")
    kind = str(kind).strip() if kind is not None else None

    if kind == "finish":
        return Action(type="finish", raw=raw)

    if kind == "tool_call" or (kind is None and "name" in obj):
        name = obj.get("name")
        arguments = obj.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            return _invalid(raw, 'tool_call requires a non-empty string "name"')
        if not isinstance(arguments, dict):
            return _invalid(
                raw,
                f'tool_call "arguments" must be a JSON object, got '
                f"{type(arguments).__name__}; emit e.g. "
                '<action>{"type": "tool_call", "name": "%s", "arguments": {}}</action>' % name,
            )
        # NOTE: a raw JSON object with a "name" key and no "type" is accepted as a
        # best-effort tool_call (legacy prompt formats); it is never executed
        # before registry validation.
        return Action(type="tool_call", name=name, arguments=arguments, raw=raw)

    if kind is None:
        return _invalid(
            raw, 'action object is missing "type"; expected "tool_call" or "finish"'
        )
    return _invalid(
        raw,
        f'unknown action type {kind!r}; expected "tool_call" or "finish" '
        "(finish is the stop marker and is never a tool name)",
    )


def parse_action(text: str) -> Action:
    """Parse the last complete ``<action>...</action>`` block out of ``text``.

    Tolerates surrounding prose, whitespace and markdown fences. Never raises:
    garbage input yields ``Action(type="invalid", error=...)`` with a message
    that can be fed back to the model verbatim.
    """
    raw = text if isinstance(text, str) else ""
    if not raw.strip():
        return _invalid(raw, "empty model output: expected exactly one <action>...</action> block")

    close_idx = raw.rfind(ACTION_CLOSE)
    if close_idx == -1:
        # Legacy tolerance: some prompts/models emit a bare JSON action object.
        # Accepted only with a "name" key (best-effort tool_call) or an explicit
        # "type" — registry validation still gates execution.
        bare = _strip_fences(raw)
        if bare.startswith("{") and bare.endswith("}"):
            try:
                obj = json.loads(bare)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and (obj.get("name") or obj.get("type")):
                return _parse_payload(bare, raw)
        return _invalid(
            raw,
            "no complete <action>...</action> block found; emit exactly one action block",
        )
    open_idx = raw.rfind(ACTION_OPEN, 0, close_idx)
    if open_idx == -1:
        return _invalid(raw, "found </action> without a matching <action> opening tag")
    return _parse_payload(raw[open_idx + len(ACTION_OPEN) : close_idx], raw)


__all__ = [
    "ACTION_CLOSE",
    "ACTION_OPEN",
    "FINISH_ACTION",
    "Action",
    "make_tool_call",
    "parse_action",
    "render_action",
]
