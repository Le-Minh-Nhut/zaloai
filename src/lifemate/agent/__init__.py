"""Agent package: frozen action protocol, prompt builder, loop, helpers."""

from __future__ import annotations

from lifemate.agent.agent import AgentResult, ToolAgent
from lifemate.agent.protocol import (
    ACTION_CLOSE,
    ACTION_OPEN,
    FINISH_ACTION,
    Action,
    make_tool_call,
    parse_action,
    render_action,
)

__all__ = [
    "ACTION_CLOSE",
    "ACTION_OPEN",
    "FINISH_ACTION",
    "Action",
    "AgentResult",
    "ToolAgent",
    "make_tool_call",
    "parse_action",
    "render_action",
]
