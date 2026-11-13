"""Step / loop / budget stopping rules for the agent loop.

Every stop is reported with a short reason string that ends up in
``AgentResult.finish_reason`` and in evaluation diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from lifemate.agent.protocol import Action

#: Canonical reason strings emitted by :meth:`StoppingPolicy.should_stop`.
REASON_FINISH = "finish"
REASON_MAX_STEPS = "max_steps"
REASON_TIME_BUDGET = "time_budget"
REASON_REPEATED_FAILURE = "repeated_failure"
REASON_INVALID_REPEAT = "invalid_repeat"


def signature(action: Any) -> str:
    """Stable hash of an action (type + name + arguments) for loop detection."""
    if isinstance(action, Action):
        payload: dict[str, Any] = {"type": action.type}
        if action.type == "tool_call":
            payload["name"] = action.name
            payload["arguments"] = action.arguments or {}
        elif action.type == "invalid":
            payload["error"] = action.error or ""
    elif isinstance(action, dict):
        payload = {
            "type": action.get("type"),
            "name": action.get("name"),
            "arguments": action.get("arguments") or {},
        }
    else:
        payload = {"repr": repr(action)}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class StoppingPolicy:
    """Decides whether the loop must stop before the next model call."""

    def __init__(
        self,
        max_steps: int = 8,
        time_budget_seconds: float = 120.0,
        stop_on_finish: bool = True,
        max_consecutive_failures: int = 2,
        max_repeated_identical_calls: int = 2,
    ) -> None:
        self.max_steps = int(max_steps)
        self.time_budget_seconds = float(time_budget_seconds)
        self.stop_on_finish = bool(stop_on_finish)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.max_repeated_identical_calls = int(max_repeated_identical_calls)

    def should_stop(
        self,
        step_index: int,
        action: Any = None,
        elapsed_seconds: float = 0.0,
        repeated_signature: int = 1,
        consecutive_failures: int = 0,
    ) -> tuple[bool, str]:
        """Return ``(stop, reason)`` for the state *after* ``action`` was handled.

        ``step_index`` is the number of completed steps, ``repeated_signature``
        the count of consecutive identical action signatures including the
        current one, ``consecutive_failures`` the current failure streak.
        """
        if action is not None:
            is_finish = action.is_finish if isinstance(action, Action) else (
                isinstance(action, dict) and action.get("type") == "finish"
            )
            if is_finish and self.stop_on_finish:
                return True, REASON_FINISH
        if step_index >= self.max_steps:
            return True, REASON_MAX_STEPS
        if elapsed_seconds >= self.time_budget_seconds:
            return True, REASON_TIME_BUDGET
        if consecutive_failures >= self.max_consecutive_failures and consecutive_failures > 0:
            return True, REASON_REPEATED_FAILURE
        if repeated_signature > self.max_repeated_identical_calls:
            return True, REASON_INVALID_REPEAT
        return False, ""

    @classmethod
    def from_config(cls, config: Any, *, max_consecutive_failures: int = 2) -> "StoppingPolicy":
        """Build from an ``AppConfig.agent`` section (missing fields use defaults)."""
        agent_cfg = getattr(config, "agent", None) if config is not None else None

        def get(key: str, default: Any) -> Any:
            """Read one agent-config field from a dataclass or a dict."""
            if agent_cfg is None:
                return default
            if isinstance(agent_cfg, dict):
                return agent_cfg.get(key, default)
            return getattr(agent_cfg, key, default)

        return cls(
            max_steps=get("max_steps", 8),
            time_budget_seconds=get("time_budget_seconds", 120.0),
            stop_on_finish=get("stop_on_finish", True),
            max_consecutive_failures=max_consecutive_failures,
        )


__all__ = [
    "REASON_FINISH",
    "REASON_INVALID_REPEAT",
    "REASON_MAX_STEPS",
    "REASON_REPEATED_FAILURE",
    "REASON_TIME_BUDGET",
    "StoppingPolicy",
    "signature",
]
