"""Thin execution layer between the agent loop and the ToolEnvironment.

Contract: observations are returned exactly as the environment produced them;
this module never mutates environment state directly and never swallows a failed
call (it stays in the env call log, which evaluation reads verbatim).
"""

from __future__ import annotations

from typing import Any

from lifemate.agent.protocol import Action
from lifemate.errors import ProhibitedAccessError


class ToolExecutor:
    """Executes validated actions against a ``ToolEnvironment``."""

    def __init__(self, env: Any, registry: Any, max_consecutive_failures: int = 2) -> None:
        if env is None:
            raise ValueError("ToolExecutor requires a ToolEnvironment")
        self.env = env
        self.registry = registry
        self.max_consecutive_failures = int(max_consecutive_failures)
        self._failure_streak = 0

    # ------------------------------------------------------------------ state
    @property
    def failure_streak(self) -> int:
        """Number of consecutive failures (env ``ok: False`` or skipped)."""
        return self._failure_streak

    def reset(self) -> None:
        """Clear the failure streak (called at the start of each run)."""
        self._failure_streak = 0

    # ---------------------------------------------------------------- execute
    def execute(self, action: Action) -> dict[str, Any]:
        """Call the environment and return its observation dict untouched.

        Tool-level failures are data, not exceptions: they update the failure
        streak and are returned to the caller.
        """
        if not action.is_tool_call or not action.name:
            return {"ok": False, "error": "execute() requires a tool_call action with a name"}
        try:
            observation = self.env.call(action.name, dict(action.arguments or {}))
        except ProhibitedAccessError:
            # A compliance violation is never data: it must stop the run, not
            # become an observation the model could learn to work around.
            raise
        except Exception as exc:  # env contract violation, still not fatal
            observation = {
                "ok": False,
                "error": f"environment raised {type(exc).__name__}: {exc}",
            }
        if not isinstance(observation, dict):
            observation = {"ok": False, "error": f"environment returned {type(observation).__name__}, expected dict"}
        self._note_outcome(observation)
        return observation

    def execute_validated(self, action: Action) -> dict[str, Any]:
        """Validate against the registry first, then execute.

        A validation failure returns ``{"ok": False, "error": ..., "skipped": True}``
        and is NOT sent to the environment: skipped calls do not appear in the env
        call log, which is why evaluation must count them from the trajectory.
        """
        if not action.is_tool_call:
            return self.note_skipped("execute_validated() requires a tool_call action")
        name = action.name or ""
        if hasattr(self.registry, "is_available") and not self.registry.is_available(name):
            return self.note_skipped(f"tool {name!r} is not available for this sample")
        # The environment's validator is authoritative when present: its rule set
        # (empty values, x-one-of groups, minimum, format, items) is the one that
        # actually rejects calls, so it must be the one consulted here too.
        env_validate = getattr(self.env, "validate", None)
        if callable(env_validate):
            outcome = env_validate(name, dict(action.arguments or {}))
            if not getattr(outcome, "ok", True):
                errors = list(getattr(outcome, "errors", []) or []) or ["invalid arguments"]
                return self.note_skipped(f"invalid arguments for {name}: {'; '.join(str(e) for e in errors)}")
            return self.execute(action)
        outcome = self.registry.validate(name, dict(action.arguments or {}))
        errors = outcome.get("errors") if isinstance(outcome, dict) else outcome
        if errors:
            message = "; ".join(str(e) for e in errors) if isinstance(errors, list) else str(errors)
            return self.note_skipped(f"invalid arguments for {name}: {message}")
        return self.execute(action)

    # ---------------------------------------------------------------- helpers
    def note_skipped(self, message: str) -> dict[str, Any]:
        """Observation for an action that is NOT executed (counts as a failure)."""
        self._failure_streak += 1
        return {"ok": False, "error": message, "skipped": True}

    def _note_outcome(self, observation: dict[str, Any]) -> None:
        """Update the failure streak from an env observation."""
        if observation.get("ok", True):
            self._failure_streak = 0
        else:
            self._failure_streak += 1

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Environment call log, verbatim (empty list when unavailable)."""
        log = getattr(self.env, "get_tool_calls", None)
        return log() if callable(log) else []


__all__ = ["ToolExecutor"]
