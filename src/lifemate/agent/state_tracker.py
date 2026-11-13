"""Per-run bookkeeping of agent steps. Plain data, no hidden state access."""

from __future__ import annotations

import json
from typing import Any, Sequence

from lifemate.agent.prompt_builder import StepRecord


class StateTracker:
    """Records every step of a run and renders it for logs/diagnostics.

    Observations are stored exactly as the environment returned them; the
    environment call log is only passed through when explicitly collected.
    """

    def __init__(self) -> None:
        self._steps: list[StepRecord] = []
        self._elapsed: list[float] = []
        self._call_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ record
    def record(
        self,
        action: dict[str, Any] | None,
        observation: dict[str, Any] | None = None,
        elapsed: float = 0.0,
    ) -> StepRecord:
        """Append one step and return it."""
        step = StepRecord(index=len(self._steps), action=action, observation=observation)
        self._steps.append(step)
        self._elapsed.append(float(elapsed))
        return step

    def collect_call_log(self, call_log: Sequence[dict[str, Any]] | None) -> None:
        """Passthrough of ``env.get_tool_calls()`` (never reconstructed here)."""
        self._call_log = list(call_log or [])

    # ------------------------------------------------------------------- query
    @property
    def steps(self) -> list[StepRecord]:
        """Steps in execution order (copy; the tracker stays authoritative)."""
        return list(self._steps)

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Environment call log passthrough (empty until ``collect_call_log``)."""
        return list(self._call_log)

    @property
    def elapsed(self) -> list[float]:
        """Per-step elapsed seconds (same indexing as ``steps``)."""
        return list(self._elapsed)

    # --------------------------------------------------------------- rendering
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dump of the whole run (steps with elapsed + call log)."""
        return {
            "steps": [
                {**step.to_dict(), "elapsed": self._elapsed[i]}
                for i, step in enumerate(self._steps)
            ],
            "call_log": list(self._call_log),
        }

    def summary(self) -> dict[str, Any]:
        """Compact, log-friendly view of the run."""
        failed = sum(
            1 for s in self._steps if s.observation is not None and s.observation.get("ok") is False
        )
        return {
            "n_steps": len(self._steps),
            "n_failed_steps": failed,
            "total_elapsed": round(sum(self._elapsed), 3),
            "actions": [
                (s.action or {}).get("name") or (s.action or {}).get("type") for s in self._steps
            ],
        }

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StateTracker({json.dumps(self.summary(), ensure_ascii=False)})"


__all__ = ["StateTracker"]
