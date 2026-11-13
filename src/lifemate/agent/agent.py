"""Closed-loop tool-calling agent.

Duck-typed on purpose: ``generator`` only needs ``.generate(messages) -> str``
and ``env`` must satisfy ``lifemate.env.base.ToolEnvironment``. No torch import
at module load time, so this file is testable with stdlib only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from lifemate.agent.prompt_builder import (
    DEFAULT_HISTORY_LIMIT,
    StepRecord,
    build_decision_prompt,
    to_messages,
)
from lifemate.agent.state_tracker import StateTracker
from lifemate.agent.stopping_policy import (
    REASON_FINISH,
    REASON_INVALID_REPEAT,
    REASON_MAX_STEPS,
    REASON_REPEATED_FAILURE,
    REASON_TIME_BUDGET,
    StoppingPolicy,
    signature,
)
from lifemate.agent.tool_executor import ToolExecutor
from lifemate.agent.tool_parser import parse_model_output
from lifemate.agent.tool_registry import ToolRegistry

#: Stop-policy reasons that map directly onto an ``AgentResult.status``.
_STATUS_BY_REASON = {
    REASON_FINISH: "finished",
    REASON_MAX_STEPS: "max_steps",
    REASON_TIME_BUDGET: "time_budget",
    REASON_REPEATED_FAILURE: "repeated_failure",
    REASON_INVALID_REPEAT: "repeated_failure",
}

_INVALID_NOTE = "invalid action, not executed"


@dataclass
class AgentResult:
    """Outcome of one ``run()`` call."""

    status: str
    steps: list[StepRecord] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state_hash: str = ""
    transcript: str = ""
    n_llm_calls: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    finish_reason: str = ""
    sample_id: str = ""

    @property
    def ok(self) -> bool:
        """True only when the agent reached its own ``finish``."""
        return self.status == "finished"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dump for logs and evaluation rows."""
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "state_hash": self.state_hash,
            "n_steps": len(self.steps),
            "n_llm_calls": self.n_llm_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "errors": self.errors,
            "steps": [s.to_dict() for s in self.steps],
        }


def _installed_apps(context: Any, fields: dict[str, Any]) -> list[str]:
    """Installed apps for prompt/registry narrowing (context wins, then fields)."""
    if isinstance(context, dict):
        apps = context.get("installed_apps")
    else:
        apps = getattr(context, "installed_apps", None)
    return list(apps or fields.get("installed_apps") or [])


def _sample_fields(sample_or_fields: Any) -> dict[str, Any]:
    """Normalise a ``Sample`` (or a plain dict of fields) into a plain dict."""
    if isinstance(sample_or_fields, dict):
        fields = dict(sample_or_fields)
        if "transcript" not in fields:
            fields["transcript"] = fields.get("audio_text", "")
        return fields
    context = getattr(sample_or_fields, "context", None)
    apps = getattr(context, "installed_apps", None) if context is not None else None
    return {
        "id": getattr(sample_or_fields, "id", ""),
        "transcript": getattr(sample_or_fields, "audio_text", "") or "",
        "context": context,
        "history": list(getattr(sample_or_fields, "history", []) or []),
        "installed_apps": list(apps or []),
    }


class ToolAgent:
    """One-step-at-a-time tool-calling loop over a real ``ToolEnvironment``."""

    def __init__(
        self,
        generator: Any,
        registry: ToolRegistry,
        env: Any,
        prompt_builder: Callable[..., str] | None = None,
        policy: StoppingPolicy | None = None,
        config: Any = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        max_steps: int | None = None,
        time_budget_seconds: float | None = None,
        **extra: Any,
    ) -> None:
        if callable(generator) and not hasattr(generator, "generate"):
            from lifemate.models.generation import CallableGenerator
            generator = CallableGenerator(generator)
        if generator is None or not callable(getattr(generator, "generate", None)):
            raise ValueError("ToolAgent requires a generator with .generate(messages) -> str")
        if registry is None:
            raise ValueError("ToolAgent requires a ToolRegistry")
        if env is None:
            raise ValueError("ToolAgent requires a ToolEnvironment")
        self.generator = generator
        self.registry = registry
        self.env = env
        self.prompt_builder = prompt_builder or build_decision_prompt
        self.config = config
        self.history_limit = int(history_limit)
        self.policy = policy if policy is not None else StoppingPolicy.from_config(config)
        if max_steps is not None:
            self.policy.max_steps = int(max_steps)
        if time_budget_seconds is not None:
            self.policy.time_budget_seconds = float(time_budget_seconds)
        self.executor = ToolExecutor(env, registry, self.policy.max_consecutive_failures)
    # ------------------------------------------------------------------- run
    def run(
        self,
        sample_or_fields: Any,
        transcript: str | None = None,
        deadline: float | None = None,
    ) -> AgentResult:
        """Run the loop for one sample.

        ``transcript`` overrides the sample's ASR text; ``deadline`` is an
        absolute ``time.monotonic()`` value (injectable clock for tests via
        ``self.clock``).
        """
        fields = _sample_fields(sample_or_fields)
        text = transcript if transcript is not None else str(fields.get("transcript") or "")
        return self._run_loop(fields, text, deadline)

    def run_batch(
        self, samples: Iterable[Any], transcripts: Sequence[str] | None = None
    ) -> list[AgentResult]:
        """``run()`` over several samples, with optional per-sample transcripts."""
        items = list(samples)
        if transcripts is not None and len(transcripts) != len(items):
            raise ValueError(
                f"transcripts length {len(transcripts)} != samples length {len(items)}"
            )
        results: list[AgentResult] = []
        for index, sample in enumerate(items):
            text = transcripts[index] if transcripts is not None else None
            results.append(self.run(sample, transcript=text))
        return results

    # --------------------------------------------------------------- helpers
    #: Injectable monotonic clock (tests replace it to fake the budget).
    clock: Callable[[], float] = staticmethod(time.monotonic)

    def _run_loop(self, fields: dict[str, Any], transcript: str, deadline: float | None) -> AgentResult:
        """The single generate → parse → execute loop shared by all entry points."""
        registry = self._registry_for(fields)
        tracker = StateTracker()
        errors: list[str] = []
        started = self.clock()
        n_llm_calls = 0
        status = "max_steps"
        finish_reason = ""
        context = fields.get("context")
        history = fields.get("history") or []
        installed = _installed_apps(context, fields)

        policy = self.policy
        executor = self.executor
        executor.reset()
        step_index = 0
        repeat_count = 0
        last_sig = ""
        result = AgentResult(status="error", transcript=transcript, sample_id=str(fields.get("id") or ""))

        while True:
            elapsed = self.clock() - started
            if deadline is not None and self.clock() >= deadline:
                status, finish_reason = "time_budget", REASON_TIME_BUDGET
                break
            stop, reason = policy.should_stop(
                step_index,
                action=None,
                elapsed_seconds=elapsed,
                repeated_signature=repeat_count,
                consecutive_failures=executor.failure_streak,
            )
            if stop:
                status, finish_reason = _STATUS_BY_REASON.get(reason, reason), reason
                break

            prompt = self.prompt_builder(
                transcript=transcript,
                context=context,
                history=history,
                installed_apps=installed,
                catalog=registry,
                prior_steps=tracker.steps,
                history_limit=self.history_limit,
            )
            messages = to_messages(prompt)
            try:
                raw_output = self.generator.generate(messages)
                n_llm_calls += 1
            except Exception as exc:
                errors.append(f"generator failed at step {step_index}: {type(exc).__name__}: {exc}")
                status, finish_reason = "error", "generator_error"
                break

            action = parse_model_output(str(raw_output or ""), registry, self.env)
            step_sig = signature(action)
            repeat_count = repeat_count + 1 if step_sig == last_sig else 1
            last_sig = step_sig

            if action.is_finish:
                # ``finish`` is the frozen stop marker: the loop always ends here.
                # ``policy.stop_on_finish=False`` only affects callers of
                # StoppingPolicy.should_stop (teacher-forced trajectory rollouts).
                step_index += 1
                status, finish_reason = "finished", REASON_FINISH
                break

            if action.type == "invalid":
                # Invalid actions never reach env: synthesised error notice counted
                # toward the failure streak, so the loop cannot spin forever.
                observation = executor.note_skipped(action.error or "invalid action")
                observation["note"] = _INVALID_NOTE
                errors.append(action.error or "invalid action")
            else:
                observation = executor.execute(action)
                if not observation.get("ok", True) and "environment raised" in str(observation.get("error", "")):
                    errors.append(str(observation["error"]))
            tracker.record(action.to_dict(), observation, elapsed)
            step_index += 1

            stop, reason = policy.should_stop(
                step_index,
                action=action,
                elapsed_seconds=self.clock() - started,
                repeated_signature=repeat_count,
                consecutive_failures=executor.failure_streak,
            )
            if stop:
                status, finish_reason = _STATUS_BY_REASON.get(reason, reason), reason
                break

        call_log = self._read(self.env, "get_tool_calls", [])
        state_hash = self._read(self.env, "get_state_hash", "")
        tracker.collect_call_log(call_log)
        result.status = status
        result.finish_reason = finish_reason
        result.steps = tracker.steps
        result.tool_calls = call_log
        result.state_hash = state_hash
        result.n_llm_calls = n_llm_calls
        result.elapsed_seconds = self.clock() - started
        result.errors = errors
        return result

    def _registry_for(self, fields: dict[str, Any]) -> ToolRegistry:
        """Per-sample registry view (installed apps only).

        The registry handed to the constructor is often already scoped per
        sample; re-filtering happens only when the sample actually declares its
        apps, so a fields dict without context cannot silently narrow it away.
        """
        for_sample = getattr(self.registry, "for_sample", None)
        if for_sample is None:
            return self.registry
        has_context = fields.get("context") is not None
        if not has_context and "installed_apps" not in fields:
            return self.registry
        return for_sample(fields)

    @staticmethod
    def _read(obj: Any, name: str, default: Any) -> Any:
        """Call ``obj.<name>()`` when it exists (env accessors), else ``default``."""
        getter = getattr(obj, name, None)
        return getter() if callable(getter) else default


__all__ = ["AgentResult", "ToolAgent"]
