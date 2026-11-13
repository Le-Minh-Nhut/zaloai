"""Parse a model output string into an :class:`Action`, registry-checked.

Pure: no environment calls, no state mutation. An invalid action is a first-class
value the agent loop feeds back to the model.
"""

from __future__ import annotations

from typing import Any

from lifemate.agent.protocol import Action, parse_action


def _format_problems(problems: Any) -> str:
    """Join validation errors; empty string when there are none."""
    if isinstance(problems, dict):
        errors = problems.get("errors") or []
        return "; ".join(str(p) for p in errors)
    if isinstance(problems, (list, tuple)):
        return "; ".join(str(p) for p in problems)
    return str(problems) if problems else ""


def _outcome_errors(outcome: Any) -> str:
    """Message text of a validation outcome (``ValidationOutcome`` or dict)."""
    if hasattr(outcome, "ok") and hasattr(outcome, "errors"):
        return "" if outcome.ok else _format_problems(list(outcome.errors)) or "invalid arguments"
    return _format_problems(outcome)


def parse_model_output(text: str, registry: Any, env: Any = None) -> Action:
    """Protocol-parse ``text`` then validate against ``registry``.

    When ``env`` exposes ``validate`` it is AUTHORITATIVE and used instead of the
    local catalog validator: the official simulator's rule set (empty values,
    ``x-one-of`` groups, ``minimum``, ``format``, ``items``) is the one that
    rejects calls and therefore the one the agent must not disagree with. The
    registry still decides which tools may be called at all.

    Returns ``Action(type="invalid", error=...)`` when the tool is unknown or not
    available for this sample, or when the arguments fail validation.
    """
    action = parse_action(text)
    if not action.is_tool_call:
        return action

    name = action.name or ""
    available = registry.tool_names() if hasattr(registry, "tool_names") else []
    if not registry.is_available(name):
        listing = ", ".join(available) if available else "(none)"
        return Action(
            type="invalid",
            name=name,
            arguments=action.arguments,
            raw=action.raw,
            error=(
                f"tool {name!r} is not available for this sample; "
                f"available tools: {listing}"
            ),
        )

    validate = getattr(env, "validate", None)
    outcome = validate(name, dict(action.arguments or {})) if callable(validate) else registry.validate(name, action.arguments)
    errors = _outcome_errors(outcome)
    if errors:
        return Action(
            type="invalid",
            name=name,
            arguments=action.arguments,
            raw=action.raw,
            error=f"invalid arguments for {name}: {errors}",
        )
    return action


__all__ = ["parse_model_output"]
