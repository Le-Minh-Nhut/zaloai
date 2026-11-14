"""Optional second-layer schema guardrail for parsed tool calls.

Layer position (never a replacement)::

    model text -> parse_model_output (protocol -> registry -> env/registry validate)
               -> check_action(...)                       # THIS module

The existing parser is authoritative. This layer may only ADD rejections for the
checks it declares; it MUST NOT weaken a rejection into an acceptance and MUST
NOT rewrite an ``env``-accepted call into something else silently. Allowed
outcomes are therefore:

* ``accepted``   — nothing to report; ``action`` is the input object, untouched.
* ``unchanged``  — ``enabled=False``; the input object is returned as-is.
* ``rejected``   — at least one reason; ``action`` is the input object and
  ``original_action`` carries it for debugging.
* ``normalized`` — only for coercions the tool schema explicitly permits (the
  only one implemented is a JSON number given for a declared ``integer``), with
  the reason recorded. ``original_action`` still carries the model's output.

Conflict handling: when an ``env`` object is supplied and its ``validate``
accepts the call while this layer rejects it, every reason is prefixed
``conflict:`` and ``env_said_ok=True`` is exposed so the caller can choose. The
default policy documented here is to keep the rejection (the guardrail is a
belt-and-braces layer, not an override), but the flag makes the disagreement
visible instead of silent.

Pure stdlib, duck-typed registry (``is_available``/``get``), no imports from
``lifemate.agent.tool_parser`` so either layer can be tested alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

__all__ = ["GuardrailOutcome", "check_action"]

#: Values a model emits as filler and that never reach a real tool.
_PLACEHOLDERS = frozenset({"<...>", "...", "TODO", "None", "null", ""})
#: ``<anything>`` counts too: ``<time>`` is the same filler with a longer label.
_PLACEHOLDER_RE = re.compile(r"^<[^<>]*>$")

_JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}

def _action_get(action: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict action or a duck-typed ``Action``."""
    if isinstance(action, Mapping):
        return action.get(key, default)
    return getattr(action, key, default)

def _action_dict(action: Any) -> dict[str, Any]:
    """Best-effort plain dict of the action, for ``original_action``."""
    if isinstance(action, Mapping):
        return dict(action)
    to_dict = getattr(action, "to_dict", None)
    if callable(to_dict):
        try:
            produced = to_dict()
        except Exception:  # pragma: no cover - defensive, never crash a guard
            produced = None
        if isinstance(produced, dict):
            return produced
    return {key: _action_get(action, key) for key in ("type", "name", "arguments", "error")}

def _with_arguments(action: Any, arguments: dict[str, Any]) -> Any:
    """``action`` with replaced arguments (a new object; the input is untouched)."""
    if isinstance(action, Mapping):
        updated = dict(action)
        updated["arguments"] = dict(arguments)
        return updated
    try:
        return replace(action, arguments=dict(arguments))
    except TypeError:  # pragma: no cover - duck-typed action, keep the original
        return action

def _argument_schema(registry: Any, tool: str) -> dict[str, Any] | None:
    """The tool's ``parameters`` mapping, or ``None`` when the registry lacks it."""
    spec = None
    getter = getattr(registry, "get", None)
    if callable(getter):
        try:
            spec = getter(tool)
        except Exception:  # pragma: no cover - defensive
            spec = None
    if spec is None:
        catalogue = getattr(registry, "catalog", None)
        spec = getattr(catalogue, "get", lambda _n: None)(tool) if catalogue is not None else None
    if spec is None:
        return None
    params = getattr(spec, "parameters", None)
    return dict(params) if isinstance(params, Mapping) else None

def _declared_types(schema: Mapping[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, (list, tuple)):
        return [str(t) for t in declared]
    return [str(declared)] if declared is not None else []

def _type_name(value: Any) -> str:
    return type(value).__name__

def _matches(value: Any, declared: str) -> bool:
    expected = _JSON_TYPE_CHECKS.get(declared)
    if expected is None:
        return True  # unknown type keyword: never invent a verdict
    if isinstance(value, bool) and declared in ("integer", "number"):
        return False
    return isinstance(value, expected)

def _bound_problems(arg: str, value: Any, schema: Mapping[str, Any]) -> list[str]:
    """Range violations, named by their bounds. Never clamps: 25 stays 25."""
    low = schema.get("minimum")
    high = schema.get("maximum")
    if isinstance(low, (int, float)) and not isinstance(low, bool) and isinstance(high, (int, float)) and not isinstance(high, bool):
        try:
            if value < low or value > high:
                return [f"range: {arg}={value!r} outside {low}..{high}"]
        except TypeError:
            return []
        return []
    problems: list[str] = []
    for key, label, violated in (
        ("minimum", "the minimum", lambda v, b: v < b),
        ("maximum", "the maximum", lambda v, b: v > b),
        ("exclusiveMinimum", "the exclusive minimum", lambda v, b: v <= b),
        ("exclusiveMaximum", "the exclusive maximum", lambda v, b: v >= b),
    ):
        if key not in schema:
            continue
        bound = schema[key]
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            continue  # malformed spec: do not invent a verdict
        try:
            if violated(value, bound):
                problems.append(f"range: {arg}={value!r} outside {label} {bound}")
        except TypeError:
            continue
    return problems

def _check_argument(arg: str, value: Any, schema: Mapping[str, Any]) -> tuple[list[str], Any]:
    """Problems for one argument plus the possibly-coerced value."""
    reasons: list[str] = []
    if not isinstance(schema, Mapping):
        return reasons, value
    declared = _declared_types(schema)
    coerced = value
    if declared:
        if not any(_matches(value, t) for t in declared):
            # The one schema-permitted coercion: a whole JSON number used where
            # an integer is declared. Everything else is rejected un-coerced.
            if (
                "integer" in declared
                and isinstance(value, float)
                and not isinstance(value, bool)
                and value == int(value)
            ):
                coerced = int(value)
                reasons.append(f"normalized: {arg}={value!r} -> {coerced} (integer declared)")
            else:
                reasons.append(
                    f"type: {arg} expected {'/'.join(declared)}, got {_type_name(value)}"
                )
                return reasons, value
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and coerced not in enum:
        reasons.append(f"enum: {arg}={coerced!r} not in {list(enum)}")
    if any(t in ("integer", "number") for t in declared):
        reasons.extend(_bound_problems(arg, coerced, schema))
    return reasons, coerced

def _placeholder_problems(argument: str, value: Any) -> list[str]:
    if isinstance(value, str):
        if value.strip() in _PLACEHOLDERS or _PLACEHOLDER_RE.match(value.strip()):
            return [f"placeholder: unrecognized placeholder {argument}={value!r}"]
        return []
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, str)
        and (item.strip() in _PLACEHOLDERS or _PLACEHOLDER_RE.match(item.strip()))
        for item in value
    ):
        return [f"placeholder: unrecognized placeholder inside {argument}"]
    return []

def _env_errors(env: Any, tool: str, args: dict[str, Any]) -> list[str]:
    """Errors from an ``env.validate`` (empty list == the env accepts)."""
    validate = getattr(env, "validate", None)
    if not callable(validate):
        return []
    try:
        outcome = validate(tool, dict(args))
    except Exception as exc:  # pragma: no cover - an env crash is not an approval
        return [f"env validate raised {type(exc).__name__}: {exc}"]
    if hasattr(outcome, "ok") and hasattr(outcome, "errors"):
        return [] if outcome.ok else [str(e) for e in (outcome.errors or [])]
    if isinstance(outcome, Mapping):
        errors = outcome.get("errors") or []
        return [str(e) for e in errors]
    return [] if not outcome else [str(outcome)]

@dataclass(frozen=True)
class GuardrailOutcome:
    """Result of the optional guardrail layer."""

    action: Any
    verdict: str
    reasons: tuple[str, ...] = ()
    original_action: dict[str, Any] | None = None
    env_said_ok: bool = False

    @property
    def accepted(self) -> bool:
        """True unless this layer rejected the call."""
        return self.verdict != "rejected"

def check_action(
    action: Any,
    registry: Any,
    *,
    enabled: bool = False,
    env: Any = None,
) -> GuardrailOutcome:
    """Validate ``action`` against the tool schema; never weaken the parser.

    ``enabled=False`` returns ``verdict="unchanged"`` with the SAME ``action``
    object and no reasons, so a disabled call keeps the baseline byte-for-byte.
    """
    if not enabled:
        return GuardrailOutcome(action=action, verdict="unchanged", reasons=())

    kind = str(_action_get(action, "type") or "")
    if kind == "finish":
        return GuardrailOutcome(action=action, verdict="accepted", reasons=())

    tool = str(_action_get(action, "name") or "")
    arguments = _action_get(action, "arguments") or {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    arguments = dict(arguments)
    original = _action_dict(action)

    if kind != "tool_call":
        # The protocol layer already rejected this; agree without double-reporting.
        error = str(_action_get(action, "error") or "not a tool_call")
        return GuardrailOutcome(
            action=action,
            verdict="rejected",
            reasons=tuple(sorted([f"protocol: {error}"])),
            original_action=original,
        )

    available = getattr(registry, "is_available", None)
    if callable(available) and not available(tool):
        listing = ", ".join(getattr(registry, "tool_names", lambda: [])()) or "(none)"
        return GuardrailOutcome(
            action=action,
            verdict="rejected",
            reasons=(f"tool: {tool!r} is not available for this sample; available: {listing}",),
            original_action=original,
        )

    parameters = _argument_schema(registry, tool)
    if parameters is None:
        # No schema to check against: this layer has nothing to say. It never
        # rejects on its own ignorance (the parser already gated availability).
        return GuardrailOutcome(action=action, verdict="accepted", reasons=())

    properties = parameters.get("properties") or {}
    properties = dict(properties) if isinstance(properties, Mapping) else {}
    required = [str(name) for name in (parameters.get("required") or [])]
    extra_allowed = parameters.get("additionalProperties") is True

    reasons: list[str] = []
    normalized: list[str] = []
    coerced_arguments = dict(arguments)
    for arg in sorted(arguments):
        value = arguments[arg]
        schema = properties.get(arg)
        if schema is None:
            if not extra_allowed:
                reasons.append(f"unsupported: unknown argument {arg!r}")
            continue
        problems, coerced = _check_argument(arg, value, schema)
        coerced_arguments[arg] = coerced
        for problem in problems:
            if problem.startswith("normalized:"):
                normalized.append(problem)
            else:
                reasons.append(problem)
        if arg in required:
            # Only a required argument is worth rejecting on a placeholder: the
            # official validator reads an empty OPTIONAL value as "not given".
            reasons.extend(_placeholder_problems(arg, value))

    env_errors = _env_errors(env, tool, arguments) if env is not None else []
    env_said_ok = env is not None and not env_errors

    if reasons and env_said_ok:
        reasons = [f"conflict: {reason} (env accepted this call)" for reason in reasons]
    elif env_errors:
        # An env rejection is repeated, never dropped into an acceptance: a
        # caller reading only this verdict must not start executing the call.
        reasons = [f"env: {error}" for error in env_errors]

    reasons.sort()
    if reasons:
        return GuardrailOutcome(
            action=action,
            verdict="rejected",
            reasons=tuple(reasons),
            original_action=original,
            env_said_ok=env_said_ok,
        )
    if normalized:
        return GuardrailOutcome(
            action=_with_arguments(action, coerced_arguments),
            verdict="normalized",
            reasons=tuple(sorted(normalized)),
            original_action=original,
            env_said_ok=env_said_ok,
        )
    return GuardrailOutcome(action=action, verdict="accepted", reasons=(), env_said_ok=env_said_ok)
