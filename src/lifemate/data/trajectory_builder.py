"""Causal SFT construction: turn gold trajectories into next-action examples.

The intellectual core of the LifeMate training pipeline. One *decision point*
is one (prompt, target action) pair where the prompt contains only what the
model is allowed to see at that moment:

* the spoken request, the world context, the conversation history,
* the tools of the sample's **installed apps** (plus the meta tool),
* the observations of steps ``0..k-1`` — never step ``k``'s own result and
  never ``label.final_state``.

Every decision point is therefore a strictly causal prediction: given the
visible past, emit the next action. The same prompt renderer
(:func:`lifemate.agent.prompt_builder.build_decision_prompt`) is used at
inference time, so train and test prompt bytes are identical for identical
inputs.

Refusals to guess (all recorded as issues/flags, never silently repaired):

* result lists shorter/longer than call lists,
* a result whose explicit index contradicts its position ⇒ pairing is provably
  wrong and the sample yields **no** decision points (the "past" of step k could
  really be its future),
* a result whose ``name`` disagrees with the call at that position ⇒ flagged
  (``name_mismatch``) but still built: result names are free-form in real data,
  so a spelling difference is not proof of mispairing,
* unknown ``ok`` flags,
* a final result that is missing ⇒ no terminal ``finish`` example
  (``visibility_limited``),
* write-argument values that cannot be derived from anything the model saw
  (``unobservable_ref_args``) — that decision point is **dropped**, never
  repaired and never injected into the prompt.

Pure stdlib; importable without torch/transformers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lifemate.agent.prompt_builder import StepRecord, build_decision_prompt, compact_history, to_messages
from lifemate.agent.protocol import FINISH_ACTION, make_tool_call, parse_action, render_action
from lifemate.agent.tool_registry import META_TOOL_NAME, ToolRegistry
from lifemate.errors import LifeMateError

from .preprocessing import estimate_tokens
from .schemas import Sample, ToolCall, ToolResult

__all__ = [
    "Alignment",
    "DecisionPoint",
    "align_calls_and_results",
    "assert_no_label_leak",
    "build_decision_points",
    "build_sft_dataset",
    "from_jsonl",
    "to_hf_dataset_records",
    "to_jsonl",
    "tool_result_to_observation",
]

#: Literal keys that must never appear in a prompt (label-only fields).
LABEL_MARKERS = ("final_state", "final_app_state", "end_state")

#: Write-call argument names treated as references when the schema says nothing.
_REF_ARG_SUFFIXES = ("_id", "_ref", "_name")

# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
@dataclass
class Alignment:
    """Positional call/result alignment plus everything suspicious about it.

    ``steps`` pairs every gold call with a result or ``None`` (result missing).
    ``truncated`` means the result list was *shorter* than the call list, so the
    tail of the trajectory has no observations. Extra results are listed in
    ``unaligned_results`` and are never attached to a step.
    """

    steps: list[tuple[ToolCall, ToolResult | None]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    truncated: bool = False
    unaligned_results: list[ToolResult] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def last_result_missing(self) -> bool:
        return bool(self.steps) and self.steps[-1][1] is None

    @property
    def misaligned(self) -> bool:
        """True when a result provably belongs to a *different* call step.

        Only ``result_out_of_order`` qualifies: an explicit index is the record's
        own statement of which call produced the result, so a position that
        contradicts it means the observation attached to step ``k`` came from
        another call, and the prompt of step ``k`` would show the future.

        ``name_mismatch`` deliberately does NOT qualify. Result names are
        free-form in real data (``list`` vs ``alarm.list``, a variant spelling),
        so a name difference is a data-quality signal — flagged, never a proof of
        mispairing, and never a reason to discard an otherwise sound sample.
        """
        return any(issue.startswith("result_out_of_order:") for issue in self.issues)

def _explicit_index(raw_entry: Any) -> int | None:
    """Index if the *raw* result entry exposes one explicitly, else ``None``.

    ``schemas.ToolResult`` drops unknown fields, so the lookup goes through the
    label's raw record (``label.raw["tool_results"][i]``). When the dataset
    carries no explicit index this returns ``None`` for every entry and the
    out-of-order check is a no-op — it never guesses.
    """
    if not isinstance(raw_entry, Mapping):
        return None
    for key in ("index", "step", "position", "call_index"):
        value = raw_entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _raw_result_entries(sample: Sample) -> list[Any]:
    """Raw result entries from ``label.raw`` (empty when unavailable)."""
    label = getattr(sample, "label", None)
    raw = getattr(label, "raw", None)
    if not isinstance(raw, Mapping):
        return []
    for key in ("tool_results", "results", "observations"):
        value = raw.get(key)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []

def align_calls_and_results(sample: Sample) -> Alignment:
    """Pair gold calls with gold results positionally, conservatively.

    Rules (deliberate, documented, tested):

    * counts equal ⇒ plain 1:1 positional pairing;
    * ``len(results) < len(calls)`` ⇒ pair the prefix, remaining calls get
      ``None`` and ``truncated=True`` (a later result is never borrowed);
    * ``len(results) > len(calls)`` ⇒ pair the prefix, extras go to
      ``unaligned_results`` and are ignored (never attached to a step);
    * a result carrying a ``name`` different from the call at that position ⇒
      ``name_mismatch`` issue (pairing stays positional: remapping would be a
      guess);
    * an explicit index in the record that disagrees with its position ⇒
      ``result_out_of_order`` issue;
    * unknown ``ok`` ⇒ ``ok_unknown`` issue.
    """
    label = getattr(sample, "label", None)
    calls: list[ToolCall] = list(getattr(label, "tool_calls", []) or []) if label else []
    results: list[ToolResult] = list(getattr(label, "tool_results", []) or []) if label else []
    sample_id = str(getattr(sample, "id", "?"))

    n_pairs = min(len(calls), len(results))
    raw_entries = _raw_result_entries(sample)
    steps: list[tuple[ToolCall, ToolResult | None]] = []
    issues: list[str] = []

    for index in range(len(calls)):
        call = calls[index]
        result = results[index] if index < n_pairs else None
        if result is not None:
            if result.name and call.name and str(result.name) != str(call.name):
                issues.append(
                    f"name_mismatch: step {index} call {call.name!r} has result named {result.name!r}"
                )
            explicit = _explicit_index(raw_entries[index]) if index < len(raw_entries) else None
            if explicit is not None and explicit != index:
                issues.append(f"result_out_of_order: step {index} carries explicit index {explicit}")
            if result.ok is None:
                issues.append(f"ok_unknown: step {index} ({call.name!r}) has no ok flag")
        steps.append((call, result))

    unaligned = results[n_pairs:]
    if unaligned:
        issues.append(f"extra_results: {len(unaligned)} result(s) beyond {len(calls)} call(s) ignored")

    alignment = Alignment(
        steps=steps,
        issues=issues,
        truncated=len(results) < len(calls),
        unaligned_results=list(unaligned),
    )
    if not calls and results:
        issues.append(f"results_without_calls: {sample_id} has {len(results)} result(s) and no call")
    return alignment

# --------------------------------------------------------------------------- #
# Observation conversion
# --------------------------------------------------------------------------- #
def tool_result_to_observation(result: ToolResult | None) -> dict[str, Any]:
    """Convert a gold :class:`ToolResult` to the env's observation dict shape.

    Output is exactly what ``env.call()`` returns and what the inference-time
    prompt shows: ``{"ok": True, "result": ...}`` or ``{"ok": False, "error": ...}``.

    ``ok`` resolution (documented, tested):

    * ``ok is True`` ⇒ success; ``result`` field used (``None`` allowed);
    * ``ok is False`` ⇒ failure; ``error`` string used, synthesised when absent;
    * ``ok is None`` (unknown) ⇒ True when the record carries **no** error
      field, otherwise False — the same inference ``ToolResult.from_raw`` uses,
      applied here explicitly so the rule is visible and testable;
    * ``result is None`` (missing gold result) ⇒ ``{"ok": False, "error":
      "missing_tool_result"}``: an unobserved outcome is never reported as
      success.
    """
    if result is None:
        return {"ok": False, "error": "missing_tool_result"}
    ok = result.ok
    if ok is None:
        ok = result.error is None
    if ok:
        payload = getattr(result, "payload", None)
        if payload:
            # Exactly what ``env.call()`` returns: the payload inline, not nested.
            return {"ok": True, **payload}
        if result.result is None:
            return {"ok": True}  # e.g. alarm.delete -> {"ok": True}
        return {"ok": True, "result": result.result}
    return {"ok": False, "error": str(result.error) if result.error is not None else "tool_error"}

# --------------------------------------------------------------------------- #
# Label-leak guard
# --------------------------------------------------------------------------- #
def assert_no_label_leak(prompt_body: str, final_state: Any = None) -> None:
    """Hard-fail when prompt text contains gold-only state.

    Checks, in order:

    1. label-only key literals (``final_state`` / ``final_app_state`` /
       ``end_state``) anywhere in the prompt, including the ``"key":`` JSON form;
    2. when ``final_state`` is supplied: its canonical JSON must not appear, and
       no nested container of it (canonicalised, ≥ 2 entries) may appear either
       — that is the signature of a copied state block.

    Leaf *values* are deliberately NOT checked. ``final_state`` is derived from
    tool results, so values like an ``alarm_id`` are legitimately model-visible
    through the observations of earlier steps; matching them would reject
    correct prompts. What must never appear is the state *structure* or a
    label-only key.

    Raising here is intentional: a leak is a data-construction bug, not a sample
    to skip.
    """
    text = prompt_body or ""
    for marker in LABEL_MARKERS:
        if re.search(rf'"{marker}"|{marker}\s*:', text) or f"'{marker}'" in text:
            raise LifeMateError(
                f"label leak: prompt contains label-only key {marker!r}; "
                "final state must never be model input"
            )
    if final_state is None or not final_state:
        # An empty final_state canonicalises to "{}", which occurs in any prompt
        # carrying an empty object; it leaks nothing.
        return
    canonical = _canonical(final_state)
    if _MIN_NEEDLE_CHARS <= len(canonical) and canonical in text:
        raise LifeMateError("label leak: canonical JSON of label.final_state appears in the prompt")
    for needle in _structural_needles(final_state):
        if needle in text:
            raise LifeMateError(
                f"label leak: final_state sub-structure {needle[:120]!r}… appears in the prompt"
            )

def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

#: A container shorter than this carries no state structure, only values that
#: legitimately appear in observations (``{"ok": true}``).
_MIN_NEEDLE_CHARS = 30

def _structural_needles(value: Any) -> list[str]:
    """Canonical forms of every nested container inside ``final_state``.

    Containers whose canonical form is shorter than :data:`_MIN_NEEDLE_CHARS`
    are skipped: tiny dicts like ``{"ok": true}`` occur in every prompt, so
    matching them would reject correct data. Multi-key/master-state structures
    are long and unique, which is exactly the leak signature we want.
    """
    needles: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            canonical = _canonical(node)
            if len(canonical) >= _MIN_NEEDLE_CHARS:
                needles.append(canonical)
            for key in node:
                walk(node[key])
        elif isinstance(node, (list, tuple)):
            # A list of scalars is a leaf *value* (e.g. an email ``to: [...]``
            # recipient list), legitimately model-visible; only lists that
            # contain containers carry state structure.
            if any(isinstance(item, (Mapping, list, tuple)) for item in node):
                canonical = _canonical(list(node))
                if len(canonical) >= _MIN_NEEDLE_CHARS:
                    needles.append(canonical)
            for item in node:
                walk(item)

    walk(value)
    return needles

# --------------------------------------------------------------------------- #
# Decision points
# --------------------------------------------------------------------------- #
@dataclass
class DecisionPoint:
    """One supervised example: visible context + the action to emit."""

    sample_id: str
    step_index: int
    prompt_body: str
    target_action: dict[str, Any]
    observation_after: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def messages(self) -> list[dict[str, str]]:
        """Chat messages (system + user) used for tokenisation."""
        return to_messages(self.prompt_body)

    def target_text(self) -> str:
        """Target rendered exactly as :func:`render_action` does."""
        return render_action(self.target_action)

    def to_record(self) -> dict[str, Any]:
        """Compact, self-contained training record (prompts are never rebuilt)."""
        return {
            "sample_id": self.sample_id,
            "step_index": self.step_index,
            "messages": self.messages(),
            "target": self.target_text(),
            "meta": dict(self.meta),
        }

def _spec_lookup(registry: ToolRegistry) -> dict[str, Any]:
    return {str(getattr(spec, "name", "")): spec for spec in registry.specs}

def _is_meta_call(name: Any, specs: Mapping[str, Any]) -> bool:
    """True when ``name`` is the ask-for-missing-information meta tool.

    The catalog may spell it ``sys.ask_missing_info`` or (in the ``sys`` app)
    ``ask_missing_info``; both are recognised by
    :func:`lifemate.data.catalog._infer_kind` as ``kind == "meta"``. Checking the
    resolved spec means a renamed alias stays terminal instead of earning a
    ``finish`` point right after it.
    """
    text = str(name or "")
    if text == META_TOOL_NAME:
        return True
    spec = specs.get(text)
    if spec is not None and str(getattr(spec, "kind", "") or "").lower() == "meta":
        return True
    return text.split(".")[-1] == META_TOOL_NAME.split(".")[-1]

def _is_write_call(spec: Any) -> bool:
    """Catalog kinds are ``read``/``write``/``meta``/``unknown``; writes only."""
    return str(getattr(spec, "kind", "") or "").lower() == "write"

def _ref_keys(spec: Any) -> set[str]:
    """Argument names the catalog marks as *value references* (``x-ref``).

    ``x-match`` is deliberately excluded: in the official catalog all 40
    occurrences are ``"fuzzy"`` free text the assistant composes itself
    (``content``, ``title``, ``body``, ``note``, ``subject``, ``address``,
    ``pickup``/``dropoff``), not a value copied from an observation. Treating
    them as references flagged 409 of 410 reported arguments for values the user
    dictated, drowning the 1 real case. The EDA's own ``_ref_of`` already keeps
    the two kinds apart.
    """
    keys: set[str] = set()
    params = getattr(spec, "parameters", None)
    properties = params.get("properties") if isinstance(params, Mapping) else None
    if isinstance(properties, Mapping):
        for name, schema in properties.items():
            if isinstance(schema, Mapping) and any(
                k in schema for k in ("x-ref", "x_ref", "ref", "x-from", "x_from")
            ):
                keys.add(str(name))
    return keys

def _visible_text(
    sample: Sample,
    prior_observations: Sequence[dict[str, Any]],
    history_limit: int | None = None,
    prior_calls: Sequence[ToolCall] = (),
) -> str:
    """Everything the *prompt* shows so far, as one searchable blob.

    Mirrors :func:`lifemate.agent.prompt_builder.build_decision_prompt`: the
    history turns are windowed by ``history_limit``, and the ``# Các bước đã
    thực hiện`` section renders each *earlier action* (the model's own emitted
    ``{name, arguments}``) alongside its observation. Omitting the earlier
    actions would report a value as unseen while the prompt literally shows it
    (real case: ``chat_ott.send_location.chat_id=chat_9017`` at step 2 of
    ``train_1657``, visible in step 1's action).
    ``history_limit=None`` keeps every turn (the old, over-claiming behaviour).
    """
    parts: list[str] = [str(getattr(sample, "audio_text", "") or "")]
    context = getattr(sample, "context", None)
    if context is not None:
        parts.append(json.dumps(context.to_dict(), ensure_ascii=False, sort_keys=True, default=str))
    turns = list(getattr(sample, "history", []) or [])
    if history_limit is not None:
        turns = compact_history(turns, history_limit)
    for turn in turns:
        parts.append(str(getattr(turn, "content", "") or ""))
        call = getattr(turn, "tool_call", None)
        if call:
            parts.append(json.dumps(call, ensure_ascii=False, sort_keys=True, default=str))
        result = getattr(turn, "tool_result", None)
        if result:
            parts.append(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    for call in prior_calls:
        parts.append(json.dumps(make_tool_call(call.name, call.arguments), ensure_ascii=False, sort_keys=True, default=str))
    for observation in prior_observations:
        parts.append(json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str))
    return "\n".join(parts)

def _unobservable_ref_args(
    calls: Sequence[ToolCall],
    specs: Mapping[str, Any],
    sample: Sample,
    observations: Sequence[dict[str, Any]],
    history_limit: int | None = None,
) -> dict[int, list[str]]:
    """Reference-like write arguments whose value is nowhere in the visible past.

    Only *write* calls are inspected, and only arguments the catalog declares as
    references (``x-ref``/``x-from``) or that look like identifiers
    (``*_id``/``*_ref``/``*_name``). A value found in the request, context, the
    *rendered* history or earlier observations is observable and is not
    reported. Reported, never repaired: injecting the gold value would leak the
    answer.

    Returns ``{call index: ["stepK:tool.arg=value", ...]}`` so a caller can drop
    exactly the decision point whose own target is affected instead of tarring
    every point of the sample.

    ``history_limit`` MUST be the window the prompt is rendered with: searching
    turns the renderer drops reports a value as observable that the model never
    sees. In the real train split 13 samples (19 arguments) had a gold ID living
    only in a turn beyond the 10-turn window.
    """
    flagged: dict[int, list[str]] = {}
    for index, call in enumerate(calls):
        spec = specs.get(call.name)
        if spec is None or not _is_write_call(spec):
            continue
        declared = _ref_keys(spec)
        visible = _visible_text(sample, observations[:index], history_limit, calls[:index])
        for key, value in (call.arguments or {}).items():
            if not (key in declared or str(key).endswith(_REF_ARG_SUFFIXES)):
                continue
            if value is None or isinstance(value, (dict, list)):
                continue
            needle = str(value)
            if needle and needle not in visible:
                flagged.setdefault(index, []).append(f"step{index}:{call.name}.{key}={needle}")
    return flagged

def build_decision_points(
    sample: Sample,
    catalog: Any,
    registry: ToolRegistry | None = None,
    history_limit: int = 10,
) -> list[DecisionPoint]:
    """Build every causal decision point of one gold trajectory.

    Layout for a sample with ``N`` calls (``N >= 1``) whose last call is not the
    meta tool::

        DP 0 .. DP N-1 : target = call k, prompt shows steps < k
        DP N           : target = finish, prompt shows steps 0..N-1

    Special cases (all documented in ``docs/TRAINING_GUIDE.md``):

    * ``N == 0`` ⇒ a single ``finish`` decision point with no steps
      (out-of-scope / nothing-to-do requests);
    * the last gold call is ``sys.ask_missing_info`` ⇒ the trajectory is
      **terminal at that question**: targets are ``[ask_missing_info]`` and no
      ``finish`` point is emitted (the user's answer is not part of the label);
    * the last aligned result is missing ⇒ no ``finish`` point and the sample is
      flagged ``visibility_limited`` (the outcome was never observed);
    * ``ask_missing_info`` elsewhere in the trajectory is an ordinary step.
    """
    resolved = registry if registry is not None else ToolRegistry(catalog).for_sample(sample)
    specs = _spec_lookup(resolved)
    alignment = align_calls_and_results(sample)
    if alignment.misaligned:
        # Positional pairing is contradicted by the record: the observation of
        # step k may belong to another call, so its prompt would show the future.
        # No decision point is trustworthy here; refuse to emit any.
        return []
    calls = [call for call, _res in alignment.steps]
    observations = [tool_result_to_observation(result) for _call, result in alignment.steps]

    sample_flags: list[str] = []
    if alignment.truncated:
        sample_flags.append("missing_results")
    if alignment.last_result_missing:
        sample_flags.append("visibility_limited")
    if any("name_mismatch" in issue for issue in alignment.issues):
        sample_flags.append("name_mismatch")
    if alignment.misaligned:
        # An observation attached to the wrong step turns into a *future*
        # observation as soon as the pairing is reversed: the prompt of step k
        # would show a result produced by a later call. Report it here; the
        # dataset builder drops the sample instead of teaching that past.
        sample_flags.append("misaligned_results")

    category = str((getattr(sample, "raw", {}) or {}).get("category") or "unknown")
    n_total = len(calls)
    prompt_kwargs = {
        "transcript": getattr(sample, "audio_text", "") or "",
        "context": getattr(sample, "context", None),
        "history": getattr(sample, "history", []) or [],
        "installed_apps": list(getattr(sample, "context", None).installed_apps or []),
        "catalog": resolved,
        "history_limit": history_limit,
    }

    def make_point(index: int, target: dict[str, Any], prior: int, observation_after: Any) -> DecisionPoint:
        steps = [
            StepRecord(
                index=i,
                action=make_tool_call(calls[i].name, calls[i].arguments),
                observation=observations[i],
            )
            for i in range(prior)
        ]
        prompt = build_decision_prompt(prior_steps=steps, **prompt_kwargs)
        return DecisionPoint(
            sample_id=str(getattr(sample, "id", "?")),
            step_index=index,
            prompt_body=prompt,
            target_action=target,
            observation_after=observation_after,
            meta={
                "token_estimate": estimate_tokens(prompt),
                "n_steps_total": n_total,
                "source": str(getattr(sample, "split", "") or ""),
                "category": category,
                "n_calls": n_total,
                "n_calls_bucket": "3+" if n_total >= 3 else str(n_total),
                "is_terminal": bool(target.get("type") == "finish"),
                "visibility_limited": alignment.last_result_missing,
                "sample_flags": list(sample_flags),
                "history_limit": history_limit,
            },
        )

    points: list[DecisionPoint] = []

    if token_prompt_is_empty(prompt_kwargs):
        return points

    if n_total == 0:
        point = make_point(0, dict(FINISH_ACTION), 0, None)
        points.append(point)
        return points

    unobservable = _unobservable_ref_args(calls, specs, sample, observations, history_limit)
    for index, call in enumerate(calls):
        point = make_point(
            index,
            make_tool_call(call.name, call.arguments),
            index,
            observations[index],
        )
        if unobservable.get(index):
            # Only the point whose *own* target carries the unseen value is
            # marked; the rest of the trajectory is unaffected.
            point.meta["unobservable_ref_args"] = list(unobservable[index])
            point.meta["sample_flags"] = sorted(set(point.meta["sample_flags"] + ["unobservable_ref"]))
        points.append(point)

    ends_with_ask = _is_meta_call(calls[-1].name, specs)
    if not ends_with_ask and not alignment.last_result_missing:
        points.append(make_point(n_total, dict(FINISH_ACTION), n_total, None))

    for point in points:
        assert_no_label_leak(point.prompt_body, getattr(sample.label, "final_state", None))
    return points

def token_prompt_is_empty(prompt_kwargs: Mapping[str, Any]) -> bool:
    """True when the rendered prompt would carry no request and no context."""
    transcript = str(prompt_kwargs.get("transcript") or "").strip()
    context = prompt_kwargs.get("context")
    has_context = False
    if context is not None:
        try:
            has_context = any(
                value not in (None, "", [], {}) for value in context.to_dict().values()
            )
        except Exception:  # pragma: no cover - duck-typed context
            has_context = True
    return not transcript and not has_context

# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
_DROP_REASONS = (
    "no_label",
    "zero_calls_no_finish_possible",
    "catalog_missing_tool",
    "unexecutable_gold_target",
    "misaligned_results",
    "empty_prompt",
)

#: Decision-point-level drops (the sample survives; these points do not).
_DP_DROP_REASONS = (
    "unobservable_ref_target",
)

def build_sft_dataset(
    samples: Sequence[Sample],
    catalog: Any,
    *,
    history_limit: int = 10,
) -> tuple[list[DecisionPoint], dict[str, Any]]:
    """Build the whole SFT dataset plus accounting stats.

    Drop reasons (exact keys/counts, never silent):

    * ``no_label`` — no gold trajectory at all;
    * ``catalog_missing_tool`` — a gold call is not available to this sample
      (tool absent from the catalog or belonging to an uninstalled app): the
      prompt could not list it, so training on it would teach an impossible
      action;
    * ``zero_calls_no_finish_possible`` — zero calls but stray results, so the
      only possible target (``finish``) would contradict the record;
    * ``unexecutable_gold_target`` — a gold call fails catalog validation
      (missing required argument, wrong type, bad enum). Supervising it teaches
      the model to emit an action the agent loop rejects as ``invalid`` and
      never executes; the missing-information case must be a
      ``sys.ask_missing_info`` call in the label, not a fabricated argument;
    * ``misaligned_results`` — a result carries an explicit index that
      contradicts its position, so pairing is provably wrong and its prompt
      would show a future observation. A ``name_mismatch`` alone is *not*
      enough (result names are free-form in real data): it is flagged on the
      decision points and the sample is still built;
    * ``empty_prompt`` — neither request text nor context to condition on.
    """
    points: list[DecisionPoint] = []
    dropped = {reason: 0 for reason in _DROP_REASONS}
    dropped_dp = {reason: 0 for reason in _DP_DROP_REASONS}
    visibility_limited = 0
    unobservable_ref = 0
    trajectory_histogram: dict[str, int] = {}
    target_char_lengths: list[int] = []

    for sample in samples:
        label = getattr(sample, "label", None)
        if label is None:
            dropped["no_label"] += 1
            continue
        calls = list(label.tool_calls or [])
        results = list(label.tool_results or [])
        if not calls and results:
            dropped["zero_calls_no_finish_possible"] += 1
            continue
        bucket = "3+" if len(calls) >= 3 else str(len(calls))
        trajectory_histogram[bucket] = trajectory_histogram.get(bucket, 0) + 1

        registry = ToolRegistry(catalog).for_sample(sample)
        available = registry.as_set()
        missing = sorted({c.name for c in calls if c.name not in available})
        if missing:
            dropped["catalog_missing_tool"] += 1
            continue
        if any(registry.validate(c.name, c.arguments).get("errors") for c in calls):
            # A gold action the loop would reject before reaching the env
            # (missing/ill-typed argument) must never be supervised: the model
            # would learn an action that is syntactically valid and unusable.
            dropped["unexecutable_gold_target"] += 1
            continue

        sample_points = build_decision_points(sample, catalog, registry=registry, history_limit=history_limit)
        if not sample_points:
            # Misalignment is reported separately so the drop reason stays honest.
            dropped["misaligned_results" if align_calls_and_results(sample).misaligned else "empty_prompt"] += 1
            continue
        # A target whose reference value the prompt cannot show is unsupervised,
        # not repaired: dropping the single affected decision point keeps the
        # rest of the trajectory while never teaching a guess.
        affected = [p for p in sample_points if p.meta.get("unobservable_ref_args")]
        if affected:
            sample_points = [p for p in sample_points if not p.meta.get("unobservable_ref_args")]
            dropped_dp["unobservable_ref_target"] += len(affected)
            unobservable_ref += 1
        points.extend(sample_points)
        if any(p.meta.get("visibility_limited") for p in sample_points):
            visibility_limited += 1
        target_char_lengths.extend(len(p.target_text()) for p in sample_points)

    target_estimate = [estimate_tokens(p.target_text()) for p in points]
    prompt_estimate = [int(p.meta.get("token_estimate") or 0) for p in points]
    stats: dict[str, Any] = {
        "n_samples": len(samples),
        "n_samples_kept": len(samples) - sum(dropped.values()),
        "n_decision_points": len(points),
        "dropped": dropped,
        "n_dropped": sum(dropped.values()),
        "dropped_decision_points": dropped_dp,
        "n_dropped_decision_points": sum(dropped_dp.values()),
        "visibility_limited": visibility_limited,
        "unobservable_ref": unobservable_ref,
        "target_chars": _describe(target_char_lengths),
        "target_tokens_estimate": _describe(target_estimate),
        "prompt_tokens_estimate": _describe(prompt_estimate),
        "trajectory_length_histogram": dict(sorted(trajectory_histogram.items())),
        "tokenization": "chars/4 estimate (tokenizer absent)",
        "history_limit": history_limit,
    }
    return points, stats

def _describe(values: Sequence[int]) -> dict[str, Any]:
    """min/mean/max for an int sequence (all zeros when empty)."""
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0, "n": 0}
    ordered = sorted(int(v) for v in values)
    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "max": ordered[-1],
        "n": len(ordered),
    }

# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def to_jsonl(points: Iterable[DecisionPoint], path: str | Path) -> Path:
    """Write compact SFT records (one JSON object per line); atomic replace."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(target_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(json.dumps(point.to_record(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    tmp.replace(target_path)
    return target_path

def from_jsonl(path: str | Path) -> list[DecisionPoint]:
    """Read records written by :func:`to_jsonl` back into decision points."""
    points: list[DecisionPoint] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LifeMateError(f"{path}:{lineno}: invalid JSON in SFT record ({exc.msg})") from exc
            messages = record.get("messages") or []
            prompt_body = str(messages[-1].get("content", "")) if messages else ""
            action = parse_action(str(record.get("target") or ""))
            if action.type == "invalid":
                raise LifeMateError(
                    f"{path}:{lineno}: target is not a valid action block ({action.error})"
                )
            points.append(
                DecisionPoint(
                    sample_id=str(record.get("sample_id", "")),
                    step_index=int(record.get("step_index", 0)),
                    prompt_body=prompt_body,
                    target_action=action.to_dict(),
                    observation_after=None,
                    meta=dict(record.get("meta") or {}),
                )
            )
    return points

def to_hf_dataset_records(points: Iterable[DecisionPoint]) -> list[dict[str, Any]]:
    """Plain list of dicts (``datasets.Dataset.from_list``-ready, no HF import)."""
    return [point.to_record() for point in points]
