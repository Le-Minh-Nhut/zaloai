"""Prediction metrics for LifeMate tool-call decisions.

Pure stdlib: importable without torch/transformers and safe to use in reports.
Nothing here reads the dataset, and nothing here invents a number: every
function returns ``n == 0`` plus ``value = None`` when the metric is undefined
on the given input, and :func:`aggregate_teacher_forced` turns those into
explicit ``{"available": False}`` blocks.

Canonicalisation choices (they change scores, so they are documented):

* **Key order is irrelevant.** Arguments are re-serialised with sorted keys.
* **``1`` and ``1.0`` are equal.** Integral floats are folded to ``int``;
  other floats are rounded to 9 significant decimals so that
  ``0.30000000000000004`` and ``0.3`` match.
* **Strings are stripped.** Leading/trailing whitespace in argument values is
  an artefact of generation, not a decision; interior whitespace is kept.
* **Lists keep their order.** Argument lists (e.g. a list of alarms) are
  order-sensitive unless the tool schema says otherwise; reordering them
  changes behaviour, so it must not score as equal.
* **Booleans are not integers.** ``True`` never equals ``1``.
* **Call order matters.** Tool calls are a sequence (a later call may depend on
  an earlier result), so index alignment is used, never a multiset.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "METRIC_DEFINITIONS",
    "aggregate_teacher_forced",
    "argument_exact_match",
    "call_count_delta",
    "canonical_json",
    "canonical_signature",
    "decision_emitted",
    "lcs_length",
    "missing_info_detection",
    "next_action_accuracy",
    "out_of_scope_accuracy",
    "sequence_precision_recall_f1",
    "stop_accuracy",
    "tool_call_exact_match",
    "tool_selection_accuracy",
]

#: ``sys.ask_missing_info`` is the meta tool the agent uses to request data.
META_TOOL = "sys.ask_missing_info"

METRIC_DEFINITIONS: dict[str, str] = {
    "tool_call_exact_match": (
        "1.0 when the whole predicted tool-call sequence equals the gold sequence "
        "after canonicalisation (names, arguments, order); 0.0 otherwise. "
        "A record where both sides are empty counts as a match."
    ),
    "sequence_f1": (
        "Precision/recall/F1 of the predicted call sequence against the gold "
        "sequence, computed from the longest common subsequence (LCS) of "
        "canonical call signatures. LCS is used instead of a multiset because "
        "call order carries meaning (a call may consume an earlier result), and "
        "instead of exact alignment because one inserted call should not "
        "penalise every later call. It is *not* a scheduling metric: two calls "
        "are only aligned when they keep their relative order."
    ),
    "argument_exact_match": (
        "Among positionally aligned call pairs that share the same tool name, "
        "the fraction whose arguments are canonically identical. Undefined "
        "when no pair shares a name."
    ),
    "tool_selection_accuracy": (
        "Fraction of samples whose first predicted tool name equals the first "
        "gold tool name. Two empty sequences count as correct (nothing to "
        "select); exactly one empty sequence counts as wrong."
    ),
    "next_action_accuracy": (
        "Per-step accuracy over an episode: the prediction and gold of every "
        "step index are compared after canonicalisation; a missing predicted "
        "step (prediction stopped early) counts as wrong. Undefined when no "
        "sample provides per-step data."
    ),
    "stop_accuracy": (
        "Fraction of samples where the agent stopped at the right point: the "
        "predicted number of calls equals the gold number of calls *and* the "
        "predicted finish flag matches the gold finish flag when the gold flag "
        "is known."
    ),
    "missing_info_detection": (
        "Confusion counts (tp/fp/fn) for the meta tool ``sys.ask_missing_info`` "
        "at sample level: positive = the agent called it (or emitted a "
        "missing-information request), negative = it did not. "
        "Precision/recall/F1 are undefined when the corresponding denominator "
        "is zero."
    ),
    "out_of_scope_accuracy": (
        "Fraction of gold samples with zero expected calls (out-of-scope / "
        "nothing to do) where the agent also emitted zero calls. Undefined when "
        "no such sample exists."
    ),
    "call_count_delta": (
        "Mean of (predicted call count - gold call count) across samples. "
        "Positive = the agent over-calls, negative = it stops early. This is a "
        "diagnostic, not a score."
    ),
}

# --------------------------------------------------------------------------- #
# Canonicalisation
# --------------------------------------------------------------------------- #
def _canon_value(value: Any) -> Any:
    """Recursively normalise a JSON-ish value (see module docstring)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 9)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return {str(k).strip(): _canon_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]).strip())}
    if isinstance(value, (list, tuple)):
        return [_canon_value(item) for item in value]
    return str(value)


def canonical_json(value: Any) -> str:
    """Canonical JSON text of ``value``, stable across key order / 1 vs 1.0."""
    return json.dumps(_canon_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _as_calls(calls: Any) -> list[dict[str, Any]]:
    """Coerce a call container into ``[{"name": str, "arguments": dict}]``."""
    if calls is None:
        return []
    if isinstance(calls, Mapping):
        calls = [calls]
    out: list[dict[str, Any]] = []
    for call in calls or []:
        if isinstance(call, Mapping):
            name = call.get("name", call.get("tool", ""))
            args = call.get("arguments", call.get("args", {}))
        else:  # duck-typed ToolCall / Action
            name = getattr(call, "name", "")
            args = getattr(call, "arguments", {})
        out.append({"name": str(name or "").strip(), "arguments": dict(args) if isinstance(args, Mapping) else {}})
    return out


def canonical_signature(call: Mapping[str, Any]) -> str:
    """One call's comparable signature: name plus canonical arguments."""
    name = str(call.get("name", "")).strip()
    return name + canonical_json(dict(call.get("arguments") or {}))


def _normalise_pred(pred: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise a prediction record to the internal record shape."""
    pred = dict(pred or {})
    return {
        "id": str(pred.get("id", "")),
        "tool_calls": _as_calls(pred.get("tool_calls")),
        "finish": pred.get("finish", pred.get("finished")),
        "steps": pred.get("steps"),
        "tool_results": list(pred.get("tool_results") or []),
        "status": pred.get("status"),
    }


# --------------------------------------------------------------------------- #
# Sequence metrics
# --------------------------------------------------------------------------- #
def lcs_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Length of the longest common subsequence (O(len a * len b) memory-light)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for item_a in a:
        cur = [0]
        for idx_b, item_b in enumerate(b, start=1):
            if item_a == item_b:
                cur.append(prev[idx_b - 1] + 1)
            else:
                cur.append(max(prev[idx_b], cur[idx_b - 1]))
        prev = cur
    return prev[-1]


def tool_call_exact_match(pred: list[dict] | None, gold: list[dict] | None) -> bool:
    """True when both sequences are identical after canonicalisation."""
    pred_sigs = [canonical_signature(c) for c in _as_calls(pred)]
    gold_sigs = [canonical_signature(c) for c in _as_calls(gold)]
    return pred_sigs == gold_sigs


def sequence_precision_recall_f1(pred: list[dict] | None, gold: list[dict] | None) -> dict[str, Any]:
    """LCS-based precision/recall/F1 of the call sequence.

    ``lcs / len(pred)`` is precision, ``lcs / len(gold)`` recall. With both
    sequences empty the metric is defined and perfect (1.0): doing nothing on a
    nothing-to-do sample is correct.
    """
    pred_sigs = [canonical_signature(c) for c in _as_calls(pred)]
    gold_sigs = [canonical_signature(c) for c in _as_calls(gold)]
    lcs = lcs_length(pred_sigs, gold_sigs)
    precision = 1.0 if not pred_sigs and not gold_sigs else (lcs / len(pred_sigs) if pred_sigs else 0.0)
    recall = 1.0 if not gold_sigs and not pred_sigs else (lcs / len(gold_sigs) if gold_sigs else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lcs_len": lcs,
        "pred_n": len(pred_sigs),
        "gold_n": len(gold_sigs),
    }


def argument_exact_match(pred: list[dict] | None, gold: list[dict] | None) -> dict[str, Any]:
    """Canonical argument equality over positionally aligned same-name pairs."""
    pred_calls = _as_calls(pred)
    gold_calls = _as_calls(gold)
    correct = 0
    compared = 0
    for pred_call, gold_call in zip(pred_calls, gold_calls):
        if pred_call["name"] != gold_call["name"]:
            continue
        compared += 1
        if canonical_json(pred_call["arguments"]) == canonical_json(gold_call["arguments"]):
            correct += 1
    return {
        "value": None if compared == 0 else correct / compared,
        "n": compared,
        "correct": correct,
    }


def tool_selection_accuracy(pred: list[dict] | None, gold: list[dict] | None) -> dict[str, Any]:
    """First-call name match for one sample."""
    pred_calls = _as_calls(pred)
    gold_calls = _as_calls(gold)
    if not pred_calls and not gold_calls:
        return {"value": 1.0, "n": 1, "correct": 1}
    if not pred_calls or not gold_calls:
        return {"value": 0.0, "n": 1, "correct": 0}
    hit = pred_calls[0]["name"] == gold_calls[0]["name"]
    return {"value": 1.0 if hit else 0.0, "n": 1, "correct": int(hit)}


def next_action_accuracy(pred_steps: Sequence[Any] | None, gold_steps: Sequence[Any] | None) -> dict[str, Any]:
    """Per-step canonical equality; missing predicted steps count as wrong.

    A step may be a call dict, an action dict (``{"type": "finish"}``) or a
    plain string. Call dicts are compared by canonical signature; everything
    else by canonical JSON.
    """
    gold_list = list(gold_steps or [])
    pred_list = list(pred_steps or [])
    if not gold_list:
        return {"value": None, "n": 0, "correct": 0}
    correct = 0
    for index, gold_step in enumerate(gold_list):
        if index >= len(pred_list):
            continue
        if _step_signature(pred_list[index]) == _step_signature(gold_step):
            correct += 1
    return {"value": correct / len(gold_list), "n": len(gold_list), "correct": correct}


def _step_signature(step: Any) -> str:
    if isinstance(step, Mapping):
        if step.get("name") or step.get("type") == "tool_call":
            return canonical_signature({"name": step.get("name", ""), "arguments": step.get("arguments") or {}})
        return "action:" + canonical_json(step)
    return "raw:" + str(step).strip()


def stop_accuracy(pred: Mapping[str, Any] | None, gold: Mapping[str, Any] | None) -> dict[str, Any]:
    """Did the agent stop at the right point?

    Compares call counts and, when both sides carry a boolean finish flag, the
    flag itself. An unknown flag is not a failure.
    """
    pred_rec = _normalise_pred(pred)
    gold_rec = _normalise_pred(gold)
    counts_ok = len(pred_rec["tool_calls"]) == len(gold_rec["tool_calls"])
    pred_finish = pred_rec["finish"]
    gold_finish = gold_rec["finish"]
    flag_ok = True
    if isinstance(pred_finish, bool) and isinstance(gold_finish, bool):
        flag_ok = pred_finish == gold_finish
    hit = counts_ok and flag_ok
    return {"value": 1.0 if hit else 0.0, "n": 1, "correct": int(hit)}


def missing_info_detection(preds: Sequence[Any], golds: Sequence[Any]) -> dict[str, Any]:
    """Sample-level confusion counts for the ``ask_missing_info`` meta tool."""
    tp = fp = fn = tn = 0
    for pred, gold in zip(preds, golds):
        pred_pos = _asks_missing_info(pred)
        gold_pos = _asks_missing_info(gold)
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": tp + fp + fn + tn,
    }


def _asks_missing_info(record: Any) -> bool:
    rec = _normalise_pred(record) if isinstance(record, Mapping) else {"tool_calls": _as_calls(record)}
    for call in rec["tool_calls"]:
        if call["name"] == META_TOOL:
            return True
    return bool(rec.get("ask_missing_info"))


def out_of_scope_accuracy(preds: Sequence[Any], golds: Sequence[Any]) -> dict[str, Any]:
    """Among gold zero-call samples, the fraction where the agent did nothing."""
    correct = 0
    total = 0
    for pred, gold in zip(preds, golds):
        gold_calls = _normalise_pred(gold)["tool_calls"]
        if gold_calls:
            continue
        total += 1
        if not _normalise_pred(pred)["tool_calls"]:
            correct += 1
    return {"value": None if total == 0 else correct / total, "n": total, "correct": correct}


def call_count_delta(pred: Sequence[Any] | Mapping[str, Any] | None, gold: Sequence[Any] | Mapping[str, Any] | None) -> int:
    """Predicted call count minus gold call count (diagnostic)."""
    pred_calls = _normalise_pred(pred)["tool_calls"] if isinstance(pred, Mapping) else _as_calls(pred)
    gold_calls = _normalise_pred(gold)["tool_calls"] if isinstance(gold, Mapping) else _as_calls(gold)
    return len(pred_calls) - len(gold_calls)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _block(value: Any, n: int, **extra: Any) -> dict[str, Any]:
    if n == 0 or value is None:
        return {"available": False, "value": None, "n": n, **extra}
    return {"available": True, "value": value, "n": n, **extra}


def _mean(values: Iterable[float]) -> tuple[float | None, int]:
    items = list(values)
    if not items:
        return None, 0
    return sum(items) / len(items), len(items)


#: Placeholder call name marking a row whose prediction was never produced.
_UNMEASURED = "__unmeasured__"

def decision_emitted(pred: Mapping[str, Any]) -> bool:
    """Did the model actually produce a decision for this step?

    A crashed generator or an output with no complete ``<action>`` block made no
    decision. Scoring it as "nothing to compare" would report a perfect match
    against a do-nothing gold and hide the failure. Every view of a score
    (aggregate and per-record) must apply this gate, or a crash reads as a
    correct "do nothing".
    """
    if str(pred.get("status") or "").lower() in {"error", "timeout", "crash"}:
        return False
    return not pred.get("parse_error")

def aggregate_teacher_forced(preds: list[dict], golds: list[dict]) -> dict[str, Any]:
    """Aggregate all decision metrics over aligned prediction/gold records.

    Each record is a mapping with at least ``tool_calls``; optional keys are
    ``id``, ``finish`` (bool), ``steps`` (per-step decisions), ``status`` and
    ``tool_results``. Metrics with no applicable sample are returned as
    ``{"available": False, ...}`` rather than as a made-up zero.

    A prediction that emitted no decision (``status == "error"`` or a parse
    error) counts as a miss, never as a match: an empty comparison would
    otherwise score a crash as a correct "do nothing".
    """
    pairs = list(zip(preds, golds))
    n = len(pairs)
    emitted = [decision_emitted(p) for p in preds[:n]]
    n_unmeasured = emitted.count(False)

    exact = [
        1.0 if ok and tool_call_exact_match(p.get("tool_calls"), g.get("tool_calls")) else 0.0
        for (p, g), ok in zip(pairs, emitted)
    ]
    exact_value, exact_n = _mean(exact)

    precisions, recalls, f1s = [], [], []
    for (pred, gold), ok in zip(pairs, emitted):
        seq = sequence_precision_recall_f1(pred.get("tool_calls"), gold.get("tool_calls"))
        precisions.append(seq["precision"] if ok else 0.0)
        recalls.append(seq["recall"] if ok else 0.0)
        f1s.append(seq["f1"] if ok else 0.0)
    precision_value, seq_n = _mean(precisions)
    recall_value, _ = _mean(recalls)
    f1_value, _ = _mean(f1s)

    arg_correct = arg_total = 0
    for (pred, gold), ok in zip(pairs, emitted):
        block = argument_exact_match(pred.get("tool_calls"), gold.get("tool_calls"))
        if ok:
            arg_correct += block["correct"]
            arg_total += block["n"]
        elif _as_calls(gold.get("tool_calls")):
            arg_total += 1  # gold needed arguments and none were produced: a miss

    arg_value = None if arg_total == 0 else arg_correct / arg_total

    selection_hits, selection_n = 0, 0
    stop_hits, stop_n = 0, 0
    for (pred, gold), ok in zip(pairs, emitted):
        if not ok:
            selection_n += 1
            stop_n += 1
            continue
        sel = tool_selection_accuracy(pred.get("tool_calls"), gold.get("tool_calls"))
        selection_hits += sel["correct"]
        selection_n += sel["n"]
        stop = stop_accuracy(pred, gold)
        stop_hits += stop["correct"]
        stop_n += stop["n"]

    next_hits, next_n = 0, 0
    for (pred, gold), ok in zip(pairs, emitted):
        block = next_action_accuracy(pred.get("steps"), gold.get("steps"))
        if not ok:
            next_n += 1
            continue
        next_hits += block["correct"]
        next_n += block["n"]

    # Unmeasured rows must not be credited as a correct "do nothing" / "did not ask".
    scored = [
        pred if ok else {**pred, "tool_calls": [{"name": _UNMEASURED, "arguments": {}}]}
        for (pred, _gold), ok in zip(pairs, emitted)
    ]
    missing = missing_info_detection(scored, golds)
    oos = out_of_scope_accuracy(scored, golds)
    deltas = [call_count_delta(pred, gold) for pred, gold in pairs]
    delta_value, delta_n = _mean([float(d) for d in deltas])

    return {
        "n_records": n,
        "n_unmeasured": n_unmeasured,
        "tool_call_exact_match": _block(exact_value, exact_n),
        "sequence_precision": _block(precision_value, seq_n),
        "sequence_recall": _block(recall_value, seq_n),
        "sequence_f1": _block(f1_value, seq_n),
        "argument_exact_match": _block(arg_value, arg_total),
        "tool_selection_accuracy": _block(None if selection_n == 0 else selection_hits / selection_n, selection_n),
        "next_action_accuracy": _block(None if next_n == 0 else next_hits / next_n, next_n),
        "stop_accuracy": _block(None if stop_n == 0 else stop_hits / stop_n, stop_n),
        "missing_info_detection": {
            "available": missing["n"] > 0,
            **{k: v for k, v in missing.items() if k != "n"},
            "n": missing["n"],
        },
        "out_of_scope_accuracy": _block(oos["value"], oos["n"], correct=oos["correct"]),
        "call_count_delta": {
            "available": delta_n > 0,
            "value": None if delta_n == 0 else delta_value,
            "n": delta_n,
            "total_delta": sum(deltas),
        },
        "definitions": METRIC_DEFINITIONS,
    }
