"""Failure taxonomy for LifeMate decisions.

One explicit, ordered rule set turns a wrong prediction into a single bucket so
the report can say *where* the pipeline breaks (ASR vs prompt vs model vs env)
instead of just "score low". Rules are pure functions of the records; nothing is
read from disk and no number is invented.

Buckets (in rule priority order):

1. ``runtime``            — the run itself failed (timeout, crash, deadline).
2. ``ASR``                — the transcript handed to the model was broken or empty.
3. ``prompt_construction``— the prompt was truncated/missing/oversized.
4. ``llm_planning``       — the model emitted nothing usable (invalid action).
5. ``tool_selection``     — a tool name that is unknown or unavailable.
6. ``argument_generation``— correct tool, arguments that fail schema/values.
7. ``tool_execution``     — the environment rejected a call (``ok: false``).
8. ``stopping``           — right prefix, wrong length (stops early / over-calls).
9. ``matched``            — not a failure (kept so counts always sum to n).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .metrics import META_TOOL, _as_calls, canonical_json, canonical_signature

__all__ = [
    "FAILURE_BUCKETS",
    "classify_failure",
    "analyze",
    "summarize_asr_errors",
]

FAILURE_BUCKETS: tuple[str, ...] = (
    "ASR",
    "prompt_construction",
    "llm_planning",
    "tool_selection",
    "argument_generation",
    "tool_execution",
    "stopping",
    "runtime",
)

#: Keys that may carry a user message / profile; never copied into a report.
_PII_KEYS = ("user_profile", "messages", "audio_text", "transcript", "content")


def _safe_id(value: Any) -> str:
    """Sample id only: ids are structural, message bodies are not."""
    text = str(value or "").strip()
    return text[:64]


def _calls(record: Any) -> list[dict[str, Any]]:
    if isinstance(record, Mapping):
        return _as_calls(record.get("tool_calls"))
    return _as_calls(record)


def _status(record: Any) -> str:
    return str((record or {}).get("status", "") or "").lower() if isinstance(record, Mapping) else ""


def _is_bad_transcript(asr_result: Any) -> bool:
    if asr_result is None:
        return False
    if isinstance(asr_result, Mapping):
        if asr_result.get("error"):
            return True
        text = asr_result.get("text")
        return text is not None and not str(text).strip()
    if isinstance(asr_result, str):
        return not asr_result.strip()
    if getattr(asr_result, "error", None):
        return True
    text = getattr(asr_result, "text", None)
    return text is not None and not str(text).strip()


def _env_rejected(record: Mapping[str, Any]) -> bool:
    for result in record.get("tool_results") or []:
        if isinstance(result, Mapping) and result.get("ok") is False:
            return True
        if getattr(result, "ok", None) is False:
            return True
    for error in record.get("errors") or []:
        text = str(error).lower()
        if "environment" in text or "tool execution" in text:
            return True
    return False


def classify_failure(
    pred: Mapping[str, Any] | None,
    gold: Mapping[str, Any] | None,
    *,
    status: str | None = None,
    asr_result: Any = None,
    registry: Any = None,
    prompt_truncated: bool | None = None,
) -> dict[str, Any]:
    """Bucket one decision record. Returns ``{"bucket", "reason"}``.

    ``registry`` is a duck-typed :class:`~lifemate.agent.tool_registry.ToolRegistry`;
    when omitted, tool names are only checked for emptiness (an unknown-tool
    verdict then cannot be made, so the next rule applies).
    """
    pred_rec = dict(pred or {})
    gold_rec = dict(gold or {})
    pred_calls = _calls(pred_rec)
    gold_calls = _calls(gold_rec)
    run_status = str(status or _status(pred_rec) or "").lower()

    # 0. A correct sequence is never a failure, whatever happened upstream — unless
    #    the environment itself rejected one of those correct calls (that is a real
    #    execution failure worth counting).
    if [canonical_signature(c) for c in pred_calls] == [canonical_signature(c) for c in gold_calls] and not _env_rejected(pred_rec):
        finish_pred, finish_gold = pred_rec.get("finish"), gold_rec.get("finish")
        if not (isinstance(finish_pred, bool) and isinstance(finish_gold, bool) and finish_pred != finish_gold):
            return {"bucket": "matched", "reason": "sequences identical after canonicalisation"}

    # 1. runtime
    if run_status in {"timeout", "error", "crash", "deadline_exceeded", "time_budget"}:
        return {"bucket": "runtime", "reason": f"run status {run_status!r}"}
    if run_status == "max_steps":
        return {"bucket": "stopping", "reason": "run stopped at the step budget (max_steps)"}
    if run_status == "repeated_failure":
        return {"bucket": "tool_execution", "reason": "run stopped after repeated tool failures"}
    if "runtime" in str(pred_rec.get("failure", "")).lower():
        return {"bucket": "runtime", "reason": "runtime failure reported by the runner"}

    # 2. ASR
    if _is_bad_transcript(asr_result):
        return {"bucket": "ASR", "reason": "transcript empty or ASR error"}

    # 3. prompt construction
    truncated = pred_rec.get("prompt_truncated") if prompt_truncated is None else prompt_truncated
    if truncated or pred_rec.get("prompt_tokens_exceeded"):
        return {"bucket": "prompt_construction", "reason": "prompt truncated or oversized"}
    if not pred_rec.get("prompt") and pred_rec.get("prompt_missing"):
        return {"bucket": "prompt_construction", "reason": "stored prompt missing"}

    # 4. llm_planning: nothing usable emitted
    if not pred_calls and gold_calls:
        invalid = int(pred_rec.get("invalid_actions") or 0)
        if invalid or pred_rec.get("parse_error") or pred_rec.get("raw_output"):
            return {
                "bucket": "llm_planning",
                "reason": "no executable call emitted (unparseable or empty model output)",
            }
        return {"bucket": "stopping", "reason": "no call emitted for a sample that needs calls"}

    # 5. tool_selection
    predicted_names = [call["name"] for call in pred_calls]
    gold_names = [call["name"] for call in gold_calls]
    unknown = [name for name in predicted_names if not name]
    if unknown:
        return {"bucket": "tool_selection", "reason": "empty tool name in a call"}
    if registry is not None:
        available = set(registry.tool_names()) if hasattr(registry, "tool_names") else set()
        for name in predicted_names:
            if available and name not in available:
                return {"bucket": "tool_selection", "reason": f"tool {name} is not available for this sample"}
    first_mismatch = next(
        (i for i, (p_name, g_name) in enumerate(zip(predicted_names, gold_names)) if p_name != g_name),
        None,
    )
    if first_mismatch is not None:
        return {
            "bucket": "tool_selection",
            "reason": f"step {first_mismatch}: predicted {predicted_names[first_mismatch]} "
            f"but gold is {gold_names[first_mismatch]}",
        }
    if len(predicted_names) > len(gold_names) and predicted_names[: len(gold_names)] == gold_names:
        return {"bucket": "stopping", "reason": "over-calls after the gold sequence ends"}

    # 6. argument_generation
    for index, (pred_call, gold_call) in enumerate(zip(pred_calls, gold_calls)):
        if pred_call["name"] != gold_call["name"]:
            continue
        if canonical_json(pred_call["arguments"]) != canonical_json(gold_call["arguments"]):
            return {
                "bucket": "argument_generation",
                "reason": f"step {index}: arguments differ for {pred_call['name']}",
            }

    # 7. tool_execution
    if _env_rejected(pred_rec):
        return {"bucket": "tool_execution", "reason": "environment returned ok=false"}

    # 8. stopping
    if len(pred_calls) != len(gold_calls):
        return {
            "bucket": "stopping",
            "reason": f"predicted {len(pred_calls)} calls, gold {len(gold_calls)}",
        }
    if [canonical_signature(c) for c in pred_calls] == [canonical_signature(c) for c in gold_calls]:
        # Identical calls but the run stopped for another reason (finish flag or status).
        return {"bucket": "stopping", "reason": f"calls match but the run ended with status {run_status!r}"}
    return {"bucket": "argument_generation", "reason": "sequence differs (fallback)"}


def _dominant(counts: Mapping[str, int]) -> tuple[str, int]:
    ranked = sorted((kv for kv in counts.items() if kv[0] != "matched"), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0] if ranked else ("", 0)


def _recommendations(bucket: str, counts: Mapping[str, int], n_failed: int) -> list[str]:
    """Text-only guidance derived from the dominant bucket (no invented numbers)."""
    total = max(n_failed, 1)
    share = counts.get(bucket, 0) / total
    base = {
        "ASR": [
            "Inspect transcripts for empty/whitespace hypotheses; the audio layer is upstream of every decision.",
            "Compare oracle vs whisper runs on the same ids before touching the agent or prompts.",
        ],
        "prompt_construction": [
            "Check SFT records for truncated prompts and raise training.max_seq_length or shorten tool schemas.",
            "Re-render the prompt for a failing id and diff it against the stored record.",
        ],
        "llm_planning": [
            "Re-run the failing ids with a higher max_new_tokens and inspect the raw outputs.",
            "Add the malformed outputs to the protocol tests; if the model ignores the format, retrain on more SFT records.",
        ],
        "tool_selection": [
            "Verify the per-sample registry only exposes installed apps' tools.",
            "Check the tool catalogue text in the prompt renders the exact dotted tool names.",
        ],
        "argument_generation": [
            "Validate predicted arguments against the tool schema and diff them field by field.",
            "Grow SFT coverage for the tools with the most argument mismatches.",
        ],
        "tool_execution": [
            "Look at env `ok:false` results: wrong ids, missing entities, permission-like errors.",
            "Decide per tool whether a failed call should be retried or the episode finished.",
        ],
        "stopping": [
            "Check whether the agent emits `finish` too early (compare call counts with gold).",
            "Enforce max_steps/time budget defaults and log finish_reason per sample.",
        ],
        "runtime": [
            "Check per-sample elapsed time against agent.time_budget_seconds.",
            "Re-run the timing-out ids alone with debug dumps to find the slow step.",
        ],
    }
    lines = list(base.get(bucket, ["No dominant failure bucket: inspect the per-record classifications."]))
    if share >= 0.5:
        lines.append(f"`{bucket}` accounts for the majority of classified failures.")
    else:
        lines.append("Failures are spread across buckets: fix the largest one first, then re-measure.")
    return lines


def analyze(
    preds: Sequence[Mapping[str, Any]],
    golds: Sequence[Mapping[str, Any]],
    asr_results: Mapping[str, Any] | Sequence[Any] | None = None,
    closed_loop: Mapping[str, Any] | None = None,
    *,
    registries: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate failure buckets, signatures and sanitised example ids."""
    counts = {bucket: 0 for bucket in FAILURE_BUCKETS}
    counts["matched"] = 0
    signatures: dict[str, int] = {}
    examples: dict[str, list[str]] = {bucket: [] for bucket in FAILURE_BUCKETS}
    records: list[dict[str, Any]] = []

    for index, (pred, gold) in enumerate(zip(preds, golds)):
        sample_id = _safe_id(pred.get("id") or gold.get("id") or f"#{index}")
        asr = None
        if isinstance(asr_results, Mapping):
            asr = asr_results.get(sample_id)
        elif isinstance(asr_results, Sequence) and not isinstance(asr_results, (str, bytes)):
            asr = asr_results[index] if index < len(asr_results) else None
        registry = (registries or {}).get(sample_id)
        verdict = classify_failure(pred, gold, asr_result=asr, registry=registry)
        bucket = verdict["bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        if bucket == "matched":
            continue
        signature = f"{bucket}: {verdict['reason']}"
        signatures[signature] = signatures.get(signature, 0) + 1
        if len(examples.setdefault(bucket, [])) < 10:
            examples[bucket].append(sample_id)
        records.append({"id": sample_id, **verdict})

    n_total = sum(counts.values())
    n_failed = n_total - counts.get("matched", 0)
    dominant, dominant_n = _dominant(counts)
    top_signatures = sorted(signatures.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    report: dict[str, Any] = {
        "n_total": n_total,
        "n_failed": n_failed,
        "counts": counts,
        "dominant_bucket": dominant or None,
        "dominant_count": dominant_n,
        "top_signatures": [{"signature": sig, "count": count} for sig, count in top_signatures],
        "examples": {bucket: ids for bucket, ids in examples.items() if ids},
        "recommendations": _recommendations(dominant, counts, n_failed) if dominant else [],
        "per_record": records,
        "bucket_definitions": {
            "ASR": "transcript handed to the model was empty or the ASR backend errored",
            "prompt_construction": "prompt truncated, oversized or missing",
            "llm_planning": "model emitted no parseable action",
            "tool_selection": "unknown/unavailable tool name, or wrong tool at a step",
            "argument_generation": "right tool, wrong arguments",
            "tool_execution": "environment rejected the call (ok=false)",
            "stopping": "right prefix, wrong number of calls (early finish or over-calling)",
            "runtime": "timeout/crash/deadline in the run itself",
        },
    }
    if closed_loop is not None:
        report["closed_loop"] = dict(closed_loop)
    return report


def summarize_asr_errors(asr_pairs: Iterable[Any]) -> dict[str, Any]:
    """WER/CER summary for ``(reference, hypothesis)`` or ``TranscriptResult`` pairs.

    Uses :mod:`lifemate.asr.metrics` when worker C shipped it, else the
    normalization helpers. Returns ``{"available": False}`` for an empty input.
    """
    refs: list[str] = []
    hyps: list[str] = []
    ids: list[str] = []
    n_errors = 0
    for index, pair in enumerate(asr_pairs or []):
        ref = hyp = ""
        pair_id = f"#{index}"
        if isinstance(pair, Mapping):
            ref = str(pair.get("reference", pair.get("ref", "")) or "")
            hyp = str(pair.get("hypothesis", pair.get("hyp", pair.get("text", ""))) or "")
            pair_id = _safe_id(pair.get("id") or pair_id)
            if pair.get("error"):
                n_errors += 1
        elif isinstance(pair, (tuple, list)) and len(pair) == 2:
            ref, hyp = str(pair[0] or ""), str(pair[1] or "")
        elif isinstance(pair, (tuple, list)) and len(pair) >= 3:
            pair_id, ref, hyp = _safe_id(pair[0]), str(pair[1] or ""), str(pair[2] or "")
        else:  # TranscriptResult-like
            ref = str(getattr(pair, "reference", "") or "")
            hyp = str(getattr(pair, "text", "") or "")
            if getattr(pair, "error", None):
                n_errors += 1
        refs.append(ref)
        hyps.append(hyp)
        ids.append(pair_id)

    if not refs:
        return {"available": False, "reason": "no ASR pairs evaluated", "n": 0}

    try:  # worker C's metrics module, when present (expects id/ref/hyp triples)
        from lifemate.asr.metrics import evaluate_asr as _evaluate  # type: ignore

        summary = _evaluate(list(zip(ids, refs, hyps)))
        if isinstance(summary, dict):
            return {"available": True, "n": len(refs), "backend_errors": n_errors, **summary}
    except Exception:
        pass

    from lifemate.asr.normalization import cer, normalize_vi, wer  # pure stdlib

    wers = [wer(normalize_vi(ref), normalize_vi(hyp)) for ref, hyp in zip(refs, hyps)]
    cers = [cer(normalize_vi(ref), normalize_vi(hyp)) for ref, hyp in zip(refs, hyps)]
    empty = sum(1 for hyp in hyps if not hyp.strip())
    return {
        "available": True,
        "n": len(refs),
        "backend_errors": n_errors,
        "empty_hypotheses": empty,
        "wer": sum(wers) / len(wers),
        "cer": sum(cers) / len(cers),
        "source": "lifemate.asr.normalization (fallback; lifemate.asr.metrics unavailable)",
    }


# Keep the meta-tool name discoverable for report code without re-importing.
ASK_MISSING_INFO = META_TOOL
