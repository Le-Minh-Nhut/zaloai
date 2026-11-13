"""Gold-trajectory EDA: call counts, ordering, tool usage, challenge taxonomy.

Only *counts and structures* are derived from labels; ``label.final_state`` is
inspected solely to count how many apps changed and is never treated as
model-visible input (``Label.is_model_input`` is False by construction).

No causal claims: groupings are descriptive.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

__all__ = [
    "analyze_trajectories",
    "alignment_report",
    "classify_challenge",
    "call_kind",
    "REQUIRED_CHALLENGES",
    "TEMPORAL_PATTERN",
]

MISSING_INFO_TOOL = "sys.ask_missing_info"
REQUIRED_CHALLENGES = (
    "missing_information",
    "fuzzy_match",
    "reference_lookup",
    "long_history",
    "multi_app",
    "temporal_expression",
    "ambiguous_entity",
)
LONG_HISTORY_TURNS = 8
MULTI_APP_MIN = 2

#: Vietnamese/English temporal words typical in spoken scheduling requests.
#: ``\b`` is unreliable for Vietnamese, so alternatives are listed explicitly
#: and matched case-insensitively against the NFC-lowercased transcript.
TEMPORAL_PATTERN = re.compile(
    r"hôm nay|hôm qua|ngày mai|ngày kia|tuần sau|tuần trước|tuần này|"
    r"tháng sau|tháng trước|tháng này|năm sau|năm ngoái|"
    r"sáng nay|chiều nay|tối nay|trưa nay|sáng mai|chiều mai|tối mai|"
    r"thứ\s*(hai|ba|tư|năm|sáu|bảy|chủ nhật)|chủ nhật|"
    r"đầu tuần|cuối tuần|giờ|phút|tiếng nữa|lát nữa|"
    r"sắp tới|vừa rồi|gần đây|"
    r"\d{1,2}\s*(giờ|h|phút|p)\b|\d{1,2}/\d{1,2}|"
    r"tomorrow|today|tonight|next week|last week",
    re.IGNORECASE,
)

#: Quoted spans / capitalized proper-noun candidates ⇒ "ambiguous entity" heuristic.
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{1,60})[\"'“”‘’]")
_KINSHIP_RE = re.compile(
    r"\b(anh|chị|em|cô|chú|bác|ông|bà|bạn|ba|mẹ|vợ|chồng)\s+"
    r"([A-ZÀ-Ỹ][a-zà-ỹ]+)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _history(sample: Any) -> list[Any]:
    return list(getattr(sample, "history", []) or [])


def _label(sample: Any) -> Any:
    return getattr(sample, "label", None)


def _calls(sample: Any) -> list[Any]:
    label = _label(sample)
    return list(getattr(label, "tool_calls", []) or []) if label is not None else []


def _results(sample: Any) -> list[Any]:
    label = _label(sample)
    return list(getattr(label, "tool_results", []) or []) if label is not None else []


def _call_args(call: Any) -> dict:
    args = getattr(call, "arguments", None)
    return dict(args) if isinstance(args, dict) else {}


def _app_of(tool_name: str) -> str:
    return tool_name.split(".")[0] if "." in tool_name else ""


def _deep_count(value: Any) -> int:
    """Count nested containers/values (argument complexity proxy)."""
    if isinstance(value, dict):
        return sum(1 + _deep_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(1 + _deep_count(v) for v in value)
    return 0


def _history_call_names(sample: Any) -> list[str]:
    """Tool names already executed inside the visible conversation history."""
    names: list[str] = []
    for turn in _history(sample):
        payload = getattr(turn, "tool_call", None)
        if not isinstance(payload, dict):
            continue
        batch = payload.get("tool_calls")
        items = batch if isinstance(batch, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.append(str(name))
    return names


def _catalog_kinds(catalog: Any) -> dict[str, str]:
    tools = getattr(catalog, "tools", None)
    if not hasattr(tools, "items"):
        return {}
    return {name: str(getattr(spec, "kind", "unknown")) for name, spec in tools.items()}


def _catalog_ref_sources(catalog: Any) -> dict[str, set[str]]:
    """Map ``tool name -> {referenced tool/ref targets}`` from x-ref annotations."""
    out: dict[str, set[str]] = defaultdict(set)
    tools = getattr(catalog, "tools", None)
    if not hasattr(tools, "items"):
        return {}
    for name, spec in tools.items():
        props = getattr(spec, "properties", None)
        if not hasattr(props, "items"):
            continue
        for _param, schema in props.items():
            if not hasattr(schema, "get"):
                continue
            for key in ("x-ref", "x_ref", "ref", "x-match", "x_match"):
                target = schema.get(key)
                if isinstance(target, str) and target:
                    out[name].add(target)
                    break
    return dict(out)


def call_kind(name: str, kinds: dict[str, str]) -> str:
    """``read`` / ``write`` / ``meta`` / ``unknown`` for one tool name."""
    kind = kinds.get(name)
    if kind in ("read", "write", "meta"):
        return kind
    if name == MISSING_INFO_TOOL or "ask_missing" in name:
        return "meta"
    return "unknown"


# ---------------------------------------------------------------------------
# Challenge classification
# ---------------------------------------------------------------------------
def classify_challenge(
    sample: Any,
    *,
    kinds: dict[str, str] | None = None,
    ref_sources: dict[str, set[str]] | None = None,
) -> list[str]:
    """Heuristic challenge tags for one sample (documented, non-causal)."""
    kinds = kinds or {}
    ref_sources = ref_sources or {}
    tags: list[str] = []
    text = str(getattr(sample, "audio_text", "") or "")
    lowered = text.lower()
    calls = _calls(sample)
    history = _history(sample)

    names = [str(getattr(c, "name", "")) for c in calls]
    if MISSING_INFO_TOOL in names or any("ask_missing" in n for n in names):
        tags.append("missing_information")

    apps_used = {_app_of(n) for n in names if _app_of(n)}
    if len(apps_used) >= MULTI_APP_MIN:
        tags.append("multi_app")

    if len(history) >= LONG_HISTORY_TURNS:
        tags.append("long_history")

    has_ref_arg = False
    for call in calls:
        args = _call_args(call)
        for value in _iter_strings(args):
            # A reference-style value: id-ish or a tool-qualified token.
            if (
                re.fullmatch(r"[a-z_]+\.[a-z_]+", value)
                or re.fullmatch(r"[A-Z]{2,}-\d+", value)
                or re.fullmatch(r"\d{4,}", value)
            ):
                has_ref_arg = True
                break
        if name_ref_key(args):
            has_ref_arg = True
        if ref_sources.get(str(getattr(call, "name", ""))):
            has_ref_arg = True
    if has_ref_arg:
        tags.append("reference_lookup")
        tags.append("fuzzy_match")

    if TEMPORAL_PATTERN.search(lowered):
        tags.append("temporal_expression")

    if _QUOTED_RE.search(text) or _KINSHIP_RE.search(text):
        tags.append("ambiguous_entity")

    return [t for t in REQUIRED_CHALLENGES if t in tags]


def name_ref_key(args: dict) -> bool:
    """Heuristic: an argument whose *name* implies a cross-record reference."""
    keys = " ".join(str(k).lower() for k in args)
    return bool(re.search(r"(^|_)(id|ref|key|uuid|code)($|_)", keys))


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


# ---------------------------------------------------------------------------
# Alignment report (data visibility limitations)
# ---------------------------------------------------------------------------
def alignment_report(samples: Iterable[Any]) -> dict[str, Any]:
    """Structural label/history inconsistencies that limit what can be learned.

    These are *dataset visibility limitations*, not model errors: a mismatched
    call/result list means the gold trajectory cannot be replayed verbatim.
    """
    samples = list(samples or [])
    mismatched: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    results_before_calls: list[dict[str, Any]] = []
    final_state_only: list[str] = []
    unnamed_calls = 0
    unnamed_results = 0

    for sample in samples:
        sid = str(getattr(sample, "id", ""))
        label = _label(sample)
        if label is None:
            missing_labels.append(sid)
            continue
        calls = _calls(sample)
        results = _results(sample)
        if not calls and not results:
            empty_labels.append(sid)
        if len(calls) != len(results):
            mismatched.append(
                {
                    "id": sid,
                    "n_calls": len(calls),
                    "n_results": len(results),
                    "delta": len(calls) - len(results),
                }
            )
        unnamed_calls += sum(1 for c in calls if not str(getattr(c, "name", "")))
        unnamed_results += sum(1 for r in results if getattr(r, "ok", None) is None)
        if not calls and getattr(label, "final_state", None):
            final_state_only.append(sid)

        # Result-before-call anomaly inside the visible history.
        seen_call = False
        for index, turn in enumerate(_history(sample)):
            if getattr(turn, "tool_call", None):
                seen_call = True
            elif getattr(turn, "tool_result", None) and not seen_call:
                results_before_calls.append(
                    {"id": sid, "turn_index": index, "kind": "tool_result_without_prior_call"}
                )
                break

    n = len(samples)
    return {
        "n_samples": n,
        "n_missing_labels": len(missing_labels),
        "missing_labels": missing_labels[:50],
        "n_empty_labels": len(empty_labels),
        "empty_labels": empty_labels[:50],
        "n_call_result_mismatch": len(mismatched),
        "call_result_mismatches": mismatched[:50],
        "n_results_before_calls": len(results_before_calls),
        "results_before_calls": results_before_calls[:50],
        "n_unnamed_calls": unnamed_calls,
        "n_results_without_status": unnamed_results,
        "n_final_state_only_labels": len(final_state_only),
        "final_state_only": final_state_only[:50],
        "has_limitations": bool(
            missing_labels
            or empty_labels
            or mismatched
            or results_before_calls
            or unnamed_calls
            or final_state_only
        ),
        "note": (
            "Counts describe label/history structure only. No content of user "
            "profiles or transcripts is reproduced here."
        ),
    }


# ---------------------------------------------------------------------------
# Main trajectory analysis
# ---------------------------------------------------------------------------
def analyze_trajectories(samples: Iterable[Any], catalog: Any = None) -> dict[str, Any]:
    """Trajectory statistics over gold labels."""
    samples = list(samples or [])
    kinds = _catalog_kinds(catalog)
    ref_sources = _catalog_ref_sources(catalog)

    call_counts: list[int] = []
    zero = single = multi = 0
    tool_freq: Counter[str] = Counter()
    app_freq: Counter[str] = Counter()
    transition_freq: dict[int, Counter[tuple[str, ...]]] = {1: Counter(), 2: Counter(), 3: Counter()}
    n_args: list[int] = []
    n_nested: list[int] = []
    missing_info_count = 0
    out_of_scope: list[str] = []
    failed_results = 0
    total_results = 0
    apps_changed_counts: list[int] = []
    final_state_apps: Counter[str] = Counter()
    read_before_write_ok = 0
    read_before_write_violations: list[dict[str, Any]] = []
    samples_with_write = 0
    challenge_counts: Counter[str] = Counter()
    challenge_examples: dict[str, list[str]] = defaultdict(list)
    by_length_duration: dict[str, dict[str, Any]] = {}
    by_length_apps: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    app_counts: list[int] = []
    lengths: list[int] = []

    for sample in samples:
        sid = str(getattr(sample, "id", ""))
        calls = _calls(sample)
        results = _results(sample)
        names = [str(getattr(c, "name", "")) for c in calls if str(getattr(c, "name", ""))]
        n_calls = len(calls)
        call_counts.append(n_calls)
        lengths.append(n_calls)
        if n_calls == 0:
            zero += 1
        elif n_calls == 1:
            single += 1
        else:
            multi += 1

        tool_freq.update(names)
        app_freq.update(a for a in (_app_of(n) for n in names) if a)

        for order in (1, 2, 3):
            for i in range(len(names) - order + 1):
                transition_freq[order][tuple(names[i : i + order])] += 1

        for call in calls:
            args = _call_args(call)
            n_args.append(len(args))
            n_nested.append(_deep_count(args))

        has_missing_info = any("ask_missing" in n for n in names)
        if has_missing_info:
            missing_info_count += 1
        if n_calls == 0 and not has_missing_info:
            out_of_scope.append(sid)

        total_results += len(results)
        failed_results += sum(1 for r in results if getattr(r, "ok", None) is False)

        # read-before-write: a read (or an x-ref source) must precede a write.
        # Reads already executed in the visible history count as preceding.
        history_reads = _history_call_names(sample)
        kinds_here = [
            call_kind(str(getattr(c, "name", "")), kinds) for c in calls
        ]
        if "write" in kinds_here:
            samples_with_write += 1
            ok_here = True
            for i, kind in enumerate(kinds_here):
                if kind != "write":
                    continue
                write_name = str(getattr(calls[i], "name", ""))
                write_app = _app_of(write_name)
                preceding = history_reads + [
                    str(getattr(p, "name", "")) for p in calls[:i]
                ]
                targets = ref_sources.get(write_name, set())
                satisfied = any(
                    call_kind(name, kinds) == "read"
                    and (
                        _app_of(name) == write_app
                        or name in targets
                        or any(t == name for t in targets)
                    )
                    for name in preceding
                )
                if not satisfied:
                    ok_here = False
                    read_before_write_violations.append(
                        {
                            "id": sid,
                            "write_index": i,
                            "write_tool": write_name,
                            "preceding_tools": list(preceding),
                        }
                    )
                    break
            if ok_here:
                read_before_write_ok += 1

        label = _label(sample)
        final_state = getattr(label, "final_state", None) if label is not None else None
        if isinstance(final_state, dict):
            apps_changed_counts.append(len(final_state))
            for key in final_state:
                final_state_apps[str(key)] += 1

        context = getattr(sample, "context", None)
        installed = list(getattr(context, "installed_apps", []) or [])
        app_counts.append(len(installed))

        audio = getattr(sample, "audio_path", None)
        duration = _wav_duration(audio)
        if duration is not None:
            durations.append(duration)

        tags = classify_challenge(sample, kinds=kinds, ref_sources=ref_sources)
        for tag in tags:
            challenge_counts[tag] += 1
            if len(challenge_examples[tag]) < 5:
                challenge_examples[tag].append(sid)

    # Grouped stats: trajectory length vs audio duration / installed apps.
    by_length_duration = _grouped(durations, lengths)
    by_length_apps = _grouped(app_counts, lengths)

    from .dataset_analysis import describe  # local import avoids a heavy cycle

    n = len(samples)
    transition_dict: dict[str, Any] = {}
    top_chains: list[dict[str, Any]] = []
    for order, counter in transition_freq.items():
        seqs = [
            {"sequence": list(seq), "count": c}
            for seq, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
        ]
        transition_dict[str(order)] = {"top": seqs, "n_distinct": len(counter)}
        if order >= 2:
            top_chains.extend(seqs[:10])
    top_chains.sort(key=lambda item: (-item["count"], len(item["sequence"])))

    return {
        "n_samples": n,
        "n_gold_calls": {"values": call_counts, "stats": describe(call_counts)},
        "zero_calls": zero,
        "single_call": single,
        "multi_call": multi,
        "fraction_zero_calls": (zero / n) if n else None,
        "tool_usage": [
            {"tool": t, "count": c, "fraction": (c / n) if n else None}
            for t, c in tool_freq.most_common()
        ],
        "app_usage": [{"app": a, "count": c} for a, c in app_freq.most_common()],
        "transitions": transition_dict,
        "top_chains": top_chains[:15],
        "argument_complexity": {
            "n_args": describe(n_args),
            "n_nested_values": describe(n_nested),
            "max_args": max(n_args) if n_args else 0,
        },
        "missing_info": {
            "count": missing_info_count,
            "fraction": (missing_info_count / n) if n else None,
            "tool": MISSING_INFO_TOOL,
        },
        "out_of_scope": {
            "count": len(out_of_scope),
            "ids": out_of_scope[:50],
            "definition": "zero gold tool calls AND no sys.ask_missing_info call",
        },
        "failed_tool_results": {
            "count": failed_results,
            "total_results": total_results,
            "fraction": (failed_results / total_results) if total_results else None,
        },
        "read_before_write": {
            "samples_with_write": samples_with_write,
            "samples_satisfied": read_before_write_ok,
            "samples_unsatisfied": samples_with_write - read_before_write_ok,
            "unsatisfied": read_before_write_violations[:50],
            "definition": (
                "a write call is satisfied when an earlier call (in the visible history or "
                "preceding gold calls) is a read of the same app or an x-ref source tool "
                "of the write"
            ),
            "note": (
                "unsatisfied does NOT imply a data defect: a single-shot request "
                "(e.g. 'set an alarm') legitimately writes without reading first"
            ),
        },
        "final_state": {
            "n_labels_with_final_state": len(apps_changed_counts),
            "apps_changed_stats": describe(apps_changed_counts),
            "apps_changed_distribution": {
                str(k): v for k, v in sorted(Counter(apps_changed_counts).items())
            },
            "apps_named": [{"app": a, "count": c} for a, c in final_state_apps.most_common()],
            "note": "counts only; final_state is never model input",
        },
        "length_vs_audio_duration": by_length_duration,
        "length_vs_installed_apps": by_length_apps,
        "challenges": {
            "counts": {tag: challenge_counts.get(tag, 0) for tag in REQUIRED_CHALLENGES},
            "fractions": {
                tag: ((challenge_counts.get(tag, 0) / n) if n else None)
                for tag in REQUIRED_CHALLENGES
            },
            "examples": {tag: list(challenge_examples.get(tag, [])) for tag in REQUIRED_CHALLENGES},
            "definitions": _challenge_definitions(),
        },
        "alignment": alignment_report(samples),
    }


def _wav_duration(path: Any) -> float | None:
    if path is None:
        return None
    from .audio_analysis import analyze_wav  # local import: only when needed

    result = analyze_wav(path)
    return result.get("duration_s") if result.get("ok") else None


def _grouped(values: list[float | int], lengths: list[int]) -> dict[str, Any]:
    """Group ``values`` (per sample) by trajectory-length bucket."""
    from .dataset_analysis import describe

    buckets: dict[str, list[float]] = defaultdict(list)
    for value, length in zip(values, lengths):
        if length == 0:
            key = "0"
        elif length == 1:
            key = "1"
        else:
            key = "2+"
        buckets[key].append(float(value))
    return {
        "buckets": {k: describe(v) for k, v in sorted(buckets.items())},
        "definition": "bucket = number of gold tool calls; stats are per-sample means",
        "caveat": "descriptive grouping only; no causal relationship is implied",
    }


def _challenge_definitions() -> dict[str, str]:
    return {
        "missing_information": f"gold trajectory contains {MISSING_INFO_TOOL}",
        "fuzzy_match": "an argument carries a reference-style value (id/number/tool token)",
        "reference_lookup": "an x-ref/x-match annotated argument or a reference-named argument",
        "long_history": f"visible history has >= {LONG_HISTORY_TURNS} turns",
        "multi_app": f"gold trajectory touches >= {MULTI_APP_MIN} distinct apps",
        "temporal_expression": "transcript matches a Vietnamese/English time expression",
        "ambiguous_entity": "transcript contains quoted spans or kinship+name patterns",
    }
