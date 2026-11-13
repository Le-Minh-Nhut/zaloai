"""Report assembly: summary.json, report.md, charts/, tables/.

Deterministic by construction: no timestamps, no absolute-path-dependent
ordering, no randomness. ``--seed`` only affects sampling upstream.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["build_report", "SYNTHETIC_BANNER", "write_csv"]

#: Exact two lines required at the very top of a synthetic-data report.
SYNTHETIC_BANNER = (
    "> **SYNTHETIC TEST DATA**\n"
    "> **NOT OFFICIAL COMPETITION STATISTICS**"
)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], columns: list[str] | None = None) -> int:
    """Write rows to CSV; returns the row count. Empty rows still write a header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(r) for r in rows or []]
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _stats_line(name: str, stats: dict[str, Any]) -> str:
    if not stats or not stats.get("count"):
        return f"- {name}: no data"
    return (
        f"- {name}: n={stats.get('count')}, mean={_fmt(stats.get('mean'))}, "
        f"median={_fmt(stats.get('median'))}, min={_fmt(stats.get('min'))}, "
        f"max={_fmt(stats.get('max'))}, p95={_fmt(stats.get('p95'))}"
    )


# ---------------------------------------------------------------------------
# Markdown sections
# ---------------------------------------------------------------------------
def _section_header(bundle: dict[str, Any], split: str, data_root: str, synthetic: bool) -> list[str]:
    ds = bundle.get("dataset") or {}
    audio = bundle.get("audio") or {}
    tools = bundle.get("tools") or {}
    lines = [f"# EDA report — split `{split}`", ""]
    lines.append(f"- data root: `{data_root}`")
    lines.append(f"- samples: {ds.get('n_samples', 0)}")
    lines.append(f"- audio files analyzed: {audio.get('n_files_analyzed', 0)}")
    lines.append(
        f"- tool catalog: "
        + (
            f"{tools.get('n_tools', 0)} tools / {tools.get('n_apps', 0)} apps"
            if tools.get("available")
            else "absent for this split"
        )
    )
    lines.append("")
    return lines


def _section_quality(bundle: dict[str, Any]) -> list[str]:
    ds = bundle.get("dataset") or {}
    audio = bundle.get("audio") or {}
    tools = bundle.get("tools") or {}
    lines = ["## Data quality issues", ""]
    issues: list[str] = []
    dup = ds.get("duplicate_ids") or {}
    if dup:
        issues.append(f"{len(dup)} duplicate sample id(s): {', '.join(list(dup)[:10])}")
    malformed = (ds.get("malformed_records") or {})
    if malformed.get("n_malformed"):
        issues.append(f"{malformed['n_malformed']} malformed JSONL record(s) skipped")
    if audio.get("n_files_failed"):
        issues.append(f"{audio['n_files_failed']} unreadable audio file(s)")
    if audio.get("n_missing_audio_refs"):
        issues.append(f"{audio['n_missing_audio_refs']} sample(s) without an audio reference")
    if audio.get("n_unexpected_sample_rate"):
        issues.append(
            f"{audio['n_unexpected_sample_rate']} file(s) not at 16 kHz — resampling required"
        )
    for flag, count in sorted((audio.get("flag_counts") or {}).items()):
        if flag in ("corrupt", "silent", "clipping", "very_short", "very_long", "empty"):
            issues.append(f"audio flag `{flag}`: {count} file(s)")
    if not tools.get("available"):
        issues.append("no tool catalog discovered for this split")
    missing = ds.get("missing_fields") or {}
    for key, count in sorted(missing.items()):
        issues.append(f"missing `{key}` in {count} sample(s)")
    lines.extend(f"- {item}" for item in issues) if issues else lines.append(
        "- no structural issues detected by the automated checks"
    )
    lines.append("")
    return lines


def _section_missing_fields(bundle: dict[str, Any]) -> list[str]:
    ds = bundle.get("dataset") or {}
    missing = ds.get("missing_fields") or {}
    inventory = ds.get("raw_field_inventory") or {}
    lines = ["## Missing fields", ""]
    if missing:
        lines.append("| field | samples missing |")
        lines.append("|---|---|")
        for key, count in sorted(missing.items()):
            lines.append(f"| `{key}` | {count} |")
    else:
        lines.append("- no tracked field is missing")
    lines.append("")
    if inventory.get("available"):
        always_missing = inventory.get("always_missing") or []
        lines.append(
            f"Raw JSONL inventory over {inventory.get('n_records', 0)} record(s); "
            f"never present: {', '.join(f'`{k}`' for k in always_missing) or 'none'}."
        )
        seen = list((inventory.get("fields") or {}).items())[:20]
        if seen:
            lines.append("")
            lines.append("Most common top-level keys: " + ", ".join(f"`{k}` ({v['count']})" for k, v in seen) + ".")
    else:
        lines.append(f"Raw JSONL inventory: unavailable ({inventory.get('reason', 'not provided')})")
    lines.append("")
    return lines


def _section_token_stats(bundle: dict[str, Any]) -> list[str]:
    tokens = (bundle.get("dataset") or {}).get("tokens") or {}
    lines = ["## Token statistics", ""]
    if tokens.get("tokenizer") == "available":
        stats = tokens.get("stats") or {}
        lines.append(f"- tokenizer: `{tokens.get('name') or 'injected'}`")
        lines.append(_stats_line("tokens per transcript", stats))
    else:
        lines.append("Token statistics: unavailable (tokenizer not loaded)")
        if tokens.get("reason"):
            lines.append(f"- reason: {tokens['reason']}")
    lines.append("")
    return lines


def _section_audio(bundle: dict[str, Any]) -> list[str]:
    audio = bundle.get("audio") or {}
    lines = ["## Audio", ""]
    lines.append(_stats_line("duration (s)", audio.get("duration_stats") or {}))
    lines.append(
        f"- total duration: {_fmt(audio.get('total_duration_hours'), 3)} h "
        f"({_fmt(audio.get('total_duration_seconds'), 1)} s)"
    )
    lines.append(f"- sample-rate distribution: {audio.get('sample_rate_distribution') or 'n/a'}")
    lines.append(f"- bit depth distribution: {audio.get('bit_depth_distribution') or 'n/a'}")
    lines.append(f"- channels distribution: {audio.get('channels_distribution') or 'n/a'}")
    lines.append(f"- silence threshold: |sample| <= {audio.get('silence_threshold')} (int16 scale)")
    flag_counts = audio.get("flag_counts") or {}
    lines.append(f"- flags: {flag_counts or 'none'}")
    corr = audio.get("duration_vs_transcript_words") or {}
    if corr.get("available"):
        lines.append(
            f"- duration vs transcript words: r={_fmt(corr.get('r'))} (n={corr.get('n')}); "
            "descriptive only"
        )
    else:
        lines.append(f"- duration vs transcript words: not computed ({corr.get('reason', 'n/a')})")
    lines.append("")
    return lines


def _section_tools(bundle: dict[str, Any]) -> list[str]:
    tools = bundle.get("tools") or {}
    lines = ["## Tool catalog", ""]
    if not tools.get("available"):
        lines.append(f"- catalog unavailable: {tools.get('reason', 'not provided')}")
        lines.append("")
        return lines
    lines.append(f"- apps: {tools.get('n_apps')}, tools: {tools.get('n_tools')}")
    lines.append(f"- kind distribution: {tools.get('kind_distribution')}")
    lines.append(f"- tools per app: {dict(list((tools.get('tools_per_app') or {}).items())[:15])}")
    lines.append(f"- parameter type frequency: {tools.get('parameter_type_frequency')}")
    xref = tools.get("x_ref") or {}
    lines.append(
        f"- x-ref annotations: {xref.get('n_refs')} "
        f"(unresolved targets: {xref.get('n_unresolved_targets', 0)})"
    )
    one_of = tools.get("x_one_of") or {}
    lines.append(f"- x-one-of tools: {one_of.get('n_tools', 0)}")
    lines.append(f"- meta tool present: {_fmt(tools.get('has_meta_tool'))}")
    graph = tools.get("dependency_graph") or {}
    lines.append(f"- dependency edges: {graph.get('n_edges', 0)}")
    lines.append("")
    if graph.get("mermaid"):
        lines.append("### Dependency graph")
        lines.append("")
        lines.append(graph["mermaid"])
        lines.append("")
    return lines


def _section_trajectory(bundle: dict[str, Any]) -> list[str]:
    traj = bundle.get("trajectory") or {}
    lines = ["## Gold trajectories", ""]
    if not traj.get("n_samples"):
        lines.append("- no labeled samples")
        lines.append("")
        return lines
    n = traj["n_samples"]
    lines.append(
        f"- tool calls: zero={traj.get('zero_calls')}, single={traj.get('single_call')}, "
        f"multi={traj.get('multi_call')} (n={n})"
    )
    lines.append(_stats_line("tool calls per sample", (traj.get("n_gold_calls") or {}).get("stats") or {}))
    lines.append(f"- missing-info requests: {traj.get('missing_info', {}).get('count')} "
                 f"({_pct(traj.get('missing_info', {}).get('fraction'))})")
    lines.append(f"- out-of-scope (0 calls, no ask): {traj.get('out_of_scope', {}).get('count')}")
    failed = traj.get("failed_tool_results") or {}
    lines.append(
        f"- failed gold tool results: {failed.get('count')} / {failed.get('total_results')} "
        f"({_pct(failed.get('fraction'))})"
    )
    rbw = traj.get("read_before_write") or {}
    lines.append(
        f"- read-before-write: {rbw.get('samples_satisfied')}/{rbw.get('samples_with_write')} "
        f"trajectories with a write are satisfied"
    )
    lines.append(f"- argument complexity (n args): {_fmt((traj.get('argument_complexity') or {}).get('n_args', {}).get('mean'))} mean")
    lines.append("")
    usage = traj.get("tool_usage") or []
    if usage:
        lines.append("| tool | count | share of samples |")
        lines.append("|---|---|---|")
        for row in usage[:20]:
            lines.append(f"| `{row['tool']}` | {row['count']} | {_pct(row.get('fraction'))} |")
        lines.append("")
    chains = traj.get("top_chains") or []
    if chains:
        lines.append("Top chains (length >= 2):")
        lines.append("")
        for row in chains[:10]:
            lines.append(f"- `{' > '.join(str(s) for s in row['sequence'])}` x{row['count']}")
        lines.append("")
    return lines


def _section_challenges(bundle: dict[str, Any]) -> list[str]:
    traj = bundle.get("trajectory") or {}
    challenges = traj.get("challenges") or {}
    lines = ["## Challenge distribution (heuristic tags)", ""]
    counts = challenges.get("counts") or {}
    if not counts:
        lines.append("- no samples analyzed")
        lines.append("")
        return lines
    lines.append("| challenge | samples | share | definition |")
    lines.append("|---|---|---|---|")
    defs = challenges.get("definitions") or {}
    for tag, count in counts.items():
        lines.append(
            f"| `{tag}` | {count} | {_pct((challenges.get('fractions') or {}).get(tag))} | "
            f"{defs.get(tag, '')} |"
        )
    lines.append("")
    lines.append("Tags are heuristics over labels and transcripts; a sample may carry several.")
    lines.append("")
    return lines


def _section_imbalance(bundle: dict[str, Any]) -> list[str]:
    traj = bundle.get("trajectory") or {}
    ds = bundle.get("dataset") or {}
    lines = ["## Imbalance", ""]
    usage = traj.get("tool_usage") or []
    if usage:
        total = sum(r["count"] for r in usage) or 1
        top = usage[0]
        lines.append(
            f"- most frequent tool `{top['tool']}` accounts for {_pct(top['count'] / total)} of all calls"
        )
        empty_or_rare = [r["tool"] for r in usage if r["count"] <= max(1, 0.01 * total)]
        if empty_or_rare:
            lines.append(f"- rare tools (<=1% of calls): {', '.join(f'`{t}`' for t in empty_or_rare[:15])}")
    else:
        lines.append("- tool usage unavailable")
    zero = traj.get("zero_calls")
    n = traj.get("n_samples") or 0
    if n and zero is not None:
        lines.append(f"- zero-call (out-of-scope style) samples: {_pct(zero / n)}")
    apps = ds.get("installed_apps") or {}
    dist = apps.get("count_distribution") or {}
    if dist:
        lines.append(f"- installed-app count distribution: {dist}")
    lines.append("")
    return lines


def _section_visibility(bundle: dict[str, Any]) -> list[str]:
    alignment = (bundle.get("trajectory") or {}).get("alignment") or {}
    lines = ["## Data visibility limitations", ""]
    if not alignment:
        lines.append("- alignment analysis unavailable (no labels loaded)")
        lines.append("")
        return lines
    lines.append(f"- samples inspected: {alignment.get('n_samples', 0)}")
    lines.append(f"- samples without a label block: {alignment.get('n_missing_labels', 0)}")
    lines.append(f"- samples with an empty label block: {alignment.get('n_empty_labels', 0)}")
    mismatched = alignment.get("n_call_result_mismatch", 0)
    lines.append(
        f"- call/result count mismatches: {mismatched}"
        + (" (gold trajectory cannot be replayed verbatim)" if mismatched else "")
    )
    lines.append(
        f"- tool results appearing before any tool call: {alignment.get('n_results_before_calls', 0)}"
    )
    lines.append(f"- calls with no tool name: {alignment.get('n_unnamed_calls', 0)}")
    lines.append(
        f"- labels carrying only final_state (no calls): {alignment.get('n_final_state_only_labels', 0)}"
    )
    mismatches = alignment.get("call_result_mismatches") or []
    if mismatches:
        lines.append("")
        lines.append("| sample | calls | results | delta |")
        lines.append("|---|---|---|---|")
        for row in mismatches[:20]:
            lines.append(f"| `{row['id']}` | {row['n_calls']} | {row['n_results']} | {row['delta']} |")
    lines.append("")
    lines.append(alignment.get("note", ""))
    lines.append("")
    return lines


def _section_asr(bundle: dict[str, Any]) -> list[str]:
    asr = bundle.get("asr") or {}
    lines = ["## ASR evaluation", ""]
    if not asr.get("available"):
        lines.append("ASR evaluation framework present; no predictions available ⇒ WER/CER not reported")
        if asr.get("reason"):
            lines.append(f"- reason: {asr['reason']}")
        lines.append("")
        return lines

    norm = asr.get("normalized") or {}
    raw = asr.get("raw") or {}
    lines.append(f"ASR evaluation: {asr.get('backend')}, n={asr.get('n')}")
    if asr.get("source") == "file" or asr.get("predictions_file"):
        lines.append("")
        lines.append(
            "> **Caveat**: predictions were supplied from a file, not produced by a benchmark run. "
            "These are file-supplied predictions (not a benchmark result); they are only as good "
            "as the file's references."
        )
    lines.append("")
    lines.append("| Metric | Raw | Normalized (`for_wer`) |")
    lines.append("|---|---|---|")
    for label, key in (("WER (mean)", "wer_mean"), ("CER (mean)", "cer_mean")):
        lines.append(f"| {label} | {_fmt(raw.get(key), 4)} | {_fmt(norm.get(key), 4)} |")
    for label, key in (("WER (median)", "wer_median"), ("WER (p95)", "wer_p95"),
                       ("CER (median)", "cer_median"), ("CER (p95)", "cer_p95")):
        lines.append(f"| {label} | — | {_fmt(norm.get(key), 4)} |")
    lines.append(f"| Sentence exact match | — | {_pct(asr.get('sentence_exact_match'))} |")
    lines.append(f"| Empty hypotheses | {asr.get('empty_hypotheses', 0)} | — |")
    lines.append("")

    latency = asr.get("latency_s") or {}
    if latency.get("count"):
        lines.append(
            f"- latency: n={latency.get('count')}, mean={_fmt(latency.get('mean'))}s, "
            f"p50={_fmt(latency.get('p50'))}s, p95={_fmt(latency.get('p95'))}s"
        )
    else:
        lines.append("- latency: no `latency_s` values in the predictions file")
    lines.append(f"- rows skipped as malformed: {asr.get('n_skipped_rows', 0)}")
    lines.append(f"- split samples without a prediction: {asr.get('n_missing', 0)}")
    lines.append(f"- predictions for ids outside the split: {asr.get('n_extra', 0)}")
    lines.append("")

    critical = asr.get("critical_tokens") or {}
    classes = critical.get("classes") or {}
    lines.append(f"### Critical token diagnostics (n={critical.get('n_pairs', 0)})")
    lines.append("")
    if not classes:
        lines.append("Critical token diagnostics unavailable for this run.")
        lines.append("")
        return lines
    lines.append("| Class | Ref tokens | Hyp tokens | Exact set match | Substitution | Magnitude errors |")
    lines.append("|---|---|---|---|---|---|")
    total_magnitude = 0
    for name, stats in classes.items():
        esm = "n/a" if stats.get("exact_set_match_rate") is None else _fmt(stats["exact_set_match_rate"], 3)
        sub = "n/a" if stats.get("substitution_rate") is None else _fmt(stats["substitution_rate"], 3)
        magnitude = stats.get("magnitude_errors", 0)
        if name in ("numbers", "money"):
            total_magnitude = max(total_magnitude, magnitude)
        lines.append(
            f"| {name} | {stats.get('n_ref', 0)} | {stats.get('n_hyp', 0)} | {esm} | {sub} | {magnitude} |"
        )
    lines.append("")
    lines.append(
        f"- magnitude errors (scale drift, e.g. `500.000đ` → `5.000.000đ`): {total_magnitude}"
    )
    lines.append("")
    return lines


def _section_recommendations(bundle: dict[str, Any]) -> list[str]:
    ds = bundle.get("dataset") or {}
    audio = bundle.get("audio") or {}
    traj = bundle.get("trajectory") or {}
    tokens = ds.get("tokens") or {}
    lines = ["## Preprocessing recommendations", ""]
    recs: list[str] = []
    if audio.get("n_unexpected_sample_rate"):
        recs.append("Resample all audio to 16 kHz mono before ASR (linear resampler in `lifemate.asr.interface`).")
    if (audio.get("flag_counts") or {}).get("silent"):
        recs.append("Drop or flag silent clips; they yield empty transcripts and pollute WER.")
    if (audio.get("flag_counts") or {}).get("clipping"):
        recs.append("Inspect clipped files; clipping degrades ASR on loud speech.")
    if tokens.get("tokenizer") != "available":
        recs.append("Load the model tokenizer to size the prompt budget (`--tokenizer`).")
    if (traj.get("multi_call") or 0) > 0:
        recs.append("Keep multi-step supervision: train on full tool-call sequences, not single calls.")
    if (traj.get("missing_info") or {}).get("count"):
        recs.append("Preserve `sys.ask_missing_info` trajectories; they teach the agent to abstain.")
    if (traj.get("out_of_scope") or {}).get("count"):
        recs.append("Keep out-of-scope samples; they teach graceful refusal.")
    if (traj.get("read_before_write") or {}).get("unsatisfied"):
        recs.append("Normalize gold trajectories so dependency reads precede writes.")
    alignment = (bundle.get("trajectory") or {}).get("alignment") or {}
    if alignment.get("n_call_result_mismatch"):
        recs.append("Reconcile call/result mismatches before closed-loop scoring; expected results are incomplete.")
    if (traj.get("challenges") or {}).get("counts", {}).get("temporal_expression"):
        recs.append("Inject `context.current_time`/`timezone` into prompts; temporal requests are frequent.")
    if not recs:
        recs.append("No preprocessing issues detected by the automated checks.")
    lines.extend(f"- {rec}" for rec in recs)
    lines.append("")
    return lines


def _section_training_implications(bundle: dict[str, Any]) -> list[str]:
    traj = bundle.get("trajectory") or {}
    tools = bundle.get("tools") or {}
    lines = ["## Training implications", ""]
    items: list[str] = []
    stats = (traj.get("n_gold_calls") or {}).get("stats") or {}
    if stats.get("max"):
        items.append(
            f"longest gold trajectory has {int(stats['max'])} tool calls — budget prompt+generation "
            "sequence length accordingly (multi-turn tool loop, not single-shot)"
        )
    if (tools.get("kind_distribution") or {}).get("write"):
        items.append("write tools exist — the model must compose arguments from earlier read results")
    if (tools.get("x_ref") or {}).get("n_refs"):
        items.append("x-ref annotations define hard dependencies — supervision must include the read step")
    if (tools.get("x_one_of") or {}).get("n_tools"):
        items.append("x-one-of constraints are checkable post-hoc; add a constrained-decoding/validator pass")
    if (traj.get("zero_calls") or 0):
        items.append("zero-call samples require an explicit refusal/termination behaviour")
    if not items:
        items.append("insufficient label coverage to derive training implications")
    lines.extend(f"- {item}" for item in items)
    lines.append("")
    return lines


def _section_limitations(synthetic: bool, chart_note: str, bundle: dict[str, Any]) -> list[str]:
    lines = ["## Limitations", ""]
    if synthetic:
        lines.append("- This run used **synthetic fixtures**, not the official dataset. No number here is a competition finding.")
    lines.append("- EDA is descriptive; no significance testing or causal inference is performed.")
    lines.append(f"- charts: {chart_note}")
    audio = bundle.get("audio") or {}
    if audio.get("per_file_truncated"):
        lines.append(
            f"- per-file audio table truncated to {audio.get('per_file_rows_returned')} of "
            f"{audio.get('per_file_rows_total')} rows"
        )
    if audio.get("truncated_by_max_files"):
        lines.append(f"- audio analysis truncated to {audio.get('max_files')} files")
    lines.append("- Challenge tags are regex/heuristic classifications, not organizer annotations.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _write_tables(bundle: dict[str, Any], out_dir: Path) -> list[str]:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    tools = bundle.get("tools") or {}
    if tools.get("table_rows"):
        written.append(str(tables / "tools.csv"))
        write_csv(
            tables / "tools.csv",
            tools["table_rows"],
            [
                "tool", "app", "kind", "n_parameters", "n_required", "n_optional",
                "required", "params_with_enum", "params_with_ref", "has_one_of", "description_chars",
            ],
        )
    else:
        written.append(str(tables / "tools.csv"))
        write_csv(tables / "tools.csv", [], ["tool", "app", "kind", "n_parameters", "n_required"])

    usage = (bundle.get("trajectory") or {}).get("tool_usage") or []
    write_csv(tables / "tool_usage.csv", usage, ["tool", "count", "fraction"])
    written.append(str(tables / "tool_usage.csv"))

    transitions: list[dict[str, Any]] = []
    for order, block in ((bundle.get("trajectory") or {}).get("transitions") or {}).items():
        for row in (block or {}).get("top") or []:
            transitions.append(
                {"order": int(order), "count": row.get("count"), "sequence": " > ".join(str(s) for s in row.get("sequence") or [])}
            )
    write_csv(tables / "transitions.csv", transitions, ["order", "count", "sequence"])
    written.append(str(tables / "transitions.csv"))

    per_app: list[dict[str, Any]] = []
    tools_per_app = tools.get("tools_per_app") or {}
    app_usage = {row["app"]: row["count"] for row in ((bundle.get("trajectory") or {}).get("app_usage") or [])}
    for app in sorted(set(tools_per_app) | set(app_usage)):
        per_app.append(
            {
                "app": app,
                "n_tools": tools_per_app.get(app, 0),
                "n_trajectory_calls": app_usage.get(app, 0),
            }
        )
    write_csv(tables / "per_app.csv", per_app, ["app", "n_tools", "n_trajectory_calls"])
    written.append(str(tables / "per_app.csv"))

    audio_rows = (bundle.get("audio") or {}).get("per_file") or []
    write_csv(
        tables / "audio_per_file.csv",
        audio_rows,
        [
            "sample_id", "path", "ok", "error", "duration_s", "sample_rate", "channels",
            "sample_width_bits", "n_frames", "peak_amplitude", "rms", "silence_ratio",
            "clipping_ratio", "flags",
        ],
    )
    written.append(str(tables / "audio_per_file.csv"))

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_report(
    analysis_bundle: dict[str, Any],
    *,
    split: str,
    data_root: str,
    synthetic: bool,
    out_dir: str | Path,
    charts_enabled: bool = True,
) -> dict[str, Any]:
    """Write summary.json, report.md, charts/, tables/. Returns a written-artifact map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = dict(analysis_bundle)
    summary["meta"] = {
        "split": split,
        "data_root": str(data_root),
        "synthetic": bool(synthetic),
        "n_samples": (analysis_bundle.get("dataset") or {}).get("n_samples", 0),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Charts (graceful when matplotlib is absent).
    charts_dir = out / "charts"
    chart_note = "skipped (matplotlib unavailable)"
    chart_paths: list[str] = []
    try:
        from .visualization import make_charts, matplotlib_available  # noqa: PLC0415

        if not charts_enabled:
            chart_note = "skipped (--no-charts)"
        elif matplotlib_available():
            chart_paths = make_charts(analysis_bundle, charts_dir)
            chart_note = f"{len(chart_paths)} PNG written to `charts/`"
        else:
            # Still write the manifest, so downstream tooling sees the reason.
            make_charts(analysis_bundle, charts_dir)
    except Exception as exc:  # noqa: BLE001 - charts must never break the report
        chart_note = f"skipped ({type(exc).__name__}: {exc})"

    table_paths = _write_tables(analysis_bundle, out)

    lines: list[str] = []
    if synthetic:
        lines.append(SYNTHETIC_BANNER)
        lines.append("")
    lines.extend(_section_header(analysis_bundle, split, str(data_root), synthetic))
    lines.extend(_section_quality(analysis_bundle))
    lines.extend(_section_missing_fields(analysis_bundle))
    lines.extend(_section_token_stats(analysis_bundle))
    lines.extend(_section_audio(analysis_bundle))
    lines.extend(_section_tools(analysis_bundle))
    lines.extend(_section_trajectory(analysis_bundle))
    lines.extend(_section_challenges(analysis_bundle))
    lines.extend(_section_imbalance(analysis_bundle))
    lines.extend(_section_visibility(analysis_bundle))
    lines.extend(_section_asr(analysis_bundle))
    lines.extend(_section_training_implications(analysis_bundle))
    lines.extend(_section_recommendations(analysis_bundle))
    lines.extend(_section_limitations(synthetic, chart_note, analysis_bundle))

    report_path = out / "report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "out_dir": str(out),
        "summary": str(summary_path),
        "report": str(report_path),
        "charts": chart_paths,
        "charts_note": chart_note,
        "tables": table_paths,
        "synthetic": bool(synthetic),
    }
