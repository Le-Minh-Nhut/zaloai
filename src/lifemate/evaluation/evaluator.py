"""Evaluation drivers: teacher-forced decisions and closed-loop episodes.

Both paths are duck-typed so they run on CPU with test doubles and never import
torch at module load:

* teacher-forced only needs an object with ``.generate(messages) -> str``;
* closed loop needs an ``agent_factory(sample)`` returning a
  :class:`~lifemate.agent.agent.ToolAgent` bound to a real environment.

``evaluate_split`` always writes ``metrics.json``/``metrics.md``/
``predictions.jsonl``/``error_analysis.json``/``comparisons.md`` plus provenance
(model mode, checkpoint, config hash, git commit, dataset manifest hash) and an
explicit ``environment`` block. Values that cannot be measured are reported as
``"unavailable"`` or ``{"available": False}``; nothing is estimated.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lifemate.agent.protocol import Action, parse_action, render_action
from lifemate.errors import ProhibitedAccessError
from lifemate.data.history import HistoryTranscriptError

from . import error_analysis, metrics

__all__ = [
    "artifact_paths",
    "config_hash",
    "dataset_manifest_hash",
    "environment_report",
    "evaluate_closed_loop",
    "evaluate_split",
    "evaluate_teacher_forced",
    "git_commit",
    "load_decision_records",
    "write_report",
]


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def git_commit(root: str | Path | None = None) -> str:
    """Current git commit, or ``"unavailable"`` (no repo, no git binary)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable"


def config_hash(cfg: Any) -> str:
    """Stable hash of the resolved config (provenance without secrets)."""
    try:
        payload = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg or {})
    except Exception:
        payload = {"repr": repr(cfg)}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dataset_manifest_hash(data_root: str | Path | None, split: str | None = None) -> str:
    """Hash of the split's JSONL file size/mtime (never its contents' labels).

    Reading labels here would be fine (they are ours), but hashing metadata keeps
    the call cheap and avoids loading a big file into memory.
    """
    if data_root is None:
        return "unavailable"
    try:
        from lifemate.data.discovery import discover_splits

        splits = discover_splits(data_root)
    except Exception:
        return "unavailable"
    entries: list[str] = []
    names = [split] if split else sorted(splits)
    for name in names:
        paths = splits.get(name) if name else None
        if paths is None or paths.jsonl is None:
            continue
        try:
            stat = paths.jsonl.stat()
        except OSError:
            continue
        entries.append(f"{name}:{paths.jsonl.name}:{stat.st_size}:{int(stat.st_mtime)}")
    if not entries:
        return "unavailable"
    return hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()[:16]


def environment_report() -> dict[str, Any]:
    """Real GPU/torch/cuda facts, or ``"unavailable"`` — never a guess."""
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:  # lazy: torch is heavy and absent in CI images
        import torch  # type: ignore

        report["torch"] = getattr(torch, "__version__", "unavailable")
        report["cuda"] = getattr(getattr(torch, "version", None), "cuda", None) or "unavailable"
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_count"] = int(torch.cuda.device_count())
            report["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
        else:
            report["gpu"] = "unavailable (torch reports no CUDA device)"
    except Exception as exc:
        report["torch"] = "unavailable"
        report["cuda"] = "unavailable"
        report["cuda_available"] = "unavailable"
        report["gpu"] = "unavailable"
        report["torch_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        import transformers  # type: ignore

        report["transformers"] = getattr(transformers, "__version__", "unavailable")
    except Exception:
        report["transformers"] = "unavailable"
    return report


# --------------------------------------------------------------------------- #
# Teacher-forced evaluation
# --------------------------------------------------------------------------- #
_TARGET_KEYS = ("target", "completion", "output", "response", "assistant", "action", "decision")
_PROMPT_KEYS = ("prompt", "input", "body", "system", "messages", "conversation")


def _messages_of(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Recover the exact stored prompt as chat messages."""
    for key in ("messages", "conversation", "prompt_messages"):
        value = record.get(key)
        if isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
            return [{"role": str(item.get("role", "user")), "content": str(item.get("content", ""))} for item in value]
    for key in ("prompt", "input", "body", "rendered_prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            # Never append the gold target: it would enter the model context and
            # an echoing model would score 1.0 without making the decision.
            return [{"role": "user", "content": value}]
    return []


def _target_of(record: Mapping[str, Any]) -> str:
    for key in _TARGET_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            return render_action(dict(value))
    return ""


_GOLD_KEYS = ("gold_tool_calls", "expected_tool_calls")

def _is_gradeable(record: Mapping[str, Any]) -> bool:
    """Gradeable = a prompt to answer AND a gold target to compare against.

    Both halves are required: without a prompt the model is scored on an empty
    context (its answer is unrelated to the sample), and without gold there is
    nothing to compare, so an empty prediction would report a perfect 1.0.
    """
    if not _messages_of(record):
        return False
    if any(record.get(key) is not None for key in _GOLD_KEYS):
        return True
    return bool(_target_of(record))

def _expected_calls(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Gold calls of a decision record (from the target action or a field).

    ``tool_calls`` is deliberately NOT a gold source: it is the prediction key
    of both ``predictions.jsonl`` and the submission format, so reading it here
    would score a model's own output against itself.
    """
    for key in _GOLD_KEYS:
        value = record.get(key)
        if isinstance(value, list):
            return [dict(call) for call in value if isinstance(call, Mapping)]
    action = parse_action(_target_of(record))
    if action.type == "tool_call":
        return [{"name": action.name or "", "arguments": dict(action.arguments)}]
    return []


def load_decision_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read SFT decision records (JSONL) without importing the data layer."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                records.append(dict(payload))
            if limit is not None and len(records) >= limit:
                break
    return records


def _generate(generator: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    """Call the generator, honouring an optional ``max_new_tokens`` override."""
    if hasattr(generator, "max_new_tokens"):
        try:
            previous = generator.max_new_tokens
            generator.max_new_tokens = max_new_tokens
            try:
                return str(generator.generate(messages))
            finally:
                generator.max_new_tokens = previous
        except TypeError:
            pass
    if callable(generator) and not hasattr(generator, "generate"):
        return str(generator(messages))
    return str(generator.generate(messages))


def evaluate_teacher_forced(
    model_or_generator: Any,
    records_jsonl: str | Path | Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Generate the next action for each stored prompt and score it.

    ``model_or_generator`` is anything with ``.generate(messages) -> str`` (a
    :class:`~lifemate.models.generation.HFGenerator` or a test double), or a
    callable taking messages. Returns ``{"records": [...], "metrics": {...},
    "predictions": [...]}``; the metrics block comes from
    :func:`lifemate.evaluation.metrics.aggregate_teacher_forced`.
    """
    if isinstance(records_jsonl, (str, os.PathLike)):
        records = load_decision_records(records_jsonl, limit=limit)
    else:
        records = [dict(row) for row in records_jsonl]
        if limit is not None:
            records = records[:limit]

    if not records:
        return {
            "records": [],
            "predictions": [],
            "metrics": {
                "available": False,
                "reason": "no decision records found",
                "n_records": 0,
                "definitions": metrics.METRIC_DEFINITIONS,
            },
            "n_records": 0,
        }

    preds: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    per_record: list[dict[str, Any]] = []

    # Only records with a prompt AND a gold target are scored: no prompt means the
    # model answers an empty context, no gold means an empty prediction would
    # otherwise equal an empty "gold" and report a perfect 1.0.
    gold_records = [record for record in records if _is_gradeable(record)]
    n_ungradeable = len(records) - len(gold_records)

    for index, record in enumerate(gold_records):
        messages = _messages_of(record)
        target = _target_of(record)
        sample_id = str(record.get("id") or record.get("sample_id") or f"#{index}")
        started = time.perf_counter()
        error: str | None = None
        setter = getattr(model_or_generator, "set_sample", None)
        if callable(setter):  # scripted test doubles need to know which row they answer
            setter(sample_id)
        try:
            output = _generate(model_or_generator, messages, max_new_tokens)
        except Exception as exc:  # a broken generator must not abort the sweep
            output = ""
            error = f"{type(exc).__name__}: {exc}"[:300]
        elapsed = time.perf_counter() - started

        action: Action = parse_action(output)
        pred_calls = (
            [{"name": action.name or "", "arguments": dict(action.arguments)}]
            if action.type == "tool_call"
            else []
        )
        gold_action = parse_action(target)
        gold_calls = _expected_calls(record)
        if not gold_calls and gold_action.type == "tool_call":
            gold_calls = [{"name": gold_action.name or "", "arguments": dict(gold_action.arguments)}]

        pred = {
            "id": sample_id,
            "tool_calls": pred_calls,
            "finish": action.type == "finish",
            "steps": pred_calls or ([{"type": "finish"}] if action.type == "finish" else []),
            "status": "error" if error else "ok",
            "raw_output": output[:2000],
            "parse_error": action.error,
        }
        gold = {
            "id": sample_id,
            "tool_calls": gold_calls,
            "finish": gold_action.type == "finish",
            "steps": gold_calls or ([{"type": "finish"}] if gold_action.type == "finish" else []),
        }
        preds.append(pred)
        golds.append(gold)
        per_record.append(
            {
                "id": sample_id,
                "step": record.get("step", record.get("step_index")),
                "predicted": pred_calls,
                "gold": gold_calls,
                "predicted_finish": pred["finish"],
                "gold_finish": gold["finish"],
                "exact_match": metrics.decision_emitted(pred)
                and metrics.tool_call_exact_match(pred_calls, gold_calls),
                "parse_error": action.error,
                "generator_error": error,
                "elapsed_seconds": round(elapsed, 4),
                # prompt text is stored by the caller, not here: PII stays out
                "n_messages": len(messages),
            }
        )

    if not gold_records:
        return {
            "records": [],
            "predictions": [],
            "metrics": {
                "available": False,
                "reason": f"none of the {len(records)} record(s) has both a prompt and a gold target",
                "n_records": 0,
                "n_ungradeable": n_ungradeable,
                "definitions": metrics.METRIC_DEFINITIONS,
            },
            "n_records": 0,
            "n_ungradeable": n_ungradeable,
        }

    aggregated = metrics.aggregate_teacher_forced(preds, golds)
    aggregated["n_ungradeable"] = n_ungradeable
    aggregated["error_analysis"] = error_analysis.analyze(preds, golds)
    return {
        "records": per_record,
        "predictions": [
            {**pred, "gold_tool_calls": gold["tool_calls"], "gold_finish": gold["finish"]}
            for pred, gold in zip(preds, golds)
        ],
        "metrics": aggregated,
        "n_records": len(gold_records),
        "n_ungradeable": n_ungradeable,
    }


# --------------------------------------------------------------------------- #
# Closed-loop evaluation
# --------------------------------------------------------------------------- #
def evaluate_closed_loop(
    agent_factory: Callable[[Any], Any],
    samples: Sequence[Any],
    *,
    split: str,
    data_root: str | Path,
    max_samples: int | None = None,
    deadline_seconds: float | None = None,
    split_paths: Any = None,
    evaluation_mode: str = "text_to_action",
    transcripts: Mapping[str, str] | None = None,
    asr_info: Mapping[str, Any] | None = None,
    prepare_history: Any = None,
    transcribe_sample: Any = None,
) -> dict[str, Any]:
    """Run the agent against a REAL environment per sample.

    ``agent_factory(sample)`` must return a ``ToolAgent`` bound to an env created
    with ``create_environment(..., use_mock=False)``. Rows are taken from the
    environment (``get_tool_calls`` / ``get_state_hash``), never rebuilt from the
    agent's own notes.

    There is deliberately no hidden-state metric: the official scoring tooling is
    not available here, so ``final_state_correctness`` is
    ``{"available": False}``.
    """
    chosen = list(samples)
    if max_samples is not None:
        chosen = chosen[:max_samples]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()

    for sample in chosen:
        sample_id = str(getattr(sample, "id", "") or "")
        sample_started = time.monotonic()
        budget_deadline = None if deadline_seconds is None else sample_started + deadline_seconds
        history_seconds = 0.0
        asr_seconds = 0.0
        history_error = None
        try:
            if prepare_history is not None:
                stage_started = time.monotonic()
                try:
                    prepare_history(sample, budget_deadline)
                except (ProhibitedAccessError, HistoryTranscriptError):
                    raise
                except Exception as exc:
                    history_error = f"{type(exc).__name__}: {exc}"[:300]
                history_seconds = time.monotonic() - stage_started
            runtime_transcript = None
            runtime_info = None
            if transcribe_sample is not None and (budget_deadline is None or time.monotonic() < budget_deadline):
                stage_started = time.monotonic()
                runtime_transcript, runtime_info = transcribe_sample(sample)
                asr_seconds = time.monotonic() - stage_started
            agent = agent_factory(sample)
            if history_error is not None:
                env = getattr(agent, "env", None)
                rows.append({"id": sample_id, "status": "history_incomplete",
                             "tool_calls": _from_env(env, "get_tool_calls", []),
                             "state_hash": _from_env(env, "get_state_hash", ""),
                             "errors": [history_error], "history_seconds": history_seconds,
                             "current_asr_seconds": asr_seconds,
                             "total_seconds": time.monotonic() - sample_started,
                             "timed_out": budget_deadline is not None and time.monotonic() >= budget_deadline})
                failures.append({"id": sample_id, "stage": "history", "error": history_error})
                continue
        except (ProhibitedAccessError, HistoryTranscriptError):
            raise
        except Exception as exc:
            failures.append({"id": sample_id, "stage": "agent_factory", "error": f"{type(exc).__name__}: {exc}"[:300]})
            rows.append({"id": sample_id, "status": "error", "tool_calls": [], "state_hash": "", "errors": [str(exc)],
                         "total_seconds": time.monotonic() - sample_started,
                         "timed_out": budget_deadline is not None and time.monotonic() >= budget_deadline})
            continue
        try:
            transcript = runtime_transcript if transcribe_sample is not None else (transcripts.get(sample_id) if transcripts is not None else None)
            current_info = runtime_info if runtime_info is not None else (asr_info or {}).get(sample_id, {})
            if evaluation_mode == "audio_to_action" and (not transcript or current_info.get("error")):
                env = getattr(agent, "env", None)
                reason = str(current_info.get("error") or "empty ASR transcript")
                rows.append({"id": sample_id, "status": "asr_failed",
                             "tool_calls": _from_env(env, "get_tool_calls", []),
                             "state_hash": _from_env(env, "get_state_hash", ""), "errors": [reason],
                             "history_seconds": history_seconds, "current_asr_seconds": asr_seconds,
                             "total_seconds": time.monotonic() - sample_started,
                             "timed_out": budget_deadline is not None and time.monotonic() >= budget_deadline})
                failures.append({"id": sample_id, "stage": "asr", "error": reason})
                continue
            kwargs = {}
            if runtime_info is not None:
                kwargs["asr_hypotheses"] = runtime_info.get("hypotheses")
            elif asr_info is not None:
                kwargs["asr_hypotheses"] = asr_info.get(sample_id, {}).get("hypotheses")
            result = agent.run(sample, transcript=transcript, deadline=budget_deadline, **kwargs)
        except ProhibitedAccessError:
            raise
        except Exception as exc:
            failures.append({"id": sample_id, "stage": "run", "error": f"{type(exc).__name__}: {exc}"[:300]})
            rows.append({"id": sample_id, "status": "error", "tool_calls": [], "state_hash": "", "errors": [str(exc)],
                         "total_seconds": time.monotonic() - sample_started,
                         "timed_out": budget_deadline is not None and time.monotonic() >= budget_deadline})
            continue

        env = getattr(agent, "env", None)
        tool_calls = _from_env(env, "get_tool_calls", getattr(result, "tool_calls", None) or [])
        state_hash = _from_env(env, "get_state_hash", getattr(result, "state_hash", "") or "")
        status = str(getattr(result, "status", "unknown"))
        row = {
            "id": sample_id,
            "status": status,
            "tool_calls": tool_calls,
            "state_hash": state_hash,
            "finish_reason": str(getattr(result, "finish_reason", "") or ""),
            "n_llm_calls": int(getattr(result, "n_llm_calls", 0) or 0),
            "elapsed_seconds": round(float(getattr(result, "elapsed_seconds", 0.0) or 0.0), 4),
            "errors": list(getattr(result, "errors", []) or []),
            "history_seconds": history_seconds,
            "current_asr_seconds": asr_seconds,
            "total_seconds": time.monotonic() - sample_started,
            "timed_out": budget_deadline is not None and time.monotonic() >= budget_deadline,
        }
        rows.append(row)
        if status not in {"finished", "ok"}:
            failures.append({"id": sample_id, "stage": "run", "error": f"status={status}"})

    golds = []
    for sample in chosen:
        label = getattr(sample, "label", None)
        gold_calls = []
        if label is not None:
            gold_calls = [
                call.to_dict() if hasattr(call, "to_dict") else dict(call)
                for call in (getattr(label, "tool_calls", None) or [])
            ]
        golds.append({"id": str(getattr(sample, "id", "") or ""), "tool_calls": gold_calls})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    total_elapsed = time.monotonic() - started
    def distribution(key: str) -> dict[str, Any]:
        values = sorted(float(row[key]) for row in rows if key in row)
        if not values:
            return {"available": False}
        def percentile(q: float) -> float:
            pos = (len(values) - 1) * q
            lower = int(pos)
            upper = min(lower + 1, len(values) - 1)
            return values[lower] + (values[upper] - values[lower]) * (pos - lower)
        return {"mean": sum(values) / len(values), "p50": percentile(.5),
                "p95": percentile(.95), "max": values[-1], "n": len(values)}
    summary: dict[str, Any] = {
        "split": split,
        "data_root": str(data_root),
        "n_requested": len(chosen),
        "n_run": len(rows),
        "status_counts": counts,
        "wall_seconds": round(total_elapsed, 3),
        "elapsed_seconds_mean": (sum(row["elapsed_seconds"] for row in rows if "elapsed_seconds" in row) /
                                 sum("elapsed_seconds" in row for row in rows)) if any("elapsed_seconds" in row for row in rows) else None,
        "latency": {"history_asr": distribution("history_seconds"),
                    "current_asr": distribution("current_asr_seconds"),
                    "total": distribution("total_seconds"),
                    "llm": {"available": False}, "tool_env": {"available": False},
                    "prompt": {"available": False},
                    "timeout_count": sum(bool(row.get("timed_out")) for row in rows)},
        "failures": failures[:50],
        "n_failures": len(failures),
        "rows": rows,
        "evaluation_mode": evaluation_mode,
        "runtime_protocol": "end_to_end_uncached" if transcribe_sample is not None else "quality_asr_latency_excluded",
        "n_successful": sum(1 for r in rows if r["status"] in {"finished", "ok"}),
        "n_failed": len(failures),
        "final_state_correctness": {
            "available": False,
            "reason": "official scoring tooling unavailable",
        },
    }
    if any(g["tool_calls"] for g in golds):
        summary["decision_metrics"] = metrics.aggregate_teacher_forced(rows, golds)
        summary["error_analysis"] = error_analysis.analyze(rows, golds)
    else:
        summary["decision_metrics"] = {
            "available": False,
            "reason": "no gold labels on this split (test split or label-free sample)",
        }
    if split_paths is not None and getattr(split_paths, "has_environment_dir", False):
        summary["environment_dir_present"] = True
    return summary


def _from_env(env: Any, method: str, fallback: Any) -> Any:
    """Prefer the environment's value; the agent's copy is only a fallback."""
    getter = getattr(env, method, None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return fallback
    return fallback


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    """The five artifacts ``evaluate_split`` writes."""
    base = Path(output_dir)
    return {
        "metrics_json": base / "metrics.json",
        "metrics_md": base / "metrics.md",
        "predictions": base / "predictions.jsonl",
        "error_analysis": base / "error_analysis.json",
        "comparisons_md": base / "comparisons.md",
    }


def _fmt(block: Any) -> str:
    if isinstance(block, Mapping):
        if block.get("available") is False:
            return "UNAVAILABLE"
        value = block.get("value")
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is not None:
            return str(value)
        return "UNAVAILABLE"
    if isinstance(block, float):
        return f"{block:.4f}"
    if isinstance(block, (int, str)):
        return str(block)
    return "UNAVAILABLE"


def render_metrics_md(payload: Mapping[str, Any]) -> str:
    """Markdown table of the metric blocks plus provenance."""
    metric_blocks = payload.get("metrics", {})
    lines = ["# Evaluation metrics", ""]
    if payload.get("mock_env"):
        lines += ["> **MOCK ENV / SYNTHETIC** — pipeline self-test only, not a competition result.", ""]
    provenance = payload.get("provenance", {})
    lines.append("| item | value |")
    lines.append("| --- | --- |")
    for key in ("split", "evaluation_mode", "model_mode", "checkpoint", "n_samples", "config_hash", "git_commit", "dataset_manifest_hash"):
        lines.append(f"| {key} | {provenance.get(key, payload.get(key, 'unavailable'))} |")
    env = payload.get("environment", {})
    for key in ("python", "torch", "cuda", "cuda_available", "gpu", "transformers"):
        lines.append(f"| env.{key} | {env.get(key, 'unavailable')} |")
    lines += ["", "## Metrics", ""]
    if payload.get("closed_loop") or payload.get("teacher_forced"):
        source = "teacher_forced" if payload.get("teacher_forced") else "closed_loop"
        lines += [f"Source: `{source}` decision metrics (see comparisons.md for the other source).", ""]
    lines += ["| metric | value | n |", "| --- | --- | --- |"]
    for name, block in metric_blocks.items():
        if name in {"definitions", "error_analysis"}:
            continue
        n = block.get("n") if isinstance(block, Mapping) else None
        lines.append(f"| {name} | {_fmt(block)} | {n if n is not None else ''} |")
    unavailable = [name for name, block in metric_blocks.items() if isinstance(block, Mapping) and block.get("available") is False]
    if unavailable:
        lines += ["", f"Undefined on this run (no applicable sample): {', '.join(sorted(unavailable))}."]
    lines += ["", "## Metric definitions", ""]
    for name, text in metrics.METRIC_DEFINITIONS.items():
        lines.append(f"- **{name}** — {text}")
    return "\n".join(lines) + "\n"


def render_comparisons_md(payload: Mapping[str, Any]) -> str:
    """Side-by-side view of available metric sources (teacher-forced vs closed loop)."""
    lines = ["# Comparisons", ""]
    blocks = payload.get("comparisons") or {}
    if not blocks:
        return "\n".join(lines + ["No comparison sources were produced in this run.", ""])
    names: list[str] = []
    skip = {"definitions", "error_analysis", "n_records", "n_ungradeable", "n_unmeasured"}
    for source in blocks.values():
        for name in source:
            if name not in names and name not in skip:
                names.append(name)
    header = "| metric | " + " | ".join(blocks) + " |"
    lines += [header, "| --- | " + " | ".join("---" for _ in blocks) + " |"]
    for name in names:
        cells = []
        for source in blocks.values():
            block = source.get(name)
            cells.append(_fmt(block) if block is not None else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    notes = payload.get("comparison_notes") or []
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def write_report(payload: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write ``metrics.json/.md``, ``error_analysis.json``, ``comparisons.md``."""
    paths = artifact_paths(output_dir)
    paths["metrics_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["metrics_md"].write_text(render_metrics_md(payload), encoding="utf-8")
    paths["error_analysis"].write_text(
        json.dumps(payload.get("error_analysis", {"available": False, "reason": "not run"}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["comparisons_md"].write_text(render_comparisons_md(payload), encoding="utf-8")
    return paths


def evaluate_split(
    *,
    cfg: Any,
    split: str,
    output_dir: str | Path | None = None,
    teacher_forced: Mapping[str, Any] | None = None,
    closed_loop: Mapping[str, Any] | None = None,
    predictions: Sequence[Mapping[str, Any]] | None = None,
    data_root: str | Path | None = None,
    checkpoint: str | None = None,
    model_mode: str | None = None,
    mock_env: bool = False,
    extra: Mapping[str, Any] | None = None,
    evaluation_mode: str | None = None,
) -> dict[str, Any]:
    """Assemble provenance + metrics for one split and write all artifacts."""
    base = Path(output_dir or getattr(getattr(cfg, "evaluation", None), "output_dir", "outputs/evaluation"))
    data_root = data_root if data_root is not None else getattr(getattr(cfg, "data", None), "root", None)
    provenance = {
        "split": split,
        "model_mode": model_mode or getattr(getattr(cfg, "inference", None), "model_mode", "unknown"),
        "checkpoint": checkpoint or getattr(getattr(cfg, "inference", None), "checkpoint", None) or "none",
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(Path(__file__).resolve().parents[3]),
        "dataset_manifest_hash": dataset_manifest_hash(data_root, split),
        "n_samples": (closed_loop or {}).get("n_run")
        or (teacher_forced or {}).get("n_records")
        or len(predictions or []),
        "evaluation_mode": evaluation_mode
        or (closed_loop or {}).get("evaluation_mode")
        or (teacher_forced or {}).get("evaluation_mode")
        or "text_to_action",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    comparisons: dict[str, Any] = {}
    if teacher_forced and isinstance(teacher_forced.get("metrics"), Mapping):
        comparisons["teacher_forced"] = teacher_forced["metrics"]
    if closed_loop and isinstance(closed_loop.get("decision_metrics"), Mapping):
        comparisons["closed_loop"] = closed_loop["decision_metrics"]

    combined = dict(teacher_forced.get("metrics", {})) if teacher_forced else {}
    if not combined and closed_loop:
        combined = dict(closed_loop.get("decision_metrics") or {})

    if "final_state_correctness" not in combined:
        fsc = (closed_loop or {}).get("final_state_correctness")
        combined["final_state_correctness"] = fsc if fsc else {
            "available": False,
            "reason": "official scoring tooling unavailable",
        }
    error_payload: dict[str, Any] = {"available": False, "reason": "error analysis not requested"}
    if teacher_forced and teacher_forced.get("metrics", {}).get("error_analysis"):
        error_payload = teacher_forced["metrics"]["error_analysis"]
    elif closed_loop and closed_loop.get("error_analysis"):
        error_payload = closed_loop["error_analysis"]

    notes: list[str] = []
    if "/tmp/" in str(checkpoint or "") or "tiny" in str(checkpoint or "").lower():
        notes.append("TINY MODEL CPU VALIDATION: synthetic/random weights, not competition results.")
    if mock_env:
        notes.append("MOCK ENV / SYNTHETIC: these numbers come from MockToolEnv, not the official simulator.")
    if closed_loop:
        notes.append(
            "final_state_correctness is unavailable: the official scoring tooling is not installed."
        )
    eff_mode = provenance["evaluation_mode"]
    if eff_mode == "text_to_action":
        notes.append("evaluation_mode is text_to_action: LLM evaluated with gold/fixed transcript (not an ASR evaluation).")
    elif eff_mode == "audio_to_action":
        notes.append("evaluation_mode is audio_to_action: full pipeline audio -> ASR -> LLM -> actions.")
        notes.append("teacher_forced scores SFT decision records; closed_loop scores full episodes.")

    payload: dict[str, Any] = {
        "evaluation_mode": eff_mode,
        "n_successful": (closed_loop or {}).get("n_successful")
        if closed_loop
        else (teacher_forced or {}).get("n_records", 0),
        "n_failed": (closed_loop or {}).get("n_failed") if closed_loop else (teacher_forced or {}).get("n_ungradeable", 0),
        "provenance": provenance,
        "environment": environment_report(),
        "metrics": combined,
        "teacher_forced": teacher_forced,
        "closed_loop": closed_loop,
        "comparisons": comparisons,
        "comparison_notes": notes,
        "error_analysis": error_payload,
        "mock_env": bool(mock_env),
        **(dict(extra or {})),
    }

    paths = write_report(payload, base)
    if predictions is not None:
        with paths["predictions"].open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload["artifacts"] = {key: str(value) for key, value in paths.items()}
    return payload
