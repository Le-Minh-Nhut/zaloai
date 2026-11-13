"""Dataset-level EDA over typed :class:`~lifemate.data.schemas.Sample` objects.

Import-safe without numpy/pandas/matplotlib: all statistics come from
:mod:`statistics`, and every aggregation is a plain list/dict.

Privacy: ``user_profile`` **values** are never copied into any output — only key
names and their frequencies. The same policy applies to transcripts (only
lengths/statistics are reported, plus a small example set only when the caller
explicitly asks for examples).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "describe",
    "analyze_dataset",
    "parse_iso_like",
    "field_inventory",
]

#: Timestamps that carry no usable information for analysis but are fine to parse.
_ISO_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ](?P<time>\d{2}:\d{2}(?::\d{2})?)(?P<frac>\.\d+)?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)
_LOOSE_DATE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})$")
_LOOSE_TIME_RE = re.compile(r"^(?P<time>\d{1,2}:\d{2}(?::\d{2})?)$")
_SEP_RE = re.compile(r"[/_]")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def describe(values: Iterable[float | int]) -> dict[str, Any]:
    """Summary statistics using stdlib only; safe on empty and single-item input.

    ``stdev`` is the *population* standard deviation (``pstdev``): the inputs are
    the whole analysed split, not a sample of it.
    """
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "stdev": pstdev(clean) if len(clean) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p5": _percentile_sorted(ordered, 5),
        "p25": _percentile_sorted(ordered, 25),
        "p75": _percentile_sorted(ordered, 75),
        "p95": _percentile_sorted(ordered, 95),
        "p99": _percentile_sorted(ordered, 99),
    }


def _percentile_sorted(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def _top(counter: Counter, n: int = 30) -> list[dict[str, Any]]:
    return [{"key": k, "count": c} for k, c in counter.most_common(n)]


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def parse_iso_like(value: Any) -> datetime | None:
    """Parse common timestamp spellings; ``None`` when unparseable.

    Accepts ``YYYY-MM-DDTHH:MM[:SS][.fff][Z|±HH:MM]``, space-separated variants,
    bare dates and bare times (bare times are anchored to 1970-01-01 so their
    hour/minute distribution is still usable, which is noted in the output).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            ts = float(value)
            if ts > 1e11:  # milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = _SEP_RE.sub("-", value.strip()) if value.strip()[:4].isdigit() else value.strip()
    text = text.strip()
    m = _ISO_RE.match(text)
    if m:
        base = f"{m.group('date')}T{m.group('time')}"
        if m.group("frac"):
            base += m.group("frac")
        tz = m.group("tz") or ""
        if tz == "Z":
            tz = "+00:00"
        elif tz and ":" not in tz:
            tz = f"{tz[:3]}:{tz[3:]}"
        try:
            return datetime.fromisoformat(base + tz)
        except ValueError:
            return None
    m = _LOOSE_DATE_RE.match(text)
    if m:
        try:
            return datetime.fromisoformat(text + "T00:00:00")
        except ValueError:
            return None
    m = _LOOSE_TIME_RE.match(text)
    if m:
        try:
            hour, minute, *rest = m.group("time").split(":")
            return datetime(1970, 1, 1, int(hour), int(minute), int(rest[0]) if rest else 0)
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Field inventory
# ---------------------------------------------------------------------------
def field_inventory(raw_records: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    """Presence/absence inventory of top-level keys in the raw JSONL records."""
    if raw_records is None:
        return {"available": False, "reason": "raw records not provided"}
    records = [r for r in raw_records if isinstance(r, dict) and "__error__" not in r]
    if not records:
        return {"available": True, "n_records": 0, "fields": {}, "always_missing": []}
    total = len(records)
    counts: Counter[str] = Counter()
    for rec in records:
        counts.update(rec.keys())
    fields = {
        key: {
            "count": c,
            "fraction": c / total,
            "present_in_all": c == total,
        }
        for key, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    # Keys seen in some records but never in others are already in ``fields``;
    # "always_missing" reports *structural* gaps the parser knew to look for.
    expected = ("id", "audio_path", "audio_text", "context", "history", "label")
    always_missing = [k for k in expected if k not in counts]
    return {
        "available": True,
        "n_records": total,
        "fields": fields,
        "always_missing": always_missing,
    }


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def _tokenizer_callable(tokenizer: Any) -> Callable[[str], list] | None:
    """Adapt either an HF tokenizer (``.encode``) or any ``callable(text)->list``.

    ``prefer`` the ``.encode`` attribute: a real HF tokenizer is *also* callable
    (it returns a ``BatchEncoding``), and ``len()`` of that is 2 — the number of
    fields, not tokens — so the callable branch would report 2 tokens per string.
    """
    if tokenizer is None:
        return None
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return encode
    if callable(tokenizer):
        return tokenizer
    raise TypeError(
        "tokenizer must be callable(text)->list or expose .encode(text)->list; "
        f"got {type(tokenizer).__name__}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def analyze_dataset(
    samples: Iterable[Any],
    raw_records: Iterable[dict[str, Any]] | None = None,
    catalog: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Full dataset-level analysis. Never raises on odd sample shapes."""
    samples = list(samples or [])
    n = len(samples)

    ids = [str(getattr(s, "id", "")) for s in samples]
    id_counts = Counter(ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1 and k}

    history_lengths: list[int] = []
    text_chars: list[int] = []
    text_words: list[int] = []
    profile_keys: Counter[str] = Counter()
    app_counts: Counter[int] = Counter()
    app_combos: Counter[str] = Counter()
    app_usage: Counter[str] = Counter()
    context_presence = Counter()
    times_raw: list[Any] = []
    parsed_hours: list[int] = []
    parse_failures: list[dict[str, Any]] = []
    missing_fields = Counter()
    labels_present = 0

    texts_for_tokens: list[str] = []

    for sample in samples:
        history = list(getattr(sample, "history", []) or [])
        history_lengths.append(len(history))
        text = getattr(sample, "audio_text", "") or ""
        texts_for_tokens.append(text)
        text_chars.append(len(text))
        text_words.append(len(text.split()) if text.strip() else 0)

        context = getattr(sample, "context", None)
        if context is None:
            missing_fields["context"] += 1
            continue
        context_presence["context"] += 1
        if not getattr(context, "current_time", None):
            missing_fields["context.current_time"] += 1
        else:
            times_raw.append(context.current_time)
        if not getattr(context, "timezone", None):
            missing_fields["context.timezone"] += 1
        if not getattr(context, "current_location", None):
            missing_fields["context.current_location"] += 1
        profile = getattr(context, "user_profile", None)
        if not isinstance(profile, dict) or not profile:
            missing_fields["context.user_profile"] += 1
            profile = profile if isinstance(profile, dict) else {}
        # PRIVACY: key names and counts only, never values.
        profile_keys.update(profile.keys())

        apps = list(getattr(context, "installed_apps", []) or [])
        if not apps:
            missing_fields["context.installed_apps"] += 1
        app_counts[len(apps)] += 1
        app_usage.update(apps)
        if apps:
            app_combos[",".join(sorted(str(a) for a in apps))] += 1

        label = getattr(sample, "label", None)
        if label is not None:
            labels_present += 1
        else:
            missing_fields["label"] += 1

    for raw_value in times_raw:
        parsed = parse_iso_like(raw_value)
        if parsed is None:
            parse_failures.append({"value_type": type(raw_value).__name__})
            continue
        parsed_hours.append(parsed.hour)

    hour_hist = Counter(parsed_hours)
    time_dist = {
        "available": bool(times_raw),
        "n_present": len(times_raw),
        "n_parsed": len(parsed_hours),
        "n_parse_failures": len(parse_failures),
        "parse_failure_examples": parse_failures[:5],
        "hour_histogram": {f"{h:02d}": hour_hist.get(h, 0) for h in range(24)} if parsed_hours else {},
        "hour_coverage": len(hour_hist) / 24 if parsed_hours else 0.0,
        "busiest_hours": [h for h, _ in hour_hist.most_common(3)] if parsed_hours else [],
        "note": "bare times are anchored to 1970-01-01; only hour/minute are meaningful",
    }

    # --- token statistics (optional tokenizer) ---------------------------
    token_stats: dict[str, Any]
    encode = None
    try:
        encode = _tokenizer_callable(tokenizer)
    except TypeError as exc:
        token_stats = {"tokenizer": "invalid", "reason": str(exc)}
    if encode is None:
        token_stats = {"tokenizer": "unavailable", "reason": "tokenizer not loaded"}
    else:
        lengths: list[int] = []
        token_errors = 0
        for text in texts_for_tokens:
            try:
                ids_out = encode(text) if text else []
                lengths.append(len(ids_out or []))
            except Exception:  # noqa: BLE001 - tokenizer quirks must not kill EDA
                token_errors += 1
        token_stats = {
            "tokenizer": "available",
            "name": getattr(tokenizer, "name_or_path", None),
            "stats": describe(lengths),
            "n_errors": token_errors,
        }

    return {
        "n_samples": n,
        "n_unique_ids": len(id_counts),
        "duplicate_ids": dict(sorted(duplicates.items())),
        "n_duplicate_ids": len(duplicates),
        "n_with_labels": labels_present,
        "n_without_labels": n - labels_present,
        "field_presence": {
            "context": context_presence.get("context", 0),
            "label": labels_present,
            "audio_text_nonempty": sum(1 for t in texts_for_tokens if t.strip()),
            "history_nonempty": sum(1 for h in history_lengths if h > 0),
            "fraction": {
                "context": (context_presence.get("context", 0) / n) if n else None,
                "label": (labels_present / n) if n else None,
                "audio_text_nonempty": (
                    sum(1 for t in texts_for_tokens if t.strip()) / n if n else None
                ),
                "history_nonempty": (sum(1 for h in history_lengths if h > 0) / n if n else None),
            },
        },
        "missing_fields": dict(sorted(missing_fields.items())),
        "raw_field_inventory": field_inventory(raw_records),
        "history_length": {
            "values": history_lengths,
            "stats": describe(history_lengths),
            "distribution": dict(sorted(Counter(history_lengths).items())),
            "zero_turn_samples": sum(1 for h in history_lengths if h == 0),
            "long_history_samples": sum(1 for h in history_lengths if h >= 8),
        },
        "input_text": {
            "chars": {"values": text_chars, "stats": describe(text_chars)},
            "words": {"values": text_words, "stats": describe(text_words)},
            "empty_text_samples": sum(1 for t in texts_for_tokens if not t.strip()),
        },
        "user_profile": {
            "n_samples_with_profile": n - missing_fields.get("context.user_profile", 0),
            "key_frequency": _top(profile_keys),
            "n_distinct_keys": len(profile_keys),
            "privacy": "only key names and counts are reported; values are never copied",
        },
        "installed_apps": {
            "count_stats": describe(list(app_counts.elements())),
            "count_distribution": {str(k): v for k, v in sorted(app_counts.items())},
            "app_frequency": _top(app_usage),
            "n_distinct_apps": len(app_usage),
            "top_combinations": _top(app_combos, 20),
        },
        "context_time": time_dist,
        "tokens": token_stats,
        "catalog": {
            "available": catalog is not None,
            "n_tools": len(getattr(catalog, "tools", {}) or {}),
            "n_apps": len(getattr(catalog, "apps", []) or []),
        },
        "malformed_records": _malformed_inventory(raw_records),
    }


def _malformed_inventory(raw_records: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    if raw_records is None:
        return {"available": False, "n_malformed": 0, "items": []}
    items = []
    for rec in raw_records:
        if isinstance(rec, dict) and "__error__" in rec:
            items.append(
                {
                    "line": rec.get("__line__"),
                    "error": rec.get("__error__"),
                    "raw_preview": str(rec.get("__raw__", ""))[:200],
                }
            )
    return {"available": True, "n_malformed": len(items), "items": items[:50]}
