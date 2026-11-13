"""ASR metric aggregation (WER/CER/latency/exact match) and critical token diagnostics.

Pure stdlib. Structurally mirrors the EDA contract: always returns a dict with
``available`` so downstream consumers can branch without exceptions.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .normalization import cer, for_wer, normalize_vi, wer

__all__ = [
    "evaluate_asr",
    "critical_token_diagnostics",
    "format_metrics_table",
    "save_asr_metrics",
    "unavailable_metrics",
    "percentile",
]

Pair = tuple[str, str, str]  # (id, reference, hypothesis)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile; ``q`` in [0, 100]. ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def _summ(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": percentile(values, 50),
        "min": min(values),
        "max": max(values),
    }


def unavailable_metrics(reason: str, backend_name: str | None = None) -> dict[str, Any]:
    """Explicit 'no predictions' structure (never zero-filled fake scores)."""
    return {
        "available": False,
        "reason": reason,
        "backend": backend_name,
        "n": 0,
        "raw": None,
        "normalized": None,
        "sentence_exact_match": None,
        "latency_s": None,
        "per_utterance": [],
    }


def evaluate_asr(
    pairs: Iterable[Pair] | None,
    backend_name: str = "unknown",
    *,
    latencies: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Aggregate WER/CER over ``(id, reference, hypothesis)`` triples.

    ``pairs`` empty or ``None`` -> ``{"available": False, ...}``. Scores are
    reported both raw and after :func:`for_wer`; the normalized view is the
    headline number. Also reports the ASCII-folded CER as a supplementary view.

    The ``raw`` view is case-insensitive (``str.lower`` on both sides) but
    otherwise untouched: no number verbalisation, no punctuation stripping, no
    abbreviation expansion. Case folding is not cosmetic — Gipformer emits
    ALL-UPPERCASE text, so a case-sensitive raw CER scores 0.78 where the
    identical-text score is 0.11, i.e. it measures the casediff, not the ASR.
    """
    rows = list(pairs or [])
    if not rows:
        return unavailable_metrics("no predictions provided", backend_name)

    per_utterance: list[dict[str, Any]] = []
    raw_cer: list[float] = []
    raw_wer: list[float] = []
    norm_cer: list[float] = []
    norm_wer: list[float] = []
    exact = 0
    empty_hyp = 0

    for item in rows:
        uid, ref, hyp = (list(item) + ["", "", ""])[:3]
        ref = "" if ref is None else str(ref)
        hyp = "" if hyp is None else str(hyp)
        if not hyp.strip():
            empty_hyp += 1
        nref, nhyp = for_wer(ref), for_wer(hyp)
        # Raw diagnostic: case-insensitive only, so an ALL-UPPERCASE recogniser
        # is not scored as if every character were wrong.
        cf_ref, cf_hyp = ref.lower(), hyp.lower()
        r_cer = cer(cf_ref, cf_hyp)
        r_wer = wer(cf_ref, cf_hyp)
        n_cer = cer(nref, nhyp)
        n_wer = wer(nref, nhyp)
        raw_cer.append(r_cer)
        raw_wer.append(r_wer)
        norm_cer.append(n_cer)
        norm_wer.append(n_wer)
        if nref == nhyp:
            exact += 1
        per_utterance.append(
            {
                "id": uid,
                "reference": ref,
                "hypothesis": hyp,
                "reference_normalized": nref,
                "hypothesis_normalized": nhyp,
                "cer_raw": r_cer,
                "wer_raw": r_wer,
                "cer_normalized": n_cer,
                "wer_normalized": n_wer,
                "exact_match": nref == nhyp,
            }
        )

    n = len(rows)
    lat = [float(x) for x in (latencies or [])]
    return {
        "available": True,
        "reason": None,
        "backend": backend_name,
        "n": n,
        "raw": {
            **_summ(raw_cer),
            "cer_mean": mean(raw_cer),
            "wer_mean": mean(raw_wer),
        },
        "normalized": {
            **_summ(norm_cer),
            "cer_mean": mean(norm_cer),
            "wer_mean": mean(norm_wer),
            "cer_median": percentile(norm_cer, 50),
            "cer_p95": percentile(norm_cer, 95),
            "wer_median": percentile(norm_wer, 50),
            "wer_p95": percentile(norm_wer, 95),
        },
        "sentence_exact_match": exact / n if n else None,
        "empty_hypotheses": empty_hyp,
        "latency_s": {
            "count": len(lat),
            "mean": mean(lat) if lat else None,
            "p50": percentile(lat, 50),
            "p95": percentile(lat, 95),
            "min": min(lat) if lat else None,
            "max": max(lat) if lat else None,
        },
        "per_utterance": per_utterance,
    }


# ---------------------------------------------------------------------------
# Critical Token Diagnostics & Magnitude Detection
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    "không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu",
    "bảy", "bẩy", "tám", "chín", "mười", "mươi", "trăm", "nghìn", "nghàn",
    "triệu", "tỷ", "tỉ", "linh", "lẻ",
}

_COMMON_VI_ASCII = {
    "va", "co", "la", "cho", "di", "den", "lam", "ra", "vao", "toi", "ban",
    "anh", "em", "chi", "con", "nguoi", "mot", "hai", "ba", "bon", "nam",
    "sau", "bay", "tam", "chin", "muoi", "tram", "nghin", "trieu", "ty",
    "so", "duong", "pho", "quan", "huyen", "phuong", "xa", "ngay", "thang",
    "nam", "gio", "phut", "sang", "chieu", "toi", "khong", "duoc", "voi",
    "tat", "bat", "mo", "dong", "gui", "nhan", "xem", "nghe", "doc", "viet",
}


def _extract_phrase_magnitudes(text: str) -> list[int]:
    """Extract magnitude scale (100, 10^3, 10^6, 10^9) for each money/number phrase."""
    t = text.lower()
    scales: list[int] = []

    # 1. Thousand-separated dotted numbers (e.g. 500.000, 1.500.000)
    for m in re.finditer(r"\b\d{1,3}(?:\.\d{3})+", t):
        val = int(m.group(0).replace(".", ""))
        if val >= 10**9:
            scales.append(10**9)
        elif val >= 10**6:
            scales.append(10**6)
        elif val >= 10**3:
            scales.append(10**3)
        elif val >= 100:
            scales.append(100)

    # 2. Number words / digits with scale words: e.g. "năm trăm nghìn", "500 nghìn", "5 triệu", "1,5 triệu"
    scale_re = re.compile(
        r"(?:(?:\d+[.,]?\d*|[không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười|mươi|trăm|linh|lẻ]+)\s+)*(tỷ|tỉ|triệu|nghìn|nghàn|ngàn)",
        re.IGNORECASE,
    )
    for m in scale_re.finditer(t):
        last_scale = m.group(1).lower()
        if last_scale in ("tỷ", "tỉ"):
            scales.append(10**9)
        elif last_scale == "triệu":
            scales.append(10**6)
        elif last_scale in ("nghìn", "nghàn", "ngàn"):
            scales.append(10**3)

    # 3. Currency symbol amounts not covered by dotted: e.g. "500k", "20k"
    for m in re.finditer(r"\b(\d+)\s*k\b", t):
        val = int(m.group(1)) * 1000
        if val >= 10**6:
            scales.append(10**6)
        elif val >= 10**3:
            scales.append(10**3)

    return scales


def _count_magnitude_errors(ref: str, hyp: str) -> int:
    """Count missing or mismatched magnitudes between reference and hypothesis multisets."""
    ref_mags = Counter(_extract_phrase_magnitudes(ref))
    hyp_mags = Counter(_extract_phrase_magnitudes(hyp))
    if not ref_mags and not hyp_mags:
        return 0
    # Count ref magnitudes not matched in hypothesis multiset
    diff = ref_mags - hyp_mags
    return sum(diff.values())


def _extract_tokens_by_class(text: str) -> dict[str, set[str]]:
    """Extract classified tokens from a text string."""
    classes: dict[str, set[str]] = {
        "numbers": set(),
        "times": set(),
        "money": set(),
        "entities": set(),
        "addresses": set(),
        "english_loanwords": set(),
    }
    if not text:
        return classes

    # 1. Times
    for m in re.finditer(r"\b\d{1,2}[h:]\d{2}\b|\b\d{1,2}\s*(?:giờ|phút)\b|\b(?:giờ|phút)\b", text, flags=re.IGNORECASE):
        classes["times"].add(m.group(0).lower())

    # 2. Money
    for m in re.finditer(r"\b\d+(?:[.,]\d+)?\s*(?:đ|vnđ|đồng|nghìn|nghàn|ngàn|triệu|tỷ|tỉ|k)\b|\b(?:\w+\s+)?(?:nghìn|nghàn|ngàn|triệu|tỷ|tỉ)\s*(?:đồng|đ)?\b", text, flags=re.IGNORECASE):
        classes["money"].add(m.group(0).lower().strip())

    # 3. Numbers (digits and Vietnamese number words)
    words = re.findall(r"\b[A-Za-z0-9\u00c0-\u1ef9]+\b", text)
    for w in words:
        wl = w.lower()
        if wl in _NUM_WORDS or w.isdigit():
            classes["numbers"].add(wl)

    # 4. Entities: capitalized tokens in reference + kinship heuristics
    kinship = r"\b(?:anh|chị|em|cô|chú|bác|ông|bà)\s+([A-Z\u00c0-\u1ef8][a-z0-9\u00c0-\u1ef9]*)"
    for m in re.finditer(kinship, text):
        classes["entities"].add(m.group(0).strip())
    # Capitalized tokens (excluding sentence start if sole capitalization)
    caps = re.findall(r"\b[A-Z\u00c0-\u1ef8][a-z0-9\u00c0-\u1ef9]*\b", text)
    for c in caps:
        classes["entities"].add(c)

    # 5. Addresses
    for m in re.finditer(r"\bsố\s+\d+[A-Za-z]?\b|\b(?:đường|phố|quận|huyện|phường|xã)\s+[A-Za-z0-9\u00c0-\u1ef9\s]+?(?=[,.]|$)", text, flags=re.IGNORECASE):
        classes["addresses"].add(m.group(0).strip().lower())

    # 6. English loanwords: ASCII-only alphabetic tokens >= 2 chars not in common VI ASCII words
    for w in words:
        if re.match(r"^[A-Za-z]{2,}$", w):
            wl = w.lower()
            if wl not in _COMMON_VI_ASCII:
                classes["english_loanwords"].add(wl)

    return classes


def critical_token_diagnostics(pairs: Iterable[Pair] | None) -> dict[str, Any]:
    """Compute error rates and magnitude drift across token classes.

    Classes: numbers, times, money, entities, addresses, english_loanwords.
    Reports: n_ref, n_hyp, exact_set_match_rate, substitution_rate, magnitude_errors.
    """
    rows = list(pairs or [])
    if not rows:
        return {
            "available": False,
            "reason": "no pairs provided",
            "n_pairs": 0,
            "classes": {},
        }

    class_names = ["numbers", "times", "money", "entities", "addresses", "english_loanwords"]
    ref_counts = {c: 0 for c in class_names}
    hyp_counts = {c: 0 for c in class_names}
    exact_set_matches = {c: 0 for c in class_names}
    active_pairs = {c: 0 for c in class_names}
    substitutions = {c: 0 for c in class_names}
    magnitude_errors = {c: 0 for c in class_names}

    for item in rows:
        _, ref, hyp = (list(item) + ["", "", ""])[:3]
        ref = "" if ref is None else str(ref)
        hyp = "" if hyp is None else str(hyp)

        ref_classes = _extract_tokens_by_class(ref)
        hyp_classes = _extract_tokens_by_class(hyp)

        # Magnitude check on money / numbers
        # Multiset magnitude comparison on money / numbers
        mag_err = _count_magnitude_errors(ref, hyp)
        if mag_err > 0:
            magnitude_errors["numbers"] += mag_err
            magnitude_errors["money"] += mag_err

        for c in class_names:
            rtoks = ref_classes[c]
            htoks = hyp_classes[c]
            ref_counts[c] += len(rtoks)
            hyp_counts[c] += len(htoks)

            if rtoks or htoks:
                active_pairs[c] += 1
                if rtoks == htoks:
                    exact_set_matches[c] += 1

            if rtoks:
                substitutions[c] += len(rtoks - htoks)

    report_classes: dict[str, Any] = {}
    for c in class_names:
        n_r = ref_counts[c]
        n_h = hyp_counts[c]
        n_active = active_pairs[c]

        report_classes[c] = {
            "n_ref": n_r,
            "n_hyp": n_h,
            "exact_set_match_rate": (exact_set_matches[c] / n_active) if n_active else None,
            "substitution_rate": (substitutions[c] / n_r) if n_r else 0.0,
            "magnitude_errors": magnitude_errors[c] if c in ("numbers", "money") else 0,
        }

    return {
        "available": True,
        "reason": None,
        "n_pairs": len(rows),
        "classes": report_classes,
    }


def format_metrics_table(data: dict[str, Any]) -> str:
    """Format evaluation or diagnostic metrics into a markdown report table."""
    if not data or not data.get("available", False):
        return f"ASR metrics unavailable: {data.get('reason', 'no data') if data else 'empty'}"

    lines: list[str] = []
    if "classes" in data:
        # Diagnostic table
        lines.append(f"### Critical Token Diagnostics (n={data.get('n_pairs', 0)})")
        lines.append("| Class | Ref Tokens | Hyp Tokens | Exact Set Match Rate | Sub Rate | Magnitude Errors |")
        lines.append("|---|---|---|---|---|---|")
        for c, stats in data["classes"].items():
            esm = f"{stats['exact_set_match_rate']:.3f}" if stats['exact_set_match_rate'] is not None else "N/A"
            sub = f"{stats['substitution_rate']:.3f}" if stats['substitution_rate'] is not None else "N/A"
            mag = str(stats['magnitude_errors'])
            lines.append(f"| {c} | {stats['n_ref']} | {stats['n_hyp']} | {esm} | {sub} | {mag} |")
    else:
        # Evaluate table
        lines.append(f"### ASR Evaluation ({data.get('backend', 'unknown')}, n={data.get('n', 0)})")
        norm = data.get("normalized", {})
        raw = data.get("raw", {})
        lines.append("| Metric | Raw | Normalized (for_wer) |")
        lines.append("|---|---|---|")
        lines.append(f"| Mean CER | {raw.get('cer_mean', 0.0):.4f} | {norm.get('cer_mean', 0.0):.4f} |")
        lines.append(f"| Mean WER | {raw.get('wer_mean', 0.0):.4f} | {norm.get('wer_mean', 0.0):.4f} |")
        lines.append(f"| Sentence Exact Match | - | {data.get('sentence_exact_match', 0.0):.4f} |")

    return "\n".join(lines)


def save_asr_metrics(path: str | Path, data: dict[str, Any]) -> Path:
    """Write metrics JSON, creating parent dirs. Returns the written path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
