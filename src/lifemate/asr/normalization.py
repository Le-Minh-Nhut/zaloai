"""Vietnamese text normalization + WER/CER for ASR evaluation and LLM ingestion.

DESTRUCTIVE NORMALIZATION BAN:
LifeMate transcripts contain Vietnamese names, locations, time expressions,
money amounts, contact references, smart-home commands, and mixed Vietnamese/English.
Benchmark-oriented normalization is DESTRUCTIVE: "chuyển năm trăm nghìn" (500,000)
must never become "chuyển năm triệu" (5,000,000).

Therefore, there are TWO DISTINCT PATHS:
1. `for_wer(text)` — evaluation only: Unicode NFC, lowercase, punctuation stripped,
   whitespace collapsed, numbers converted to spoken form (only reversible/faithful rules),
   diacritics PRESERVED by default with an explicit `fold_diacritics` variant.
   NEVER apply `for_wer` in any inference or agent ingestion path!
2. `to_llm_text(text)` — transcript fed to Qwen. CONSERVATIVE: NFC, trim/collapse whitespace,
   fix spacing around punctuation, keep diacritics, keep digits as digits, keep currency
   symbols/units (`500.000đ`, `15h30`, `20/11`, `+84900...`), keep English loanwords,
   do NOT lowercase, do NOT expand or convert numbers, do NOT strip punctuation that carries
   meaning. Idempotent.

Pure stdlib: nothing here imports numpy or torch.
"""

from __future__ import annotations

import re
import unicodedata
import warnings

__all__ = [
    "NORMALIZATION_RULES",
    "for_wer",
    "to_llm_text",
    "normalize_vi",
    "fold_diacritics",
    "tokenize_words",
    "int_to_vietnamese_words",
    "cer",
    "wer",
    "normalization_report",
    "levenshtein",
]

# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------

#: Chat-style abbreviations common in Vietnamese spoken-request transcripts.
ABBREVIATIONS: dict[str, str] = {
    "ko": "không",
    "khong": "không",
    "hok": "không",
    "dc": "được",
    "đc": "được",
    "vs": "với",
    "mn": "mọi người",
    "sp": "sản phẩm",
    "nv": "nhân viên",
    "ck": "chuyển khoản",
    "bn": "bao nhiêu",
    "ntn": "như thế nào",
    "hnay": "hôm nay",
    "hn": "hôm nay",
}

#: Unit abbreviations. Expanded to syllable-separated Vietnamese so the word
#: count stays stable under careful vs casual spelling.
UNIT_ABBREVIATIONS: dict[str, str] = {
    "kg": "ki lô gam",
    "g": "gam",
    "km": "ki lô mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "gb": "gi ga bai",
    "mb": "mê ga bai",
    "tb": "tê ra bai",
    "ph": "phút",
    "p": "phút",
}

NORMALIZATION_RULES: list[str] = [
    "unicode_nfc: canonical composition (e.g. 'e'+combining acute == 'é')",
    "lowercase: str.lower() (Vietnamese-aware)",
    "number_expansion (for_wer only): conservative spoken form for faithful integers (0..10^12) "
    "and dotted thousand groups (e.g. '1.500.000' -> 'một triệu năm trăm nghìn', '21' -> 'hai mươi mốt'). "
    "Ambiguous numbers (times '15h30', dates '20/11', phones '+84900000000') left as digits.",
    "strip_punctuation (for_wer only): keep Unicode categories L* (letters) and N* (digits); "
    "all other characters become space",
    "collapse_whitespace: runs of whitespace -> single space, then strip ends",
    "abbreviations (for_wer only): whole-token chat abbreviations expanded (see ABBREVIATIONS)",
    "units (for_wer only): whole-token unit abbreviations expanded (see UNIT_ABBREVIATIONS)",
    "diacritics_preserved: Vietnamese diacritics preserved by default in both paths.",
    "llm_conservative_spacing (to_llm_text only): fix spacing around punctuation without stripping "
    "or altering digits, currencies, case, or loanwords.",
]

_WS_RE = re.compile(r"\s+")
_DIGITS_VI = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _strip_punctuation(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def int_to_vietnamese_words(n: int) -> str:
    """Convert a positive integer or zero (0..10^12) into canonical Vietnamese spoken words."""
    if n < 0:
        return ""
    if n == 0:
        return _DIGITS_VI[0]

    def _read_two(d10: int, d1: int) -> list[str]:
        words = []
        if d10 == 1:
            words.append("mười")
        else:
            words.append(f"{_DIGITS_VI[d10]} mươi")
        if d1 == 0:
            pass
        elif d1 == 1:
            words.append("mốt" if d10 >= 2 else "một")
        elif d1 == 4 and d10 >= 2:
            words.append("bốn")
        elif d1 == 5:
            words.append("lăm")
        else:
            words.append(_DIGITS_VI[d1])
        return words

    def _read_three(val: int, has_higher: bool) -> list[str]:
        d100 = val // 100
        rem = val % 100
        d10 = rem // 10
        d1 = rem % 10
        words = []
        if d100 > 0 or has_higher:
            words.append(f"{_DIGITS_VI[d100]} trăm")
        if d10 > 0:
            words.extend(_read_two(d10, d1))
        elif d1 > 0:
            if d100 > 0 or has_higher:
                words.append(f"linh {_DIGITS_VI[d1]}")
            else:
                words.append(_DIGITS_VI[d1])
        return words

    groups: list[int] = []
    tmp = n
    while tmp > 0:
        groups.append(tmp % 1000)
        tmp //= 1000

    parts: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            continue
        has_higher = any(groups[j] > 0 for j in range(idx + 1, len(groups)))
        g_words = _read_three(g, has_higher)

        scale_mod = idx % 3
        ty_count = idx // 3
        scale_parts: list[str] = []
        if scale_mod == 1:
            scale_parts.append("nghìn")
        elif scale_mod == 2:
            scale_parts.append("triệu")
        if ty_count > 0:
            scale_parts.extend(["tỷ"] * ty_count)
        elif scale_mod == 0 and idx > 0:
            scale_parts.append("tỷ")

        chunk = " ".join(g_words)
        if scale_parts:
            chunk = chunk + " " + " ".join(scale_parts)
        parts.append(chunk)

    return " ".join(parts).strip()


def _expand_numbers_vi(text: str) -> str:
    """Expand faithful numeric expressions to spoken Vietnamese words.

    Leaves unexpandable numbers (phones, time expressions, dates, codes) as digits.
    """
    # 1. Thousand-separated integers: e.g. 1.500.000, 500.000
    def _repl_thousand(m: re.Match) -> str:
        full = m.group(1)
        val = int(full.replace(".", ""))
        words = int_to_vietnamese_words(val)
        suffix = m.group(2) or ""
        return f" {words} {suffix} "

    text = re.sub(
        r"\b(\d{1,3}(?:\.\d{3})+)(đ|vnđ|đồng)?\b",
        _repl_thousand,
        text,
        flags=re.IGNORECASE,
    )

    # 2. Decimal numbers with comma: e.g. "1,5" -> "một phẩy năm"
    def _repl_decimal(m: re.Match) -> str:
        integer_part = int(m.group(1))
        dec_str = m.group(2)
        int_words = int_to_vietnamese_words(integer_part)
        dec_words = " ".join(_DIGITS_VI[int(d)] for d in dec_str)
        return f" {int_words} phẩy {dec_words} "

    text = re.sub(r"\b(\d+),(\d+)\b", _repl_decimal, text)

    # 3. Standalone integers
    def _repl_int(m: re.Match) -> str:
        s = m.group(0)
        # Leading zero with length > 1 (e.g. phone '09...', code '01') -> keep digits
        if len(s) > 1 and s.startswith("0"):
            return s
        # 10+ digits without separator is phone / id -> keep digits
        if len(s) >= 10:
            return s
        val = int(s)
        return f" {int_to_vietnamese_words(val)} "

    # Lookbehind / lookahead ensures we don't touch times (15h30, 15:30), dates (20/11), or phones (+84)
    text = re.sub(r"(?<![+\w/:\.])\d+(?![/\w:\.])", _repl_int, text)
    return text


def for_wer(
    text: str | None,
    *,
    fold_diacritics: bool = False,
    expand_numbers: bool = True,
) -> str:
    """Evaluation-only normalization for WER/CER benchmark scoring.

    Rules: Unicode NFC, lowercase, number expansion (if enabled), punctuation
    stripping, whitespace collapse, and abbreviation expansion. Diacritics
    preserved by default; fold_diacritics=True folds them to ASCII.

    NEVER use in an agent inference pipeline.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text)).lower()
    s = s.replace("\u00a0", " ")

    if expand_numbers:
        s = _expand_numbers_vi(s)

    s = _strip_punctuation(s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""

    tokens = [
        ABBREVIATIONS.get(tok, UNIT_ABBREVIATIONS.get(tok, tok))
        for tok in s.split(" ")
    ]
    res = _WS_RE.sub(" ", " ".join(tokens)).strip()
    if fold_diacritics:
        res = _fold_diacritics_impl(res)
    return res


def to_llm_text(text: str | None) -> str:
    """Conservative transcript normalization for LLM (Qwen) ingestion.

    Guarantees:
    - Unicode NFC canonical composition.
    - Idempotent: `to_llm_text(to_llm_text(x)) == to_llm_text(x)`.
    - Preserves case (proper nouns, entity names like 'anh Nam', 'Lê Lợi').
    - Preserves diacritics.
    - Preserves digits as digits and currency/units (`500.000đ`, `15h30`, `20/11`, `+84900...`).
    - Preserves English loanwords (`book`, `shop`, `ok`).
    - Fixes spacing around punctuation without stripping meaning-bearing symbols.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("\u00a0", " ")

    lines = s.splitlines()
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    s = " ".join([ln for ln in cleaned_lines if ln])

    # Remove space before punctuation: " , " -> ", "
    s = re.sub(r"\s+([,.:;?!])", r"\1", s)

    # Add space after punctuation if followed by word character, protecting
    # numbers/currencies (1,5 / 500.000đ) and times (15:30)
    s = re.sub(r"([,;?!])(?=[^\s\d,/])", r"\1 ", s)

    # Period followed by uppercase letter when preceded by lowercase letter: "xong.Tiếp" -> "xong. Tiếp"
    # Protects abbreviations like "TP.HCM"
    s = re.sub(r"([a-z\u00e0-\u1ef9])\.([A-Z\u00c0-\u1ef8])", r"\1. \2", s)

    return s.strip()


def normalize_vi(text: str | None) -> str:
    """DEPRECATED alias for `for_wer(text)`.

    Kept for backwards compatibility with legacy callers.
    New code must explicitly call `for_wer(text)` for evaluation or
    `to_llm_text(text)` for LLM prompt ingestion.
    """
    return for_wer(text)


def _fold_diacritics_impl(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", str(text))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "d")


def fold_diacritics(text: str) -> str:
    """ASCII-fold Vietnamese diacritics (supplementary metric view only)."""
    return _fold_diacritics_impl(text)


def tokenize_words(text: str) -> list[str]:
    """Whitespace tokenization of text."""
    if not text:
        return []
    return str(text).split()


def levenshtein(a: str | list[str], b: str | list[str]) -> int:
    """Edit distance with two rolling rows (O(min(len a, len b)) memory)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        append = current.append
        for j, cb in enumerate(b, 1):
            append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (0 if ca == cb else 1),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. Calls for_wer internally for normalized scoring."""
    ref = "" if reference is None else str(reference)
    hyp = "" if hypothesis is None else str(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on whitespace tokens."""
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def normalization_report() -> dict:
    """Machine-readable description of what normalization rules do."""
    return {
        "rules": list(NORMALIZATION_RULES),
        "rule_details": {
            "unicode_nfc": {
                "description": "Unicode canonical decomposition then canonical composition.",
                "example_before": "e\\u0301",
                "example_after": "é",
            },
            "lowercase": {
                "description": "Lowercase string (Vietnamese-aware) in for_wer only.",
                "example_before": "Hà Nội",
                "example_after": "hà nội",
            },
            "strip_punctuation": {
                "description": "Strip non-alphanumeric punctuation in for_wer.",
                "example_before": "Xin chào, bạn!",
                "example_after": "xin chào bạn",
            },
            "collapse_whitespace": {
                "description": "Collapse multiple tabs/spaces into single space and trim ends.",
                "example_before": "  a   b  ",
                "example_after": "a b",
            },
            "number_expansion": {
                "description": "Expand numbers to spoken Vietnamese words in for_wer.",
                "example_before": "500.000đ",
                "example_after": "năm trăm nghìn đ",
            },
            "to_llm_conservative": {
                "description": "Preserve case, digits, currencies, and fix punctuation spacing for LLM.",
                "example_before": "Xin chào , có 500.000đ !",
                "example_after": "Xin chào, có 500.000đ!",
            },
        },
        "n_abbreviations": len(ABBREVIATIONS),
        "n_unit_abbreviations": len(UNIT_ABBREVIATIONS),
        "abbreviations": dict(sorted(ABBREVIATIONS.items())),
        "unit_abbreviations": dict(sorted(UNIT_ABBREVIATIONS.items())),
        "diacritics_preserved": True,
        "two_paths": True,
        "number_verbalization": True,
    }
