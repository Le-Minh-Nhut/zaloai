"""Vietnamese inverse text normalization (spoken words -> surface digits).

ADDITIONAL, opt-in pass. ``normalize_itn(text, enabled=False)`` returns exactly
``lifemate.asr.normalization.to_llm_text(text)`` and no changes, so a disabled
caller is byte-identical to the baseline path.

Deliberately conservative; everything not matched by a rule below is untouched:

* stdlib ``re`` only, no ``num2words``. The cardinal parser is a real parser
  (not a lookup table) and is round-trip consistent with
  ``normalization.int_to_vietnamese_words``, which does the inverse direction.
* Nothing is guessed. A clock time without a period-of-day marker stays as
  spoken and is recorded as a ``skipped_ambiguous`` change. A bare number is
  never rewritten. A 12h hour outside the marker's window is recorded as
  ``skipped_out_of_window``.
* Clause-level refusals: any clause holding a phone number / long ID, an email,
  an ``@handle`` or an ``địa chỉ`` frame is skipped whole. A candidate directly
  followed by a capitalized token (a name) is skipped.
* Idempotent: every replacement is emitted in its canonical form, so a second
  pass finds nothing left to change, and a skipped span is locked so no
  lower-priority rule can rewrite part of it.

Change offsets index the text this module produced from ``to_llm_text`` (the
``original`` field keeps the raw input).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from lifemate.asr.normalization import to_llm_text

__all__ = [
    "ItnChange",
    "ItnResult",
    "ItnRule",
    "apply_itn_pipeline",
    "itn_rules",
    "normalize_itn",
    "parse_vi_cardinal",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_DIGIT_VALUE: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
}

_DIGIT = "không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín"
_SCALE = "mươi|mười|trăm|nghìn|triệu|tỷ|linh"

#: One spoken-number phrase: it may start on a digit word or bare ``mười``
#: (``mười một``) and continues on digit/scale words separated by one space.
_NUMW = rf"(?:{_DIGIT}|mười)(?:\s+(?:{_DIGIT}|{_SCALE}))*"

_MARKER = r"sáng|trưa|chiều|tối|đêm|hôm\s+nay|ngày\s+mai"

#: Period-of-day -> inclusive 24h hour window. ``hôm nay``/``ngày mai`` state a
#: day, not an hour, so they never resolve an ambiguous 12h time.
_WINDOWS: dict[str, tuple[int, int]] = {
    "sáng": (0, 11),
    "trưa": (11, 13),
    "chiều": (12, 17),
    "tối": (18, 23),
    "đêm": (19, 23),
}

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItnChange:
    """One auditable rule application (a skip carries ``replacement==original``)."""

    rule: str
    original: str
    replacement: str
    start: int
    end: int

    @property
    def is_noop(self) -> bool:
        """True for ``skipped_*`` records: the text was deliberately left alone."""
        return self.original == self.replacement

@dataclass(frozen=True)
class ItnResult:
    """ITN output plus its audit trail."""

    text: str
    original: str
    changes: tuple[ItnChange, ...]
    enabled: bool

    @property
    def changed(self) -> bool:
        """True when at least one recorded change really altered the text."""
        return any(not change.is_noop for change in self.changes)

@dataclass(frozen=True)
class ItnRule:
    """One introspectable rule: a compiled pattern plus its handler."""

    name: str
    ipattern: re.Pattern[str]
    handler: Callable[[str, re.Match[str]], "str | None"]

#: Handler results meaning "record a skip, change nothing".
SKIP_AMBIGUOUS = "\x00skip_ambiguous"
SKIP_OUT_OF_WINDOW = "\x00skip_out_of_window"


# ---------------------------------------------------------------------------
# Cardinal parsing (words -> number)
# ---------------------------------------------------------------------------

def parse_vi_cardinal(words: str | list[str]) -> int | float | None:
    """Parse a Vietnamese cardinal phrase, or ``None`` when not fully understood.

    Handles ``không..chín``, ``mươi``/``mười``, ``trăm``, ``nghìn``, ``triệu``,
    ``tỷ`` (up to 10^12+), the ``lăm``/``mốt``/``tư`` variants and the
    fractional half ``rưỡi`` (``bảy rưỡi`` -> ``7.5``). A phrase with any
    unknown token returns ``None``: a half-understood number is never rewritten.
    """
    tokens = words.split() if isinstance(words, str) else list(words)
    if not tokens:
        return None
    total = 0
    group = 0
    pending: int | None = None
    half = False
    for raw in tokens:
        token = unicodedata.normalize("NFC", raw).lower()
        if token in _DIGIT_VALUE:
            if pending is not None:
                return None  # two digits in a row: not a number shape we know
            pending = _DIGIT_VALUE[token]
        elif token in ("mươi", "mười"):
            if pending is None:
                if token == "mươi":
                    return None  # "mươi" cannot start a number
                group += 10
            else:
                group += pending * 10
                pending = None
        elif token == "trăm":
            group += (pending if pending is not None else 1) * 100
            pending = None
        elif token == "linh":
            if pending is not None:
                return None
        elif token in ("nghìn", "triệu"):
            group += pending or 0
            pending = None
            total += group * (1_000 if token == "nghìn" else 1_000_000)
            group = 0
        elif token == "tỷ":
            group += pending or 0
            pending = None
            total = (total + group) * 1_000_000_000
            group = 0
        elif token == "rưỡi":
            if total + group + (pending or 0) == 0:
                return None  # "rưỡi" alone is not a number
            half = True
        else:
            return None
    value = total + group + (pending or 0)
    return value + 0.5 if half else value

def _parse_number_token(text: str) -> float | None:
    """Parse ``hai mươi`` / ``500.000`` / ``500`` into a number, or ``None``."""
    stripped = text.strip()
    if not stripped:
        return None
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+|\d+", stripped):
        return float(re.sub(r"\.", "", stripped))
    value = parse_vi_cardinal(stripped)
    return None if value is None else float(value)

def _format_amount(value: float) -> str | None:
    """``500000`` -> ``500.000``; ``None`` when the value is not a whole amount."""
    if value <= 0 or value != int(value):
        return None
    return f"{int(value):,}".replace(",", ".")

def _render_time(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"

# ---------------------------------------------------------------------------
# Clause protection
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"\+?\d[\d .-]{7,}|\d{6,}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE_RE = re.compile(r"(?<!\w)@\w+")
_ADDRESS_RE = re.compile(r"\bđịa\s+chỉ\b", re.IGNORECASE)
#: Clause boundaries for the "skip the whole clause" guard. ``.`` is NOT a
#: boundary: it lives inside emails, currency amounts and abbreviations, so
#: splitting there would let a rule rewrite the tail of a protected span.
_CLAUSE_SPLIT_RE = re.compile(r"[,;!?\n]")

def _protected_ranges(text: str) -> list[tuple[int, int]]:
    """Clause ranges that must never be rewritten (phones, IDs, mail, address)."""
    blocked: list[tuple[int, int]] = []
    start = 0
    bounds = [m.start() for m in _CLAUSE_SPLIT_RE.finditer(text)] + [len(text)]
    for bound in bounds:
        clause = text[start:bound]
        if any(p.search(clause) for p in (_PHONE_RE, _EMAIL_RE, _HANDLE_RE, _ADDRESS_RE)):
            blocked.append((start, bound))
        start = bound + 1
    return blocked

def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < hi and lo < end for lo, hi in spans)

def _next_token(text: str, end: int) -> str:
    """The next whitespace-delimited token after ``end`` ('' at end of text)."""
    match = re.match(r"\s*(\S+)", text[end:])
    return match.group(1) if match else ""

def _previous_token(text: str, start: int) -> str:
    """The whitespace-delimited token ending right before ``start`` ('' at BOS)."""
    match = re.search(r"(\S+)\s*$", text[:start])
    return match.group(1) if match else ""

#: Frame words that own the number after them. A standalone cardinal behind one
#: of these is part of a time/date frame this module handles (or refuses) below:
#: ``giờ ba mươi`` and ``tháng mười một`` must not decay into ``giờ 30``.
_FRAME_WORDS = frozenset({"giờ", "h", "tháng", "ngày", "năm", "phút"})

def _followed_by_name(text: str, end: int) -> bool:
    """True when the candidate is glued to a capitalized (name-like) token."""
    token = _next_token(text, end)
    return bool(token) and token[0].isupper()

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _window_hour(hour: int, marker: str | None) -> int | str:
    """Resolve a spoken hour to 24h, or return a skip sentinel.

    No marker: only 13..23 is unambiguous (already 24h). A marker restricts the
    allowed window, so ``bảy giờ chiều`` -> 19 but ``mười chín giờ sáng`` cannot
    be resolved and is skipped rather than guessed.
    """
    if marker is None:
        return hour if 13 <= hour <= 23 else SKIP_AMBIGUOUS
    window = _WINDOWS.get(re.sub(r"\s+", " ", marker.lower()))
    if window is None:
        return SKIP_AMBIGUOUS
    low, high = window
    for candidate in (hour, hour + 12):
        if low <= candidate <= high:
            return candidate
    return SKIP_OUT_OF_WINDOW

def _minutes(words: str | None) -> int | None:
    if not words:
        return 0
    value = parse_vi_cardinal(words)
    if value is None or value != int(value) or not 0 <= int(value) <= 59:
        return None
    return int(value)

def _handle_time_words(text: str, match: re.Match[str]) -> str | None:
    """``bảy giờ ba mươi sáng`` / ``bảy rưỡi tối`` / ``bảy giờ kém mười lăm chiều``."""
    hour_value = parse_vi_cardinal(match.group("hour"))
    if hour_value is None or not 0 <= hour_value <= 23:
        return None
    marker = match.group("marker")
    if match.group("half") is not None:
        resolved = _window_hour(int(hour_value), marker)
        return resolved if isinstance(resolved, str) else _render_time(resolved, 30)
    kem = match.group("kem_minutes")
    if kem is not None:
        spoken = _minutes(kem)
        if spoken is None or spoken == 0:
            return None
        resolved = _window_hour(int(hour_value), marker)
        if isinstance(resolved, str):
            return resolved
        return _render_time((resolved - 1) % 24, 60 - spoken)
    minutes = _minutes(match.group("minutes"))
    if minutes is None:
        return None
    resolved = _window_hour(int(hour_value), marker)
    return resolved if isinstance(resolved, str) else _render_time(resolved, minutes)

def _handle_time_digits(text: str, match: re.Match[str]) -> str | None:
    """``7 giờ 30`` / ``7h30`` / ``19h30`` (+ optional period marker)."""
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if hour > 23 or minute > 59:
        return None
    resolved = _window_hour(hour, match.group("marker"))
    return resolved if isinstance(resolved, str) else _render_time(resolved, minute)

def _handle_date(text: str, match: re.Match[str]) -> str | None:
    """``ngày hai mươi tháng mười một`` -> ``20/11`` (word forms only)."""
    day_raw, month_raw = match.group("day"), match.group("month")
    if day_raw.strip().isdigit() and month_raw.strip().isdigit():
        return None  # already surface digits: nothing to gain
    day, month = _parse_number_token(day_raw), _parse_number_token(month_raw)
    if day is None or month is None:
        return None
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{int(day)}/{int(month)}"

def _handle_percent(text: str, match: re.Match[str]) -> str | None:
    """``hai mươi phần trăm`` -> ``20%``."""
    value = _parse_number_token(match.group("num"))
    return None if value is None else f"{int(value)}%"

def _handle_money_words(text: str, match: re.Match[str]) -> str | None:
    """``năm trăm nghìn đồng`` -> ``500.000đ``.

    An explicit marker is trusted as-is. Without one, the phrase's own trailing
    scale word (``nghìn``/``triệu``/``tỷ``) is the only acceptable currency
    signal, and only at a clause end: ``hai nghìn người`` is a count, and
    ``hai mươi`` alone is a bare number that must never gain a currency.
    """
    phrase = match.group("num")
    value = _parse_number_token(phrase)
    if value is None:
        return None
    if match.group("marker") is None:
        if phrase.split()[-1] not in ("nghìn", "triệu", "tỷ"):
            return None
        token = _next_token(text, match.end())
        if token and token[0] not in ",.;!?)":
            return None  # a noun follows: never invent a currency
        if value < 1_000:
            return None
    amount = _format_amount(value)
    return None if amount is None else f"{amount}đ"

def _handle_money_digits(text: str, match: re.Match[str]) -> str | None:
    """``500 nghìn`` / ``500k`` / ``500.000đ`` -> ``500.000đ``."""
    value = _parse_number_token(match.group("num"))
    if value is None:
        return None
    if match.group("marker").lower() in ("nghìn", "k"):
        value *= 1_000
    elif match.group("marker").lower() == "triệu":
        value *= 1_000_000
    amount = _format_amount(value)
    return None if amount is None else f"{amount}đ"

def _handle_cardinal(text: str, match: re.Match[str]) -> str | None:
    """Clause-final compound: ``tổng cộng hai mươi`` -> ``tổng cộng 20``."""
    phrase = match.group("num")
    if len(phrase.split()) < 2 or not re.search(r"\b(mươi|mười|trăm|nghìn|triệu|tỷ)\b", phrase):
        return None
    value = _parse_number_token(phrase)
    if value is None or value != int(value):
        return None
    token = _next_token(text, match.end())
    if token and token[0].isalpha():
        return None  # a noun follows ("hai mươi tuổi"): not a standalone number
    if _previous_token(text, match.start()).lower() in _FRAME_WORDS:
        return None  # "giờ ba mươi", "tháng mười một": a frame this module owns
    return str(int(value))

# ---------------------------------------------------------------------------
# Rule table (order == priority: earlier rules lock their spans first)
# ---------------------------------------------------------------------------

_RULES: tuple[ItnRule, ...] = (
    ItnRule(
        "date",
        re.compile(
            rf"\bngày\s+(?P<day>{_NUMW}|\d{{1,2}})\s+tháng\s+(?P<month>{_NUMW}|\d{{1,2}})\b"
        ),
        _handle_date,
    ),
    ItnRule(
        "time_words",
        re.compile(
            rf"(?<!\w)(?P<hour>{_NUMW})\s+"
            r"(?:giờ(?:\s+kém\s+(?P<kem_minutes>"
            + _NUMW
            + r")|\s+(?P<minutes>"
            + _NUMW
            + r"))?|(?P<half>rưỡi))"
            rf"(?:\s+(?P<marker>{_MARKER}))?"
        ),
        _handle_time_words,
    ),
    ItnRule(
        "time_digits",
        re.compile(
            rf"(?<![\d\w])(?P<hour>\d{{1,2}})\s*(?:h|giờ)\s*(?P<minute>\d{{1,2}})?(?!\d)"
            rf"(?:\s+(?P<marker>{_MARKER}))?"
        ),
        _handle_time_digits,
    ),
    ItnRule(
        "percent",
        re.compile(rf"(?<!\w)(?P<num>{_NUMW})\s+phần\s+trăm\b"),
        _handle_percent,
    ),
    ItnRule(
        "money_words",
        re.compile(rf"(?<!\w)(?P<num>{_NUMW})(?:\s+(?P<marker>đồng|vnđ|đ|k))?\b"),
        _handle_money_words,
    ),
    ItnRule(
        "money_digits",
        re.compile(
            r"(?<![\d\w])(?P<num>\d{1,3}(?:\.\d{3})+|\d+)\s*(?P<marker>nghìn|triệu|vnđ|đồng|k|đ)\b"
        ),
        _handle_money_digits,
    ),
    ItnRule(
        "cardinal",
        re.compile(rf"(?<!\w)(?P<num>{_NUMW})(?=\s*[,.;!?]|\s*$)"),
        _handle_cardinal,
    ),
)

def itn_rules() -> tuple[ItnRule, ...]:
    """The rule table in application order (``.name`` / ``.ipattern`` inspectable)."""
    return _RULES

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def normalize_itn(text: str | None, *, enabled: bool = False) -> ItnResult:
    """Return ``to_llm_text(text)``, plus the ITN pass only when ``enabled``.

    ``enabled=False`` is the baseline: the text is exactly ``to_llm_text``'s and
    ``changes`` is empty.
    """
    raw = "" if text is None else str(text)
    base = to_llm_text(raw)
    if not enabled:
        return ItnResult(text=base, original=raw, changes=(), enabled=False)

    occupied = _protected_ranges(base)
    edits: list[tuple[int, int, str, str]] = []
    skips: list[ItnChange] = []
    for rule in _RULES:
        for match in rule.ipattern.finditer(base):
            start, end = match.span()
            if start == end or _overlaps(occupied, start, end):
                continue
            candidate = match.group(0)
            try:
                replacement = rule.handler(base, match)
            except (KeyError, IndexError, ValueError):
                replacement = None
            if replacement is None:
                continue
            if replacement in (SKIP_AMBIGUOUS, SKIP_OUT_OF_WINDOW):
                occupied.append((start, end))
                skips.append(
                    ItnChange(
                        rule=("skipped_ambiguous" if replacement == SKIP_AMBIGUOUS
                              else "skipped_out_of_window"),
                        original=candidate,
                        replacement=candidate,
                        start=start,
                        end=end,
                    )
                )
                continue
            if not replacement or replacement == candidate:
                continue  # never emit an empty replacement for a non-empty span
            if _followed_by_name(base, end):
                continue
            occupied.append((start, end))
            edits.append((start, end, replacement, rule.name))

    out = base
    for start, end, replacement, _rule in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + replacement + out[end:]
    changes = tuple(sorted(
        [ItnChange(rule=r, original=base[s:e], replacement=rep, start=s, end=e)
         for s, e, rep, r in edits] + skips,
        key=lambda c: (c.start, c.end),
    ))
    return ItnResult(text=out, original=raw, changes=changes, enabled=True)

def apply_itn_pipeline(text: str | None, *, enabled: bool = False) -> str:
    """``to_llm_text`` then ITN when enabled; ``to_llm_text`` alone when disabled."""
    return normalize_itn(text, enabled=enabled).text
