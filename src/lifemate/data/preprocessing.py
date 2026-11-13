"""Small, dependency-free preprocessing helpers shared by the training slice.

Two jobs:

1. **Normalisation** used by the split builder's near-duplicate leakage guard.
   Normalising aggressively (NFKD, diacritics stripped, casefolded, punctuation
   removed) means ``"Đặt báo thức"`` and ``"dat bao thuc"`` hash identically, so
   near-duplicate requests can never land on both sides of the train/val split.
2. **Features** used for stratification (call-count bucket, category flags) and
   for the token-size estimates written into SFT records when no tokenizer is
   available.

Pure stdlib; importable before torch/transformers exist.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable

__all__ = [
    "CHARS_PER_TOKEN",
    "collapse_whitespace",
    "estimate_tokens",
    "n_calls_bucket",
    "normalize_transcript",
    "sample_category",
    "sample_features",
    "sample_hash",
    "sample_n_calls",
]

#: Rough Vietnamese/multilingual byte-pair ratio used when no tokenizer is loaded.
CHARS_PER_TOKEN = 4

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^0-9a-z ]+")

def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return _WS_RE.sub(" ", text or "").strip()

def normalize_transcript(text: str) -> str:
    """Aggressive normalisation for duplicate detection (not for prompting).

    NFKD-decompose, drop combining marks (Vietnamese diacritics), map ``đ`` to
    ``d``, casefold, keep only ``[0-9a-z ]``, collapse whitespace. Deterministic
    and idempotent.
    """
    raw = str(text or "")
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.replace("đ", "d").replace("Đ", "D").casefold()
    return collapse_whitespace(_NON_WORD_RE.sub(" ", lowered))

def estimate_tokens(text: str, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """Character-count token estimate (ceil division); ``0`` for empty text.

    Only a budget heuristic. When a real tokenizer is available, use it instead
    (``tokenizer.encode``); this function is what the SFT records store as
    ``token_estimate`` so training can re-check the budget at load time.
    """
    if not isinstance(text, str) or not text:
        return 0
    return max(1, -(-len(text) // max(1, int(chars_per_token))))

def sample_n_calls(sample: Any) -> int:
    """Number of gold tool calls on a sample (0 when it has no label)."""
    label = getattr(sample, "label", None)
    return len(list(getattr(label, "tool_calls", []) or [])) if label is not None else 0

def n_calls_bucket(n_calls: int) -> str:
    """Coarse bucket used as a stratification key: ``0``, ``1``, ``2``, ``3+``."""
    count = max(0, int(n_calls))
    return "3+" if count >= 3 else str(count)

def sample_category(sample: Any) -> str:
    """Category label from the raw record, or ``"unknown"``.

    Read from ``raw`` on purpose: the typed schema deliberately ignores extra
    fields, and ``category`` is a metadata/stratification field, never a model
    input.
    """
    raw = getattr(sample, "raw", None) or {}
    if isinstance(raw, dict):
        value = raw.get("category") or raw.get("task") or raw.get("type")
        if value:
            return str(value)
    return "unknown"

def sample_features(sample: Any, stratify_keys: Iterable[str] = ("n_calls_bucket", "categories")) -> dict[str, Any]:
    """Stratification features for one sample, restricted to ``stratify_keys``."""
    available: dict[str, Any] = {
        "n_calls_bucket": n_calls_bucket(sample_n_calls(sample)),
        "categories": sample_category(sample),
        "n_calls": sample_n_calls(sample),
    }
    return {key: available[key] for key in stratify_keys if key in available}

def sample_hash(sample: Any) -> str:
    """Stable duplicate-group hash over normalised request text + history.

    Two samples with the same spoken request but different ids (near-duplicate
    leakage) collide here and must stay in the same split.
    """
    transcript = normalize_transcript(getattr(sample, "audio_text", "") or "")
    turns: list[str] = []
    for turn in getattr(sample, "history", []) or []:
        role = str(getattr(turn, "role", "user") or "user")
        content = normalize_transcript(getattr(turn, "content", "") or "")
        call = getattr(turn, "tool_call", None)
        turns.append(f"{role}:{content}:{json.dumps(call, sort_keys=True, ensure_ascii=False) if call else ''}")
    payload = transcript + "\u241f" + "\u241e".join(turns)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def hash_ids(sample_ids: Iterable[str]) -> str:
    """sha256 over the sorted sample ids (split manifest fingerprint)."""
    joined = "\n".join(sorted(str(s) for s in sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
