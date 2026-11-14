"""Noisy-ASR and N-best prompt variants built from a PRE-COMPUTED transcript file.

Design rule: this module **never loads an ASR model**. Hypotheses come from a
JSONL file produced ahead of time by ``scripts/transcribe_split.py
--transcript-records``, so dataset building stays dependency-free (stdlib only)
and byte-reproducible. A missing file raises with the exact producing command
instead of falling back to the gold transcript.

A variant reuses the *same* ``sample_id`` as its original: it is the same
sample, so the group that ``split_builder`` assigned before augmentation still
covers it (a variant can never leak across the train/val boundary).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from lifemate.compliance import assert_path_allowed, safe_open
from lifemate.data.augmentation import (
    NBEST_HEADER,
    REQUEST_HEADER,
    replace_first_section,
)
from lifemate.errors import LifeMateError

__all__ = [
    "NBEST_HEADER",
    "NOISY_ASR_MODES",
    "REQUEST_HEADER",
    "TranscriptHypothesis",
    "TranscriptRecord",
    "apply_nbest_format",
    "apply_noisy_asr",
    "load_transcripts",
    "render_request_section",
    "replace_request_section",
]

#: Command that produces a transcript file (kept exact: it is the error text).
_PRODUCER = (
    "python3 scripts/transcribe_split.py --split <split> --data-root <data-root> "
    "--output <predictions.jsonl> --transcript-records <path>"
)

_SELECT_BUCKETS = 1_000_000

@dataclass(frozen=True)
class TranscriptHypothesis:
    """One recogniser hypothesis for one utterance."""

    text: str
    score: float | None = None
    rank: int = 0
    source: str = ""

    @classmethod
    def from_raw(cls, raw: Any, rank: int = 0) -> "TranscriptHypothesis":
        """Tolerant construction from a string or a mapping."""
        if isinstance(raw, Mapping):
            score = raw.get("score")
            return cls(
                text=str(raw.get("text") or ""),
                score=float(score) if isinstance(score, (int, float)) else None,
                rank=int(raw.get("rank", rank) or 0),
                source=str(raw.get("source") or ""),
            )
        return cls(text=str(raw or ""), rank=rank)

@dataclass(frozen=True)
class TranscriptRecord:
    """All hypotheses for one sample, ordered best first after construction."""

    sample_id: str
    hypotheses: tuple[TranscriptHypothesis, ...] = ()
    asr_model: str = ""
    asr_revision: str = ""
    asr_config: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        # Identical texts are one hypothesis (the best-ranked one wins) and are
        # never duplicated in the prompt; nothing is synthesised here.
        ordered = sorted(
            enumerate(self.hypotheses), key=lambda item: (int(item[1].rank), item[0])
        )
        seen: set[str] = set()
        unique: list[TranscriptHypothesis] = []
        for _index, hypothesis in ordered:
            text = hypothesis.text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(replace(hypothesis, text=text))
        object.__setattr__(self, "hypotheses", tuple(unique))

    @property
    def best(self) -> str:
        """Rank-0 text (empty when the recogniser produced nothing)."""
        return self.hypotheses[0].text if self.hypotheses else ""

    def at_rank(self, rank: int) -> TranscriptHypothesis | None:
        """Hypothesis by 1-based rank (``rank=1`` is the recogniser's best output)."""
        index = int(rank) - 1
        return self.hypotheses[index] if 0 <= index < len(self.hypotheses) else None

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_transcripts(path: str | Path, *, require_multi: bool = False) -> dict[str, TranscriptRecord]:
    """Read a transcript JSONL written by ``scripts/transcribe_split.py``.

    ``require_multi`` raises when any record carries fewer than two *distinct*
    hypotheses (an N-best experiment on a 1-best file would silently degrade to
    the gold prompt distribution).
    """
    target = assert_path_allowed(path)
    if not target.is_file():
        raise LifeMateError(
            f"transcript file not found: {target}\n"
            f"produce it first with:\n  {_PRODUCER}"
        )
    records: dict[str, TranscriptRecord] = {}
    with safe_open(target, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LifeMateError(f"{target}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, Mapping):
                raise LifeMateError(f"{target}:{lineno}: expected a JSON object")
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if not sample_id:
                raise LifeMateError(f"{target}:{lineno}: missing 'sample_id'")
            if sample_id in records:
                raise LifeMateError(f"{target}:{lineno}: duplicate sample_id {sample_id!r}")
            raw_hyps = row.get("hypotheses")
            if raw_hyps is None:
                # Single-hypothesis producer row: one real hypothesis, not a guess.
                text = str(row.get("hypothesis") or "")
                raw_hyps = [{"text": text, "rank": 0}] if text else []
            if not isinstance(raw_hyps, (list, tuple)):
                raise LifeMateError(f"{target}:{lineno}: 'hypotheses' must be a list")
            records[sample_id] = TranscriptRecord(
                sample_id=sample_id,
                hypotheses=tuple(
                    TranscriptHypothesis.from_raw(item, rank=index)
                    for index, item in enumerate(raw_hyps)
                ),
                asr_model=str(row.get("asr_model") or row.get("backend") or ""),
                asr_revision=str(row.get("asr_revision") or row.get("revision") or ""),
                asr_config=dict(row.get("asr_config") or {}),
            )
    if require_multi:
        thin = sorted(key for key, value in records.items() if len(value.hypotheses) < 2)
        if thin:
            raise LifeMateError(
                f"{target}: {len(thin)} record(s) carry fewer than 2 distinct hypotheses "
                f"(e.g. {thin[0]!r}); an N-best prompt needs a multi-hypothesis file"
            )
    return records

# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_request_section(
    record: TranscriptRecord, *, nbest_format: bool = False, k: int = 3
) -> str:
    """The prompt's request section for one transcript record.

    The single form is byte-identical to what ``build_decision_prompt`` emits
    today (``"# Yêu cầu\\n" + transcript.strip()``). ``nbest_format`` uses the
    numbered list only when more than one hypothesis survives deduplication: a
    one-item list would move the prompt distribution for no signal.
    """
    if nbest_format and len(record.hypotheses) > 1:
        limit = max(1, int(k))
        lines = [f"{index}. {hyp.text}" for index, hyp in enumerate(record.hypotheses[:limit], 1)]
        return NBEST_HEADER + "\n\n" + "\n".join(lines)
    return REQUEST_HEADER + "\n" + record.best

def replace_request_section(prompt_body: str, section_text: str) -> str | None:
    """Swap the whole request section (header+body), keeping every other byte."""
    return replace_first_section(prompt_body, section_text)

# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _bucket(seed: Any, sample_id: str) -> int:
    """Stable per-id bucket: independent of iteration order and of ``hash()``."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % _SELECT_BUCKETS

def _selected(seed: Any, sample_id: str, ratio: float) -> bool:
    return _bucket(seed, sample_id) < round(float(ratio) * _SELECT_BUCKETS)

def _get(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)

def _prompt_of(record: Any) -> str:
    messages = _get(record, "messages")
    if not messages:
        return ""
    last = messages[-1]
    return str(last.get("content", "")) if isinstance(last, Mapping) else str(getattr(last, "content", ""))

def _with_prompt(record: Any, prompt_body: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``record`` with a new prompt body and merged meta (target verbatim)."""
    out = dict(record) if isinstance(record, Mapping) else dict(vars(record))
    messages = [dict(message) for message in (out.get("messages") or [])]
    if not messages:
        raise LifeMateError("record has no messages to rewrite")
    messages[-1]["content"] = prompt_body
    out["messages"] = messages
    merged = dict(out.get("meta") or {})
    merged.update(meta)
    out["meta"] = merged
    return out

def _ids_hash(ids: Iterable[str]) -> str:
    joined = "\n".join(sorted(str(item) for item in ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
def _dataset_fingerprint(records: Iterable[Any], key: str = "sample_id") -> str:
    tokens = []
    for r in records:
        sid = str(_get(r, key) or "")
        aug = str((_get(r, "meta") or {}).get("augmentation") or "")
        prompt = _prompt_of(r)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        tokens.append(f"{sid}:{aug}:{prompt_hash}")
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Noisy ASR
# --------------------------------------------------------------------------- #
NOISY_ASR_MODES = ("augment", "replace")

def apply_noisy_asr(
    records: Sequence[Any],
    transcripts: Mapping[str, TranscriptRecord] | None,
    *,
    enabled: bool = False,
    ratio: float = 0.0,
    seed: int = 42,
    key: str = "sample_id",
    hypothesis_rank: int = 1,
    mode: str = "augment",
) -> tuple[list[Any], dict[str, Any]]:
    """Swap a ``ratio``-sampled record's transcript for a noisy hypothesis.

    ``hypothesis_rank`` is 1-based over the deduplicated hypotheses: rank 1 is
    the recogniser's own best output, which replaces the gold transcript (the
    classic noisy-ASR setting — the model must act on what the ASR heard, not on
    the clean reference). The sampled set is fixed per sample id by
    ``sha256(seed:id)``, so reordering or subsetting the input cannot change
    which ids are chosen.

    ``mode`` decides what the sampled set *means*:

    * ``"augment"`` — every clean record is kept and one variant is appended
      after it. The record count grows, so the noisy SHARE of the result is
      ``ratio / (1 + ratio)``, not ``ratio`` (0.5 -> 33 % noisy).
    * ``"replace"`` — the sampled clean record is replaced by its noisy variant.
      The record count is unchanged and the noisy share really is ``ratio``
      (0.5 -> 50 % clean / 50 % noisy). Samples with no usable transcript stay
      clean and are counted in ``n_skipped``/``n_missing_transcripts``.

    The variant preserves ``sample_id``, ``step_index``, the rendered history,
    context, installed apps, tool trajectory and the TARGET action verbatim: the
    target is never regenerated and no model is called.
    """
    if not enabled:
        return records, {}
    if not 0.0 < float(ratio) <= 1.0:
        raise LifeMateError(f"noisy ASR ratio must be in (0, 1], got {ratio!r}")
    if str(mode) not in NOISY_ASR_MODES:
        raise LifeMateError(
            f"noisy ASR mode must be one of {NOISY_ASR_MODES}, got {mode!r}"
        )
    if not transcripts:
        raise LifeMateError(
            "noisy ASR is enabled but no transcripts were loaded\n"
            f"produce them first with:\n  {_PRODUCER}"
        )
    if int(hypothesis_rank) < 1:
        raise LifeMateError(f"hypothesis_rank is 1-based, got {hypothesis_rank!r}")

    mode = str(mode)
    out: list[Any] = []
    selected_ids: list[str] = []
    n_created = n_missing = n_thin = n_no_section = 0
    for record in records:
        sample_id = str(_get(record, key) or "")
        if not sample_id or not _selected(seed, sample_id, ratio):
            out.append(record)
            continue
        selected_ids.append(sample_id)
        variant = None
        transcript = transcripts.get(sample_id)
        if transcript is None:
            n_missing += 1
        else:
            hypothesis = transcript.at_rank(int(hypothesis_rank))
            if hypothesis is None:
                n_thin += 1
            else:
                section = render_request_section(
                    TranscriptRecord(
                        sample_id=sample_id,
                        hypotheses=(hypothesis,),
                        asr_model=transcript.asr_model,
                        asr_revision=transcript.asr_revision,
                        asr_config=transcript.asr_config,
                    )
                )
                prompt = replace_request_section(_prompt_of(record), section)
                if prompt is None:
                    n_no_section += 1
                else:
                    variant = _with_prompt(
                        record,
                        prompt,
                        {
                            "augmentation": "noisy_asr",
                            "asr_model": transcript.asr_model,
                            "asr_revision": transcript.asr_revision,
                            "asr_rank": int(hypothesis_rank),
                            "noisy_asr_ratio": float(ratio),
                            "noisy_asr_mode": mode,
                            "augmentation_seed": int(seed),
                            "gold_transcript_replaced": True,
                        },
                    )
        if variant is None:
            out.append(record)
            continue
        n_created += 1
        if mode == "replace":
            out.append(variant)
        else:
            out.extend([record, variant])
    n_total = len(out)
    report = {
        "enabled": True,
        "ratio": float(ratio),
        "seed": int(seed),
        "hypothesis_rank": int(hypothesis_rank),
        "augmentation_mode": mode,
        "n_original_records": len(records),
        "n_clean_records": n_total - n_created,
        "n_noisy_records": n_created,
        "n_total_records": n_total,
        "requested_noisy_ratio": float(ratio),
        "effective_noisy_ratio": round(n_created / n_total, 6) if n_total else 0.0,
        "n_skipped": len(selected_ids) - n_created,
        "n_missing_transcripts": n_missing,
        "dataset_fingerprint": _dataset_fingerprint(out, key),
        "n_records_in": len(records),
        "n_variants_created": n_created,
        "n_missing_transcript": n_missing,
        "n_insufficient_hypotheses": n_thin,
        "n_prompt_without_request_section": n_no_section,
        "sample_ids_selected_hash": _ids_hash(selected_ids),
    }
    return out, report

# --------------------------------------------------------------------------- #
# N-best prompt format
# --------------------------------------------------------------------------- #
def apply_nbest_format(
    records: Sequence[Any],
    transcripts: Mapping[str, TranscriptRecord] | None,
    *,
    enabled: bool = False,
    k: int = 3,
    seed: int = 42,
) -> tuple[list[Any], dict[str, Any]]:
    """Add one variant per record whose request section lists up to ``k`` hypotheses.

    Targets stay verbatim; records whose transcript has no second hypothesis are
    left alone (the single form already is their prompt).
    """
    if not enabled:
        return records, {}
    if not transcripts:
        raise LifeMateError(
            "N-best prompt format is enabled but no transcripts were loaded\n"
            f"produce them first with:\n  {_PRODUCER}"
        )
    out: list[Any] = []
    n_created = n_missing = n_thin = n_no_section = 0
    for record in records:
        out.append(record)
        sample_id = str(_get(record, "sample_id") or "")
        transcript = transcripts.get(sample_id)
        if transcript is None:
            n_missing += 1
            continue
        if len(transcript.hypotheses) < 2:
            n_thin += 1
            continue
        section = render_request_section(transcript, nbest_format=True, k=k)
        prompt = replace_request_section(_prompt_of(record), section)
        if prompt is None:
            n_no_section += 1
            continue
        out.append(
            _with_prompt(
                record,
                prompt,
                {
                    "augmentation": "nbest_format",
                    "asr_model": transcript.asr_model,
                    "asr_revision": transcript.asr_revision,
                    "nbest_k": max(1, int(k)),
                    "n_hypotheses_shown": min(max(1, int(k)), len(transcript.hypotheses)),
                    "augmentation_seed": int(seed),
                },
            )
        )
        n_created += 1
    report = {
        "enabled": True,
        "k": max(1, int(k)),
        "seed": int(seed),
        "n_records_in": len(records),
        "n_variants_created": n_created,
        "n_missing_transcript": n_missing,
        "n_single_hypothesis": n_thin,
        "n_prompt_without_request_section": n_no_section,
    }
    return out, report
