"""History transcripts produced by ASR, injected into sample history turns.

The dataset ships a *text* history for ``train`` (``audio_text`` on every history
user turn) but ``public_test`` ships only the audio, so a test-time prompt would
carry an empty prior turn for 419/634 samples. ``outputs/asr/history_transcripts.jsonl``
holds real ASR output for those turns; this module loads it and fills the gap
without ever inventing text.

Rules enforced here:

* the file's ``text`` is ALREADY normalised (lower-cased by the same
  ``lower_if_uppercase_only`` pass as the final transcript): it is copied
  verbatim, never re-normalised;
* existing text is never overwritten — on ``train`` the gold ``audio_text`` wins;
* every ``audio`` path in the file must resolve under ``<data_root>/<split>/audio/``
  and pass :func:`lifemate.compliance.assert_path_allowed`. A row pointing
  anywhere else is a data-integrity failure and raises, it is not skipped;
* nothing here opens audio or any ``*_environment`` path.

Stdlib only, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..compliance import assert_path_allowed
from ..errors import LifeMateError

__all__ = [
    "HistoryTranscriptError",
    "apply_history_texts",
    "load_history_texts",
]


class HistoryTranscriptError(LifeMateError):
    """The history-transcript file is unusable (integrity failure, not a soft skip)."""


def _row_parts(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
    """``(split, id, turn_index)`` of one JSONL row, or ``None`` when malformed."""
    try:
        return str(row["split"]), str(row["id"]), int(row["turn_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _audio_path(row: Mapping[str, Any], split: str, data_root: str | Path | None) -> Path | None:
    """Validate the row's ``audio`` field; raise when it escapes the split's audio dir."""
    audio = row.get("audio")
    if not audio:
        return None
    if data_root is None:
        raise HistoryTranscriptError(
            f"history transcript row {row.get('id')!r} carries an 'audio' path but no data_root was "
            "given, so it cannot be validated against <split>/audio/"
        )
    allowed_root = (Path(data_root) / split / "audio").resolve()
    candidate = assert_path_allowed(audio)  # *_environment paths are refused outright
    absolute = candidate if candidate.is_absolute() else Path(data_root) / split / candidate
    resolved = absolute.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise HistoryTranscriptError(
            f"history transcript row {row.get('id')!r}: audio {audio!r} resolves to {resolved}, "
            f"outside {allowed_root}; refusing a file that points out of the split's audio directory"
        ) from exc
    return resolved


def load_history_texts(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    strict_audio: bool = True,
) -> dict[tuple[str, str, int], str]:
    """Read the ASR history JSONL into ``{(split, id, turn_index): text}``.

    Rows with an ``error`` or a blank ``text`` are dropped (they carry no usable
    transcript). Every row is validated: a malformed row or an ``audio`` path
    outside ``<data_root>/<split>/audio/`` raises :class:`HistoryTranscriptError`.
    """
    target = Path(path)
    if not target.is_file():
        raise HistoryTranscriptError(f"history transcripts file not found: {target}")
    texts: dict[tuple[str, str, int], str] = {}
    duplicates = 0
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HistoryTranscriptError(f"{target}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, Mapping):
                raise HistoryTranscriptError(f"{target}:{lineno}: expected a JSON object")
            key = _row_parts(row)
            if key is None:
                raise HistoryTranscriptError(
                    f"{target}:{lineno}: needs string 'split', string 'id' and integer 'turn_index'"
                )
            if strict_audio:
                _audio_path(row, key[0], data_root)
            if row.get("error") or not str(row.get("text") or "").strip():
                continue
            if key in texts:
                duplicates += 1
                continue
            texts[key] = str(row["text"])
    if duplicates:  # first occurrence wins; the file is regenerated, never merged
        raise HistoryTranscriptError(
            f"{target}: {duplicates} duplicate (split, id, turn_index) key(s); refusing an ambiguous file"
        )
    return texts


def _turns_of(record: Any) -> Iterable[Any]:
    """History turns of a ``Sample``, a plain record dict, or a list of them."""
    if isinstance(record, Mapping):
        return record.get("history") or []
    history = getattr(record, "history", None)
    if history is not None:
        return history or []
    raw = getattr(record, "raw", None)
    if isinstance(raw, Mapping):
        return raw.get("history") or []
    return []


def _writeable_turn(turn: Any) -> dict | None:
    """The dict to fill for one turn, or ``None`` when the turn cannot take text."""
    if isinstance(turn, dict):
        return turn
    raw = getattr(turn, "raw", None)
    return raw if isinstance(raw, dict) else None


def _turn_text(turn: Any) -> str:
    """Text already present on a turn (mirrors ``prompt_builder._turn_text``)."""
    if isinstance(turn, dict):
        sources = (turn,)
    else:
        sources = (getattr(turn, "raw", None) or {}, {"content": getattr(turn, "content", "")})
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("content", "audio_text", "transcript", "text", "asr_text", "utterance"):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _turn_role(turn: Any) -> str:
    if isinstance(turn, dict):
        return str(turn.get("role") or "user").lower()
    return str(getattr(turn, "role", None) or "user").lower()


def _sample_id(record: Any) -> str:
    if isinstance(record, Mapping):
        return str(record.get("id") or record.get("sample_id") or "")
    return str(getattr(record, "id", "") or "")


def apply_history_texts(
    samples_or_records: Iterable[Any],
    texts: Mapping[tuple[str, str, int], str],
    split: str,
) -> dict[str, int]:
    """Fill ``audio_text`` on history user turns that carry no text.

    ``turn_index`` counts the sample's USER turns from 1 (it matches the
    ``<id>_u<N>.wav`` file names). Existing text is preserved: on ``train`` the
    gold ``audio_text`` always wins over the ASR copy.

    Returns ``{"filled": n, "existing": n, "missing": n, "samples_with_history": n,
    "samples_touched": n}`` — ``missing`` counts user turns the file does not
    cover, so a silent partial application is impossible.
    """
    stats = {"filled": 0, "existing": 0, "missing": 0, "samples_with_history": 0, "samples_touched": 0}
    for record in samples_or_records:
        turns = _turns_of(record)
        user_turns = [turn for turn in turns if _turn_role(turn) == "user"]
        if not user_turns:
            continue
        stats["samples_with_history"] += 1
        sample_id = _sample_id(record)
        touched = False
        for index, turn in enumerate(user_turns, start=1):
            if _turn_text(turn):
                stats["existing"] += 1
                continue
            asr_text = texts.get((split, sample_id, index))
            target = _writeable_turn(turn)
            if not asr_text or target is None:
                stats["missing"] += 1
                continue
            target["audio_text"] = asr_text      # verbatim: already normalised upstream
            if not isinstance(turn, dict) and hasattr(turn, "content"):
                turn.content = asr_text           # keep HistoryTurn.content coherent
            stats["filled"] += 1
            touched = True
        if touched:
            stats["samples_touched"] += 1
    return stats
