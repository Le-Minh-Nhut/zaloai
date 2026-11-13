"""Checkpoint persistence: save, inspect, version and merge.

Purpose of the versioning rule: **never overwrite a checkpoint silently**.
Writing into an existing non-empty directory without ``force=True`` lands in
``<dir>-v2``, ``-v3`` … and logs which path was chosen, so a training run can
never destroy the artifact that produced yesterday's submission.

All writes are atomic (temp file/dir + ``os.replace`` on the same filesystem).
Metadata is plain JSON: model name, mode, step, config, git commit, timestamp,
seed, dataset manifest hash — the exact set needed to reproduce a run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lifemate.errors import LifeMateError

__all__ = [
    "CheckpointInfo",
    "META_FILENAME",
    "load_checkpoint_meta",
    "merge_lora_adapter",
    "resolve_checkpoint",
    "save_checkpoint",
    "versioned_dir",
]

META_FILENAME = "lifemate_meta.json"
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")

# Metadata keys every checkpoint carries.
_META_KEYS = (
    "model_name",
    "model_mode",
    "step",
    "config",
    "git_commit",
    "timestamp",
    "seed",
    "dataset_manifest_hash",
)


@dataclass
class CheckpointInfo:
    """Result of :func:`resolve_checkpoint`."""

    kind: str  # "full" | "lora"
    path: str
    step: int | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form (``meta`` may be ``None`` when the file is absent)."""
        return {"kind": self.kind, "path": self.path, "step": self.step, "meta": self.meta}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    """Short HEAD hash, or ``None`` when not in a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def versioned_dir(output_dir: str | Path, *, force: bool = False) -> Path:
    """Return a writable target: ``output_dir`` or the next free ``-vN`` sibling.

    Never returns a path that already holds files unless ``force=True``.
    """
    target = Path(str(output_dir))
    if force or not target.exists() or not any(target.iterdir()):
        return target
    version = 2
    while True:
        candidate = target.with_name(f"{target.name}-v{version}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        version += 1


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temp file + ``os.replace`` (never a torn file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)  # atomic on the same filesystem
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _build_meta(output_dir: Path, meta: dict[str, Any], is_lora: bool, step: int | None) -> dict[str, Any]:
    """Metadata payload with a stable key set (missing values are ``None``).

    Every checkpoint carries the same keys so downstream tooling can read them
    without guarding for absence: model name, mode, step, config, git commit,
    timestamp, seed, dataset manifest hash.
    """
    payload: dict[str, Any] = {key: meta.get(key) for key in _META_KEYS}
    payload["model_mode"] = meta.get("model_mode") or ("lora" if is_lora else "full")
    if step is not None:
        payload["step"] = step
    if payload["timestamp"] is None:
        payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if payload["git_commit"] is None:
        payload["git_commit"] = _git_commit()
    payload["is_lora"] = bool(is_lora)
    payload["output_dir"] = str(output_dir)
    for key, value in meta.items():  # keep extras, never drop caller data
        payload.setdefault(key, value)
    return payload


def _step_from_name(path: Path) -> int | None:
    """Parse a training step from ``checkpoint-1500`` / ``step_1500`` names."""
    for part in reversed(path.parts):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            return int(digits)
    return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    meta: dict[str, Any],
    is_lora: bool = False,
    step: int | None = None,
    force: bool = False,
) -> Path:
    """Save model+tokenizer (+ metadata) and return the directory actually used.

    Existing non-empty target without ``force`` → next free ``<dir>-vN``.
    Raises :class:`LifeMateError` when the save itself fails; the partially
    written directory is left for inspection rather than deleted silently.
    """
    target = versioned_dir(output_dir, force=force)
    if target != Path(str(output_dir)):
        print(f"[checkpoint] {output_dir} already populated; writing version {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    if model is None or tokenizer is None:
        raise LifeMateError("save_checkpoint requires both a model and a tokenizer")

    staging = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=f".{target.name}."))
    try:
        model.save_pretrained(str(staging))
        tokenizer.save_pretrained(str(staging))
        _atomic_write_json(
            staging / META_FILENAME, _build_meta(target, dict(meta or {}), is_lora, step)
        )
        if target.exists():
            if force and any(target.iterdir()):
                # Explicitly requested overwrite: move the old tree aside first so
                # the rename below stays atomic and nothing is lost half-written.
                backup = versioned_dir(target.with_name(f"{target.name}-replaced"), force=False)
                os.replace(target, backup)
                print(f"[checkpoint] force=True: previous {target} moved to {backup}")
            else:
                target.rmdir()  # only ever empty here (versioned_dir guarantees it)
        os.replace(staging, target)  # atomic rename within one filesystem
    except Exception as exc:
        raise LifeMateError(
            f"failed to write checkpoint to {target}: {exc}. "
            "Check disk space and that the model object exposes save_pretrained(); "
            f"partial output left in {staging}"
        ) from exc
    return target


def load_checkpoint_meta(path: str | Path) -> dict[str, Any]:
    """Read ``lifemate_meta.json``; ``{}`` when absent, error when corrupt."""
    meta_path = Path(str(path))
    if meta_path.is_dir():
        meta_path = meta_path / META_FILENAME
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise LifeMateError(f"checkpoint metadata at {meta_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LifeMateError(f"checkpoint metadata at {meta_path} must be a JSON object")
    return data


def resolve_checkpoint(path: str | Path) -> CheckpointInfo:
    """Classify a checkpoint directory as ``full`` or ``lora``.

    A directory containing ``adapter_config.json`` (or adapter weights) is a
    LoRA adapter; a directory containing full weights/config is a full
    checkpoint. Anything else raises, naming what was looked for.
    """
    target = Path(str(path))
    if not target.exists():
        raise LifeMateError(f"checkpoint path does not exist: {target}")
    if target.is_file():
        raise LifeMateError(
            f"checkpoint path {target} is a file; pass the checkpoint directory"
        )
    meta = load_checkpoint_meta(target)
    step = meta.get("step")
    if step is None:
        step = _step_from_name(target)

    if any((target / name).exists() for name in _ADAPTER_FILES):
        return CheckpointInfo(kind="lora", path=str(target), step=step, meta=meta or None)

    if (target / "config.json").exists() or any(target.glob("*.safetensors")) or any(
        target.glob("pytorch_model*.bin")
    ):
        return CheckpointInfo(kind="full", path=str(target), step=step, meta=meta or None)
    candidates: list[tuple[int, Path]] = []
    for candidate in target.rglob("*"):
        if candidate.is_dir() and candidate != target:
            if any((candidate / name).exists() for name in _ADAPTER_FILES) or (
                (candidate / "config.json").exists()
                or any(candidate.glob("*.safetensors"))
                or any(candidate.glob("pytorch_model*.bin"))
            ):
                c_meta = load_checkpoint_meta(candidate)
                c_step = c_meta.get("step") if c_meta else _step_from_name(candidate)
                candidates.append((int(c_step) if c_step is not None else -1, candidate))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return resolve_checkpoint(candidates[0][1])

    raise LifeMateError(
        f"cannot classify checkpoint {target}: expected adapter_config.json (LoRA) or "
        "config.json / *.safetensors / pytorch_model*.bin (full fine-tune)"
    )


def merge_lora_adapter(
    base_model_name: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    torch_dtype: str = "bfloat16",
    force: bool = False,
) -> Path:
    """Merge a LoRA adapter into its base model and save the merged weights.

    Requires ``peft`` + ``transformers`` + ``torch`` at call time. The merged
    directory can be loaded as a plain full checkpoint (no PEFT at inference).
    """
    adapter_path = Path(str(adapter_dir))
    if not adapter_path.exists():
        raise LifeMateError(f"LoRA adapter directory not found: {adapter_path}")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - env dependent
        raise LifeMateError(
            "merge_lora_adapter requires peft, transformers and torch: "
            f"pip install peft transformers torch ({exc})"
        ) from exc

    from lifemate.models.loader import is_local_or_offline, resolve_dtype

    dtype = resolve_dtype(torch_dtype)
    local_only = is_local_or_offline(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, local_files_only=local_only
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()

    target = versioned_dir(output_dir, force=force)
    if target != Path(str(output_dir)):
        print(f"[checkpoint] {output_dir} already populated; writing version {target}")
    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(target))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=local_only)
    tokenizer.save_pretrained(str(target))
    _atomic_write_json(
        target / META_FILENAME,
        _build_meta(
            target,
            {"model_name": base_model_name, "merged_from": str(adapter_path)},
            is_lora=False,
            step=_step_from_name(adapter_path),
        ),
    )
    return target
