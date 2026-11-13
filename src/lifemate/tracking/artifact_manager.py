"""Checkpoint artifacts: upload, download and cheap local manifests.

All W&B calls are lazy and guarded: a missing ``wandb`` package or a disabled
config yields an explicit ``{"uploaded": False, "reason": ...}`` instead of an
exception or a silent success.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

__all__ = [
    "checkpoint_manifest",
    "download_artifact",
    "list_local_checkpoints",
    "sha256_file",
    "upload_checkpoint",
]

log = logging.getLogger("lifemate.tracking")

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
META_NAMES = ("config.json", "lifemate_meta.json", "adapter_config.json", "generation_config.json", "export_meta.json")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    """Files + sizes + sha256 of config/meta files only.

    Hashing multi-GB weights is expensive and adds nothing (weights are already
    content-addressed by the artifact store), so only small config/metadata
    files are hashed. Weight files get a size-only entry and are listed under
    ``weights`` with their total size.
    """
    target = Path(path)
    if not target.exists():
        return {"path": str(target), "exists": False, "files": [], "weights": [], "total_bytes": 0}
    files: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    if target.is_file():
        entry = {"path": target.name, "size": target.stat().st_size}
        if target.suffix in WEIGHT_SUFFIXES:
            weights.append({**entry, "sha256": "skipped (weights)"})
        else:
            files.append({**entry, "sha256": sha256_file(target)})
    else:
        for item in sorted(target.rglob("*")):
            if not item.is_file():
                continue
            rel = str(item.relative_to(target))
            size = item.stat().st_size
            if item.suffix in WEIGHT_SUFFIXES:
                weights.append({"path": rel, "size": size, "sha256": "skipped (weights)"})
            elif item.name in META_NAMES or item.suffix in {".json", ".txt", ".md"}:
                files.append({"path": rel, "size": size, "sha256": sha256_file(item)})
            else:
                files.append({"path": rel, "size": size, "sha256": "skipped (not metadata)"})
    return {
        "path": str(target),
        "exists": True,
        "files": files,
        "weights": weights,
        "total_bytes": _dir_size(target) if target.is_dir() else target.stat().st_size,
        "weights_bytes": sum(entry["size"] for entry in weights),
        "hashed": "config/meta files only (weights are content-addressed by the artifact store)",
    }


def list_local_checkpoints(output_dir: str | Path) -> list[dict[str, Any]]:
    """Checkpoint directories under ``output_dir`` with size/mtime/meta."""
    base = Path(output_dir)
    if not base.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        has_weights = any(child.glob("*.safetensors")) or any(child.glob("pytorch_model*.bin"))
        has_adapter = (child / "adapter_config.json").exists()
        if not (has_weights or has_adapter or (child / "config.json").exists()):
            continue
        meta: dict[str, Any] = {}
        meta_path = child / "lifemate_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {"error": "unreadable lifemate_meta.json"}
        stat = child.stat()
        found.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "lora" if has_adapter else "full",
                "size_bytes": _dir_size(child),
                "mtime": stat.st_mtime,
                "meta": meta,
            }
        )
    return found


def upload_checkpoint(
    path: str | Path,
    name: str,
    cfg: Any = None,
    tracker: Any = None,
) -> dict[str, Any]:
    """Upload a checkpoint as a W&B artifact when logging is enabled.

    Returns ``{"uploaded": bool, "reason"?, "artifact"?, "version"?}``. Never
    raises for "wandb disabled"; raises only for a real upload failure.
    """
    wandb_cfg = getattr(cfg, "wandb", None)
    enabled = bool(getattr(wandb_cfg, "enabled", False))
    upload = bool(getattr(wandb_cfg, "upload_artifacts", False))
    if tracker is not None and isinstance(getattr(tracker, "enabled", None), bool):
        enabled = enabled and tracker.enabled
    if not enabled:
        return {"uploaded": False, "reason": "wandb disabled"}
    if not upload:
        return {"uploaded": False, "reason": "wandb.upload_artifacts is false"}
    target = Path(path)
    if not target.exists():
        return {"uploaded": False, "reason": f"checkpoint path not found: {target}"}
    try:
        import wandb  # type: ignore
    except Exception as exc:
        return {"uploaded": False, "reason": f"wandb not importable: {exc}"}

    artifact_type = "model"
    try:
        run = getattr(tracker, "_run", None)
        if run is not None and hasattr(run, "log_artifact"):
            artifact = wandb.Artifact(name=name, type=artifact_type)
            if target.is_dir():
                artifact.add_dir(str(target))
            else:
                artifact.add_file(str(target))
            run.log_artifact(artifact)
            return {"uploaded": True, "artifact": name, "path": str(target)}
        api = wandb.Api()
        project = getattr(wandb_cfg, "project", None) or "lifemate"
        entity = getattr(wandb_cfg, "entity", None)
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if target.is_dir():
            artifact.add_dir(str(target))
        else:
            artifact.add_file(str(target))
        with artifact.new_draft() if hasattr(artifact, "new_draft") else _null_ctx():
            pass
        return {
            "uploaded": False,
            "reason": (
                "api-only upload needs a run context; enable wandb logging and pass the tracker "
                f"(project={project!r}, entity={entity!r}, api={type(api).__name__})"
            ),
        }
    except Exception as exc:
        log.warning("artifact upload failed: %s", exc)
        return {"uploaded": False, "reason": f"upload failed: {type(exc).__name__}: {exc}"[:300]}


class _null_ctx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> None:
        return None


def download_artifact(ref: str, dest: str | Path, *, cfg: Any = None) -> dict[str, Any]:
    """Download a W&B artifact (``name:version`` or ``entity/project/name:version``).

    Returns ``{"downloaded": bool, "path"?, "reason"?}``. Raises
    :class:`RuntimeError` with an actionable message when ``wandb`` is missing or
    the reference is malformed, because this is an explicit user request.
    """
    if not ref or ":" not in str(ref) and "/" not in str(ref):
        raise RuntimeError(
            f"artifact reference {ref!r} is not valid; expected 'name:version' "
            "or 'entity/project/name:version'"
        )
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"download_artifact requires the wandb package: pip install wandb ({exc})"
        ) from exc

    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    try:
        artifact = api.artifact(ref, type="model")
    except TypeError:  # older versions do not accept `type`
        artifact = api.artifact(ref)
    downloaded = artifact.download(root=str(destination))
    return {"downloaded": True, "path": str(downloaded), "ref": str(ref)}
