#!/usr/bin/env python3
"""Fetch a checkpoint: W&B artifact, local copy, or symlink.

Usage::

    python scripts/download_checkpoint.py --artifact lifemate-full:v3 --dest outputs/checkpoints
    python scripts/download_checkpoint.py --artifact ent/proj/lifemate-full:latest --project lifemate --entity me
    python scripts/download_checkpoint.py --from-uri /mnt/checkpoints/full --dest outputs/checkpoints --link

Prints the resulting manifest (files, sizes, metadata hashes). Never prints
credentials: the W&B token is read by the SDK from the environment/config file.

Exit codes: 0 ok, 2 source missing / download failed, 3 wandb missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402
from lifemate.tracking.artifact_manager import checkpoint_manifest, list_local_checkpoints  # noqa: E402

EXIT_OK = 0
EXIT_SOURCE = 2
EXIT_WANDB = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Download or copy a LifeMate checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", help="W&B artifact reference, e.g. name:version")
    source.add_argument("--from-uri", help="local path or file:// URI to copy (or link)")
    parser.add_argument("--dest", default="outputs/checkpoints", help="destination directory")
    parser.add_argument("--project", default=None, help="W&B project (default: config wandb.project)")
    parser.add_argument("--entity", default=None, help="W&B entity (default: config wandb.entity)")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--link", action="store_true", help="symlink instead of copying (--from-uri only)")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    return parser.parse_args(argv)


def _local_transfer(source: Path, dest: Path, *, link: bool, force: bool) -> Path:
    if not source.exists():
        raise LifeMateError(f"source path does not exist: {source}")
    target = dest / source.name
    if target.exists() or target.is_symlink():
        if not force:
            raise LifeMateError(f"destination already exists: {target} (pass --force to replace)")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    dest.mkdir(parents=True, exist_ok=True)
    if link:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def _artifact_transfer(args: argparse.Namespace, cfg: Any) -> Path:
    try:
        import wandb  # noqa: F401
    except Exception as exc:
        print(f"ERROR: --artifact needs the wandb package: pip install wandb ({exc})", file=sys.stderr)
        raise SystemExit(EXIT_WANDB)
    from lifemate.tracking.artifact_manager import download_artifact

    project = args.project or getattr(cfg.wandb, "project", None) or "lifemate"
    entity = args.entity or getattr(cfg.wandb, "entity", None)
    ref = args.artifact
    if "/" not in str(ref):  # plain name:version -> qualify with project/entity
        ref = f"{entity}/{project}/{ref}" if entity else f"{project}/{ref}"
    try:
        result = download_artifact(ref, args.dest, cfg=cfg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_SOURCE)
    except Exception as exc:
        print(
            f"ERROR: artifact download failed for {ref!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_SOURCE)
    return Path(result["path"])


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_SOURCE

    destination = Path(args.dest)
    try:
        if args.from_uri:
            raw = str(args.from_uri)
            source = Path(raw[7:]) if raw.startswith("file://") else Path(raw)
            target = _local_transfer(source, destination, link=args.link, force=args.force)
        else:
            target = _artifact_transfer(args, cfg)
    except SystemExit as exc:
        return int(exc.code or EXIT_SOURCE)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_SOURCE

    manifest = checkpoint_manifest(target)
    print(f"checkpoint available at {target}")
    print(json.dumps(manifest, indent=2)[:4000])
    siblings = list_local_checkpoints(target.parent if target.is_dir() else target.parent)
    if siblings:
        print(f"local checkpoints under {target.parent}: " + ", ".join(item["name"] for item in siblings))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
