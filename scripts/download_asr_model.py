#!/usr/bin/env python3
"""Download Gipformer-68M-RNNT ONNX model weights and assets from Hugging Face Hub.

Usage:
    python scripts/download_asr_model.py --output-dir models/gipformer-68M-rnnt --quantization int8
    python scripts/download_asr_model.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.compliance import assert_path_allowed

PINNED_MODEL_ID = "g-group-ai-lab/gipformer-68M-rnnt"
PINNED_REVISION = "318322b009200f8e77a3751fa64638e5cc85a7dc"

# Exact file sizes from upstream repo at pinned revision
KNOWN_FILES: dict[str, int] = {
    "tokens.txt": 25847,
    "bpe.model": 270695,
    "config.json": 615,
    "encoder.int8.onnx": 70876359,
    "decoder.int8.onnx": 1308690,
    "joiner.int8.onnx": 1033417,
    "encoder.onnx": 261057847,
    "decoder.onnx": 5165084,
    "joiner.onnx": 4104465,
}

COMMON_FILES = ["tokens.txt", "bpe.model", "config.json"]
INT8_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]
FP32_FILES = ["encoder.onnx", "decoder.onnx", "joiner.onnx"]


def resolve_file_list(quantization: str) -> list[str]:
    files = list(COMMON_FILES)
    if quantization in ("int8", "both"):
        files.extend(INT8_FILES)
    if quantization in ("fp32", "both"):
        files.extend(FP32_FILES)
    return files


_TREE_API = "https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"


def fetch_pinned_hashes(
    model_id: str, revision: str, timeout: float = 20.0
) -> dict[str, tuple[str, str]] | None:
    """Filename -> (algorithm, hexdigest) as published for ``revision``.

    The Hub tree API carries two digest kinds and this returns whichever the file
    has, so no hash is ever hardcoded here:

    * LFS files (``.onnx``, ``bpe.model``): ``lfs.oid`` is the SHA-256 of the bytes.
    * plain files (``tokens.txt``, ``config.json``): ``oid`` is a git blob id,
      i.e. SHA-1 over ``b"blob {len}\\0" + bytes``. It is a content hash of the
      file, not of the enclosing tree, so it detects a substituted file.

    Returns ``None`` when the metadata cannot be obtained (offline, no CA bundle,
    private repo, bad revision); the caller then degrades to the size check.
    """
    import json
    import urllib.request  # noqa: PLC0415 — only reached by --verify

    url = _TREE_API.format(repo_id=model_id, revision=revision)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None

    hashes: dict[str, tuple[str, str]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("path")
        if not isinstance(name, str):
            continue
        lfs = entry.get("lfs") or {}
        lfs_oid = lfs.get("oid") if isinstance(lfs, dict) else None
        if isinstance(lfs_oid, str) and len(lfs_oid) == 64:
            hashes[name] = ("sha256", lfs_oid.lower())
        elif isinstance(entry.get("oid"), str) and len(entry["oid"]) == 40:
            hashes[name] = ("git-sha1", entry["oid"].lower())
    return hashes or None


def file_digest(path: Path, algorithm: str) -> str:
    """Digest one file: plain SHA-256, or a git blob SHA-1 for ``git-sha1``."""
    hasher = hashlib.sha1() if algorithm == "git-sha1" else hashlib.sha256()
    data = path.read_bytes()
    if algorithm == "git-sha1":
        hasher.update(b"blob " + str(len(data)).encode() + b"\0")
    hasher.update(data)
    return hasher.hexdigest()


def verify_files(
    output_path: Path, files: list[str], hashes: dict[str, tuple[str, str]] | None
) -> int:
    """Check downloaded files. Returns 0 when acceptable, 2 on a real mismatch.

    Hash-checked files are verified byte-exactly; files whose hash is unknown, and
    every file when the metadata is unreachable, fall back to the pinned size.
    The printed summary always states which of the two happened.
    """
    if hashes is None:
        print("hash verification unavailable (offline); falling back to pinned sizes.")

    hashed = 0
    size_only: list[str] = []
    failures: list[str] = []
    for name in files:
        local_file = output_path / name
        if not local_file.is_file():
            failures.append(f"{name}: missing")
            continue
        expected_sz = KNOWN_FILES.get(name)
        actual_sz = local_file.stat().st_size
        if expected_sz is not None and actual_sz != expected_sz:
            failures.append(f"{name}: size mismatch (expected {expected_sz} B, got {actual_sz} B)")
            continue

        published = hashes.get(name) if hashes else None
        if published is None:
            size_only.append(name)
            print(f"  OK      {name} (size {actual_sz:,d} B; no published hash)")
            continue
        algorithm, expected = published
        actual = file_digest(local_file, algorithm)
        if actual != expected:
            failures.append(
                f"{name}: {algorithm} mismatch (expected {expected}, got {actual})"
            )
            continue
        hashed += 1
        print(f"  OK      {name} ({algorithm} {actual[:16]}…)")

    for failure in failures:
        print(f"ERROR: verification failed: {failure}", file=sys.stderr)

    if failures:
        return 2
    if hashed:
        print(f"Verification OK: {hashed} file(s) hash-verified.")
    else:
        print("Verification OK (sizes only; no published hashes were available).")
    if size_only and hashed:
        print(f"  {len(size_only)} file(s) size-checked only: {', '.join(size_only)}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Gipformer-68M-RNNT model weights for offline inference."
    )
    parser.add_argument(
        "--output-dir",
        default="models/gipformer-68M-rnnt",
        help="Target directory for model files (default: models/gipformer-68M-rnnt)",
    )
    parser.add_argument(
        "--quantization",
        choices=("int8", "fp32", "both"),
        default="int8",
        help="Quantization flavour to download (default: int8)",
    )
    parser.add_argument(
        "--revision",
        default=PINNED_REVISION,
        help=f"Git commit revision (default pinned: {PINNED_REVISION})",
    )
    parser.add_argument(
        "--model-id",
        default=PINNED_MODEL_ID,
        help=f"Hugging Face Hub repo id (default: {PINNED_MODEL_ID})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify file sizes after download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print file list, sizes, and total MB without downloading",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output_dir)

    try:
        assert_path_allowed(output_path)
    except Exception as exc:
        print(f"ERROR: output directory path prohibited: {exc}", file=sys.stderr)
        return 2

    if "1.5" in args.model_id:
        print(
            f"ERROR: model_id {args.model_id!r} is rejected: gipformer1.5 is a different model.",
            file=sys.stderr,
        )
        return 2

    files = resolve_file_list(args.quantization)
    total_bytes = sum(KNOWN_FILES.get(f, 0) for f in files)
    total_mb = total_bytes / (1024 * 1024)

    if args.dry_run:
        print("Gipformer-68M-RNNT download plan (dry run):")
        print(f"  Model ID:     {args.model_id}")
        print(f"  Revision:     {args.revision}")
        print(f"  Output Dir:   {output_path}")
        print(f"  Quantization: {args.quantization}")
        print(f"  Files ({len(files)}):")
        for f in files:
            size_b = KNOWN_FILES.get(f, 0)
            print(f"    - {f:20s} {size_b:>12,d} B ({size_b / (1024 * 1024):>6.2f} MB)")
        print(f"  Total Size:   {total_bytes:,d} B ({total_mb:.2f} MB)")
        return 0

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: 'huggingface_hub' is required to download model weights. "
            "Install with: pip install \"huggingface-hub>=0.20.0\" or pip install -e \".[asr]\".",
            file=sys.stderr,
        )
        return 2

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} files ({total_mb:.2f} MB) to {output_path}...")
    for f in files:
        try:
            hf_hub_download(
                repo_id=args.model_id,
                filename=f,
                revision=args.revision,
                local_dir=str(output_path),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            print(f"ERROR: failed to download {f}: {exc}", file=sys.stderr)
            return 2

    if args.verify:
        print("Verifying downloaded files...")
        hashes = fetch_pinned_hashes(args.model_id, args.revision)
        return verify_files(output_path, files, hashes)

    print(f"Download complete: {len(files)} files, {total_mb:.2f} MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
