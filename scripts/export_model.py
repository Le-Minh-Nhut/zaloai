#!/usr/bin/env python3
"""Export a checkpoint for submission: merged weights + tokenizer + metadata.

Usage::

    python scripts/export_model.py --ckpt outputs/checkpoints/full --output outputs/export
    python scripts/export_model.py --ckpt outputs/checkpoints/lora --output outputs/export_lora --merge-lora

The exported directory can be loaded with ``--model-mode full --checkpoint
<export dir>``. ``export_meta.json`` records parameter counts, dtype bytes and an
estimated on-disk size so a reviewer can sanity-check the artifact without torch.

Exit codes: 0 ok, 2 checkpoint missing/invalid, 3 torch/peft missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402

EXIT_OK = 0
EXIT_CHECKPOINT = 2
EXIT_DEPS = 3

_DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Export a LifeMate checkpoint.")
    parser.add_argument("--ckpt", default="outputs/checkpoints/full", help="checkpoint directory")
    parser.add_argument("--output", default="outputs/export", help="export directory")
    parser.add_argument("--merge-lora", action="store_true", help="merge a LoRA adapter into the base model")
    parser.add_argument("--base-model", default=None, help="base model name/id (default: config model.name)")
    parser.add_argument("--dtype", default=None, help="torch dtype for the merged model (default: config)")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    return parser.parse_args(argv)


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parameter_accounting(model: Any, dtype_name: str) -> dict[str, Any]:
    """Parameter counts and estimated weight bytes (real numbers, no guesses)."""
    try:
        total = sum(int(p.numel()) for p in model.parameters())
        trainable = sum(int(p.numel()) for p in model.parameters() if getattr(p, "requires_grad", False))
    except Exception as exc:  # pragma: no cover - exotic model objects
        return {"available": False, "reason": f"cannot count parameters: {exc}"}
    bytes_per = _DTYPE_BYTES.get(str(dtype_name).lower())
    return {
        "available": True,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "dtype": dtype_name,
        "bytes_per_parameter": bytes_per,
        # Estimated: safetensors stores raw tensor bytes + a small header.
        "estimated_weight_bytes": None if bytes_per is None else int(total * bytes_per),
        "estimated_weight_gib": None if bytes_per is None else round(total * bytes_per / 1024**3, 3),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run the export and return the metadata payload."""
    from lifemate.models.checkpoint import load_checkpoint_meta, merge_lora_adapter, resolve_checkpoint

    cfg = load_config(args.config)
    ckpt = Path(args.ckpt)
    output = Path(args.output)
    if not ckpt.exists():
        raise LifeMateError(f"checkpoint path does not exist: {ckpt}")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise LifeMateError(
            f"output directory {output} is not empty; pass --force to overwrite (nothing was changed)"
        )

    info = resolve_checkpoint(ckpt)
    kind = info.kind
    ckpt = Path(info.path)
    source_meta = load_checkpoint_meta(ckpt)
    adapter_cfg_file = ckpt / "adapter_config.json"
    discovered_base = None
    if adapter_cfg_file.is_file():
        try:
            discovered_base = json.loads(adapter_cfg_file.read_text()).get("base_model_name_or_path")
        except Exception:
            pass
    if not discovered_base and source_meta:
        discovered_base = source_meta.get("model_name")
    base_model = args.base_model or discovered_base or cfg.model.name
    discovered_dtype = None
    base_config_file = Path(base_model) / "config.json"
    if base_config_file.is_file():
        try:
            discovered_dtype = json.loads(base_config_file.read_text()).get("torch_dtype")
        except Exception:
            pass
    dtype = args.dtype or discovered_dtype or (source_meta.get("config", {}).get("training", {}).get("precision") if source_meta else None) or cfg.model.torch_dtype

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        return _deps_error(exc)

    if args.merge_lora:
        try:
            import peft  # noqa: F401
        except ImportError as exc:
            _deps_error(exc, "peft")
        if kind != "lora":
            raise LifeMateError(f"--merge-lora requires a LoRA adapter checkpoint, got {kind!r}")
        merged = merge_lora_adapter(base_model, ckpt, output, torch_dtype=dtype, force=args.force)
        # source_meta already loaded above
        payload_source = f"merged LoRA adapter from {ckpt}"
        model = AutoModelForCausalLM.from_pretrained(str(merged), torch_dtype=getattr(torch, dtype, None))
        tokenizer = AutoTokenizer.from_pretrained(str(merged))
    else:
        if kind == "lora":
            raise LifeMateError(
                f"{ckpt} is a LoRA adapter; pass --merge-lora to merge it with {base_model} first"
            )
        output.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=getattr(torch, dtype, None))
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))
        source_meta = load_checkpoint_meta(ckpt)
        payload_source = str(ckpt)

    accounting = _parameter_accounting(model, dtype)
    actual = _dir_size(output)
    meta = {
        "source": payload_source,
        "kind": "merged_lora" if args.merge_lora else "full",
        "base_model": base_model,
        "dtype": dtype,
        "accounting": accounting,
        "files": sorted(item.name for item in output.rglob("*") if item.is_file()),
        "actual_bytes": actual,
        "checkpoint_meta": source_meta,
        "how_to_run": (
            f"python scripts/infer.py --model-mode full --checkpoint {output} "
            "--split public_test --data-root ./data --output submission.jsonl"
        ),
    }
    (output / "export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def _deps_error(exc: Exception, package: str = "torch + transformers") -> NoReturn:
    """Print an actionable dependency error and exit 3."""
    print(
        f"ERROR: export requires {package}: pip install {package.replace(' + ', ' ')} "
        f"(import failed: {exc})",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_DEPS)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        meta = export(args)
    except SystemExit as exc:
        return int(exc.code or EXIT_DEPS)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT
    accounting = meta.get("accounting", {})
    print(f"exported to {args.output}")
    if accounting.get("available"):
        print(
            f"  parameters: {accounting['total_parameters']:,} "
            f"({accounting['dtype']}, ~{accounting.get('estimated_weight_gib')} GiB estimated)"
        )
    print(f"  files: {len(meta['files'])}, on disk: {meta['actual_bytes'] / 1024**2:.1f} MiB")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
