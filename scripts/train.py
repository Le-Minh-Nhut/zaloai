#!/usr/bin/env python3
"""Train the LifeMate action policy with LoRA (default) or full-parameter SFT.

Usage:
    # LoRA (DEFAULT: the only mode that fits one consumer GPU)
    python3 scripts/train.py
    # full fine-tuning (explicit; needs ~80 GB, see configs/training_full.yaml)
    python3 scripts/train.py --config configs/training_full.yaml --tuning full
    # planning only: no weight loading
    python3 scripts/train.py --dry-run --param-count 4000000000 --tuning full
    # CPU smoke test: tiny model, 2 steps, synthetic records
    python3 scripts/train.py --smoke

Exit codes: 0 ok, 1 unexpected failure, 2 dataset not found or unreadable,
3 environment cannot run the requested configuration (no torch/CUDA, or the
model cannot be loaded). Never silently downgrades full → LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import TUNING_MODES, dump_config, load_config  # noqa: E402
from lifemate.errors import ConfigError, DatasetNotFoundError, LifeMateError  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NO_DATA = 2
EXIT_ENV = 3

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line interface (no CLI override; the config decides, and the config default is ``lora``)."""
    parser = argparse.ArgumentParser(
        description="Full-parameter (default) or LoRA SFT for the LifeMate action policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/training_lora.yaml",
        help="YAML config path (default: the LoRA config, the only mode that fits a "
        "single consumer GPU; pass configs/training_full.yaml explicitly for full FT)",
    )
    parser.add_argument(
        "--tuning",
        choices=TUNING_MODES,
        default=None,
        help="tuning mode; overrides training.tuning_mode from the config (default: from config, or 'lora'). "
        "'qlora' = LoRA on a 4-bit nf4 base and also sets training.quantization=nf4",
    )
    parser.add_argument("--dataset-dir", default=None, help="dir holding train.jsonl/val.jsonl")
    parser.add_argument("--model", default=None, help="override model.name (wins over YAML)")
    parser.add_argument("--output-dir", default=None, help="override training.output_dir")
    parser.add_argument(
        "--resume",
        "--resume-from",
        dest="resume",
        default=None,
        help="resume from this step-<n> checkpoint dir (alias: --resume-from)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after N optimizer steps (early stop; it does NOT define the LR schedule)",
    )
    parser.add_argument(
        "--planned-steps",
        type=int,
        default=None,
        help="LR schedule horizon in optimizer steps (default: derived from num_train_epochs). "
        "Persisted in every checkpoint so a resume continues the same trajectory",
    )
    parser.add_argument(
        "--extend-horizon",
        action="store_true",
        help="authorise resuming into a DIFFERENT horizon than the checkpoint recorded "
        "(without this a horizon mismatch is refused: the run would be a different schedule)",
    )
    parser.add_argument(
        "--best-effort-resume",
        action="store_true",
        help="accept a schema-1 checkpoint (no sampler position). The continuation uses a "
        "DIFFERENT sample order and is NOT equivalent to the uninterrupted run",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="checkpoint every N optimizer steps for save_strategy=steps "
        "(default: 0 = fall back to logging_steps)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="validate every N optimizer steps for eval_strategy=steps "
        "(default: 0 = fall back to logging_steps)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and preflight only")
    parser.add_argument(
        "--param-count",
        type=int,
        default=None,
        help="parameter count for --dry-run (avoids loading weights)",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny CPU model, 2 steps, synthetic data")
    parser.add_argument("--seed", type=int, default=None, help="override training.seed")
    parser.add_argument(
        "--run-name",
        default=None,
        help="checkpoint subdirectory name (default: the tuning mode, e.g. 'full'/'lora')",
    )
    parser.add_argument(
        "--seq-len",
        "--max-seq-length",
        dest="seq_len",
        type=int,
        default=None,
        help="override training.max_seq_length (alias: --max-seq-length)",
    )
    parser.add_argument("--device", default=None, help="override model.device (e.g. 'cpu' or 'cuda')")
    # --enable-feature / --disable-feature (optional slice; absent = no flags).
    try:
        from lifemate.features.overrides import add_feature_arguments  # noqa: PLC0415

        add_feature_arguments(parser)
    except ImportError:  # pragma: no cover - features slice absent
        pass
    return parser.parse_args(argv)

def _print_feature_report(cfg: Any) -> dict[str, Any]:
    """Print the effective feature configuration once; return its JSON half.

    ``{}`` when the optional features slice is absent, so the summary/schema stay
    stable on a checkout without it.
    """
    try:  # noqa: PLC0415
        from lifemate.features.report import (  # noqa: PLC0415
            feature_report_dict,
            format_feature_report,
        )
    except ImportError:  # pragma: no cover - features slice absent
        return {}
    if getattr(cfg, "features", None) is None:  # pragma: no cover - slice present but off
        return {}
    report = feature_report_dict(
        cfg.features, model_id=cfg.model.name, asr_id=cfg.asr.model_id
    )
    print(
        format_feature_report(
            cfg.features, model_id=cfg.model.name, asr_id=cfg.asr.model_id
        )
    )
    return report

def _refuse_non_training_features(args: argparse.Namespace, cfg: Any) -> str | None:
    """Message when the CLI names a feature that cannot affect a TRAINING run.

    An inference-only technique changes no supervised token, so accepting
    ``--enable-feature asr.multi_asr`` here would silently promise an experiment
    the run does not perform. Names the flag, the id and its scope.
    """
    requested: list[tuple[str, str]] = []
    for flag, values in (
        ("--enable-feature", getattr(args, "enable_feature", None) or []),
        ("--disable-feature", getattr(args, "disable_feature", None) or []),
    ):
        for value in values:
            requested.append((flag, str(value).split("=", 1)[0]))
    if not requested:
        return None
    try:  # noqa: PLC0415
        from lifemate.features.registry import FEATURES  # noqa: PLC0415
    except ImportError:  # pragma: no cover - features slice absent
        return None
    by_id = {spec.id: spec for spec in FEATURES}
    problems: list[str] = []
    for flag, feature_id in requested:
        spec = by_id.get(feature_id)
        if spec is None:
            problems.append(f"{flag} {feature_id}: unknown feature id")
            continue
        if spec.scope == "inference":
            problems.append(
                f"{flag} {feature_id}: this flag is inference-only (scope=inference) and cannot "
                "change a training run; drop it from the training command and pass it to "
                "scripts/infer.py / scripts/evaluate.py instead"
            )
    return "; ".join(problems) if problems else None

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LifeMateError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
    return records

def load_records(dataset_dir: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Load ``train.jsonl``/``val.jsonl`` + ``manifest.json`` from a dataset dir."""
    root = Path(str(dataset_dir))
    train_path = root / "train.jsonl"
    val_path = root / "val.jsonl"
    if not train_path.is_file():
        raise DatasetNotFoundError("train", str(train_path))
    manifest_path = root / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LifeMateError(f"{manifest_path}: invalid JSON ({exc.msg})") from exc
    conditioned = manifest.get("conditioned_multi_asr")
    if conditioned:
        for path, key in ((train_path, "train_sha256"), (val_path, "validation_sha256")):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != conditioned.get(key):
                raise LifeMateError(f"conditioned dataset checksum mismatch: {path}")
        for key in ("artifact_sha256", "prompt_version"):
            if not conditioned.get(key):
                raise LifeMateError(f"conditioned dataset missing provenance: {key}")
    train = _read_jsonl(train_path)
    val = _read_jsonl(val_path) if val_path.is_file() else []
    if not train:
        raise LifeMateError(f"{train_path} holds no SFT records; run scripts/prepare_dataset.py")
    return train, val, manifest

def _dataset_dir(args: argparse.Namespace, cfg: Any) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir)
    if args.smoke:
        return Path("/nonexistent")  # never read in smoke mode: records are synthesised
    return Path(args.dataset_dir or "outputs/datasets") / cfg.data.train_split

def _synthetic_records(n: int = 8) -> list[dict[str, Any]]:
    """Deterministic synthetic SFT records for the CPU smoke test."""
    records: list[dict[str, Any]] = []
    for index in range(n):
        tool = "alarm.set" if index % 2 else "alarm.list"
        target = (
            '<action>{"type": "tool_call", "name": "%s", "arguments": {}}</action>' % tool
            if index % 3
            else '<action>{"type": "finish"}</action>'
        )
        records.append(
            {
                "sample_id": f"smoke_{index}",
                "step_index": index,
                "messages": [
                    {"role": "system", "content": "Emit one action block."},
                    {"role": "user", "content": f"# Yêu cầu\nDat bao thuc so {index}.\n"},
                ],
                "target": target,
                "meta": {"synthetic": True},
            }
        )
    return records

# --------------------------------------------------------------------------- #
# Smoke model
# --------------------------------------------------------------------------- #
def build_smoke_model(vocab_size: int = 64):
    """Tiny ``LlamaForCausalLM``-shaped model on CPU (transformers + torch)."""
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415 - lazy

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(config)

class CharTokenizer:
    """Character-level tokenizer for smoke tests (deterministic, no downloads).

    Implements just enough of the HF surface for the collator:
    ``apply_chat_template``, ``encode``, ``eos_token_id``, ``pad_token_id``.
    """

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._offset = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._offset + (ord(ch) % (self.vocab_size - self._offset)) for ch in text]

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False) -> str:
        del tokenize
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def save_pretrained(self, path: str) -> None:  # pragma: no cover - smoke only
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text(
            json.dumps({"type": "char", "vocab_size": self.vocab_size}), encoding="utf-8"
        )

# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _preflight(cfg: Any, args: argparse.Namespace, model: Any = None) -> dict:
    """Run the heuristic memory preflight for the requested configuration."""
    from lifemate.training.memory_preflight import format_report, run_memory_preflight  # noqa: PLC0415

    seq_len = int(args.seq_len or cfg.training.max_seq_length)
    report = run_memory_preflight(
        model=model,
        param_count=args.param_count if model is None else None,
        cfg=cfg,
        seq_len=seq_len,
    )
    print(format_report(report))
    return report

#: Preflight verdicts that mean "this will not fit".
_NO_FIT_VERDICTS = frozenset({"does_not_fit"})


_FP16_WARNED = False


def _warn_fp16(cfg: Any) -> None:
    """One loud warning for any fp16-on-CUDA run; no sequence threshold guessed.

    The preflight heuristic UNDER-estimates this case (it projects 8.54 GB for
    LoRA at seq 8192, while the MEASURED float16 step reserved 15.14 GB during an
    earlier run), so the verdict
    guard below cannot be the only signal. On a bf16-capable card bf16 is
    strictly better: MEASURED 14.37 GB allocated on the train-only step (batch
    padded to exactly 8192) and a completed step, where float16 OOM'd on the
    first backward.
    """
    global _FP16_WARNED
    if _FP16_WARNED:
        return
    precision = str(getattr(cfg.training, "precision", "bfloat16") or "bfloat16").strip().lower()
    if precision not in {"float16", "fp16", "half"}:
        return
    if not str(getattr(getattr(cfg, "model", None), "device", "") or "").startswith("cuda"):
        return
    _FP16_WARNED = True
    print(
        "warning: precision=float16 uses GradScaler loss scaling. MEASURED on a "
        "16.57 GB RTX 5070 Ti: LoRA bf16 at seq 8192 (train-only step) completes at "
        "14.37 GB allocated, "
        "while the identical float16 run OOMs on the first backward (15.14 GB reserved, "
        "earlier measurement). "
        "Prefer precision: bfloat16 on a bf16-capable card."
    )


def _refuse_fp16_overcommit(cfg: Any, report: dict[str, Any]) -> str | None:
    """Message for an fp16 run whose preflight says it cannot fit, else ``None``.

    MEASURED on an RTX 5070 Ti (16.57 GB): one LoRA bf16 **train-only** step at
    seq 8192 (batch padded to exactly 8192 tokens) peaks at 14.37 GB allocated
    and completes; the same run at ``precision: float16``
    OOMs on the first backward ("Tried to allocate 510.00 MiB … 14.22 GiB in
    use"). fp16 keeps its own fp32 master/logit copies and gained nothing here,
    so refusing up front beats a multi-hour run that dies at step 1. The
    decision is the preflight VERDICT, not a hardcoded sequence limit.
    """
    precision = str(getattr(cfg.training, "precision", "bfloat16") or "bfloat16").strip().lower()
    if precision not in {"float16", "fp16", "half"}:
        return None
    if str(report.get("verdict") or "") not in _NO_FIT_VERDICTS:
        return None
    return (
        f"precision=float16 at max_seq_length={report.get('seq_len')} does not fit: the "
        f"preflight estimates {report.get('total_gb')} GB against "
        f"{report.get('available_gb')} GB free (verdict {report.get('verdict')}). "
        "MEASURED on a 16.57 GB RTX 5070 Ti: the identical LoRA run at seq 8192 "
        "peaks at 14.37 GB allocated in bf16 on the train-only step and SUCCEEDS, "
        "while float16 OOMs on the "
        "first backward (earlier measurement: peak 14.53 GB allocated, 15.14 GB reserved). float16 needs "
        "grad scaling and buys nothing here. Use precision: bfloat16 (or float32) on "
        "a bf16-capable card, or lower training.max_seq_length / "
        "per_device_train_batch_size until the preflight verdict is 'fits'."
    )

def _load_model(cfg: Any, args: argparse.Namespace):
    """Load the real base model + tokenizer for training.

    ``load_model_and_tokenizer`` is keyword-only, so every field is expanded
    explicitly here instead of passing the config object.

    ``checkpoint=None`` on purpose: ``--resume`` is owned by
    :meth:`SFTTrainer.resume`, which additionally verifies step continuity, the
    optimizer state, the dataset manifest hash and the tuning mode. Letting the
    loader load the checkpoint too would give the run two owners of resume
    semantics and, in LoRA mode, attach the adapter twice (peft refuses that).
    """
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415 - lazy

    # Full fine-tuning trains fp32 master weights (see
    # ``full_finetuning.master_dtype``): AdamW's update is quantised to the
    # *parameter* dtype, so a bf16 master loses every 5e-6-sized step. It must be
    # requested here — downcasting later would already have rounded the weights.
    # bf16 speed comes from autocast in the trainer. LoRA keeps the base at
    # ``precision`` because the base is frozen and only adapters train.
    # QLoRA stores the base as nf4 and computes in ``precision``; the k-bit
    # preparation (peft) decides the trainable dtypes, so no fp32 master is
    # requested here.
    adapter_mode = cfg.training.tuning_mode in ("lora", "qlora")
    load_dtype = cfg.training.precision if adapter_mode else "float32"
    try:
        bundle = load_model_and_tokenizer(
            cfg.model.name,
            torch_dtype=load_dtype,
            device=cfg.model.device,
            tuning_mode=cfg.training.tuning_mode,
            checkpoint=None,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
            trainable=True,
            for_training=True,
            max_new_tokens=cfg.agent.max_new_tokens,
            quantization=cfg.training.quantization,
        )
    except Exception as exc:  # heavy ML stack raises a wide variety
        raise LifeMateError(
            "cannot load the base model: "
            f"{type(exc).__name__}: {exc}\n"
            "This workstation has no model weights and no CUDA-enabled torch build. Use --dry-run / --smoke here "
            "and run the real job on the cloud GPU (see docs/TRAINING_GUIDE.md)."
        ) from exc
    model = getattr(bundle, "model", None)
    tokenizer = getattr(bundle, "tokenizer", None)
    if model is None or tokenizer is None:
        raise LifeMateError("model loader returned no model/tokenizer")
    return model, tokenizer

def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415 - lazy
    except Exception as exc:
        raise LifeMateError(
            f"torch is not installed ({exc}); real training needs torch. "
            "Use --dry-run or --smoke on this machine."
        ) from exc
    return torch

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    started = time.time()

    try:
        cfg = load_config(args.config)
    except (ConfigError, LifeMateError) as exc:
        print(f"invalid config {args.config}: {exc}")
        return EXIT_FAIL

    # --tuning CLI flag overrides config only when explicitly passed
    if args.tuning is not None and args.tuning != cfg.training.tuning_mode:
        print(f"tuning mode: {cfg.training.tuning_mode} (config) -> {args.tuning} (--tuning)")
        cfg.training.tuning_mode = args.tuning
    if args.tuning is not None:
        # Quantization follows the mode explicitly: qlora means a 4-bit base,
        # every other mode means full-precision weights. Never implied.
        cfg.training.quantization = "nf4" if args.tuning == "qlora" else "none"
    if args.model:
        cfg.model.name = args.model
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.seed is not None:
        cfg.training.seed = int(args.seed)
    if args.seq_len is not None:
        cfg.training.max_seq_length = int(args.seq_len)
    if args.resume:
        cfg.training.resume_from_checkpoint = args.resume
    if args.device:
        cfg.model.device = args.device
    if args.save_steps is not None:
        cfg.training.save_steps = int(args.save_steps)
        # Without this the cadence is inert: the trainer only honours
        # save_steps when save_strategy == "steps".
        if int(args.save_steps) > 0:
            cfg.training.save_strategy = "steps"
    if args.eval_steps is not None:
        cfg.training.eval_steps = int(args.eval_steps)
        if int(args.eval_steps) > 0:
            cfg.training.eval_strategy = "steps"

    # Features: CLI beats env beats YAML beats defaults. Training-scope ids change
    # this run; inference-only ids are refused here because silently accepting them
    # would imply they changed the supervised data.
    try:
        from lifemate.features.overrides import build_feature_config  # noqa: PLC0415

        cfg.features, _feature_provenance = build_feature_config(cfg, args)
        refused = _refuse_non_training_features(args, cfg)
        if refused:
            print(f"configuration error: {refused}")
            return EXIT_FAIL
    except LifeMateError as exc:
        print(f"configuration error: {exc}")
        return EXIT_FAIL
    except ImportError:  # pragma: no cover - features slice absent
        _feature_provenance = {}
    if args.smoke:
        cfg.model.device = "cpu"
        cfg.training.precision = "float32"
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 2
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.gradient_checkpointing = False
        cfg.training.save_total_limit = 1
        cfg.training.max_seq_length = min(int(cfg.training.max_seq_length), 512)
        cfg.training.logging_steps = 1

    print(f"tuning mode : {cfg.training.tuning_mode}  (default: lora)")
    print(f"model       : {cfg.model.name}  device={cfg.model.device} precision={cfg.training.precision}")
    print(f"output dir  : {cfg.training.output_dir}")
    feature_report = _print_feature_report(cfg)

    # ------------------------------------------------------------------ data
    train_records: list[dict] = []
    val_records: list[dict] = []
    manifest: dict = {}
    if args.smoke and not args.dataset_dir:
        train_records = _synthetic_records(8)
        val_records = _synthetic_records(2)
        manifest = {"synthetic": True, "n_train": len(train_records)}
        print(f"smoke mode  : {len(train_records)} synthetic train / {len(val_records)} val records")
    elif args.dry_run and not args.dataset_dir:
        print("dataset     : not loaded (--dry-run without --dataset-dir)")
    else:
        dataset_dir = _dataset_dir(args, cfg)
        try:
            train_records, val_records, manifest = load_records(dataset_dir)
        except DatasetNotFoundError as exc:
            print(str(exc))
            print(
                "\nBuild it first:\n"
                "  python3 scripts/prepare_dataset.py --data-root ./data --split train "
                "--output outputs/datasets"
            )
            return EXIT_NO_DATA
        except LifeMateError as exc:
            print(f"failed to read dataset: {exc}")
            return EXIT_NO_DATA
        print(
            f"dataset     : {dataset_dir} "
            f"({len(train_records)} train / {len(val_records)} val records)"
        )

    # -------------------------------------------------------------- preflight
    if args.dry_run and args.param_count is None and not args.smoke:
        print("\n(--dry-run without --param-count: using the model profile's verified parameter count)")
    try:
        preflight = _preflight(cfg, args)
    except LifeMateError as exc:
        print(f"preflight failed: {exc}")
        return EXIT_FAIL
    _warn_fp16(cfg)
    refusal = _refuse_fp16_overcommit(cfg, preflight)
    if refusal:
        print(f"environment error: {refusal}")
        return EXIT_ENV

    if args.dry_run:
        plan = {
            "tuning_mode": cfg.training.tuning_mode,
            "model": cfg.model.name,
            "output_dir": cfg.training.output_dir,
            "max_seq_length": cfg.training.max_seq_length,
            "num_train_epochs": cfg.training.num_train_epochs,
            "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "n_train_records": len(train_records),
            "n_val_records": len(val_records),
            "planned_total_steps": args.planned_steps,
            "stop_after_steps": args.max_steps,
            "save_steps": cfg.training.save_steps,
            "eval_steps": cfg.training.eval_steps,
            "dataset_manifest": {
                key: manifest.get(key) for key in ("n_train", "n_val", "seed", "ratio", "group_hash")
            },
        }
        print("\nplan (dry run, no model loaded, no training launched)")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"dry run complete in {time.time() - started:.2f}s")
        return EXIT_OK

    # ----------------------------------------------------------------- model
    try:
        torch = _require_torch()
    except LifeMateError as exc:
        print(f"environment error: {exc}")
        return EXIT_ENV

    try:
        if args.smoke:
            model = build_smoke_model()
            tokenizer = CharTokenizer()
            print(f"smoke model : {sum(p.numel() for p in model.parameters()):,} parameters (CPU)")
        else:
            model, tokenizer = _load_model(cfg, args)
    except LifeMateError as exc:
        print(f"environment error: {exc}")
        return EXIT_ENV
    except ImportError as exc:
        print(
            f"environment error: {exc}. Real training needs torch+transformers "
            "(pip install -e '.[train]'). Use --dry-run or --smoke here."
        )
        return EXIT_ENV
    except Exception as exc:  # model download/config failures
        print(f"environment error: cannot build the model ({type(exc).__name__}: {exc})")
        return EXIT_ENV

    # ------------------------------------------------------------- configure
    is_lora = cfg.training.tuning_mode in ("lora", "qlora")
    try:
        if is_lora:
            from lifemate.training import lora_finetuning  # noqa: PLC0415

            if cfg.training.tuning_mode == "qlora":
                report = lora_finetuning.configure_qlora(model, tokenizer, cfg)
            else:
                report = lora_finetuning.configure_lora(model, cfg)
            model = report.pop("model")
            print(
                f"LoRA        : r={report['r']} alpha={report['alpha']} "
                f"targets={report['n_target_modules']} ({', '.join(report['target_modules'])}) "
                f"trainable={report['trainable_params']:,} ({report['trainable_percentage']}%) "
                f"grad_ckpt={report['gradient_checkpointing']}"
            )
        else:
            from lifemate.training.full_finetuning import configure_full_finetuning  # noqa: PLC0415

            report = configure_full_finetuning(model, tokenizer, cfg)
            print(
                f"full FT     : trainable={report['trainable_params']:,} / "
                f"total={report['total_params']:,} (all parameters)"
            )
            if report.get("multimodal_towers"):
                print(
                    f"multimodal  : policy={report.get('multimodal_policy', 'everything')} "
                    f"(towers: {', '.join(report['multimodal_towers'])})"
                )
    except LifeMateError as exc:
        print(f"configuration error: {exc}")
        return EXIT_FAIL

    # Second preflight, now that the real adapter/trainable set exists: the first
    # call ran before any weights were loaded, so it could only report the 0.5%
    # placeholder. This one is the number the run actually has to fit.
    try:
        preflight = _preflight(cfg, args, model=model)
    except LifeMateError as exc:
        print(f"preflight failed: {exc}")
        return EXIT_FAIL
    _warn_fp16(cfg)
    # ``report`` below is the adapter/full-FT report and goes into the run summary:
    # keep the preflight in its own name so it can never clobber it.
    refusal = _refuse_fp16_overcommit(cfg, preflight)
    if refusal:
        print(f"environment error: {refusal}")
        return EXIT_ENV

    # ---------------------------------------------------------------- train
    from lifemate.training.callbacks import StopController  # noqa: PLC0415
    from lifemate.training.collator import SupervisedCollator  # noqa: PLC0415
    from lifemate.training.trainer import (  # noqa: PLC0415
        SFTTrainer,
        TrainingInterrupted,
        set_seed,
    )

    set_seed(cfg.training.seed)
    tracker = _build_tracker(cfg, args, run_name=args.run_name or cfg.training.tuning_mode)
    try:
        collator = SupervisedCollator.from_tokenizer(tokenizer)
    except LifeMateError as exc:
        print(f"configuration error: {exc}")
        return EXIT_FAIL

    manifest_hash = (hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                     if manifest.get("conditioned_multi_asr") else manifest.get("group_hash"))
    # SIGINT/SIGTERM set a cooperative stop flag: the current optimizer step
    # finishes, an interrupt-<step> checkpoint is written, exit code 130.
    stop = StopController().install()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        collator=collator,
        train_records=train_records,
        val_records=val_records,
        cfg=cfg,
        tracker=tracker,
        device=cfg.model.device if not args.smoke else "cpu",
        run_name=args.run_name or cfg.training.tuning_mode,
        # ``--max-steps`` is an early stop, never the LR horizon: a 2-step probe
        # must not define a 2-step cosine schedule for the real run.
        stop_after_steps=args.max_steps or (2 if args.smoke else None),
        planned_total_steps=args.planned_steps,
        extend_horizon=bool(args.extend_horizon),
        best_effort=bool(args.best_effort_resume),
        is_lora=is_lora,
        dataset_manifest_hash=manifest_hash,
        stop=stop,
    )

    if args.resume:
        try:
            step = trainer.resume(args.resume)
        except LifeMateError as exc:
            # Never silently start from zero when --resume was requested.
            print(f"resume failed: {exc}")
            return EXIT_FAIL
        mode = "exact" if trainer.exact_resume else "best-effort"
        print(f"resumed     : {args.resume} at step {step}")
        print(f"resume mode : {mode}")
        print(
            f"epoch       : {trainer._epoch_index}/{int(math.ceil(trainer.epochs))}  "
            f"next micro-batch: {trainer._cursor}"
        )
        horizon = int(trainer.planned_total_steps or 0)
        print(
            f"scheduler   : {'rebuilt' if trainer.schedule_modified else 'restored'} "
            f"(horizon {horizon})"
        )
        print("rng         : python+cpu+cuda restored")

    try:
        result = trainer.run()
    except TrainingInterrupted as exc:
        print(f"interrupted at step {exc.step}; checkpoint: {exc.checkpoint_dir}")
        print(
            f"resume with: python3 scripts/train.py --config {args.config} "
            f"--resume {exc.checkpoint_dir}"
        )
        return exc.exit_code
    except LifeMateError as exc:
        print(f"training failed: {exc}")
        return EXIT_FAIL
    finally:
        stop.uninstall()

    print(
        f"done in {result.runtime_s:.1f}s: steps={result.steps} epochs={result.epochs} "
        f"train_loss={result.final_loss} val_loss={result.val_loss}"
    )
    print(f"checkpoints : {result.checkpoint_dir}")

    summary_path = Path(cfg.training.output_dir) / "train_summary.json"
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "result": result.to_dict(),
                    "report": report,
                    "tuning_mode": cfg.training.tuning_mode,
                    "model": cfg.model.name,
                    "dataset_manifest": manifest,
                    # Resume provenance: a modified schedule / approximate
                    # continuation must never be mistaken for a clean run.
                    "schedule_modified": bool(result.schedule_modified),
                    "exact_resume": bool(result.exact_resume),
                    "planned_total_steps": result.planned_total_steps,
                    "stop_after_steps": result.stop_after_steps,
                    "skipped_steps": result.skipped_steps,
                    # Human-readable half of the feature identity; the fingerprint
                    # itself is in training_state.json / lifemate_meta.json.
                    "features": feature_report,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        dump_config(cfg, Path(cfg.training.output_dir) / "effective_config.yaml")
        print(f"summary     : {summary_path}")
    except OSError as exc:
        print(f"warning: could not write the run summary: {exc}")

    return EXIT_OK

def _build_tracker(cfg: Any, args: argparse.Namespace, run_name: str | None = None):
    """Tracker for this run, honouring ``training.report_to``.

    ``"wandb"`` (or ``wandb.enabled: true``) delegates to
    :class:`lifemate.tracking.WandbTracker`, which owns the redaction policy:
    secrets (``api_key``/``token``/``secret``/``password`` …) are replaced with
    ``[REDACTED]`` and PII keys (``user_profile``/``audio_text``/``messages`` …)
    are dropped. Never hand-roll ``wandb.init`` here — an unredacted config would
    put user profile contents on a third-party service.

    The tracking import is lazy and failure-tolerant. If it cannot be imported
    the run falls back to :class:`lifemate.training.callbacks.JsonlTracker`
    (metrics still recorded locally, one warning), because losing metrics is
    worse than losing a dashboard — and resurrecting a raw ``wandb.init`` path
    would silently leak.
    """
    from lifemate.training.callbacks import JsonlTracker, NullTracker  # noqa: PLC0415

    wandb_cfg = getattr(cfg, "wandb", None)
    report_to = [str(item).lower() for item in (getattr(cfg.training, "report_to", []) or [])]
    enabled = bool(getattr(wandb_cfg, "enabled", False)) or "wandb" in report_to
    if args.smoke:
        return NullTracker()
    if not enabled:
        import os
        if os.environ.get("LIFEMATE_LOCAL_METRICS") == "1":
            return JsonlTracker(cfg.training.output_dir, run_name=run_name or cfg.training.tuning_mode)
        return NullTracker()

    name = run_name or getattr(wandb_cfg, "run_name", None) or cfg.training.tuning_mode
    try:
        from lifemate.tracking import WandbTracker  # noqa: PLC0415 - optional package
    except Exception as exc:  # tracking slice missing or broken
        print(
            f"warning: lifemate.tracking unavailable ({type(exc).__name__}: {exc}); "
            "falling back to a local JSONL metrics log"
        )
        return JsonlTracker(cfg.training.output_dir, run_name=name)

    tracker = WandbTracker(cfg, run_name=name, config=cfg.to_dict())
    if not getattr(tracker, "enabled", False):
        print(
            f"warning: W&B disabled ({getattr(tracker, 'disabled_reason', 'unknown')}); "
            f"metrics go to {Path(cfg.training.output_dir) / name / 'metrics.jsonl'}"
        )
        return JsonlTracker(cfg.training.output_dir, run_name=name)
    return tracker

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted")
        sys.exit(130)
