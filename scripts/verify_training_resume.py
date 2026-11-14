#!/usr/bin/env python3
"""Prove that an interrupted-and-resumed run is EXACTLY the uninterrupted one.

The experiment, per invocation:

* **A (reference)** — one fresh trainer runs to the planned horizon.
* **B (resumed)** — a fresh trainer with the same seed stops at ``--interrupt-at``
  steps (MID-EPOCH, on purpose), then a completely new trainer + new model object
  resumes from that checkpoint and runs to the same horizon.

Both branches record, per optimizer step: the consumed sample ids, the loss, the
LR, the step/epoch/cursor position. The comparison prints and writes the max
absolute weight difference, the AdamW ``exp_avg``/``exp_avg_sq`` difference, the
scheduler position and the ENTIRE per-step LR list, the full consumed sample
order, the loss trajectory and the end-of-run python/torch RNG state. Exit code
is non-zero when the comparison fails.

Fresh process for the resumed branch: ``--as-child`` makes this same script run
ONLY the resume half and dump its result as JSON, which the parent invokes through
``subprocess``. A resume that only works because of inherited in-process state
(module caches, a warm RNG, a mutated class attribute) therefore cannot pass.

Usage:
    PYTHONPATH=src python3 scripts/verify_training_resume.py --tiny --steps 12 \\
        --interrupt-at 5 --accum 3 --out outputs/reports/resume_equivalence.json

    # the real model (needs the checkpoint on disk / hub access):
    PYTHONPATH=src python3 scripts/verify_training_resume.py --real \\
        --model models/MiniCPM5-2B --steps 40 --interrupt-at 20 --accum 8

``--tiny`` builds a 2-layer d=32 LLaMA on CPU and synthetic records: no downloads,
no dataset, deterministic. Exit codes: 0 equivalent, 1 not equivalent or BLOCKED,
2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2

# --------------------------------------------------------------------------- #
# Fixtures (tiny CPU model + synthetic records; mirrors scripts/train.py)
# --------------------------------------------------------------------------- #
def build_tiny_model(vocab_size: int = 64):
    """2-layer d=32 ``LlamaForCausalLM`` on CPU, deterministically initialised."""
    import torch  # noqa: PLC0415
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config).to(torch.float32)
    # The trainer's own resume path reloads a plain state dict, so a checkpoint
    # dir must be reproducible from weights alone.
    return model

class CharTokenizer:
    """Character-level tokenizer: deterministic, no downloads."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._offset = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._offset + (ord(ch) % (self.vocab_size - self._offset)) for ch in text]

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str:
        del tokenize
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text(
            json.dumps({"type": "char", "vocab_size": self.vocab_size}), encoding="utf-8"
        )

def synthetic_records(n: int = 12) -> list[dict[str, Any]]:
    """Deterministic synthetic SFT records (same shape as scripts/train.py)."""
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
                    {"role": "user", "content": f"# Yeu cau\nDat bao thuc so {index}.\n"},
                ],
                "target": target,
                "meta": {"synthetic": True},
            }
        )
    return records

# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def build_config(*, accum: int, work: Path, real_model: str | None = None):
    from lifemate.config import load_config  # noqa: PLC0415

    cfg = load_config(None)
    cfg.training.precision = "float32"
    cfg.training.tuning_mode = "full"
    cfg.training.quantization = "none"
    cfg.training.gradient_checkpointing = False
    cfg.training.gradient_clip_norm = 1.0
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = max(1, int(accum))
    cfg.training.num_train_epochs = 2
    cfg.training.learning_rate = 1e-3
    cfg.training.warmup_ratio = 0.1
    cfg.training.lr_scheduler_type = "cosine"
    cfg.training.logging_steps = 1
    cfg.training.save_strategy = "steps"
    cfg.training.save_steps = 1
    cfg.training.eval_strategy = "no"
    cfg.training.save_total_limit = 0
    cfg.training.max_seq_length = 128
    cfg.training.seed = 0
    cfg.training.output_dir = str(work / "checkpoints")
    cfg.model.device = "cpu"
    cfg.model.name = real_model or "tiny-resume"
    return cfg

class _TracedTrainer:
    """Runs one SFTTrainer and records the full trajectory it consumed."""

    def __init__(self, *, cfg, model, tokenizer, records, work: Path, run_name: str, **kwargs):
        from lifemate.training.collator import SupervisedCollator  # noqa: PLC0415
        from lifemate.training.trainer import CheckpointManager, SFTTrainer  # noqa: PLC0415

        self.cfg = cfg
        self.model = model
        self.sample_order: list[int] = []
        self.losses: list[float] = []
        self.lrs: list[float] = []
        self.positions: list[dict[str, Any]] = []
        collator = SupervisedCollator.from_tokenizer(tokenizer)
        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            train_records=records,
            val_records=[],
            cfg=cfg,
            device="cpu",
            run_name=run_name,
            checkpoint_manager=CheckpointManager(
                work / "checkpoints", run_name=run_name, save_total_limit=0
            ),
            **kwargs,
        )
        self._tokenizer = tokenizer
        self._records = records
        self._wrap_collator()

    def _wrap_collator(self) -> None:
        """Record which example ids each consumed batch actually came from.

        ``to_torch_batch`` is called with the collated batch, so the trainer's own
        ``self.collator([train_examples[i] for i in chunk])`` is the only place the
        example ids are visible. Wrap the trainer's collator to capture the call
        argument list; the ids are the trainer's positions into its own example
        list, which is what a sample-order comparison must compare.
        """
        trainer = self.trainer
        original = trainer.collator
        seen: list[list[int]] = []

        def recorder(examples):
            ids = [int(ex.get("_verify_index", -1)) for ex in examples]
            seen.append(ids)
            return original(examples)

        trainer.collator = recorder  # type: ignore[assignment]
        self._seen = seen

    def prepare_indices(self) -> None:
        """Stamp a stable id on every tokenised example (once, in order)."""
        from lifemate.training.collator import build_supervised_example  # noqa: PLC0415

        examples = []
        for index, record in enumerate(self._records):
            example = build_supervised_example(
                self._tokenizer,
                list(record.get("messages") or []),
                str(record.get("target") or ""),
                int(self.cfg.training.max_seq_length),
            )
            example["_verify_index"] = index
            examples.append(example)
        self._examples = examples

    def run(self):
        trainer = self.trainer
        self.prepare_indices()
        # Feed the trainer pre-built examples (with ids) instead of raw records.
        trainer.train_records = self._examples
        original_prepare = trainer._prepare_examples

        def prepare(records, split):
            kept = [dict(item) for item in records if item.get("kept")]
            if not kept:
                from lifemate.errors import LifeMateError  # noqa: PLC0415

                raise LifeMateError(f"no usable {split} examples")
            return kept

        trainer._prepare_examples = prepare  # type: ignore[assignment]
        original_log = trainer._maybe_log

        def log(step, loss):
            original_log(step, loss)
            if trainer.optimizer is not None:
                self.lrs.append(round(float(trainer.optimizer.param_groups[0]["lr"]), 15))
            self.losses.append(loss)
            self.positions.append(
                {
                    "global_step": int(step),
                    "epoch_index": int(trainer._epoch_index),
                    "next_micro_index": int(trainer._cursor),
                }
            )

        trainer._maybe_log = log  # type: ignore[assignment]
        try:
            result = trainer.run()
        finally:
            trainer._prepare_examples = original_prepare  # type: ignore[assignment]
        self.sample_order = [i for batch in self._seen for i in batch]
        self.result = result
        return result

    # ------------------------------------------------------------ comparison
    def final_state(self) -> dict[str, Any]:
        import torch  # noqa: PLC0415

        trainer = self.trainer
        weights = {k: v.detach().clone() for k, v in speaker_dict(self.model).items()}
        optics: dict[str, Any] = {}
        moments = ("exp_avg", "exp_avg_sq")
        state = trainer.optimizer.state_dict().get("state", {})
        for key, entry in state.items():
            for name in moments:
                if name in entry:
                    optics[f"{key}:{name}"] = entry[name].detach().clone()
        scheduler = trainer.scheduler
        sched_state = scheduler.state_dict() if scheduler is not None else {}
        return {
            "weights": weights,
            "optics": optics,
            "scheduler_last_epoch": sched_state.get("last_epoch"),
            "scheduler_step_count": sched_state.get("_step_count"),
            "lr_list": list(self.lrs),
            "python_rng_state": list(__import__("random").getstate()[1]),
            "torch_rng_state": torch.get_rng_state().tolist(),
            "global_step": int(trainer.progress_state()["global_step"]),
            "epoch_index": int(trainer._epoch_index),
            "next_micro_index": int(trainer._cursor),
            "sample_order": list(self.sample_order),
            "losses": list(self.losses),
            "positions": list(self.positions),
            "skipped_steps": int(getattr(self.result, "skipped_steps", 0)),
        }

def speaker_dict(model: Any) -> dict[str, Any]:
    """state_dict of the underlying module (PEFT-safe)."""
    base = getattr(model, "module", None)
    if base is not None and hasattr(base, "state_dict"):
        return base.state_dict()
    return model.state_dict()

def _tensor_diff(a: Any, b: Any) -> tuple[float, float]:
    """``(max_abs, max_rel)`` over two tensors, ``0.0`` when both are empty."""
    import torch  # noqa: PLC0415

    if a is None and b is None:
        return 0.0, 0.0
    diff = (a - b).abs()
    scale = b.abs().clamp(min=1e-12)
    return float(diff.max().item()), float((diff / scale).max().item())

def compare(reference: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    """Every number the acceptance criterion asks for, plus a pass/fail verdict."""
    import torch  # noqa: PLC0415

    weight_abs = weight_rel = 0.0
    worst_weight = None
    missing = sorted(set(reference["weights"]) ^ set(resumed["weights"]))
    for name in sorted(set(reference["weights"]) & set(resumed["weights"])):
        a, b = reference["weights"][name], resumed["weights"][name]
        absdiff, reldiff = _tensor_diff(a, b)
        if absdiff > weight_abs:
            weight_abs, worst_weight = absdiff, name
        weight_rel = max(weight_rel, reldiff)

    opt_abs = 0.0
    worst_opt = None
    opt_keys = sorted(set(reference["optics"]) | set(resumed["optics"]))
    for key in opt_keys:
        absdiff, _ = _tensor_diff(reference["optics"].get(key), resumed["optics"].get(key))
        if absdiff > opt_abs:
            opt_abs, worst_opt = absdiff, key

    order_match = reference["sample_order"] == resumed["sample_order"]
    first_divergence = None
    for index, (a, b) in enumerate(zip(reference["sample_order"], resumed["sample_order"])):
        if a != b:
            first_divergence = index
            break
    lr_match = reference["lr_list"] == resumed["lr_list"]
    loss_abs = max(
        (abs(a - b) for a, b in zip(reference["losses"], resumed["losses"])), default=0.0
    )
    loss_len_match = len(reference["losses"]) == len(resumed["losses"])
    rng_py_match = reference["python_rng_state"] == resumed["python_rng_state"]
    rng_torch_match = reference["torch_rng_state"] == resumed["torch_rng_state"]

    checks = {
        "weights_bit_identical": weight_abs == 0.0 and not missing,
        "optimizer_moments_identical": opt_abs == 0.0,
        "scheduler_last_epoch_identical": (
            reference["scheduler_last_epoch"] == resumed["scheduler_last_epoch"]
        ),
        "scheduler_step_count_identical": (
            reference["scheduler_step_count"] == resumed["scheduler_step_count"]
        ),
        "lr_list_identical": lr_match,
        "sample_order_identical": order_match,
        "loss_trajectory_identical": loss_abs == 0.0 and loss_len_match,
        "global_step_identical": reference["global_step"] == resumed["global_step"],
        "position_identical": (
            reference["epoch_index"],
            reference["next_micro_index"],
        )
        == (resumed["epoch_index"], resumed["next_micro_index"]),
        "python_rng_identical": rng_py_match,
        "torch_rng_identical": rng_torch_match,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "weights": {
            "max_abs_diff": weight_abs,
            "max_rel_diff": weight_rel,
            "worst_tensor": worst_weight,
            "tensor_set_mismatch": missing[:10],
            "n_tensors": len(reference["weights"]),
        },
        "optimizer": {"max_abs_diff": opt_abs, "worst_entry": worst_opt, "n_entries": len(opt_keys)},
        "scheduler": {
            "reference": {
                "last_epoch": reference["scheduler_last_epoch"],
                "_step_count": reference["scheduler_step_count"],
                "lr_list": reference["lr_list"],
            },
            "resumed": {
                "last_epoch": resumed["scheduler_last_epoch"],
                "_step_count": resumed["scheduler_step_count"],
                "lr_list": resumed["lr_list"],
            },
        },
        "sample_order": {
            "identical": order_match,
            "lengths": [len(reference["sample_order"]), len(resumed["sample_order"])],
            "first_divergence_index": first_divergence,
            "reference_head": reference["sample_order"][:24],
            "resumed_head": resumed["sample_order"][:24],
        },
        "loss": {
            "max_abs_diff": loss_abs,
            "identical": loss_abs == 0.0 and loss_len_match,
            "reference": reference["losses"][:24],
            "resumed": resumed["losses"][:24],
        },
        "position": {
            "global_step": [reference["global_step"], resumed["global_step"]],
            "epoch_index": [reference["epoch_index"], resumed["epoch_index"]],
            "next_micro_index": [reference["next_micro_index"], resumed["next_micro_index"]],
        },
        "rng": {"python": rng_py_match, "torch": rng_torch_match},
        "skipped_steps": [reference["skipped_steps"], resumed["skipped_steps"]],
        "losses_count": [len(reference["losses"]), len(resumed["losses"])],
    }

# --------------------------------------------------------------------------- #
# Branches
# --------------------------------------------------------------------------- #
def _load_real(model_path: str, cfg):
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415

    bundle = load_model_and_tokenizer(
        model_path, device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True
    )
    return bundle.model, bundle.tokenizer

def run_reference(args, work: Path) -> dict[str, Any]:
    records = synthetic_records(args.records)
    cfg = build_config(accum=args.accum, work=work, real_model=args.model if args.real else None)
    if args.real:
        model, tokenizer = _load_real(args.model, cfg)
    else:
        model, tokenizer = build_tiny_model(), CharTokenizer()
    traced = _TracedTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        records=records,
        work=work,
        run_name="reference",
        planned_total_steps=args.steps,
    )
    traced.run()
    return traced.final_state()

def run_resumed(args, work: Path, corrupt_checkpoint=None) -> dict[str, Any]:
    """Interrupt at ``--interrupt-at``, then resume in a NEW trainer + model.

    ``corrupt_checkpoint(path)`` is the NEGATIVE CONTROL hook: it tampers with the
    saved state before the resume reads it, so a test can prove the comparison
    actually discriminates instead of always passing.
    """
    records = synthetic_records(args.records)
    cfg = build_config(accum=args.accum, work=work, real_model=args.model if args.real else None)
    real = args.real
    if real:
        model_a, tokenizer_a = _load_real(args.model, cfg)
    else:
        model_a, tokenizer_a = build_tiny_model(), CharTokenizer()
    interrupted = _TracedTrainer(
        cfg=cfg,
        model=model_a,
        tokenizer=tokenizer_a,
        records=records,
        work=work,
        run_name="interrupted",
        planned_total_steps=args.steps,
        stop_after_steps=args.interrupt_at,
    )
    interrupted.run()
    checkpoint = interrupted.trainer.checkpoint_manager.latest()
    if checkpoint is None:
        raise SystemExit(f"BLOCKED: no checkpoint written at step {args.interrupt_at}")
    if corrupt_checkpoint is not None:
        corrupt_checkpoint(checkpoint)

    # A COMPLETELY NEW trainer AND a new model object.
    cfg_b = build_config(accum=args.accum, work=work, real_model=args.model if real else None)
    if real:
        model_b, tokenizer_b = _load_real(args.model, cfg_b)
    else:
        model_b, tokenizer_b = build_tiny_model(), CharTokenizer()
    resumed = _TracedTrainer(
        cfg=cfg_b,
        model=model_b,
        tokenizer=tokenizer_b,
        records=records,
        work=work,
        run_name="resumed",
        planned_total_steps=args.steps,
    )
    resumed.trainer.resume(checkpoint)
    resumed.prepare_indices()
    # The resumed branch records only the work it did; the uninterrupted reference
    # consumed the interrupted branch's batches first, so the full trajectory is
    # the concatenation. Comparing anything shorter would hide a divergence in the
    # pre-interrupt epoch.
    resumed.sample_order = []
    resumed.run()
    state = resumed.final_state()
    state["sample_order"] = list(interrupted.sample_order) + list(state["sample_order"])
    state["losses"] = list(interrupted.losses) + list(state["losses"])
    state["positions"] = list(interrupted.positions) + list(state["positions"])
    state["lr_list"] = list(interrupted.lrs) + list(state["lr_list"])
    state["checkpoint"] = str(checkpoint)
    state["resume_step"] = int(args.interrupt_at)
    return state

def _run_child(args, work: Path) -> dict[str, Any]:
    """``--as-child``: run only the resumed branch in THIS process.

    Tensors (weights / AdamW moments) are far too large for stdout, so they go to
    ``--dump`` via ``torch.save``; the JSON printed on the last line carries every
    scalar and list the comparison needs.
    """
    state = run_resumed(args, work)
    if args.dump:
        import torch  # noqa: PLC0415

        torch.save({"weights": state["weights"], "optics": state["optics"]}, args.dump)
    return _jsonable(state)

def _jsonable(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in {"weights", "optics"}}

# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def render(report: dict[str, Any]) -> str:
    lines = ["", "== resume equivalence ==", ""]
    w = report["weights"]
    lines.append(
        f"weights      max_abs_diff={w['max_abs_diff']:.3e}  max_rel_diff={w['max_rel_diff']:.3e}  "
        f"tensors={w['n_tensors']}  worst={w['worst_tensor']}"
    )
    lines.append(
        f"optimizer    max_abs_diff={report['optimizer']['max_abs_diff']:.3e}  "
        f"entries={report['optimizer']['n_entries']}  worst={report['optimizer']['worst_entry']}"
    )
    sched = report["scheduler"]
    lines.append(
        "scheduler    last_epoch "
        f"{sched['reference']['last_epoch']} vs {sched['resumed']['last_epoch']}  "
        "_step_count "
        f"{sched['reference']['_step_count']} vs {sched['resumed']['_step_count']}  "
        f"lr_list_equal={report['checks']['lr_list_identical']}"
    )
    order = report["sample_order"]
    lines.append(
        f"sample order identical={order['identical']} lengths={order['lengths']} "
        f"first_divergence={order['first_divergence_index']}"
    )
    lines.append(f"  reference head: {order['reference_head']}")
    lines.append(f"  resumed   head: {order['resumed_head']}")
    lines.append(
        f"loss         max_abs_diff={report['loss']['max_abs_diff']:.3e} "
        f"identical={report['loss']['identical']} n={report['losses_count']}"
    )
    pos = report["position"]
    lines.append(
        f"position     global_step={pos['global_step']} epoch_index={pos['epoch_index']} "
        f"next_micro_index={pos['next_micro_index']}"
    )
    lines.append(f"rng          python={report['rng']['python']} torch={report['rng']['torch']}")
    lines.append(f"skipped      {report['skipped_steps']}")
    lines.append("")
    for name, ok in report["checks"].items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    lines.append("")
    lines.append("VERDICT: " + ("EQUIVALENT" if report["passed"] else "NOT EQUIVALENT"))
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiny", action="store_true", help="built-in tiny CPU model (default path)")
    parser.add_argument("--real", action="store_true", help="use --model (may be BLOCKED here)")
    parser.add_argument("--model", default="models/MiniCPM5-2B", help="base model for --real")
    parser.add_argument("--steps", type=int, default=12, help="planned_total_steps horizon")
    parser.add_argument("--interrupt-at", type=int, default=5, help="early-stop step for branch B")
    parser.add_argument("--accum", type=int, default=3, help="gradient_accumulation_steps")
    parser.add_argument("--records", type=int, default=12, help="number of synthetic records")
    parser.add_argument("--out", default=None, help="write the comparison JSON here")
    parser.add_argument(
        "--as-child",
        action="store_true",
        help="internal: run only the resume branch in this process (used via subprocess)",
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="internal: with --as-child, torch.save the final weights + optimizer moments here",
    )
    args = parser.parse_args(argv)
    if args.steps <= 0 or not (0 < args.interrupt_at < args.steps):
        print(
            f"error: need 0 < --interrupt-at ({args.interrupt_at}) < --steps ({args.steps}); "
            "the interrupt must be mid-run so the resume has work left",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        import torch  # noqa: F401,PLC0415 - probe
    except Exception as exc:
        print(f"BLOCKED: torch is not installed in this interpreter ({exc})", file=sys.stderr)
        return EXIT_MISMATCH

    with tempfile.TemporaryDirectory(prefix="resume_verify_") as tmp:
        work = Path(tmp)
        if args.as_child:
            print(json.dumps(_run_child(args, work)))
            return EXIT_OK

        reference = run_reference(args, work)
        dump_path = work / "resumed_tensors.pt"
        # B-resumed runs in a NEW PYTHON PROCESS via this same script.
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *argv_with_child(args, dump_path)],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            env={**os.environ, "PYTHONPATH": str(_SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        if child.returncode != 0:
            print(f"BLOCKED: the resumed branch failed in its own process:\n{child.stderr[-4000:]}")
            return EXIT_MISMATCH
        payload = None
        for line in reversed(child.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                payload = json.loads(line)
                break
        if payload is None:
            print(f"BLOCKED: no JSON from the child process:\n{child.stdout[-2000:]}")
            return EXIT_MISMATCH
        if not dump_path.is_file():
            print("BLOCKED: the child process wrote no tensor dump")
            return EXIT_MISMATCH
        import torch  # noqa: PLC0415

        tensors = torch.load(dump_path, map_location="cpu", weights_only=True)
        payload["weights"] = tensors["weights"]
        payload["optics"] = tensors["optics"]

    report = compare(reference, payload)
    report["mode"] = "real" if args.real else "tiny"
    report["steps"] = args.steps
    report["interrupt_at"] = args.interrupt_at
    report["accum"] = args.accum
    report["records"] = args.records
    report["sample_order_reference_len"] = len(reference["sample_order"])
    print(render(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return EXIT_OK if report["passed"] else EXIT_MISMATCH

def argv_with_child(args: argparse.Namespace, dump_path: Path) -> list[str]:
    """Arguments for the child process: same experiment, ``--as-child`` added."""
    argv = ["--steps", str(args.steps), "--interrupt-at", str(args.interrupt_at),
            "--accum", str(args.accum), "--records", str(args.records), "--as-child",
            "--dump", str(dump_path)]
    argv.append("--real" if args.real else "--tiny")
    if args.real:
        argv += ["--model", args.model]
    return argv

if __name__ == "__main__":
    raise SystemExit(main())
