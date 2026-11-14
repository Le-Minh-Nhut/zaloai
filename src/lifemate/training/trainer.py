"""Own training loop for action SFT (deliberately not ``transformers.Trainer``).

Why a hand-written loop: ~300 lines instead of a 5000-line dependency surface,
every branch unit-testable on CPU with a fake model, and checkpoint/interrupt
semantics that are ours to guarantee rather than to discover.

Training signal: cross-entropy on ``<action>...</action>`` tokens only. The
collator masks the prompt with ``-100``, so the loss never rewards reproducing
the prompt. Validation reports two numbers, both teacher-forced (no sampling, so
runs are reproducible):

* ``val_loss`` — mean token cross-entropy on target tokens;
* ``val_token_accuracy`` — share of target tokens whose argmax matches the gold
  token. This is *not* an action-match metric (that lives in
  ``lifemate.evaluation``, which re-parses the emitted block); it is a cheap
  in-training canary for "is the target distribution being learned at all".

Checkpoints: ``<output_dir>/<run_name>/step-<n>``, written atomically
(model+tokenizer via ``models.checkpoint``, then ``training_state.json`` last so
a checkpoint without state is visibly incomplete). Never overwrites an existing
step directory. ``save_total_limit`` prunes the oldest step dirs afterwards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lifemate.errors import LifeMateError

from .callbacks import NullTracker, StopController
from .collator import SupervisedCollator, build_supervised_example, to_torch_batch

__all__ = [
    "CheckpointManager",
    "SFTTrainer",
    "TrainResult",
    "TrainingInterrupted",
    "optimizer_steps_per_epoch",
    "read_training_state",
    "set_seed",
]

STATE_FILENAME = "training_state.json"

#: Fallback when a config carries no sequence budget. Matches
#: ``lifemate.config.TrainingConfig.max_seq_length`` so a hand-built TrainingConfig
#: and the shipped YAMLs cannot disagree (4096 middle-truncated 75 % of records).
DEFAULT_MAX_SEQ_LENGTH = 8192

# ``config_hash`` records the training knobs that change the math. These keys are
# excluded because they are bookkeeping, not optimisation: ``--resume`` sets
# ``resume_from_checkpoint`` *after* the checkpoint was written, resuming into a
# new ``--output-dir`` is normal practice, and logging/saving cadence plus the
# worker count and the (constant) ``method`` name cannot change a gradient.
CONFIG_HASH_IGNORED = (
    "resume_from_checkpoint",
    "output_dir",
    "logging_steps",
    "save_strategy",
    "eval_strategy",
    "save_total_limit",
    "report_to",
    "dataloader_num_workers",
    "method",
)

# Inside ``model``, device placement and the tokenizer location are not maths.
CONFIG_HASH_MODEL_IGNORED = ("device", "tokenizer_name")

# ``config_fingerprint`` hashes ONLY these sections (whitelist). Every other
# section — ``data.*``, ``asr.*``, ``agent.*``, ``inference.*``,
# ``evaluation.*``, ``wandb.*``, ``cloud.*`` — is excluded *by construction*, so a
# resume may move the dataset mount, switch GPUs or rename the W&B run.
# ``lora`` is hashed only when ``training.tuning_mode == "lora"``.
CONFIG_HASHED_SECTIONS = {
    "training": CONFIG_HASH_IGNORED,
    "model": CONFIG_HASH_MODEL_IGNORED,
}

class TrainingInterrupted(LifeMateError):
    """Raised after a stop was requested and an interrupt checkpoint was saved."""

    exit_code = 130

    def __init__(self, message: str, step: int, checkpoint_dir: str | None = None) -> None:
        super().__init__(message)
        self.step = step
        self.checkpoint_dir = checkpoint_dir

@dataclass
class TrainResult:
    """Outcome of one ``SFTTrainer.run()``."""

    steps: int = 0
    epochs: float = 0.0
    final_loss: float | None = None
    val_loss: float | None = None
    runtime_s: float = 0.0
    checkpoint_dir: str | None = None
    metrics_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "val_loss": self.val_loss,
            "runtime_s": self.runtime_s,
            "checkpoint_dir": self.checkpoint_dir,
            "metrics_history": list(self.metrics_history),
        }

def set_seed(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed ``random``, ``numpy`` (if present) and ``torch`` (if present)."""
    seed = int(seed)
    random.seed(seed)
    applied: dict[str, Any] = {"seed": seed, "python": True, "numpy": False, "torch": False}
    try:  # pragma: no cover - optional dependency
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
        applied["numpy"] = True
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        applied["torch"] = True
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            applied["deterministic"] = True
    except Exception:
        pass
    return applied

class CheckpointManager:
    """Versioned, atomic, never-overwriting step checkpoints.

    ``save_fn(target_dir, payload) -> Path`` is injectable so the directory
    policy can be tested without torch. ``payload`` carries ``model``,
    ``tokenizer``, ``state`` and ``meta``; the default writer delegates to
    :func:`lifemate.models.checkpoint.save_checkpoint`.
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str = "sft",
        save_total_limit: int = 2,
        save_fn: Callable[[Path, Mapping[str, Any]], Path] | None = None,
    ) -> None:
        self.output_dir = Path(str(output_dir))
        self.run_name = str(run_name or "sft")
        self.save_total_limit = int(save_total_limit)
        self.save_fn = save_fn or _default_save_fn
        self.root = self.output_dir / self.run_name

    def step_dir(self, step: int, tag: str = "step") -> Path:
        return self.root / f"{tag}-{int(step)}"

    def save(self, step: int, payload: Mapping[str, Any], tag: str = "step") -> Path:
        """Write one checkpoint; refuses to reuse a populated directory."""
        target = self.step_dir(step, tag)
        if target.exists() and any(target.iterdir()):
            raise LifeMateError(
                f"checkpoint directory already populated: {target}. "
                "Refusing to overwrite (resume with --resume or pick another --output-dir)."
            )
        target.mkdir(parents=True, exist_ok=True)
        written = Path(self.save_fn(target, payload))
        _atomic_write_json(target / STATE_FILENAME, dict(payload.get("state") or {}))
        if tag == "step":
            self._prune()
        return written

    def _prune(self) -> None:
        """Keep the newest ``save_total_limit`` step dirs (0 = keep everything)."""
        if self.save_total_limit <= 0:
            return
        dirs = sorted(
            (path for path in self.root.glob("step-*") if path.is_dir()),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        for stale in dirs[: max(0, len(dirs) - self.save_total_limit)]:
            _remove_tree(stale)

    def latest(self) -> Path | None:
        """Newest ``step-*`` directory, or ``None``."""
        dirs = [path for path in self.root.glob("step-*") if path.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda path: int(path.name.split("-")[-1]))

def _remove_tree(path: Path) -> None:
    import shutil  # noqa: PLC0415 - only on the prune path

    shutil.rmtree(path, ignore_errors=True)

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path

def read_training_state(path: str | Path) -> dict[str, Any]:
    """Read ``training_state.json`` from a checkpoint dir; ``{}`` when absent."""
    state_path = Path(str(path)) / STATE_FILENAME
    if not state_path.is_file():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LifeMateError(f"corrupt training state at {state_path}: {exc.msg}") from exc

def _default_save_fn(target: Path, payload: Mapping[str, Any]) -> Path:
    """Save model+tokenizer via ``models.checkpoint`` (lazy import)."""
    model = payload.get("model")
    tokenizer = payload.get("tokenizer")
    if model is None:
        raise LifeMateError("checkpoint payload has no model")
    try:  # pragma: no cover - depends on whether the models slice is complete
        from lifemate.models.checkpoint import save_checkpoint  # noqa: PLC0415

        return save_checkpoint(
            model,
            tokenizer,
            target,
            meta=dict(payload.get("meta") or {}),
            is_lora=bool(payload.get("is_lora")),
            step=payload.get("state", {}).get("step"),
            force=True,  # CheckpointManager already guarantees the dir is empty
        )
    except ImportError:  # pragma: no cover - fallback writer
        saver = getattr(model, "save_pretrained", None)
        if callable(saver):
            saver(str(target))
        tok_saver = getattr(tokenizer, "save_pretrained", None)
        if callable(tok_saver):
            tok_saver(str(target))
        return target

def optimizer_steps_per_epoch(n_examples: int, micro_batch: int, accum: int) -> int:
    """Optimizer steps one epoch produces, matching the loop's trailing flush.

    ``ceil(n_examples / (micro_batch * accum))``: every ``accum`` accumulated
    micro-groups is one step, and the leftover group at the epoch end is flushed
    as a final partial step (rescaled, not dropped). Guarantees ≥ 1 so a dataset
    smaller than one effective batch still trains.
    """
    effective = max(1, int(micro_batch)) * max(1, int(accum))
    return max(1, math.ceil(max(0, int(n_examples)) / effective))

def chunked_causal_lm_loss(logits: Any, labels: Any, chunk_tokens: int) -> Any:
    """Causal-LM cross-entropy in sequence chunks, mathematically exact.

    ``transformers``' ``ForCausalLMLoss`` upcasts the WHOLE ``[seq, vocab]``
    logits tensor to fp32 before the fused ``cross_entropy``. At vocab 130560 and
    seq 8192 that is one 3.98 GiB allocation (measured OOM on a 16.57 GB card).
    Chunking over the sequence dimension caps that temporary at
    ``chunk_tokens * vocab * 4`` bytes.

    Equivalence is preserved by accumulating a **sum** of per-chunk token losses
    and dividing once by the number of supervised tokens: dividing per chunk and
    averaging would over-weight a short trailing chunk.
    """
    import torch  # noqa: PLC0415 - lazy on purpose
    import torch.nn.functional as F  # noqa: PLC0415

    shift_labels = labels[..., 1:]
    seq = int(shift_labels.shape[-1])
    vocab = int(logits.shape[-1])
    total = None
    supervised = 0
    step = max(1, int(chunk_tokens))
    for start in range(0, seq, step):
        stop = min(start + step, seq)
        target = shift_labels[..., start:stop]
        piece = logits[..., start:stop, :].float()  # only this slice is upcast
        piece = piece.reshape(-1, vocab)
        loss_sum = F.cross_entropy(
            piece, target.reshape(-1), ignore_index=-100, reduction="sum"
        )
        total = loss_sum if total is None else total + loss_sum
        supervised += int((target != -100).sum())
    if total is None:  # pragma: no cover - seq == 0 is impossible for a real batch
        return logits.sum() * 0.0
    # A zero-grad-linked zero when nothing is supervised, instead of nan.
    return total / max(supervised, 1)


def _forward_loss(model: Any, batch: Mapping[str, Any], loss_chunk_tokens: int = 0) -> Any:
    """Run the model on a tensor batch and return its scalar loss tensor.

    ``loss_chunk_tokens <= 0`` keeps the original single-shot path (the model
    computes the loss itself). A positive value asks for logits and computes the
    chunked loss here; a model that returns no logits falls back to the original
    path rather than failing.
    """
    if loss_chunk_tokens > 0 and "labels" in batch:
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        try:
            output = model(**inputs)
        except TypeError:  # pragma: no cover - model requires labels
            output = None
        logits = getattr(output, "logits", None)
        if logits is not None:
            return chunked_causal_lm_loss(logits, batch["labels"], loss_chunk_tokens)

    output = model(**batch)
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, (tuple, list)) and output:
        loss = output[0]
    if loss is None:
        raise LifeMateError(
            "model returned no loss; pass labels= so the causal-LM head computes cross-entropy"
        )
    return loss

#: Precision spellings that mean fp16 / bf16, lowercased (the loader's
#: ``DTYPE_MAP`` accepts exactly these). fp16 needs loss scaling; bf16 does not.
FP16_PRECISIONS = frozenset({"float16", "fp16", "half"})
BF16_PRECISIONS = frozenset({"bfloat16", "bf16"})


class SFTTrainer:
    """Minimal SFT loop: epochs → micro-batches → clip → optimizer/scheduler step.

    Every collaborator is injected, so the loop is exercised on CPU with a fake
    model and without transformers/peft.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        collator: SupervisedCollator,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] = (),
        cfg: Any = None,
        tracker: Any = None,
        device: Any = "cpu",
        run_name: str = "sft",
        max_steps: int | None = None,
        stop: StopController | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        is_lora: bool = False,
        dataset_manifest_hash: str | None = None,
        optimizer: Any = None,
        scheduler: Any = None,
        log_every: int | None = None,
    ) -> None:
        training = getattr(cfg, "training", cfg)
        if training is None:
            raise LifeMateError("SFTTrainer requires a config with a training section")
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.cfg = cfg
        self.training = training
        self.train_records = list(train_records)
        self.val_records = list(val_records)
        self.tracker = tracker or NullTracker()
        self.device = device
        self.is_lora = bool(is_lora)
        self.dataset_manifest_hash = dataset_manifest_hash
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stop = stop or StopController()
        self.max_seq_length = int(getattr(training, "max_seq_length", DEFAULT_MAX_SEQ_LENGTH))
        self.epochs = float(getattr(training, "num_train_epochs", 2) or 2)
        self.micro_batch = max(1, int(getattr(training, "per_device_train_batch_size", 1) or 1))
        self.accum = max(1, int(getattr(training, "gradient_accumulation_steps", 1) or 1))
        self.grad_clip = float(getattr(training, "gradient_clip_norm", 1.0) or 0.0)
        self.log_every = int(log_every if log_every is not None else getattr(training, "logging_steps", 10) or 10)
        self.seed = int(getattr(training, "seed", 42) or 42)
        self.max_steps = max_steps
        self.run_name = run_name
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            getattr(training, "output_dir", "outputs/checkpoints"),
            run_name=run_name,
            save_total_limit=int(getattr(training, "save_total_limit", 2) or 0),
        )
        self.save_strategy = str(getattr(training, "save_strategy", "epoch") or "epoch")
        self.eval_strategy = str(getattr(training, "eval_strategy", "epoch") or "epoch")
        # Normalised the same way the loader's DTYPE_MAP is (strip + lower), so
        # "FP16"/"Half" reach the same autocast + scaler branches as "float16".
        self.precision = str(getattr(training, "precision", "bfloat16") or "bfloat16").strip().lower()
        # fp16 on CUDA needs loss scaling: fp16's exponent range overflows on the
        # first large grad and the adapter dies silently (fp16 has no bf16's
        # range). bf16/fp32 and CPU runs get ``None`` and keep the plain
        # backward -> clip -> step order untouched.
        self.scaler = self._make_grad_scaler()
        # 0 => the model's own (whole-vocab, fp32-upcast) loss; >0 => chunked CE.
        self.loss_chunk_tokens = int(getattr(training, "loss_chunk_tokens", 0) or 0)
        # Resolved from ``ModelConfig.name`` (the single identity field) and
        # threaded into every supervised example, so the terminator and template
        # kwargs do not depend on the tokenizer's (checkpoint-relative) metadata.
        from lifemate.models.profiles import resolve_profile  # noqa: PLC0415 - lazy

        self.profile = resolve_profile(
            getattr(getattr(cfg, "model", None), "name", "") if cfg is not None else ""
        )
        self.start_step = 0
        self._last_saved_step: int | None = None
        self._last_evaluated_step: int | None = None
        self.metrics_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ setup
    def _prepare_examples(self, records: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        dropped: dict[str, int] = {}
        for item in records:
            example = build_supervised_example(
                self.tokenizer,
                list(item.get("messages") or []),
                str(item.get("target") or ""),
                self.max_seq_length,
                profile=self.profile,
            )
            if example.get("kept"):
                examples.append(example)
            else:
                reason = str(example.get("reason") or "unknown")
                dropped[reason] = dropped.get(reason, 0) + 1
        # Always recorded, including for an empty/degenerate split: the counters are
        # the diagnosis when the LifeMateError below fires.
        self.metrics_history.append(
            {"split": split, "kept": len(examples), "dropped": dict(sorted(dropped.items()))}
        )
        if not examples:
            raise LifeMateError(
                f"no usable {split} examples after tokenisation (dropped: {dict(sorted(dropped.items()))}); "
                "check max_seq_length and the target format"
            )
        return examples

    def _ensure_optimizer(self, num_training_steps: int) -> None:
        if self.optimizer is not None and self.scheduler is not None:
            return
        from .full_finetuning import build_optimizer_and_scheduler  # noqa: PLC0415 - lazy

        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, self.cfg, num_training_steps
        )
        self.optimizer = self.optimizer or optimizer
        self.scheduler = self.scheduler or scheduler

    def _make_grad_scaler(self):
        """A ``torch.amp.GradScaler`` for fp16-on-CUDA runs, else ``None``.

        Only fp16 needs it. The scaler's state (the current scale and its
        growth/backoff counters) IS persisted next to the optimizer tensors and
        restored by :meth:`_load_optimizer_state`, so a resumed fp16 run does
        not restart the scale search from 65536.
        """
        if self.precision not in FP16_PRECISIONS:
            return None
        if not (isinstance(self.device, str) and self.device.startswith("cuda")):
            return None
        try:
            import torch  # noqa: PLC0415

            return torch.amp.GradScaler("cuda")
        except Exception:  # pragma: no cover - old torch / no CUDA build
            return None

    def _autocast(self):
        """Autocast context manager (null context on CPU / fp32)."""
        try:
            import torch  # noqa: PLC0415

            if isinstance(self.device, str) and self.device.startswith("cuda"):
                if self.precision in BF16_PRECISIONS:
                    return torch.autocast("cuda", dtype=torch.bfloat16)
                if self.precision in FP16_PRECISIONS:
                    return torch.autocast("cuda", dtype=torch.float16)
            return _null_context()
        except Exception:  # pragma: no cover - torch absent
            return _null_context()

    # ------------------------------------------------------------------- loop
    def run(self) -> TrainResult:
        """Train, evaluate per ``eval_strategy``, checkpoint per ``save_strategy``."""
        started = time.time()
        # A resumed run already restored torch/random state from the checkpoint;
        # re-seeding here would silently restart the stream from scratch, so the
        # RNG is only seeded for a fresh run (start_step == 0).
        if self.start_step <= 0:
            set_seed(self.seed)
        train_examples = self._prepare_examples(self.train_records, "train")
        val_examples = self._prepare_examples(self.val_records, "val") if self.val_records else []

        steps_per_epoch = optimizer_steps_per_epoch(len(train_examples), self.micro_batch, self.accum)
        total_steps = self.max_steps if self.max_steps is not None else int(math.ceil(steps_per_epoch * self.epochs))
        total_steps = max(1, total_steps)
        self._ensure_optimizer(total_steps)
        if self.start_step > 0:
            # A resumed run restored the checkpoint's scheduler, which was built for
            # the *interrupted* run's step count. Left alone, every remaining step's
            # LR comes from a schedule that already reached its horizon: the run
            # silently trains at LR 0.0 (no learning, full compute cost). Rebuild the
            # schedule on the restored optimizer for this run's real horizon, keeping
            # the AdamW moments untouched.
            from .full_finetuning import build_scheduler  # noqa: PLC0415 - lazy

            self.scheduler = build_scheduler(
                self.optimizer, self.cfg, total_steps, last_epoch=self.start_step
            )

        self.model.train()
        step = self.start_step
        epoch_index = 0
        last_loss: float | None = None
        val_loss: float | None = None
        epochs_done = 0.0
        order_source = list(range(len(train_examples)))
        rng = random.Random(self.seed + self.start_step)
        checkpoint_dir = str(self.checkpoint_manager.root)

        pending = 0  # micro-batches accumulated since the last optimizer step
        def optimizer_step() -> None:
            """One optimizer step; shared by the accumulate path and the flush."""
            nonlocal step, pending, val_loss, checkpoint_dir
            if self.scaler is not None:
                # fp16: gradients are still multiplied by the loss scale here, and
                # both the partial-group rescale and clip_grad_norm_ must see TRUE
                # magnitudes, so unscale first and never scale twice.
                self.scaler.unscale_(self.optimizer)
            if 0 < pending < self.accum:
                # Partial trailing group: gradients were scaled by 1/accum but only
                # ``pending`` micro-batches contributed, so rescale by accum/pending
                # to keep the effective loss scale identical to a full group.
                factor = self.accum / pending
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(factor)
            self._clip_gradients()
            if self.scaler is not None:
                # Inf/NaN guard: ``step`` skips the update when a grad is not
                # finite; ``update`` then backs the scale off.
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            pending = 0
            step += 1
            self._maybe_log(step, last_loss)
            # Step-based evaluation & saving
            if val_examples and self.eval_strategy == "steps" and self._due("steps", step, total_steps):
                metrics = self.evaluate(val_examples)
                val_loss = metrics["val_loss"]
                self.metrics_history.append({"step": step, "epoch": epoch_index, **metrics})
                self.tracker.log({"step": step, **metrics})
                self._last_evaluated_step = step
            if self.save_strategy == "steps" and self._due("steps", step, total_steps):
                checkpoint_dir = str(self._save(step, epoch_index))

        try:
            while step < total_steps:
                rng.shuffle(order_source)
                epoch_index += 1
                self.optimizer.zero_grad(set_to_none=True)

                for micro in range(0, len(order_source), self.micro_batch):
                    if step >= total_steps:
                        break
                    chunk = order_source[micro : micro + self.micro_batch]
                    batch = self.collator([train_examples[i] for i in chunk])
                    tensors = to_torch_batch(batch, self.device)
                    with self._autocast():
                        loss = _forward_loss(self.model, tensors, self.loss_chunk_tokens)
                    pending += 1
                    # Scale by the accumulation count; a short trailing group is
                    # rescaled by accum/pending before its optimizer step. Under
                    # fp16 the scaler multiplies that by its dynamic loss scale.
                    scaled_loss = loss / self.accum
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    last_loss = float(loss.detach().item())

                    if pending >= self.accum:
                        optimizer_step()
                        if self.stop.requested:
                            checkpoint_dir = str(self._save(step, epoch_index, tag="interrupt"))
                            raise TrainingInterrupted(
                                f"stop requested; interrupt checkpoint written at step {step}",
                                step=step,
                                checkpoint_dir=checkpoint_dir,
                            )
                        if step >= total_steps:
                            break

                # Flush a partial accumulation group so no grad is silently dropped
                # and the loop always makes progress (a dataset smaller than one
                # effective batch would otherwise never step).
                if pending and step < total_steps:
                    optimizer_step()
                    if self.stop.requested:
                        checkpoint_dir = str(self._save(step, epoch_index, tag="interrupt"))
                        raise TrainingInterrupted(
                            f"stop requested; interrupt checkpoint written at step {step}",
                            step=step,
                            checkpoint_dir=checkpoint_dir,
                        )
                epochs_done = epoch_index

                # Epoch-based evaluation & saving (or final step if not yet evaluated/saved)
                if val_examples and (self.eval_strategy == "epoch" or (step >= total_steps and self.eval_strategy != "no")):
                    if self._last_evaluated_step != step:
                        metrics = self.evaluate(val_examples)
                        val_loss = metrics["val_loss"]
                        self.metrics_history.append({"step": step, "epoch": epoch_index, **metrics})
                        self.tracker.log({"step": step, **metrics})
                        self._last_evaluated_step = step
                if (self.save_strategy == "epoch" or (step >= total_steps and self.save_strategy != "no")) and self._due(self.save_strategy, step, total_steps):
                    if self._last_saved_step != step:
                        checkpoint_dir = str(self._save(step, epoch_index))
        except TrainingInterrupted:
            raise
        except BaseException:
            self._save_crash_safe(step, epoch_index)
            raise
        finally:
            self.tracker.finish()

        result = TrainResult(
            steps=step,
            epochs=float(epochs_done),
            final_loss=last_loss,
            val_loss=val_loss,
            runtime_s=round(time.time() - started, 3),
            checkpoint_dir=checkpoint_dir,
            metrics_history=list(self.metrics_history),
        )
        return result

    def _due(self, strategy: str, step: int, total_steps: int) -> bool:
        """``epoch`` (here: end of each epoch) / ``steps`` / ``no``."""
        key = str(strategy or "epoch").lower()
        if key in {"no", "none"}:
            return False
        if key == "steps":
            return step > 0 and step % max(1, self.log_every) == 0
        return True

    def _clip_gradients(self) -> None:
        if self.grad_clip <= 0:
            return
        try:
            import torch  # noqa: PLC0415

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        except Exception:  # pragma: no cover - torch absent
            return

    def _maybe_log(self, step: int, loss: float | None) -> None:
        if loss is None or self.log_every <= 0 or step % self.log_every != 0:
            return
        lr = None
        try:
            lr = float(self.optimizer.param_groups[0]["lr"])
        except Exception:  # pragma: no cover - fake optimizer
            lr = None
        payload = {"step": step, "train_loss": round(loss, 6)}
        if lr is not None:
            payload["learning_rate"] = lr
        self.metrics_history.append(dict(payload))
        self.tracker.log(payload)

    def evaluate(self, examples: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Teacher-forced validation loss + per-target-token accuracy.

        Two paths, and the shipped default is the first:

        * ``loss_chunk_tokens > 0`` (default 1024): the forward runs WITHOUT
          ``labels``, so the model's own ``ForCausalLMLoss`` — which upcasts the
          whole ``[seq, vocab]`` logits tensor to fp32 — is never invoked, and
          both the loss sum and the argmax count run on one sequence chunk at a
          time. At seq 8192 / vocab 130560 that removes a 4.28 GB fp32 upcast
          plus a 2.14 GB ``.contiguous()`` shift copy from the eval step.
        * ``loss_chunk_tokens <= 0``: the original single-shot path, kept
          verbatim so the numbers can be compared against the chunked one.

        Shift semantics are identical in both paths: ``logits[..., :-1, :]``
        against ``labels[..., 1:]``. A model that returns no logits falls back to
        its own loss, exactly as before.
        """
        import torch  # noqa: PLC0415 - lazy
        import torch.nn.functional as F  # noqa: PLC0415

        data = list(examples) if examples is not None else self._prepare_examples(self.val_records, "val")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        with torch.no_grad():
            for start in range(0, len(data), self.micro_batch):
                batch = self.collator(data[start : start + self.micro_batch])
                tensors = to_torch_batch(batch, self.device)
                labels = tensors.get("labels")
                handled = False
                if self.loss_chunk_tokens > 0 and labels is not None:
                    inputs = {k: v for k, v in tensors.items() if k != "labels"}
                    try:
                        with self._autocast():
                            output = self.model(**inputs)
                    except TypeError:  # pragma: no cover - model requires labels
                        output = None
                    logits = getattr(output, "logits", None)
                    if logits is not None:
                        shift_labels = labels[..., 1:]
                        seq = int(shift_labels.shape[-1])
                        vocab = int(logits.shape[-1])
                        width = max(1, self.loss_chunk_tokens)
                        for lo in range(0, seq, width):
                            hi = min(lo + width, seq)
                            target = shift_labels[..., lo:hi]
                            slice_logits = logits[..., lo:hi, :]
                            total_loss += float(
                                F.cross_entropy(
                                    slice_logits.float().reshape(-1, vocab),
                                    target.reshape(-1),
                                    ignore_index=-100,
                                    reduction="sum",
                                ).item()
                            )
                            # argmax on the slice only, never on [seq, vocab].
                            preds = slice_logits.argmax(dim=-1)
                            match = target != -100
                            correct += int((preds[match] == target[match]).sum().item())
                            total_tokens += int(match.sum().item())
                        handled = True

                if not handled:
                    with self._autocast():
                        output = self.model(**tensors)
                    loss = getattr(output, "loss", None)
                    if loss is None and isinstance(output, (tuple, list)) and output:
                        loss = output[0]
                    logits = getattr(output, "logits", None)
                    if logits is not None and labels is not None:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        mask = shift_labels != -100
                        supervised = int(mask.sum().item())
                        if loss is not None and supervised:
                            total_loss += float(loss.detach().item()) * supervised
                            total_tokens += supervised
                        if supervised:
                            preds = shift_logits.argmax(dim=-1)
                            correct += int((preds[mask] == shift_labels[mask]).sum().item())
                    elif loss is not None and labels is not None:
                        supervised = int((labels != -100).sum().item())
                        if supervised:
                            total_loss += float(loss.detach().item()) * supervised
                            total_tokens += supervised
        self.model.train()
        return {
            "val_loss": round(total_loss / total_tokens, 6) if total_tokens else None,
            "val_token_accuracy": round(correct / total_tokens, 6) if total_tokens else None,
            "val_target_tokens": total_tokens,
            "n_supervised_tokens": total_tokens,
        }

    # ------------------------------------------------------------- checkpoint
    def _state(self, step: int, epoch: int) -> dict[str, Any]:
        config_hash = config_fingerprint(self.cfg)
        state = {
            "step": step,
            "epoch": epoch,
            "seed": self.seed,
            "config_hash": config_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "is_lora": self.is_lora,
            "precision": self.precision,
        }
        try:
            import torch  # noqa: PLC0415

            state["torch_rng_state"] = torch.get_rng_state().tolist()
            if torch.cuda.is_available():  # pragma: no cover - GPU only
                state["cuda_rng_state"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception:  # pragma: no cover - torch absent
            state["torch_rng_state"] = None
        state["python_rng_state"] = list(random.getstate()[1])
        return state

    def _save(self, step: int, epoch: int, tag: str = "step") -> Path:
        if tag == "step" and self._last_saved_step == step:
            return self.checkpoint_manager.step_dir(step, tag)
        payload = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "is_lora": self.is_lora,
            "state": self._state(step, epoch),
            "meta": {
                "model_name": getattr(getattr(self.cfg, "model", None), "name", None),
                # Profile family makes an adapter/base mismatch detectable before
                # any weight is merged (see models.checkpoint compat check).
                "model_profile": self.profile.name if self.profile else None,
                "seed": self.seed,
                "config": _cfg_dict(self.cfg),
                "dataset_manifest_hash": self.dataset_manifest_hash,
                "tuning_mode": str(
                    getattr(self.training, "tuning_mode", "lora" if self.is_lora else "full")
                ),
            },
        }
        written = self.checkpoint_manager.save(step, payload, tag=tag)
        self._save_optimizer_state(written, step, epoch)
        if tag == "step":
            self._last_saved_step = step
        return written

    def _save_optimizer_state(self, checkpoint_dir: Path, step: int, epoch: int) -> None:
        """Optimizer/scheduler tensors via ``torch.save`` (skipped without torch)."""
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return
        if self.optimizer is None:
            return
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            # fp16 grad-scaler state (None on bf16/fp32/CPU runs).
            "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": step,
            "epoch": epoch,
        }
        tmp = checkpoint_dir / "optimizer_state.pt.tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, checkpoint_dir / "optimizer_state.pt")
        except Exception:  # pragma: no cover - fake optimizers
            tmp.unlink(missing_ok=True)

    def _save_crash_safe(self, step: int, epoch: int) -> None:
        """Best-effort ``crash-<step>`` state; never masks the original error."""
        try:
            target = self.checkpoint_manager.step_dir(step, "crash")
            target.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(target / STATE_FILENAME, self._state(step, epoch))
        except Exception:  # pragma: no cover - disk full etc.
            return

    def resume(self, checkpoint_dir: str | Path) -> int:
        """Load weights + optimizer state from ``checkpoint_dir``; return its step.

        Fails loudly when the checkpoint is unreadable, when its step is not
        monotonic with respect to ``start_step``, or when a LoRA checkpoint is
        resumed into a full fine-tuning run (or vice versa).
        """
        path = Path(str(checkpoint_dir))
        state = read_training_state(path)
        if not state:
            raise LifeMateError(
                f"cannot resume: {path} has no {STATE_FILENAME}. "
                "Point --resume at a step-<n> directory produced by this trainer."
            )
        step = int(state.get("step") or 0)
        if step <= 0:
            raise LifeMateError(f"cannot resume: {path} records step {step}; expected a positive step")
        stored_lora = state.get("is_lora")
        if isinstance(stored_lora, bool) and stored_lora != self.is_lora:
            raise LifeMateError(
                f"cannot resume: checkpoint is {'LoRA' if stored_lora else 'full'} "
                f"but this run is {'LoRA' if self.is_lora else 'full'}"
            )
        stored_hash = state.get("dataset_manifest_hash")
        if self.dataset_manifest_hash and stored_hash and stored_hash != self.dataset_manifest_hash:
            raise LifeMateError(
                "cannot resume: dataset manifest hash differs from the checkpoint "
                f"({stored_hash} != {self.dataset_manifest_hash}); the split changed"
            )
        stored_config_hash = state.get("config_hash")
        if stored_config_hash:
            current = config_fingerprint(self.cfg)
            if stored_config_hash != current:
                raise LifeMateError(
                    "cannot resume: training config differs from the checkpoint "
                    f"({stored_config_hash} != {current}); the run would not continue the "
                    "same optimisation (the guard hashes training.* minus bookkeeping, "
                    "model.*, and lora.* only in LoRA mode; check lr, seed, precision, "
                    "max_seq_length, batch settings, epochs, scheduler, tuning_mode)"
                )

        self._load_weights(path)
        # The optimizer does not exist yet on a fresh trainer, so build it before
        # restoring; otherwise ``_load_optimizer_state`` returns early and the AdamW
        # moments are silently dropped (resume would restart from zero moments).
        # Only do it when the checkpoint actually carries optimizer state, so models
        # without parameters (test doubles) keep working.
        if self.optimizer is None and (path / "optimizer_state.pt").is_file():
            self._ensure_optimizer(self.max_steps or max(1, step))
        self._load_optimizer_state(path)
        torch_rng = state.get("torch_rng_state")
        if torch_rng is not None:
            try:
                import torch  # noqa: PLC0415
                torch.set_rng_state(torch.tensor(torch_rng, dtype=torch.uint8))
            except Exception:  # pragma: no cover
                pass
        py_rng = state.get("python_rng_state")
        if py_rng is not None:
            try:
                random.setstate((3, tuple(py_rng), None))
            except Exception:  # pragma: no cover
                pass
        self.start_step = step
        return step

    def _load_weights(self, path: Path) -> None:
        """Restore weights: peft adapter, a test hook, or a plain state dict."""
        if self.is_lora:
            self._load_adapter_weights(path)
            return
        hook = getattr(self.model, "load_state_dict_from_checkpoint", None)
        if callable(hook):  # documented hook for fake/test models
            hook(path)
            return
        state_files = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
        if not state_files:
            raise LifeMateError(
                f"cannot resume: no weight files under {path} "
                "(expected *.safetensors or pytorch_model*.bin)"
            )
        state = _read_state_dict(state_files)
        try:
            missing, unexpected = _load_state_dict_into(self.model, state)
        except Exception as exc:
            raise LifeMateError(f"failed to load weights from {path}: {exc}") from exc
        # A shape/name mismatch means this checkpoint is not this model: fail loudly
        # instead of silently training from fresh weights.
        if missing:
            raise LifeMateError(
                f"cannot resume: {len(missing)} parameter(s) missing from {path} "
                f"(e.g. {sorted(missing)[0]}); the checkpoint does not match this model"
            )
        if unexpected:
            raise LifeMateError(
                f"cannot resume: {len(unexpected)} unexpected parameter(s) in {path} "
                f"(e.g. {sorted(unexpected)[0]}); the checkpoint does not match this model"
            )

    def _load_adapter_weights(self, path: Path) -> None:
        """Restore LoRA adapter weights through peft's own API.

        ``PeftModel.load_adapter`` requires an explicit ``adapter_name`` (it is
        positional in peft >= 0.21), and reloading into the *active* adapter name
        is what restores the tensors in place; loading under a new name would add
        a second adapter and leave the active one at its fresh random init.
        """
        load = getattr(self.model, "load_adapter", None)
        if not callable(load):
            raise LifeMateError(
                f"cannot resume: is_lora=True but {type(self.model).__name__} exposes "
                "no load_adapter(); pass a peft model"
            )
        name = getattr(self.model, "active_adapter", None) or "default"
        try:
            load(str(path), adapter_name=name, is_trainable=True)
        except Exception as exc:
            raise LifeMateError(f"failed to load LoRA adapter from {path}: {exc}") from exc
        active = getattr(self.model, "active_adapter", name)
        if isinstance(active, (list, tuple, set)):
            active = next(iter(active), name)
        if active != name:
            raise LifeMateError(
                f"cannot resume: adapter loaded as {active!r} but the model trains {name!r}; "
                "the restored weights would be inactive"
            )

    def _load_optimizer_state(self, path: Path) -> None:
        state_file = path / "optimizer_state.pt"
        if not state_file.is_file() or self.optimizer is None:
            return
        try:
            import torch  # noqa: PLC0415
        except Exception:  # pragma: no cover - torch absent
            return
        payload = torch.load(state_file, map_location="cpu", weights_only=False)
        try:
            self.optimizer.load_state_dict(payload["optimizer"])
            if self.scheduler is not None and payload.get("scheduler") is not None:
                self.scheduler.load_state_dict(payload["scheduler"])
            if self.scaler is not None and payload.get("grad_scaler"):
                self.scaler.load_state_dict(payload["grad_scaler"])
        except Exception as exc:
            raise LifeMateError(f"failed to restore optimizer state from {state_file}: {exc}") from exc

class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> None:
        return None

def _read_state_dict(state_files: Sequence[Path]) -> dict[str, Any]:
    """Load and merge ``.safetensors``/``.bin`` shards into one state dict."""
    merged: dict[str, Any] = {}
    for state_file in state_files:
        if state_file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file  # noqa: PLC0415 - optional
            except Exception as exc:  # pragma: no cover - safetensors absent
                raise LifeMateError(
                    f"cannot read {state_file.name}: safetensors is not installed ({exc})"
                ) from exc
            merged.update(load_file(str(state_file)))
        else:
            import torch  # noqa: PLC0415 - lazy

            payload = torch.load(state_file, map_location="cpu", weights_only=True)
            if isinstance(payload, Mapping):
                payload = payload.get("state_dict", payload)
            merged.update(dict(payload))
    return merged


def _load_state_dict_into(model: Any, state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Load ``state`` into ``model``, honouring its checkpoint key conversions.

    ``save_pretrained`` writes keys through the model's *reverse* weight
    conversions, so a hybrid checkpoint (e.g. ``Qwen3_5``, whose released
    multimodal weights use a ``model.language_model.*`` prefix) cannot be
    reloaded with a raw ``load_state_dict``: every non-``lm_head`` key is
    prefixed and would be reported as 55 missing + 55 unexpected. transformers'
    own loader applies the conversions, so use it when it is available and fall
    back to plain ``load_state_dict`` otherwise.

    Returns ``(missing, unexpected)`` with tied weights (``lm_head`` sharing
    ``embed_tokens`` — ``tie_word_embeddings`` checkpoints omit one of the two)
    treated as satisfied, because transformers re-ties them on load.
    """
    conversions = getattr(model, "_weight_conversions", None)
    load_state_dict = getattr(model, "load_state_dict", None)
    if conversions:
        try:
            from transformers.modeling_utils import (  # noqa: PLC0415 - optional
                LoadStateDictConfig,
                convert_and_load_state_dict_in_model,
            )
        except Exception:  # pragma: no cover - older transformers
            convert_and_load_state_dict_in_model = None
        if convert_and_load_state_dict_in_model is not None:
            info, _ = convert_and_load_state_dict_in_model(
                model,
                dict(state),
                LoadStateDictConfig(weight_mapping=conversions),
                disk_offload_index=None,
            )
            return _without_tied(model, info.missing_keys, info.unexpected_keys)
    missing, unexpected = load_state_dict(dict(state), strict=False)
    return _without_tied(model, missing, unexpected)


def _without_tied(model: Any, missing: Iterable[str], unexpected: Iterable[str]) -> tuple[list[str], list[str]]:
    """Drop tied-weight names from a load report.

    A ``tie_word_embeddings`` checkpoint stores only one of ``embed_tokens`` /
    ``lm_head``, so the other is reported missing even though transformers has
    already re-tied it to the loaded tensor. Those are not mismatches.
    """
    tied: set[str] = set()
    for attr, target in (getattr(model, "_tied_weights_keys", None) or {}).items():
        tied.add(attr)
        if isinstance(target, str):
            tied.add(target)
    return (
        sorted(key for key in missing if key not in tied),
        sorted(key for key in unexpected if key not in tied),
    )


def _cfg_dict(cfg: Any) -> dict[str, Any]:
    """Effective config as a plain dict (never raises on exotic configs)."""
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:  # pragma: no cover
            pass
    return {"repr": repr(cfg)}


def _section(cfg: Any, name: str) -> dict[str, Any]:
    """Field→value view of one config section (``{}`` when absent/opaque)."""
    section = cfg.get(name) if isinstance(cfg, Mapping) else getattr(cfg, name, None)
    if section is None and name == "training" and dataclasses.is_dataclass(cfg):
        # ``SFTTrainer`` also accepts a bare ``TrainingConfig`` as ``cfg``.
        section = cfg
    if section is None:
        return {}
    if dataclasses.is_dataclass(section) and not isinstance(section, type):
        # Fields come from the dataclass, so this projection cannot drift.
        return {f.name: getattr(section, f.name, None) for f in dataclasses.fields(section)}
    if isinstance(section, Mapping):
        return dict(section)
    return dict(vars(section)) if hasattr(section, "__dict__") else {}

def config_fingerprint(cfg: Any) -> str:
    """16-hex digest of only the knobs that change the optimisation.

    Whitelist, not blacklist: the hash covers exactly

    * ``training.*`` minus :data:`CONFIG_HASH_IGNORED` — the bookkeeping keys
      ``resume_from_checkpoint``, ``output_dir``, ``logging_steps``,
      ``save_strategy``, ``eval_strategy``, ``save_total_limit``, ``report_to``,
      ``dataloader_num_workers`` and ``method`` are dropped; every optimisation
      knob (lr, weight decay, epochs, batch/accum, warmup, scheduler, clip,
      checkpointing, seed, ``max_seq_length``, ``validation_ratio``, optimizer,
      precision, ``tuning_mode``) stays in,
    * ``model.*`` minus :data:`CONFIG_HASH_MODEL_IGNORED` (``device``, which is
      placement not maths, and ``tokenizer_name``),
    * every ``lora.*`` field — but only when ``training.tuning_mode == "lora"``;
      in full fine-tuning a LoRA knob cannot change the maths, so it is skipped.

    Everything else — ``data.root``, ``asr.*``, ``agent.*``, ``inference.*``,
    ``evaluation.*``, ``wandb.*``, ``cloud.*`` — is excluded by construction, so a
    resume may move the dataset mount, switch GPUs or rename the W&B run. Fields
    are read from the dataclasses themselves (or the mapping's keys), so the set
    cannot drift.
    """
    sections = dict(CONFIG_HASHED_SECTIONS)
    if str(_section(cfg, "training").get("tuning_mode") or "full") == "lora":
        sections["lora"] = ()
    payload = {
        name: {k: v for k, v in _section(cfg, name).items() if k not in ignored}
        for name, ignored in sections.items()
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
