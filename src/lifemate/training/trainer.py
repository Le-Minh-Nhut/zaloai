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
import copy
import hashlib
import json
import math
import os
import random
import struct
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
    "TRAINING_STATE_SCHEMA",
    "TrainResult",
    "TrainingInterrupted",
    "TrainingProgress",
    "chunked_causal_lm_loss",
    "dataset_order_hash",
    "forward_loss",
    "optimizer_steps_per_epoch",
    "read_training_state",
    "set_seed",
]

STATE_FILENAME = "training_state.json"

#: Version of the ``progress`` block in ``training_state.json``. Schema 2 is the
#: first version that records THE NEXT WORK ITEM (epoch permutation + cursor +
#: every RNG stream), i.e. the first one an EXACT resume can be reconstructed
#: from. A schema-1 checkpoint only knows the step counter, so its sampler
#: position is unknowable; :meth:`SFTTrainer.resume` refuses it unless the caller
#: explicitly opts into ``best_effort``.
TRAINING_STATE_SCHEMA = 2

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
    "save_steps",
    "eval_steps",
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
    #: Optimizer steps the GradScaler SKIPPED because a gradient was not finite.
    #: They are not counted in ``steps`` (no parameter moved, no LR consumed).
    skipped_steps: int = 0
    #: The LR schedule horizon (independent of any early stop).
    planned_total_steps: int | None = None
    #: Early-stop limit, ``None`` when the run was not cut short by one.
    stop_after_steps: int | None = None
    #: ``False`` only after a best-effort (schema-1) resume.
    exact_resume: bool = True
    #: ``True`` when the horizon was deliberately changed on resume.
    schedule_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "val_loss": self.val_loss,
            "runtime_s": self.runtime_s,
            "checkpoint_dir": self.checkpoint_dir,
            "metrics_history": list(self.metrics_history),
            "skipped_steps": self.skipped_steps,
            "planned_total_steps": self.planned_total_steps,
            "stop_after_steps": self.stop_after_steps,
            "exact_resume": self.exact_resume,
            "schedule_modified": self.schedule_modified,
        }

@dataclass
class TrainingProgress:
    """Resumable position: THE NEXT WORK ITEM, plus every stream needed to replay it.

    ``global_step`` counts COMPLETED optimizer steps that actually updated the
    parameters (a GradScaler-skipped step does not count). ``epoch_index`` is
    1-based and names the epoch *in progress*; ``next_micro_index`` counts the
    micro-batches already consumed in that epoch, so the next batch is
    ``permutation[next_micro_index * micro_batch : ...]``. ``permutation`` is the
    literal example order of that epoch and ``sampler_rng_state`` is the sampler
    RNG *after* the shuffle that produced it, so epoch N+1.. match the
    uninterrupted run.
    """

    schema_version: int = TRAINING_STATE_SCHEMA
    global_step: int = 0
    epoch_index: int = 0
    next_micro_index: int = 0
    permutation: list[int] = field(default_factory=list)
    sampler_rng_state: list[int] = field(default_factory=list)
    sampler_rng_gauss_next: float | None = None
    planned_total_steps: int = 0
    stop_after_steps: int | None = None
    optimizer_steps_in_epoch: int = 0
    epochs: float = 1.0
    dataset_size: int = 0
    dataset_order_hash: str | None = None
    torch_rng_state: list[int] | None = None
    cuda_rng_state: list[list[int]] | None = None
    numpy_rng_state: dict[str, Any] | None = None
    python_rng_state: list[int] | None = None
    python_rng_gauss_next: float | None = None
    pending_microbatches: int = 0
    last_evaluated_step: int | None = None
    last_saved_step: int | None = None
    exact_resume: bool = True
    schedule_modified: bool = False
    #: Digest of the TRAINING-affecting feature switches (``None`` when the features
    #: slice is absent). A resume across a change must be refused: the supervised
    #: data distribution is not the one the optimizer was stepping on.
    training_feature_fingerprint: str | None = None
    #: Enabled feature ids, so a refusal can name what differs.
    training_features: list[str] = field(default_factory=list)
    world: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingProgress":
        """Build from a persisted ``progress`` block (unknown keys ignored)."""
        known = {f.name for f in dataclasses.fields(cls)}
        values = {key: value for key, value in dict(payload or {}).items() if key in known}
        progress = cls(**values)
        progress.permutation = [int(i) for i in (progress.permutation or [])]
        return progress

def dataset_order_hash(examples: Sequence[Mapping[str, Any]]) -> str:
    """16-hex digest of the training EXAMPLES in order (ids + labels).

    Two runs whose hash differs see different tokens in a different order, so
    their sampler positions are not interchangeable and a resume must refuse.
    """
    digest = hashlib.sha256()
    for example in examples:
        for key in ("input_ids", "labels"):
            values = [int(v) for v in (example.get(key) or [])]
            digest.update(len(values).to_bytes(4, "little"))
            digest.update(struct.pack(f"<{len(values)}i", *values))
    return digest.hexdigest()[:16]

def _numpy_rng_state() -> dict[str, Any] | None:
    """``numpy.random`` legacy MT19937 state as JSON-safe primitives (``None`` if absent)."""
    try:  # pragma: no cover - optional dependency
        import numpy as np  # noqa: PLC0415
    except Exception:
        return None
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "kind": str(kind),
        "keys": [int(k) for k in keys],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached": float(cached),
    }

def _restore_numpy_rng(payload: Mapping[str, Any]) -> None:
    import numpy as np  # noqa: PLC0415 - only reached when a state was saved

    np.random.set_state(
        (
            str(payload.get("kind") or "MT19937"),
            np.array([int(k) for k in payload["keys"]], dtype=np.uint32),
            int(payload["pos"]),
            int(payload.get("has_gauss") or 0),
            float(payload.get("cached") or 0.0),
        )
    )

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

    WRITE ORDER (a resumable checkpoint must never look complete while a piece is
    missing). Inside one checkpoint directory the sequence is:

    1. weights + tokenizer + ``lifemate_meta.json`` (``save_fn``),
    2. ``finalize(target)`` — the optimizer/scheduler/GradScaler tensors,
    3. ``training_state.json``, LAST.

    ``training_state.json`` is the marker: schema 2 makes it claim
    ``exact_resume: true``, so everything it vouches for must already be durable.
    ``finalize`` is optional and defaults to ``None`` (weight-only checkpoints,
    which is what the stdlib/test-double path uses); its return value is ignored.
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

    def save(
        self,
        step: int,
        payload: Mapping[str, Any],
        tag: str = "step",
        finalize: Callable[[Path], Any] | None = None,
    ) -> Path:
        """Write one checkpoint; refuses to reuse a populated directory.

        ``finalize`` runs AFTER the weights and BEFORE ``training_state.json``, so
        the file that advertises resumability is written last (see the class
        docstring for the full order). It may return anything; it is ignored.
        """
        target = self.step_dir(step, tag)
        if target.exists() and any(target.iterdir()):
            raise LifeMateError(
                f"checkpoint directory already populated: {target}. "
                "Refusing to overwrite (resume with --resume or pick another --output-dir)."
            )
        target.mkdir(parents=True, exist_ok=True)
        written = Path(self.save_fn(target, payload))
        if finalize is not None:
            finalize(target)
        _atomic_write_json(target / STATE_FILENAME, dict(payload.get("state") or {}))
        if tag == "step":
            self._prune()
        return written

    def _prune(self) -> None:
        """Keep the newest ``save_total_limit`` step dirs (0 = keep everything).

        The NEWEST step dir is never pruned, even with ``save_total_limit=1`` and
        a resume about to read it: a checkpoint that a resume needs must survive
        the write that follows it.
        """
        if self.save_total_limit <= 0:
            return
        dirs = sorted(
            (path for path in self.root.glob("step-*") if path.is_dir()),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        keep = max(1, self.save_total_limit)
        for stale in dirs[: max(0, len(dirs) - keep)]:
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


def _target_suffix_inputs(model: Any, inputs: dict[str, Any], labels: Any) -> tuple[dict[str, Any], Any]:
    """Keep the predecessor of the first target so causal shifting stays exact."""
    import inspect

    underlying = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    try:
        supports_suffix = "logits_to_keep" in inspect.signature(underlying.forward).parameters
    except (TypeError, ValueError, AttributeError):
        supports_suffix = False
    if supports_suffix:
        positions = (labels != -100).nonzero(as_tuple=False)
        if positions.numel():
            first = max(0, int(positions[:, -1].min().item()) - 1)
            if first:
                inputs = dict(inputs)
                inputs["logits_to_keep"] = int(labels.shape[-1]) - first
                labels = labels[..., first:]
    return inputs, labels


def _forward_loss(model: Any, batch: Mapping[str, Any], loss_chunk_tokens: int = 0) -> Any:
    """Run the model on a tensor batch and return its scalar loss tensor.

    ``loss_chunk_tokens <= 0`` keeps the original single-shot path (the model
    computes the loss itself). A positive value asks for logits and computes the
    chunked loss here; a model that returns no logits falls back to the original
    path rather than failing.
    """
    if loss_chunk_tokens > 0 and "labels" in batch:
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        labels = batch["labels"]
        inputs, labels = _target_suffix_inputs(model, inputs, labels)
        try:
            output = model(**inputs)
        except TypeError:  # pragma: no cover - model requires labels
            output = None
        logits = getattr(output, "logits", None)
        if logits is not None:
            return chunked_causal_lm_loss(logits, labels, loss_chunk_tokens)

    output = model(**batch)
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, (tuple, list)) and output:
        loss = output[0]
    if loss is None:
        raise LifeMateError(
            "model returned no loss; pass labels= so the causal-LM head computes cross-entropy"
        )
    return loss

def forward_loss(model: Any, batch: Mapping[str, Any], loss_chunk_tokens: int = 0) -> Any:
    """Public alias of the trainer's own loss entry point for external callers.

    Same maths, same single-shape contract: ``loss_chunk_tokens > 0`` asks for
    logits and computes the chunked CE here, ``<= 0`` keeps the model's own
    single-shot loss. Notebooks and probes must call THIS, never a hand-rolled
    ``model(**_batch).loss``, or they re-introduce the fp32 ``[seq, vocab]``
    upcast the chunking exists to avoid.
    """
    return _forward_loss(model, batch, loss_chunk_tokens)

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
        stop_after_steps: int | None = None,
        planned_total_steps: int | None = None,
        extend_horizon: bool = False,
        best_effort: bool = False,
        stop: StopController | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        is_lora: bool = False,
        dataset_manifest_hash: str | None = None,
        experiment_provenance: dict | None = None,
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
        self.exposure_counts = {"clean": 0, "conditioned": 0}
        self.experiment_provenance = experiment_provenance
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
        # ``max_steps`` is the historical spelling of the early stop. It NEVER
        # defines the LR schedule any more: a 2-step probe would otherwise define
        # a 2-step cosine schedule for the real run.
        self.stop_after_steps = (
            int(stop_after_steps) if stop_after_steps is not None
            else (int(max_steps) if max_steps is not None else None)
        )
        if self.stop_after_steps is not None and self.stop_after_steps <= 0:
            self.stop_after_steps = None
        # Schedule horizon, resolved in ``run()`` when ``None``.
        self.planned_total_steps = int(planned_total_steps) if planned_total_steps else None
        self.extend_horizon = bool(extend_horizon)
        self.best_effort = bool(best_effort)
        # Legacy attribute kept: some callers/tests read ``trainer.max_steps``.
        self.max_steps = max_steps
        # Cadence: 0 falls back to logging_steps (documented in TrainingConfig).
        self.save_every = int(getattr(training, "save_steps", 0) or 0) or None
        self.eval_every = int(getattr(training, "eval_steps", 0) or 0) or None
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
        # Exact-resume bookkeeping (see TrainingProgress).
        self.progress = TrainingProgress(epochs=float(self.epochs))
        self.exact_resume = True
        self.schedule_modified = False
        self.stop_after_requested = self.stop_after_steps
        # Sampler RNG: seeded once for a fresh run, restored from the checkpoint
        # for a resume so epoch N+1.. reproduce the uninterrupted permutations.
        self._sampler_rng = random.Random(self.seed)
        # Recorder proving whether the scaler actually applied an update.
        self._update_recorder: dict[str, int] = {"applied": 0, "skipped": 0}
        self.skipped_steps = 0
        self.horizon: int = 0
        self.scheduler_rebuilt = False
        self._epoch_index = 0
        self._current_step = 0
        self._permutation: list[int] = []
        self._cursor = 0
        self._steps_in_epoch = 0
        self._pending = 0
        self._resumed_optimizer_steps: int | None = None
        self._resumed_scheduler_state = False
        self._scheduler_state_loaded = False
        # Set by resume(): does the checkpoint carry the optimizer tensors it claims?
        self._optimizer_state_present = True
        # Set by resume(): did a guard accept this continuation only approximately?
        self._exact_downgraded = False

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
                example["conditioned_exposure"] = (item.get("meta") or {}).get("augmentation") == "conditioned_multi_asr"
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
        """Build the optimizer and scheduler, instrumenting the step BEFORE the schedule.

        Order matters twice: the update recorder must be installed on the optimizer
        before ``LambdaLR`` wraps ``optimizer.step`` (otherwise torch warns on every
        ``scheduler.step()`` that the optimizer was overridden afterwards), and the
        scheduler must be built over the instrumented optimizer for the same reason.
        """
        from .full_finetuning import build_optimizer_and_scheduler, build_scheduler  # noqa: PLC0415

        if self.optimizer is None:
            optimizer, _scheduler = build_optimizer_and_scheduler(
                self.model, self.cfg, num_training_steps
            )
            self.optimizer = optimizer
        self._install_update_recorder()
        if self.scheduler is None:
            self.scheduler = build_scheduler(self.optimizer, self.cfg, num_training_steps)

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
        """Train, evaluate per ``eval_strategy``, checkpoint per ``save_strategy``.

        Two independent numbers drive the loop:

        * ``self.horizon`` (``planned_total_steps``) — the LR schedule length,
          derived from ``num_train_epochs`` and PERSISTED in every checkpoint so a
          resume continues the original cosine trajectory;
        * ``self.stop_after_steps`` — an early stop that ends the run without
          redefining the schedule.
        """
        started = time.time()
        fresh = self.start_step <= 0
        # A resumed run restored torch/random state from the checkpoint; re-seeding
        # here would silently restart the stream from scratch.
        if fresh:
            set_seed(self.seed)
            self._sampler_rng = random.Random(self.seed)
        train_examples = self._prepare_examples(self.train_records, "train")
        val_examples = self._prepare_examples(self.val_records, "val") if self.val_records else []
        self._train_examples = train_examples
        self._val_examples = val_examples
        self._ensure_dataset_order_hash(train_examples)

        persisted = None if fresh else self.progress
        if persisted is not None and persisted.dataset_order_hash and persisted.dataset_order_hash != self._dataset_order_hash:
            raise LifeMateError(
                "cannot resume: dataset order hash differs from the checkpoint "
                f"(checkpoint {persisted.dataset_order_hash} != this run {self._dataset_order_hash}). "
                "The training examples are not the same tokens in the same order (the dataset was "
                "rebuilt, max_seq_length changed, or the records were reordered), so the sampler "
                "position would replay different data. Re-run from scratch, or point --dataset-dir "
                "at the dataset this checkpoint was trained on."
            )

        steps_per_epoch = optimizer_steps_per_epoch(len(train_examples), self.micro_batch, self.accum)
        derived = max(1, int(math.ceil(steps_per_epoch * self.epochs)))
        horizon = self._resolve_horizon(persisted, derived)
        self.horizon = horizon
        self._ensure_optimizer(horizon)
        if not fresh and (self.schedule_modified or not self._resumed_scheduler_state):
            # Rebuild ONLY when the checkpoint carries no scheduler state, or when
            # the horizon was deliberately changed. A restored scheduler already
            # sits at ``last_epoch == start_step`` on the ORIGINAL horizon, which is
            # exactly the trajectory an uninterrupted run would have taken.
            from .full_finetuning import build_scheduler  # noqa: PLC0415 - lazy

            self.scheduler = build_scheduler(
                self.optimizer, self.cfg, horizon, last_epoch=self.start_step
            )
            self.scheduler_rebuilt = True

        self.model.train()
        step = int(self.start_step)
        last_loss: float | None = None
        val_loss: float | None = None
        epochs_done = 0.0
        checkpoint_dir = str(self.checkpoint_manager.root)
        micro_count = max(1, math.ceil(len(train_examples) / self.micro_batch))
        if fresh:
            self._epoch_index, self._permutation, self._cursor, self._steps_in_epoch = 0, [], 0, 0
        self._pending = 0

        def optimizer_step() -> bool:
            """One optimizer step; shared by the accumulate path and the flush.

            Returns ``True`` when the parameters actually moved. A GradScaler
            SKIPS the update on an inf/NaN gradient; that must not consume a step
            number nor a scheduler tick.
            """
            nonlocal step, checkpoint_dir
            if self.scaler is not None:
                # fp16: gradients are still multiplied by the loss scale here, and
                # both the partial-group rescale and clip_grad_norm_ must see TRUE
                # magnitudes, so unscale first and never scale twice.
                self.scaler.unscale_(self.optimizer)
            if 0 < self._pending < self.accum:
                # Partial trailing group: gradients were scaled by 1/accum but only
                # ``pending`` micro-batches contributed, so rescale by accum/pending
                # to keep the effective loss scale identical to a full group.
                factor = self.accum / self._pending
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(factor)
            self._clip_gradients()
            before = self._update_recorder["applied"]
            if self.scaler is not None:
                # Inf/NaN guard: ``step`` skips the update when a grad is not
                # finite; ``update`` then backs the scale off.
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            updated = self._update_recorder["applied"] > before
            self.optimizer.zero_grad(set_to_none=True)
            self._pending = 0
            if not updated:
                self.skipped_steps += 1
                self.metrics_history.append({"step": step, "skipped_step": True})
                self.tracker.log({"step": step, "skipped_step": 1})
                return False
            if self.scheduler is not None:
                self.scheduler.step()
            step += 1
            self._current_step = step
            self._steps_in_epoch += 1
            self._maybe_log(step, last_loss)
            if val_examples and self.eval_strategy == "steps" and self._due(self.eval_strategy, step, self.eval_every):
                metrics = self.evaluate(val_examples)
                val_loss = metrics["val_loss"]
                self.metrics_history.append({"step": step, "epoch": self._epoch_index, **metrics})
                self.tracker.log({"step": step, **metrics})
                self._last_evaluated_step = step
            if self.save_strategy == "steps" and self._due(self.save_strategy, step, self.save_every):
                checkpoint_dir = str(self._save(step, self._epoch_index))
            return True

        def finish_interrupt() -> None:
            """Finish a partial group, then write the interrupt checkpoint."""
            nonlocal checkpoint_dir
            if self._pending:
                optimizer_step()  # C: never checkpoint on a half-accumulated group
            if self._last_saved_step == step:
                # This step already has a durable checkpoint (steps cadence or the
                # final save); reuse it instead of failing on a populated dir.
                checkpoint_dir = str(self.checkpoint_manager.step_dir(step))
            else:
                checkpoint_dir = str(self._save(step, self._epoch_index, tag="interrupt"))
            raise TrainingInterrupted(
                f"stop requested; interrupt checkpoint written at step {step}",
                step=step,
                checkpoint_dir=checkpoint_dir,
            )

        try:
            epoch_start_step = step
            while not self._limit_reached(step):
                if not self._permutation or self._cursor >= micro_count:
                    self._advance_epoch(len(train_examples))
                    epoch_start_step = step
                # Consume the epoch's micro-batches from the stored cursor. On a
                # resume this skips exactly the ones the interrupted run already
                # consumed: the permutation is replayed, never regenerated.
                while self._cursor < micro_count and not self._limit_reached(step):
                    start = self._cursor * self.micro_batch
                    chunk = self._permutation[start : start + self.micro_batch]
                    for i in chunk:
                        exposure_key = 'conditioned' if train_examples[i].get('conditioned_exposure') else 'clean'
                        self.exposure_counts[exposure_key] += 1
                    batch = self.collator([train_examples[i] for i in chunk])
                    tensors = to_torch_batch(batch, self.device)
                    with self._autocast():
                        loss = _forward_loss(self.model, tensors, self.loss_chunk_tokens)
                    loss_value = float(loss.detach().item())
                    if not math.isfinite(loss_value):
                        raise LifeMateError(f"non-finite training loss at step {step}: {loss_value}")
                    self._pending += 1
                    self._cursor += 1
                    # Scale by the accumulation count; a short trailing group is
                    # rescaled by accum/pending before its optimizer step. Under
                    # fp16 the scaler multiplies that by its dynamic loss scale.
                    scaled_loss = loss / self.accum
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    last_loss = loss_value
                    if self._pending >= self.accum:
                        optimizer_step()
                        if self.stop.requested:
                            finish_interrupt()

                # Flush a partial accumulation group so no grad is silently dropped
                # and the loop always makes progress (a dataset smaller than one
                # effective batch would otherwise never step). Skipped at the run's
                # limit so the run never overshoots its horizon / stop step.
                if self._cursor >= micro_count and self._pending and not self._limit_reached(step):
                    optimizer_step()
                    if self.stop.requested:
                        finish_interrupt()

                boundary = self._cursor >= micro_count
                if boundary:
                    epochs_done = float(self._epoch_index)
                    if step == epoch_start_step:
                        raise LifeMateError(
                            f"epoch {self._epoch_index} produced no optimizer update: every gradient "
                            "was non-finite and the scaler skipped all of them. The run cannot make "
                            "progress; check the data (inf/NaN targets) or reduce the learning rate."
                        )
                final = self._limit_reached(step)
                # Epoch-based evaluation & saving (always also at the run's final
                # step, so an early stop leaves a resumable checkpoint). Both are
                # deduplicated against the PERSISTED step numbers, so a resume never
                # re-runs a step that was already evaluated/saved.
                want_eval = self.eval_strategy not in {"no", "none"} and (
                    final or self._due(self.eval_strategy, step, self.eval_every)
                )
                if val_examples and want_eval and self._last_evaluated_step != step:
                    metrics = self.evaluate(val_examples)
                    val_loss = metrics["val_loss"]
                    self.metrics_history.append({"step": step, "epoch": self._epoch_index, **metrics})
                    self.tracker.log({"step": step, **metrics})
                    self._last_evaluated_step = step
                want_save = self.save_strategy not in {"no", "none"} and (
                    final or self._due(self.save_strategy, step, self.save_every)
                )
                if want_save and self._last_saved_step != step:
                    checkpoint_dir = str(self._save(step, self._epoch_index))
                if boundary and self.stop.requested:
                    finish_interrupt()
        except TrainingInterrupted:
            raise
        except BaseException:
            self._save_crash_safe(step, self._epoch_index)
            raise
        finally:
            self.tracker.finish()

        if self._pending:  # pragma: no cover - every exit path commits or flushes
            raise LifeMateError(
                f"internal: run() finished with {self._pending} uncommitted micro-batch(es) of "
                "gradient accumulation; the checkpoint boundary invariant was violated"
            )

        return TrainResult(
            steps=step,
            epochs=float(epochs_done),
            final_loss=last_loss,
            val_loss=val_loss,
            runtime_s=round(time.time() - started, 3),
            checkpoint_dir=checkpoint_dir,
            metrics_history=list(self.metrics_history),
            skipped_steps=self.skipped_steps,
            planned_total_steps=self.horizon,
            stop_after_steps=self.stop_after_steps,
            exact_resume=self.exact_resume,
            schedule_modified=self.schedule_modified,
        )

    def _limit_reached(self, step: int) -> bool:
        """``True`` when the schedule horizon or the early stop is reached."""
        if step >= int(self.horizon or 0):
            return True
        return self.stop_after_steps is not None and step >= self.stop_after_steps

    def _resolve_horizon(self, persisted: TrainingProgress | None, derived: int) -> int:
        """The LR schedule length, refusing an unannounced change on resume."""
        stored = int(persisted.planned_total_steps) if persisted is not None and persisted.planned_total_steps else None
        requested = self.planned_total_steps
        if stored is None:
            return max(1, int(requested or derived))
        if requested is None or int(requested) == stored:
            return max(1, stored)
        if not self.extend_horizon:
            raise LifeMateError(
                "cannot resume: the schedule horizon differs from the checkpoint's "
                f"(checkpoint planned_total_steps={stored}, this run requests {requested}). "
                "Continuing would silently train a different LR trajectory, not a continuation. "
                "Pass --extend-horizon (SFTTrainer(extend_horizon=True)) to authorise the change, "
                "or drop --planned-steps to continue the recorded schedule."
            )
        self.schedule_modified = True
        self._announce_horizon_change(stored, int(requested))
        return max(1, int(requested))

    def _announce_horizon_change(self, old: int, new: int) -> None:
        """One loud line per run, however many times the change is observed."""
        if getattr(self, "_horizon_change_announced", None) == (old, new):
            return
        self._horizon_change_announced = (old, new)
        print(
            f"schedule: HORIZON EXTENDED {old} -> {new} "
            "(this run is a modified training schedule, not a continuation)"
        )

    def _ensure_dataset_order_hash(self, examples: Sequence[Mapping[str, Any]] | None = None) -> str | None:
        """Digest of the tokenised examples in order (cached after the first call).

        Shared by ``run()`` and ``resume()``: a checkpoint whose recorded order hash
        differs must be refused BEFORE its weights are loaded.
        """
        if getattr(self, "_dataset_order_hash", None):
            return self._dataset_order_hash
        if examples is None:
            if not self.train_records:
                return None
            # ``_prepare_examples`` records a diagnostic entry; a resume must not
            # add a second one for the same split, so restore the list afterwards.
            before = len(self.metrics_history)
            try:
                examples = self._prepare_examples(self.train_records, "train")
            except LifeMateError:  # pragma: no cover - unusable records
                del self.metrics_history[before:]
                return None
            del self.metrics_history[before:]
            self._train_examples = examples
        self._dataset_order_hash = dataset_order_hash(examples)
        return self._dataset_order_hash

    def _advance_epoch(self, n_examples: int) -> None:
        """Start the next epoch: one shuffle from the persisted sampler stream."""
        self._epoch_index += 1
        order = list(range(int(n_examples)))
        self._sampler_rng.shuffle(order)
        self._permutation = order
        self._cursor = 0
        self._steps_in_epoch = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _due(self, strategy: str, step: int, every: int | None = None) -> bool:
        """``epoch`` (end of each epoch) / ``steps`` (every ``every``) / ``no``.

        ``every`` is ``save_steps`` for ``save_strategy`` and ``eval_steps`` for
        ``eval_strategy`` (0 in the config falls back to ``logging_steps``), so the
        two cadences are independent. It was previously ``logging_steps`` for both.
        """
        key = str(strategy or "epoch").lower()
        if key in {"no", "none"} or step <= 0:
            return False
        if key == "steps":
            cadence = int(every or self.log_every or 1)
            return step % max(1, cadence) == 0
        return True

    def _clip_gradients(self) -> None:
        if self.grad_clip <= 0:
            return
        try:
            import torch  # noqa: PLC0415

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip,
                                           error_if_nonfinite=self.scaler is None)
        except ImportError:  # pragma: no cover - torch absent
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
        if self.experiment_provenance:
            payload["conditioned_exposures"] = self.exposure_counts["conditioned"]
            payload["clean_exposures"] = self.exposure_counts["clean"]
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
                    inputs, eval_labels = _target_suffix_inputs(self.model, inputs, labels)
                    try:
                        with self._autocast():
                            output = self.model(**inputs)
                    except TypeError:  # pragma: no cover - model requires labels
                        output = None
                    logits = getattr(output, "logits", None)
                    if logits is not None:
                        shift_labels = eval_labels[..., 1:]
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
    def _install_update_recorder(self) -> None:
        """Wrap the optimizer's bound ``step`` so a skipped update is detectable.

        ``GradScaler.step(optimizer)`` calls ``optimizer.step()`` only when no
        inf/NaN was found, so an invocation count IS the ground truth for "did the
        parameters move". The recorder is a monkeypatched bound method kept on the
        trainer (not an optimizer PROXY): ``build_scheduler``/``torch.save`` require
        a real ``torch.optim.Optimizer``, and a proxy fails ``isinstance`` checks.
        Idempotent per optimizer object.
        """
        import types  # noqa: PLC0415 - one local import, keeps the top clean

        target = self.optimizer
        if target is None or getattr(target, "_omp_step_recorded", False):
            return
        recorder = self._update_recorder
        original = target.step

        def recorded_step(_opt: Any, *args: Any, **kwargs: Any) -> Any:
            recorder["applied"] = int(recorder.get("applied", 0)) + 1
            return original(*args, **kwargs)

        try:
            # ``MethodType``, not a bare function: torch's LR scheduler re-wraps
            # ``opt.step`` and reads ``step_fn.__func__``, which a plain function
            # assigned to the instance does not have.
            target.step = types.MethodType(recorded_step, target)  # type: ignore[method-assign]
            target._omp_step_recorded = True  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # pragma: no cover - exotic optimizer
            return

    def _feature_identity(self) -> dict[str, Any]:
        """Training-affecting feature identity: ``{fingerprint, enabled}`` or empty.

        Training features change the supervised DATA distribution, so a resume across
        a change is not a continuation. Inference-only ids never move the
        fingerprint (that is the registry's own split), and the enabled-id list is
        stored alongside so the refusal can name exactly which flags differ.
        """
        features = getattr(self.cfg, "features", None)
        if features is None:
            return {}
        try:  # pragma: no cover - depends on which slice the checkout carries
            from lifemate.features.config import (  # noqa: PLC0415
                enabled_features,
                training_feature_fingerprint,
            )
        except ImportError:  # pragma: no cover - features slice absent
            return {}
        return {
            "training_feature_fingerprint": training_feature_fingerprint(features),
            "training_features": list(enabled_features(features)),
        }

    def _world(self) -> dict[str, Any]:
        """Identity of the training world; a mismatch means "not the same model"."""
        from lifemate.models.profiles import resolve_profile  # noqa: PLC0415 - lazy

        model_name = getattr(getattr(self.cfg, "model", None), "name", None)
        tuning_mode = str(getattr(self.training, "tuning_mode", "lora" if self.is_lora else "full"))
        return {
            "model_id": model_name,
            "model_profile": (self.profile.name if self.profile else resolve_profile(model_name or "").name),
            "tuning_mode": tuning_mode,
            "quantization": str(getattr(self.training, "quantization", "none") or "none"),
            "precision": self.precision,
            # A full fine-tune has no adapter, so ``lora.r`` is inert there (it is
            # excluded from the config hash for the same reason): record ``None``
            # rather than a rank that would refuse a legitimate resume.
            "adapter_rank": (
                int(getattr(getattr(self.cfg, "lora", None), "r", 0) or 0)
                if tuning_mode in {"lora", "qlora"} else None
            ),
        }

    def _cuda_rng_state(self) -> list[list[int]] | None:
        try:  # pragma: no cover - GPU only
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return [s.tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception:  # pragma: no cover - torch absent
            return None
        return None

    def _optimizer_state_expected(self) -> bool:
        """``True`` when this trainer can actually persist optimizer/scheduler tensors.

        Drives ``exact_resume``: a run with no torch optimizer (or no torch at all)
        writes a weight-only checkpoint, which is NOT an exact resume target
        because AdamW moments and the GradScaler state cannot come back. That is
        legitimate (the stdlib runner and test doubles) but must never be labelled
        exact.
        """
        if self.optimizer is None:
            return False
        try:
            import torch  # noqa: F401,PLC0415 - probe only

            return True
        except Exception:  # pragma: no cover - torch absent
            return False

    def _torch_rng_state(self) -> list[int] | None:
        try:
            import torch  # noqa: PLC0415

            return torch.get_rng_state().tolist()
        except Exception:  # pragma: no cover - torch absent
            return None

    def progress_state(self) -> dict[str, Any]:
        """The next-work-item snapshot this trainer would checkpoint right now.

        Safe to call between runs and at the end of one (used by
        ``scripts/verify_training_resume.py`` and the CLI's resume banner).
        """
        return self._progress_state(
            int(self._current_step), int(self._epoch_index), allow_pending=True
        ).to_dict()

    def _progress_state(self, step: int, epoch: int, *, allow_pending: bool = False) -> TrainingProgress:
        """Snapshot describing the NEXT work item at the current boundary."""
        py_state = random.getstate()
        sampler_state = self._sampler_rng.getstate()
        if int(self._pending or 0) != 0 and not allow_pending:
            raise LifeMateError(
                f"internal: refusing to write a resumable checkpoint with {self._pending} "
                "micro-batch(es) of uncommitted gradients; checkpoint only at an optimizer-step "
                "or epoch boundary"
            )
        identity = self._feature_identity()
        progress = TrainingProgress(
            schema_version=TRAINING_STATE_SCHEMA,
            global_step=int(step),
            epoch_index=int(epoch),
            next_micro_index=int(self._cursor if self._permutation else 0),
            permutation=list(self._permutation),
            # Compact JSON-safe form: the internal 625-word tuple + gauss cache.
            sampler_rng_state=[int(v) for v in sampler_state[1]],
            sampler_rng_gauss_next=(
                float(sampler_state[2]) if sampler_state[2] is not None else None
            ),
            planned_total_steps=int(getattr(self, "horizon", 0) or 0),
            stop_after_steps=self.stop_after_steps,
            optimizer_steps_in_epoch=int(self._steps_in_epoch),
            epochs=float(self.epochs),
            dataset_size=len(getattr(self, "_train_examples", []) or []),
            dataset_order_hash=getattr(self, "_dataset_order_hash", None),
            torch_rng_state=self._torch_rng_state(),
            cuda_rng_state=self._cuda_rng_state(),
            numpy_rng_state=_numpy_rng_state(),
            python_rng_state=[int(v) for v in py_state[1]],
            python_rng_gauss_next=(
                float(py_state[2]) if py_state[2] is not None else None
            ),
            pending_microbatches=int(self._pending or 0),
            last_evaluated_step=self._last_evaluated_step,
            last_saved_step=self._last_saved_step,
            exact_resume=bool(self.exact_resume) and self._optimizer_state_expected(),
            schedule_modified=bool(self.schedule_modified),
            training_feature_fingerprint=identity.get("training_feature_fingerprint"),
            training_features=list(identity.get("training_features") or []),
            world={**self._world(), **identity},
        )
        return progress

    def _state(self, step: int, epoch: int, *, allow_pending: bool = False) -> dict[str, Any]:
        """``training_state.json`` payload: legacy keys + the schema-2 ``progress`` block."""
        progress = self._progress_state(step, epoch, allow_pending=allow_pending)
        world = progress.world
        state = {
            # --- legacy keys: read by existing notebooks/tests, keep every one ---
            "step": step,
            "epoch": epoch,
            "seed": self.seed,
            "config_hash": config_fingerprint(self.cfg),
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "experiment_provenance": self.experiment_provenance,
            "exposure_counts": dict(self.exposure_counts),
            "is_lora": self.is_lora,
            "precision": self.precision,
            "torch_rng_state": progress.torch_rng_state,
            "cuda_rng_state": progress.cuda_rng_state,
            "python_rng_state": progress.python_rng_state,
            # --- schema 2 ---
            "schema_version": TRAINING_STATE_SCHEMA,
            "exact_resume": progress.exact_resume,
            "schedule_modified": progress.schedule_modified,
            "progress": progress.to_dict(),
            "world": world,
            "training_feature_fingerprint": progress.training_feature_fingerprint,
            "training_features": list(progress.training_features),
        }
        # One step counter, asserted: the legacy ``step`` and the progress block
        # cannot disagree (a second counter is how resume bugs are born).
        assert state["step"] == state["progress"]["global_step"], (
            state["step"],
            state["progress"]["global_step"],
        )
        return state

    def _save(self, step: int, epoch: int, tag: str = "step") -> Path:
        """Write one checkpoint: weights -> optimizer_state.pt -> training_state.json.

        The order is the guarantee. ``training_state.json`` claims
        ``exact_resume: true`` and carries the sampler cursor, so it must not exist
        until the optimizer/scheduler/scaler tensors it vouches for are durable —
        otherwise a crash in between leaves a directory that ``resume()`` would
        accept as exact while AdamW silently restarts from zero moments. The
        tensors are therefore written through ``CheckpointManager.save``'s
        ``finalize`` hook, which runs before the last (state) write.
        """
        if tag == "step" and self._last_saved_step == step:
            return self.checkpoint_manager.step_dir(step, tag)
        world = self._world()
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
                "experiment_provenance": self.experiment_provenance,
                "dataset_order_hash": getattr(self, "_dataset_order_hash", None),
                "config_hash": config_fingerprint(self.cfg),
                "tuning_mode": world["tuning_mode"],
                "quantization": world["quantization"],
                "precision": self.precision,
                "adapter_rank": world["adapter_rank"],
                "planned_total_steps": int(getattr(self, "horizon", 0) or 0),
                "global_step": int(step),
                "schema_version": TRAINING_STATE_SCHEMA,
                "exact_resume": bool(self.exact_resume) and self._optimizer_state_expected(),
                "schedule_modified": bool(self.schedule_modified),
                **self._feature_identity(),
            },
        }
        written = self.checkpoint_manager.save(
            step,
            payload,
            tag=tag,
            finalize=lambda target: self._save_optimizer_state(target, step, epoch),
        )
        if tag == "step":
            self._last_saved_step = step
        return written

    def _save_optimizer_state(self, checkpoint_dir: Path, step: int, epoch: int) -> bool:
        """Optimizer/scheduler tensors via ``torch.save``; ``True`` when written.

        Runs as ``CheckpointManager.save(finalize=...)``, i.e. BEFORE
        ``training_state.json``. Returns ``False`` when there is nothing to write
        (no torch, or no optimizer) so the caller can refuse to claim exactness;
        a genuine write failure with a live optimizer raises instead, because that
        must never be published as a resumable step.
        """
        try:
            import torch  # noqa: PLC0415
        except Exception:  # pragma: no cover - torch absent (stdlib runner)
            return False
        if self.optimizer is None:
            return False
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            # fp16 grad-scaler state (None on bf16/fp32/CPU runs).
            "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": step,
            "epoch": epoch,
            # The horizon the scheduler state belongs to; a resume compares it
            # against the persisted ``planned_total_steps`` before trusting it.
            "planned_total_steps": int(getattr(self, "horizon", 0) or 0),
            "scheduler_class": type(self.scheduler).__name__ if self.scheduler is not None else None,
        }
        tmp = checkpoint_dir / "optimizer_state.pt.tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, checkpoint_dir / "optimizer_state.pt")
        except Exception as exc:  # pragma: no cover - unserialisable state
            tmp.unlink(missing_ok=True)
            raise LifeMateError(
                f"failed to write optimizer state to {checkpoint_dir}: {exc}. The checkpoint "
                "would resume with AdamW moments reset to zero, so it was not published."
            ) from exc
        return True

    def _save_crash_safe(self, step: int, epoch: int) -> None:
        """Best-effort ``crash-<step>`` state; never masks the original error.

        A crash can land mid-accumulation, so this checkpoint deliberately records
        a nonzero ``pending_microbatches``: it exists to be diagnosed, and
        :meth:`resume` refuses it for an EXACT resume.
        """
        try:
            target = self.checkpoint_manager.step_dir(step, "crash")
            target.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(target / STATE_FILENAME, self._state(step, epoch, allow_pending=True))
        except Exception:  # pragma: no cover - disk full etc.
            return

    def resume(self, checkpoint_dir: str | Path, *, best_effort: bool | None = None) -> int:
        """Restore weights + optimizer + sampler position from ``checkpoint_dir``.

        Returns the completed optimizer step. Fails loudly when the checkpoint is
        unreadable/stateless, when its tuning mode, dataset or config differ, when
        the world (model profile, quantization, adapter rank) differs, when it has
        no ``optimizer_state.pt`` (see :meth:`_require_optimizer_state`), when the
        schedule horizon differs without ``extend_horizon``, when it was written
        mid-accumulation, or when it predates schema 2.
        ``best_effort`` defaults to the trainer's own flag (CLI:
        ``--best-effort-resume``); it downgrades every one of those to an
        approximate continuation and sets ``exact_resume = False``.
        """
        path = Path(str(checkpoint_dir))
        effort = self.best_effort if best_effort is None else bool(best_effort)
        state = read_training_state(path)
        if not state:
            raise LifeMateError(
                f"cannot resume: {path} has no {STATE_FILENAME}. "
                "Point --resume at a step-<n> directory produced by this trainer."
            )
        step = int(state.get("step") or 0)
        if step <= 0:
            raise LifeMateError(f"cannot resume: {path} records step {step}; expected a positive step")
        stored_provenance = state.get('experiment_provenance')
        if stored_provenance != self.experiment_provenance and (stored_provenance or self.experiment_provenance):
            oldp=dict(stored_provenance or {}); newp=dict(self.experiment_provenance or {})
            oldp.pop('source_commit',None); newp.pop('source_commit',None)
            old_horizon=int((state.get('progress') or {}).get('planned_total_steps') or 0)
            if not (isinstance(stored_provenance, dict) and isinstance(self.experiment_provenance, dict)
                    and stored_provenance.get('source_commit') and self.experiment_provenance.get('source_commit')
                    and self.extend_horizon and self.planned_total_steps and self.planned_total_steps > old_horizon > 0 and oldp == newp):
                raise LifeMateError('cannot resume: warm-start provenance differs from checkpoint')
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
                recorded_progress = state.get('progress') or {}
                old_epochs = recorded_progress.get('epochs')
                old_horizon = int(recorded_progress.get('planned_total_steps') or 0)
                only_epoch_extension = False
                if (self.extend_horizon and self.planned_total_steps and old_horizon > 0
                        and self.planned_total_steps > old_horizon and isinstance(old_epochs, (int, float))
                        and self.epochs > old_epochs > 0):
                    for old_value in (old_epochs, int(old_epochs) if float(old_epochs).is_integer() else old_epochs):
                        previous_cfg = copy.deepcopy(self.cfg)
                        if isinstance(previous_cfg, dict):
                            previous_cfg['training']['num_train_epochs'] = old_value
                        else:
                            previous_cfg.training.num_train_epochs = old_value
                        if config_fingerprint(previous_cfg) == stored_config_hash:
                            only_epoch_extension = True
                            break
                if not only_epoch_extension:
                    raise LifeMateError(
                    "cannot resume: training config differs from the checkpoint "
                    f"({stored_config_hash} != {current}); the run would not continue the "
                    "same optimisation (the guard hashes training.* minus bookkeeping, "
                    "model.*, and lora.* only in LoRA mode; check lr, seed, precision, "
                    "max_seq_length, batch settings, epochs, scheduler, tuning_mode)"
                )

        progress = self._read_progress(path, state, effort)
        if progress.dataset_order_hash:
            current = self._ensure_dataset_order_hash()
            if current and current != progress.dataset_order_hash:
                raise LifeMateError(
                    "cannot resume: dataset order hash differs from the checkpoint "
                    f"(checkpoint {progress.dataset_order_hash} != this run {current}). "
                    "The training examples are not the same tokens in the same order (the dataset "
                    "was rebuilt, max_seq_length changed, or the records were reordered), so the "
                    "sampler position would replay different data. Point --dataset-dir at the "
                    "dataset this checkpoint was trained on, or re-run from scratch."
                )
        self._check_world(path, state, progress)
        self._check_manifest_compat(path)
        stored_horizon = int(progress.planned_total_steps or 0)
        if stored_horizon and self.planned_total_steps and stored_horizon != int(self.planned_total_steps):
            if not self.extend_horizon:
                raise LifeMateError(
                    "cannot resume: the schedule horizon differs from the checkpoint's "
                    f"(checkpoint planned_total_steps={stored_horizon}, this run requests "
                    f"{int(self.planned_total_steps)}). Continuing would silently train a "
                    "different LR trajectory. Pass --extend-horizon / SFTTrainer(extend_horizon=True) "
                    "to authorise it, or drop --planned-steps to keep the recorded schedule."
                )
            self.schedule_modified = True
            self._announce_horizon_change(stored_horizon, int(self.planned_total_steps))

        self._check_feature_identity(path, progress, effort)
        self._require_optimizer_state(path, step, effort)
        self._load_weights(path)
        # The optimizer does not exist yet on a fresh trainer, so build it before
        # restoring; otherwise ``_load_optimizer_state`` returns early and the AdamW
        # moments are silently dropped (resume would restart from zero moments).
        # Only do it when the checkpoint actually carries optimizer state, so models
        # without parameters (test doubles) keep working.
        if self.optimizer is None and (path / "optimizer_state.pt").is_file():
            self._ensure_optimizer(stored_horizon or int(self.planned_total_steps or 0) or max(1, step))
        self._install_update_recorder()
        self._load_optimizer_state(path)
        self._resumed_scheduler_state = bool(self._scheduler_state_loaded) and not self.schedule_modified
        if self.scheduler is not None and not self.schedule_modified:
            # Guard the restored schedule: a scheduler state built for a different
            # horizon would put every remaining step at the wrong LR.
            last_epoch = int(getattr(self.scheduler, "last_epoch", step) or 0)
            if last_epoch > 0 and last_epoch != step:
                raise LifeMateError(
                    f"cannot resume: the checkpoint's scheduler sits at last_epoch={last_epoch} "
                    f"but the checkpoint records step {step}. The saved schedule does not belong "
                    "to this step; re-run from scratch or resume an older checkpoint."
                )

        self._restore_rng(state, progress, exact=not effort)
        self.exposure_counts = dict(state.get("exposure_counts") or {"clean": 0, "conditioned": 0})
        self.progress = progress
        self.progress.global_step = step  # the single counter, asserted below
        self.start_step = step
        self._current_step = step
        self._epoch_index = int(progress.epoch_index)
        self._permutation = list(progress.permutation)
        self._cursor = int(progress.next_micro_index)
        self._steps_in_epoch = int(progress.optimizer_steps_in_epoch)
        self._last_evaluated_step = progress.last_evaluated_step
        self._last_saved_step = progress.last_saved_step
        # Exactness needs ALL of: the checkpoint says so, no effort downgrade was
        # requested, and the optimizer state it vouches for is actually here.
        self.exact_resume = (
            bool(progress.exact_resume)
            and not effort
            and self._optimizer_state_present
            and not self._exact_downgraded
        )
        if progress.sampler_rng_state:
            self._sampler_rng.setstate(
                (3, tuple(int(v) for v in progress.sampler_rng_state), progress.sampler_rng_gauss_next)
            )
        if stored_horizon:
            self.planned_total_steps = (
                int(self.planned_total_steps) if self.schedule_modified else stored_horizon
            )
        assert self.progress.global_step == int(state.get("step") or 0)
        return step

    def _read_progress(self, path: Path, state: Mapping[str, Any], effort: bool) -> TrainingProgress:
        """The schema-2 progress block, or a best-effort/refused reconstruction."""
        schema = int(state.get("schema_version") or 1)
        raw = state.get("progress")
        block = dict(raw) if isinstance(raw, Mapping) else {}
        if schema >= TRAINING_STATE_SCHEMA and block.get("permutation"):
            progress = TrainingProgress.from_dict(block)
            if int(progress.pending_microbatches or 0) != 0:
                if not effort:
                    raise LifeMateError(
                        f"cannot resume exactly: {path} was written with "
                        f"{progress.pending_microbatches} uncommitted micro-batch(es) of gradient "
                        "accumulation, so its optimization boundary is not consistent. That "
                        "checkpoint comes from a crash, not an optimizer step. Re-run from scratch "
                        "or pass best_effort=True / --best-effort-resume to continue approximately."
                    )
                progress.exact_resume = False
            if effort:
                # Loud, and never labelable as exact (requirement J).
                print("resume: BEST EFFORT (not exact)")
            return progress
        if not effort:
            raise LifeMateError(
                f"cannot resume exactly: this checkpoint predates exact-resume state "
                f"(schema {schema}); its sampler position cannot be reconstructed. Re-run from "
                "scratch, or pass best_effort=True/--best-effort-resume to continue approximately "
                "(a DIFFERENT sample order; not equivalent to the uninterrupted run)"
            )
        print("resume: BEST EFFORT (not exact)")
        # Approximate continuation: start epoch 1 and let the sampler RNG produce
        # a fresh order. Never labelled exact.
        return TrainingProgress(
            schema_version=TRAINING_STATE_SCHEMA,
            global_step=int(state.get("step") or 0),
            epoch_index=int(state.get("epoch") or 0),
            next_micro_index=0,
            permutation=[],
            planned_total_steps=int(state.get("planned_total_steps") or 0),
            epochs=float(self.epochs),
            dataset_order_hash=None,
            exact_resume=False,
        )

    def _check_feature_identity(
        self, path: Path, progress: TrainingProgress, effort: bool
    ) -> None:
        """Refuse a resume across a change in TRAINING-affecting feature flags.

        Training features change what the model is supervised on, so the AdamW
        trajectory of the checkpoint is not reproducible under new flags — that is a
        different experiment, not a continuation. Inference-only ids are excluded by
        the registry's own ``training_feature_fingerprint`` split.
        """
        stored = progress.training_feature_fingerprint
        if not stored:
            return
        current_identity = self._feature_identity()
        current = current_identity.get("training_feature_fingerprint")
        if not current or str(current) == str(stored):
            return
        stored_ids = list(progress.training_features or [])
        current_ids = list(current_identity.get("training_features") or [])
        changed = sorted(set(stored_ids) ^ set(current_ids))
        detail = f"; differing flags: {changed}" if changed else ""
        message = (
            "cannot resume: the training-feature fingerprint differs from the checkpoint's "
            f"(checkpoint {stored}, this run {current}){detail}. Training features change the "
            "supervised data distribution, so this is a different experiment, not a continuation "
            "of the checkpointed one. Resume with the config the checkpoint was trained under, or "
            "re-run from scratch."
        )
        if not effort:
            raise LifeMateError(message)
        # Lasting downgrade: the exactness assignment at the end of resume() composes
        # this with the other requirements (a bare ``exact_resume = False`` here
        # would be overwritten).
        print("resume: BEST EFFORT (not exact)")
        self._exact_downgraded = True

    def _require_optimizer_state(self, path: Path, step: int, effort: bool) -> None:
        """Refuse a step dir whose ``optimizer_state.pt`` is missing (weight-only).

        ``training_state.json`` is written LAST, but a crash (or OOM/disconnect)
        between an older writer's two writes — or a hand-copied directory, or the
        notebook's adapter-only export — can leave a state file that claims
        ``exact_resume: true`` beside no optimizer tensors. Accepting it would
        rebuild AdamW from zero moments while the report still said "exact", which
        is exactly the disagreement this schema exists to prevent.

        GATED on ``self.optimizer is not None or any weight files exist``: a
        ``save_fn``-injected test double has neither, and its checkpoints carry no
        optimizer file by construction. Those keep resuming as before.
        """
        if (path / "optimizer_state.pt").is_file():
            if self._optimizer_state_expected():
                self._optimizer_state_present = True
            return
        has_weights = bool(
            list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
        )
        if self.optimizer is None and not has_weights:
            # Test-double / stdlib checkpoint: no optimizer was ever persisted, so
            # there is nothing to restore and nothing to lose. Not exact, though:
            # a real continuation would still need those moments.
            self._optimizer_state_present = False
            return
        if effort:
            print("resume: BEST EFFORT (not exact)")
            self._optimizer_state_present = False
            return
        raise LifeMateError(
            f"cannot resume exactly: {path} has no optimizer_state.pt (step {step}), so the "
            "AdamW moments, the LR-scheduler position and any GradScaler state cannot be "
            "restored. Resuming would restart the optimizer from zero moments — loading the "
            "weights alone is adapter-only loading, NOT a resume. Fix: pass best_effort=True / "
            "--best-effort-resume to continue approximately (that run prints 'resume: BEST "
            "EFFORT (not exact)'), resume an older checkpoint of this run that carries "
            "optimizer_state.pt, or re-run from scratch."
        )

    def _check_manifest_compat(self, path: Path) -> None:
        """Cross-check ``lifemate_meta.json`` (Phase 3.9 manifest provenance).

        The trainer's own ``_check_world`` reads ``training_state.json``; the
        manifest is the surface a loader/export tool reads, so both are verified.
        Only fields the checkpoint actually recorded are compared.
        """
        try:
            from lifemate.models.checkpoint import check_checkpoint_compat  # noqa: PLC0415 - lazy
        except ImportError:  # pragma: no cover - models slice absent
            return
        world = self._world()
        # ``config_hash`` / ``dataset_manifest_hash`` are checked by ``resume()``
        # itself with richer messages, and ``dataset_order_hash`` needs the
        # tokenised examples, which only ``run()`` builds; do not double-report.
        check_checkpoint_compat(
            path,
            model_profile=world["model_profile"],
            tuning_mode=world["tuning_mode"],
            quantization=world["quantization"],
            precision=world["precision"],
            adapter_rank=world["adapter_rank"],
            allow_horizon_change=self.extend_horizon,
        )

    def _check_world(self, path: Path, state: Mapping[str, Any], progress: TrainingProgress) -> None:
        """Refuse a checkpoint from a different model world (profile/mode/quant/dtype)."""
        recorded = dict(progress.world or state.get("world") or {})
        if not recorded:
            return
        current = self._world()
        for key, label, fix in (
            ("model_profile", "model profile",
             "pass the --model this checkpoint was trained with"),
            ("tuning_mode", "tuning mode",
             "pass --tuning to match the checkpoint (lora into qlora is not a continuation)"),
            ("quantization", "quantization",
             "match training.quantization from the checkpoint (lora and qlora are different maths)"),
            ("precision", "precision", "match training.precision from the checkpoint"),
            ("adapter_rank", "adapter rank",
             "match lora.r from the checkpoint (the adapter shapes differ)"),
        ):
            old, new = recorded.get(key), current.get(key)
            if old in (None, "") or new in (None, "") or str(old) == str(new):
                continue
            raise LifeMateError(
                f"cannot resume: checkpoint {label} differs from this run's "
                f"(checkpoint {key}={old!r} != this run {new!r}) at {path}. "
                f"The optimizer state and weights would not describe the same model; {fix}."
            )

    def _restore_rng(
        self, state: Mapping[str, Any], progress: TrainingProgress, *, exact: bool
    ) -> None:
        """Restore python/torch/cuda/numpy RNG exactly, or raise on request.

        Never ``except: pass``: a silently unrestored stream is precisely the
        non-reproducibility this schema exists to remove.
        """
        def fail(what: str, exc: Exception) -> None:
            if exact:
                raise LifeMateError(
                    f"cannot resume exactly: restoring the {what} RNG state failed ({exc}). "
                    "The continuation would not replay the interrupted run. Pass best_effort=True "
                    "to accept an approximate continuation, or fix the environment."
                ) from exc

        py_state = progress.python_rng_state or state.get("python_rng_state")
        if py_state:
            try:
                random.setstate(
                    (3, tuple(int(v) for v in py_state), progress.python_rng_gauss_next)
                )
            except Exception as exc:  # pragma: no cover - corrupt state
                fail("python", exc)
        torch_rng = progress.torch_rng_state or state.get("torch_rng_state")
        cuda_rng = progress.cuda_rng_state or state.get("cuda_rng_state")
        if torch_rng is not None or cuda_rng is not None:
            try:
                import torch  # noqa: PLC0415

                if torch_rng is not None:
                    torch.set_rng_state(torch.tensor(list(torch_rng), dtype=torch.uint8))
                if cuda_rng and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [torch.tensor(list(s), dtype=torch.uint8) for s in cuda_rng]
                    )
            except Exception as exc:  # pragma: no cover - torch absent
                fail("torch/cuda", exc)
        numpy_state = progress.numpy_rng_state
        if numpy_state:
            try:
                _restore_numpy_rng(numpy_state)
            except Exception as exc:  # pragma: no cover - numpy absent
                fail("numpy", exc)

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
                self._scheduler_state_loaded = True
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
