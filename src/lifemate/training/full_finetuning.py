"""Full-parameter fine-tuning configuration.

Full fine-tuning means *every* parameter is trainable: no frozen blocks, no
hidden exceptions. The contract is asserted, not assumed — a base model that
arrives half-frozen (or a ``requires_grad`` typo) fails loudly here instead of
training a quietly broken adapter-shaped run.

Everything heavy (torch) is imported inside the functions.
"""

from __future__ import annotations

from typing import Any, Mapping

from lifemate.errors import ConfigError, LifeMateError
from lifemate.models.loader import count_parameters
from lifemate.training.trainer import DEFAULT_MAX_SEQ_LENGTH

__all__ = [
    "SUPPORTED_SCHEDULERS",
    "build_optimizer_and_scheduler",
    "configure_full_finetuning",
    "master_dtype",
    "resolve_dtype",
]

#: Schedulers accepted by :func:`build_optimizer_and_scheduler` (maps 1:1 to
#: ``torch.optim.lr_scheduler.get_scheduler`` names we rely on).
SUPPORTED_SCHEDULERS = ("cosine", "linear", "constant", "constant_with_warmup")

def resolve_dtype(torch_module: Any, name: Any, device: str = "cuda") -> Any:
    """Map a config dtype string to a ``torch.dtype`` (bf16 → fp32 on CPU).

    bfloat16 on CPU is slow/unsupported for many kernels, so a CPU run silently
    becoming fp32 is a documented, deliberate downgrade — never a silent switch
    between bf16 and fp16, which changes numerics.
    """
    key = str(name or "bfloat16").lower()
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if key not in mapping:
        raise ConfigError(f"unsupported dtype {name!r}; expected one of {sorted(mapping)}")
    dtype = mapping[key]
    if dtype is torch_module.bfloat16 and str(device).startswith("cpu"):
        return torch_module.float32
    return dtype

def master_dtype(torch_module: Any, name: Any, device: str = "cuda") -> Any:
    """Dtype the *trainable parameters* must be stored in.

    AdamW applies ``lr * m / (sqrt(v) + eps)`` straight into the parameter
    tensor, so a bf16 parameter quantises every update to a bf16 ulp (~0.4% of
    its magnitude). Measured at ``lr = 5e-6`` (``configs/training_full.yaml``):
    after 20 steps only **7% of bf16 tensors had moved at all** in a
    ``N(0, 0.02)``-scale weight, versus 100% in fp32. Full fine-tuning therefore
    keeps fp32 master weights and gets its bf16 speed from
    ``torch.autocast`` (the trainer's ``_autocast``), not from a bf16 parameter
    dtype. LoRA is unaffected: adapters start at zero, so their magnitudes stay
    large enough relative to the ~3e-4 / 2e-4 updates they take.
    """
    dtype = resolve_dtype(torch_module, name, device)
    if dtype in (torch_module.bfloat16, torch_module.float16):
        return torch_module.float32
    return dtype

def _training_cfg(cfg: Any) -> Any:
    return getattr(cfg, "training", cfg)

def configure_full_finetuning(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    *,
    freeze_nothing: bool = True,
    multimodal_policy: str = "everything",
) -> dict:
    """Enable training on every parameter and return an auditable report.

    Dtype casting is applied BEFORE unfreezing so that parameter references
    and requires_grad flags are established after any cast/device moves.

    Multimodal policy (Qwen3.5 vision/audio towers):
    - 'everything' (default): full FT unfreezes ALL modules incl. vision/audio towers.
    - 'lm_only': freezes visual/audio towers explicitly; only LM trains.
    Policy is explicit, printed at startup, and reflected in trainable counts.
    """
    import torch  # noqa: PLC0415 - lazy on purpose

    training = _training_cfg(cfg)
    if model is None:
        raise LifeMateError("configure_full_finetuning(model=None) is not allowed")

    device = "cuda" if _is_cuda(model) else "cpu"
    compute_dtype = resolve_dtype(torch, getattr(training, "precision", "bfloat16"), device)
    target_dtype = master_dtype(torch, getattr(training, "precision", "bfloat16"), device)
    dtype_before = next((str(p.dtype) for p in model.parameters()), "unknown")
    cast_applied = False
    if dtype_before != str(target_dtype) and hasattr(model, "to"):
        model = model.to(dtype=target_dtype)
        cast_applied = True

    try:
        for param in model.parameters():
            param.requires_grad_(True)
    except AttributeError as exc:  # pragma: no cover - exotic model wrappers
        raise LifeMateError(f"model does not expose torch-style parameters(): {exc}") from exc

    # Multimodal towers policy check
    policy = str(getattr(training, "multimodal_policy", multimodal_policy) or "everything").lower()
    tower_suffixes = ("visual", "vision_tower", "vision_model", "audio_tower", "audio_model", "image_tower")
    detected_towers: list[str] = []
    if hasattr(model, "named_modules"):
        for mod_name, mod in model.named_modules():
            if any(mod_name.endswith(sfx) or mod_name == sfx for sfx in tower_suffixes):
                detected_towers.append(mod_name)
                if policy == "lm_only":
                    for p in mod.parameters():
                        p.requires_grad_(False)

    counts = count_parameters(model)
    if counts["total"] <= 0:
        raise LifeMateError("model reports zero parameters; refusing to start full fine-tuning")
    if freeze_nothing and policy == "everything" and counts["trainable"] != counts["total"]:
        frozen = counts["total"] - counts["trainable"]
        raise LifeMateError(
            f"full fine-tuning requested but {frozen:,} of {counts['total']:,} parameters are frozen "
            "(freeze_nothing=True requires trainable == total)"
        )

    checkpointing = False
    if bool(getattr(training, "gradient_checkpointing", False)):
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable):
            try:
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:  # pragma: no cover - older signature
                enable()
            checkpointing = True
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except (AttributeError, ValueError):  # pragma: no cover
                pass

    return {
        "tuning_mode": "full",
        "freeze_nothing": bool(freeze_nothing),
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "frozen_params": counts["frozen"],
        "trainable_ratio": round(counts["trainable"] / counts["total"], 4) if counts["total"] else 0.0,
        "dtype_before": dtype_before,
        "dtype_target": str(target_dtype),
        "compute_dtype": str(compute_dtype),
        "dtype_cast_applied": cast_applied,
        "gradient_checkpointing": checkpointing,
        "multimodal_policy": policy,
        "multimodal_towers": detected_towers,
        "learning_rate": float(getattr(training, "learning_rate", 0.0)),
        "weight_decay": float(getattr(training, "weight_decay", 0.0)),
        "optimizer": str(getattr(training, "optimizer", "adamw_torch")),
        "max_seq_length": int(getattr(training, "max_seq_length", DEFAULT_MAX_SEQ_LENGTH)),
    }

def _is_cuda(model: Any) -> bool:
    try:
        return next(model.parameters()).is_cuda
    except Exception:  # pragma: no cover
        return False

def build_scheduler(
    optimizer: Any, cfg: Any, num_training_steps: int, *, last_epoch: int | None = None
) -> Any:
    """Warmup-then-schedule ``LambdaLR`` over an existing optimizer.

    Split out of :func:`build_optimizer_and_scheduler` so a resumed run can
    rebuild the schedule for its *real* horizon (a scheduler restored from a
    checkpoint was built for the step count the interrupted run had) without
    rebuilding the optimizer and losing the restored AdamW moments.

    ``last_epoch`` positions the schedule at an already-completed step, applying
    that step's LR immediately; ``None`` starts a fresh schedule at step 0.
    """
    import torch  # noqa: PLC0415 - lazy on purpose

    training = _training_cfg(cfg)
    scheduler_name = str(getattr(training, "lr_scheduler_type", "cosine")).lower()
    warmup_ratio = float(getattr(training, "warmup_ratio", 0.0))
    if scheduler_name not in SUPPORTED_SCHEDULERS:
        raise ConfigError(f"unsupported lr_scheduler_type {scheduler_name!r}; expected {SUPPORTED_SCHEDULERS}")
    if num_training_steps < 0:
        raise ConfigError("num_training_steps must be >= 0")
    steps = max(1, int(num_training_steps))
    warmup_steps = int(round(warmup_ratio * steps))
    if scheduler_name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    elif scheduler_name == "constant_with_warmup":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0,
        )
    elif scheduler_name == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _warmup_then_linear(step, warmup_steps, steps)
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _warmup_then_cosine(step, warmup_steps, steps)
        )
    if last_epoch is not None:
        scheduler.last_epoch = int(last_epoch)
        # Apply the LR the schedule assigns to that position. Calling step() here
        # would advance the counter and trip torch's "step before optimizer.step"
        # warning on a freshly restored optimizer.
        factor = scheduler.lr_lambdas[0](scheduler.last_epoch)
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr * factor
    return scheduler


def build_optimizer_and_scheduler(
    model: Any,
    cfg: Any,
    num_training_steps: int,
    *,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
):
    """AdamW with a no-decay group for 1-D params, plus a warmup scheduler.

    * weight decay applies only to matrices/multi-dim tensors; biases, norms and
      embeddings of dim ≤ 1 stay undecayed (standard transformer practice);
    * schedule: linear warmup for ``warmup_ratio`` of the run, then cosine
      (default), linear or constant — chosen by ``training.lr_scheduler_type``;
    * ``num_training_steps`` counts optimizer steps, not micro-batches.
    """
    import torch  # noqa: PLC0415 - lazy on purpose

    training = _training_cfg(cfg)
    lr = float(learning_rate if learning_rate is not None else getattr(training, "learning_rate", 5e-6))
    decay = float(weight_decay if weight_decay is not None else getattr(training, "weight_decay", 0.01))
    if int(num_training_steps) < 0:
        raise ConfigError("num_training_steps must be >= 0")
    steps = max(1, int(num_training_steps))

    decay_params: list[Any] = []
    no_decay_params: list[Any] = []
    embedding_param_ids: set[int] = set()
    if hasattr(model, "modules"):
        try:
            for m in model.modules():
                if isinstance(m, torch.nn.Embedding):
                    for p in m.parameters():
                        embedding_param_ids.add(id(p))
        except Exception:  # pragma: no cover
            pass

    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        # Bias, LayerNorm/RMSNorm, and embeddings stay undecayed (standard transformer practice)
        is_no_decay = (
            param.dim() <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "embed" in name.lower()
            or "wte" in name.lower()
            or "shared" in name.lower()
            or id(param) in embedding_param_ids
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [{"params": decay_params, "weight_decay": decay}]
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not any(group["params"] for group in groups):
        raise LifeMateError("optimizer got no trainable parameters; check requires_grad")

    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    return optimizer, build_scheduler(optimizer, cfg, steps)

def _warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)

def _warmup_then_linear(step: int, warmup_steps: int, total_steps: int) -> float:
    factor = _warmup_factor(step, warmup_steps)
    if step < warmup_steps:
        return factor
    span = max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - (step - warmup_steps) / span)

def _warmup_then_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    import math  # noqa: PLC0415 - tiny, keeps module import light

    factor = _warmup_factor(step, warmup_steps)
    if step < warmup_steps:
        return factor
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
