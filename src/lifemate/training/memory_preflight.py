"""Pre-flight memory estimate: would this run fit on the available GPU?

Pure arithmetic over a parameter count. It is a **heuristic projection, not a
measurement**: no allocation, no CUDA probing beyond asking total/free VRAM.

Byte accounting (decimal GB, ``1e9`` bytes — call out GB, not GiB):

===============  ==============================================  =========================
term             full fine-tuning                                LoRA
===============  ==============================================  =========================
weights          ``params * 4`` (fp32 master; AdamW writes the  ``params * bpp`` (bf16 base)
                 update into the parameter tensor, so a bf16
                 master would quantise a 5e-6 step away)
gradients        ``params * 4``                                  adapter only (32,464,896 params)
optimizer state  ``params * 8`` (AdamW fp32 m,v + fp32 master)   adapter only
activations      ``layers * seq * batch * hidden * k`` heuristic  same
===============  ==============================================  =========================

``k`` (the activation constant) is documented as 10 for full fine-tuning
without gradient checkpointing and 2 with it; gradient checkpointing trades
recompute time for a ~5x smaller activation footprint. The estimate is
deliberately conservative because underestimating VRAM loses a whole training
run.
"""

from __future__ import annotations

from typing import Any, Mapping

from lifemate.errors import LifeMateError

__all__ = ["format_report", "run_memory_preflight"]

#: Bytes per float parameter.
BYTES_PER_PARAM = 2  # bf16/fp16
#: AdamW: fp32 exp_avg + fp32 exp_avg_sq (+ fp32 master copy when the model is fp16/bf16).
OPTIMIZER_BYTES_PER_PARAM = 8
#: Activation constant without / with gradient checkpointing.
ACTIVATION_K_PLAIN = 10
ACTIVATION_K_CHECKPOINTED = 2
#: Share of a GPU we are willing to fill before calling it "tight".
TIGHT_FRACTION = 0.9
#: Defaults used when the caller supplies neither a model nor a param count.
DEFAULT_PARAMS = 4_000_000_000
#: Released ``Qwen/Qwen3.5-4B`` ``text_config`` shape (32 layers, hidden 2560).
DEFAULT_LAYERS = 32
DEFAULT_HIDDEN = 2560
#: Sequence budget assumed when neither ``seq_len`` nor a config supplies one.
DEFAULT_SEQ_LEN = 8192

def _gpu_info() -> dict[str, Any]:
    """Total/free VRAM when torch + CUDA exist; an honest ``unavailable`` else."""
    try:
        import torch  # noqa: PLC0415 - lazy on purpose
    except Exception as exc:
        return {"gpu": "unavailable", "reason": f"torch not importable: {exc}"}
    try:
        if not torch.cuda.is_available():
            return {"gpu": "unavailable", "reason": "torch.cuda.is_available() is False"}
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        return {
            "gpu": "available",
            "device": index,
            "name": str(props.name),
            "total_bytes": int(total),
            "free_bytes": int(free),
            "total_gb": round(total / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
        }
    except Exception as exc:  # pragma: no cover - driver/runtime quirks
        return {"gpu": "unavailable", "reason": f"CUDA query failed: {exc}"}

def _trainable_from_model(model: Any) -> int:
    """Trainable parameters of an already-built model (0 when unknown/absent).

    The LoRA branch used to invent ``0.5 %`` of the base as the adapter count;
    that placeholder was then reported as ``trainable_params`` and got quoted as
    a measured figure. Whenever a real (or meta-device) model is available, count
    the tensors that actually carry ``requires_grad`` instead.
    """
    total = 0
    try:
        for param in model.parameters():
            if getattr(param, "requires_grad", False):
                total += int(param.numel())
    except Exception:  # pragma: no cover - duck-typed model
        return 0
    return total


def _params_from_model(model: Any) -> int:
    total = 0
    try:
        for param in model.parameters():
            total += int(param.numel())
    except Exception:  # pragma: no cover - duck-typed model
        return 0
    return total

def _config_values(cfg: Any, model: Any = None) -> tuple[str, int, int, int, int, bool, int]:
    """``(tuning_mode, batch, accum, layers, hidden, checkpointing, bytes_per_param)``.

    ``layers``/``hidden`` come from ``model.config`` (``num_hidden_layers`` /
    ``hidden_size``) when a model is passed; otherwise the released
    Qwen3.5-4B shape is assumed — **32** layers, hidden 2560 (from the
    checkpoint's ``text_config``; the previous default of 36 overestimated the
    activation term by 12.5 %).
    """
    training = getattr(cfg, "training", cfg) if cfg is not None else None
    mode = str(getattr(training, "tuning_mode", "full") or "full")
    batch = int(getattr(training, "per_device_train_batch_size", 1) or 1)
    accum = int(getattr(training, "gradient_accumulation_steps", 1) or 1)
    checkpointing = bool(getattr(training, "gradient_checkpointing", False))
    config = getattr(model, "config", None) if model is not None else None
    layers = int(getattr(config, "num_hidden_layers", 0) or 0) or DEFAULT_LAYERS
    hidden = int(getattr(config, "hidden_size", 0) or 0) or DEFAULT_HIDDEN
    precision = str(getattr(training, "precision", "bfloat16") or "bfloat16").lower()
    bpp = 4 if precision in ("float32", "fp32") else BYTES_PER_PARAM
    return mode, batch, accum, layers, hidden, checkpointing, bpp
def run_memory_preflight(
    model: Any = None,
    param_count: int | None = None,
    cfg: Any = None,
    seq_len: int | None = None,
    gpu_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate VRAM for a full or LoRA run on ``param_count`` parameters.

    All returned numbers are estimates in decimal GB; the ``heuristic`` key
    states which formula produced the activation term. ``verdict`` is
    ``fits``/``tight``/``does_not_fit`` when VRAM is known, else ``unknown``.

    ``seq_len=None`` inherits ``training.max_seq_length`` from ``cfg`` (falling
    back to :data:`DEFAULT_SEQ_LEN`), so the estimate always matches the budget
    the run will actually use instead of a stale literal.
    """
    if seq_len is None:
        training = getattr(cfg, "training", cfg) if cfg is not None else None
        seq_len = int(getattr(training, "max_seq_length", 0) or 0) or DEFAULT_SEQ_LEN
    mode, batch, accum, layers, hidden, checkpointing, bpp = _config_values(cfg, model)
    params = int(param_count or 0)
    if not params and model is not None:
        params = _params_from_model(model)
    params = params or DEFAULT_PARAMS
    seq = int(seq_len or 0)
    if seq <= 0:
        raise LifeMateError(f"seq_len must be positive, got {seq_len!r}")

    weights = params * bpp
    # Full fine-tuning keeps fp32 master weights (``full_finetuning.master_dtype``):
    # AdamW's update lands in the parameter tensor, so a bf16 master quantises a
    # 5e-6 step away. Storage/grad/optimizer terms are therefore 4 bytes each,
    # while the activation term still runs at the compute dtype (bf16 + autocast).
    master_bpp = max(bpp, 4) if mode != "lora" else bpp
    master_weights = params * master_bpp
    if mode == "lora":
        # Prefer the adapter tensors that actually exist (``configure_lora`` was
        # run, or a meta-device model was built); fall back to the 0.5 % guess
        # and SAY SO, so a placeholder is never quoted as a measured count.
        adapter_params = _trainable_from_model(model) if model is not None else 0
        trainable_source = "measured" if adapter_params else "placeholder_0.5pct"
        if not adapter_params:
            adapter_params = max(1, int(params * 0.005))
        grads = adapter_params * bpp
        optimizer = adapter_params * OPTIMIZER_BYTES_PER_PARAM
        trainable_params = adapter_params
    else:
        grads = params * master_bpp
        optimizer = params * OPTIMIZER_BYTES_PER_PARAM
        trainable_params = params
        trainable_source = "all_parameters"

    k = ACTIVATION_K_CHECKPOINTED if checkpointing else ACTIVATION_K_PLAIN
    activations = int(layers * seq * batch * hidden * k * bpp)
    total = master_weights + grads + optimizer + activations

    gpu = dict(gpu_info) if gpu_info is not None else _gpu_info()
    available_bytes = gpu.get("free_bytes") or gpu.get("total_bytes") or 0
    if available_bytes:
        ratio = total / float(available_bytes)
        verdict = "fits" if ratio <= TIGHT_FRACTION else ("tight" if ratio <= 1.0 else "does_not_fit")
    else:
        verdict = "unknown"

    return {
        "tuning_mode": mode,
        "param_count": params,
        "trainable_params": trainable_params,
        "trainable_source": trainable_source,
        "seq_len": seq,
        "microbatch": batch,
        "gradient_accumulation_steps": accum,
        "gradient_checkpointing": checkpointing,
        "weights_gb": round(master_weights / 1e9, 2),
        "weight_bytes_per_param": master_bpp,
        "gradients_gb": round(grads / 1e9, 2),
        "optimizer_gb": round(optimizer / 1e9, 2),
        "activations_gb": round(activations / 1e9, 2),
        "total_gb": round(total / 1e9, 2),
        "checkpoint_size_gb": round((master_weights + grads) / 1e9, 2),
        "gpu": gpu,
        "available_gb": round(available_bytes / 1e9, 2) if available_bytes else None,
        "verdict": verdict,
        "heuristic": (
            f"activations = layers({layers}) * seq({seq}) * microbatch({batch}) * "
            f"hidden({hidden}) * k({k}) * {bpp} bytes; k=2 with gradient "
            "checkpointing else 10. Point estimate, not a measurement."
        ),
        "suggestions": _suggestions(verdict, mode, batch, seq, checkpointing),
    }

def _suggestions(verdict: str, mode: str, batch: int, seq: int, checkpointing: bool) -> list[str]:
    """Actionable next steps ordered by effort."""
    if verdict == "fits":
        return ["estimate fits the available VRAM; still watch the first few steps' peak memory"]
    tips: list[str] = []
    if not checkpointing:
        tips.append("enable training.gradient_checkpointing (trades ~20% speed for a ~5x smaller activation term)")
    if batch > 1:
        tips.append(f"reduce training.per_device_train_batch_size ({batch} -> 1) and raise gradient_accumulation_steps")
    if seq > 2048:
        tips.append(f"reduce training.max_seq_length ({seq} -> 2048 or 1024)")
    if mode == "full":
        tips.append("switch to LoRA (--tuning lora): only adapter gradients/optimizer states are stored")
        tips.append("use optimizer offload / DeepSpeed ZeRO-3 to shard optimizer state across GPUs")
    tips.append("measure with torch.cuda.max_memory_allocated() during the first optimizer step")
    return tips

def format_report(report: Mapping[str, Any]) -> str:
    """Human-readable multi-line rendering of :func:`run_memory_preflight`."""
    gpu = report.get("gpu") or {}
    if gpu.get("gpu") == "available":
        gpu_line = f"{gpu.get('name')} — free {gpu.get('free_gb')} GB / total {gpu.get('total_gb')} GB"
    else:
        gpu_line = f"unavailable ({gpu.get('reason', 'no CUDA')})"
    lines = [
        "Memory pre-flight (estimate, not a measurement)",
        f"  tuning mode        : {report.get('tuning_mode')}",
        f"  parameters         : {int(report.get('param_count') or 0):,} "
        f"(trainable {int(report.get('trainable_params') or 0):,}; "
        f"count from {report.get('trainable_source') or 'unknown'})",
        f"  sequence length    : {report.get('seq_len')} (microbatch {report.get('microbatch')}, "
        f"accum {report.get('gradient_accumulation_steps')})",
        f"  weights            : {report.get('weights_gb')} GB",
        f"  gradients          : {report.get('gradients_gb')} GB",
        f"  optimizer state    : {report.get('optimizer_gb')} GB",
        f"  activations        : {report.get('activations_gb')} GB (heuristic)",
        f"  TOTAL              : {report.get('total_gb')} GB",
        f"  checkpoint size    : {report.get('checkpoint_size_gb')} GB",
        f"  GPU                : {gpu_line}",
        f"  verdict            : {report.get('verdict')}",
        f"  assumptions        : {report.get('heuristic')}",
    ]
    suggestions = list(report.get("suggestions") or [])
    if suggestions:
        lines.append("  suggestions        :")
        lines.extend(f"    - {tip}" for tip in suggestions)
    return "\n".join(lines)
