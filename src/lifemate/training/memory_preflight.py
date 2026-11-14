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
gradients        ``params * 4``                                  adapter only (measured for MiniCPM5-2B: 25,116,672)
optimizer state  ``params * 8`` (AdamW fp32 m,v + fp32 master)   adapter only
activations      ``layers * seq * batch * hidden * k`` heuristic  same
logits + CE      ``chunk * batch * vocab * 4`` (fp32 upcast)      same
                 ``chunk = min(loss_chunk_tokens, seq)`` when chunked,
                 else ``seq``; the unchunked cost is reported as
                 ``logits_unchunked_gb`` for contrast
===============  ==============================================  =========================
For Qwen3.5-4B the LoRA adapter count is 32,464,896 (12 suffixes, hybrid
linear-attention set); the preflight reports whatever the loaded model actually
carries and labels the value ``measured`` or ``placeholder_0.5pct``.

``k`` (the activation constant) is documented as 10 for full fine-tuning
without gradient checkpointing and 2 with it; gradient checkpointing trades
recompute time for a ~5x smaller activation footprint. The estimate is
deliberately conservative because underestimating VRAM loses a whole training
run.
"""

from __future__ import annotations

import json
from pathlib import Path
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
#: Fallbacks used when neither a model, a local config.json nor a param count is
#: available. They come from the DEFAULT profile (MiniCPM5-2B, measured from its
#: released config.json + safetensors index). They used to be Qwen3.5-4B's shape,
#: which under-estimated nothing and over-estimated everything for MiniCPM.
DEFAULT_PARAMS = 2_516_756_480
DEFAULT_LAYERS = 42
DEFAULT_HIDDEN = 2048
#: Qwen3.5-4B shape, kept for the explicit Qwen profile.
QWEN_PARAMS = 4_000_000_000
QWEN_LAYERS = 32
QWEN_HIDDEN = 2560
#: nf4 storage ≈ 0.5 byte/param (4-bit + per-block absmax scale); the frozen base
#: is not kept in 16-bit alongside it.
NF4_BYTES_PER_PARAM = 0.5
#: Bytes per logit in the cross-entropy step. transformers' ``ForCausalLMLoss``
#: upcasts the logits tensor it is handed to fp32 before the fused cross_entropy.
#: The SHIPPED default is chunked CE (``training.loss_chunk_tokens: 1024``), so
#: the tensor that gets upcast is a ``[chunk, vocab]`` slice, not the whole
#: ``[seq, vocab]``: 1024 * 130560 * 4 = 0.53 GB, versus 4.28 GB unchunked.
#: MEASURED on a 16.57 GB RTX 5070 Ti at seq 8192 / vocab 130560: one
#: TRAIN-ONLY step peaks at 14.37 GB allocated / 15.17 GB reserved (batch padded
#: to exactly 8192 tokens, no ``PYTORCH_CUDA_ALLOC_CONF``); the unchunked
#: whole-vocab upcast added ~4.28 GB and OOM'd the same class of card.
LOGIT_BYTES = 4
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

def _model_shape(cfg: Any, model: Any = None) -> tuple[int, int, int, int, str]:
    """``(params, layers, hidden, vocab, source)`` for the configured model.

    Resolution order, most trustworthy first:

    1. a loaded ``model`` (``config.text_config`` then ``config``
       ``num_hidden_layers``/``hidden_size`` + real parameter count);
    2. ``config.json`` of a LOCAL model directory named by ``cfg.model.name``
       (stdlib JSON read, offline, no network);
    3. the model profile's verified fallbacks (MiniCPM5-2B: 2,516,756,480 params
       / 42 layers / 2048 hidden — measured; Qwen3.5-4B: 4e9 / 32 / 2560).

    ``source`` is stamped into the report so an estimate states where its shape
    came from instead of implying a measurement.
    """
    from lifemate.models.profiles import resolve_profile  # noqa: PLC0415 - lazy

    model_cfg = getattr(cfg, "model", None) if cfg is not None else None
    model_name = str(getattr(model_cfg, "name", "") or "")
    profile = resolve_profile(model_name)
    vocab = 0

    if model is not None:
        config = getattr(model, "config", None)
        layers, hidden, vocab = _shape_triple(config)
        params = _params_from_model(model)
        # ``_shape_triple`` returns (0, 0, 0) when neither level answers, so this
        # guard IS the assertion that both resolved values are positive.
        if layers > 0 and hidden > 0 and params > 0:
            return params, layers, hidden, vocab, "model (loaded config + parameter count)"

    payload = _read_local_config_json(model_name)
    if payload:
        layers, hidden, vocab = _shape_triple(payload)
        params = _params_from_config_json(payload)
        if layers > 0 and hidden > 0 and params > 0:
            return params, layers, hidden, vocab, f"local config.json ({model_name})"

    return (
        profile.preflight_params or DEFAULT_PARAMS,
        profile.preflight_layers or DEFAULT_LAYERS,
        profile.preflight_hidden or DEFAULT_HIDDEN,
        profile.preflight_vocab,
        f"profile fallback {profile.name} ({profile.preflight_source}); "
        "no config reported both num_hidden_layers > 0 and hidden_size > 0 "
        "(checked text_config then the top level)",
    )


def _shape_triple(config: Any) -> tuple[int, int, int]:
    """``(layers, hidden, vocab)`` from a config object or dict, nested first.

    Qwen3.5-4B keeps ``num_hidden_layers``/``hidden_size`` under
    ``config.text_config`` (32 / 2560) and exposes NOTHING at the top level, so
    a top-level-only read silently returned 0 and the caller fell back to the
    profile. Descend into ``text_config`` first, then the top level.
    """
    for source in (_nested(config, "text_config"), config):
        layers, hidden, vocab = _ints(source, "num_hidden_layers", "hidden_size", "vocab_size")
        if layers > 0 and hidden > 0:
            return layers, hidden, vocab
    return 0, 0, 0


def _nested(config: Any, key: str) -> Any:
    """``config[key]`` for a dict and ``config.key`` for an object (``None`` else)."""
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _ints(source: Any, *keys: str) -> tuple[int, ...]:
    """``int()`` of each key in ``source``; missing/non-numeric values become 0."""
    values: list[int] = []
    for key in keys:
        try:
            values.append(int(_nested(source, key) or 0))
        except (TypeError, ValueError):
            values.append(0)
    return tuple(values)


def _read_local_config_json(model_name: str) -> dict[str, Any]:
    """``config.json`` for a local model directory (``{}`` when unavailable)."""
    if not model_name:
        return {}
    path = Path(model_name) / "config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _params_from_config_json(payload: Mapping[str, Any]) -> int:
    """Parameter count from a transformer ``config.json`` (Llama-style block).

    Exact for a plain decoder-only stack: ``vocab*hidden`` embeddings (plus an
    untied ``lm_head``) + per layer attention/MLP projections + RMSNorm weights
    (2 per layer and one final). Verified against the MiniCPM5-2B safetensors
    index: 2,516,756,480 exactly. Returns 0 when a field is missing, so the
    caller falls back to a profile's verified number instead of inventing one.
    """
    try:
        hidden = int(payload["hidden_size"])
        layers = int(payload["num_hidden_layers"])
        vocab = int(payload["vocab_size"])
        inter = int(payload["intermediate_size"])
        heads = int(payload["num_attention_heads"])
        kv_heads = int(payload.get("num_key_value_heads") or heads)
    except (KeyError, TypeError, ValueError):
        return 0
    head_dim = int(payload.get("head_dim") or (hidden // heads if heads else 0))
    q = hidden * (heads * head_dim)
    k = hidden * (kv_heads * head_dim)
    v = k
    o = (heads * head_dim) * hidden
    mlp = hidden * inter * 3  # gate + up + down
    norms = hidden * 2  # input_layernorm + post_attention_layernorm
    per_layer = q + k + v + o + mlp + norms
    embed = vocab * hidden
    total = layers * per_layer + embed + hidden  # + the final norm
    if not payload.get("tie_word_embeddings", True):
        total += embed
    return int(total)


def _config_values(cfg: Any, model: Any = None) -> tuple[str, int, int, bool, int, str]:
    """``(tuning_mode, batch, accum, checkpointing, bytes_per_param, quantization)``."""
    training = getattr(cfg, "training", cfg) if cfg is not None else None
    mode = str(getattr(training, "tuning_mode", "full") or "full")
    batch = int(getattr(training, "per_device_train_batch_size", 1) or 1)
    accum = int(getattr(training, "gradient_accumulation_steps", 1) or 1)
    checkpointing = bool(getattr(training, "gradient_checkpointing", False))
    precision = str(getattr(training, "precision", "bfloat16") or "bfloat16").lower()
    bpp = 4 if precision in ("float32", "fp32") else BYTES_PER_PARAM
    quantization = str(getattr(training, "quantization", "none") or "none").lower()
    return mode, batch, accum, checkpointing, bpp, quantization
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
    training = getattr(cfg, "training", cfg) if cfg is not None else None
    if seq_len is None:
        seq_len = int(getattr(training, "max_seq_length", 0) or 0) or DEFAULT_SEQ_LEN
    mode, batch, accum, checkpointing, bpp, quantization = _config_values(cfg, model)
    adapter_mode = mode in ("lora", "qlora")
    quantized = quantization != "none"
    shape_params, layers, hidden, vocab, shape_source = _model_shape(cfg, model)
    params = int(param_count or 0) or shape_params
    seq = int(seq_len or 0)
    if seq <= 0:
        raise LifeMateError(f"seq_len must be positive, got {seq_len!r}")

    # Every term below is reported separately and in decimal GB. They are
    # ESTIMATES: the only measured figure in this report is ``trainable_params``
    # when a real (or meta-device) adapter is present.
    #
    # Full fine-tuning keeps fp32 master weights (``full_finetuning.master_dtype``):
    # AdamW's update lands in the parameter tensor, so a bf16 master quantises a
    # 5e-6 step away. Storage/grad/optimizer terms are therefore 4 bytes each,
    # while the activation term still runs at the compute dtype (bf16 + autocast).
    master_bpp = max(bpp, 4) if not adapter_mode else bpp
    if quantized:
        # nf4 base: ~0.5 byte/param for the weights PLUS the fp16 absmax scales
        # and fp32 zero-offsets bitsandbytes materialises per block. Modelled as
        # a flat 0.5 B/param (the ~6% scale overhead is inside that rounding).
        master_weights = params * NF4_BYTES_PER_PARAM
        master_bpp = NF4_BYTES_PER_PARAM
    else:
        master_weights = params * master_bpp

    if adapter_mode:
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
        # LoRA/QLoRA keeps adapters in the compute dtype: their weights are a
        # separate term from the frozen base, which is what the adapter_gb key is.
        adapter_weights = adapter_params * bpp
    else:
        grads = params * master_bpp
        optimizer = params * OPTIMIZER_BYTES_PER_PARAM
        trainable_params = params
        trainable_source = "all_parameters"
        adapter_weights = 0

    k = ACTIVATION_K_CHECKPOINTED if checkpointing else ACTIVATION_K_PLAIN
    activations = int(layers * seq * batch * hidden * k * bpp)
    # The cross-entropy logits term is reported on its own: it is not part of the
    # activation heuristic (which counts hidden states), and at a 130k vocab it is
    # the single largest allocation of the step. The HEADLINE term models the path
    # the config actually runs: with ``loss_chunk_tokens > 0`` the trainer slices
    # the CE, so only ``[chunk, vocab]`` is upcast to fp32. The unchunked cost is
    # reported separately so the ~8x difference is visible, not implied.
    chunk_tokens = int(getattr(training, "loss_chunk_tokens", 0) or 0)
    chunk = min(chunk_tokens, seq) if chunk_tokens > 0 else seq
    logits = int(chunk * batch * vocab * LOGIT_BYTES) if vocab else 0
    logits_unchunked = int(seq * batch * vocab * LOGIT_BYTES) if vocab else 0
    total = master_weights + adapter_weights + grads + optimizer + activations + logits

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
        # ``weights_gb`` keeps its original meaning: the frozen/trained BASE
        # weights only. Adapter storage is a separate term, listed on its own.
        "weights_gb": round(master_weights / 1e9, 2),
        "base_weights_gb": round(master_weights / 1e9, 2),
        "adapter_gb": round(adapter_weights / 1e9, 2),
        "weight_bytes_per_param": master_bpp,
        "quantization": quantization,
        "layers": layers,
        "hidden": hidden,
        "shape_source": shape_source,
        "gradients_gb": round(grads / 1e9, 2),
        "optimizer_gb": round(optimizer / 1e9, 2),
        "activations_gb": round(activations / 1e9, 2),
        "logits_gb": round(logits / 1e9, 2),
        "logits_chunk_tokens": chunk,
        # What the same step would allocate WITHOUT chunking (``loss_chunk_tokens: 0``).
        "logits_unchunked_gb": round(logits_unchunked / 1e9, 2),
        "logits_note": (
            f"headline logits term models the {'chunked' if chunk_tokens > 0 else 'unchunked'} "
            f"path: {chunk} tokens upcast to fp32; unchunked would be "
            f"{round(logits_unchunked / 1e9, 2)} GB"
        ),
        "vocab_size": vocab,
        "total_gb": round(total / 1e9, 2),
        "checkpoint_size_gb": round((master_weights + adapter_weights + grads) / 1e9, 2),
        "gpu": gpu,
        "available_gb": round(available_bytes / 1e9, 2) if available_bytes else None,
        "verdict": verdict,
        "heuristic": (
            f"activations = layers({layers}) * seq({seq}) * microbatch({batch}) * "
            f"hidden({hidden}) * k({k}) * {bpp} bytes; k=2 with gradient "
            f"checkpointing else 10; logits = chunk({chunk}) * microbatch({batch}) * "
            f"vocab({vocab}) * {LOGIT_BYTES} bytes, modelling the "
            f"{'CHUNKED' if chunk_tokens > 0 else 'UNCHUNKED'} path "
            f"(loss_chunk_tokens={chunk_tokens if chunk_tokens > 0 else 0} -> chunk "
            f"{chunk} tokens; transformers upcasts that slice to fp32 in the loss; "
            f"unchunked would be {round(logits_unchunked / 1e9, 2)} GB). "
            f"Base weights at {master_bpp} byte(s)/param"
            f"{' (nf4)' if quantized else ''}. Shape from {shape_source}. "
            "Point estimate, not a measurement."
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
        f"  model shape        : {report.get('layers')} layers, hidden {report.get('hidden')} "
        f"(from {report.get('shape_source') or 'unknown'})",
        f"  parameters         : {int(report.get('param_count') or 0):,} "
        f"(trainable {int(report.get('trainable_params') or 0):,}; "
        f"count from {report.get('trainable_source') or 'unknown'})",
        f"  sequence length    : {report.get('seq_len')} (microbatch {report.get('microbatch')}, "
        f"accum {report.get('gradient_accumulation_steps')})",
        f"  base weights       : {report.get('base_weights_gb')} GB "
        f"({report.get('weight_bytes_per_param')} B/param, {report.get('quantization')})",
        f"  adapter tensors    : {report.get('adapter_gb')} GB",
        f"  weights (total)    : {report.get('weights_gb')} GB",
        f"  gradients          : {report.get('gradients_gb')} GB",
        f"  optimizer state    : {report.get('optimizer_gb')} GB",
        f"  activations        : {report.get('activations_gb')} GB (heuristic)",
        f"  logits + CE        : {report.get('logits_gb')} GB "
        f"(chunk {report.get('logits_chunk_tokens')} of {report.get('seq_len')} tokens * vocab fp32; "
        f"unchunked {report.get('logits_unchunked_gb')} GB)",
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
