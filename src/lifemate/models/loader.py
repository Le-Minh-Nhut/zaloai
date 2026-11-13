"""Model + tokenizer loading for Qwen-family causal LMs, with LoRA support.

Design rules enforced here:

* **No module-level heavy imports.** ``torch``/``transformers``/``peft`` are
  imported inside functions, so prompt builders, data code and tests run on a
  CPU-only stdlib box.
* **No silent substitution.** An unsupported architecture, a missing PEFT
  install or an unreadable checkpoint raises :class:`LifeMateError` with an
  actionable message. We never fall back to a different model or to random
  weights.
* **Offline friendly.** ``local_files_only`` is set when the path exists on
  disk or ``HF_HUB_OFFLINE=1`` is set, so a scoring box without network access
  fails fast instead of hanging on a hub call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lifemate.errors import LifeMateError

__all__ = [
    "DTYPE_MAP",
    "ModelBundle",
    "bf16_note",
    "count_parameters",
    "is_local_or_offline",
    "load_model_and_tokenizer",
    "resolve_dtype",
    "resolve_model_mode",
]

#: String dtype names accepted from config → torch dtype attribute names.
DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
}

#: Upper bound for ``max_new_tokens``: one decision block is a few dozen tokens.
MAX_NEW_TOKENS_CEILING = 4096


@dataclass
class ModelBundle:
    """Loaded model plus everything needed to generate and to log provenance."""

    model: Any
    tokenizer: Any
    model_mode: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> Any:
        """Device of the first parameter, or ``None`` when unknown."""
        return getattr(self.model, "device", None)

    @property
    def generation_config(self) -> dict[str, Any]:
        """Sane decode defaults (never greedy+sampling mixed)."""
        return dict(self.meta.get("generation_config") or {})


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable without torch)
# --------------------------------------------------------------------------- #
def is_local_or_offline(
    model_name_or_path: str | Path, env: dict[str, str] | None = None
) -> bool:
    """Whether ``from_pretrained`` must be called with ``local_files_only=True``.

    True when the given path exists locally, when ``HF_HUB_OFFLINE`` is truthy,
    or when ``TRANSFORMERS_OFFLINE`` is truthy.
    """
    environ = os.environ if env is None else env
    name = str(model_name_or_path or "").strip()
    if not name:
        return False  # "" resolves to cwd: never treat that as a local model
    if Path(name).exists():
        return True
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = str(environ.get(key, "")).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


def resolve_dtype(torch_dtype: Any, device: str = "cuda") -> Any:
    """Map a config string to a ``torch.dtype``.

    bfloat16 on CUDA without hardware support degrades to float16 with an
    explicit note in the returned object's ``meta`` — decided by the caller of
    :func:`load_model_and_tokenizer`; this helper only raises on unknown names.
    """
    import torch  # local import: torch is optional for the rest of the package

    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    key = str(torch_dtype or "bfloat16").strip().lower()
    if key not in DTYPE_MAP:
        raise LifeMateError(
            f"unsupported torch_dtype {torch_dtype!r}; expected one of "
            f"{sorted(DTYPE_MAP)} or a torch.dtype instance"
        )
    return getattr(torch, DTYPE_MAP[key])


def bf16_note(torch_dtype: Any, device: str) -> tuple[Any, str | None]:
    """Downgrade bf16→fp16 when the device cannot run bf16; return a note.

    Returns ``(dtype_like_torch_dtype_or_str, note_or_None)`` without importing
    torch; the caller resolves the string with :func:`resolve_dtype`.
    """
    name = str(torch_dtype).lower()
    if "bf16" not in name and "bfloat16" not in name:
        return torch_dtype, None
    try:
        import torch

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            return "float16", "CUDA unavailable: bfloat16 requested, using float16 on CPU"
        if str(device).startswith("cuda") and not torch.cuda.is_bf16_supported():
            return "float16", "GPU lacks bfloat16 support: falling back to float16"
    except Exception:  # pragma: no cover - torch missing/broken
        return torch_dtype, None
    return torch_dtype, None


def resolve_model_mode(tuning_mode: str | None, checkpoint: str | Path | None) -> str:
    """``base`` | ``full`` | ``lora`` from the config's tuning mode.

    ``full`` without a checkpoint means "base weights, will be trained"; with a
    checkpoint it means a full-finetune checkpoint directory. ``lora`` requires
    an adapter checkpoint to be usable for inference.
    """
    mode = "full" if tuning_mode is None else str(tuning_mode).strip().lower()
    if mode not in ("full", "lora"):
        raise LifeMateError(
            f"unsupported tuning_mode {tuning_mode!r}; expected 'full' or 'lora'"
        )
    if checkpoint is None or str(checkpoint).strip() == "":
        return "base"
    return mode


def count_parameters(model: Any) -> dict[str, int]:
    """Parameter counts; ``trainable`` uses ``requires_grad``."""
    total = 0
    trainable = 0
    for param in getattr(model, "parameters", lambda: [])():
        n = int(param.numel())
        total += n
        if bool(getattr(param, "requires_grad", False)):
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _default_generation_config(tokenizer: Any, temperature: float, max_new_tokens: int) -> dict[str, Any]:
    """Decode defaults: greedy unless ``temperature > 0``; bounded new tokens."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return {
        "do_sample": bool(temperature and temperature > 0),
        "temperature": float(temperature) if temperature and temperature > 0 else None,
        # Bounded so a bad config can never ask for an unbounded decode.
        "max_new_tokens": max(1, min(int(max_new_tokens or MAX_NEW_TOKENS_CEILING), MAX_NEW_TOKENS_CEILING)),
        "pad_token_id": pad_id if pad_id is not None else eos_id,
        "eos_token_id": eos_id,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    torch_dtype: str = "bfloat16",
    device: str = "cuda",
    tuning_mode: str = "full",
    checkpoint: str | Path | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    lora_config: dict[str, Any] | None = None,
    trainable: bool = False,
    for_training: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> ModelBundle:
    """Load a causal LM (+ optional LoRA adapter) and its tokenizer.

    ``trust_remote_code=True`` executes Python shipped with the weights; only
    enable it for a repository you have audited, and never for an arbitrary hub
    id at scoring time. The default is False, which is what the competition
    submission uses.

    Raises :class:`LifeMateError` when the architecture is not a supported
    causal LM, when PEFT is required but missing, or when an offline load would
    need the network.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LifeMateError(
            "load_model_and_tokenizer requires torch and transformers: "
            f"pip install torch transformers ({exc})"
        ) from exc

    mode = resolve_model_mode(tuning_mode, checkpoint)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise LifeMateError(
            f"device={device!r} requested but torch.cuda.is_available() is False; "
            "pass device='cpu' for a dry run or run on a GPU host"
        )
    dtype_name, note = bf16_note(torch_dtype, device)
    dtype = resolve_dtype(dtype_name, device)
    # Offline when either the base id/path or the checkpoint resolves locally.
    local_only = is_local_or_offline(model_name_or_path) or is_local_or_offline(
        str(checkpoint or "")
    )

    # A checkpoint usually carries its own tokenizer; prefer it over the base id.
    tokenizer_source = str(checkpoint) if _has_tokenizer_files(checkpoint) else model_name_or_path
    tokenizer = _load_tokenizer(AutoTokenizer, tokenizer_source, local_only, trust_remote_code)
    _ensure_pad_token(tokenizer)

    base_source = model_name_or_path
    if mode == "full" and checkpoint:
        base_source = str(checkpoint)

    model = _load_causal_lm(
        AutoModelForCausalLM, base_source, dtype, local_only, trust_remote_code, attn_implementation
    )

    if mode == "lora" and checkpoint:
        model = _attach_lora_adapter(model, checkpoint, trainable=trainable)

    if str(device):
        try:
            model.to(device)
        except Exception as exc:  # pragma: no cover - device/driver issues
            raise LifeMateError(
                f"could not move model to device {device!r}: {exc}; "
                "check torch.cuda.is_available() and the device string"
            ) from exc

    effective_trainable = trainable or for_training
    if effective_trainable:
        model.train()
        if mode in ("base", "full") and tuning_mode != "lora":
            for param in model.parameters():
                param.requires_grad_(True)
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    params = count_parameters(model)
    gen_cfg = _default_generation_config(tokenizer, temperature, max_new_tokens)
    meta: dict[str, Any] = {
        "model_name": model_name_or_path,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_mode": mode,
        "torch_dtype": str(dtype),
        "device": str(device),
        "attn_implementation": attn_implementation,
        "trust_remote_code": bool(trust_remote_code),
        "local_files_only": local_only,
        "for_training": bool(for_training),
        "trainable": bool(effective_trainable),
        "n_parameters": params,
        "generation_config": gen_cfg,
        "note": note,
    }
    return ModelBundle(model=model, tokenizer=tokenizer, model_mode=mode, meta=meta)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _load_tokenizer(AutoTokenizer: Any, source: str, local_only: bool, trust: bool) -> Any:
    """Load the tokenizer, converting every failure into ``LifeMateError``."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=trust, local_files_only=local_only
        )
    except Exception as exc:
        raise LifeMateError(
            f"could not load tokenizer for {source!r} "
            f"(local_files_only={local_only}): {exc}"
        ) from exc
    if tokenizer is None:
        raise LifeMateError(f"tokenizer loading returned None for {source!r}")
    return tokenizer


def _has_tokenizer_files(checkpoint: str | Path | None) -> bool:
    """Whether ``checkpoint`` is a directory holding tokenizer artifacts."""
    if not checkpoint:
        return False
    path = Path(str(checkpoint))
    if not path.is_dir():
        return False
    return any(
        (path / name).exists()
        for name in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "spiece.model")
    )


def _ensure_pad_token(tokenizer: Any) -> None:
    """Decoder-only models often have no pad token; reuse EOS (documented)."""
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token


def _tied_names(model: Any) -> set[str]:
    """Names transformers re-ties on load (``lm_head`` sharing ``embed_tokens``)."""
    tied: set[str] = set()
    for attr, target in (getattr(model, "_tied_weights_keys", None) or {}).items():
        tied.add(attr)
        if isinstance(target, str):
            tied.add(target)
    return tied


def _missing_parameters(model: Any, info: Any) -> list[str]:
    """Parameters the checkpoint did not provide (tied weights excluded).

    ``from_pretrained`` only *prints* a MISSING table and keeps the random
    initialisation, so a truncated download or a config that claims more layers
    than the weight files hold silently trains a partly random model. Reading
    the loading info is the only way to turn that into an error.
    """
    if not isinstance(info, dict):
        return []
    tied = _tied_names(model)
    return sorted(str(key) for key in (info.get("missing_keys") or []) if str(key) not in tied)


def _load_causal_lm(
    AutoModelForCausalLM: Any,
    source: str,
    dtype: Any,
    local_only: bool,
    trust: bool,
    attn_implementation: str | None,
) -> Any:
    """Load the causal LM; unsupported architectures raise ``LifeMateError``."""
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(trust),
        "local_files_only": local_only,
        "output_loading_info": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        result = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except ValueError as exc:
        raise LifeMateError(
            f"architecture of {source!r} is not supported as a causal LM "
            f"(AutoModelForCausalLM). Use a decoder-only checkpoint or pass a "
            f"matching Auto* class yourself. Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise LifeMateError(
            f"could not load model weights from {source!r} "
            f"(local_files_only={local_only}): {exc}"
        ) from exc
    loaded, info = result if isinstance(result, tuple) else (result, None)
    if loaded is None:
        raise LifeMateError(f"model loading returned None for {source!r}")
    missing = _missing_parameters(loaded, info)
    if missing:
        raise LifeMateError(
            f"checkpoint {source!r} is incomplete: {len(missing)} parameter(s) have no "
            f"weights (e.g. {missing[0]}). Training from this would silently fine-tune a "
            "partly random model. Re-download/complete the checkpoint."
        )
    return loaded


def _attach_lora_adapter(model: Any, checkpoint: str | Path, *, trainable: bool) -> Any:
    """Wrap ``model`` with the PEFT adapter stored in ``checkpoint``."""
    path = Path(str(checkpoint))
    if not path.exists():
        raise LifeMateError(
            f"LoRA adapter checkpoint not found: {path}. Pass the directory written by "
            "models.checkpoint.save_checkpoint(is_lora=True)."
        )
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise LifeMateError(
            "loading a LoRA adapter requires the 'peft' package: pip install peft. "
            f"Adapter requested at {path}."
        ) from exc
    # The adapter directory's own ``adapter_config.json`` governs r/alpha/targets;
    # ``lora_config`` is only used by the training slice when creating a fresh
    # adapter, so it is deliberately not injected here.
    return PeftModel.from_pretrained(model, str(path), is_trainable=bool(trainable))
