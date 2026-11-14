"""LoRA configuration with architecture-aware target-module discovery.

Discovery must never silently no-op: asking for LoRA and getting a frozen model
is the classic silent failure. So the resolved module set is *derived from the
model's actual module names* (intersection with known projection suffixes), and
an empty intersection raises :class:`LifeMateError` listing what was found.

Peft/torch are imported inside the functions only.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from lifemate.errors import ConfigError, LifeMateError
from lifemate.models.loader import count_parameters

__all__ = [
    "ATTENTION_MLP_SUFFIXES",
    "HYBRID_SUFFIXES",
    "configure_lora",
    "configure_qlora",
    "discover_target_modules",
    "load_adapter",
    "merge_adapter",
    "save_adapter",
]

#: Standard attention + MLP projections (Llama/Qwen/Mistral/Gemma families).
ATTENTION_MLP_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
)

#: Hybrid / mamba-like projections (Qwen3-Next, Jamba, Mamba-2 style blocks).
HYBRID_SUFFIXES = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "in_proj", "out_proj", "x_proj", "dt_proj")

def _module_suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]

def discover_target_modules(
    model: Any,
    candidates: Sequence[str] = ATTENTION_MLP_SUFFIXES + HYBRID_SUFFIXES,
) -> list[str]:
    """Suffixes from ``candidates`` that exist as leaf module names in ``model``.

    Returns a sorted, de-duplicated list (peft matches by suffix, so suffixes are
    what it needs). Empty when the architecture exposes none of them.
    """
    present = {_module_suffix(name) for name, _module in getattr(model, "named_modules", lambda: [])()}
    return sorted({suffix for suffix in candidates if suffix in present})

def configure_qlora(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    target_modules: Sequence[str] | None = None,
) -> dict:
    """QLoRA: 4-bit frozen base + LoRA adapters, then the normal LoRA checks.

    The base must ALREADY be loaded with ``quantization='nf4'`` (see
    ``models.loader.load_model_and_tokenizer``); this function only performs the
    k-bit preparation peft requires and then delegates, so every invariant
    (nothing but ``lora_`` parameters trainable, targets resolved from the real
    module names) is checked in one place.

    ``prepare_model_for_kbit_training`` upcasts LayerNorms/outputs to fp32 and
    never casts the 4-bit weights, which is exactly why a plain
    ``model.to(dtype)`` must not be used around it.
    """
    try:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415 - lazy
    except Exception as exc:  # pragma: no cover - peft absent
        raise LifeMateError(f"QLoRA requires the 'peft' package: {exc}") from exc
    if not bool(getattr(getattr(model, "config", None), "quantization_config", None)):
        raise LifeMateError(
            "configure_qlora received an unquantized model; load it with "
            "quantization='nf4' first (a QLoRA run on 16-bit weights is just LoRA "
            "with a misleading name)"
        )
    training = getattr(cfg, "training", cfg)
    prepared = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(getattr(training, "gradient_checkpointing", False)),
    )
    report = configure_lora(prepared, cfg, target_modules)
    report["tuning_mode"] = "qlora"
    report["quantization"] = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": str(getattr(training, "precision", "bfloat16")),
        "bnb_4bit_use_double_quant": True,
        "source": "active checkpoint config.quantization_config",
    }
    return report

def _resolve_targets(model: Any, configured: Iterable[str] | None) -> list[str]:
    if configured:
        wanted = [str(item) for item in configured if str(item).strip()]
        present = {_module_suffix(name) for name, _ in getattr(model, "named_modules", lambda: [])()}
        missing = [item for item in wanted if _module_suffix(item) not in present]
        if missing:
            raise LifeMateError(
                f"LoRA target_modules configured but not found in model: {missing}. "
                f"Discovered leaf modules (sample): {_sample_modules(model)}."
            )
        resolved = sorted({item for item in wanted if _module_suffix(item) in present})
        if not resolved:
            raise LifeMateError(
                "LoRA target_modules were configured but none exist in this model. "
                f"Configured: {wanted}. Discovered leaf modules (sample): {_sample_modules(model)}. "
                "Remove lora.target_modules from the config to use automatic discovery."
            )
        return resolved
    resolved = discover_target_modules(model)
    if not resolved:
        raise LifeMateError(
            "LoRA target-module discovery found nothing to adapt: no attention/MLP projection "
            f"matched. Discovered leaf modules (sample): {_sample_modules(model)}. "
            "Set lora.target_modules explicitly for this architecture."
        )
    return resolved

def _sample_modules(model: Any, limit: int = 40) -> list[str]:
    names: list[str] = []
    for name, module in getattr(model, "named_modules", lambda: [])():
        if not list(getattr(module, "children", lambda: [])()) and name:
            names.append(name)
        if len(names) >= limit:
            break
    return names

def configure_lora(
    model: Any,
    cfg: Any,
    target_modules: Sequence[str] | None = None,
) -> dict:
    """Freeze the base model, attach a LoRA adapter, and report what happened.

    ``target_modules`` precedence: explicit argument → ``cfg.lora.target_modules``
    → automatic discovery. Raises :class:`LifeMateError` when the resolved set is
    empty or when any trainable parameter is not a LoRA parameter.
    """
    try:
        from peft import LoraConfig as PeftLoraConfig  # noqa: PLC0415 - lazy
        from peft import get_peft_model  # noqa: PLC0415 - lazy
    except Exception as exc:  # pragma: no cover - peft absent
        raise LifeMateError(
            f"LoRA tuning requires the 'peft' package (pip install peft): {exc}"
        ) from exc

    lora_cfg = getattr(cfg, "lora", cfg)
    training_cfg = getattr(cfg, "training", cfg)
    r = int(getattr(lora_cfg, "r", 16))
    alpha = int(getattr(lora_cfg, "alpha", 32))
    dropout = float(getattr(lora_cfg, "dropout", 0.05))
    bias = str(getattr(lora_cfg, "bias", "none") or "none")
    if r <= 0:
        raise ConfigError(f"lora.r must be positive, got {r}")
    if bias not in {"none", "all", "lora_only"}:
        raise ConfigError(f"lora.bias must be one of none/all/lora_only, got {bias!r}")

    configured = list(target_modules) if target_modules else getattr(lora_cfg, "target_modules", None)
    if not configured:
        # Profile default: empty for Qwen (keep hybrid discovery), the 7 verified
        # Llama projections for MiniCPM. Always intersected with the model's real
        # module names by ``_resolve_targets``, which raises when none match.
        from lifemate.models.profiles import resolve_profile  # noqa: PLC0415 - lazy

        model_cfg = getattr(cfg, "model", None)
        profile = resolve_profile(getattr(model_cfg, "name", "") or "")
        configured = profile.lora_targets() or None
        if configured:
            # A profile's list is authoritative: if the model exposes projections
            # the profile does not name (a Qwen hybrid block behind a MiniCPM
            # profile), adapting only the listed subset would be a silent
            # downgrade. Fail and say which knob to turn.
            extra = sorted(set(discover_target_modules(model)) - set(configured))
            if extra:
                raise LifeMateError(
                    f"model exposes LoRA-able projections the {profile.name!r} profile does "
                    f"not list (e.g. {', '.join(extra)}); the configured model "
                    f"({getattr(model_cfg, 'name', None)!r}) does not match its profile. "
                    "Pass --model with the matching base, or set lora.target_modules explicitly."
                )
    resolved = _resolve_targets(model, configured)

    for param in model.parameters():
        param.requires_grad_(False)

    peft_config = PeftLoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM",
        target_modules=resolved,
    )
    peft_model = get_peft_model(model, peft_config)
    counts = count_parameters(peft_model)

    checkpointing = False
    if bool(getattr(training_cfg, "gradient_checkpointing", False)):
        enable = getattr(peft_model, "gradient_checkpointing_enable", None)
        if callable(enable):
            try:
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:  # pragma: no cover - older peft signature
                enable()
            checkpointing = True
        # The base model is frozen, so checkpointed blocks see no grad-requiring
        # input and the adapter gradients silently vanish unless the embeddings
        # output is marked as requiring grad. peft's helper re-registers the hook.
        require = getattr(peft_model, "enable_input_require_grads", None)
        if callable(require):
            require()
        config = getattr(peft_model, "config", None)
        if config is not None:
            try:
                config.use_cache = False
            except (AttributeError, ValueError):  # pragma: no cover
                pass

    trainable_names = [
        name for name, param in peft_model.named_parameters() if getattr(param, "requires_grad", False)
    ]
    non_lora = sorted(name for name in trainable_names if "lora_" not in name)
    if not trainable_names:
        raise LifeMateError("LoRA adapter attached but no parameter is trainable; refusing to train")
    if non_lora:
        raise LifeMateError(
            f"LoRA attached but {len(non_lora)} non-LoRA parameter(s) are trainable "
            f"(e.g. {non_lora[0]}); only 'lora_' parameters may train"
        )

    total = counts["total"] or 1
    return {
        "tuning_mode": "lora",
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "bias": bias,
        "target_modules": list(resolved),
        "n_target_modules": len(resolved),
        "n_adapter_tensors": len(trainable_names),
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "frozen_params": counts["frozen"],
        "trainable_percentage": round(100.0 * counts["trainable"] / total, 4),
        "gradient_checkpointing": checkpointing,
        "model": peft_model,
    }

def _unwrap(peft_model: Any) -> Any:
    for attr in ("merge_and_unload",):
        if not callable(getattr(peft_model, attr, None)):
            raise LifeMateError(
                "expected a peft model exposing merge_and_unload(); got "
                f"{type(peft_model).__name__}"
            )
    return peft_model

def save_adapter(peft_model: Any, path: Any) -> str:
    """Write the adapter files only (never the base weights)."""
    save = getattr(peft_model, "save_pretrained", None)
    if not callable(save):
        raise LifeMateError(f"cannot save adapter: {type(peft_model).__name__} has no save_pretrained()")
    save(str(path))
    return str(path)

def load_adapter(base_model: Any, path: Any, adapter_name: str = "default") -> Any:
    """Attach a saved adapter to ``base_model`` (base weights untouched)."""
    try:
        from peft import PeftModel  # noqa: PLC0415 - lazy
    except Exception as exc:  # pragma: no cover - peft absent
        raise LifeMateError(f"loading a LoRA adapter requires 'peft': {exc}") from exc
    return PeftModel.from_pretrained(base_model, str(path), adapter_name=adapter_name)

def merge_adapter(peft_model: Any) -> Any:
    """Merge the adapter into the base weights and return the plain model."""
    merged = _unwrap(peft_model).merge_and_unload()
    return merged
