"""Per-model-family behaviour, in one place.

Everything that differs between the two supported LLMs lives here and nowhere
else: which chat-template kwargs to pass, which text terminates an assistant
turn, which override ids to add to the decode stop set, which LoRA projections
the architecture actually has, and the shape used when no ``config.json`` can be
read for a memory estimate.

``ModelConfig.name`` stays the single source of truth for identity: callers pass
that name (or a tokenizer/model object) and get a profile back. No other module
may branch on a model name.

Facts recorded below were measured from the released checkpoints; where a claim
in the original task description did not survive measurement, the measured value
is used and the discrepancy is noted.

Pure stdlib: importable without torch/transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "GENERIC_PROFILE",
    "MINICPM5_PROFILE",
    "MODEL_PROFILES",
    "ModelProfile",
    "NORMALIZED_KEY_LIMIT",
    "profile_for_model",
    "profile_for_tokenizer",
    "resolve_profile",
]

#: Keys are matched against a lower-cased, ``_``→``-`` normalised identity. Only
#: the descriptor part matters, so a hub id and a local directory both resolve.
NORMALIZED_KEY_LIMIT = 512

#: The closed-think tail both supported chat templates emit for
#: ``enable_thinking=False``: an empty block opened and closed, so the assistant
#: turn starts outside it. Escaped so the angle brackets stay string data.
#:
#: Measured on ``models/MiniCPM5-2B/chat_template.jinja`` and the Qwen3 tail: the
#: tags contain NO U+200B zero-width space (the raw template has no ``\\xe2\\x80\\x8b``
#: byte at all), and they tokenize to ids 8 / 9 in the MiniCPM tokenizer.
CLOSED_THINK_SUFFIX = "\x3cthink\x3e\n\n\x3c/think\x3e\n\n"

#: Terminator the templates put after every assistant turn.
IM_END = "<|im_end|>"


@dataclass(frozen=True)
class ModelProfile:
    """How one model family must be prompted, stopped and adapted."""

    #: Registry key (``minicpm5`` / ``qwen3_5`` / ``generic``).
    name: str
    #: Coarse family used for checkpoint compatibility ("minicpm" / "qwen").
    family: str
    #: Kwargs handed to ``tokenizer.apply_chat_template``.
    chat_template_kwargs: Mapping[str, Any]
    #: Extra decode stop ids added to whatever the checkpoint declares.
    speculative_stop_ids: tuple[int, ...]
    #: Projections LoRA should adapt; empty means "discover from the model".
    lora_target_modules: tuple[str, ...]
    #: ``max_new_tokens`` when the caller does not pass one.
    default_max_new_tokens: int
    #: Sequence budget assumed when a caller supplies none.
    default_seq_length: int
    #: Architectures accepted for this family (documentation + error messages).
    architecture_fallbacks: tuple[str, ...]
    #: Whether the template implements an explicit thinking on/off switch.
    supports_thinking: bool
    #: Substrings that select this profile from a hub id or a local path.
    aliases: tuple[str, ...] = ()
    #: Text that ends an assistant turn in the chat template. The supervised
    #: terminator is this text's token id, NOT ``tokenizer.eos_token_id``.
    response_terminator: str | None = None
    #: Closed-think suffix for the no-chat-template fallback path.
    no_think_suffix: str = CLOSED_THINK_SUFFIX
    #: What to pass as ``add_special_tokens`` when tokenizing a rendered prompt.
    #: False when the template already emits the BOS token (a second one would be
    #: a duplicated ``<s>``: measured 2x id 0 on MiniCPM with the default True).
    prompt_add_special_tokens: bool = True
    #: Verified parameter/layer/hidden fallbacks (used only when no config.json).
    preflight_params: int = 0
    preflight_layers: int = 0
    preflight_hidden: int = 0
    #: Vocabulary size, needed to price the fp32 logits tensor of the CE loss.
    preflight_vocab: int = 0
    #: Where ``preflight_*`` came from, quoted verbatim in the report.
    preflight_source: str = "unknown"
    #: Free-form notes surfaced in logs/tests.
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- helpers ---------------------------------------------------------- #
    @property
    def template_kwargs(self) -> dict[str, Any]:
        """Mutable copy of the chat-template kwargs (callers add their own)."""
        return dict(self.chat_template_kwargs)

    def lora_targets(self) -> list[str]:
        """Configured LoRA projections, or ``[]`` to mean auto-discovery."""
        return [str(item) for item in self.lora_target_modules]


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
MINICPM5_PROFILE = ModelProfile(
    name="minicpm5",
    family="minicpm",
    # enable_thinking=False is an officially supported mode for this template: it
    # renders an EMPTY CLOSED think block, which is byte-identical to what the
    # template emits for a training assistant turn without reasoning_content.
    chat_template_kwargs={"enable_thinking": False},
    # config.json / generation_config.json declare eos [1, 130073]; the profile
    # keeps 130073 as a defensive union in case a checkpoint drops it.
    speculative_stop_ids=(130073,),
    # Measured from the architecture (42 layers, Llama blocks): the 7 standard
    # projections. 295 Linear modules total, 294 of them match these suffixes
    # (lm_head is excluded on purpose: it is not an attention/MLP projection).
    lora_target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    # Measured action targets in outputs/datasets/train/train.jsonl with the real
    # tokenizer: 4368 targets, p50 33, p90 58, p99 113, max 165 (terminator
    # included). 192 covers every measured target; it is not 512 because a
    # thinking-disabled turn emits one short block and a bigger budget only buys
    # more rambling before the truncation guard fires.
    default_max_new_tokens=192,
    default_seq_length=8192,
    architecture_fallbacks=("LlamaForCausalLM",),
    supports_thinking=True,
    aliases=("minicpm",),
    response_terminator=IM_END,
    no_think_suffix=CLOSED_THINK_SUFFIX,
    prompt_add_special_tokens=False,
    preflight_params=2_516_756_480,
    preflight_layers=42,
    preflight_hidden=2048,
    preflight_vocab=130560,
    preflight_source="released config.json + safetensors index (measured)",
    notes=(
        "add_bos_token/add_eos_token are false; the template emits <s> itself",
        "two eos ids: </s>=1 and <|im_end|>=130073; the template terminates with 130073",
    ),
)

QWEN3_5_PROFILE = ModelProfile(
    name="qwen3_5",
    family="qwen",
    chat_template_kwargs={"enable_thinking": False},
    speculative_stop_ids=(),
    # Empty => keep the existing architecture-aware discovery, including the
    # hybrid linear-attention projections. Unchanged behaviour on purpose.
    lora_target_modules=(),
    default_max_new_tokens=512,
    default_seq_length=8192,
    architecture_fallbacks=("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"),
    supports_thinking=True,
    aliases=("qwen",),
    response_terminator=IM_END,
    no_think_suffix=CLOSED_THINK_SUFFIX,
    prompt_add_special_tokens=True,
    preflight_params=4_000_000_000,
    preflight_layers=32,
    preflight_hidden=2560,
    preflight_vocab=248320,
    preflight_source="released text_config (32 layers, hidden 2560, vocab 248320)",
    notes=("tokenizer.eos_token_id IS <|im_end|>, so the old terminator was correct",),
)

GENERIC_PROFILE = ModelProfile(
    name="generic",
    family="generic",
    chat_template_kwargs={"enable_thinking": False},
    speculative_stop_ids=(),
    lora_target_modules=(),
    default_max_new_tokens=512,
    default_seq_length=8192,
    architecture_fallbacks=(),
    supports_thinking=False,
    aliases=(),
    response_terminator=None,
    no_think_suffix=CLOSED_THINK_SUFFIX,
    prompt_add_special_tokens=True,
    preflight_params=4_000_000_000,
    preflight_layers=32,
    preflight_hidden=2560,
    preflight_vocab=0,
    preflight_source="no verified shape for this family (generic fallback)",
    notes=("unknown family: terminator falls back to tokenizer.eos_token_id",),
)

#: Registry, in match order (specific aliases first).
MODEL_PROFILES: tuple[ModelProfile, ...] = (
    MINICPM5_PROFILE,
    QWEN3_5_PROFILE,
)


def _normalize(name: Any) -> str:
    """Lower-case, ``_``→``-``; strips a trailing path separator."""
    text = str(name or "").strip().lower().replace("_", "-")
    return text[:NORMALIZED_KEY_LIMIT].rstrip("/")


def resolve_profile(model_name_or_path: Any) -> ModelProfile:
    """Profile for a hub id, a local directory or a tokenizer/model identity.

    Unmatched names yield :data:`GENERIC_PROFILE`, which reproduces the previous
    behaviour exactly (``enable_thinking=False``, ``eos_token_id`` terminator,
    auto-discovered LoRA targets, 512 new tokens).
    """
    key = _normalize(model_name_or_path)
    if not key:
        return GENERIC_PROFILE
    for profile in MODEL_PROFILES:
        if any(alias in key for alias in profile.aliases):
            return profile
    return GENERIC_PROFILE


def profile_for_tokenizer(tokenizer: Any) -> ModelProfile:
    """Profile from a tokenizer's recorded identity.

    A tokenizer saved into a training checkpoint reports the checkpoint
    directory, which carries no family name: callers that know the model name
    (the trainer, the inference scripts) MUST pass the profile explicitly so the
    terminator does not silently change after a save/reload round trip.
    """
    return resolve_profile(getattr(tokenizer, "name_or_path", ""))


def profile_for_model(model: Any) -> ModelProfile:
    """Profile from ``model.config._name_or_path`` (best effort)."""
    config = getattr(model, "config", None)
    return resolve_profile(getattr(config, "_name_or_path", ""))
