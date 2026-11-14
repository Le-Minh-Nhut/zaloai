"""Supervised-example construction and batching for action SFT.

Labels come only from the target action text. The prompt is masked with
``-100`` so the loss is computed exclusively on ``<action>...</action>`` tokens
(that is what stops the model from learning to regenerate its own prompt, and
what makes the prompt/answer split auditable in tests).

Truncation policy is deliberately asymmetric:

* the **target is never truncated** — a chopped action block teaches broken
  JSON, so an example whose target does not fit is dropped with an explicit
  reason (``target_too_long``);
* when the target fits, the prompt is truncated from the **middle** rather than
  the left, keeping both the head (spoken request and world context) and the tail
  (tools, prior steps, format spec) while dropping older conversation history;
* the ``# Yêu cầu`` section and the ``# Công cụ khả dụng`` catalog are *pinned*:
  the request to the prefix, then the catalog. When the catalog plus the
  steps/format tail fits the remaining budget the whole tail is pinned and only
  the middle is cut. When it does NOT fit (MiniCPM5-2B's tokenizer renders the
  catalog at ~7833 tokens against 7714 remaining at 8192) the whole catalog is
  kept anyway, the tail is added only if it still fits, and the cut moves into
  the context/history head; only a catalog that alone exceeds the budget is cut
  inside. Measured with the real MiniCPM tokenizer over 3170 train / 352 val
  ``tool_call`` records: 0 kept prompts lose the ``### <tool> (kind=...)`` block
  or the catalog at 8192 (70 train / 5 val records are instead DROPPED as
  ``target_value_truncated``, because the deeper head cut removes the history
  value the target copies), and 0 lost, 0 dropped, 0 truncated at 12288.
  The pre-fix behaviour severed the block for 70/3170 train and 5/352 val
  records (the fallback middle cut), which is what this path replaces.
  Residual risk: the splice boundary can still land inside a history turn, and a
  catalog longer than the whole budget is still cut inside;
* a ``tool_call`` target whose **reference value** (``*_id``/``*_ref``/``*_name``)
  is no longer a substring of the truncated prompt is dropped with
  ``reason="target_value_truncated"``. The data builder performs the same check
  before truncation; after it, the record would supervise an identifier the
  model was never shown. Measured with the DEFAULT model (MiniCPM5-2B tokenizer)
  at max_seq_length 8192: 70 / 4368 train (1.60 %) and 5 / 482 val (1.04 %)
  records dropped, 488 train / 41 val prompts middle-truncated. At 5632 the same
  rule drops 608 / 4368 (13.92 %) and 67 / 482 — which is why 8192 is the
  shipped default. The earlier "0 records at 8192, 122 at 4096" figures were
  measured with the Qwen3.5-4B tokenizer and do not apply to MiniCPM.
  Non-reference arguments are exempt: the model legitimately composes text and
  dates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from lifemate.agent.protocol import ACTION_CLOSE, ACTION_OPEN
from lifemate.errors import LifeMateError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lifemate.models.profiles import ModelProfile

__all__ = [
    "CHATML_FALLBACK_TEMPLATE",
    "NO_THINK_SUFFIX",
    "SupervisedCollator",
    "apply_chat_template_text",
    "assert_loss_masking",
    "build_supervised_example",
    "render_prompt_text",
    "resolve_response_terminator",
    "to_torch_batch",
]

#: Fallback when ``tokenizer.apply_chat_template`` is unavailable.
#: MUST be identical at inference time; ``models/generation.py`` uses the
#: tokenizer's own chat template, so this fallback is only for bare tokenizers.
#: A tokenizer that *has* a template and raises is a real failure: both sides
#: raise (:class:`LifeMateError`) instead of diverging into different formats.
CHATML_FALLBACK_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"

#: What Qwen3-style templates emit for ``enable_thinking=False``: the think
#: block is opened and closed empty, so the assistant turn starts outside it.
#: Escaped on purpose: the angle brackets must be part of the STRING, not markup.
NO_THINK_SUFFIX = "\x3cthink\x3e\n\n\x3c/think\x3e\n\n"

_FALLBACK_GENERATION_SUFFIX = "<|im_start|>assistant\n" + NO_THINK_SUFFIX


def apply_chat_template_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = True,
    profile: ModelProfile | None = None,
) -> str:
    """Render chat messages with the profile's ``chat_template_kwargs``.

    Thinking must be off on both sides of the prompt contract: a template that
    ends with an *open* ``think`` block would train the model to emit the action
    inside its reasoning, and at inference leaves it mid-reasoning with nothing
    telling it to stop. Both supported profiles pass ``enable_thinking=False``,
    which each of their templates turns into an empty closed block.

    Only a ``TypeError`` naming one of the kwargs means "this tokenizer does not
    know it"; the plain call is retried. Any other exception is a real template
    failure and propagates to the caller.
    """
    from lifemate.models.profiles import profile_for_tokenizer  # noqa: PLC0415 - lazy: models/__init__ imports this module

    active = profile or profile_for_tokenizer(tokenizer)
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(
            messages, **active.template_kwargs, **kwargs
        )
    except TypeError as exc:
        unknown = [key for key in active.chat_template_kwargs if key in str(exc)]
        if not unknown:
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)

def render_prompt_text(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any = None,
    profile: ModelProfile | None = None,
) -> str:
    """Render chat messages to text, preferring the tokenizer's chat template.

    Deterministic for identical inputs. Raises :class:`LifeMateError` when the
    messages are malformed, or when a tokenizer that *has* a chat template fails
    to render, rather than emitting a silently different prompt than inference.
    The ChatML fallback covers only tokenizers with no template at all.
    """
    normalised: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping) or "role" not in message:
            raise LifeMateError(f"chat message must be a mapping with a role, got {message!r}")
        normalised.append({"role": str(message["role"]), "content": str(message.get("content") or "")})
    if not normalised:
        raise LifeMateError("cannot render an empty message list")

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            rendered = apply_chat_template_text(
                tokenizer, normalised, add_generation_prompt=True, profile=profile
            )
        except Exception as exc:
            # Parity with inference (HFGenerator.build_prompt raises too): quietly
            # reformatting here would train on different prompt bytes than we serve.
            raise LifeMateError(
                "tokenizer.apply_chat_template failed; the tokenizer has no usable chat "
                f"template for these messages ({type(tokenizer).__name__}): {exc}"
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise LifeMateError(
                f"tokenizer.apply_chat_template returned {type(rendered).__name__}, expected a non-empty str"
            )
        return rendered
    body = "".join(
        CHATML_FALLBACK_TEMPLATE.format(role=m["role"], content=m["content"]) for m in normalised
    )
    return body + _FALLBACK_GENERATION_SUFFIX

def _encode(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    """Tokenize ``text`` (ids only); empty text yields ``[]``.

    ``add_special_tokens`` mirrors ``ModelProfile.prompt_add_special_tokens`` so
    the prompt encode step is the same on both sides: the training path passes
    the profile's value for the PROMPT, and ``HFGenerator`` passes the same value
    when it tokenizes the rendered prompt. Targets never get special tokens.
    """
    if not text:
        return []
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=add_special_tokens)
        except TypeError:  # pragma: no cover - minimal tokenizers
            ids = encode(text)
    else:  # pragma: no cover - callable tokenizer
        ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    return [int(i) for i in ids]

def eos_token_ids(tokenizer_or_config: Any) -> list[int]:
    """Every EOS id an object declares, as ints.

    MiniCPM declares TWO (``[1, 130073]``) and the old ``isinstance(value, int)``
    test silently ignored the list, so the collator supervised ``</s>`` instead
    of the ``<|im_end|>`` the chat template actually emits.
    """
    value = getattr(tokenizer_or_config, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return sorted({int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)})
    return []

def resolve_response_terminator(
    tokenizer: Any, profile: ModelProfile | None = None
) -> tuple[int | None, str]:
    """Token id that ends an assistant turn, plus where it came from.

    The supervised terminator must be the token the CHAT TEMPLATE emits
    (``<|im_end|>`` for both supported families), not ``tokenizer.eos_token_id``:
    MiniCPM's ``eos_token`` is ``</s>`` (1) while its template closes every turn
    with ``<|im_end|>`` (130073), so the old code supervised a terminator the
    model never produces. Qwen's ``eos_token`` already IS ``<|im_end|>``, which
    is why the bug was invisible with one model.

    Returns ``(id, source)``; ``source`` is ``profile:<name>`` when the profile's
    terminator text resolved to exactly one id, else ``tokenizer_eos:<ids>``.
    The trailing newline the template writes after the terminator is NOT
    supervised: it carries no decision, and hiding it keeps the loss on the
    action block plus its terminator only.
    """
    from lifemate.models.profiles import profile_for_tokenizer  # noqa: PLC0415 - lazy

    active = profile or profile_for_tokenizer(tokenizer)
    terminator = active.response_terminator
    if terminator:
        ids = _encode(tokenizer, terminator)
        if len(ids) == 1:
            return ids[0], f"profile:{active.name}"
    eos = eos_token_ids(tokenizer)
    if len(eos) == 1:
        return eos[0], f"tokenizer_eos:{eos[0]}"
    if eos:
        return eos[0], f"tokenizer_eos:{eos}[0]"
    return None, "none"

def _eos_id(tokenizer: Any) -> int | None:
    """First EOS id, kept for callers that only need "the" terminator."""
    eos = eos_token_ids(tokenizer)
    return eos[0] if eos else None

#: Section header that opens the spoken request in a decision prompt
#: (:func:`lifemate.agent.prompt_builder.build_decision_prompt`). The request is
#: the one part of a prompt that can never be recovered from context, so
#: truncation protects it even at the cost of older history.
REQUEST_MARKER = "# Yêu cầu"
#: Section header that opens the tool catalog. A target ``tool_call`` names a
#: tool whose ``### <tool> (kind=...)`` block must be readable, and the block can
#: sit anywhere inside the catalog, so the catalog is pinned to the tail instead
#: of being cut in half by the middle cut.
#:
#: The earlier note that the catalog "fits the tail budget of every prepared
#: record" was measured with the Qwen tokenizer and is FALSE for MiniCPM5-2B:
#: its tokenizer renders the catalog at ~7833 tokens, leaving 7714 of the 8192
#: budget, so the pin was skipped and the fallback middle cut severed the
#: ``### <tool> (kind=...)`` block of the tool the target calls for 45/2979
#: (1.51 %) train and 2/326 (0.61 %) val ``tool_call`` prompts. The pin now has
#: a second, catalog-aware path instead of falling back to a full middle cut.
TOOLS_MARKER = "# Công cụ khả dụng"
_SECTION_SEP = "\n\n"

def _split_at_request(text: str) -> tuple[str, str] | None:
    """Split prompt text into ``(system+request, rest)``; ``None`` when unmarked.

    The split is textual on purpose: re-tokenising two halves instead of one
    string costs at most a couple of boundary merges, while guaranteeing the
    request survives truncation.
    """
    start = text.find(REQUEST_MARKER)
    if start < 0:
        return None
    body_start = start + len(REQUEST_MARKER)
    nxt = text.find(_SECTION_SEP + "# ", body_start)
    end = len(text) if nxt < 0 else nxt
    return text[:end], text[end:]

def _split_catalog(text: str) -> tuple[str, str, str] | None:
    """Split ``(before, catalog, after)``; ``None`` when the catalog is unmarked.

    ``catalog`` starts at its ``# Công cụ khả dụng`` heading and ends before the
    next ``\n\n# `` heading, so the section can be pinned as one indivisible unit
    while ``after`` (prior steps, response format) stays separately placeable.
    """
    split = _split_at_tools_section(text)
    if split is None:
        return None
    before, catalog_and_after = split
    nxt = catalog_and_after.find(_SECTION_SEP + "# ", len(_SECTION_SEP)) 
    if nxt < 0:
        return before, catalog_and_after, ""
    return before, catalog_and_after[:nxt], catalog_and_after[nxt:]


def _split_at_tools_section(text: str) -> tuple[str, str] | None:
    """Split ``(before, catalog-and-after)``; ``None`` when unmarked.

    Same textual approach as :func:`_split_at_request`: the catalog is pinned to
    the preserved tail so a middle cut can never delete the schema block of the
    tool the target calls.
    """
    start = text.find(_SECTION_SEP + TOOLS_MARKER)
    if start < 0:
        return None
    return text[:start], text[start:]

#: Argument-name suffixes the data builder treats as identifying a value produced
#: elsewhere (``*_id``/``*_ref``/``*_name``); kept in sync with
#: ``lifemate.data.trajectory_builder._REF_ARG_SUFFIXES``, redefined here so this
#: module stays importable without the data layer.
_REF_ARG_SUFFIXES = ("_id", "_ref", "_name")

def _target_values(target_text: str) -> list[str]:
    """Reference-valued arguments of a ``tool_call`` target, as plain strings.

    ``build_decision_points`` already drops a point whose reference value is
    absent from the *untruncated* prompt. Truncation can delete that value
    afterwards, so the same question has to be asked of the FINAL prompt text.
    Only non-empty scalar values of reference-like arguments are returned: other
    arguments are routinely composed by the model (a greeting, a computed
    date), and demanding them verbatim would drop correct examples.
    """
    text = target_text or ""
    start, end = text.find(ACTION_OPEN), text.rfind(ACTION_CLOSE)
    if start < 0 or end < 0:
        return []
    try:
        action = json.loads(text[start + len(ACTION_OPEN) : end])
    except (ValueError, TypeError):
        return []
    if not isinstance(action, Mapping) or action.get("type") != "tool_call":
        return []
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return []
    values: list[str] = []
    for key, value in arguments.items():
        if not str(key).endswith(_REF_ARG_SUFFIXES):
            continue
        if isinstance(value, (Mapping, list, tuple)) or value is None:
            continue
        needle = str(value)
        if needle:
            values.append(needle)
    return values

def _values_hidden_by_truncation(tokenizer: Any, prompt_ids: Sequence[int], target_text: str) -> list[str]:
    """Reference values of ``target_text`` that the truncated prompt no longer shows.

    A plain substring test on the decoded prompt — the same question
    ``_unobservable_ref_args`` asks of the untruncated text, moved after the cut.
    Decoding happens only for records that were actually truncated, so the
    normal path costs nothing.
    """
    values = _target_values(target_text)
    if not values:
        return []
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return []  # cannot inspect: never drop on a guess
    text = decode(list(prompt_ids), skip_special_tokens=False)
    return [value for value in values if value not in text]

def _middle_truncate(ids: list[int], budget: int, head_keep_fraction: float) -> list[int]:
    """Drop the middle of ``ids`` to ``budget`` tokens (head fraction first)."""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return list(ids)
    head_budget = int(budget * head_keep_fraction)
    tail_budget = budget - head_budget
    head = ids[:head_budget]
    tail = ids[-tail_budget:] if tail_budget > 0 else []
    return head + tail

@dataclass
class SupervisedExample:
    """One tokenised decision point plus the bookkeeping training needs."""

    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    kept: bool = True
    reason: str | None = None
    n_prompt_tokens: int = 0
    n_target_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
            "kept": self.kept,
            "reason": self.reason,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_target_tokens": self.n_target_tokens,
        }

def build_supervised_example(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    target_text: str,
    max_seq_length: int,
    head_keep_fraction: float = 0.35,
    profile: ModelProfile | None = None,
) -> dict[str, Any]:
    """Tokenise one (prompt, target) pair with target-only labels.

    The supervised sequence is ``prompt + target + terminator``: the terminator
    comes from the model profile (see :func:`resolve_response_terminator`), and
    the newline the chat template writes after it is deliberately NOT supervised.

    Returns the dict documented on :class:`SupervisedExample`. A dropped example
    comes back with ``kept=False``, a ``reason`` and empty token lists, so the
    caller can count it without special-casing.
    """
    if isinstance(head_keep_fraction, bool) or not isinstance(head_keep_fraction, (int, float)) or not (0.0 <= head_keep_fraction <= 1.0):
        raise LifeMateError(f"head_keep_fraction must be a float between 0.0 and 1.0, got {head_keep_fraction!r}")
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise LifeMateError(f"max_seq_length must be a positive int, got {max_seq_length!r}")
    if not isinstance(target_text, str) or not target_text.strip():
        return SupervisedExample(kept=False, reason="empty_target").to_dict()

    prompt_text = render_prompt_text(messages, tokenizer, profile)
    # Same encode flag as HFGenerator.generate() uses on the identical string.
    from lifemate.models.profiles import profile_for_tokenizer  # noqa: PLC0415 - lazy

    prompt_flags = (profile or profile_for_tokenizer(tokenizer)).prompt_add_special_tokens
    # Only the FIRST piece carries the flag: the render already contains the BOS,
    # so applying it to a later fragment would inject a second one mid-prompt.
    prompt_ids = _encode(tokenizer, prompt_text, add_special_tokens=prompt_flags)
    target_ids = _encode(tokenizer, target_text)
    if not target_ids:
        return SupervisedExample(kept=False, reason="empty_target").to_dict()

    eos, _source = resolve_response_terminator(tokenizer, profile)
    has_eos = bool(target_ids) and eos is not None and target_ids[-1] == eos
    if not has_eos:
        if eos is None:
            return SupervisedExample(kept=False, reason="no_eos_token").to_dict()
        target_ids = target_ids + [eos]

    n_target = len(target_ids)
    if n_target >= max_seq_length:
        return SupervisedExample(
            kept=False, reason="target_too_long", n_prompt_tokens=len(prompt_ids), n_target_tokens=n_target
        ).to_dict()

    budget = max_seq_length - n_target
    truncated_prompt = 0
    if len(prompt_ids) > budget:
        truncated_prompt = len(prompt_ids) - budget
        split = _split_at_request(prompt_text)
        if split is not None:
            # Keep the system+request prefix intact and truncate only what
            # follows it. The request is the single element the target cannot be
            # inferred without, and it lives at the very start of the body, so a
            # plain middle cut would discard it as soon as history grows.
            prefix_text, rest_text = split
            prefix_ids = _encode(tokenizer, prefix_text, add_special_tokens=prompt_flags)
            if len(prefix_ids) < budget:
                remaining = budget - len(prefix_ids)
                # Pin the tool catalog (and everything after it) to the tail so a
                # middle cut cannot delete the schema block of the tool the target
                # calls; only what precedes the catalog is middle-truncated.
                catalog = _split_catalog(rest_text)
                tail_ids = (
                    _encode(tokenizer, catalog[1] + catalog[2], add_special_tokens=False)
                    if catalog is not None
                    else []
                )
                if 0 < len(tail_ids) < remaining:
                    head_ids = _middle_truncate(
                        _encode(tokenizer, catalog[0]), remaining - len(tail_ids), head_keep_fraction
                    )
                    prompt_ids = prefix_ids + head_ids + tail_ids
                elif catalog is not None and _encode(tokenizer, catalog[1]):
                    # The catalog plus everything after it does not fit. Keep the
                    # WHOLE catalog first (it is what makes the target answerable),
                    # then the response-format/steps tail if it still fits, and
                    # give only what remains to the context+history head. Only the
                    # catalog's own overflow forces a cut inside it.
                    before_text, catalog_text, after_text = catalog
                    catalog_ids = _encode(tokenizer, catalog_text)
                    after_ids = _encode(tokenizer, after_text)
                    if len(catalog_ids) <= remaining:
                        keep_after = (
                            after_ids if len(catalog_ids) + len(after_ids) <= remaining else []
                        )
                        head_budget = remaining - len(catalog_ids) - len(keep_after)
                        head_ids = _middle_truncate(
                            _encode(tokenizer, before_text, add_special_tokens=prompt_flags),
                            head_budget,
                            head_keep_fraction,
                        )
                        prompt_ids = prefix_ids + head_ids + catalog_ids + keep_after
                    else:
                        prompt_ids = prefix_ids + _middle_truncate(
                            catalog_ids, remaining, head_keep_fraction
                        )
                else:
                    rest_ids = _encode(tokenizer, rest_text)
                    prompt_ids = prefix_ids + _middle_truncate(rest_ids, remaining, head_keep_fraction)
            else:
                prompt_ids = _middle_truncate(prefix_ids, budget, head_keep_fraction)
        else:
            prompt_ids = _middle_truncate(prompt_ids, budget, head_keep_fraction)
    n_prompt = len(prompt_ids)
    if truncated_prompt:
        # The untruncated prompt was checked by the data builder; the cut happens
        # here, so ask the same question of the FINAL text. A ``tool_call`` whose
        # reference value (an ID/name copied from history or an earlier result) no
        # longer appears in the prompt is unanswerable: drop it by name instead of
        # supervising a target the model cannot derive.
        hidden = _values_hidden_by_truncation(tokenizer, prompt_ids, target_text)
        if hidden:
            return SupervisedExample(
                kept=False,
                reason="target_value_truncated",
                n_prompt_tokens=n_prompt,
                n_target_tokens=n_target,
            ).to_dict()
    input_ids = prompt_ids + target_ids
    labels = [-100] * n_prompt + list(target_ids)
    return SupervisedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        kept=True,
        reason=f"prompt_middle_truncated_{truncated_prompt}" if truncated_prompt else None,
        n_prompt_tokens=n_prompt,
        n_target_tokens=n_target,
    ).to_dict()

def assert_loss_masking(labels: Sequence[int], prompt_len: int) -> None:
    """Assert labels mask exactly the prompt and supervise exactly the target."""
    if prompt_len < 1 or prompt_len >= len(labels):
        raise LifeMateError(f"prompt_len {prompt_len} out of range for labels of length {len(labels)}")
    head = list(labels[:prompt_len])
    if any(value != -100 for value in head):
        raise LifeMateError("loss masking broken: prompt positions must all be -100")
    tail = list(labels[prompt_len:])
    if not tail or any(value == -100 for value in tail):
        raise LifeMateError("loss masking broken: every target position must be supervised")

@dataclass
class SupervisedCollator:
    """Pad a batch of kept examples to the longest length.

    ``pad_token_id`` defaults to ``0`` only when the tokenizer exposes nothing:
    a real tokenizer MUST pass ``tokenizer.pad_token_id`` or ``eos_token_id``.
    Padding uses ``attention_mask=0`` and ``labels=-100``, so padded positions
    can never contribute to the loss.
    """

    pad_token_id: int
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pad_token_id, int) or isinstance(self.pad_token_id, bool):
            raise LifeMateError("SupervisedCollator requires an integer pad_token_id")

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, pad_to_multiple_of: int | None = None) -> "SupervisedCollator":
        """Build from a tokenizer, falling back to ``eos_token_id`` then ``0``."""
        pad = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad, int):
            pad = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(pad, int):
            raise LifeMateError(
                "tokenizer has neither pad_token_id nor eos_token_id; "
                "set tokenizer.pad_token before training (see models/loader._ensure_pad_token)"
            )
        return cls(pad_token_id=pad, pad_to_multiple_of=pad_to_multiple_of)

    def _width(self, longest: int) -> int:
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 0:
            multiple = int(self.pad_to_multiple_of)
            return max(multiple, -(-longest // multiple) * multiple)
        return longest

    def __call__(self, examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Return padded lists (no tensors; see :func:`to_torch_batch`)."""
        kept = [dict(ex) for ex in examples if ex.get("kept", True)]
        if not kept:
            raise LifeMateError("collator received no usable examples (all dropped)")
        width = self._width(max(len(ex["input_ids"]) for ex in kept))
        batch: dict[str, Any] = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in kept:
            ids = list(example["input_ids"])
            mask = list(example.get("attention_mask") or [1] * len(ids))
            labels = list(example["labels"])
            if not (len(ids) == len(mask) == len(labels)):
                raise LifeMateError(
                    f"example length mismatch: ids={len(ids)} mask={len(mask)} labels={len(labels)}"
                )
            pad = width - len(ids)
            batch["input_ids"].append(ids + [self.pad_token_id] * pad)
            batch["attention_mask"].append(mask + [0] * pad)
            batch["labels"].append(labels + [-100] * pad)
        return batch

def to_torch_batch(batch: Mapping[str, Any], device: Any = None) -> dict[str, Any]:
    """Convert a padded list batch to torch tensors (lazy torch import)."""
    try:
        import torch  # noqa: PLC0415 - lazy on purpose
    except Exception as exc:  # pragma: no cover - torch absent
        raise LifeMateError(f"torch is required to build tensor batches: {exc}") from exc
    tensors = {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }
    if device is not None:
        tensors = {key: value.to(device) for key, value in tensors.items()}
    return tensors
