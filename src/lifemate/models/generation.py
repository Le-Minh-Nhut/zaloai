"""Generation wrappers: the real HF sampler and duck-typed test doubles.

``HFGenerator`` is the only place in the package that touches
``transformers`` at call time. Everything else depends on the duck type
``.generate(messages: list[dict]) -> str``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from lifemate.errors import LifeMateError
from lifemate.models.profiles import ModelProfile, profile_for_model, profile_for_tokenizer, resolve_profile
from lifemate.training.collator import apply_chat_template_text, eos_token_ids

__all__ = [
    "CallableGenerator",
    "GenerationStats",
    "HFGenerator",
    "STOP_MARKER",
    "resolve_stop_ids",
    "stop_on_action_criteria",
    "truncate_at_marker",
]

#: Stop generation as soon as this appears: one decision per turn, always.
STOP_MARKER = "</action>"


@dataclass
class GenerationStats:
    """Token/time accounting of one ``generate`` call."""

    n_calls: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    stop_marker_hit: bool = False
    truncated_post_hoc: bool = False
    #: Set when an output was cut off by ``max_new_tokens`` before the stop
    #: marker appeared. Such an output is rejected, never parsed as an action.
    truncated_by_budget: bool = False
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (``stop_marker_hit`` is cumulative over calls)."""
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "stop_marker_hit": self.stop_marker_hit,
            "truncated_post_hoc": self.truncated_post_hoc,
            "truncated_by_budget": self.truncated_by_budget,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

def resolve_stop_ids(
    tokenizer: Any, model: Any = None, profile: ModelProfile | None = None
) -> list[int]:
    """Every id decoding must stop on, in first-seen order.

    Sources: the tokenizer, then the checkpoint's own ``generation_config`` and
    ``config`` (MiniCPM declares ``[1, 130073]`` there), then the profile's
    defensive extras. Stopping on only the tokenizer's single id would let a
    second EOS id decode into visible text and derail parsing.
    """
    ordered: list[int] = []
    for source in (tokenizer, getattr(model, "generation_config", None), getattr(model, "config", None)):
        for value in eos_token_ids(source):
            if value not in ordered:
                ordered.append(value)
    active = profile or profile_for_model(model) or profile_for_tokenizer(tokenizer)
    for value in active.speculative_stop_ids:
        if int(value) not in ordered:
            ordered.append(int(value))
    return ordered


def truncate_at_marker(text: str, marker: str = STOP_MARKER) -> tuple[str, bool]:
    """Keep everything up to and including the first ``marker`` occurrence.

    Returns ``(text, hit)``; ``hit`` is False when the marker never appears, in
    which case the caller has a truncated decision to reject downstream.
    """
    index = text.find(marker)
    if index == -1:
        return text, False
    return text[: index + len(marker)], True


def stop_on_action_criteria(tokenizer: Any, marker: str = STOP_MARKER) -> Any:
    """``StoppingCriteriaList`` that aborts decoding right after ``marker``.

    Returns ``None`` when transformers is unavailable (post-hoc truncation then
    applies) or when the marker encodes to no tokens.
    """
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:  # pragma: no cover - transformers absent
        return None
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if not marker_ids:
        return None

    class StopOnActionCriteria(StoppingCriteria):  # type: ignore[misc]
        """Stop as soon as the tail of the sequence is the marker's ids."""

        def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
            tail = input_ids[0][-len(marker_ids) :].tolist()
            return tail == list(marker_ids)

    return StoppingCriteriaList([StopOnActionCriteria()])


class HFGenerator:
    """Greedy/chat-templated generator over a loaded ``ModelBundle``.

    Early stop: a ``StoppingCriteria`` aborts sampling at ``</action>`` when
    transformers is available, so we never spend tokens after the decision.
    With a tokenizer that cannot stop mid-sequence the same truncation is
    applied post-hoc and flagged in :attr:`stats`.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        do_sample: bool | None = None,
        device: Any = None,
        stop_on_action: bool = True,
        top_p: float | None = None,
        profile: ModelProfile | None = None,
        model_name: str | None = None,
        constrained: bool = False,
        allowed_tools: Iterable[str] | None = None,
    ) -> None:
        if model is None or tokenizer is None:
            raise LifeMateError("HFGenerator requires both a model and a tokenizer")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise LifeMateError(
                "tokenizer has no apply_chat_template(); use a chat/instruct tokenizer "
                "(e.g. openbmb/MiniCPM5-2B, Qwen/Qwen3.5-4B) or build the prompt text yourself"
            )
        self.model = model
        self.tokenizer = tokenizer
        # ``model_name`` (usually cfg.model.name) is authoritative when given: a
        # checkpoint directory carries no family name in its tokenizer metadata.
        if profile is not None:
            self.profile = profile
        elif model_name:
            self.profile = resolve_profile(model_name)
        else:
            guessed = profile_for_model(model)
            self.profile = guessed if guessed.family != "generic" else profile_for_tokenizer(tokenizer)
        # temperature==0 is greedy: do_sample must stay False or decode is invalid.
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample) if do_sample is not None else self.temperature > 0
        self.top_p = float(top_p) if top_p is not None else None
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise LifeMateError(f"top_p must be in (0, 1], got {top_p!r}")
        self.max_new_tokens = int(
            max_new_tokens if max_new_tokens is not None else self.profile.default_max_new_tokens
        )
        self.device = device
        self.stop_on_action = bool(stop_on_action)
        # Action-grammar masking (features.decoding.constrained_decoding_enabled).
        # ``allowed_tools`` is PER SAMPLE — each sample has its own installed apps
        # and registry view, while the generator is built once — hence the setter
        # below. An EMPTY allow-list is legitimate for a finish-only sample: the
        # grammar then permits only ``{"type": "finish"}``. It is not an error.
        self.constrained = bool(constrained)
        self.allowed_tools: tuple[str, ...] = tuple(allowed_tools or ())
        self.stop_ids = resolve_stop_ids(tokenizer, model, self.profile)
        self.stats = GenerationStats()
        try:
            import torch
            if str(self.device or getattr(self.model, "device", "")).startswith("cpu") and torch.get_num_threads() > 4:
                torch.set_num_threads(4)
        except Exception:
            pass
        self._last_stopping: Any = None  # keeps the criterion alive during decode

    def set_allowed_tools(self, tools: Iterable[str] | None) -> None:
        """Set the tool names the constrained grammar may emit (per sample)."""
        self.allowed_tools = tuple(str(name) for name in (tools or ()) if str(name).strip())

    # ------------------------------------------------------------------ chat
    def build_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        """Chat-template the messages with a generation prompt appended.

        Identical bytes to :func:`lifemate.training.collator.render_prompt_text`
        for the same messages and profile (asserted by tests): both pass the
        profile's ``enable_thinking=False`` so the assistant turn never starts
        inside an open ``think`` block.
        """
        try:
            return apply_chat_template_text(self.tokenizer, list(messages), profile=self.profile)
        except Exception as exc:
            raise LifeMateError(
                "tokenizer.apply_chat_template failed; the tokenizer has no usable chat "
                f"template for these messages: {exc}"
            ) from exc

    # -------------------------------------------------------------- generate
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        """Generate exactly one assistant turn (text, not token ids).

        Raises :class:`LifeMateError` when the decode was cut off by
        ``max_new_tokens`` before ``</action>`` appeared: a truncated decision
        must never be parsed into an apparently-valid action.
        """
        import time

        import torch

        prompt = self.build_prompt(messages)
        started = time.monotonic()
        # ``add_special_tokens=False`` when the chat template already emits BOS:
        # measured 2x token id 0 on MiniCPM with the default (template emits <s>
        # and the tokenizer prepends another one).
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.profile.prompt_add_special_tokens,
        )
        device = self.device or getattr(self.model, "device", None) or "cpu"
        inputs = self._to_device(inputs, device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        eos_ids = list(self.stop_ids)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            # Stop on ALL declared EOS ids (MiniCPM has two); a single int keeps
            # the previous kwargs byte-identical for single-EOS checkpoints.
            "eos_token_id": eos_ids[0] if len(eos_ids) == 1 else eos_ids,
        }
        if self.do_sample:
            kwargs["temperature"] = max(self.temperature, 1e-5)
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        stopping = self._stopping_criteria()
        if stopping is not None:
            kwargs["stopping_criteria"] = stopping

        with torch.no_grad():
            if self.constrained:
                # Exactly the same kwargs as the unconstrained call below; the
                # grammar only adds a logits processor. An empty allow-list is
                # passed through: the grammar then permits finish only.
                from lifemate.models.constrained import constrained_generate  # noqa: PLC0415

                output = constrained_generate(
                    self.model,
                    self.tokenizer,
                    inputs,
                    enabled=True,
                    allowed_tools=self.allowed_tools,
                    **kwargs,
                )
            else:
                output = self.model.generate(**inputs, **kwargs)

        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text, hit = truncate_at_marker(text) if self.stop_on_action else (text, False)

        n_new = int(new_tokens.shape[-1])
        self.stats.n_calls += 1
        self.stats.prompt_tokens += prompt_len
        self.stats.generated_tokens += n_new
        self.stats.stop_marker_hit = self.stats.stop_marker_hit or hit
        self.stats.truncated_post_hoc = self.stats.truncated_post_hoc or (hit and stopping is None)
        self.stats.elapsed_seconds += time.monotonic() - started

        # A decision cut off by the token budget must fail loudly instead of
        # reaching the parser: a half-written JSON body can still look like a
        # valid action to a lenient reader, and executing it is worse than a
        # visible error. Only checked when we asked for the stop marker; a
        # deliberately unbounded decode (stop_on_action=False) is not a decision.
        if self.stop_on_action and not hit and n_new >= self.max_new_tokens:
            self.stats.truncated_by_budget = True
            raise LifeMateError(
                f"generation hit the {self.max_new_tokens}-token budget without emitting "
                f"{STOP_MARKER} ({n_new} new tokens); refusing to parse a truncated decision"
            )
        return text

    def _to_device(self, inputs: Any, device: Any) -> Any:
        """Move tokenized inputs to ``device`` (no-op for 4-bit/placement maps)."""
        if getattr(self.model, "hf_device_map", None):
            # Accelerate already placed the layers; moving the batch is still fine
            # because generate() reads the embedding device from the map.
            target = next(iter(self.model.hf_device_map.values()), device)
            return {key: value.to(target) for key, value in inputs.items()}
        return {key: value.to(device) for key, value in inputs.items()}

    # --------------------------------------------------------------- helpers
    def _stopping_criteria(self) -> Any:
        """``StoppingCriteriaList`` aborting at ``</action>``, else ``None``.

        ``None`` means the marker is applied post-hoc by :func:`truncate_at_marker`
        and ``GenerationStats.truncated_post_hoc`` records that.
        """
        if not self.stop_on_action:
            return None
        stopping = stop_on_action_criteria(self.tokenizer)
        self._last_stopping = stopping  # keep the criterion alive during decode
        return stopping


class CallableGenerator:
    """Wrap any ``f(messages) -> str`` as a generator (tests, ablations)."""

    def __init__(self, fn: Callable[[list[dict[str, str]]], str]) -> None:
        if not callable(fn):
            raise ValueError("CallableGenerator requires a callable")
        self._fn = fn
        self.calls: list[list[dict[str, str]]] = []
        self.stats = GenerationStats()

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Call the wrapped function; raises ``LifeMateError`` on a non-str result."""
        self.calls.append([dict(m) for m in messages])
        self.stats.n_calls += 1
        out = self._fn(messages)
        if not isinstance(out, str):
            raise LifeMateError(
                f"generator callable returned {type(out).__name__}, expected str"
            )
        return out
