"""Generation wrappers: the real HF sampler and duck-typed test doubles.

``HFGenerator`` is the only place in the package that touches
``transformers`` at call time. Everything else depends on the duck type
``.generate(messages: list[dict]) -> str``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from lifemate.errors import LifeMateError
from lifemate.training.collator import apply_chat_template_text

__all__ = [
    "CallableGenerator",
    "GenerationStats",
    "HFGenerator",
    "STOP_MARKER",
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
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (``stop_marker_hit`` is cumulative over calls)."""
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "stop_marker_hit": self.stop_marker_hit,
            "truncated_post_hoc": self.truncated_post_hoc,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


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
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool | None = None,
        device: Any = None,
        stop_on_action: bool = True,
    ) -> None:
        if model is None or tokenizer is None:
            raise LifeMateError("HFGenerator requires both a model and a tokenizer")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise LifeMateError(
                "tokenizer has no apply_chat_template(); use a chat/instruct tokenizer "
                "(e.g. Qwen3.5-4B-Instruct) or build the prompt text yourself"
            )
        self.model = model
        self.tokenizer = tokenizer
        # temperature==0 is greedy: do_sample must stay False or decode is invalid.
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample) if do_sample is not None else self.temperature > 0
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.stop_on_action = bool(stop_on_action)
        self.stats = GenerationStats()
        try:
            import torch
            if str(self.device or getattr(self.model, "device", "")).startswith("cpu") and torch.get_num_threads() > 4:
                torch.set_num_threads(4)
        except Exception:
            pass
        self._last_stopping: Any = None  # keeps the criterion alive during decode

    # ------------------------------------------------------------------ chat
    def build_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        """Chat-template the messages with a generation prompt appended.

        Identical bytes to :func:`lifemate.training.collator.render_prompt_text`
        for the same messages (asserted by tests): both render with
        ``enable_thinking=False`` so the assistant turn never starts inside an
        open `` thinking`` block.
        """
        try:
            return apply_chat_template_text(self.tokenizer, list(messages))
        except Exception as exc:
            raise LifeMateError(
                "tokenizer.apply_chat_template failed; the tokenizer has no usable chat "
                f"template for these messages: {exc}"
            ) from exc

    # -------------------------------------------------------------- generate
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        """Generate exactly one assistant turn (text, not token ids)."""
        import time

        import torch

        prompt = self.build_prompt(messages)
        started = time.monotonic()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.device or getattr(self.model, "device", None) or "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])

        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if self.do_sample:
            kwargs["temperature"] = max(self.temperature, 1e-5)
        stopping = self._stopping_criteria()
        if stopping is not None:
            kwargs["stopping_criteria"] = stopping

        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)

        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text, hit = truncate_at_marker(text) if self.stop_on_action else (text, False)

        self.stats.n_calls += 1
        self.stats.prompt_tokens += prompt_len
        self.stats.generated_tokens += int(new_tokens.shape[-1])
        self.stats.stop_marker_hit = self.stats.stop_marker_hit or hit
        self.stats.truncated_post_hoc = self.stats.truncated_post_hoc or (hit and stopping is None)
        self.stats.elapsed_seconds += time.monotonic() - started
        return text

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
