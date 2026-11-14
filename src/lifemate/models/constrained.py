"""Constrained decoding for the LifeMate action grammar, via transformers logits processors.

No vllm/sglang/outlines: one ``transformers.LogitsProcessor`` that masks the next
token to the legal continuations of the frozen action format::

    <action>{"type": "finish"}</action>
    <action>{"type": "tool_call", "name": "<allowed.tool>", "arguments": {...}}</action>

What is constrained, and what is deliberately not
-------------------------------------------------
Forced: the ``<action>`` prefix, ``{"type": "``, the type literal, the
``", "name": "`` glue, the tool name (restricted to ``allowed_tools``), the
``", "arguments": `` glue, the opening ``{`` of the arguments object, and the
closing ``</action>`` once that object closes.

NOT constrained: the inside of the arguments object. A full JSON grammar masker is
a large machine and the wrong owner of that decision — arguments are semantic
values the tool schema must validate, so the guardrail stays authoritative. The
processor only tracks brace/string/escape state of the decoded text to know when
the object has closed.

Two safety rules, both tested:

* An unsafe step never masks: if the next-token set cannot be proven safe, the
  logits are left untouched rather than forcing a wrong literal.
* A mask is never all-``-inf``: :func:`apply_mask` detects that, logs it and skips
  masking for that step, so generation can never be poisoned into NaN.

``enabled=False`` → :func:`constrained_generate` calls ``model.generate`` with the
caller's kwargs untouched and no processor attached.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ACTION_CLOSE_TEXT",
    "ACTION_OPEN_TEXT",
    "ActionGrammarLogitsProcessor",
    "GrammarState",
    "PrefixTrie",
    "apply_mask",
    "build_prefix_trie",
    "cached_piece_map",
    "constrained_generate",
    "token_piece_map",
]

ACTION_OPEN_TEXT = "<action>"
ACTION_CLOSE_TEXT = "</action>"
TYPE_GLUE_OPEN = '{"type": "'
TYPE_FINISH = 'finish"}'
TYPE_TOOL_CALL = 'tool_call", "name": "'
ARGS_GLUE = '", "arguments": '
OBJECT_OPEN = "{"
#: Prefix a chat template may already have closed; tolerated before ``<action>``.
THINK_CLOSE = "</think>"

#: Cap on how far a single token may reach into the upcoming forced literals.
_CONTINUATION_CHARS = 64


# ---------------------------------------------------------------------------
# Token pieces
# ---------------------------------------------------------------------------
def _require_tokenizer_support(tokenizer: Any) -> tuple[Any, Any]:
    """Tokenizer pieces as ``(convert_ids_to_tokens, convert_tokens_to_string)``."""
    ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(ids_to_tokens):
        raise TypeError(
            "constrained decoding needs a tokenizer with convert_ids_to_tokens(); "
            f"{type(tokenizer).__name__} has none. Pass an HF tokenizer that exposes its "
            "vocabulary, or turn the feature off "
            "(features.decoding.constrained_decoding_enabled=false / "
            "HFGenerator(constrained=False)) to leave decoding unconstrained."
        )
    tokens_to_string = getattr(tokenizer, "convert_tokens_to_string", None)
    if not callable(tokens_to_string):
        raise TypeError(
            "constrained decoding needs a tokenizer with convert_tokens_to_string(); "
            f"{type(tokenizer).__name__} has none. Pass an HF tokenizer, or turn the feature "
            "off (features.decoding.constrained_decoding_enabled=false / "
            "HFGenerator(constrained=False))."
        )
    return ids_to_tokens, tokens_to_string


def token_piece_map(tokenizer: Any, vocab_size: int | None = None) -> list[str]:
    """Decoded surface text of every token id in ``[0, vocab_size)``.

    One ``convert_tokens_to_string`` call per entry: ~0.1 s for a 130k-token
    vocabulary (measured on the local MiniCPM5-2B tokenizer), paid once per
    processor rather than once per decode step.
    """
    ids_to_tokens, tokens_to_string = _require_tokenizer_support(tokenizer)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    pieces: list[str] = []
    for index in range(int(vocab_size)):
        try:
            piece = tokens_to_string([ids_to_tokens(index)])
        except Exception:  # noqa: BLE001 - an undecodable id cannot be legal here
            piece = ""
        pieces.append(str(piece or ""))
    return pieces


# ---------------------------------------------------------------------------
# Grammar as an explicit small FSM over forced literals + one free tail
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Literal:
    text: str


@dataclass(frozen=True)
class _Choice:
    """One of several literals; each candidate is ``(text, next_step)``."""

    candidates: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _FreeJson:
    """Unconstrained until the arguments object AND its enclosing action object close."""


@dataclass
class GrammarState:
    """Legal next-token set plus the literal it is working through.

    ``allowed_ids is None`` means "this processor cannot prove anything here", and
    the caller must leave the logits alone. ``literal`` is kept for diagnostics.
    """

    allowed_ids: frozenset[int] | None = None
    literal: str = ""
    free_json: bool = False

    @property
    def constrained(self) -> bool:
        return self.allowed_ids is not None


class PrefixTrie:
    """Legal-continuation automaton for one sample's action grammar.

    The forced part of the grammar is a chain of literal steps; tool names form a
    choice step, and the arguments object is the single free branch. State is
    always recomputed from the decoded text, so no hidden mutable cursor can drift
    from what the model actually emitted.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        tokenizer: Any,
        allowed_actions: Sequence[dict[str, Any]] | None = None,
        allowed_tools: Iterable[str] = (),
        pieces: Sequence[str] | None = None,
    ) -> None:
        if vocab_size is None or int(vocab_size) <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size!r}")
        self.vocab_size = int(vocab_size)
        self.tokenizer = tokenizer
        self.pieces = (
            list(pieces) if pieces is not None else token_piece_map(tokenizer, self.vocab_size)
        )
        self.allowed_tools = tuple(dict.fromkeys(str(t) for t in allowed_tools if str(t).strip()))

        actions = [a for a in (allowed_actions or []) if isinstance(a, dict)]
        if not actions:
            # No sample vocabulary: the generic grammar (both kinds, tools from the allow-list).
            self.finish_allowed = True
            self.sample_tools: tuple[str, ...] = ()
            self.tool_call_allowed = bool(self.allowed_tools)
        else:
            kinds = {str(a.get("type") or "").strip() for a in actions}
            self.finish_allowed = "finish" in kinds
            self.sample_tools = tuple(
                dict.fromkeys(
                    str(a.get("name")).strip()
                    for a in actions
                    if str(a.get("type") or "").strip() == "tool_call" and a.get("name")
                )
            )
            self.tool_call_allowed = bool(self.sample_tools)

        # The tool set actually emit-able: the sample's vocabulary, further
        # intersected with the caller's allow-list when one is given. A tool in
        # neither list can never be produced.
        if self.allowed_tools:
            pool = self.sample_tools or self.allowed_tools
            self.tools = tuple(tool for tool in pool if tool in self.allowed_tools)
        else:
            self.tools = self.sample_tools
        if self.tool_call_allowed and not self.tools:
            self.tool_call_allowed = False

        self.steps = self._build_steps()
        self._ids_cache: dict[str, frozenset[int]] = {}
        self._piece_index: dict[str, list[int]] | None = None
        # Once the whole grammar is satisfied the decode is finished; only EOS may
        # follow, so a stray token cannot continue past the closing tag.
        eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = frozenset({int(eos)}) if isinstance(eos, int) else frozenset()

    # -- grammar ------------------------------------------------------------
    def _build_steps(self) -> list[Any]:
        """``<action>{\"type\": \"`` → choice → (finish | tool) → free → ``</action>``."""
        gluing = 4 if self.tool_call_allowed else 3
        finish_end = 3
        candidate_list: list[tuple[str, int]] = []
        if self.finish_allowed:
            candidate_list.append((TYPE_FINISH, finish_end))
        if self.tool_call_allowed:
            candidate_list.append((TYPE_TOOL_CALL, gluing))
        steps: list[Any] = [
            _Literal(ACTION_OPEN_TEXT),
            _Literal(TYPE_GLUE_OPEN),
            _Choice(tuple(candidate_list)) if candidate_list else _FreeJson(),
            _Literal(ACTION_CLOSE_TEXT),
            # Tool names only: the choice must not swallow the closing quote that
            # the next literal starts with, or the walk cannot resynchronise.
            _Choice(tuple((tool, 5) for tool in self.tools)),
            # ``", "arguments": {`` — the opening brace belongs to the forced part,
            # so the free span starts INSIDE the object and its depth is 1.
            _Literal(ARGS_GLUE + OBJECT_OPEN),
            _FreeJson(),
            _Literal(ACTION_CLOSE_TEXT),
        ]
        return steps

    def type_choices(self) -> tuple[str, ...]:
        step = self.steps[2]
        if isinstance(step, _Choice):
            return tuple(text for text, _ in step.candidates)
        return ()

    # -- id lookup ---------------------------------------------------------
    def _index(self) -> dict[str, list[int]]:
        """Exact piece → ids. Empty pieces (special tokens) are never legal here."""
        if self._piece_index is None:
            index: dict[str, list[int]] = {}
            for token_id, piece in enumerate(self.pieces):
                if piece:
                    index.setdefault(piece, []).append(token_id)
            self._piece_index = index
        return self._piece_index

    def ids_completing(self, continuation: str) -> frozenset[int]:
        """Ids whose decoded piece is a non-empty prefix of ``continuation``.

        A token may span several forced literals (``'"type": "finish"}'``), so the
        continuation, not just the current literal, is tested.
        """
        if not continuation:
            return frozenset()
        cached = self._ids_cache.get(continuation)
        if cached is not None:
            return cached
        found: set[int] = set()
        for piece, ids in self._index().items():
            if continuation.startswith(piece):
                found.update(ids)
        result = frozenset(found)
        self._ids_cache[continuation] = result
        return result

    def ids_for_choices(self, remainders: Sequence[str]) -> frozenset[int]:
        """Ids that are a prefix of at least one candidate remainder."""
        found: set[int] = set()
        for remainder in remainders:
            found |= self.ids_completing(remainder)
        return frozenset(found)

    # -- state -------------------------------------------------------------
    def state_for(self, text_so_far: str) -> GrammarState:
        """Legal next-token set after ``text_so_far`` (unconstrained when unsure)."""
        text = self._tolerated_prefix(text_so_far)
        if ACTION_CLOSE_TEXT in text:
            # A complete action exists: constrain to EOS only (never "correct" it,
            # and never let a stray token continue past the closing tag).
            return GrammarState(allowed_ids=self.eos_ids or None)
        return self._walk(text, start=0, consumed=0)

    @staticmethod
    def _tolerated_prefix(text: str) -> str:
        """Drop an already-closed think block and leading whitespace.

        The chat template emits the closed-think suffix in the *prompt*, so a
        well-behaved decode starts at ``<action>``; a model that re-emits
        ``</think>`` or leading spaces first is still constrained afterwards
        instead of losing the mask for the whole turn.
        """
        trimmed = text.lstrip()
        while trimmed.startswith(THINK_CLOSE):
            trimmed = trimmed[len(THINK_CLOSE):].lstrip()
        return trimmed

    def _walk(self, text: str, *, start: int, consumed: int) -> GrammarState:
        steps = self.steps
        index = start
        offset = consumed
        while index < len(steps):
            step = steps[index]
            available = text[offset:]
            if isinstance(step, _FreeJson):
                # The free span starts INSIDE the arguments object, while the
                # enclosing action object is still open too, so depth 2 must return
                # to 0: the frozen format closes both (``...{}}</action>``).
                scan = _JsonScan.scan(available, start_depth=2)
                if scan.closed and scan.close_index:
                    # Resume the walk right AFTER the closing brace: any text the
                    # model already emitted past it must be matched against what
                    # follows (re-forcing the literal would emit it twice).
                    index += 1
                    offset += scan.close_index
                    continue
                return GrammarState(free_json=True)
            if isinstance(step, _Literal):
                if available.startswith(step.text):
                    index += 1
                    offset += len(step.text)
                    continue
                if step.text.startswith(available):
                    return GrammarState(
                        allowed_ids=self.ids_completing(
                            self._continuation(index, offset, available)
                        )
                        or None,
                        literal=step.text[len(available):],
                    )
                return GrammarState()  # diverged from the forced grammar
            # _Choice: a candidate is matched when the available text STARTS with
            # it (one iteration may consume several literals at once).
            matched = next(
                ((candidate, next_step) for candidate, next_step in step.candidates
                 if available.startswith(candidate)),
                None,
            )
            if matched is not None:
                index = matched[1]
                offset += len(matched[0])
            else:
                remainders = [
                    candidate[len(available):]
                    for candidate, _ in step.candidates
                    if candidate.startswith(available)
                ]
                if not remainders:
                    return GrammarState()
                return GrammarState(
                    allowed_ids=self.ids_for_choices(remainders) or None,
                    literal=remainders[0] if len(remainders) == 1 else "",
                )
            continue
        # Grammar complete: the only legal continuation is end-of-sequence.
        return GrammarState(allowed_ids=self.eos_ids or None)

    def _continuation(self, index: int, offset: int, available: str) -> str:
        """Remaining literal + as much of the following forced text as is certain.

        Lets one token span stage boundaries (a tokenizer piece such as
        ``'"type": "finish"}'``) while never guessing across a choice or into the
        free JSON span.
        """
        pieces = [self.steps[index].text[len(available):]]
        for later in self.steps[index + 1:]:
            if isinstance(later, _Literal):
                pieces.append(later.text)
            elif isinstance(later, _Choice):
                pieces.append(_common_prefix([text for text, _ in later.candidates]))
                break
            else:
                break
            if sum(len(piece) for piece in pieces) >= _CONTINUATION_CHARS:
                break
        return "".join(pieces)[:_CONTINUATION_CHARS]


def _common_prefix(strings: Sequence[str]) -> str:
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for index, char in enumerate(shortest):
        if any(other[index] != char for other in strings):
            return shortest[:index]
    return shortest


#: ``(id(tokenizer), vocab_size) -> (tokenizer, pieces)``. The piece table costs
#: ~0.1 s and 130k strings to build, and the generator is built once while
#: :func:`constrained_generate` runs per sample — so it is built once and reused.
#: The tokenizer itself is kept in the entry, which also prevents ``id()`` reuse.
_PIECES_CACHE: dict[tuple[int, int], tuple[Any, list[str]]] = {}


def cached_piece_map(tokenizer: Any, vocab_size: int) -> list[str]:
    """Token piece table for ``tokenizer``, built once per (tokenizer, vocab size)."""
    key = (id(tokenizer), int(vocab_size))
    hit = _PIECES_CACHE.get(key)
    if hit is not None:
        return hit[1]
    pieces = token_piece_map(tokenizer, vocab_size)
    _PIECES_CACHE[key] = (tokenizer, pieces)
    return pieces


def build_prefix_trie(
    vocab_size: int,
    tokenizer: Any,
    allowed_actions: Sequence[dict[str, Any]] | None = None,
    *,
    allowed_tools: Iterable[str] = (),
    pieces: Sequence[str] | None = None,
) -> PrefixTrie:
    """Legal-token automaton for one sample's action grammar.

    ``allowed_actions`` is the sample's action vocabulary (``{"type": "finish"}``
    and/or ``{"type": "tool_call", "name": ...}`` entries). ``allowed_tools``
    further restricts which tool names may be emitted at all.
    """
    return PrefixTrie(
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        allowed_actions=allowed_actions,
        allowed_tools=allowed_tools,
        pieces=pieces,
    )


# ---------------------------------------------------------------------------
# Brace tracking (free JSON tail)
# ---------------------------------------------------------------------------
@dataclass
class _JsonScan:
    depth: int = 0
    in_string: bool = False
    escaped: bool = False
    started: bool = False
    #: Characters consumed up to and including the brace that closed THIS object
    #: (0 while it is still open).
    close_index: int = 0

    @classmethod
    def scan(cls, text: str, *, start_depth: int = 0) -> _JsonScan:
        # A non-zero start depth means the object's opening brace was part of the
        # forced prefix, so the object has already started.
        state = cls(depth=int(start_depth), started=bool(start_depth))
        for index, char in enumerate(text):
            state.feed(char)
            closed_here = char == "}" and state.depth == 0 and not state.in_string
            if closed_here and state.close_index == 0:
                state.close_index = index + 1
        return state

    def feed(self, char: str) -> None:
        if self.escaped:
            self.escaped = False
            return
        if self.in_string:
            if char == "\\":
                self.escaped = True
            elif char == '"':
                self.in_string = False
            return
        if char == '"':
            self.in_string = True
        elif char == "{":
            self.depth += 1
            self.started = True
        elif char == "}":
            self.depth = max(0, self.depth - 1)

    @property
    def closed(self) -> bool:
        return bool(self.started and self.depth == 0 and not self.in_string and self.close_index)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def apply_mask(scores: Any, allowed: frozenset[int], *, step: int | None = None) -> Any:
    """Mask ``scores`` to ``allowed``; return it untouched when that is unsafe.

    Never writes an all-``-inf`` row: an empty/out-of-range allowed set, or a mask
    that leaves nothing finite, is logged and skipped.
    """
    width = int(scores.shape[-1])
    legal = sorted(index for index in allowed if 0 <= index < width)
    if not legal:
        logger.warning(
            "constrained decoding: no legal token at step %s; leaving logits unmasked "
            "(an all -inf row would poison generation)",
            step,
        )
        return scores
    import torch  # noqa: PLC0415 - only reached when a real processor runs

    masked = torch.full_like(scores, float("-inf"))
    masked[..., legal] = scores[..., legal]
    if not bool(torch.isfinite(masked).any()):
        logger.warning(
            "constrained decoding: masking to %d token(s) left no finite logit at step %s; "
            "skipping the mask",
            len(legal),
            step,
        )
        return scores
    return masked


def _processor_base() -> type:
    try:
        from transformers import LogitsProcessor  # noqa: PLC0415 - optional dependency

        return LogitsProcessor
    except ImportError:  # pragma: no cover - stdlib-only interpreter

        class _Base:
            """Stand-in base when transformers is absent (never used for generation)."""

            supports_continuous_batching = None

            def __call__(self, input_ids: Any, scores: Any) -> Any:
                raise TypeError(
                    "transformers is not installed: ActionGrammarLogitsProcessor cannot be used "
                    "as a logits processor here. Install transformers or use enabled=False."
                )

        return _Base


class ActionGrammarLogitsProcessor(_processor_base()):  # type: ignore[misc,valid-type]
    """Mask logits so only tokens continuing the LifeMate action grammar appear."""

    def __init__(
        self,
        trie: PrefixTrie,
        tokenizer: Any,
        *,
        prompt_length: int = 0,
    ) -> None:
        # The trie already resolved the vocabulary (either through the tokenizer or
        # from explicit ``pieces``), so decoding new tokens needs no tokenizer
        # introspection here: ``trie.pieces`` is the fallback decoder.
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.steps = 0
        self.masked_steps = 0
        self.skipped_steps = 0
        self.last_state: GrammarState | None = None

    def decode_generated(self, token_ids: Sequence[int]) -> str:
        """Decode only the new tokens (never the prompt)."""
        decoder = getattr(self.tokenizer, "decode", None)
        if callable(decoder):
            try:
                return str(decoder(list(token_ids), skip_special_tokens=True))
            except TypeError:  # tokenizers without kwargs
                return str(decoder(list(token_ids)))
        return "".join(self.trie.pieces[int(t)] for t in token_ids if 0 <= int(t) < len(self.trie.pieces))

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        self.steps += 1
        generated = input_ids[0][self.prompt_length:].tolist()
        state = self.trie.state_for(self.decode_generated(generated))
        self.last_state = state
        if not state.constrained:
            return scores
        self.masked_steps += 1
        masked = apply_mask(scores, state.allowed_ids or frozenset(), step=self.steps)
        if masked is scores:
            self.skipped_steps += 1
        return masked


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def constrained_generate(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    *,
    enabled: bool = False,
    allowed_tools: Sequence[str] = (),
    allowed_actions: Sequence[dict[str, Any]] | None = None,
    max_new_tokens: int = 256,
    pieces: Sequence[str] | None = None,
    **gen_kwargs: Any,
) -> Any:
    """``model.generate`` with the action grammar masked, or byte-identically plain.

    ``enabled=False`` passes ``gen_kwargs`` straight through and attaches no
    processor. ``prompt`` may be tokenized inputs (anything ``model.generate``
    accepts); ``prompt_length`` is derived from its ``input_ids`` when possible so
    only newly generated tokens are inspected.
    """
    if not enabled:
        return _call_generate(model, prompt, max_new_tokens=max_new_tokens, **gen_kwargs)

    vocab_size = len(tokenizer)
    trie = build_prefix_trie(
        vocab_size,
        tokenizer,
        allowed_actions,
        allowed_tools=allowed_tools,
        pieces=pieces if pieces is not None else cached_piece_map(tokenizer, vocab_size),
    )
    prompt_length = 0
    input_ids = getattr(prompt, "input_ids", None)
    if input_ids is None and isinstance(prompt, dict):
        input_ids = prompt.get("input_ids")
    if input_ids is not None and hasattr(input_ids, "shape"):
        prompt_length = int(input_ids.shape[-1])

    processor = ActionGrammarLogitsProcessor(trie, tokenizer, prompt_length=prompt_length)

    from transformers import LogitsProcessorList  # noqa: PLC0415 - enabled path only

    existing = gen_kwargs.pop("logits_processor", None)
    processors = list(existing) if existing is not None else []
    processors.append(processor)
    return _call_generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList(processors),
        **gen_kwargs,
    )


def _call_generate(model: Any, prompt: Any, **kwargs: Any) -> Any:
    """Call ``generate`` the way transformers expects for this prompt type.

    transformers 5 ignores a mapping passed positionally (``inputs`` becomes the
    mapping itself and ``.shape`` fails), so tokenized inputs are splatted as
    keywords — the same call the caller would write by hand. Everything else
    (a tensor, ``input_ids=`` already in kwargs) is passed through untouched.
    """
    if isinstance(prompt, Mapping):
        return model.generate(**dict(prompt), **kwargs)
    return model.generate(prompt, **kwargs)
