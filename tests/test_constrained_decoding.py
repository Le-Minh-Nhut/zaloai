"""Constrained-decoding tests: grammar masking over a tiny vocabulary, torch-gated.

Two layers:

* the **grammar** (``PrefixTrie``) is pure Python over a hand-built piece table, so
  it is asserted on any interpreter;
* the **masking** (``apply_mask`` / ``ActionGrammarLogitsProcessor``) needs torch,
  and is skipped where torch is absent.

The grammar-only assertions use a tiny synthetic vocabulary whose pieces spell the
forced literals, so "which ids are legal here" is checkable by inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import pytest
except ImportError:
    pytest = None  # stdlib runner: assertions only

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.models.constrained import (  # noqa: E402
    ACTION_CLOSE_TEXT,
    ACTION_OPEN_TEXT,
    ActionGrammarLogitsProcessor,
    PrefixTrie,
    apply_mask,
    build_prefix_trie,
    constrained_generate,
    token_piece_map,
)

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if pytest is not None:
    _needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch unavailable")

# A tiny vocabulary: the literals are spelled out plus a few obvious distractors.
TINY_PIECES = [
    "<action>",          # 0
    '{"type": "',        # 1
    "finish",            # 2
    '"}',                # 3
    "tool_call",         # 4
    '", "name": "',      # 5
    "alarm.list",        # 6
    '", "arguments": ',  # 7
    "{",                 # 8
    "}",                 # 9
    "</action>",         # 10
    "hello",             # 11  (illegal at every forced step)
    '"',                 # 12
    "weather.get",       # 13
]

#: Ids that MUST be legal at a state, ids that MUST NOT be. A mask may also admit a
#: token that is only a partial piece of the same literal (e.g. a bare "{"), which
#: is still a legal continuation — so assertions are "must include" + "must exclude"
#: rather than exact set equality.
MUST = {
    "open": {0},
    "type": {1},
    "type_choice": {2, 4},
    "finish_tail": {3},
    "close": {10},
    "tool_name": {6, 13},
    "args": {7},
}
#: "hello" (11) never continues the forced grammar; a bare quote is only legal where
#: a quote is part of the next literal, so it is not in this global never-set.
NEVER = {11}


#: Same vocabulary plus one piece that spans the whole arguments glue, so a decode
#: can reach the free-JSON span in a single step.
FREE_PIECES = [*TINY_PIECES, '", "arguments": {']

_ARGS_GLUE_ID = len(FREE_PIECES) - 1


def _tiny_trie(
    tools: tuple[str, ...] = ("alarm.list", "weather.get"), pieces: list[str] | None = None
) -> PrefixTrie:
    table = list(pieces or TINY_PIECES)
    return build_prefix_trie(
        len(table),
        tokenizer=object(),  # pieces given explicitly; the tokenizer is unused
        allowed_actions=[
            {"type": "finish"},
            {"type": "tool_call", "name": "alarm.list"},
            {"type": "tool_call", "name": "weather.get"},
            {"type": "tool_call", "name": "evil.tool"},
        ],
        allowed_tools=tools,
        pieces=table,
    )


def _allowed(state) -> set[int]:
    return set(state.allowed_ids or ())


def _check(state, must: set[int], label: str) -> set[int]:
    allowed = _allowed(state)
    assert state.constrained, f"{label}: expected a constrained state"
    assert must <= allowed, f"{label}: missing legal ids {sorted(must - allowed)}"
    assert not (allowed & NEVER), f"{label}: illegal ids allowed {sorted(allowed & NEVER)}"
    return allowed


# ---------------------------------------------------------------------------
# Grammar (any interpreter)
# ---------------------------------------------------------------------------
def test_trie_allows_only_legal_continuations_at_each_state():
    trie = _tiny_trie()

    _check(trie.state_for(""), MUST["open"], "start")
    _check(trie.state_for(ACTION_OPEN_TEXT), MUST["type"], "type glue")
    _check(trie.state_for(ACTION_OPEN_TEXT + '{"type": "'), MUST["type_choice"], "type choice")
    _check(trie.state_for(ACTION_OPEN_TEXT + '{"type": "finish'), MUST["finish_tail"], "finish tail")

    finished = ACTION_OPEN_TEXT + '{"type": "finish"}'
    _check(trie.state_for(finished), MUST["close"], "finish close")

    tool_prefix = ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "'
    _check(trie.state_for(tool_prefix), MUST["tool_name"], "tool name")

    named = tool_prefix + "alarm.list"
    _check(trie.state_for(named), MUST["args"], "arguments glue")


def test_a_disallowed_tool_can_never_be_emitted():
    trie = _tiny_trie(tools=("alarm.list",))
    tool_prefix = ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "'
    allowed = _allowed(trie.state_for(tool_prefix))
    assert 6 in allowed
    # "weather.get" and the sample's "evil.tool" are both outside the allow-list.
    assert 13 not in allowed
    assert 13 not in allowed
    # The grammar's tool choice holds only the allow-listed name, so no state can
    # ever legalise another one — that is what the mask acts on.
    assert trie.tools == ("alarm.list",)
    for state_text in (
        ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "',
        ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "wea',
    ):
        state = trie.state_for(state_text)
        assert 13 not in _allowed(state), f"disallowed tool legal at {state_text!r}"


def test_arguments_object_is_free_then_the_close_tag_is_forced():
    trie = _tiny_trie()
    opened = ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "alarm.list", "arguments": {'
    state = trie.state_for(opened)
    assert state.free_json is True
    assert state.allowed_ids is None  # nothing is masked inside the arguments

    # A still-open object (even with a nested one) stays free.
    inner_open = opened + '"a": {"b": 1'
    assert trie.state_for(inner_open).free_json is True

    # Two `{` are already open at this point (the action object in the forced
    # prefix and the arguments object), so two `}` are needed to close both.
    # A nested object and a brace inside a string do not close them early.
    nested_but_open = opened + '"a": {"b": "}"}'
    assert trie.state_for(nested_but_open).free_json is True
    # One brace short: still free (the action object is still open).
    assert trie.state_for(nested_but_open + "}").free_json is True
    # Both closed → the closing tag is re-armed.
    both_closed = nested_but_open + "}}"
    assert trie.state_for(both_closed).free_json is False
    _check(trie.state_for(both_closed), MUST["close"], "both objects closed")

    # Flat arguments: the frozen format still closes both objects.
    closed = opened + '"a": 1}}'
    assert trie.state_for(closed).free_json is False
    _check(trie.state_for(closed), MUST["close"], "closed")


def test_illegal_text_makes_the_processor_stop_masking():
    trie = _tiny_trie()
    # A diverged decode must be left alone, never "corrected" into a literal.
    assert trie.state_for("The answer is").constrained is False
    assert trie.state_for(ACTION_OPEN_TEXT + '{"type": "weird').constrained is False


def test_a_complete_action_is_never_constrained_further():
    trie = _tiny_trie()
    done = ACTION_OPEN_TEXT + '{"type": "finish"}' + ACTION_CLOSE_TEXT
    assert trie.state_for(done).constrained is False


def test_finish_only_grammar_excludes_tool_calls():
    trie = build_prefix_trie(
        len(TINY_PIECES),
        object(),
        allowed_actions=[{"type": "finish"}],
        pieces=TINY_PIECES,
    )
    assert trie.type_choices() == ('finish"}',)
    assert _allowed(trie.state_for(ACTION_OPEN_TEXT + '{"type": "')) == {2}


def test_tool_only_grammar_excludes_finish():
    trie = build_prefix_trie(
        len(TINY_PIECES),
        object(),
        allowed_actions=[{"type": "tool_call", "name": "alarm.list"}],
        pieces=TINY_PIECES,
    )
    assert trie.type_choices() == ('tool_call", "name": "',)
    assert _allowed(trie.state_for(ACTION_OPEN_TEXT + '{"type": "')) == {4}


def test_finish_is_refused_when_the_allow_list_excludes_all_tools():
    """No legal action is left: the grammar must fall back, not mask to empty."""
    trie = build_prefix_trie(
        len(TINY_PIECES),
        object(),
        allowed_actions=[{"type": "tool_call", "name": "alarm.list"}],
        allowed_tools=("something.else",),
        pieces=TINY_PIECES,
    )
    assert trie.finish_allowed is False
    assert trie.tool_call_allowed is False
    assert trie.type_choices() == ()


def test_tokenizer_without_vocabulary_introspection_raises_clearly():
    class _Opaque:
        pass

    trie = _tiny_trie()
    assert trie.pieces == TINY_PIECES
    try:
        token_piece_map(_Opaque())
    except TypeError as exc:
        assert "convert_ids_to_tokens" in str(exc)
        # The message names the knob an operator turns off.
        assert "constrained_decoding_enabled=false" in str(exc)
        assert "HFGenerator(constrained=False)" in str(exc)
    else:
        raise AssertionError("a tokenizer with no vocabulary access must raise, not silently unconstrain")

    # Building a trie FROM an opaque tokenizer fails the same way (no pieces given).
    try:
        build_prefix_trie(len(TINY_PIECES), _Opaque())
    except TypeError as exc:
        assert "convert_ids_to_tokens" in str(exc)
    else:
        raise AssertionError("build_prefix_trie must not silently unconstrain with an opaque tokenizer")


def test_prefix_trie_validates_vocab_size():
    try:
        build_prefix_trie(0, object(), pieces=[])
    except ValueError as exc:
        assert "vocab_size" in str(exc)
    else:
        raise AssertionError("vocab_size must be positive")


# ---------------------------------------------------------------------------
# Masking (torch)
# ---------------------------------------------------------------------------
def test_no_step_produces_an_all_masked_row():
    if not HAVE_TORCH:
        return
    trie = _tiny_trie()
    walk = [
        "",
        ACTION_OPEN_TEXT,
        ACTION_OPEN_TEXT + '{"type": "',
        ACTION_OPEN_TEXT + '{"type": "finish',
        ACTION_OPEN_TEXT + '{"type": "finish"}',
        ACTION_OPEN_TEXT + '{"type": "finish"}' + ACTION_CLOSE_TEXT,
        ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "',
        ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "alarm.list',
        ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "alarm.list", "arguments": {',
    ]
    for text in walk:
        state = trie.state_for(text)
        scores = torch.zeros(1, len(TINY_PIECES))
        masked = apply_mask(scores, state.allowed_ids or frozenset(), step=1)
        assert torch.isfinite(masked).any(), f"all-masked row at {text!r}"
        if state.constrained and state.allowed_ids:
            finite = set(torch.isfinite(masked[0]).nonzero().flatten().tolist())
            # The mask is exactly the allowed set intersected with the row width.
            assert finite == {i for i in _allowed(state) if i < len(TINY_PIECES)}, (
                f"mask mismatch at {text!r}"
            )


def test_empty_allowed_set_never_masks():
    if not HAVE_TORCH:
        return
    scores = torch.zeros(1, len(TINY_PIECES))
    # Everything out of range: the row must come back untouched, not all -inf.
    assert apply_mask(scores, frozenset({999, 1000}), step=3) is scores
    assert torch.isfinite(scores).all()
    # Only -inf legal logits: also skipped, rather than producing a NaN row.
    dead = torch.full((1, len(TINY_PIECES)), float("-inf"))
    assert apply_mask(dead, frozenset({0}), step=4) is dead


def test_processor_masks_a_forced_bad_token():
    if not HAVE_TORCH:
        return
    trie = _tiny_trie()
    # All four ids are the prompt, so nothing has been generated yet: the grammar
    # demands the "<action>" literal and must veto the model's chosen "hello".
    processor = ActionGrammarLogitsProcessor(trie, object(), prompt_length=4)
    input_ids = torch.tensor([[5, 6, 7, 11]])
    scores = torch.zeros(1, len(TINY_PIECES))
    scores[0, 11] = 50.0  # the model really wants the illegal token
    masked = processor(input_ids, scores)
    assert not torch.isfinite(masked[0, 11])
    finite = set(torch.isfinite(masked[0]).nonzero().flatten().tolist())
    assert MUST["open"] <= finite and not (finite & NEVER)
    assert processor.masked_steps == 1 and processor.skipped_steps == 0

    # After "<action>" is generated (one new id past the 4-id prompt), only the
    # JSON glue is legal.
    input_ids = torch.tensor([[5, 6, 7, 11, 0]])
    masked2 = processor(input_ids, torch.zeros(1, len(TINY_PIECES)))
    finite2 = set(torch.isfinite(masked2[0]).nonzero().flatten().tolist())
    assert MUST["type"] <= finite2 and not (finite2 & NEVER)


def test_processor_leaves_free_and_complete_states_untouched():
    if not HAVE_TORCH:
        return
    pieces = FREE_PIECES
    trie = _tiny_trie(pieces=pieces)
    processor = ActionGrammarLogitsProcessor(trie, object(), prompt_length=0)

    # These ids spell <action>{"type": "tool_call", "name": "alarm.list", "arguments": {
    open_ids = [0, 1, 4, 5, 6, _ARGS_GLUE_ID]
    scores = torch.zeros(1, len(pieces))
    out = processor(input_ids=torch.tensor([open_ids]), scores=scores)
    assert out is scores, "the arguments object must stay unconstrained"
    assert processor.last_state is not None and processor.last_state.free_json is True

    # An all -inf input stays all -inf (no NaN, no invented values) instead of
    # producing an unusable masked row.
    hostile = torch.full((1, len(pieces)), float("-inf"))
    barrier = ActionGrammarLogitsProcessor(trie, object(), prompt_length=0)
    result = barrier(torch.tensor([[0]]), hostile)
    assert not torch.isnan(result).any()


class _StubModel:
    """Records the generate() call so 'disabled == baseline' is checkable."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: Any = None, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return "generated"


class _StubTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    def __len__(self) -> int:
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids: Any) -> Any:
        return [f"tok{int(i)}" for i in ids] if isinstance(ids, (list, tuple)) else f"tok{int(ids)}"

    def convert_tokens_to_string(self, tokens: Any) -> str:
        return self.pieces[int(str(tokens[0]).removeprefix("tok"))]

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "".join(self.pieces[int(i)] for i in ids)


def test_disabled_is_byte_identical_to_plain_generate():
    model = _StubModel()
    tokenizer = _StubTokenizer(TINY_PIECES)
    kwargs = {"do_sample": False, "eos_token_id": [1, 2], "pad_token_id": 1}

    sentinel = object()
    constrained_generate(model, tokenizer, sentinel, enabled=False, max_new_tokens=17, **kwargs)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["prompt"] is sentinel
    # Nothing added, nothing changed: exactly the caller's kwargs.
    assert call["kwargs"] == {"max_new_tokens": 17, **kwargs}
    assert "logits_processor" not in call["kwargs"]

    # A tokenized mapping is splatted, which is what the caller would write by hand.
    mapping = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    constrained_generate(model, tokenizer, mapping, enabled=False, max_new_tokens=4)
    assert model.calls[-1]["kwargs"] == {"max_new_tokens": 4, **mapping}


def test_enabled_attaches_a_processor_and_keeps_caller_kwargs():
    if not HAVE_TORCH:
        return
    model = _StubModel()
    tokenizer = _StubTokenizer(TINY_PIECES)
    constrained_generate(
        model,
        tokenizer,
        {"input_ids": None},
        enabled=True,
        allowed_tools=["alarm.list"],
        max_new_tokens=5,
        do_sample=False,
    )
    kwargs = model.calls[0]["kwargs"]
    assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 5
    processors = list(kwargs["logits_processor"])
    assert len(processors) == 1
    assert isinstance(processors[0], ActionGrammarLogitsProcessor)
    assert processors[0].trie.tools == ("alarm.list",)


def test_enabled_appends_to_an_existing_logits_processor_list():
    if not HAVE_TORCH:
        return

    class _Existing:
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            return scores

    model = _StubModel()
    constrained_generate(
        model,
        _StubTokenizer(TINY_PIECES),
        {"input_ids": None},
        enabled=True,
        allowed_tools=["alarm.list"],
        logits_processor=[_Existing()],
    )
    processors = list(model.calls[0]["kwargs"]["logits_processor"])
    assert len(processors) == 2
    assert isinstance(processors[0], _Existing)
    assert isinstance(processors[1], ActionGrammarLogitsProcessor)


def test_processor_reports_its_counters():
    if not HAVE_TORCH:
        return
    trie = _tiny_trie()
    processor = ActionGrammarLogitsProcessor(trie, object(), prompt_length=0)
    processor(torch.tensor([[0]]), torch.zeros(1, len(TINY_PIECES)))
    processor(torch.tensor([[0, 1]]), torch.zeros(1, len(TINY_PIECES)))
    assert processor.steps == 2
    assert processor.masked_steps == 2
    assert processor.skipped_steps == 0


# ---------------------------------------------------------------------------
# Wiring into HFGenerator: the flag must actually reach model.generate
# ---------------------------------------------------------------------------
class _KwargSpy:
    """Model stand-in recording the EXACT kwargs each generate() call received."""

    def __init__(self) -> None:
        import torch

        self.calls: list[dict[str, Any]] = []
        self.device = "cpu"
        self.config = None

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        record = {"args": args, "kwargs": dict(kwargs)}
        record["has_logits_processor"] = "logits_processor" in kwargs
        self.calls.append(record)
        return torch.tensor([[0, 1, 2]])


class _ChatTokenizer:
    """Minimal chat-template tokenizer over the tiny vocabulary."""

    pad_token_id = 0
    eos_token_id = 10
    chat_template = "unused"

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    def __len__(self) -> int:
        return len(self.pieces)

    def apply_chat_template(self, messages: Any, tokenize: bool = False, **kwargs: Any) -> str:
        return "|".join(str(m.get("content", "")) for m in messages) + "|"

    def __call__(self, text: Any, return_tensors: str | None = None, **kwargs: Any) -> Any:
        import torch

        ids = torch.tensor([[0, 1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "".join(self.pieces[int(i)] for i in ids if int(i) < len(self.pieces))

    def convert_ids_to_tokens(self, ids: Any) -> Any:
        return [f"tok{int(i)}" for i in ids] if isinstance(ids, (list, tuple)) else f"tok{int(ids)}"

    def convert_tokens_to_string(self, tokens: Any) -> str:
        return self.pieces[int(str(tokens[0]).removeprefix("tok"))]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [E2E_EOS]  # the stop-marker ids; only ``is not None`` matters here


def _generator(constrained: bool, tools: tuple[str, ...] = ()):
    from lifemate.models.generation import HFGenerator

    tokenizer = _ChatTokenizer(FREE_PIECES)
    model = _KwargSpy()
    generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=7,
        temperature=0.0,
        constrained=constrained,
        allowed_tools=tools,
        stop_on_action=False,
    )
    return generator, model


def test_generator_unconstrained_call_is_byte_identical_and_has_no_processor():
    if not HAVE_TORCH:  # HFGenerator.generate imports torch
        return
    """The historical call: same kwargs, no logits_processor, no extra branch."""
    generator, model = _generator(constrained=False)
    generator.generate([{"role": "user", "content": "hi"}])

    assert len(model.calls) == 1
    kwargs = model.calls[0]["kwargs"]
    # Exactly the pre-change kwargs dict, values included (prompt splatted as kwargs).
    assert set(kwargs) == {
        "max_new_tokens", "do_sample", "pad_token_id", "eos_token_id",
        "input_ids", "attention_mask",
    }
    assert kwargs["max_new_tokens"] == 7
    assert kwargs["do_sample"] is False
    assert kwargs["pad_token_id"] == 0
    assert kwargs["eos_token_id"] == 10  # single EOS id stays a bare int
    assert model.calls[0]["has_logits_processor"] is False
    # The prompt was splatted as keyword inputs, exactly as before.
    assert "input_ids" in kwargs and "attention_mask" in kwargs


def test_generator_constrained_call_adds_only_a_processor():
    if not HAVE_TORCH:  # HFGenerator.generate imports torch
        return
    generator, model = _generator(constrained=True, tools=("alarm.list",))
    generator.generate([{"role": "user", "content": "hi"}])

    assert len(model.calls) == 1
    kwargs = model.calls[0]["kwargs"]
    assert model.calls[0]["has_logits_processor"] is True
    processors = list(kwargs.pop("logits_processor"))
    assert len(processors) == 1 and isinstance(processors[0], ActionGrammarLogitsProcessor)
    # Every other kwarg is unchanged from the unconstrained call.
    assert set(kwargs) == {"max_new_tokens", "do_sample", "pad_token_id", "eos_token_id", "input_ids", "attention_mask"}
    assert kwargs["max_new_tokens"] == 7
    assert kwargs["eos_token_id"] == 10
    assert kwargs["do_sample"] is False


def test_set_allowed_tools_changes_the_grammar_per_sample():
    if not HAVE_TORCH:  # HFGenerator.generate imports torch
        return
    generator, model = _generator(constrained=True)
    generator.generate([{"role": "user", "content": "a"}])
    first = list(model.calls[0]["kwargs"]["logits_processor"])[0]
    assert first.trie.tools == ()
    assert first.trie.type_choices() == ('finish"}',)  # finish-only sample

    generator.set_allowed_tools(["alarm.list", "weather.get"])
    generator.generate([{"role": "user", "content": "b"}])
    second = list(model.calls[1]["kwargs"]["logits_processor"])[0]
    assert second.trie.tools == ("alarm.list", "weather.get")
    assert set(second.trie.type_choices()) == {'finish"}', 'tool_call", "name": "'}

    # An empty set is legitimate (finish-only) and must not raise.
    generator.set_allowed_tools([])
    generator.generate([{"role": "user", "content": "c"}])
    third = list(model.calls[2]["kwargs"]["logits_processor"])[0]
    assert third.trie.tools == ()
    assert third.trie.finish_allowed is True


def test_piece_table_is_built_once_per_tokenizer():
    """Per-sample generation must not rebuild 130k pieces every call."""
    from lifemate.models.constrained import _PIECES_CACHE, cached_piece_map

    tokenizer = _ChatTokenizer(FREE_PIECES)
    first = cached_piece_map(tokenizer, len(FREE_PIECES))
    second = cached_piece_map(tokenizer, len(FREE_PIECES))
    assert first is second
    key = (id(tokenizer), len(FREE_PIECES))
    assert key in _PIECES_CACHE and _PIECES_CACHE[key][0] is tokenizer


# ---------------------------------------------------------------------------
# Tiny-vocab end-to-end (torch): the grammar decides the output
# ---------------------------------------------------------------------------
def _tiny_lm_and_tokenizer(pieces: list[str]):
    """A 2-layer random LM whose vocabulary IS the grammar's literals."""
    import torch
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

    tokenizer = _ChatTokenizer(pieces)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=len(pieces),
        max_position_embeddings=256,
        bos_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.generation_config = GenerationConfig(
        bos_token_id=0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
    )
    model.eval()
    return model, tokenizer


def _tiny_run(pieces: list[str], tools: tuple[str, ...], constrained: bool) -> str:
    import torch

    from lifemate.models.constrained import constrained_generate

    model, tokenizer = _tiny_lm_and_tokenizer(pieces)
    inputs = {"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.tensor([[1, 1]])}
    with torch.no_grad():
        out = constrained_generate(
            model,
            tokenizer,
            inputs,
            enabled=constrained,
            allowed_tools=tools,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][2:].tolist())


#: Vocabulary for the end-to-end check: the grammar literals AND every character
#: they are built from. A character-level table means no grammar position is a
#: dead end, so an allowed path always exists and the mask — not the vocabulary —
#: decides the output.
GRAMMAR_LITERALS = [
    ACTION_OPEN_TEXT,
    ACTION_CLOSE_TEXT,
    "finish",
    "tool_call",
    "{",
    "}",
    '"type": "',
    '", "name": "',
    '", "arguments": ',
    "alarm.list",
    "weather.get",
]
GHOST_TOOL = "ghost.tool"
_JUNK = ["<unk>", "</s>", "<s>", "<pad>", "~", "?", "hello"]


def _char_vocab() -> list[str]:
    chars = sorted({ch for literal in GRAMMAR_LITERALS + [GHOST_TOOL] for ch in literal})
    pieces = list(_JUNK)
    pieces += [ch for ch in chars if ch not in pieces]
    pieces += [literal for literal in GRAMMAR_LITERALS + [GHOST_TOOL] if literal not in pieces]
    return pieces


E2E_PIECES = _char_vocab()
E2E_EOS = 1  # "</s>"
E2E_PAD = 3  # "<pad>"


class _Bias:
    """Logits processor that boosts given ids: steers only the BRANCH choice."""

    def __init__(self, boosts: dict[int, float]) -> None:
        self.boosts = boosts

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for token_id, delta in self.boosts.items():
            scores[..., token_id] += delta
        return scores


def _tiny_lm_and_tokenizer(pieces: list[str]):
    """A 2-layer random LM over the grammar's own vocabulary."""
    import torch
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

    tokenizer = _ChatTokenizer(pieces)
    tokenizer.eos_token_id = E2E_EOS
    tokenizer.pad_token_id = E2E_PAD
    tokenizer.special_ids = frozenset({0, 1, 2, 3})
    tokenizer.decode = lambda ids, skip_special_tokens=True: "".join(
        tokenizer.pieces[int(i)]
        for i in ids
        if int(i) < len(tokenizer.pieces)
        and not (skip_special_tokens and int(i) in tokenizer.special_ids)
    )
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=len(pieces),
        max_position_embeddings=256,
        bos_token_id=2,
        eos_token_id=E2E_EOS,
        pad_token_id=E2E_PAD,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=E2E_EOS, pad_token_id=E2E_PAD
    )
    model.eval()
    return model, tokenizer


def _tiny_run(
    pieces: list[str],
    tools: tuple[str, ...],
    constrained: bool,
    boosts: dict[str, float] | None = None,
) -> str:
    import torch
    from transformers import LogitsProcessorList

    from lifemate.models.constrained import constrained_generate

    model, tokenizer = _tiny_lm_and_tokenizer(pieces)
    inputs = {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.tensor([[1, 1]])}
    bias = _Bias({pieces.index(name): delta for name, delta in (boosts or {}).items()})
    kwargs: dict[str, Any] = {
        # Character-level pieces need room for ~45 steps; the grammar stops at EOS.
        "max_new_tokens": 80,
        "do_sample": False,
        "pad_token_id": E2E_PAD,
        "eos_token_id": E2E_EOS,
    }
    if not constrained:
        # Baseline: the bias is the only influence, no grammar at all.
        with torch.no_grad():
            out = model.generate(
                **inputs, logits_processor=LogitsProcessorList([bias]), **kwargs
            )
        return tokenizer.decode(out[0][2:].tolist())
    with torch.no_grad():
        out = constrained_generate(
            model,
            tokenizer,
            inputs,
            enabled=True,
            allowed_tools=tools,
            logits_processor=[bias],
            **kwargs,
        )
    return tokenizer.decode(out[0][2:].tolist())


def test_tiny_vocab_finish_only_generation_is_exactly_the_action_block():
    """A finish-only sample: the grammar forces exactly one legal block."""
    if not HAVE_TORCH:
        return
    text = _tiny_run(E2E_PIECES, (), constrained=True)
    assert text == ACTION_OPEN_TEXT + '{"type": "finish"}' + ACTION_CLOSE_TEXT, text


def test_tiny_vocab_tool_call_uses_an_allowed_tool_and_parses():
    """Steered to the tool_call branch, the grammar picks only allowed names."""
    if not HAVE_TORCH:
        return
    from lifemate.agent.protocol import parse_action

    text = _tiny_run(
        E2E_PIECES,
        ("alarm.list",),
        constrained=True,
        # Steer the unconstrained type choice to tool_call and close the free
        # arguments object immediately (a huge boost is safe: at every forced
        # position the grammar masks "}" to -inf anyway).
        boosts={"tool_call": 60.0, "}": 400.0},
    )
    parsed = parse_action(text)
    assert parsed is not None and parsed.type == "tool_call", text
    assert parsed.name == "alarm.list", text
    # "ghost.tool" exists in the vocabulary and is outside the allow-list.
    assert GHOST_TOOL not in text


def test_negative_control_without_the_grammar_the_disallowed_tool_is_emitted():
    """Same model, same bias, grammar off: the blocked text DOES appear."""
    if not HAVE_TORCH:
        return
    text = _tiny_run(
        E2E_PIECES,
        ("alarm.list",),
        constrained=False,
        boosts={"g": 80.0, "t": 40.0},
    )
    # Unconstrained, the boosted junk character wins and the output is not a
    # parseable action at all: proof that the grammar, not the vocabulary, is what
    # makes the constrained run well-formed.
    assert text != '<action>{"type": "finish"}</action>', text
    from lifemate.agent.protocol import parse_action

    parsed = parse_action(text)
    assert parsed is None or parsed.type == "invalid", text


def test_processor_blocks_the_disallowed_token_on_a_forced_row():
    """Token-level negative control: same scores, mask on vs mask off."""
    if not HAVE_TORCH:
        return
    import torch

    trie = _tiny_trie(pieces=E2E_PIECES)
    # Literal pieces that spell <action>{"type": "tool_call", "name": "
    prefix_text = [ACTION_OPEN_TEXT, "{", '"type": "', "tool_call", '", "name": "']
    tool_prefix_ids = [E2E_PIECES.index(piece) for piece in prefix_text]
    ghost_id = E2E_PIECES.index(GHOST_TOOL)
    allowed_id = E2E_PIECES.index("alarm.list")

    scores = torch.zeros(1, len(E2E_PIECES))
    scores[0, ghost_id] = 50.0  # the model loudly prefers the disallowed tool
    assert int(scores.argmax()) == ghost_id  # unmasked: it IS chosen

    processor = ActionGrammarLogitsProcessor(trie, object(), prompt_length=0)
    masked = processor(torch.tensor([tool_prefix_ids]), scores.clone())
    # The disallowed name is unreachable, and whatever wins is a legal continuation
    # of the tool-name literal (with a character-level vocabulary that is "a"/...,
    # i.e. a prefix of an allowed name — never the ghost).
    assert not torch.isfinite(masked[0, ghost_id])
    chosen = int(masked.argmax())
    assert chosen != ghost_id
    state = trie.state_for(
        ACTION_OPEN_TEXT + '{"type": "tool_call", "name": "'
    )
    assert chosen in (state.allowed_ids or frozenset())
    assert allowed_id in (state.allowed_ids or frozenset())
