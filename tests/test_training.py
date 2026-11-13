"""Training-slice tests: masking, tuning-mode selection, preflight, checkpoints.

Stdlib-only where possible. Every test that needs torch/transformers prints a
``SKIP`` line naming the reason instead of failing, so this file runs on the
CPU/stdlib box and again on the GPU box without edits.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.config import LoraConfig, TrainingConfig, load_config  # noqa: E402
from lifemate.data.trajectory_builder import DecisionPoint  # noqa: E402
from lifemate.errors import ConfigError, LifeMateError  # noqa: E402
from lifemate.training.collator import (  # noqa: E402
    NO_THINK_SUFFIX,
    SupervisedCollator,
    assert_loss_masking,
    build_supervised_example,
    render_prompt_text,
    to_torch_batch,
)
from lifemate.training.memory_preflight import format_report, run_memory_preflight  # noqa: E402
from lifemate.training.trainer import (  # noqa: E402
    CheckpointManager,
    SFTTrainer,
    optimizer_steps_per_epoch,
    read_training_state,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _CharTokenizer:
    """Char-level tokenizer: one token per character, offset so ids avoid 0/1."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [2 + (ord(ch) % 90) for ch in text]

class _ChatTokenizer(_CharTokenizer):
    """Char tokenizer that also exposes a chat template."""

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False):
        del tokenize
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


class _DecodingChatTokenizer(_ChatTokenizer):
    """``_ChatTokenizer`` that can also ``decode`` what it encoded.

    The truncation guard compares the target's reference values against the
    FINAL prompt text, which needs a decoder. The char table maps several
    characters onto one id (``ord % 90``), so it is not reversible in general;
    this fixture keeps a forward table instead, which is exact for the ASCII
    fixtures the collator tests use.
    """

    def __init__(self) -> None:
        self._forward: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens)
        for ch, token_id in zip(text, ids):
            self._forward[token_id] = ch
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(self._forward.get(int(i), "?") for i in ids)

class _ThinkTemplateTokenizer(_CharTokenizer):
    """Char tokenizer whose template behaves like Qwen3's (opt-out thinking).

    Mirrors the real ``chat_template.jinja`` tail: ``enable_thinking=False``
    emits an empty, immediately closed think block; anything else (including an
    absent flag) leaves the block open.
    """

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
    ):
        del tokenize
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if not add_generation_prompt:
            return body
        suffix = "\x3cthink\x3e\n\n\x3c/think\x3e\n\n" if enable_thinking is False else "\x3cthink\x3e\n"
        return body + "<|im_start|>assistant\n" + suffix

def _messages(prompt: str = "Dat bao thuc") -> list[dict[str, str]]:
    return [{"role": "system", "content": "S"}, {"role": "user", "content": prompt}]

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401,PLC0415 - probe only

        return True
    except Exception:
        return False

def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401,PLC0415 - probe only

        return True
    except Exception:
        return False

# --------------------------------------------------------------------------- #
# collator
# --------------------------------------------------------------------------- #
def test_collator_masks_exactly_the_prompt() -> None:
    """labels[:prompt_len] == -100 and the target is supervised verbatim."""
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    example = build_supervised_example(tokenizer, _messages(), target, 256)
    assert example["kept"] is True, example
    prompt_len = example["n_prompt_tokens"]
    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert example["labels"][prompt_len:] == example["input_ids"][prompt_len:]
    assert example["labels"][-1] == tokenizer.eos_token_id, "target must end with EOS"
    assert example["n_target_tokens"] == len(tokenizer.encode(target)) + 1
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])
    assert_loss_masking(example["labels"], prompt_len)
    print(
        f"  masking -> prompt {prompt_len} tokens -100, target {example['n_target_tokens']} supervised"
    )

def test_collator_drops_instead_of_truncating_target() -> None:
    """A target that does not fit is dropped, never truncated."""
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    target_len = len(tokenizer.encode(target)) + 1
    for budget in (target_len - 1, target_len):
        example = build_supervised_example(tokenizer, _messages(), target, budget)
        assert example["kept"] is False
        assert example["reason"] == "target_too_long", example
        assert example["input_ids"] == [] and example["labels"] == []
    print("  target too long -> dropped with reason target_too_long")

def test_collator_middle_truncates_prompt() -> None:
    """Over-budget prompts drop from the middle (history); request and tail survive."""
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    request_marker = "REQ-HEAD-MARKER"
    history_marker = "HIST-MID-" + ("x" * 400)
    tail_marker = "TAIL-MARK"
    messages = [
        {"role": "user", "content": f"{request_marker} {history_marker} {tail_marker}"}
    ]
    target_len = len(tokenizer.encode(target)) + 1
    budget = target_len + 50
    example = build_supervised_example(tokenizer, messages, target, budget, head_keep_fraction=0.35)
    assert example["kept"] is True, example
    assert len(example["input_ids"]) == budget
    assert example["n_target_tokens"] == target_len
    assert example["reason"].startswith("prompt_middle_truncated_"), example["reason"]
    assert example["input_ids"][-target_len:] == tokenizer.encode(target) + [tokenizer.eos_token_id]
    assert_loss_masking(example["labels"], example["n_prompt_tokens"])

    kept_text = _decode(tokenizer, example["input_ids"][: example["n_prompt_tokens"]])
    assert "REQ-HEAD" in kept_text, "request marker must survive in preserved head"
    assert "TAIL-MARK" in kept_text, "tail marker must survive in preserved tail"
    assert "HIST-MID" not in kept_text, "history in middle must be dropped"

def test_collator_truncation_keeps_the_request_section() -> None:
    """A decision prompt's request survives even when the tail alone fills the budget.

    The old middle cut kept a fixed head fraction of the *whole* text. Once the
    steps section grew, that fraction was consumed by the system prompt and the
    spoken request (the first section of the user body) was cut away — the target
    then had no request to condition on. Truncation now protects the ``# Yêu cầu``
    section and cuts only what follows it.
    """
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    request = "REQ-REQUEST-MARKER"
    steps = "\n".join(f"Bước {i}:\n<action>{{}}</action>\nKết quả: {{'i': {i}}}" for i in range(200))
    prompt = (
        f"# Yêu cầu\nDat bao thuc {request}.\n\n"
        "# Ngữ cảnh\ncurrent_time: now\n\n"
        f"# Các bước đã thực hiện\n{steps}\n\n"
        '# Định dạng trả lời\n<action>{"type": "finish"}</action>\n'
    )
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": prompt}]
    target_len = len(tokenizer.encode(target)) + 1
    full_len = len(tokenizer.encode(render_prompt_text(messages, tokenizer)))
    assert full_len > 200 + target_len

    for budget in (target_len + 60, target_len + 150, target_len + 400):
        example = build_supervised_example(tokenizer, messages, target, budget)
        assert example["kept"] is True, example
        assert example["n_prompt_tokens"] <= budget - target_len
        assert example["reason"] is None or example["reason"].startswith("prompt_middle_truncated_")
        assert_loss_masking(example["labels"], example["n_prompt_tokens"])
        assert example["input_ids"][-target_len:] == tokenizer.encode(target) + [tokenizer.eos_token_id]
        kept = _decode(tokenizer, example["input_ids"][: example["n_prompt_tokens"]])
        assert "REQ-REQUEST-MARKER" in kept, f"request lost at budget={budget}: {kept[:200]!r}"
    print("  truncation -> '# Yêu cầu' section survives every over-budget cut")

def test_collator_truncation_keeps_the_target_tool_schema_block() -> None:
    """Truncation never deletes the schema block of the tool the target calls.

    The catalog sits in the middle of the prompt, so a plain head/tail cut could
    splice a hole through ``### <tool> (kind=`` — measured on the prepared build:
    3 of 3324 ``tool_call`` prompts lost the block of the tool they were told to
    call (all three ``contacts.search``, whose ``query`` argument name appears
    nowhere else). The catalog is now pinned to the tail region.
    """
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "tool_call", "name": "contacts.search", "arguments": {"query": "Lan"}}</action>'
    schema_block = tokenizer.encode("### contacts.search (kind=")
    history = "x" * 3000
    prompt = (
        "# Yêu cầu\nNhắn Lan giúp em REQ-MARK.\n\n"
        "# Ngữ cảnh\ncurrent_time: 2026-11-20T09:00:00+07:00\n\n"
        f"# Lịch sử hội thoại\nNgười dùng: {history}\n\n"
        "# Công cụ khả dụng\n## contacts\n"
        "### contacts.get (kind=read)\nChi tiết liên hệ\nrequired: contact_id\n"
        'parameters: {"properties": {"contact_id": {"type": "string"}}, "required": ["contact_id"], "type": "object"}\n'
        "### contacts.search (kind=read)\nTìm liên hệ theo tên\nrequired: (none)\n"
        'parameters: {"properties": {"query": {"type": "string"}}, "required": [], "type": "object"}\n'
        "## notes\n### notes.create (kind=write)\nTạo ghi chú\nrequired: content\n"
        'parameters: {"properties": {"content": {"type": "string"}}, "required": ["content"], "type": "object"}\n\n'
        '# Các bước đã thực hiện\nBước 1:\n<action>{}</action>\nKết quả: {"ok": true}\n\n'
        '# Định dạng trả lời\n<action>{"type": "tool_call", "name": "<ten>", "arguments": {}}</action>\n'
    )
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": prompt}]
    target_len = len(tokenizer.encode(target)) + 1

    def keeps_block(ids: list[int]) -> bool:
        return any(ids[i : i + len(schema_block)] == schema_block for i in range(len(ids)))

    # Budget chosen so the old middle cut fell inside the catalog (it did: block
    # lost at +800) while the trigger keeps exactly the budget.
    example = build_supervised_example(tokenizer, messages, target, target_len + 800)
    assert example["kept"] is True, example
    assert example["reason"].startswith("prompt_middle_truncated_"), example["reason"]
    assert example["n_prompt_tokens"] == 800
    assert_loss_masking(example["labels"], example["n_prompt_tokens"])
    assert keeps_block(example["input_ids"][: example["n_prompt_tokens"]]), "schema block lost"
    kept = _decode(tokenizer, example["input_ids"][: example["n_prompt_tokens"]])
    assert "REQ-MARK" in kept, "request must still survive"
    print("  truncation -> target tool's '### <tool> (kind=' block survives")

def test_collator_never_supervises_a_target_whose_reference_value_truncation_removed() -> None:
    """Every kept ``tool_call`` can still see the ID it is asked to emit.

    ``build_decision_points`` drops such a point when the *untruncated* prompt
    lacks the value. Truncation happens later, inside the collator, so the same
    question has to be asked of the final text: otherwise the model is trained to
    emit an identifier it was never shown. Measured at ``max_seq_length=4096`` on
    the prepared build: 122 records. At 8192 the pinned catalog keeps the value.
    """
    tokenizer = _DecodingChatTokenizer()
    target = (
        '<action>{"type": "tool_call", "name": "sms.send", '
        '"arguments": {"contact_id": "c_777", "content": "toi nha"}}</action>'
    )
    # c_777 exists only in the history section, which the middle cut removes.
    prompt = (
        "# Yêu cầu\nNhan tin cho Lan.\n\n"
        f"# Lịch sử hội thoại\nNgười dùng: ma lien he c_777 nam o day.\n{'h' * 600}\n\n"
        "# Công cụ khả dụng\n## sms\n"
        "### sms.send (kind=write)\nGửi tin\nrequired: contact_id, content\n"
        'parameters: {"properties": {"contact_id": {"type": "string"}, '
        '"content": {"type": "string"}}, "required": ["contact_id", "content"], "type": "object"}\n\n'
        "# Định dạng trả lời\n<action></action>\n"
    )
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": prompt}]
    target_len = len(tokenizer.encode(target)) + 1

    # The cut geometry is not monotone in the budget (the pinned catalog shifts it),
    # so the invariant is checked at every budget rather than at one magic number.
    dropped = 0
    kept = 0
    for extra in range(0, 1200, 50):
        example = build_supervised_example(tokenizer, messages, target, target_len + extra)
        if not example["kept"]:
            assert example["reason"] in {"target_value_truncated", "target_too_long"}, example["reason"]
            assert example["input_ids"] == [] and example["labels"] == []
            dropped += example["reason"] == "target_value_truncated"
            continue
        kept += 1
        kept_text = _decode(tokenizer, example["input_ids"][: example["n_prompt_tokens"]])
        assert "c_777" in kept_text, f"supervised a target whose ID was cut (budget +{extra})"
        assert_loss_masking(example["labels"], example["n_prompt_tokens"])
    assert kept > 0, "fixture never fit: the positive branch is untested"
    assert dropped > 0, "fixture never truncated the ID away: the drop branch is untested"

    # A value the model may compose is not a reference: its absence never drops.
    no_ref = (
        '<action>{"type": "tool_call", "name": "sms.send", '
        '"arguments": {"content": "chao ban"}}</action>'
    )
    assert build_supervised_example(tokenizer, messages, no_ref, target_len + 100)["kept"] is True
    print(f"  truncation -> {dropped} over-budget points dropped, {kept} kept with the ID visible")

def test_collator_pathological_long_prompt_target_intact() -> None:
    """A 20k-char prompt middle-truncates while target and head/tail survive intact."""
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    long_noise = "Z" * 20000
    messages = [{"role": "user", "content": f"REQ_START_{long_noise}_REQ_END"}]
    target_len = len(tokenizer.encode(target)) + 1
    budget = target_len + 100
    example = build_supervised_example(tokenizer, messages, target, budget)
    assert example["kept"] is True, example
    assert example["n_target_tokens"] == target_len
    assert example["input_ids"][-target_len:] == tokenizer.encode(target) + [tokenizer.eos_token_id]
    assert example["labels"][-target_len:] == tokenizer.encode(target) + [tokenizer.eos_token_id]
    assert_loss_masking(example["labels"], example["n_prompt_tokens"])
    kept_text = _decode(tokenizer, example["input_ids"][: example["n_prompt_tokens"]])
    assert "REQ_START" in kept_text
    assert "REQ_END" in kept_text

def test_collator_head_keep_fraction_validation() -> None:
    """head_keep_fraction must be a float between 0.0 and 1.0."""
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    for bad_fraction in (-0.1, 1.1, -1.0, 2.0, "bad", True):
        try:
            build_supervised_example(
                tokenizer, _messages(), target, 128, head_keep_fraction=bad_fraction
            )
        except LifeMateError as exc:
            assert "head_keep_fraction" in str(exc)
        else:
            raise AssertionError(f"expected LifeMateError for head_keep_fraction={bad_fraction!r}")


def test_collator_fitting_prompt_has_no_truncation() -> None:
    """A prompt that fits within budget has reason=None and is completely un-truncated."""
    tokenizer = _ChatTokenizer()
    target = '<action>{"type": "finish"}</action>'
    messages = _messages("Fit within budget")
    example = build_supervised_example(tokenizer, messages, target, 256)
    assert example["kept"] is True
    assert example["reason"] is None
    assert example["n_prompt_tokens"] == len(tokenizer.encode(render_prompt_text(messages, tokenizer)))
    assert_loss_masking(example["labels"], example["n_prompt_tokens"])
def _decode(tokenizer: _CharTokenizer, ids: list[int]) -> str:
    table = {2 + (ord(ch) % 90): ch for ch in map(chr, range(0x20, 0x7F))}
    return "".join(table.get(i, "?") for i in ids)

def test_collator_fallback_template_is_documented_and_stable() -> None:
    """Without a chat template the fallback renders ChatML deterministically."""
    bare = _CharTokenizer()
    first = build_supervised_example(bare, _messages(), '<action>{"type": "finish"}</action>', 256)
    second = build_supervised_example(bare, _messages(), '<action>{"type": "finish"}</action>', 256)
    assert first["input_ids"] == second["input_ids"], "fallback rendering is not deterministic"
    text = _decode(bare, first["input_ids"][: first["n_prompt_tokens"]])
    assert "<|im_start|>system" in text and "<|im_start|>assistant" in text
    print("  fallback template -> ChatML, deterministic")


def test_rendered_prompt_closes_the_think_block() -> None:
    """A prompt must never end inside an open `` thinking`` block.

    Qwen3's template emits ``<|im_start|>assistant\\n thinking\\n`` unless it is
    called with ``enable_thinking=False``. Rendering the default would supervise
    the action *inside* reasoning and leave inference mid-thought with no cue to
    stop, so both call sites pass the flag and must agree byte for byte.
    """
    from lifemate.models.generation import HFGenerator

    messages = _messages()
    think = _ThinkTemplateTokenizer()
    train = render_prompt_text(messages, think)
    infer = HFGenerator(object(), think).build_prompt(messages)
    for name, text in (("train", train), ("infer", infer)):
        assert "<|im_start|>assistant" in text, f"{name}: no assistant generation prompt"
        tail = text.split("<|im_start|>assistant")[-1]
        assert tail.count("</think>") >= tail.count(" thinking"), f"{name}: unclosed think block {tail!r}"
        assert tail.endswith("\n\n"), f"{name}: target would land inside reasoning {tail!r}"
    assert train == infer, "train/inference prompt bytes diverge"
    assert train.endswith("<|im_start|>assistant\n" + NO_THINK_SUFFIX)

    # A tokenizer whose template rejects the kwarg still renders, identically.
    plain = _ChatTokenizer()
    assert render_prompt_text(messages, plain) == HFGenerator(object(), plain).build_prompt(messages)
    assert render_prompt_text(messages, plain).endswith("<assistant>")

    # The bare-tokenizer ChatML fallback uses the same closed-think prefix.
    fallback_tail = render_prompt_text(messages, _CharTokenizer()).split("<|im_start|>assistant")[-1]
    assert fallback_tail.count("</think>") >= fallback_tail.count(" thinking")
    assert fallback_tail.endswith("\n\n")

    # A genuinely broken template must surface on BOTH sides: silently falling
    # back to ChatML in training while inference raises (or renders the model's
    # own template) would train on prompt bytes we never serve.
    class _Broken(_ThinkTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("template exploded")

    for name, render in (
        ("train", lambda: render_prompt_text(messages, _Broken())),
        ("infer", lambda: HFGenerator(object(), _Broken()).build_prompt(messages)),
    ):
        try:
            render()
        except LifeMateError as exc:
            assert "apply_chat_template failed" in str(exc), f"{name}: unexpected error {exc}"
        else:  # pragma: no cover
            raise AssertionError(f"real template failure must not be swallowed at {name}")
    print("  prompt tail -> closed think block, train/infer identical")


def test_no_think_suffix_is_a_real_closed_think_block() -> None:
    """``NO_THINK_SUFFIX`` must be the literal Qwen tail, not a lookalike.

    The constant once read ``" think...`` (space, no opening tag): every
    bare-tokenizer prompt then ended with a plain word instead of the closed
    ``think`` block the real template emits. Building the expectation from
    ``chr()`` keeps the test independent of how the literal is spelled.
    """
    lt, gt = chr(60), chr(62)
    expected = lt + "think" + gt + "\n\n" + lt + "/think" + gt + "\n\n"
    assert NO_THINK_SUFFIX == expected, f"NO_THINK_SUFFIX is {NO_THINK_SUFFIX!r}, expected {expected!r}"
    rendered = render_prompt_text(_messages(), _ThinkTemplateTokenizer())
    assert rendered.endswith("<|im_start|>assistant\n" + expected)
    print("  think suffix -> closed block with real angle brackets")

def test_collator_padding_mask_and_labels() -> None:
    """Padding: attention_mask 0, labels -100, ids pad_token_id."""
    tokenizer = _CharTokenizer()
    collator = SupervisedCollator.from_tokenizer(tokenizer)
    short = build_supervised_example(tokenizer, _messages("a"), '<action>{"type": "finish"}</action>', 256)
    long = build_supervised_example(tokenizer, _messages("b" * 40), '<action>{"type": "finish"}</action>', 256)
    batch = collator([short, long])
    width = max(len(short["input_ids"]), len(long["input_ids"]))
    for ids, mask, labels in zip(batch["input_ids"], batch["attention_mask"], batch["labels"]):
        assert len(ids) == len(mask) == len(labels) == width
        pad = width - sum(mask)
        assert mask[: len(ids) - pad] == [1] * (len(ids) - pad)
        assert mask[-pad:] == [0] * pad if pad else True
        assert labels[-pad:] == [-100] * pad if pad else True
        assert ids[-pad:] == [tokenizer.pad_token_id] * pad if pad else True
    assert batch["attention_mask"][0].count(0) > 0, "shorter example must be padded"
    print(f"  padding -> width {width}, mask 0 / labels -100 on pad positions")

def test_collator_rejects_bad_pad_id_and_empty_batch() -> None:
    """A non-integer pad id and an all-dropped batch both fail loudly."""
    for bad in (None, "0", 1.5):
        try:
            SupervisedCollator(pad_token_id=bad)
        except LifeMateError:
            pass
        else:
            raise AssertionError(f"pad_token_id={bad!r} should be rejected")
    collator = SupervisedCollator(pad_token_id=0)
    try:
        collator([{"input_ids": [], "labels": [], "kept": False}])
    except LifeMateError:
        pass
    else:
        raise AssertionError("an all-dropped batch must raise, not produce empty tensors")
    print("  validation -> bad pad id and empty batch rejected")

def test_loss_masking_assertion_catches_leaks() -> None:
    """assert_loss_masking rejects a supervised prompt position or a masked target."""
    assert_loss_masking([-100, -100, 5, 6], 2)
    for labels, prompt_len in (
        ([-100, 5, 5, 6], 2),      # prompt position supervised
        ([-100, -100, -100, 6], 2),  # target position masked
        ([-100, -100], 2),          # no target at all
        ([5, 6], 0),                # zero prompt len (causal LM requires >= 1 prompt token)
    ):
        try:
            assert_loss_masking(labels, prompt_len)
        except LifeMateError:
            continue
        raise AssertionError(f"labels={labels} prompt_len={prompt_len} should be rejected")
    print("  assert_loss_masking -> 3 broken layouts rejected, 1 valid accepted")

# --------------------------------------------------------------------------- #
# anti-leak: gradient must not depend on prompt tokens
# --------------------------------------------------------------------------- #
def test_gradient_independent_of_prompt_tokens() -> None:
    """The loss (not just the label tensor) ignores prompt tokens.

    Same target, two different prompts ⇒ gradients agree to float tolerance. The
    positive control — an *unmasked* label vector, i.e. prompt tokens leaking into
    the loss — must produce gradients that disagree by orders of magnitude, which
    is what makes this test able to fail.
    """
    if not _torch_available():
        print("  SKIP gradient anti-leak test: torch not installed")
        return
    import torch  # noqa: PLC0415

    from lifemate.training.trainer import _forward_loss  # noqa: PLC0415

    tokenizer = _CharTokenizer()
    target = '<action>{"type": "finish"}</action>'
    left = build_supervised_example(tokenizer, _messages("Dat bao thuc"), target, 256)
    right = build_supervised_example(tokenizer, _messages("Mo nhac"), target, 256)
    assert left["n_target_tokens"] == right["n_target_tokens"]
    assert left["labels"][-left["n_target_tokens"] :] == right["labels"][-right["n_target_tokens"] :]
    assert left["n_prompt_tokens"] != right["n_prompt_tokens"], "prompts must actually differ"

    masked = _grad_delta(left, right)
    assert masked["max_rel_delta"] < 1e-4, masked

    # positive control: unmask the prompt so the loss *does* see prompt tokens
    leaky_left = dict(left, labels=list(left["input_ids"]))
    leaky_right = dict(right, labels=list(right["input_ids"]))
    leaky = _grad_delta(leaky_left, leaky_right)
    assert leaky["max_rel_delta"] > 1e-2, leaky
    assert leaky["max_rel_delta"] > masked["max_rel_delta"] * 100, (leaky, masked)
    print(
        f"  gradient anti-leak -> masked max_rel_delta={masked['max_rel_delta']:.2e} "
        f"vs unmasked (control) {leaky['max_rel_delta']:.2e}"
    )

def _grad_delta(left: dict, right: dict) -> dict:
    """Relative gradient difference between two examples on a fresh tiny model.

    A fresh model per call keeps the comparison pure: absolute float32 reduction
    order differs slightly with sequence length, so the metric is relative and
    the leaky control shows the scale a real leak produces.
    """
    import torch  # noqa: PLC0415

    from lifemate.training.trainer import _forward_loss  # noqa: PLC0415

    deltas: list[float] = []
    for example in (left, right):
        torch.manual_seed(0)
        model = _tiny_lm(vocab_size=200, hidden=16)
        model.zero_grad(set_to_none=True)
        batch = to_torch_batch(SupervisedCollator(pad_token_id=0)([example]))
        _forward_loss(model, batch).backward()
        deltas.append(
            torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        )
    diff = (deltas[0] - deltas[1]).abs()
    # Scale by the gradient magnitude, not per-element: tiny gradients (~1e-8)
    # are dominated by float32 reduction order, which is not a leak.
    scale = max(float(deltas[0].abs().max().item()), 1e-6)
    return {
        "max_rel_delta": float(diff.max().item()) / scale,
        "max_abs_delta": float(diff.max().item()),
        "n_tensors": int(deltas[0].numel()),
    }

def _tiny_lm(vocab_size: int = 200, hidden: int = 16):
    """Minimal causal LM: embedding → linear head, returns an object with .loss."""
    import torch  # noqa: PLC0415

    class TinyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden)
            self.head = torch.nn.Linear(hidden, vocab_size)
            self.config = type("Cfg", (), {"num_hidden_layers": 1, "hidden_size": hidden})()

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            del attention_mask
            logits = self.head(self.embed(input_ids))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )
            return type("Out", (), {"loss": loss, "logits": logits})()

    return TinyLM()

# --------------------------------------------------------------------------- #
# tuning mode selection
# --------------------------------------------------------------------------- #
def test_full_is_the_default_tuning_mode() -> None:
    """Config defaults and the CLI default are both ``full``."""
    assert load_config().training.tuning_mode == "full"
    assert load_config(ROOT / "configs" / "base.yaml").training.tuning_mode == "full"
    assert load_config(ROOT / "configs" / "training_full.yaml").training.tuning_mode == "full"
    out = _run([sys.executable, "scripts/train.py", "--help"])
    assert out.returncode == 0, out.stderr
    assert "--tuning {full,lora}" in out.stdout, out.stdout
    assert "default: from config, or 'full'" in out.stdout, "the CLI must advertise full as the default"
    assert load_config(ROOT / "configs" / "training_lora.yaml").training.tuning_mode == "lora"
    print("  tuning mode -> full by default (config + CLI), lora only when configured")

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600, check=False
    )

def test_configs_round_trip_through_load_config() -> None:
    """Every shipped config parses with load_config (unknown keys would raise)."""
    for name in ("base", "training_full", "training_lora", "inference", "evaluation", "wandb"):
        path = ROOT / "configs" / f"{name}.yaml"
        assert path.is_file(), f"missing configs/{name}.yaml"
        cfg = load_config(path)
        assert cfg.training.tuning_mode in {"full", "lora"}
    lora_cfg = load_config(ROOT / "configs" / "training_lora.yaml")
    assert (lora_cfg.lora.r, lora_cfg.lora.alpha, lora_cfg.lora.dropout) == (16, 32, 0.05)
    full_cfg = load_config(ROOT / "configs" / "training_full.yaml")
    assert full_cfg.training.gradient_checkpointing is True
    assert full_cfg.training.precision == "bfloat16"
    print("  configs -> 6 files parse, LoRA r/alpha/dropout = 16/32/0.05")

def test_shipped_configs_agree_on_max_seq_length() -> None:
    """Every shipped training config must agree on the sequence budget.

    The value decides data coverage, not just memory: measured on the real SFT
    build with the real Qwen3.5 tokenizer, ``4096`` middle-truncates
    **3299/4385 (75.2 %)** records and removes the ``# Công cụ khả dụng`` section
    from **176** of them (the tool list the model must choose from), dropping 6
    reference values a target needs. At ``8192`` truncation is **47/4385
    (1.07 %)**, none lose the tool section, and the largest record needs 9016
    tokens (prompt + target + eos, measured with the closed think-block
    suffix). A config that silently keeps 4096 would train on different data
    than the run it is compared with, so drift is a test failure, not a
    preference.
    """
    expected = 8192
    paths = ("base.yaml", "training_full.yaml", "training_lora.yaml")
    seen = {}
    for name in paths:
        cfg = load_config(ROOT / "configs" / name)
        seen[name] = cfg.training.max_seq_length
        assert cfg.training.max_seq_length == expected, (
            f"configs/{name} sets max_seq_length={cfg.training.max_seq_length}, "
            f"expected {expected} for full data coverage ({seen})"
        )
    assert len(set(seen.values())) == 1, f"shipped configs disagree: {seen}"
    # The dataclass default must match too, or a config-less run silently differs.
    assert load_config().training.max_seq_length == expected
    print(f"  configs -> all {len(paths)} shipped configs + dataclass default use max_seq_length={expected}")

def test_configure_full_finetuning_trainable_equals_total() -> None:
    """Full mode unfreezes everything and asserts trainable == total."""
    if not _torch_available():
        print("  SKIP full-FT configure test: torch not installed")
        return
    from lifemate.training.full_finetuning import configure_full_finetuning  # noqa: PLC0415

    model = _tiny_lm()
    for param in model.parameters():
        param.requires_grad_(False)  # arrive half-frozen, as a LoRA checkpoint would
    cfg = load_config()
    cfg.training.precision = "float32"
    cfg.training.gradient_checkpointing = False
    report = configure_full_finetuning(model, None, cfg)
    assert report["trainable_params"] == report["total_params"], report
    assert report["frozen_params"] == 0
    assert report["tuning_mode"] == "full"
    print(
        f"  full FT -> {report['trainable_params']:,}/{report['total_params']:,} trainable "
        "(half-frozen model repaired, asserted)"
    )

def test_configure_full_finetuning_fails_when_frozen() -> None:
    """freeze_nothing=True must refuse a model whose parameters stay frozen."""
    if not _torch_available():
        print("  SKIP full-FT frozen guard: torch not installed")
        return
    import torch  # noqa: PLC0415

    from lifemate.training.full_finetuning import configure_full_finetuning  # noqa: PLC0415

    class Stubborn(torch.nn.Module):
        """``parameters()`` reports a parameter that refuses to be unfrozen."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def parameters(self, recurse: bool = True):
            del recurse
            self.lin.weight.requires_grad_(False)
            return [self.lin.weight]

    cfg = load_config()
    cfg.training.gradient_checkpointing = False
    try:
        configure_full_finetuning(Stubborn(), None, cfg)
    except LifeMateError as exc:
        assert "frozen" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("a permanently frozen model must fail full fine-tuning")
    print("  full FT guard -> refuses a model whose parameters stay frozen")

# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
def test_lora_discovery_and_empty_set() -> None:
    """Discovery finds real projections; an empty intersection raises."""
    if not _torch_available():
        print("  SKIP LoRA discovery: torch not installed")
        return
    import torch  # noqa: PLC0415

    from lifemate.training.lora_finetuning import discover_target_modules  # noqa: PLC0415
    model = _tiny_lm()
    assert discover_target_modules(model) == [], "TinyLM has no attention projections"

    class Attn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.k_proj = torch.nn.Linear(8, 8)
            self.v_proj = torch.nn.Linear(8, 8)
            self.o_proj = torch.nn.Linear(8, 8)

    class Llama(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attn()
            self.gate_proj = torch.nn.Linear(8, 8)

    found = discover_target_modules(Llama())
    assert found == ["gate_proj", "k_proj", "o_proj", "q_proj", "v_proj"], found

    from lifemate.training.lora_finetuning import _resolve_targets  # noqa: PLC0415

    try:
        _resolve_targets(model, None)  # the discovery path configure_lora uses
    except LifeMateError as exc:
        message = str(exc)
        assert "discovery found nothing" in message, message
        assert "embed" in message or "head" in message, "the error must list discovered modules"
    else:
        raise AssertionError("empty target set must raise, never silently no-op")

    # an explicitly configured module that does not exist also raises
    try:
        _resolve_targets(model, ["q_proj"])
    except LifeMateError as exc:
        assert "not found in model" in str(exc), str(exc)  # match actual error message from _resolve_targets
    else:
        raise AssertionError("configured-but-absent target modules must raise")
    print(f"  discovery -> {found}; empty set raises with module list")

def test_lora_freezes_base_and_trains_only_adapters() -> None:
    """Adapter params are trainable, base is frozen, gradients reach the adapter."""
    if not _torch_available():
        print("  SKIP LoRA attach test: torch not installed")
        return
    try:
        import peft  # noqa: F401,PLC0415 - probe only
    except Exception as exc:
        print(f"  SKIP LoRA attach test: peft not installed ({exc})")
        return
    import torch  # noqa: PLC0415

    from lifemate.training.lora_finetuning import configure_lora  # noqa: PLC0415

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.v_proj = torch.nn.Linear(8, 8)
            self.lm_head = torch.nn.Linear(8, 4)
            self.config = type("Cfg", (), {"model_type": "test", "hidden_size": 8})()

        def forward(self, input_ids=None, labels=None, **kwargs):
            del kwargs
            hidden = self.v_proj(self.q_proj(torch.ones(1, 2, 8)))
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                loss = logits.sum() * 0.0
            return type("Out", (), {"loss": loss, "logits": logits})()

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            """peft requires this hook for task_type=CAUSAL_LM."""
            del kwargs
            return {"input_ids": input_ids}

    cfg = load_config()
    cfg.lora = LoraConfig(r=2, alpha=4, dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none")
    report = configure_lora(Model(), cfg)
    assert report["trainable_params"] > 0
    assert report["trainable_params"] < report["total_params"]
    assert report["target_modules"] == ["q_proj", "v_proj"]
    assert 0 < report["trainable_percentage"] < 100
    model = report["model"]
    adapter = [n for n, p in model.named_parameters() if p.requires_grad]
    assert adapter and all("lora_" in name for name in adapter), adapter

    out = model(input_ids=torch.ones(1, 2, dtype=torch.long))
    out.logits.sum().backward()
    adapter_grads = [
        p.grad.abs().sum().item() for n, p in model.named_parameters() if "lora_" in n and p.grad is not None
    ]
    assert adapter_grads and any(value > 0 for value in adapter_grads), "no gradient reached the adapter"
    base_frozen = [
        p.grad for n, p in model.named_parameters() if "lora_" not in n and p.grad is not None
    ]
    assert all(g is None or float(g.abs().sum()) == 0 for g in base_frozen), "base weights got gradients"
    print(
        f"  LoRA -> {report['trainable_params']:,} trainable "
        f"({report['trainable_percentage']}%), base frozen, adapter gradients flow"
    )

# --------------------------------------------------------------------------- #
# memory preflight (hand-computed expectations)
# --------------------------------------------------------------------------- #
def test_memory_preflight_math_is_exact() -> None:
    """Hand-computed GB figures for 1e9 params, seq 1024, 32 layers, hidden 2560.

    Full fine-tuning stores **fp32 master** weights and gradients (AdamW writes
    the update into the parameter tensor, so a bf16 master would quantise a
    5e-6 step away — see ``full_finetuning.master_dtype``), while activations
    still run at the compute dtype (bf16). Layer count is the released
    Qwen3.5-4B ``text_config`` value (32; the old default of 36 overestimated
    the activation term by 12.5 %).

    weights      = 1e9 * 4             = 4.00 GB
    gradients    = 1e9 * 4             = 4.00 GB
    optimizer    = 1e9 * 8             = 8.00 GB
    activations  = 32*1024*1*2560*10*2 = 1.678 GB (k=10, no grad checkpointing)
    total        = 17.678 GB           -> 17.68
    checkpoint   = weights + grads     = 8.00 GB
    """
    cfg = load_config()
    cfg.training.tuning_mode = "full"
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.gradient_checkpointing = False
    report = run_memory_preflight(
        param_count=1_000_000_000, cfg=cfg, seq_len=1024, gpu_info={"gpu": "unavailable"}
    )
    assert report["weight_bytes_per_param"] == 4, report
    assert report["weights_gb"] == 4.0, report
    assert report["gradients_gb"] == 4.0, report
    assert report["optimizer_gb"] == 8.0, report
    assert report["activations_gb"] == 1.68, report
    assert report["total_gb"] == 17.68, report
    assert report["checkpoint_size_gb"] == 8.0, report
    assert report["verdict"] == "unknown", report["verdict"]
    assert report["available_gb"] is None
    assert "layers(32)" in report["heuristic"], report["heuristic"]

    checkpointed = run_memory_preflight(
        param_count=1_000_000_000, cfg=_cfg_with(gradient_checkpointing=True), seq_len=1024,
        gpu_info={"gpu": "unavailable"},
    )
    # k=2: 32*1024*1*2560*2*2 = 335,544,320 bytes -> 0.34 GB
    assert checkpointed["activations_gb"] == 0.34, checkpointed
    assert checkpointed["total_gb"] == 16.34, checkpointed
    print("  preflight full -> 4.00 + 4.00 + 8.00 + 1.68 = 17.68 GB (hand-checked, fp32 masters, 32 layers)")

def test_memory_preflight_verdicts_and_lora() -> None:
    """fits/tight/does_not_fit boundaries and the LoRA accounting."""
    cfg = _cfg_with(gradient_checkpointing=False, tuning_mode="full")
    total_bytes = int(17.6777216e9)  # 4+4+8 GB + 32*1024*2560*10*2 bytes
    for free, expected in (
        (total_bytes * 2, "fits"),
        (int(total_bytes / 0.95), "tight"),
        (int(total_bytes / 2), "does_not_fit"),
    ):
        report = run_memory_preflight(
            param_count=1_000_000_000,
            cfg=cfg,
            seq_len=1024,
            gpu_info={"gpu": "available", "free_bytes": free, "total_bytes": free, "name": "TestGPU",
                      "free_gb": free / 1e9, "total_gb": free / 1e9},
        )
        assert report["verdict"] == expected, (free, report["verdict"], expected)

    lora = run_memory_preflight(
        param_count=1_000_000_000,
        cfg=_cfg_with(gradient_checkpointing=True, tuning_mode="lora", per_device_train_batch_size=1),
        seq_len=1024,
        gpu_info={"gpu": "unavailable"},
    )
    # adapter = 0.5% of 1e9 = 5e6 params -> grads 0.01 GB, optimizer 0.04 GB
    assert lora["trainable_params"] == 5_000_000, lora
    assert lora["weights_gb"] == 2.0
    assert lora["gradients_gb"] == 0.01, lora
    assert lora["optimizer_gb"] == 0.04, lora
    # 2.0 weights + 0.01 grads + 0.04 optimizer + 0.34 activations
    assert lora["total_gb"] == 2.39, lora
    assert lora["activations_gb"] == 0.34, lora
    text = format_report(lora)
    assert "verdict" in text and "TOTAL" in text and "estimate, not a measurement" in text
    print("  preflight -> 3 verdicts correct; LoRA 2.39 GB vs 16.34 GB full (fp32 masters, 32 layers)")

def test_real_run_reports_measured_trainable_after_lora_configure() -> None:
    """A non-dry-run path preflights twice: first without weights, then with the model.

    The first call happens before any weights exist, so it can only print the
    ``placeholder_0.5pct`` count. The second, after ``configure_lora`` attached the
    adapters, must receive the model — that is the call whose report says
    ``measured`` (docs/FINAL_TRAINING_EXPERIMENT_READINESS.md §12 bước 3).
    """
    if not (_torch_available() and _transformers_available()):
        print("  SKIP two-preflight test: torch/transformers not installed")
        return
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("train_script_preflight", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[bool] = []
    real = module._preflight

    def recording_preflight(cfg, args, model=None):
        calls.append(model is None)
        return real(cfg, args, model=model)

    module._preflight = recording_preflight
    with tempfile.TemporaryDirectory() as tmp:
        code = module.main(["--smoke", "--tuning", "lora", "--max-steps", "1", "--output-dir", tmp])
    assert code == 0, code
    assert calls == [True, False], f"expected [no model, model], got {calls}"
    print("  two preflights -> first without model (placeholder), second with the LoRA-wrapped model (measured)")


def _cfg_with(**overrides):
    cfg = load_config()
    for key, value in overrides.items():
        setattr(cfg.training, key, value)
    return cfg

def test_memory_preflight_reports_gpu_honestly() -> None:
    """With no CUDA the report says unavailable and never invents a measurement."""
    report = run_memory_preflight(param_count=1_000_000_000, cfg=load_config(), seq_len=1024)
    gpu = report["gpu"]
    if not _torch_available():
        assert gpu["gpu"] == "unavailable"
        assert "torch" in gpu["reason"].lower()
        assert report["verdict"] == "unknown"
        assert report["suggestions"]
        print("  preflight GPU -> unavailable (torch missing), verdict unknown, suggestions given")
    else:
        assert gpu["gpu"] in {"available", "unavailable"}
        print(f"  preflight GPU -> {gpu['gpu']} (torch present)")

def test_memory_preflight_rejects_bad_seq_len() -> None:
    """seq_len must be positive."""
    try:
        run_memory_preflight(param_count=1_000_000, cfg=load_config(), seq_len=0)
    except LifeMateError:
        print("  preflight -> seq_len 0 rejected")
        return
    raise AssertionError("seq_len=0 must raise")

def test_optimizer_steps_always_progress() -> None:
    """A dataset smaller than one effective batch still yields a step per epoch.

    total_steps is derived from this helper, so the loop can never spin without
    taking an optimizer step (the bug class "accumulate forever, never step").
    """
    assert optimizer_steps_per_epoch(1, 1, 8) == 1
    assert optimizer_steps_per_epoch(0, 1, 8) == 1
    assert optimizer_steps_per_epoch(8, 1, 8) == 1
    assert optimizer_steps_per_epoch(9, 1, 8) == 2
    assert optimizer_steps_per_epoch(16, 2, 4) == 2
    assert optimizer_steps_per_epoch(17, 2, 4) == 3
    assert optimizer_steps_per_epoch(100, 4, 4) == 7   # 6 full groups + 1 flush
    print("  schedule -> 10/16 partial groups still step (no infinite accumulation)")

# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def _fake_save(target: Path, payload: dict) -> Path:
    """Writer that records what it was asked to save (no torch)."""
    (target / "weights.bin").write_text(
        json.dumps({"step": payload["state"]["step"], "tag": payload.get("tag")}), encoding="utf-8"
    )
    return target

def test_checkpoint_versioning_and_no_overwrite() -> None:
    """step-<n> dirs are created once; a second save at the same step fails."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(tmp, run_name="run", save_total_limit=0, save_fn=_fake_save)
        path = manager.save(1, {"state": {"step": 1}})
        assert path == Path(tmp) / "run" / "step-1"
        assert (path / "training_state.json").is_file()
        assert read_training_state(path)["step"] == 1
        try:
            manager.save(1, {"state": {"step": 1}})
        except LifeMateError as exc:
            assert "already populated" in str(exc), str(exc)
        else:
            raise AssertionError("re-saving an existing step must raise, never overwrite")
        interrupt = manager.save(2, {"state": {"step": 2}}, tag="interrupt")
        crash = manager.save(2, {"state": {"step": 2}}, tag="crash")
        assert interrupt.name == "interrupt-2" and crash.name == "crash-2"
        assert manager.latest().name == "step-1", manager.latest()
        print("  checkpoints -> step-1 once, re-save refused, interrupt/crash tags separate")

def test_checkpoint_pruning_keeps_newest() -> None:
    """save_total_limit prunes the oldest step dirs only."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(tmp, run_name="run", save_total_limit=2, save_fn=_fake_save)
        for step in (10, 20, 30, 40):
            manager.save(step, {"state": {"step": step}})
        remaining = sorted(p.name for p in (Path(tmp) / "run").glob("step-*"))
        assert remaining == ["step-30", "step-40"], remaining
        assert manager.latest().name == "step-40"
        print("  pruning -> kept step-30, step-40 (save_total_limit=2)")

def test_checkpoint_meta_round_trip_and_manifest_hash() -> None:
    """training_state.json round-trips step/epoch/manifest hash."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(tmp, run_name="run", save_total_limit=1, save_fn=_fake_save)
        state = {"step": 7, "epoch": 2, "seed": 42, "dataset_manifest_hash": "abc123", "is_lora": False}
        manager.save(7, {"state": state})
        restored = read_training_state(Path(tmp) / "run" / "step-7")
        assert restored["step"] == 7 and restored["epoch"] == 2
        assert restored["dataset_manifest_hash"] == "abc123"
        assert read_training_state(Path(tmp) / "run" / "nope") == {}
        print("  state -> step/epoch/manifest hash round-trip")

def test_resume_guards_mismatch() -> None:
    """Resume refuses a LoRA↔full mismatch, a missing state, and a hash change."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(tmp, run_name="run", save_total_limit=0, save_fn=_fake_save)
        path = manager.save(
            5, {"state": {"step": 5, "is_lora": False, "dataset_manifest_hash": "hash_a"}}
        )

        def _trainer(**kwargs):
            return SFTTrainer(
                model=_StubModel(),
                tokenizer=_CharTokenizer(),
                collator=SupervisedCollator(pad_token_id=0),
                train_records=[{"messages": _messages(), "target": '<action>{"type": "finish"}</action>'}],
                cfg=_cfg_with(tuning_mode="full"),
                checkpoint_manager=CheckpointManager(tmp, run_name="other", save_fn=_fake_save),
                **kwargs,
            )

        assert _trainer().resume(path) == 5
        for kwargs, needle in (
            ({"is_lora": True}, "lora"),
            ({"dataset_manifest_hash": "hash_b"}, "manifest"),
        ):
            try:
                _trainer(**kwargs).resume(path)
            except LifeMateError as exc:
                assert needle in str(exc).lower(), str(exc)
            else:
                raise AssertionError(f"resume with {kwargs} must fail")
        try:
            _trainer().resume(Path(tmp) / "run" / "step-999")
        except LifeMateError as exc:
            assert "training_state" in str(exc) or "no training_state" in str(exc).lower()
        else:
            raise AssertionError("resume from a stateless dir must fail")
        print("  resume -> step 5 restored; lora/full, hash change, stateless dir all fail")

class _StubModel:
    """Model that accepts resume calls (no torch)."""

    def load_state_dict_from_checkpoint(self, path: Path) -> None:
        self.loaded_from = str(path)

    def parameters(self):
        return []

# --------------------------------------------------------------------------- #
# trainer assembly (no torch needed)
# --------------------------------------------------------------------------- #
def test_trainer_drop_accounting_and_empty_guard() -> None:
    """Dropped examples are counted per reason; zero usable examples raises."""
    trainer = SFTTrainer(
        model=_StubModel(),
        tokenizer=_CharTokenizer(),
        collator=SupervisedCollator(pad_token_id=0),
        train_records=[
            {"messages": _messages(), "target": '<action>{"type": "finish"}</action>'},
            {"messages": _messages(), "target": '<action>{"type": "finish"}</action>' * 40},
        ],
        cfg=_cfg_with(max_seq_length=64),
        checkpoint_manager=CheckpointManager("/tmp/unused", save_fn=_fake_save),
    )
    examples = trainer._prepare_examples(trainer.train_records, "train")
    assert len(examples) == 1, len(examples)
    record = trainer.metrics_history[-1]
    assert record["dropped"]["target_too_long"] == 1, record

    bad = SFTTrainer(
        model=_StubModel(),
        tokenizer=_CharTokenizer(),
        collator=SupervisedCollator(pad_token_id=0),
        train_records=[{"messages": _messages(), "target": '<action>{"type": "finish"}</action>' * 40}],
        cfg=_cfg_with(max_seq_length=64),
        checkpoint_manager=CheckpointManager("/tmp/unused", save_fn=_fake_save),
    )
    try:
        bad._prepare_examples(bad.train_records, "train")
    except LifeMateError as exc:
        assert "target_too_long" in str(exc), str(exc)
    else:
        raise AssertionError("no usable examples must raise")
    assert bad.metrics_history[-1]["kept"] == 0, (
        "the drop counters must be recorded even when the split has no usable example: "
        f"{bad.metrics_history}"
    )

    # Regression: an empty record list must reach the LifeMateError, not NameError
    # (the per-record diagnostics entry has to be appended unconditionally), and a
    # degenerate trailing record must not suppress the counters entry.
    empty = SFTTrainer(
        model=_StubModel(),
        tokenizer=_CharTokenizer(),
        collator=SupervisedCollator(pad_token_id=0),
        train_records=[],
        cfg=_cfg_with(max_seq_length=64),
        checkpoint_manager=CheckpointManager("/tmp/unused", save_fn=_fake_save),
    )
    try:
        empty._prepare_examples([], "val")
    except LifeMateError as exc:
        assert "no usable val examples" in str(exc), str(exc)
    else:
        raise AssertionError("an empty record list must raise LifeMateError")
    assert [entry["split"] for entry in empty.metrics_history] == ["val"], empty.metrics_history
    assert empty.metrics_history[-1]["kept"] == 0, empty.metrics_history

    degenerate = SFTTrainer(
        model=_StubModel(),
        tokenizer=_CharTokenizer(),
        collator=SupervisedCollator(pad_token_id=0),
        train_records=[],
        cfg=_cfg_with(max_seq_length=64),
        checkpoint_manager=CheckpointManager("/tmp/unused", save_fn=_fake_save),
    )
    kept = degenerate._prepare_examples(
        [{"messages": _messages(), "target": '<action>{"type": "finish"}</action>'}, {}], "train"
    )
    assert len(kept) == 1, len(kept)
    assert degenerate.metrics_history[-1]["kept"] == 1, (
        f"a trailing {{}} record must not skip the counters entry: {degenerate.metrics_history}"
    )
    print(
        "  trainer prep -> target_too_long counted; all-dropped split raises; "
        "empty list raises LifeMateError with counters recorded; trailing {} keeps counters"
    )

# --------------------------------------------------------------------------- #
# trainer loop (torch-gated): interrupt + crash safety
# --------------------------------------------------------------------------- #
def _loop_trainer(model, records, cfg, tmp, **kwargs):
    tokenizer = _CharTokenizer()
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        collator=SupervisedCollator.from_tokenizer(tokenizer),
        train_records=records,
        cfg=cfg,
        device="cpu",
        checkpoint_manager=CheckpointManager(
            tmp, run_name="run", save_total_limit=0, save_fn=_fake_save
        ),
        **kwargs,
    )

def _loop_cfg(**overrides):
    cfg = load_config()
    cfg.training.precision = "float32"
    cfg.training.gradient_checkpointing = False
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.logging_steps = 1
    cfg.training.save_total_limit = 0
    for key, value in overrides.items():
        setattr(cfg.training, key, value)
    return cfg

def test_trainer_interrupt_saves_checkpoint_and_exits_130() -> None:
    """A stop request finishes the step, saves interrupt-<step>, raises exit 130."""
    if not _torch_available():
        print("  SKIP interrupt test: torch not installed")
        return
    import torch  # noqa: PLC0415

    from lifemate.training.callbacks import StopController  # noqa: PLC0415
    from lifemate.training.trainer import TrainingInterrupted  # noqa: PLC0415

    torch.manual_seed(0)
    records = _loop_records(6)
    stop = StopController()
    trainer = _loop_trainer(_tiny_lm(), records, _loop_cfg(), tempfile.mkdtemp(), max_steps=5, stop=stop)
    original = trainer._maybe_log

    def hook(step: int, loss: float | None) -> None:
        original(step, loss)
        if step == 2:
            stop.request(2)

    trainer._maybe_log = hook  # type: ignore[assignment]
    try:
        trainer.run()
    except TrainingInterrupted as exc:
        assert exc.step == 2, exc.step
        assert exc.exit_code == 130
        assert Path(exc.checkpoint_dir).name == "interrupt-2", exc.checkpoint_dir
        assert read_training_state(exc.checkpoint_dir)["step"] == 2
        print("  interrupt -> step 2 finished, interrupt-2 saved, exit code 130")
    else:
        raise AssertionError("a stop request must raise TrainingInterrupted")

def test_trainer_crash_writes_state_and_reraises() -> None:
    """An unhandled error writes crash-<step> state and re-raises the original."""
    if not _torch_available():
        print("  SKIP crash-safety test: torch not installed")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(0)

    class Boom(SFTTrainer):
        def _save(self, step: int, epoch: int, tag: str = "step") -> Path:
            raise RuntimeError("disk on fire")

    trainer = Boom(
        model=_tiny_lm(),
        tokenizer=_CharTokenizer(),
        collator=SupervisedCollator.from_tokenizer(_CharTokenizer()),
        train_records=_loop_records(4),
        cfg=_loop_cfg(),
        device="cpu",
        checkpoint_manager=CheckpointManager(
            tempfile.mkdtemp(), run_name="run", save_fn=_fake_save
        ),
        max_steps=2,
    )
    try:
        trainer.run()
    except RuntimeError as exc:
        assert "disk on fire" in str(exc), "the original error must not be masked"
        crash_states = [
            path for path in Path(trainer.checkpoint_manager.root).rglob("training_state.json")
            if path.parent.name.startswith("crash-")
        ]
        assert crash_states, "crash-<step> state must be written"
        assert read_training_state(crash_states[0].parent)["step"] == 2
        print("  crash -> crash-2 state written, original error re-raised")
    else:
        raise AssertionError("an unhandled exception must propagate")

def _loop_records(n: int) -> list[dict]:
    return [
        {"messages": _messages(f"x{index}"), "target": '<action>{"type": "finish"}</action>'}
        for index in range(n)
    ]

def test_trainer_full_loop_learns_and_checkpoints() -> None:
    """End-to-end CPU loop: steps, val metrics, checkpoint on disk (torch-gated)."""
    if not _torch_available():
        print("  SKIP trainer loop test: torch not installed")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    trainer = _loop_trainer(
        _tiny_lm(),
        _loop_records(5),
        _loop_cfg(),
        tempfile.mkdtemp(),
        max_steps=3,
        val_records=_loop_records(2),
    )
    result = trainer.run()
    assert result.steps == 3, result.steps
    assert result.final_loss is not None and result.final_loss > 0
    assert result.val_loss is not None and result.val_loss > 0
    assert result.checkpoint_dir and Path(result.checkpoint_dir).name == "step-3"
    assert read_training_state(result.checkpoint_dir)["step"] == 3
    val_records = [m for m in result.metrics_history if "val_loss" in m]
    assert val_records and "val_token_accuracy" in val_records[0]
    print(
        f"  loop -> 3 steps, train_loss={result.final_loss:.3f}, "
        f"val_loss={result.val_loss:.3f}, step-3 on disk"
    )

# --------------------------------------------------------------------------- #
# dependency contracts (signatures owned by other workers)
# --------------------------------------------------------------------------- #
def test_model_loader_call_site_matches_real_signature() -> None:
    """``scripts/train.py`` calls the keyword-only loader with only real keywords.

    The loader signature lives in another worker's slice and is keyword-only, so
    a positional/keyword drift is a silent runtime break. Parsed with ``ast`` so
    the check holds without torch/transformers installed.
    """
    import ast  # noqa: PLC0415

    loader_src = (ROOT / "src" / "lifemate" / "models" / "loader.py").read_text(encoding="utf-8")
    signature = next(
        node.args
        for node in ast.walk(ast.parse(loader_src))
        if isinstance(node, ast.FunctionDef) and node.name == "load_model_and_tokenizer"
    )
    allowed = {arg.arg for arg in signature.args} | {arg.arg for arg in signature.kwonlyargs}
    assert signature.vararg is None and signature.kwarg is None, "loader must not take *args/**kwargs"

    calls = [
        node
        for node in ast.walk(ast.parse((ROOT / "scripts" / "train.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_model_and_tokenizer"
    ]
    assert len(calls) == 1, f"expected one loader call site, found {len(calls)}"
    call = calls[0]
    assert len(call.args) == 1, f"loader takes one positional argument, call passes {len(call.args)}"
    unknown = [kw.arg for kw in call.keywords if kw.arg not in allowed]
    assert not unknown, f"unknown loader keywords passed: {unknown}"
    required = {"trainable", "for_training", "tuning_mode", "torch_dtype", "device"}
    missing = required - {kw.arg for kw in call.keywords}
    assert not missing, f"training-relevant loader keywords missing: {sorted(missing)}"
    print(f"  loader contract -> 1 positional + {len(call.keywords)} real keywords, none unknown")

def test_protocol_serialization_round_trip() -> None:
    """Targets serialise through the frozen protocol API, verbatim."""
    from lifemate.agent.protocol import (  # noqa: PLC0415
        FINISH_ACTION,
        make_tool_call,
        parse_action,
        render_action,
    )

    action = make_tool_call("alarm.set", {"time": "06:30", "label": "sáng"})
    text = render_action(action)
    assert text == (
        '<action>{"type": "tool_call", "name": "alarm.set", '
        '"arguments": {"time": "06:30", "label": "sáng"}}</action>'
    ), text
    parsed = parse_action(text)
    assert parsed.type == "tool_call" and parsed.to_dict() == action

    finish_text = render_action(FINISH_ACTION)
    assert finish_text == '<action>{"type": "finish"}</action>', finish_text
    assert parse_action(finish_text).type == "finish"

    # a DecisionPoint target round-trips the same way the trainer will read it
    point = DecisionPoint(
        sample_id="s", step_index=0, prompt_body="p", target_action=action
    )
    assert point.target_text() == text
    assert parse_action(point.target_text()).to_dict() == action
    print("  protocol -> render/parse round-trip byte-identical for call and finish")

def test_tracker_uses_redacting_wandb_tracker_with_jsonl_fallback() -> None:
    """``_build_tracker`` never hand-rolls wandb.init; redaction is active.

    Two separate assertions:

    1. the real :func:`lifemate.tracking.sanitize` (the policy WandbTracker
       applies to every payload) redacts secrets and drops PII — called directly,
       no network, no run;
    2. with W&B off/unavailable ``scripts.train._build_tracker`` returns a local
       JSONL logger, and its written lines contain no secret or PII values.

    GPU/wandb-online behaviour is BLOCKED here: no GPU, no W&B credentials/network.
    """
    import importlib.util  # noqa: PLC0415

    from lifemate.tracking.wandb_tracker import PII_KEYS, REDACTED, sanitize  # noqa: PLC0415

    payload = {
        "api_key": "sk-live-123",
        "HF_TOKEN": "hf_secret",
        "password": "hunter2",
        "client_secret": "abc",
        "user_profile": {"name": "Nguyen Van A", "phone": "+84900000000"},
        "audio_text": "Dat bao thuc luc 6 gio",
        "learning_rate": 5e-6,
        "nested": {"access_token": "tok", "step": 3},
    }
    clean = sanitize(payload)
    for key in ("api_key", "HF_TOKEN", "password", "client_secret"):
        assert clean[key] == REDACTED, (key, clean[key])
    for key in ("user_profile", "audio_text"):
        assert key in PII_KEYS and clean[key] != payload[key], (key, clean[key])
    assert clean["nested"]["access_token"] == REDACTED
    assert clean["learning_rate"] == 5e-6 and clean["nested"]["step"] == 3
    flat = json.dumps(clean, ensure_ascii=False)
    for secret in ("sk-live-123", "hf_secret", "hunter2", "abc", "Nguyen Van A", "84900000000"):
        assert secret not in flat, f"{secret!r} survived sanitisation"

    # the module's own builder: W&B disabled => local JSONL, no wandb import
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from lifemate.training.callbacks import JsonlTracker, NullTracker  # noqa: PLC0415

    cfg = load_config()
    assert isinstance(module._build_tracker(cfg, _args(), run_name="full"), NullTracker)

    cfg.training.report_to = ["wandb"]
    cfg.wandb.enabled = False
    with tempfile.TemporaryDirectory() as tmp:
        cfg.training.output_dir = tmp
        tracker = module._build_tracker(cfg, _args(), run_name="full")
        assert isinstance(tracker, JsonlTracker), type(tracker).__name__
        assert not hasattr(tracker, "_run"), "the fallback must not touch wandb"
        tracker.log({"step": 1, "train_loss": 1.5, "api_key": "sk-live-123",
                     "user_profile": {"phone": "+84900000000"}, "audio_text": "Dat bao thuc"})
        tracker.log({"step": 2, "train_loss": 1.4})
        tracker.finish()
        lines = (Path(tmp) / "full" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines
    first = json.loads(lines[0])
    assert first["train_loss"] == 1.5 and first["step"] == 1
    assert first["api_key"] == REDACTED, first
    raw = "".join(lines)
    for secret in ("sk-live-123", "84900000000", "Dat bao thuc"):
        assert secret not in raw, f"{secret!r} reached the metrics log"
    print(
        "  tracker -> secrets redacted, PII dropped, JSONL fallback used "
        "(wandb-online BLOCKED: no creds/network)"
    )

def _args():
    """Minimal argparse namespace for `_build_tracker`."""
    import argparse  # noqa: PLC0415

    return argparse.Namespace(smoke=False)

# --------------------------------------------------------------------------- #
# CLI behaviour
# --------------------------------------------------------------------------- #
def test_cli_missing_dataset_exits_2() -> None:
    """A missing SFT dataset exits 2 with an actionable message."""
    out = _run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/training_full.yaml",
            "--dataset-dir",
            "/nonexistent/sft",
        ]
    )
    assert out.returncode == 2, (out.returncode, out.stdout, out.stderr)
    assert "Dataset not found." in out.stdout, out.stdout
    assert "prepare_dataset.py" in out.stdout, "the message must say how to build the dataset"
    assert "Traceback" not in out.stdout + out.stderr
    print("  CLI -> missing dataset exits 2 with no traceback")

def test_cli_prepare_dataset_requires_data() -> None:
    """prepare_dataset exits 2 on a missing data root."""
    out = _run(
        [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--data-root",
            "/nonexistent/data",
            "--split",
            "train",
            "--output",
            "/tmp/never",
        ]
    )
    assert out.returncode == 2, (out.returncode, out.stdout, out.stderr)
    assert "Dataset not found." in out.stdout
    assert "Traceback" not in out.stdout + out.stderr
    print("  CLI -> prepare_dataset missing data exits 2, no traceback")

def test_prepare_dataset_end_to_end() -> None:
    """prepare_dataset writes train/val JSONL + manifest + stats."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _run(
            [
                sys.executable,
                "scripts/prepare_dataset.py",
                "--data-root",
                "tests/fixtures/synthetic_lifemate",
                "--split",
                "train",
                "--output",
                tmp,
            ]
        )
        assert out.returncode == 0, (out.stdout, out.stderr)
        split_dir = Path(tmp) / "train"
        for name in ("train.jsonl", "val.jsonl", "manifest.json", "stats.json"):
            assert (split_dir / name).is_file(), name
        records = [json.loads(line) for line in (split_dir / "train.jsonl").read_text().splitlines() if line]
        assert records, "train.jsonl must not be empty"
        assert set(records[0]) == {"sample_id", "step_index", "messages", "target", "meta"}
        manifest = json.loads((split_dir / "manifest.json").read_text())
        assert manifest["n_train"] + manifest["n_val"] == 14, manifest  # 14 fixture samples
        assert manifest["tokenization"].startswith("deferred to training"), manifest["tokenization"]
        assert json.loads((split_dir / "stats.json").read_text())["train"]["n_decision_points"] == len(records)
        print(f"  prepare_dataset -> {len(records)} train records + manifest/stats written")

def test_prepare_dataset_manifest_names_dropped_decision_points() -> None:
    """The manifest must name decision points the builder refused to supervise.

    Regression: ``build_sft_dataset`` drops a point whose gold argument the
    prompt cannot show (``unobservable_ref_target``), but ``prepare_dataset``
    only wrote ``n_train_decision_points`` / ``n_val_decision_points``. On the
    real split that made 4385 stored points look like a complete build while the
    current builder produces 4368 (17 dropped) — the loss was invisible in
    ``manifest.json`` and ``stats.json``'s sample-level ``n_dropped`` stayed 0.
    Both files must now carry the count and the per-reason breakdown.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = _run(
            [
                sys.executable,
                "scripts/prepare_dataset.py",
                "--data-root",
                "tests/fixtures/synthetic_lifemate",
                "--split",
                "train",
                "--output",
                tmp,
            ]
        )
        assert out.returncode == 0, (out.stdout, out.stderr)
        split_dir = Path(tmp) / "train"
        records = [json.loads(l) for l in (split_dir / "train.jsonl").read_text().splitlines() if l]
        manifest = json.loads((split_dir / "manifest.json").read_text())
        stats = json.loads((split_dir / "stats.json").read_text())
        assert manifest["n_train_decision_points"] == len(records)
        touched = manifest["n_train_dropped_decision_points"]
        assert touched == stats["train"]["n_dropped_decision_points"], (
            manifest["n_train_dropped_decision_points"],
            stats["train"]["n_dropped_decision_points"],
        )
        assert manifest["dropped_decision_points"]["train"] == stats["train"]["dropped_decision_points"]
        assert manifest["n_val_dropped_decision_points"] == stats["val"]["n_dropped_decision_points"]
        # The fixture build really does drop points, so a zero here would mean
        # the wiring silently reported nothing.
        assert touched > 0, (touched, manifest["dropped_decision_points"])
        assert out.stdout.count("decision points dropped") >= 1, out.stdout
        print(f"  prepare_dataset -> manifest names {touched} dropped decision point(s)")

def test_prepare_dataset_fit_reports_truncation_not_a_false_fit() -> None:
    """`_fit_stats` must not read "everything fits" while the collator truncates.

    Regression for a real misinformation bug: the manifest said
    `n_fit: 4385, n_over_budget: 0` for the real split while 75% of those
    records were middle-truncated at 4096. Truncation (kept, middle cut) and
    dropping (kept=False) are different outcomes and are reported separately.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_dataset_mod", ROOT / "scripts" / "prepare_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeTokenizer:
        """One token per character: a 40-char prompt needs 40 tokens."""

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))

        eos_token_id = 0

    messages = [{"role": "user", "content": "# Yêu cầu\n" + "x" * 40}]
    records = [{"messages": messages, "target": "short", "meta": {"token_estimate": 400}}]
    assert records

    # no tokenizer -> measured=None, an explicit caveat, never a silent "all fit"
    unknown = module._fit_stats(records, None, 4096)
    assert unknown["method"] == "chars4_estimate", unknown
    assert unknown["n_truncated"] is None, unknown
    assert unknown["n_dropped"] is None, unknown
    assert unknown["caveat"], "an estimate must be labelled as one"

    # real-ish tokenizer, generous budget -> nothing truncated, nothing dropped
    roomy = module._fit_stats(records, FakeTokenizer(), 4096)
    assert roomy["method"] == "tokenizer"
    assert roomy["n_truncated"] == 0 and roomy["n_dropped"] == 0, roomy
    assert roomy["n_truncated_fraction"] == 0.0

    # tight budget -> truncated, but still KEPT (never counted as a drop)
    tight = module._fit_stats(records, FakeTokenizer(), 20)
    assert tight["n_truncated"] == 1, tight
    assert tight["n_dropped"] == 0, tight
    assert tight["n_fit"] == 1, tight
    print("  _fit_stats -> truncation and dropping reported separately, estimates flagged")

def test_dry_run_prints_preflight_and_plan() -> None:
    """--dry-run --param-count prints the estimate and never loads a model."""
    out = _run(
        [
            sys.executable,
            "scripts/train.py",
            "--dry-run",
            "--param-count",
            "4000000000",
            "--tuning",
            "full",
            "--config",
            "configs/training_full.yaml",
        ]
    )
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "Memory pre-flight" in out.stdout
    # 4e9 params, seq 8192, grad ckpt, 32 layers: weights 16.0 (fp32 master)
    # + grads 16.0 + AdamW 32.0 + activations 32*8192*1*2560*2*2 = 2.68 GB
    assert "TOTAL              : 66.68 GB" in out.stdout, out.stdout
    assert "weights            : 16.0 GB" in out.stdout, out.stdout
    assert "sequence length    : 8192" in out.stdout, out.stdout
    assert "layers(32)" in out.stdout, out.stdout
    assert '"tuning_mode": "full"' in out.stdout
    assert "no model loaded, no training launched" in out.stdout
    print("  dry run full -> 66.68 GB at seq 8192 (fp32 masters, 32 layers), plan printed, exit 0")

    lora = _run(
        [
            sys.executable,
            "scripts/train.py",
            "--dry-run",
            "--param-count",
            "4000000000",
            "--tuning",
            "lora",
            "--config",
            "configs/training_lora.yaml",
        ]
    )
    assert lora.returncode == 0, (lora.stdout, lora.stderr)
    assert "tuning mode        : lora" in lora.stdout
    # configs/training_lora.yaml: batch 1, accum 8, seq 8192, grad ckpt.
    # weights 8.00 + grads 0.04 + optimizer 0.16 + activations 2.68 = 10.88 GB
    assert "gradients          : 0.04 GB" in lora.stdout, lora.stdout
    assert "optimizer state    : 0.16 GB" in lora.stdout, lora.stdout
    assert "sequence length    : 8192 (microbatch 1, accum 8)" in lora.stdout, lora.stdout
    assert "TOTAL              : 10.88 GB" in lora.stdout, lora.stdout
    print("  dry run lora -> 10.88 GB at seq 8192 (microbatch 1 x accum 8), exit 0")

def test_smoke_reports_environment_honestly() -> None:
    """--smoke runs end-to-end with a tiny model, or exits 3 saying why."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _run(
            [sys.executable, "scripts/train.py", "--smoke", "--output-dir", tmp, "--max-steps", "2"]
        )
        checkpoint_dirs = [p for p in Path(tmp).rglob("step-*") if p.is_dir()]
        if _torch_available() and _transformers_available():
            assert out.returncode == 0, (out.returncode, out.stdout, out.stderr)
            assert checkpoint_dirs, f"SMOKE (CPU, tiny model) produced no checkpoint in {tmp}"
            assert "done in" in out.stdout
            print(f"  SMOKE (CPU, tiny model) -> exit 0, {len(checkpoint_dirs)} checkpoint(s)")
        else:
            assert out.returncode == 3, (out.returncode, out.stdout, out.stderr)
            assert "environment error" in out.stdout
            assert "torch" in out.stdout or "transformers" in out.stdout
            assert not checkpoint_dirs, "no partial checkpoint may be written on an env failure"
            print(
                "  BLOCKED: torch/transformers unavailable -> --smoke exits 3 with the "
                "documented message and writes no partial checkpoint"
            )
def test_evaluation_shifted_accuracy_sft_01() -> None:
    """SFT-01 regression: evaluate() compares shifted logits against shifted labels."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP SFT-01 evaluation test: torch/transformers not installed")
        return
    import torch  # noqa: PLC0415
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=False)
        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        target = '<action>{"type": "finish"}</action>'
        ex = build_supervised_example(bundle.tokenizer, _messages(), target, 128)
        trainer = SFTTrainer(model=bundle.model, tokenizer=bundle.tokenizer, collator=collator, train_records=[], val_records=[], cfg=load_config())

        vocab_size = bundle.model.config.vocab_size

        class PerfectModel(torch.nn.Module):
            def forward(self, input_ids, labels=None, **kwargs):
                B, L = input_ids.shape
                logits = torch.zeros(B, L, vocab_size)
                for b in range(B):
                    for i in range(L - 1):
                        logits[b, i, input_ids[b, i + 1]] = 100.0
                return type("Out", (), {"loss": torch.tensor(0.1), "logits": logits})()

        trainer.model = PerfectModel()
        metrics_perf = trainer.evaluate([ex])
        assert metrics_perf["val_token_accuracy"] == 1.0, metrics_perf
        assert metrics_perf["n_supervised_tokens"] == ex["n_target_tokens"]

        class ImperfectModel(torch.nn.Module):
            def forward(self, input_ids, labels=None, **kwargs):
                B, L = input_ids.shape
                logits = torch.zeros(B, L, vocab_size)
                for b in range(B):
                    for i in range(L - 1):
                        tok = input_ids[b, i + 1] if i % 2 == 0 else (input_ids[b, i + 1] + 1) % vocab_size
                        logits[b, i, tok] = 100.0
                return type("Out", (), {"loss": torch.tensor(0.5), "logits": logits})()

        trainer.model = ImperfectModel()
        metrics_imp = trainer.evaluate([ex])
        assert 0.0 < metrics_imp["val_token_accuracy"] < 1.0, metrics_imp
        print("  SFT-01 -> shifted logits predict target tokens (perfect: 1.0, imperfect: < 1.0 and > 0)")

def test_trainer_no_double_save_on_final_step_sft_04() -> None:
    """SFT-04 regression: save_strategy='steps' with final step save does not double-save."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP SFT-04 test: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        ckpt_root = Path(tmp) / "ckpts"
        cfg = load_config(None)
        cfg.training.output_dir = str(ckpt_root)
        cfg.training.precision = "float32"
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 2
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.save_strategy = "steps"
        cfg.training.logging_steps = 2
        cfg.model.device = "cpu"

        bundle = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)
        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        trainer = SFTTrainer(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            collator=collator,
            train_records=_synthetic_records(4),
            val_records=[],
            cfg=cfg,
            device="cpu",
            max_steps=2,
        )
        trainer.run()
        step_dirs = [p for p in (ckpt_root / "sft").glob("step-*") if p.is_dir()]
        assert len(step_dirs) == 1, step_dirs
        assert step_dirs[0].name == "step-2"
        print("  SFT-04 -> exactly one step-2 dir saved; double save prevented")

#: ``text_config`` from the released ``Qwen/Qwen3.5-4B`` ``config.json`` (the
#: architecture Qwen3_5ForCausalLM builds from it). Copied verbatim so the test
#: needs neither the HF cache nor a weight download; only the numerics-free parts
#: that decide module names are kept (layers, layer_types, sizes, head counts).
QWEN3_5_4B_TEXT_CONFIG = dict(
    attention_bias=False,
    attention_dropout=0.0,
    attn_output_gate=True,
    dtype="bfloat16",
    eos_token_id=248044,
    full_attention_interval=4,
    head_dim=256,
    hidden_act="silu",
    hidden_size=2560,
    initializer_range=0.02,
    intermediate_size=9216,
    # Released pattern: full_attention at indices 3,7,11,...,31 (every 4th layer).
    layer_types=["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"]
    + ["linear_attention"] * 3 + ["full_attention"],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_value_head_dim=128,
    max_position_embeddings=32768,  # 262144 in the release; irrelevant for names
    mlp_only_layers=[],
    mtp_num_hidden_layers=1,
    mtp_use_dedicated_embeddings=False,
    num_attention_heads=16,
    num_hidden_layers=32,
    num_key_value_heads=4,
    rms_norm_eps=1e-06,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=248320,
)

def test_lora_target_modules_exist_in_released_qwen3_5_architecture() -> None:
    """Every discovered (or configured) LoRA target must name a real module.

    Building on the ``meta`` device costs no memory, so the 4.2 B-parameter
    architecture can be inspected on this CPU box. A target pattern that matches
    no module is silent no-op training (peft would adapt nothing), so the shipped
    auto-discovery and each configured suffix are asserted against the real
    module tree — including the hybrid ``linear_attention`` blocks of Qwen3.5,
    which plain attention-only suffix lists miss.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP released-architecture LoRA test: torch/transformers not installed")
        return
    import torch  # noqa: PLC0415
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: PLC0415

    from lifemate.training.lora_finetuning import (  # noqa: PLC0415
        ATTENTION_MLP_SUFFIXES,
        HYBRID_SUFFIXES,
        _resolve_targets,
        discover_target_modules,
    )

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**QWEN3_5_4B_TEXT_CONFIG))
    assert int(sum(p.numel() for p in model.parameters())) == 4_205_751_296

    leaf = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name}
    discovered = discover_target_modules(model)
    assert discovered, "discovery found no target module"
    for suffix in discovered:
        assert suffix in leaf, suffix
    # Hybrid linear-attention projections are present in this architecture and
    # must be adapted, otherwise 28 of 32 layers' input projections stay frozen.
    for required in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                     "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        assert required in discovered, (required, discovered)

    # An explicitly configured list resolves to the same real modules...
    assert _resolve_targets(model, ["q_proj", "down_proj"]) == ["down_proj", "q_proj"]
    # ...and a suffix that does not exist fails loudly instead of training nothing.
    for bogus in ("qkv_proj", "x_proj", "gate_up_proj"):
        assert bogus in ATTENTION_MLP_SUFFIXES + HYBRID_SUFFIXES
        try:
            _resolve_targets(model, [bogus])
        except LifeMateError as exc:
            assert "not found" in str(exc).lower(), str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"configured target {bogus} matches no module but was accepted")
    print(
        f"  LoRA targets -> {len(discovered)} suffixes exist in the released 4.2B "
        f"architecture (incl. linear_attention); a bogus suffix raises"
    )

def test_lora_trainable_count_matches_the_shipped_config() -> None:
    """The trainable count must be derived from the config, never quoted from prose.

    ``docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md`` printed 21,028,756, which is
    exactly ``int(4_205_751_296 * 0.005)``: the memory preflight's 0.5 % LoRA
    placeholder, mistaken for a measurement. The real adapters are
    ``r * (in + out)`` over every wrapped projection, so the expectation here is
    computed from the shipped YAML + a meta-device model, not hardcoded.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP shipped-config LoRA count: torch/transformers not installed")
        return
    try:
        import peft  # noqa: F401,PLC0415 - probe only
    except Exception as exc:
        print(f"  SKIP shipped-config LoRA count: peft not installed ({exc})")
        return
    import torch  # noqa: PLC0415
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: PLC0415

    from lifemate.training.lora_finetuning import configure_lora, discover_target_modules  # noqa: PLC0415
    from lifemate.training.memory_preflight import run_memory_preflight  # noqa: PLC0415

    cfg = load_config("configs/training_lora.yaml")
    assert cfg.lora.target_modules is None, "shipped config must stay on auto-discovery"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**QWEN3_5_4B_TEXT_CONFIG))
    base_params = int(sum(p.numel() for p in model.parameters()))
    discovered = discover_target_modules(model)

    report = configure_lora(model, cfg)
    peft_model = report["model"]
    assert report["target_modules"] == discovered, (report["target_modules"], discovered)
    assert report["r"] == cfg.lora.r and report["alpha"] == cfg.lora.alpha

    # Expectation from the architecture + config: r*(in+out) per wrapped Linear.
    wrapped = [
        (name, module)
        for name, module in peft_model.base_model.model.named_modules()
        if type(module).__module__.startswith("peft.tuners.lora")
    ]
    expected = 0
    seen_suffixes = set()
    for _, module in wrapped:
        base = getattr(module, "base_layer", module)
        expected += cfg.lora.r * (base.in_features + base.out_features)
        seen_suffixes.add(type(module).__name__)
    assert expected == report["trainable_params"], (expected, report["trainable_params"])
    assert expected == peft_model.num_parameters(only_trainable=True)
    assert 0 < expected < base_params / 10, "adapter set is implausibly large"

    # The preflight must report the measured count when handed the real model.
    preflight = run_memory_preflight(model=peft_model, cfg=cfg, gpu_info={"gpu": "unavailable"})
    assert preflight["trainable_source"] == "measured", preflight["trainable_source"]
    assert preflight["trainable_params"] == expected, preflight["trainable_params"]
    print(
        f"  shipped LoRA config -> {len(report['target_modules'])} suffixes, "
        f"{len(wrapped)} modules, trainable {expected:,} "
        f"({100.0 * expected / base_params:.3f}% of {base_params:,})"
    )

def test_full_finetuning_keeps_fp32_master_weights() -> None:
    """AdamW writes into the parameter tensor, so a bf16 master freezes the run.

    Measured (``torch 2.15``, ``N(0, 0.02)`` weights, ``lr=5e-6``, 20 steps,
    grad scale 0.01): with bf16 parameters only **7.4 %** of elements changed at
    all, with fp32 parameters 100 %. A bf16 parameter's ulp near 0.02 is ~1e-4,
    i.e. 20x the per-step update, so the update is rounded away.
    ``configure_full_finetuning`` must therefore keep fp32 master weights and get
    bf16 speed from autocast.
    """
    if not _torch_available():
        print("  SKIP fp32 master test: torch not installed")
        return
    import torch  # noqa: PLC0415

    from lifemate.training.full_finetuning import (  # noqa: PLC0415
        configure_full_finetuning,
        master_dtype,
    )

    assert master_dtype(torch, "bfloat16") is torch.float32
    assert master_dtype(torch, "bf16") is torch.float32
    assert master_dtype(torch, "float32") is torch.float32
    # The compute dtype keeps the configured bf16 on a GPU while the master stays
    # fp32; on CPU `resolve_dtype` downgrades bf16 to fp32 (documented).
    from lifemate.training.full_finetuning import resolve_dtype  # noqa: PLC0415

    assert resolve_dtype(torch, "bfloat16", "cuda") is torch.bfloat16
    assert resolve_dtype(torch, "bfloat16", "cpu") is torch.float32

    model = _tiny_lm()
    model = model.to(dtype=torch.bfloat16)  # arrives in the config's precision
    cfg = load_config()
    cfg.training.precision = "bfloat16"
    cfg.training.gradient_checkpointing = False
    report = configure_full_finetuning(model, None, cfg)
    assert all(p.dtype is torch.float32 for p in model.parameters()), "master weights must be fp32"
    assert report["dtype_target"] == "torch.float32", report
    assert report["dtype_before"] == "torch.bfloat16", report

    # The defect, reproduced at the optimizer level: same steps, same lr, only
    # the parameter dtype differs.
    def moved_fraction(dtype) -> float:
        generator = torch.Generator().manual_seed(0)
        weight = (torch.randn(50_000, generator=generator) * 0.02).to(dtype)
        param = torch.nn.Parameter(weight.clone())
        optimizer = torch.optim.AdamW([param], lr=5e-6, betas=(0.9, 0.95), eps=1e-8)
        start = param.detach().float().clone()
        for step in range(20):
            optimizer.zero_grad()
            grad = torch.randn(50_000, generator=torch.Generator().manual_seed(step)) * 0.01
            param.grad = grad.to(dtype)
            optimizer.step()
        return float((param.detach().float() != start).float().mean())

    bf16_moved = moved_fraction(torch.bfloat16)
    fp32_moved = moved_fraction(torch.float32)
    assert fp32_moved == 1.0, fp32_moved
    assert bf16_moved < 0.2, f"bf16 master unexpectedly moved {bf16_moved:.3f} of elements"
    print(
        f"  fp32 master -> {fp32_moved:.0%} of elements moved vs {bf16_moved:.1%} in bf16 "
        f"(lr 5e-6, 20 steps)"
    )

def test_adamw_decay_split_embeddings_undecayed() -> None:
    """AdamW decay split: embeddings, LayerNorms and biases stay in weight_decay=0.0."""
    if not _torch_available():
        print("  SKIP AdamW decay split test: torch not installed")
        return
    import torch  # noqa: PLC0415
    from lifemate.training.full_finetuning import build_optimizer_and_scheduler  # noqa: PLC0415

    class DummyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 8)
            self.norm = torch.nn.LayerNorm(8)
            self.proj = torch.nn.Linear(8, 8, bias=True)

        def forward(self, x):
            return self.proj(self.norm(self.embed_tokens(x)))

    model = DummyLM()
    cfg = load_config()
    opt, _ = build_optimizer_and_scheduler(model, cfg, 10)
    assert len(opt.param_groups) == 2
    decay_group, no_decay_group = opt.param_groups[0], opt.param_groups[1]
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0

    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert id(model.embed_tokens.weight) in no_decay_ids, "embeddings must not receive weight decay"
    assert id(model.norm.weight) in no_decay_ids, "norm weight must not receive weight decay"
    assert id(model.norm.bias) in no_decay_ids, "norm bias must not receive weight decay"
    assert id(model.proj.bias) in no_decay_ids, "linear bias must not receive weight decay"
    assert id(model.proj.weight) in {id(p) for p in decay_group["params"]}, "linear weight must receive weight decay"
    print("  AdamW split -> embeddings, norms, biases undecayed; linear weights decayed")

def test_gradient_scale_invariance_micro_vs_batch() -> None:
    """8 micro-batches at accum=8 vs 1 batch at accum=1 with same tokens: grads match within 1e-5."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP grad scale invariance test: torch/transformers not installed")
        return
    import torch  # noqa: PLC0415
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415
    from lifemate.training.trainer import _forward_loss  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle1 = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)
        bundle2 = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)

        # Invariance construction (i): all 8 examples have identical prompt & target,
        # meaning no padding is added and token count per micro-batch is identical.
        # Therefore mean-of-means == mean-over-all tokens.
        target = '<action>{"type": "tool_call", "name": "alarm.set", "arguments": {}}</action>'
        messages = [{"role": "system", "content": "Act."}, {"role": "user", "content": "Set."}]
        raw_examples = [build_supervised_example(bundle1.tokenizer, messages, target, 128) for _ in range(8)]
        collator = SupervisedCollator.from_tokenizer(bundle1.tokenizer)

        bundle1.model.zero_grad()
        for ex in raw_examples:
            batch_1 = collator([ex])
            tensors = to_torch_batch(batch_1, "cpu")
            loss = _forward_loss(bundle1.model, tensors)
            (loss / 8).backward()

        bundle2.model.zero_grad()
        batch_8 = collator(raw_examples)
        tensors_8 = to_torch_batch(batch_8, "cpu")
        loss_8 = _forward_loss(bundle2.model, tensors_8)
        loss_8.backward()

        max_rel = 0.0
        named1 = dict(bundle1.model.named_parameters())
        named2 = dict(bundle2.model.named_parameters())
        for n, p1 in named1.items():
            p2 = named2[n]
            diff = (p1.grad - p2.grad).abs()
            norm2 = p2.grad.norm().item()
            rel = (diff.norm().item() / norm2) if norm2 > 0 else diff.max().item()
            if rel > max_rel:
                max_rel = rel
            assert rel < 1e-5, f"{n} rel diff {rel} >= 1e-5"

        # Explicit check for embedding and linear projection
        emb_diff = (named1["model.embed_tokens.weight"].grad - named2["model.embed_tokens.weight"].grad).abs().max().item()
        lin_diff = (named1["model.layers.0.self_attn.q_proj.weight"].grad - named2["model.layers.0.self_attn.q_proj.weight"].grad).abs().max().item()
        assert emb_diff < 1e-5
        assert lin_diff < 1e-5
        print(f"  grad scale invariance -> max relative delta = {max_rel:.2e} (<= 1e-5; emb={emb_diff:.2e}, lin={lin_diff:.2e})")

def test_partial_group_rescale_matches_mean() -> None:
    """Trailing partial group (3 micro-batches at accum=8) rescales to exact mean over the 3 batches."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP partial group rescale test: torch/transformers not installed")
        return
    import torch  # noqa: PLC0415
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415
    from lifemate.training.trainer import _forward_loss, SFTTrainer  # noqa: PLC0415
    from lifemate.training.full_finetuning import build_optimizer_and_scheduler  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle_a = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)
        bundle_b = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)

        target = '<action>{"type": "finish"}</action>'
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        records = [{"messages": messages, "target": target} for _ in range(3)]

        cfg = load_config()
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 1
        cfg.training.gradient_accumulation_steps = 8
        cfg.training.gradient_checkpointing = False
        cfg.training.gradient_clip_norm = 0.0
        cfg.training.output_dir = tmp
        collator = SupervisedCollator.from_tokenizer(bundle_a.tokenizer)
        trainer = SFTTrainer(model=bundle_a.model, tokenizer=bundle_a.tokenizer, collator=collator, train_records=records, val_records=[], cfg=cfg, device="cpu")

        captured_grads_a = {}
        class HookedOptimizer:
            def __init__(self, opt):
                self.opt = opt
                self.param_groups = opt.param_groups
            def step(self):
                for n, p in bundle_a.model.named_parameters():
                    if p.grad is not None:
                        captured_grads_a[n] = p.grad.clone()
                return self.opt.step()
            def zero_grad(self, *a, **kw):
                return self.opt.zero_grad(*a, **kw)
            def state_dict(self):
                return self.opt.state_dict()
            def load_state_dict(self, s):
                return self.opt.load_state_dict(s)

        opt_a, sched_a = build_optimizer_and_scheduler(bundle_a.model, cfg, 1)
        trainer.optimizer = HookedOptimizer(opt_a)
        trainer.scheduler = sched_a
        trainer.run()

        bundle_b.model.zero_grad()
        raw_examples = [build_supervised_example(bundle_b.tokenizer, messages, target, 128) for _ in range(3)]
        total_loss = 0.0
        for ex in raw_examples:
            tensors = to_torch_batch(collator([ex]), "cpu")
            total_loss = total_loss + _forward_loss(bundle_b.model, tensors)
        (total_loss / 3.0).backward()

        max_rel = 0.0
        for n, pb in bundle_b.model.named_parameters():
            pa_grad = captured_grads_a[n]
            diff = (pa_grad - pb.grad).abs()
            norm_b = pb.grad.norm().item()
            rel = (diff.norm().item() / norm_b) if norm_b > 0 else diff.max().item()
            if rel > max_rel:
                max_rel = rel
            assert rel < 1e-5, f"{n} rel diff {rel} >= 1e-5"
        print(f"  partial group rescale -> 3 batches at accum=8 matches mean over 3; max rel diff = {max_rel:.2e}")

def test_scheduler_step_cadence_once_per_optimizer_step() -> None:
    """Assert scheduler.step() is called once per optimizer step and not per micro-batch."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP scheduler cadence test: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415
    from lifemate.training.full_finetuning import build_optimizer_and_scheduler  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)
        target = '<action>{"type": "finish"}</action>'
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        # 6 records, micro_batch=1, accum=2 -> 6 micro-batches, exactly 3 optimizer steps
        records = [{"messages": messages, "target": target} for _ in range(6)]

        cfg = load_config()
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 1
        cfg.training.gradient_accumulation_steps = 2
        cfg.training.output_dir = tmp
        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        trainer = SFTTrainer(model=bundle.model, tokenizer=bundle.tokenizer, collator=collator, train_records=records, val_records=[], cfg=cfg, device="cpu")

        opt, sched = build_optimizer_and_scheduler(bundle.model, cfg, 3)
        step_calls = []
        class StubScheduler:
            def __init__(self, s):
                self.s = s
            def step(self):
                step_calls.append(len(step_calls) + 1)
                return self.s.step()
            def state_dict(self):
                return self.s.state_dict()
            def load_state_dict(self, d):
                return self.s.load_state_dict(d)

        trainer.optimizer = opt
        trainer.scheduler = StubScheduler(sched)
        trainer.run()
        assert len(step_calls) == 3, f"Expected 3 scheduler steps, got {len(step_calls)}"
        print(f"  scheduler cadence -> 6 microbatches, 3 optimizer steps, scheduler called {len(step_calls)} times")

def test_mid_epoch_step_based_evaluation() -> None:
    """eval_strategy='steps' triggers evaluations mid-epoch at step boundaries."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP mid-epoch eval test: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)
        target = '<action>{"type": "finish"}</action>'
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        records = [{"messages": messages, "target": target} for _ in range(6)]
        val_records = [{"messages": messages, "target": target} for _ in range(2)]

        cfg = load_config()
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.eval_strategy = "steps"
        cfg.training.logging_steps = 2
        cfg.training.output_dir = tmp
        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        trainer = SFTTrainer(model=bundle.model, tokenizer=bundle.tokenizer, collator=collator, train_records=records, val_records=val_records, cfg=cfg, device="cpu")
        res = trainer.run()

        eval_steps = [m["step"] for m in res.metrics_history if "val_loss" in m]
        assert eval_steps == [2, 4, 6], f"Expected evals at steps [2, 4, 6], got {eval_steps}"
        print(f"  mid-epoch eval -> evaluated at steps {eval_steps}")
def test_full_finetuning_backward_all_parameters_receive_gradients() -> None:
    """Every parameter receives non-None gradient after backward."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP all parameters gradient test: torch/transformers not installed")
        return
    import torch  # noqa: PLC0415
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415
    from lifemate.training.full_finetuning import configure_full_finetuning  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True)
        configure_full_finetuning(bundle.model, bundle.tokenizer, None)

        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        out = bundle.model(input_ids, labels=labels)
        out.loss.backward()

        named = dict(bundle.model.named_parameters())
        assert named["model.embed_tokens.weight"].grad is not None
        assert named["model.norm.weight"].grad is not None
        assert named["lm_head.weight"].grad is not None
        for n, p in named.items():
            assert p.grad is not None, f"grad is None for {n}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {n}"
        print("  full FT grad flow -> all parameters incl. embeddings, norms, lm_head receive non-None grads")

def test_cli_tuning_precedence_preserves_config() -> None:
    """Omitting --tuning preserves config tuning_mode; explicit --tuning overrides it."""
    out_default = _run([sys.executable, "scripts/train.py", "--config", "configs/training_lora.yaml", "--dry-run"])
    assert out_default.returncode == 0
    assert "tuning mode : lora" in out_default.stdout
    assert "tuning mode: lora (config) -> full" not in out_default.stdout

    out_override = _run([sys.executable, "scripts/train.py", "--config", "configs/training_lora.yaml", "--tuning", "full", "--dry-run"])
    assert out_override.returncode == 0
    assert "tuning mode: lora (config) -> full (--tuning)" in out_override.stdout
    assert "tuning mode : full" in out_override.stdout
    print("  CLI precedence -> omitting --tuning preserves config; explicit flag overrides")

# --------------------------------------------------------------------------- #
# resume: optimizer moments, RNG stream, checkpoint layout
# --------------------------------------------------------------------------- #
def _resume_cfg(max_seq_length: int = 128):
    """Deterministic CPU config for resume tests (no checkpointing, no val)."""
    cfg = load_config(None)
    cfg.training.precision = "float32"
    cfg.training.num_train_epochs = 1
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.gradient_checkpointing = False
    cfg.training.save_strategy = "steps"
    cfg.training.logging_steps = 1
    cfg.training.max_seq_length = max_seq_length
    cfg.model.device = "cpu"
    return cfg

def _to_lora_with_r(cfg, r: int):
    """Switch a config to LoRA mode and set ``lora.r`` (the knob under test)."""
    cfg.training.tuning_mode = "lora"
    cfg.lora.r = int(r)
    return cfg

def _resume_trainer(model_dir: Path, tmp: Path, run: str, **kwargs) -> SFTTrainer:
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415

    bundle = load_model_and_tokenizer(
        str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True
    )
    return SFTTrainer(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        collator=SupervisedCollator.from_tokenizer(bundle.tokenizer),
        train_records=kwargs.pop("train_records", []),
        cfg=_resume_cfg(),
        device="cpu",
        checkpoint_manager=CheckpointManager(tmp / run, run_name=run, save_total_limit=0),
        **kwargs,
    )

def test_checkpoint_training_state_has_rng_and_optimizer_files() -> None:
    """A saved step-<n> carries RNG in training_state.json plus optimizer_state.pt."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP checkpoint layout test: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = make_tiny_model_and_tokenizer(root / "model")
        trainer = _resume_trainer(
            model_dir, root, "layout", max_steps=1, train_records=_synthetic_records(2)
        )
        result = trainer.run()
        checkpoint = Path(result.checkpoint_dir)

        state = read_training_state(checkpoint)
        assert isinstance(state.get("step"), int) and state["step"] == 1, state
        assert "torch_rng_state" in state, sorted(state)
        assert "python_rng_state" in state, sorted(state)
        assert state["torch_rng_state"] is not None, "torch present, so the RNG state must be captured"
        assert isinstance(state["python_rng_state"], list) and state["python_rng_state"], state["python_rng_state"]
        assert state.get("config_hash"), "config_hash must be recorded for resume provenance"

        optimizer_file = checkpoint / "optimizer_state.pt"
        assert optimizer_file.is_file(), f"missing {optimizer_file.name} in {checkpoint}"
        assert optimizer_file.stat().st_size > 0
        print(
            "  checkpoint layout -> training_state.json(step, torch_rng_state, python_rng_state, "
            "config_hash) + optimizer_state.pt"
        )

def test_resume_restores_optimizer_and_rng_state() -> None:
    """Resume restores AdamW moments, scheduler position and the torch/python RNG stream."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP resume restore test: torch/transformers not installed")
        return
    import random  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = make_tiny_model_and_tokenizer(root / "model")
        records = _synthetic_records(2)

        # Interrupted run: exactly one optimizer step, then a step-1 checkpoint.
        first = _resume_trainer(model_dir, root, "first", max_steps=1, train_records=records)
        first.run()
        checkpoint = Path(first.checkpoint_manager.step_dir(1))
        state = read_training_state(checkpoint)

        saved_optimizer = first.optimizer.state_dict()["state"]
        assert len(first.optimizer.state) > 0, "AdamW must have per-parameter state after one step"
        param_id = next(iter(saved_optimizer))
        saved_exp_avg = saved_optimizer[param_id]["exp_avg"].clone()
        assert torch.is_tensor(saved_exp_avg) and saved_exp_avg.abs().sum() > 0, "exp_avg must be non-trivial"
        saved_scheduler_epoch = first.scheduler.state_dict().get("last_epoch")
        saved_torch_rng = torch.tensor(state["torch_rng_state"], dtype=torch.uint8)
        saved_python_rng = list(state["python_rng_state"])

        # Advance both streams so a no-op restore cannot pass by coincidence.
        torch.randn(16)
        random.random()

        resumed = _resume_trainer(model_dir, root, "second", max_steps=3, train_records=records)
        assert resumed.resume(checkpoint) == 1
        assert resumed.start_step == 1

        # The fresh trainer had no optimizer before resume(); the moments must be
        # rebuilt AND repopulated, not silently dropped.
        assert resumed.optimizer is not None, "resume() must build the optimizer before restoring"
        restored = resumed.optimizer.state_dict()["state"]
        assert len(restored) == len(saved_optimizer), (len(restored), len(saved_optimizer))
        assert torch.allclose(restored[param_id]["exp_avg"], saved_exp_avg), "exp_avg not restored"
        restored_exp_avg = restored[param_id]["exp_avg"].clone()  # later steps mutate in place
        assert resumed.scheduler.state_dict().get("last_epoch") == saved_scheduler_epoch, (
            resumed.scheduler.state_dict().get("last_epoch"),
            saved_scheduler_epoch,
        )

        # resume() restores the streams; run() must not re-seed them away.
        assert torch.equal(torch.get_rng_state(), saved_torch_rng), "torch RNG stream not restored"
        assert list(random.getstate()[1]) == saved_python_rng, "python RNG stream not restored"

        # run() must not re-seed a resumed trainer: that would discard the restored
        # stream. Spy on the module-level set_seed the loop calls.
        import lifemate.training.trainer as trainer_module  # noqa: PLC0415

        seed_calls: list[int] = []
        real_set_seed = trainer_module.set_seed

        def _spy(seed: int, *args, **kwargs):
            seed_calls.append(int(seed))
            return real_set_seed(seed, *args, **kwargs)

        trainer_module.set_seed = _spy
        try:
            resumed.run()
        finally:
            trainer_module.set_seed = real_set_seed
        assert seed_calls == [], f"run() re-seeded a resumed trainer: {seed_calls}"

        # Data order is deliberately reseeded per start_step (random.Random(seed +
        # start_step)), so a resumed epoch visits records in a different order than
        # the uninterrupted run and byte-identical losses are unattainable. The
        # strongest deterministic property that IS guaranteed: the second optimizer
        # step must keep updating the restored moments rather than starting over.
        after_run = resumed.optimizer.state_dict()["state"][param_id]["exp_avg"]
        assert not torch.equal(after_run, restored_exp_avg), "resumed step did not update exp_avg"
        print(
            "  resume -> step 1; optimizer moments, scheduler position and both RNG "
            "streams restored and replayable; resumed step updates exp_avg"
        )

def test_resume_rejects_config_hash_mismatch() -> None:
    """A tampered/edited config_hash blocks the resume instead of silently continuing."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP resume config-hash test: torch/transformers not installed")
        return
    import hashlib  # noqa: PLC0415

    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415
    from lifemate.training.trainer import STATE_FILENAME, config_fingerprint  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = make_tiny_model_and_tokenizer(root / "model")
        records = _synthetic_records(2)

        first = _resume_trainer(model_dir, root, "first", max_steps=1, train_records=records)
        first.run()
        checkpoint = Path(first.checkpoint_manager.step_dir(1))
        state = read_training_state(checkpoint)
        original_hash = state["config_hash"]
        assert original_hash == config_fingerprint(first.cfg), "written hash must match the live fingerprint"

        # A genuine knob change (lr) must produce a different fingerprint.
        changed = _resume_trainer(model_dir, root, "changed", max_steps=2, train_records=records)
        changed.cfg.training.learning_rate = float(changed.cfg.training.learning_rate) * 2
        assert config_fingerprint(changed.cfg) != original_hash, "lr change must alter the fingerprint"
        try:
            changed.resume(checkpoint)
        except LifeMateError as exc:
            assert "config" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("resume with a changed learning_rate must fail")

        # Every math-relevant knob must be inside the whitelist: if one of these
        # slips through, the resume would silently continue under new maths.
        # ``lora.r`` only matters in LoRA mode, so that entry switches mode first.
        math_knobs = (
            ("learning_rate", lambda cfg: setattr(cfg.training, "learning_rate", 1e-4)),
            ("seed", lambda cfg: setattr(cfg.training, "seed", 7)),
            ("max_seq_length", lambda cfg: setattr(cfg.training, "max_seq_length", 64)),
            (
                "gradient_accumulation_steps",
                lambda cfg: setattr(cfg.training, "gradient_accumulation_steps", 2),
            ),
            ("tuning_mode", lambda cfg: setattr(cfg.training, "tuning_mode", "lora")),
            ("model.name", lambda cfg: setattr(cfg.model, "name", "other/model")),
            ("model.torch_dtype", lambda cfg: setattr(cfg.model, "torch_dtype", "float32")),
            ("lora.r", lambda cfg: _to_lora_with_r(cfg, 32)),
        )
        for index, (label, setter) in enumerate(math_knobs):
            edited = _resume_trainer(model_dir, root, f"edit{index}", max_steps=2, train_records=records)
            setter(edited.cfg)
            assert config_fingerprint(edited.cfg) != original_hash, f"{label} change must alter the fingerprint"
            try:
                edited.resume(checkpoint)
            except LifeMateError as exc:
                assert "config" in str(exc).lower(), f"{label}: {exc}"
            else:
                raise AssertionError(f"resume with a changed {label} must fail")

        # Isolation for lora.r: in LoRA mode, r alone moves the digest; in full
        # mode the same knob is inert (asserted in the tolerance test).
        lora_base = _resume_cfg()
        lora_base.training.tuning_mode = "lora"
        lora_hash = config_fingerprint(lora_base)
        lora_base.lora.r = int(lora_base.lora.r) + 16
        assert config_fingerprint(lora_base) != lora_hash, "lora.r must move the digest in LoRA mode"

        # Tampering the recorded hash directly must also be rejected.
        tampered = dict(state)
        tampered["config_hash"] = hashlib.sha256(b"tampered").hexdigest()[:16]
        (checkpoint / STATE_FILENAME).write_text(json.dumps(tampered), encoding="utf-8")
        other = _resume_trainer(model_dir, root, "other", max_steps=2, train_records=records)
        try:
            other.resume(checkpoint)
        except LifeMateError as exc:
            assert "config" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("resume with a tampered config_hash must fail")

        # Path/bookkeeping keys are excluded, so a different --output-dir still resumes.
        (checkpoint / STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")
        moved = _resume_trainer(model_dir, root, "moved", max_steps=2, train_records=records)
        moved.cfg.training.output_dir = str(root / "elsewhere")
        moved.cfg.training.resume_from_checkpoint = str(checkpoint)
        assert moved.resume(checkpoint) == 1, "output_dir / resume_from_checkpoint must not block a resume"
        print(
            "  resume config hash -> lr/seed/seq-len/accum/tuning_mode/model.dtype/lora.r "
            "changes and a tampered hash rejected; output_dir/resume_from_checkpoint ignored"
        )

def test_lora_resume_restores_adapter_and_enables_checkpointing() -> None:
    """LoRA resume must restore adapter tensors; checkpointing must be on, cache off.

    Two defects covered: (a) ``_load_weights`` called ``load_adapter(str(path))``
    without the now-required ``adapter_name``, so every LoRA resume raised
    TypeError; (b) ``configure_lora`` ignored ``training.gradient_checkpointing``,
    so the LoRA path silently ran without the ~5x activation reduction the config
    asked for.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP LoRA resume test: torch/transformers not installed")
        return
    try:
        import peft  # noqa: F401,PLC0415 - probe only
    except Exception:
        print("  SKIP LoRA resume test: peft not installed")
        return
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig  # noqa: PLC0415
    from lifemate.training.lora_finetuning import configure_lora  # noqa: PLC0415
    from lifemate.training.collator import SupervisedCollator  # noqa: PLC0415

    text = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    records = [
        {"messages": [{"role": "user", "content": f"dat bao thuc {i}"}],
         "target": '<action>{"type": "finish"}</action>'}
        for i in range(3)
    ]

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [2 + (ord(ch) % 60) for ch in text]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

        def save_pretrained(self, path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _build(root: Path, max_steps: int):
        cfg = _resume_cfg(max_seq_length=64)
        cfg.training.tuning_mode = "lora"
        cfg.training.num_train_epochs = 1
        cfg.training.output_dir = str(root)
        cfg.training.gradient_checkpointing = True
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(text)
        report = configure_lora(model, cfg)
        assert report["gradient_checkpointing"] is True, "configure_lora ignored gradient_checkpointing"
        assert report["model"].config.use_cache is False
        return SFTTrainer(
            model=report["model"], tokenizer=_Tok(),
            collator=SupervisedCollator.from_tokenizer(_Tok()),
            train_records=records, cfg=cfg, device="cpu", run_name="lora",
            max_steps=max_steps, is_lora=True,
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = _build(root, 2)
        first.run()
        checkpoint = Path(first.checkpoint_manager.step_dir(2))
        assert (checkpoint / "adapter_model.safetensors").is_file()
        saved = {n: p.detach().clone() for n, p in first.model.named_parameters() if "lora_" in n}

        resumed = _build(root, 4)
        assert resumed.resume(checkpoint) == 2
        restored = {n: p.detach().clone() for n, p in resumed.model.named_parameters() if "lora_" in n}
        mismatched = [n for n in saved if not torch.allclose(saved[n], restored[n])]
        assert not mismatched, f"{len(mismatched)} adapter tensor(s) not restored, e.g. {mismatched[0]}"
        print("  LoRA resume -> adapter tensors restored exactly; grad checkpointing on, use_cache off")

def test_hybrid_checkpoint_reloads_through_weight_conversions() -> None:
    """Resume must reload a hybrid checkpoint whose saved keys are prefixed.

    ``Qwen3_5`` (target 4B checkpoint) carries a ``_weight_conversions`` mapping
    and ``save_pretrained`` writes ``model.language_model.*``; a raw
    ``load_state_dict`` then reports every non-``lm_head`` key as missing
    (measured: 55 missing + 55 unexpected for 56 tensors), so resume refused a
    checkpoint this trainer had just written. Tied embeddings are also tolerated:
    a ``tie_word_embeddings`` checkpoint omits ``lm_head``.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP hybrid reload test: torch/transformers not installed")
        return
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig  # noqa: PLC0415
    from lifemate.training.trainer import _load_state_dict_into  # noqa: PLC0415

    text = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, tie_word_embeddings=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "mm"
        Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text)).save_pretrained(source)

        model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.float32)
        assert getattr(model, "_weight_conversions", None), "checkpoint defines no key conversions"

        saved = Path(tmp) / "saved"
        model.save_pretrained(saved)
        from lifemate.training.trainer import _read_state_dict  # noqa: PLC0415

        state = _read_state_dict(sorted(saved.glob("*.safetensors")))
        fresh = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.float32)
        missing, unexpected = _load_state_dict_into(fresh, state)
        assert missing == [] and unexpected == [], (missing[:3], unexpected[:3])
        names = dict(fresh.named_parameters())
        for key, tensor in state.items():
            local = key
            if "language_model" in local:
                local = local.replace("model.language_model.", "model.", 1)
            if local in names:
                assert torch.equal(names[local].detach(), tensor), f"{key} not restored"
        print("  hybrid reload -> conversions applied, tied lm_head tolerated, weights identical")

def test_resumed_run_keeps_a_real_learning_rate() -> None:
    """A resumed run must not silently train at LR 0.0.

    The checkpoint's scheduler was built for the interrupted run's step count, so
    restoring it directly makes every remaining step take the LR from a schedule
    that already reached its horizon (0.0). The resumed LR trajectory must match
    the uninterrupted run's.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP resumed-LR test: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415

    records = _synthetic_records(4)

    def _trainer(model_dir: Path, root: Path, run: str, max_steps: int | None) -> SFTTrainer:
        cfg = _resume_cfg()
        cfg.training.num_train_epochs = 2
        cfg.training.warmup_ratio = 0.25
        cfg.training.learning_rate = 1e-4  # big enough to see in a trace
        cfg.training.eval_strategy = "no"
        cfg.training.output_dir = str(root / "ckpts")
        bundle = __import__("lifemate.models.loader", fromlist=["load_model_and_tokenizer"]).load_model_and_tokenizer(
            str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True
        )
        return SFTTrainer(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            collator=SupervisedCollator.from_tokenizer(bundle.tokenizer),
            train_records=records,
            cfg=cfg,
            device="cpu",
            run_name=run,
            max_steps=max_steps,
        )

    def _trace(trainer: SFTTrainer) -> list[tuple[int, float]]:
        trace: list[tuple[int, float]] = []

        def _log(step: int, loss: float | None) -> None:
            trace.append((step, round(float(trainer.optimizer.param_groups[0]["lr"]), 12)))

        trainer._maybe_log = _log
        return trace

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = make_tiny_model_and_tokenizer(root / "model")

        reference = _trainer(model_dir, root, "ref", None)
        reference_trace = _trace(reference)
        reference.run()

        interrupted = _trainer(model_dir, root, "int", 2)
        _trace(interrupted)
        interrupted.run()
        checkpoint = Path(interrupted.checkpoint_manager.step_dir(2))

        resumed = _trainer(model_dir, root, "res", None)
        assert resumed.resume(checkpoint) == 2
        resumed_trace = _trace(resumed)
        resumed.run()

        assert resumed_trace, "the resumed run logged no step"
        assert any(lr > 0 for _, lr in resumed_trace), f"resumed run trained at LR 0: {resumed_trace}"
        assert resumed_trace == [item for item in reference_trace if item[0] > 2], (
            f"resumed LR trajectory {resumed_trace} != uninterrupted tail "
            f"{[i for i in reference_trace if i[0] > 2]}"
        )
        print(f"  resumed LR -> {resumed_trace} matches the uninterrupted trajectory")

def test_resume_tolerates_irrelevant_config_changes() -> None:
    """Dataset path, device, W&B names and inference/ASR paths are outside the hash."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP resume irrelevant-change test: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415
    from lifemate.training.trainer import config_fingerprint  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = make_tiny_model_and_tokenizer(root / "model")
        records = _synthetic_records(2)

        first = _resume_trainer(model_dir, root, "first", max_steps=1, train_records=records)
        assert first.cfg.data.root == "./data"
        first.run()
        checkpoint = Path(first.checkpoint_manager.step_dir(1))
        original_hash = read_training_state(checkpoint)["config_hash"]
        assert original_hash == config_fingerprint(first.cfg)

        # Bookkeeping knobs are not optimisation: each one alone must resume, and
        # the whole set plus a LoRA knob in full mode must leave the digest alone.
        bookkeeping = {
            "logging_steps": 99,
            "save_strategy": "no",
            "eval_strategy": "no",
            "save_total_limit": 9,
            "report_to": ["wandb"],
            "dataloader_num_workers": 4,
            "method": "sft",
        }
        for index, (key, value) in enumerate(bookkeeping.items()):
            only = _resume_trainer(model_dir, root, f"book{index}", max_steps=2, train_records=records)
            setattr(only.cfg.training, key, value)
            assert config_fingerprint(only.cfg) == original_hash, (
                f"training.{key} is bookkeeping and must not participate in the fingerprint"
            )
            assert only.resume(checkpoint) == 1, f"a changed training.{key} must not block a resume"

        resumed = _resume_trainer(model_dir, root, "resumed", max_steps=2, train_records=records)
        resumed.cfg.data.root = "/mnt/data"
        resumed.cfg.model.device = "cuda:1"
        resumed.cfg.wandb.run_name = "renamed-run"
        resumed.cfg.inference.output = str(root / "other" / "submission.jsonl")
        resumed.cfg.asr.model_dir = "/mnt/asr/gipformer"
        for key, value in bookkeeping.items():
            setattr(resumed.cfg.training, key, value)
        # A LoRA knob cannot change full fine-tuning maths, so full mode ignores it.
        resumed.cfg.lora.r = 64

        assert config_fingerprint(resumed.cfg) == original_hash, (
            "data.root/model.device/wandb.run_name/inference.output/asr.model_dir plus the "
            "bookkeeping keys and lora.r in full mode must not participate in the fingerprint"
        )
        try:
            step = resumed.resume(checkpoint)
        except LifeMateError as exc:
            raise AssertionError(f"irrelevant config changes blocked the resume: {exc}") from exc
        assert step == 1, f"expected step 1, got {step}"

        # ... but in LoRA mode the same LoRA knob IS optimisation-relevant.
        lora_cfg = _resume_cfg()
        lora_cfg.training.tuning_mode = "lora"
        before = config_fingerprint(lora_cfg)
        lora_cfg.lora.r = 64
        assert config_fingerprint(lora_cfg) != before, "lora.r must alter the fingerprint in LoRA mode"
        print(
            "  resume tolerance -> data.root, model.device, wandb.run_name, inference.output, "
            "asr.model_dir, logging/save/eval/save_total_limit/report_to/workers/method and "
            "lora.r in FULL mode changed; fingerprint unchanged, resume accepted; lora.r in "
            "LORA mode changes the fingerprint"
        )

def test_config_fingerprint_is_stable_across_dict_order() -> None:
    """Equivalent configs built in different key orders hash identically (sort_keys)."""
    from lifemate.training.trainer import config_fingerprint  # noqa: PLC0415

    left = {
        "training": {"max_seq_length": 4096, "learning_rate": 5e-6, "seed": 42},
        "model": {"torch_dtype": "bfloat16", "name": "Qwen/Qwen3.5-4B"},
        "lora": {"alpha": 32, "r": 16},
    }
    right = {
        "lora": {"r": 16, "alpha": 32},
        "model": {"name": "Qwen/Qwen3.5-4B", "torch_dtype": "bfloat16"},
        "training": {"seed": 42, "max_seq_length": 4096, "learning_rate": 5e-6},
    }
    assert config_fingerprint(left) == config_fingerprint(right), "key order must not matter"
    assert config_fingerprint({"training": {"seed": 1}}) == config_fingerprint(
        {"training": {"seed": 1}, "lora": {}}
    ), "missing/empty lora section must not crash or vary the digest"
    print("  config fingerprint -> dict order irrelevant, missing sections tolerated")
def main() -> int:
    """Run every test in this module (stdlib runner parity)."""
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - test harness
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0

if __name__ == "__main__":
    if not _torch_available():
        import types

        torch = types.SimpleNamespace()  # noqa: F841 - keep the guard below honest
    sys.exit(main())

def test_resume_rejects_a_seq_len_change_across_the_4096_to_8192_migration() -> None:
    """A checkpoint trained at 4096 must NOT resume into the 8192 setting.

    ``max_seq_length`` is inside ``config_fingerprint`` (it changes which tokens
    are seen: 75 % of records were truncated at 4096), so a half-finished
    pre-change run has to be rejected rather than silently continued on
    different data. Executed: fingerprint 97ce16b36757e736 @4096 vs
    8f57e80e923e1cc3 @8192 -> ``cannot resume: training config differs``.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP 4096->8192 resume guard: torch/transformers not installed")
        return
    from tests.test_tiny_e2e import make_tiny_model_and_tokenizer, _synthetic_records  # noqa: PLC0415
    from lifemate.models.loader import load_model_and_tokenizer  # noqa: PLC0415
    from lifemate.training.trainer import config_fingerprint  # noqa: PLC0415

    def _trainer(out: Path, msl: int, max_steps: int) -> SFTTrainer:
        cfg = _resume_cfg(max_seq_length=msl)
        cfg.training.output_dir = str(out)
        bundle = load_model_and_tokenizer(
            str(out / "model"), device="cpu", torch_dtype="float32",
            tuning_mode="full", trainable=True, for_training=True,
        )
        return SFTTrainer(
            model=bundle.model, tokenizer=bundle.tokenizer,
            collator=SupervisedCollator.from_tokenizer(bundle.tokenizer),
            train_records=_synthetic_records(4), cfg=cfg, device="cpu",
            run_name="full", max_steps=max_steps,
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = make_tiny_model_and_tokenizer(root / "model")
        assert model_dir  # loader above points at root/"model"
        old = _trainer(root, 4096, 2)
        old.run()
        checkpoint = old.checkpoint_manager.step_dir(2)

        new_cfg = _resume_cfg(max_seq_length=8192)
        assert config_fingerprint(old.cfg) != config_fingerprint(new_cfg), (
            "max_seq_length must participate in config_fingerprint"
        )
        new = _trainer(root, 8192, 4)
        try:
            new.resume(checkpoint)
        except LifeMateError as exc:
            message = str(exc)
            assert "training config differs" in message, message
            assert "max_seq_length" in message, message
        else:  # pragma: no cover
            raise AssertionError("resume across a max_seq_length change must be rejected")

        same = _trainer(root, 4096, 4)
        assert same.resume(checkpoint) == 2, "an unchanged setting must still resume"
        print("  resume guard -> 4096 checkpoint rejected under 8192 (names max_seq_length); same setting resumes")
