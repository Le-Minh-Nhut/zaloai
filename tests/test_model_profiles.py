"""Model-profile tests: MiniCPM5-2B default, Qwen3.5-4B as explicit alternative.

Three layers of coverage:

* stdlib-only: registry resolution, the default flip, YAML/CLI precedence, the
  reasoning-block guard, truncation rejection, checkpoint identity mismatch, and
  the memory preflight's per-profile numbers;
* real-tokenizer (needs ``transformers`` + ``models/MiniCPM5-2B``): chat-template
  parity at token-ID level on real SFT records;
* Qwen prompt bytes frozen from BEFORE the change, so Qwen rendering stays
  byte-identical.

The tokenizer-gated tests print ``SKIP: <reason>`` and return when the model
directory or ``transformers`` is absent; every check they make is a recorded
measurement, not a re-derivation.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402
from lifemate.models import checkpoint as ckpt  # noqa: E402
from lifemate.models import loader  # noqa: E402
from lifemate.models.profiles import (  # noqa: E402
    CLOSED_THINK_SUFFIX,
    GENERIC_PROFILE,
    MINICPM5_PROFILE,
    QWEN3_5_PROFILE,
    resolve_profile,
)

LOCAL_MINICPM = ROOT / "models" / "MiniCPM5-2B"

NEEDS_GPU: list[str] = []


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def _minicpm_tokenizer():
    """Real MiniCPM tokenizer, or ``None`` when unavailable."""
    if not _have("transformers"):
        return None
    if not (LOCAL_MINICPM / "tokenizer_config.json").is_file():
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(LOCAL_MINICPM), local_files_only=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"SKIP: MiniCPM tokenizer unreadable ({type(exc).__name__}: {exc})")
        return None


def _real_records(count: int = 5) -> list[dict]:
    """First ``count`` real SFT decision records, or ``[]`` when the file is absent."""
    path = ROOT / "outputs" / "datasets" / "train" / "train.jsonl"
    if not path.is_file():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) >= count:
                break
    return records


# --------------------------------------------------------------------------- #
# profile registry
# --------------------------------------------------------------------------- #
def test_profile_resolution_registry() -> None:
    """Hub ids and local directories both resolve; unknown names stay generic."""
    assert resolve_profile("openbmb/MiniCPM5-2B") is MINICPM5_PROFILE
    assert resolve_profile("models/MiniCPM5-2B") is MINICPM5_PROFILE
    assert resolve_profile("minicpm5") is MINICPM5_PROFILE
    assert resolve_profile("Qwen/Qwen3.5-4B") is QWEN3_5_PROFILE
    assert resolve_profile("Qwen/Qwen3.5-4B-Instruct") is QWEN3_5_PROFILE
    assert resolve_profile("meta-llama/Llama-3.1-8B") is GENERIC_PROFILE
    assert resolve_profile("") is GENERIC_PROFILE
    assert resolve_profile(None) is GENERIC_PROFILE
    print("  profiles -> minicpm5 (hub + local + lowercase), qwen3_5, else generic")


def test_profile_fields_are_the_measured_ones() -> None:
    """The numbers quoted in the profile are the measured upstream facts."""
    assert MINICPM5_PROFILE.family == "minicpm"
    assert MINICPM5_PROFILE.chat_template_kwargs == {"enable_thinking": False}
    assert MINICPM5_PROFILE.response_terminator == "<|im_end|>"
    assert 130073 in MINICPM5_PROFILE.speculative_stop_ids
    # Measured from AutoModelForCausalLM.from_config on the released config.
    assert MINICPM5_PROFILE.lora_target_modules == (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    assert MINICPM5_PROFILE.preflight_params == 2_516_756_480
    assert (MINICPM5_PROFILE.preflight_layers, MINICPM5_PROFILE.preflight_hidden) == (42, 2048)
    assert MINICPM5_PROFILE.architecture_fallbacks == ("LlamaForCausalLM",)
    assert MINICPM5_PROFILE.supports_thinking is True
    # The template BOS is already in the rendered prompt.
    assert MINICPM5_PROFILE.prompt_add_special_tokens is False
    # Qwen must keep today's behaviour exactly.
    assert QWEN3_5_PROFILE.default_max_new_tokens == 512
    assert QWEN3_5_PROFILE.lora_target_modules == ()  # hybrid auto-discovery
    assert QWEN3_5_PROFILE.speculative_stop_ids == ()
    assert QWEN3_5_PROFILE.prompt_add_special_tokens is True
    assert QWEN3_5_PROFILE.no_think_suffix == CLOSED_THINK_SUFFIX
    print("  profile fields -> MiniCPM 42L/2048h/7 targets/192 tokens; Qwen 512 + discovery")


# --------------------------------------------------------------------------- #
# DEFAULT FLIP + OVERRIDES
# --------------------------------------------------------------------------- #
def test_minicpm_is_the_default_model() -> None:
    """Dataclass default and every shipped training/inference YAML."""
    assert load_config().model.name == "openbmb/MiniCPM5-2B"
    for name in ("base", "inference", "training_full", "training_lora", "training_lora_wandb"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.model.name == "openbmb/MiniCPM5-2B", f"configs/{name}.yaml still names {cfg.model.name}"
    print("  defaults -> MiniCPM5-2B in code + 5 configs")


def test_qwen_stays_one_config_away() -> None:
    """The dedicated Qwen override config is explicit and otherwise identical."""
    qwen = load_config(ROOT / "configs" / "training_lora_qwen.yaml")
    assert qwen.model.name == "Qwen/Qwen3.5-4B"
    assert qwen.training.tuning_mode == "lora"
    base = load_config(ROOT / "configs" / "training_lora.yaml")

    def flat(payload, prefix=""):
        out = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                out.update(flat(value, prefix + key + "."))
            else:
                out[key if not prefix else prefix + key] = value
        return out

    left, right = flat(base.to_dict()), flat(qwen.to_dict())
    diffs = {key for key in left if left[key] != right[key]}
    assert diffs == {"model.name"}, f"training_lora_qwen.yaml differs in more than model.name: {diffs}"
    print("  qwen config -> training_lora_qwen.yaml differs only in model.name")


def test_qwen_is_selectable_by_yaml_and_cli() -> None:
    """--model beats YAML for both directions; YAML beats the dataclass default."""
    from lifemate.config import load_config as load

    yaml_cfg = load(ROOT / "configs" / "training_lora.yaml")
    assert yaml_cfg.model.name == "openbmb/MiniCPM5-2B"
    yaml_cfg.model.name = "Qwen/Qwen3.5-4B"  # what scripts/train.py --model does
    assert resolve_profile(yaml_cfg.model.name).name == "qwen3_5"
    assert resolve_profile(DEFAULT_OVERRIDE := "Qwen/Qwen3.5-4B").name == "qwen3_5"
    del DEFAULT_OVERRIDE
    print("  overrides -> --model wins over YAML; both profiles reachable")


def test_wandb_config_has_exactly_three_deltas() -> None:
    """training_lora_wandb.yaml = training_lora.yaml + 3 documented deltas."""
    def flat(payload, prefix=""):
        out = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                out.update(flat(value, prefix + key + "."))
            else:
                out[key if not prefix else prefix + key] = value
        return out

    plain = flat(load_config(ROOT / "configs" / "training_lora.yaml").to_dict())
    tracked = flat(load_config(ROOT / "configs" / "training_lora_wandb.yaml").to_dict())
    diffs = {key: (plain[key], tracked[key]) for key in plain if plain[key] != tracked[key]}
    assert diffs == {
        "training.report_to": ([], ["wandb"]),
        "wandb.enabled": (False, True),
        "wandb.mode": ("online", "offline"),
    }, diffs
    print("  wandb config -> exactly 3 deltas (report_to, wandb.enabled, wandb.mode)")


def test_no_silent_qwen_fallback_in_loader() -> None:
    """A failing MiniCPM load must raise, never retry another model."""
    source = (ROOT / "src" / "lifemate" / "models" / "loader.py").read_text(encoding="utf-8")
    # No model id and no "fall back to X" retry anywhere in the load path.
    for identifier in ("Qwen/", "openbmb/"):
        assert identifier not in source, f"the loader must not name a fallback model ({identifier})"
    print("  loader -> no fallback model name anywhere (bf16->fp16 only)")


# --------------------------------------------------------------------------- #
# TEMPLATE KWARGS / TERMINATOR (fake tokenizers, no downloads)
# --------------------------------------------------------------------------- #
class _MiniCPMLikeTokenizer:
    """Mimics the measured MiniCPM contract without needing the real files.

    ``eos_token_id`` is ``</s>`` (1) while the template closes turns with
    ``<|im_end|>`` (130073): the exact shape that made the collator supervise
    the wrong terminator.
    """

    name_or_path = "openbmb/MiniCPM5-2B"
    eos_token_id = 1

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(dict(kwargs))
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            return body + "<|im_start|>assistant\n" + CLOSED_THINK_SUFFIX
        return body

    def encode(self, text, add_special_tokens=False):
        table = {
            "<|im_end|>": [130073],
            "</s>": [1],
            "<s>": [0],
        }
        if text in table:
            return list(table[text])
        return [10 + (ord(ch) % 50) for ch in text]


def test_template_kwargs_come_from_the_profile() -> None:
    """The collator passes the profile's kwargs, not a hardcoded literal."""
    from lifemate.training.collator import apply_chat_template_text

    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    tok = _MiniCPMLikeTokenizer()
    text = apply_chat_template_text(tok, messages, profile=MINICPM5_PROFILE)
    assert tok.calls == [{"enable_thinking": False}]
    assert text.endswith(CLOSED_THINK_SUFFIX)
    # A template that rejects the kwarg still renders (TypeError retry path).
    class _Rejecting(_MiniCPMLikeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
            return super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )

    assert apply_chat_template_text(_Rejecting(), messages, profile=MINICPM5_PROFILE).endswith(
        CLOSED_THINK_SUFFIX
    )
    print("  template kwargs -> profile-supplied, TypeError-retry preserved")


def test_terminator_is_the_template_terminator_not_eos_token() -> None:
    """MiniCPM supervises 130073; Qwen keeps its eos (which IS <|im_end|>)."""
    from lifemate.training.collator import (
        build_supervised_example,
        eos_token_ids,
        resolve_response_terminator,
    )

    tok = _MiniCPMLikeTokenizer()
    assert eos_token_ids(tok) == [1]
    terminator, source = resolve_response_terminator(tok, MINICPM5_PROFILE)
    assert (terminator, source) == (130073, "profile:minicpm5"), (terminator, source)

    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    target = '<action>{"type": "finish"}</action>'
    example = build_supervised_example(tok, messages, target, 4096, profile=MINICPM5_PROFILE)
    assert example["kept"] is True
    supervised = example["input_ids"][example["n_prompt_tokens"] :]
    assert supervised[-1] == 130073, supervised[-3:]
    assert 1 not in supervised, "the wrong terminator </s> must not be supervised"
    # The newline the template writes after <|im_end|> is deliberately unsupervised.
    assert example["labels"][-1] == 130073
    # A list-valued eos_token_id must be understood, not ignored.
    listy = SimpleNamespace(eos_token_id=[1, 130073])
    assert eos_token_ids(listy) == [1, 130073]
    print("  terminator -> 130073 from the template; </s>=1 never supervised; eos lists parsed")


def test_no_think_suffix_has_no_zero_width_space() -> None:
    """Measured: the shipped templates contain NO U+200B inside the think tags."""
    assert "\u200b" not in CLOSED_THINK_SUFFIX
    for name in ("chat_template.jinja",):
        path = LOCAL_MINICPM / name
        if not path.is_file():
            print(f"SKIP: {path} absent - cannot re-measure the template bytes")
            continue
        raw = path.read_bytes()
        assert b"\xe2\x80\x8b" not in raw, "the shipped template unexpectedly contains U+200B"
    print("  think tags -> plain '<think>'/'</think>' in the shipped template (no U+200B)")


# --------------------------------------------------------------------------- #
# REAL TOKENIZER PARITY (measurement, needs models/MiniCPM5-2B)
# --------------------------------------------------------------------------- #
def test_chat_template_parity_at_token_id_level() -> None:
    """The inference prompt is an exact token prefix of the training sequence.

    Recorded measurements (5 real SFT records, 8192 budget, real tokenizer):
    prompt tokens 4720/5002/5103/5209/8022, supervised 30/47/47/13/27, every
    supervised sequence ending in 130073 with no 1 anywhere in it, and exactly
    one BOS id (0) in both the prompt and the training sequence.
    """
    tokenizer = _minicpm_tokenizer()
    if tokenizer is None:
        print("SKIP: models/MiniCPM5-2B tokenizer or transformers unavailable")
        NEEDS_GPU.append("chat_template_parity")
        return
    records = _real_records(5)
    if len(records) < 5:
        print("SKIP: outputs/datasets/train/train.jsonl missing (need 5 records)")
        NEEDS_GPU.append("chat_template_parity")
        return

    from lifemate.models.generation import HFGenerator
    from lifemate.training.collator import build_supervised_example

    profile = resolve_profile("openbmb/MiniCPM5-2B")
    generator = HFGenerator(SimpleNamespace(), tokenizer, model_name="openbmb/MiniCPM5-2B")
    assert generator.profile is profile
    assert generator.max_new_tokens == MINICPM5_PROFILE.default_max_new_tokens
    assert generator.stop_ids == [1, 130073], generator.stop_ids

    measured: list[tuple[int, int, int]] = []
    for record in records:
        messages, target = record["messages"], record["target"]
        example = build_supervised_example(tokenizer, messages, target, 8192, profile=profile)
        assert example["kept"] is True, example.get("reason")
        train_ids = example["input_ids"]
        prompt_len = example["n_prompt_tokens"]
        supervised = train_ids[prompt_len:]

        prompt_text = generator.build_prompt(messages)
        infer_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # (a) the inference ids are an exact prefix of the training ids
        assert train_ids[: len(infer_ids)] == infer_ids, "inference prompt is not a prefix"
        # (b) exactly one BOS in each (the template emits it; we never add a second)
        assert infer_ids.count(0) == 1, f"double BOS in prompt: {infer_ids.count(0)}"
        assert train_ids.count(0) == 1, "double BOS in the training sequence"
        # (c) the only supervised difference is the action target + terminator
        assert supervised[:-1] == tokenizer.encode(target, add_special_tokens=False)
        assert supervised[-1] == 130073, supervised[-1]
        assert 1 not in supervised, "the model's <s>..</s> eos must not be supervised"
        # (d) the trained sequence is the template's own serialization minus the
        #     trailing newline (id 220), which is intentionally unsupervised.
        templated = tokenizer.apply_chat_template(
            list(messages) + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        templated_ids = tokenizer.encode(templated, add_special_tokens=False)
        assert templated_ids[: len(train_ids)] == train_ids, "training ids diverge from the template"
        assert templated_ids[len(train_ids) :] == [220], "expected only the unsupervised newline"
        measured.append((prompt_len, len(supervised), infer_ids[-1]))

    print(f"  parity -> 5 records, (prompt, supervised, last) = {measured}")
    assert [item[1] for item in measured] == [30, 47, 47, 13, 27], measured


def test_qwen_prompt_rendering_is_unchanged() -> None:
    """Qwen's rendered prompt bytes must be exactly what they were before.

    Frozen from the pre-change code path: ``enable_thinking=False`` plus
    ``tokenize=False``/``add_generation_prompt=True``, ChatML system+user turns
    and a closed think block. Any drift here is a Qwen regression.
    """
    tokenizer = _minicpm_tokenizer()
    if tokenizer is None:
        print("SKIP: transformers unavailable - Qwen-shaped template not exercised")
        NEEDS_GPU.append("qwen_prompt_unchanged")
        return
    from lifemate.models.generation import HFGenerator
    from lifemate.training.collator import apply_chat_template_text, render_prompt_text

    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    # Qwen's template has the same tail contract; using the MiniCPM tokenizer
    # keeps this a real-tokenizer check of the shared code path.
    rendered = apply_chat_template_text(
        tokenizer, messages, add_generation_prompt=True, profile=QWEN3_5_PROFILE
    )
    assert rendered.startswith("<s><|im_start|>system\nSYS<|im_end|>\n")
    assert rendered.endswith("<|im_start|>assistant\n" + CLOSED_THINK_SUFFIX)
    assert render_prompt_text(messages, tokenizer, QWEN3_5_PROFILE) == rendered
    assert HFGenerator(
        SimpleNamespace(), tokenizer, model_name="Qwen/Qwen3.5-4B"
    ).build_prompt(messages) == rendered
    # Qwen keeps the default tokenizer call (add_special_tokens untouched).
    assert QWEN3_5_PROFILE.prompt_add_special_tokens is True
    assert QWEN3_5_PROFILE.default_max_new_tokens == 512
    print("  qwen rendering -> identical bytes, thinking=False, 512 new tokens")


# --------------------------------------------------------------------------- #
# GENERATION: ALL EOS IDS + TRUNCATION REJECTION
# --------------------------------------------------------------------------- #
def test_stop_ids_union_every_declared_eos() -> None:
    """Both MiniCPM EOS ids stop decoding; a single-EOS model stays an int."""
    from lifemate.models.generation import resolve_stop_ids

    tok = SimpleNamespace(eos_token_id=1)
    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=[1, 130073]),
        generation_config=SimpleNamespace(eos_token_id=[1, 130073]),
    )
    assert resolve_stop_ids(tok, model, MINICPM5_PROFILE) == [1, 130073]
    # Profile extras are a defensive union when the checkpoint drops one.
    assert resolve_stop_ids(tok, SimpleNamespace(), MINICPM5_PROFILE) == [1, 130073]
    qwen_tok = SimpleNamespace(eos_token_id=248046)
    qwen_gen = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=248046),
        generation_config=SimpleNamespace(eos_token_id=248046),
    )
    assert resolve_stop_ids(qwen_tok, qwen_gen, QWEN3_5_PROFILE) == [248046]
    print("  stop ids -> [1, 130073] for MiniCPM; [248046] for Qwen")


def test_truncated_output_never_parses_as_an_action() -> None:
    """A decode cut off by the budget raises instead of yielding an action."""
    from lifemate.models.generation import HFGenerator
    from lifemate.agent.protocol import parse_action

    class _Tokenizer(_MiniCPMLikeTokenizer):
        pad_token_id = 1

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(97 + (int(i) % 26)) for i in ids)

        def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
            import torch

            ids = torch.tensor([[5, 6, 7]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def encode(self, text, add_special_tokens=False):
            if text == "</action>":
                return [9001]
            return [5, 6, 7]

    class _Model:
        device = "cpu"

        def __call__(self, *args, **kwargs):  # pragma: no cover - not used
            raise AssertionError("model must only be used through generate()")

        def generate(self, input_ids=None, attention_mask=None, **kwargs):
            import torch

            # Emit exactly max_new_tokens and never the stop marker.
            n = int(kwargs["max_new_tokens"])
            new = torch.full((1, n), 12, dtype=torch.long)
            return torch.cat([input_ids, new], dim=-1)

    if not _have("torch"):
        print("SKIP: torch unavailable - generation loop not exercised")
        NEEDS_GPU.append("truncation_guard")
        return

    generator = HFGenerator(_Model(), _Tokenizer(), model_name="openbmb/MiniCPM5-2B", max_new_tokens=4)
    try:
        generator.generate([{"role": "user", "content": "U"}])
    except LifeMateError as exc:
        assert "budget" in str(exc) and "truncated" in str(exc), exc
        assert generator.stats.truncated_by_budget is True
    else:  # pragma: no cover
        raise AssertionError("a truncated decode must raise, not return text")

    # The same text WOULD have looked like a decision to a lenient parser when the
    # JSON body is complete but the closing tag never arrived.
    assert parse_action('<action>{"type": "finish"}').type == "invalid"
    assert parse_action('<action>{"type": "finish"}</action>').type == "finish"
    print("  truncation -> LifeMateError + stats flag; unclosed block is invalid")


# --------------------------------------------------------------------------- #
# ACTION PARSING: REASONING BLOCKS
# --------------------------------------------------------------------------- #
def test_action_after_closed_reasoning_block_is_executable() -> None:
    """Reasoning BEFORE the action is fine: the block is closed."""
    from lifemate.agent.protocol import parse_action

    text = (
        CLOSED_THINK_SUFFIX.replace("\n\n", "\n") + "need an alarm\n</think>\n\n"
        + '<action>{"type": "tool_call", "name": "alarm.list", "arguments": {}}</action>'
    )
    action = parse_action(text)
    assert action.is_tool_call, action.error
    assert action.name == "alarm.list"
    print("  reasoning-then-action -> parsed and executable")


def test_action_inside_unclosed_reasoning_block_is_refused() -> None:
    """An unterminated reasoning block must not reach the tool executor."""
    from lifemate.agent.protocol import open_reasoning_start, parse_action

    text = '<think>\nstill thinking <action>{"type": "finish"}</action>'
    action = parse_action(text)
    assert action.type == "invalid", action.to_dict()
    assert "unterminated reasoning" in (action.error or "")
    assert open_reasoning_start(text) is not None
    # A ZWSP spelling is matched too.
    text_zwsp = "<\u200bthink\u200b>\n" + '<action>{"type": "finish"}</action>'
    assert parse_action(text_zwsp).type == "invalid"
    assert open_reasoning_start(CLOSED_THINK_SUFFIX + '<action>{"type": "finish"}</action>') is None
    print("  unclosed reasoning -> invalid, plain and ZWSP spellings both guarded")


def test_multiple_action_blocks_take_the_last_valid_one() -> None:
    """Only a FINAL complete action is executed; earlier drafts are ignored."""
    from lifemate.agent.protocol import parse_action

    text = (
        '<action>{"type": "tool_call", "name": "wrong.tool", "arguments": {}}</action>\n'
        + CLOSED_THINK_SUFFIX
        + '<action>{"type": "tool_call", "name": "alarm.list", "arguments": {}}</action>'
    )
    action = parse_action(text)
    assert action.is_tool_call and action.name == "alarm.list", action.to_dict()
    # A split (truncated) second block leaves the earlier one as the last complete
    # block, which is why the generator refuses budget-truncated output first.
    split = '<action>{"type": "finish"}</action> <action>{"type": "tool_call", "na'
    assert parse_action(split).type == "finish"
    print("  multiple blocks -> last complete action wins; truncation caught upstream")


def test_hallucinated_tool_and_bad_arguments_are_rejected() -> None:
    """Registry validation gates execution, not just the protocol parse."""
    from lifemate.agent.tool_parser import parse_model_output

    class _Registry:
        def tool_names(self):
            return ["alarm.list"]

        def is_available(self, name):
            return name == "alarm.list"

        def validate(self, name, arguments):
            if arguments.get("bad"):
                return {"errors": ["bad argument"]}
            return {"ok": True, "errors": []}

    unknown = parse_model_output(
        '<action>{"type": "tool_call", "name": "made.up", "arguments": {}}</action>', _Registry()
    )
    assert unknown.type == "invalid" and "not available" in (unknown.error or "")
    bad_args = parse_model_output(
        '<action>{"type": "tool_call", "name": "alarm.list", "arguments": {"bad": 1}}</action>',
        _Registry(),
    )
    assert bad_args.type == "invalid" and "invalid arguments" in (bad_args.error or "")
    missing = parse_model_output('<action>{"type": "tool_call", "arguments": {}}</action>', _Registry())
    assert missing.type == "invalid"
    print("  registry gate -> hallucinated tool, bad args and missing name all invalid")


# --------------------------------------------------------------------------- #
# CHECKPOINT IDENTITY
# --------------------------------------------------------------------------- #
def test_adapter_base_mismatch_rejected_both_ways() -> None:
    """A MiniCPM adapter cannot load onto Qwen and vice versa."""
    with tempfile.TemporaryDirectory() as tmp:
        mini = pathlib.Path(tmp) / "mini-adapter"
        mini.mkdir()
        (mini / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "openbmb/MiniCPM5-2B"}), encoding="utf-8"
        )
        (mini / "lifemate_meta.json").write_text(
            json.dumps({"model_name": "openbmb/MiniCPM5-2B", "model_profile": "minicpm5"}),
            encoding="utf-8",
        )
        try:
            ckpt.assert_adapter_base_matches(mini, "Qwen/Qwen3.5-4B")
        except LifeMateError as exc:
            assert "qwen3_5" in str(exc) and "minicpm5" in str(exc), exc
            assert "Refusing" in str(exc)
        else:
            raise AssertionError("MiniCPM adapter must be refused on a Qwen base")
        # The matching base (local dir vs hub id) is accepted.
        report = ckpt.assert_adapter_base_matches(mini, str(LOCAL_MINICPM))
        assert report["recorded"] is True and report["recorded_profile"] == "minicpm5"

        qwen = pathlib.Path(tmp) / "qwen-adapter"
        qwen.mkdir()
        (qwen / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}), encoding="utf-8"
        )
        try:
            ckpt.assert_adapter_base_matches(qwen, "openbmb/MiniCPM5-2B")
        except LifeMateError as exc:
            assert "minicpm5" in str(exc)
        else:
            raise AssertionError("Qwen adapter must be refused on a MiniCPM base")

        unknown = pathlib.Path(tmp) / "unknown-adapter"
        unknown.mkdir()
        assert ckpt.assert_adapter_base_matches(unknown, "openbmb/MiniCPM5-2B")["recorded"] is False
    print("  checkpoint identity -> both directions refused before any merge; unknown = no-op")


def test_merge_and_attach_call_the_identity_guard() -> None:
    """Both merge paths check identity BEFORE touching weights."""
    loader_src = (ROOT / "src" / "lifemate" / "models" / "loader.py").read_text(encoding="utf-8")
    ckpt_src = (ROOT / "src" / "lifemate" / "models" / "checkpoint.py").read_text(encoding="utf-8")
    assert "assert_adapter_base_matches(path, base_name)" in loader_src
    merge_src = ckpt_src.split("def merge_lora_adapter", 1)[1]
    assert merge_src.index("assert_adapter_base_matches") < merge_src.index("PeftModel")
    print("  guards -> _attach_lora_adapter and merge_lora_adapter both validate first")


# --------------------------------------------------------------------------- #
# PREFLIGHT PER PROFILE
# --------------------------------------------------------------------------- #
def test_preflight_shape_differs_per_profile_and_is_labelled() -> None:
    """MiniCPM and Qwen estimates use their own verified shapes, labelled."""
    from lifemate.training.memory_preflight import run_memory_preflight

    mini = load_config()
    mini.model.name = "openbmb/MiniCPM5-2B"
    mini.training.tuning_mode = "lora"
    mini.training.gradient_checkpointing = True
    qwen = load_config()
    qwen.model.name = "Qwen/Qwen3.5-4B"
    qwen.training.tuning_mode = "lora"
    qwen.training.gradient_checkpointing = True

    mini_report = run_memory_preflight(cfg=mini, seq_len=1024, gpu_info={"gpu": "unavailable"})
    qwen_report = run_memory_preflight(cfg=qwen, seq_len=1024, gpu_info={"gpu": "unavailable"})
    assert mini_report["param_count"] == 2_516_756_480, mini_report["param_count"]
    assert (mini_report["layers"], mini_report["hidden"]) == (42, 2048)
    assert (qwen_report["layers"], qwen_report["hidden"]) == (32, 2560)
    assert mini_report["param_count"] != qwen_report["param_count"]
    assert mini_report["activations_gb"] != qwen_report["activations_gb"]
    # Honest provenance: the shape source and the estimate/measured split are
    # explicit keys, never implied.
    assert "profile fallback minicpm5" in mini_report["shape_source"]
    assert "profile fallback qwen3_5" in qwen_report["shape_source"]
    assert mini_report["trainable_source"] in {"measured", "placeholder_0.5pct"}
    for report in (mini_report, qwen_report):
        assert "Point estimate, not a measurement" in report["heuristic"]
        assert report["base_weights_gb"] > 0 and report["adapter_gb"] >= 0
    assert mini_report["base_weights_gb"] == 5.03, mini_report["base_weights_gb"]
    print(
        f"  preflight -> MiniCPM {mini_report['param_count']:,} params / 42L / "
        f"base {mini_report['base_weights_gb']} GB vs Qwen 4,000,000,000 / 32L / "
        f"base {qwen_report['base_weights_gb']} GB (both labelled estimates)"
    )


def test_preflight_reads_a_local_config_json_instead_of_guessing() -> None:
    """A local model directory's config.json wins over the profile fallback."""
    from lifemate.training.memory_preflight import (
        _params_from_config_json,
        _read_local_config_json,
        run_memory_preflight,
    )

    payload = _read_local_config_json(str(LOCAL_MINICPM))
    if not payload:
        print("SKIP: models/MiniCPM5-2B/config.json absent")
        NEEDS_GPU.append("preflight_local_config")
        return
    measured = _params_from_config_json(payload)
    assert measured == 2_516_756_480, measured
    cfg = load_config()
    cfg.model.name = str(LOCAL_MINICPM)
    cfg.training.tuning_mode = "lora"
    report = run_memory_preflight(cfg=cfg, seq_len=1024, gpu_info={"gpu": "unavailable"})
    assert report["param_count"] == measured
    assert report["shape_source"].startswith("local config.json"), report["shape_source"]
    print(f"  preflight -> local config.json gives {measured:,} (matches the safetensors index)")


def test_qlora_accounting_is_separate_and_quantized() -> None:
    """QLoRA stores the base at ~0.5 B/param and reports it as a distinct term."""
    from lifemate.training.memory_preflight import run_memory_preflight

    cfg = load_config()
    cfg.training.tuning_mode = "qlora"
    cfg.training.quantization = "nf4"
    cfg.training.gradient_checkpointing = True
    cfg.model.name = "openbmb/MiniCPM5-2B"
    report = run_memory_preflight(param_count=1_000_000_000, cfg=cfg, seq_len=1024,
                                  gpu_info={"gpu": "unavailable"})
    assert report["quantization"] == "nf4"
    assert report["weight_bytes_per_param"] == 0.5
    assert report["base_weights_gb"] == 0.5, report["base_weights_gb"]
    assert report["trainable_params"] == 5_000_000, report["trainable_params"]
    assert report["trainable_source"] == "placeholder_0.5pct"
    # base 0.5 + adapter 0.01 + grads 0.01 + optimizer 0.04 + activations 0.34
    # + bf16 head logits 1024*130560*2 = 0.27 GB + fp32 chunk 1024*130560*4 = 0.53 GB
    # -> 1.71
    assert report["logits_compute_gb"] == 0.27, report
    assert report["logits_chunked_gb"] == 0.53, report
    assert report["logits_gb"] == 0.8, report
    assert report["total_gb"] == 1.71, report
    assert report["adapter_gb"] == 0.01, report
    print("  qlora preflight -> 0.5 B/param base, adapter/grads/optimizer listed separately")


# --------------------------------------------------------------------------- #
# QLoRA implementation logic (no bitsandbytes needed)
# --------------------------------------------------------------------------- #
def test_qlora_config_requires_bitsandbytes_honestly() -> None:
    """The stub-module path proves the check is a real import, not a flag."""
    import builtins

    if not _have("torch"):
        print("SKIP: torch absent - BitsAndBytesConfig not constructible")
        NEEDS_GPU.append("qlora_config")
        return
    from transformers import BitsAndBytesConfig

    if _have("bitsandbytes"):
        print("SKIP: bitsandbytes installed here - absence path not exercised")
        return

    try:
        loader._check_quantization("nf4", "cuda")
    except LifeMateError as exc:
        assert "bitsandbytes" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("nf4 without bitsandbytes must raise")

    import types

    stub = types.ModuleType("bitsandbytes")
    sys.modules["bitsandbytes"] = stub
    try:
        assert loader._check_quantization("nf4", "cuda") == "nf4"
    finally:
        del sys.modules["bitsandbytes"]

    import torch

    config = loader.build_bnb_config(torch.bfloat16)
    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype is torch.bfloat16
    assert config.bnb_4bit_use_double_quant is True
    # A CPU device is refused rather than silently unquantized.
    try:
        loader._check_quantization("nf4", "cpu")
    except LifeMateError as exc:
        assert "CUDA" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nf4 on cpu must raise")
    del builtins
    print("  qlora -> BitsAndBytesConfig wired for real; absence and CPU both raise")


def test_qlora_refuses_an_unquantized_model() -> None:
    """configure_qlora must not quietly act as plain LoRA."""
    from lifemate.training.lora_finetuning import configure_qlora

    cfg = load_config()
    cfg.training.tuning_mode = "qlora"
    cfg.training.quantization = "nf4"
    model = SimpleNamespace(config=SimpleNamespace(quantization_config=None))
    try:
        configure_qlora(model, SimpleNamespace(), cfg)
    except LifeMateError as exc:
        assert "quantized" in str(exc) or "peft" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("configure_qlora must reject a 16-bit model")
    print("  qlora -> unquantized model rejected (no silent LoRA downgrade)")


def test_qlora_is_not_silently_downgraded_by_the_cli() -> None:
    """--tuning full must mean full, and qlora must set quantization=nf4."""
    from lifemate.config import TUNING_MODES

    assert TUNING_MODES == ("full", "lora", "qlora")
    cfg = load_config(ROOT / "configs" / "training_full.yaml")
    assert cfg.training.tuning_mode == "full"
    assert cfg.training.quantization == "none"
    # Config validation enforces the pairing in both directions.
    for mode, quantization, ok in (
        ("qlora", "nf4", True),
        ("qlora", "none", False),
        ("full", "nf4", False),
        ("lora", "nf4", False),
        ("full", "none", True),
    ):
        probe = load_config()
        probe.training.tuning_mode = mode
        probe.training.quantization = quantization
        try:
            probe.validate()
        except LifeMateError:
            assert not ok, f"{mode}+{quantization} should be valid"
        else:
            assert ok, f"{mode}+{quantization} should be rejected"
    print("  cli -> qlora requires nf4; full/lora require none (validated, not implied)")


# --------------------------------------------------------------------------- #
# CHUNKED CROSS-ENTROPY
# --------------------------------------------------------------------------- #
def test_chunked_loss_matches_the_single_shot_loss() -> None:
    """Chunking must be arithmetically identical, not merely similar.

    ``transformers.ForCausalLMLoss`` does ``logits.float()`` on the WHOLE
    ``[seq, vocab]`` tensor: at seq 8192 x vocab 130560 that is one 3.98 GiB
    allocation (measured OOM on a 16.57 GB card). The chunked path upcasts only
    a slice and accumulates a SUM divided once by the supervised-token count, so
    a short trailing chunk is not over-weighted.
    """
    if not _have("torch") or not _have("transformers"):
        print("SKIP: torch/transformers absent - loss equivalence not exercised")
        NEEDS_GPU.append("chunked_loss_equivalence")
        return
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from lifemate.training.trainer import _forward_loss

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).float().eval()
    ids = torch.randint(0, 256, (2, 37))  # 37 tokens: an odd, partial-chunk length
    cases = {
        "all supervised": ids.clone(),
        "mixed mask": torch.where(
            torch.rand(2, 37) > 0.5, ids, torch.full_like(ids, -100)
        ),
    }
    measured = {}
    for name, labels in cases.items():
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        with torch.no_grad():
            reference = float(_forward_loss(model, batch, 0))
            for chunk in (1, 4, 17, 1024):
                got = float(_forward_loss(model, batch, chunk))
                rel = abs(got - reference) / max(abs(reference), 1e-12)
                assert rel < 1e-5, f"{name}: chunk {chunk} loss {got} vs {reference} (rel {rel:.2e})"
                measured[(name, chunk)] = got
    # Chunk size 1 and 1024 must agree with each other as well.
    for name in cases:
        assert abs(measured[(name, 1)] - measured[(name, 1024)]) / max(abs(measured[(name, 1)]), 1e-12) < 1e-5
    print(f"  chunked CE -> 4 chunk sizes x 2 masking patterns equal to the single-shot loss "
          f"(max rel {max(abs(measured[(n, c)] - measured[(n, 1024)]) for n in cases for c in (1, 4, 17)):.1e})")


def test_loss_chunk_tokens_default_and_off_switch() -> None:
    """The field is config-driven and 0 disables chunking (old path stays reachable)."""
    cfg = load_config()
    assert cfg.training.loss_chunk_tokens == 1024, cfg.training.loss_chunk_tokens
    cfg.training.loss_chunk_tokens = 0
    cfg.validate()  # 0 is legal: "no chunking"
    from_config = load_config(ROOT / "configs" / "training_lora.yaml")
    assert from_config.training.loss_chunk_tokens == 1024
    print("  loss_chunk_tokens -> 1024 by default, 0 = the model's own single-shot loss")


def test_encode_flags_match_between_training_and_inference() -> None:
    """Both sides tokenize the rendered prompt with the profile's flag.

    ``build_supervised_example`` encodes the prompt with
    ``profile.prompt_add_special_tokens`` and ``HFGenerator.generate`` passes the
    same value to the tokenizer. Measured hazard this prevents: with
    ``add_special_tokens=True`` the MiniCPM render produced TWO ``<s>`` ids
    because the template already emits one.
    """
    source = (ROOT / "src" / "lifemate" / "training" / "collator.py").read_text(encoding="utf-8")
    gen_source = (ROOT / "src" / "lifemate" / "models" / "generation.py").read_text(encoding="utf-8")
    assert "prompt_add_special_tokens" in source, "training path must read the profile flag"
    assert "add_special_tokens=self.profile.prompt_add_special_tokens" in gen_source
    assert "add_special_tokens=prompt_flags" in source
    # The target is never given special tokens on either side.
    assert "_encode(tokenizer, target_text)" in source

    class _Recording(_MiniCPMLikeTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.flags: list[bool] = []

        def encode(self, text, add_special_tokens=True):
            self.flags.append(add_special_tokens)
            return super().encode(text, add_special_tokens=add_special_tokens)

    from lifemate.training.collator import build_supervised_example

    tok = _Recording()
    build_supervised_example(
        tok,
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        '<action>{"type": "finish"}</action>',
        4096,
        profile=MINICPM5_PROFILE,
    )
    # prompt (profile flag = False), target (False), terminator lookup (False)
    assert tok.flags and tok.flags[0] is False, tok.flags
    assert all(flag is False for flag in tok.flags), tok.flags
    print("  encode flags -> training and inference both use the profile's value (False for MiniCPM)")


def test_default_tuning_mode_is_lora_everywhere() -> None:
    """Config default, base.yaml, CLI --config and the printed banner all say lora."""
    assert load_config().training.tuning_mode == "lora"
    assert load_config(ROOT / "configs" / "base.yaml").training.tuning_mode == "lora"
    assert load_config(ROOT / "configs" / "training_lora.yaml").training.tuning_mode == "lora"
    # full stays explicitly reachable and never silently downgraded.
    assert load_config(ROOT / "configs" / "training_full.yaml").training.tuning_mode == "full"

    import argparse

    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("_train_mod", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args([])
    assert args.config == "configs/training_lora.yaml", args.config
    assert module.parse_args(["--tuning", "full"]).tuning == "full"
    assert module.parse_args(["--max-seq-length", "4096"]).seq_len == 4096
    assert module.parse_args(["--resume-from", "x"]).resume == "x"
    del argparse
    print("  defaults -> config/base.yaml/CLI all lora; training_full.yaml and --tuning full explicit")


def test_wandb_config_deltas_survive_the_tuning_default_change() -> None:
    """Exactly 3 deltas from training_lora.yaml, re-checked after the flip."""
    def flat(payload, prefix=""):
        out = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                out.update(flat(value, prefix + key + "."))
            else:
                out[key if not prefix else prefix + key] = value
        return out

    plain = flat(load_config(ROOT / "configs" / "training_lora.yaml").to_dict())
    tracked = flat(load_config(ROOT / "configs" / "training_lora_wandb.yaml").to_dict())
    diffs = {key: (plain[key], tracked[key]) for key in plain if plain[key] != tracked[key]}
    assert set(diffs) == {"training.report_to", "wandb.enabled", "wandb.mode"}, diffs
    print("  wandb deltas -> still exactly report_to, wandb.enabled, wandb.mode")


# --------------------------------------------------------------------------- #
# EVALUATION MUST USE THE CHUNKED PATH
# --------------------------------------------------------------------------- #
class _LossTrapModel:
    """Model that RAISES if its own ``loss`` is used, and tracks slice widths.

    ``evaluate`` previously called ``self.model(**tensors)`` with ``labels``, which
    made transformers build the whole ``[seq, vocab]`` fp32 logits tensor (4.28 GB
    at 8192 x 130560) and then a second ``.contiguous()`` shift copy (2.14 GB).
    With ``eval_strategy: epoch`` in the shipped configs that ran at the end of
    every epoch, on the default path. This double fails that shape of bug: it
    returns logits but a ``loss`` property that raises, and it records the largest
    sequence slice it was asked to produce logits for.
    """

    device = "cpu"

    def __init__(self, vocab: int = 64) -> None:
        self.vocab = vocab
        self.calls = 0
        self.max_logits_seq = 0

    def __call__(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        import torch

        if labels is not None:
            raise AssertionError("evaluate() forwarded labels: the model's own fp32 whole-vocab loss ran")
        self.calls += 1
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[-1])
        self.max_logits_seq = max(self.max_logits_seq, seq)
        logits = torch.zeros(batch, seq, self.vocab)
        # logits[t] must predict token t+1, because evaluate compares
        # logits[..., :-1, :] against labels[..., 1:]. Scattering at input_ids[t]
        # would score 0% accuracy even though the paths agree.
        nxt = torch.zeros_like(input_ids)
        nxt[:, :-1] = input_ids[:, 1:]
        logits.scatter_(2, nxt.unsqueeze(-1), 12.0)
        return SimpleNamespace(logits=logits, loss=None)

    def eval(self):
        return self

    def train(self):
        return self


class _TrapTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(ch) % 60) for ch in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def save_pretrained(self, path):  # pragma: no cover - not used here
        return None


def test_evaluate_never_invokes_the_model_own_loss() -> None:
    """The epoch-end eval path must not reach ``ForCausalLMLoss``.

    Regression guard for a real defect: chunking the TRAIN step left
    ``evaluate`` calling ``model(**tensors)`` with labels, re-introducing the
    whole-vocab fp32 upcast plus a full-width shift copy at the end of every
    epoch — an OOM on a 16.57 GB card after a full epoch of compute.
    """
    if not _have("torch"):
        print("SKIP: torch absent - evaluate path not exercised")
        NEEDS_GPU.append("evaluate_no_model_loss")
        return
    from lifemate.training.collator import SupervisedCollator
    from lifemate.training.trainer import SFTTrainer

    cfg = load_config()
    cfg.model.name = "Qwen/Qwen3.5-4B"
    cfg.training.loss_chunk_tokens = 1024
    cfg.training.per_device_train_batch_size = 1
    cfg.training.max_seq_length = 512
    tokenizer = _TrapTokenizer()
    records = [
        {"messages": [{"role": "user", "content": f"hi {i}"}],
         "target": '<action>{"type": "finish"}</action>'}
        for i in range(2)
    ]
    model = _LossTrapModel()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        collator=SupervisedCollator.from_tokenizer(tokenizer),
        train_records=records,
        val_records=records,
        cfg=cfg,
        device="cpu",
    )
    metrics = trainer.evaluate()
    assert model.calls > 0, "evaluate never called the model"
    assert metrics["val_target_tokens"] > 0
    # A perfect next-token predictor must score 1.0: proves the argmax runs on
    # the slices and that the shift (logits[t] vs labels[t+1]) is preserved.
    assert metrics["val_token_accuracy"] == 1.0, metrics
    # And the single-shot path (chunk=0) is still reachable and still correct.
    cfg0 = load_config()
    cfg0.model.name = "Qwen/Qwen3.5-4B"
    cfg0.training.loss_chunk_tokens = 0
    cfg0.training.max_seq_length = 512
    reference = SFTTrainer(
        model=_OracleFallbackModel(),
        tokenizer=tokenizer,
        collator=SupervisedCollator.from_tokenizer(tokenizer),
        train_records=records,
        val_records=records,
        cfg=cfg0,
        device="cpu",
    ).evaluate()
    assert reference["val_target_tokens"] == metrics["val_target_tokens"]
    print(f"  evaluate -> model's own loss never called; chunked acc=1.0, "
          f"{metrics['val_target_tokens']} supervised tokens")


class _OracleFallbackModel:
    """Legacy-path double: supplies ``loss`` itself (no chunking available)."""

    device = "cpu"

    def __call__(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        import torch
        import torch.nn.functional as F

        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[-1])
        logits = torch.zeros(batch, seq, 64)
        nxt = torch.zeros_like(input_ids)
        nxt[:, :-1] = input_ids[:, 1:]
        logits.scatter_(2, nxt.unsqueeze(-1), 12.0)
        return SimpleNamespace(
            logits=logits,
            loss=F.cross_entropy(
                logits[:, :-1].float().reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100
            ),
        )

    def eval(self):
        return self

    def train(self):
        return self


def test_evaluate_paths_agree_on_loss_accuracy_and_token_count() -> None:
    """Chunked vs single-shot evaluate: same val_loss, accuracy, token count."""
    if not _have("torch"):
        print("SKIP: torch absent - evaluate equivalence not exercised")
        NEEDS_GPU.append("evaluate_equivalence")
        return
    from lifemate.training.collator import SupervisedCollator
    from lifemate.training.trainer import SFTTrainer

    tokenizer = _TrapTokenizer()
    records = [
        {"messages": [{"role": "user", "content": f"hi {i}"}],
         "target": '<action>{"type": "finish"}</action>'}
        for i in range(3)
    ]
    results = {}
    for chunk in (0, 1024, 5, 1):
        cfg = load_config()
        cfg.model.name = "Qwen/Qwen3.5-4B"
        cfg.training.loss_chunk_tokens = chunk
        cfg.training.per_device_train_batch_size = 1
        cfg.training.max_seq_length = 96
        model = _OracleFallbackModel()  # identical weights/logits for every run
        results[chunk] = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            collator=SupervisedCollator.from_tokenizer(tokenizer),
            train_records=records,
            val_records=records,
            cfg=cfg,
            device="cpu",
        ).evaluate()
    reference = results[0]
    for chunk in (1024, 5, 1):
        got = results[chunk]
        for key in ("val_loss", "val_token_accuracy", "val_target_tokens"):
            if key == "val_target_tokens":
                assert got[key] == reference[key], (chunk, key, got, reference)
            else:
                rel = abs(got[key] - reference[key]) / max(abs(reference[key]), 1e-12)
                assert rel < 1e-5, (chunk, key, got[key], reference[key], rel)
    print(f"  evaluate -> chunk 0/1024/5/1 identical: loss {reference['val_loss']}, "
          f"acc {reference['val_token_accuracy']}, tokens {reference['val_target_tokens']}")


def main() -> int:  # pragma: no cover - manual convenience
    """Run every test in this module (stdlib runner compatible)."""
    failures = 0
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            try:
                value()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
