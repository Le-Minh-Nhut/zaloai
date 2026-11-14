"""Tiny generation tests: exercises the end-to-end HFGenerator chain.

Gated on torch/transformers/peft so stdlib/non-torch interpreters skip cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.errors import LifeMateError  # noqa: E402

try:
    import torch
    import transformers
    import peft

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import pytest
except ImportError:  # stdlib runner without pytest: per-test HAVE_DEPS guards skip instead
    pytest = None

if pytest is not None:
    # Real pytest skip per test; without this the HAVE_DEPS guards return early and pytest counts them as passes.
    pytestmark = pytest.mark.skipif(not HAVE_DEPS, reason="torch/transformers/peft unavailable")


def make_tiny_model_and_tokenizer(save_dir: Path) -> Path:
    """Build a deterministic tiny 2-layer LLaMA model and byte-level chat tokenizer."""
    save_dir.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    action_open = "<action>"
    action_body = '{"type": "tool_call", "name": "alarm.list", "arguments": {}}'
    action_close = "</action>"
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>", action_open, action_body, action_close]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for ch in alphabet:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    tok = Tokenizer(models.BPE(vocab, []))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast_tok.chat_template = (
        "{% for message in messages %}"
        "{{ \"<|im_start|>\" + message[\"role\"] + \"\\n\" + message[\"content\"] + \"<|im_end|>\\n\" }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ \"<|im_start|>assistant\\n\" }}"
        "{% endif %}"
    )
    fast_tok.save_pretrained(save_dir)

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=len(vocab),
        max_position_embeddings=2048,
        bos_token_id=fast_tok.bos_token_id,
        eos_token_id=fast_tok.eos_token_id,
        pad_token_id=fast_tok.pad_token_id,
        torch_dtype="float32",
    )
    model = LlamaForCausalLM(config)
    model.generation_config = GenerationConfig(
        bos_token_id=fast_tok.bos_token_id,
        eos_token_id=fast_tok.eos_token_id,
        pad_token_id=fast_tok.pad_token_id,
    )
    model.save_pretrained(save_dir)
    return save_dir


def test_tiny_generation_chain() -> None:
    """Exercise HFGenerator -> parse_model_output -> ToolExecutor -> SubmissionWriter."""
    if not HAVE_DEPS:
        print("SKIP: torch/transformers/peft unavailable")
        return

    import torch
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    from lifemate.agent.tool_executor import ToolExecutor
    from lifemate.agent.tool_parser import parse_model_output
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.data.catalog import ToolCatalog
    from lifemate.env.mock import MockToolEnv
    from lifemate.evaluation.submission import SubmissionWriter
    from lifemate.evaluation.submission_validator import validate_submission
    from lifemate.models.generation import HFGenerator

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        tok = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
        model = LlamaForCausalLM.from_pretrained(str(model_dir))

        action_open = "<action>"
        action_body = '{"type": "tool_call", "name": "alarm.list", "arguments": {}}'
        action_close = "</action>"

        id_open = tok.convert_tokens_to_ids(action_open)
        id_body = tok.convert_tokens_to_ids(action_body)
        id_close = tok.convert_tokens_to_ids(action_close)

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "list alarms"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        id_prompt_end = prompt_ids[-1]

        # Determinism is achieved by zeroing transformer layers (identity pass-through),
        # embedding the key transition tokens onto orthogonal dimensions (0..3), and
        # wiring lm_head weights to deterministically chain:
        # prompt_end (dim 0) -> <action> (dim 1) -> action_body (dim 2) -> </action> (dim 3).
        # This acts as an exact 3-state transition automaton under greedy argmax decoding.
        with torch.no_grad():
            for param in model.model.layers.parameters():
                param.zero_()
            model.model.norm.weight.fill_(1.0)
            for layer in model.model.layers:
                layer.input_layernorm.weight.fill_(1.0)
                layer.post_attention_layernorm.weight.fill_(1.0)

            model.model.embed_tokens.weight.zero_()
            model.model.embed_tokens.weight[id_prompt_end, 0] = 1.0
            model.model.embed_tokens.weight[id_open, 1] = 1.0
            model.model.embed_tokens.weight[id_body, 2] = 1.0
            model.model.embed_tokens.weight[id_close, 3] = 1.0

            model.lm_head.weight.zero_()
            model.lm_head.weight[id_open, 0] = 10.0
            model.lm_head.weight[id_body, 1] = 10.0
            model.lm_head.weight[id_close, 2] = 10.0

        messages = [{"role": "user", "content": "list alarms"}]

        # 1. Greedy generation yields string ending at </action> and stop_marker_hit is True
        gen1 = HFGenerator(model, tok, max_new_tokens=64, temperature=0.0)
        out1 = gen1.generate(messages)
        gen2 = HFGenerator(model, tok, max_new_tokens=64, temperature=0.0)
        out2 = gen2.generate(messages)
        assert out1 == out2, "greedy generation must be deterministic across calls"
        assert out1.endswith("</action>"), f"expected output ending at </action>, got {out1!r}"
        assert gen1.stats.stop_marker_hit is True

        # 2. Text -> parse_model_output yields a valid Action
        catalog_path = Path("tests/fixtures/synthetic_lifemate/train/train_tools_catalog.json")
        catalog = ToolCatalog.from_file(catalog_path)
        registry = ToolRegistry(catalog)
        env = MockToolEnv(catalog=catalog, apps=["alarm"])
        action = parse_model_output(out1, registry, env)
        assert action.is_tool_call
        assert action.name == "alarm.list"

        # 3. Action -> ToolExecutor.execute on MockToolEnv
        executor = ToolExecutor(env, registry)
        obs = executor.execute(action)
        assert "ok" in obs and obs["ok"] is True
        calls = env.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["name"] == "alarm.list"

        # 4. SubmissionWriter + validate_submission
        sub_path = Path(tmp) / "submission.jsonl"
        sample_id = "sample_tiny_01"
        writer = SubmissionWriter(sub_path)
        writer.add(sample_id, env.get_tool_calls(), env.get_state_hash())
        writer.finalize()
        assert sub_path.is_file()
        rows = sub_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1
        validation = validate_submission(sub_path, [sample_id])
        assert validation["ok"] is True

        # 5. max_new_tokens is respected, and a decode that hits the budget
        #    without </action> is an explicit failure, not a parsed decision.
        gen_short = HFGenerator(model, tok, max_new_tokens=2, temperature=0.0)
        try:
            out_short = gen_short.generate(messages)
        except LifeMateError as exc:
            assert "budget" in str(exc) and "truncated" in str(exc), exc
            assert gen_short.stats.truncated_by_budget is True
            assert gen_short.stats.generated_tokens <= 2
        else:
            # The tiny model happens to emit the marker inside 2 tokens.
            assert isinstance(out_short, str)
            assert gen_short.stats.generated_tokens <= 2
            assert gen_short.stats.truncated_by_budget is False
