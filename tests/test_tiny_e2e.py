"""End-to-end tiny model CPU tests (offline, gated on torch/transformers/peft).

Exercises:
1. Loader round-trip with a local random causal LM + tokenizer.
2. Prompt token consistency between training collator and inference HFGenerator.
3. Full fine-tuning 2 steps and checkpoint reload with identical logits.
4. LoRA 2 steps with verified adapter tensor change (gradient flow).
5. LoRA merge and reload with parameter count match and logits equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

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
    """Build a deterministic 2-layer LLaMA model and byte-level chat tokenizer."""
    save_dir.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>", "<action>", "</action>"]
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
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<action>", "</action>"],
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


def _synthetic_records(n: int = 4) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(n):
        tool = "alarm.set" if index % 2 else "alarm.list"
        target = f'<action>{{"type": "tool_call", "name": "{tool}", "arguments": {{}}}}</action>'
        records.append({
            "sample_id": f"smoke_{index}",
            "step_index": index,
            "messages": [
                {"role": "system", "content": "Emit one action block."},
                {"role": "user", "content": f"Turn on {tool} now."},
            ],
            "target": target,
        })
    return records


def test_tiny_loader_round_trip() -> None:
    if not HAVE_DEPS:
        print("SKIP: torch/transformers/peft unavailable")
        return
    from lifemate.errors import LifeMateError
    from lifemate.models.loader import count_parameters, load_model_and_tokenizer, resolve_model_mode

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle = load_model_and_tokenizer(
            str(model_dir),
            device="cpu",
            torch_dtype="float32",
            tuning_mode="full",
            for_training=True,
        )
        counts = count_parameters(bundle.model)
        assert counts["total"] == counts["trainable"] > 0
        assert counts["frozen"] == 0

        assert resolve_model_mode("full", None) == "base"
        assert resolve_model_mode("full", "/some/ckpt") == "full"
        assert resolve_model_mode("lora", "/some/ckpt") == "lora"

        try:
            load_model_and_tokenizer(str(Path(tmp) / "nonexistent"), device="cpu", torch_dtype="float32")
        except LifeMateError:
            pass
        else:
            raise AssertionError("expected LifeMateError for nonexistent path")


def test_tiny_prompt_token_consistency() -> None:
    if not HAVE_DEPS:
        print("SKIP: torch/transformers/peft unavailable")
        return
    from lifemate.agent.prompt_builder import build_decision_prompt, to_messages
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.data.loader import load_catalog, load_samples
    from lifemate.models.generation import HFGenerator
    from lifemate.models.loader import load_model_and_tokenizer
    from lifemate.training.collator import _encode, render_prompt_text

    fixtures = Path("tests/fixtures/synthetic_lifemate")
    samples = load_samples(fixtures, split="train")
    catalog = load_catalog(fixtures, split="train")
    sample = samples[0]
    registry = ToolRegistry(catalog).for_sample(sample)
    prompt_text = build_decision_prompt(
        transcript=sample.audio_text,
        context=sample.context,
        history=sample.history,
        installed_apps=sample.installed_apps,
        catalog=registry,
    )
    messages = to_messages(prompt_text)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        bundle = load_model_and_tokenizer(str(model_dir), device="cpu", torch_dtype="float32")
        collator_text = render_prompt_text(messages, bundle.tokenizer)
        collator_tokens = _encode(bundle.tokenizer, collator_text)

        generator = HFGenerator(bundle.model, bundle.tokenizer)
        gen_text = generator.build_prompt(messages)
        gen_inputs = bundle.tokenizer(gen_text, return_tensors="pt")
        gen_tokens = gen_inputs["input_ids"][0].tolist()

        assert collator_tokens == gen_tokens, "train/inference prompt token mismatch"


def test_tiny_full_train_and_reload() -> None:
    if not HAVE_DEPS:
        print("SKIP: torch/transformers/peft unavailable")
        return
    import torch
    from transformers import AutoModelForCausalLM
    from lifemate.config import load_config
    from lifemate.models.loader import load_model_and_tokenizer
    from lifemate.tracking.wandb_tracker import WandbTracker
    from lifemate.training.collator import SupervisedCollator
    from lifemate.training.trainer import SFTTrainer

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        ckpt_root = Path(tmp) / "ckpts"
        cfg = load_config(None)
        cfg.training.output_dir = str(ckpt_root)
        cfg.training.precision = "float32"
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 2
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.gradient_checkpointing = False
        cfg.training.save_strategy = "steps"
        cfg.training.logging_steps = 1
        cfg.model.device = "cpu"

        bundle = load_model_and_tokenizer(
            str(model_dir),
            device="cpu",
            torch_dtype="float32",
            tuning_mode="full",
            trainable=True,
            for_training=True,
        )
        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        trainer = SFTTrainer(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            collator=collator,
            train_records=_synthetic_records(4),
            val_records=_synthetic_records(2),
            cfg=cfg,
            tracker=WandbTracker(None),
            device="cpu",
            run_name="full",
            max_steps=2,
            is_lora=False,
        )
        result = trainer.run()
        assert result.steps == 2
        assert result.final_loss is not None and result.final_loss > 0

        ckpt_dir = ckpt_root / "full" / "step-2"
        assert (ckpt_dir / "training_state.json").exists()

        reloaded = load_model_and_tokenizer(
            str(model_dir),
            checkpoint=ckpt_dir,
            device="cpu",
            torch_dtype="float32",
            tuning_mode="full",
        )
        direct = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.float32)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            out1 = reloaded.model(input_ids).logits
            out2 = direct(input_ids).logits
        assert (out1 - out2).abs().max().item() == 0.0


def test_tiny_lora_train_and_adapter_change() -> None:
    if not HAVE_DEPS:
        print("SKIP: torch/transformers/peft unavailable")
        return
    import torch
    from safetensors.torch import load_file
    from lifemate.config import load_config
    from lifemate.models.loader import load_model_and_tokenizer
    from lifemate.tracking.wandb_tracker import WandbTracker
    from lifemate.training.collator import SupervisedCollator
    from lifemate.training.lora_finetuning import configure_lora
    from lifemate.training.trainer import SFTTrainer

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        ckpt_root = Path(tmp) / "ckpts_lora"
        cfg = load_config(None)
        cfg.training.output_dir = str(ckpt_root)
        cfg.training.precision = "float32"
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 2
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.gradient_checkpointing = False
        cfg.training.save_strategy = "steps"
        cfg.training.logging_steps = 1
        cfg.training.tuning_mode = "lora"
        cfg.model.device = "cpu"

        bundle = load_model_and_tokenizer(
            str(model_dir),
            device="cpu",
            torch_dtype="float32",
            tuning_mode="lora",
            trainable=True,
            for_training=True,
        )
        report = configure_lora(bundle.model, cfg)
        assert report["trainable_params"] > 0
        assert report["frozen_params"] > 0

        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        trainer = SFTTrainer(
            model=report["model"],
            tokenizer=bundle.tokenizer,
            collator=collator,
            train_records=_synthetic_records(4),
            val_records=_synthetic_records(2),
            cfg=cfg,
            tracker=WandbTracker(None),
            device="cpu",
            run_name="lora",
            max_steps=2,
            is_lora=True,
        )
        result = trainer.run()
        assert result.steps == 2

        ckpt_dir = ckpt_root / "lora" / "step-2"
        adapter_weights = load_file(str(ckpt_dir / "adapter_model.safetensors"))
        b_tensors = {k: v for k, v in adapter_weights.items() if "lora_B" in k}
        assert b_tensors, "no lora_B tensors found"
        # lora_B initialized to 0; after gradient flow, at least one tensor has non-zero elements
        assert any((v.abs() > 0).any().item() for v in b_tensors.values()), "adapter weights did not change"


def test_tiny_lora_merge_loads() -> None:
    if not HAVE_DEPS:
        print("SKIP: torch/transformers/peft unavailable")
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from lifemate.config import load_config
    from lifemate.models.checkpoint import merge_lora_adapter
    from lifemate.models.loader import count_parameters, load_model_and_tokenizer
    from lifemate.tracking.wandb_tracker import WandbTracker
    from lifemate.training.collator import SupervisedCollator
    from lifemate.training.lora_finetuning import configure_lora
    from lifemate.training.trainer import SFTTrainer

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "model")
        ckpt_root = Path(tmp) / "ckpts_lora"
        cfg = load_config(None)
        cfg.training.output_dir = str(ckpt_root)
        cfg.training.precision = "float32"
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 2
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.gradient_checkpointing = False
        cfg.training.logging_steps = 1
        cfg.training.save_strategy = "steps"
        cfg.training.tuning_mode = "lora"
        cfg.model.device = "cpu"

        bundle = load_model_and_tokenizer(
            str(model_dir),
            device="cpu",
            torch_dtype="float32",
            tuning_mode="lora",
            trainable=True,
            for_training=True,
        )
        report = configure_lora(bundle.model, cfg)
        collator = SupervisedCollator.from_tokenizer(bundle.tokenizer)
        trainer = SFTTrainer(
            model=report["model"],
            tokenizer=bundle.tokenizer,
            collator=collator,
            train_records=_synthetic_records(4),
            val_records=[],
            cfg=cfg,
            tracker=WandbTracker(None),
            device="cpu",
            run_name="lora",
            max_steps=2,
            is_lora=True,
        )
        trainer.run()
        ckpt_dir = ckpt_root / "lora" / "step-2"

        # Base model for reference
        base_ref = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
        base_counts = count_parameters(base_ref)

        # Base + adapter
        adapter_model = PeftModel.from_pretrained(base_ref, str(ckpt_dir))
        adapter_model.eval()

        input_ids = torch.tensor([[1, 5, 10, 20]])
        with torch.no_grad():
            adapter_logits = adapter_model(input_ids).logits

        merged_dir = Path(tmp) / "merged"
        merge_lora_adapter(str(model_dir), ckpt_dir, merged_dir, torch_dtype="float32")

        # Standalone load (no peft)
        merged_model = AutoModelForCausalLM.from_pretrained(str(merged_dir), torch_dtype=torch.float32)
        merged_model.eval()
        merged_counts = count_parameters(merged_model)

        assert merged_counts["total"] == base_counts["total"]

        with torch.no_grad():
            merged_logits = merged_model(input_ids).logits

        rel_norm = (merged_logits - adapter_logits).norm() / adapter_logits.norm()
        assert rel_norm.item() <= 1e-3, f"relative norm diff {rel_norm.item()} > 1e-3"
