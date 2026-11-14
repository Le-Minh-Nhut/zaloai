"""QLoRA invariants on a REAL 4-bit-shaped stand-in (torch/peft gated).

bitsandbytes is absent from the torch environment of record, so the 4-bit base
is modelled with a ``Params4bit``-like double: a real ``torch.nn.Parameter``
subclass whose class name and dtype read as quantized, and whose
``requires_grad_(True)`` raises ``RuntimeError`` exactly like bitsandbytes. That
is enough to exercise every invariant QLoRA actually depends on:

* the base decision (full trains, lora/qlora never),
* the adapter decision (training only),
* only ``lora_`` parameters trainable after configuration,
* a frozen base + live adapter gradient,
* an optimizer step that really moves an adapter weight,
* adapter save → reload with a bitwise-equal result,
* a trainable quantized parameter making the loader raise by name.

Without torch/peft every test prints ``SKIP`` and returns (the convention used
in ``tests/test_training.py``), so the stdlib box still runs the module.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import torch
    import transformers
    import peft

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import pytest
except ImportError:  # stdlib runner without pytest: the HAVE_DEPS guards skip instead
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not HAVE_DEPS, reason="torch/transformers/peft unavailable (needs /tmp/cudaenv)"
    )

# --------------------------------------------------------------------------- #
# the 4-bit stand-in
# --------------------------------------------------------------------------- #
def _make_params4bit(data):
    """A ``Params4bit``-like parameter: records every requires_grad attempt.

    The class name is what the loader's quantized-parameter check reads, and
    ``requires_grad_(True)`` refuses — the real bitsandbytes behaviour this test
    exists to pin. Weights stay floating-point so the *adapter* can still be
    trained through the surrounding module.
    """

    class Params4bit(torch.nn.Parameter):
        def __new__(cls, tensor):
            obj = super().__new__(cls, tensor, requires_grad=False)
            obj.journal = []
            obj._rg = False
            return obj

        @property
        def requires_grad(self):
            return self._rg

        @requires_grad.setter
        def requires_grad(self, value):
            self.journal.append(bool(value))
            self._rg = bool(value)

        def requires_grad_(self, requires_grad=True):
            self.journal.append(bool(requires_grad))
            if requires_grad:
                raise RuntimeError("Params4bit: a 4-bit parameter cannot require grad")
            self._rg = False
            return self

    return Params4bit(data)


def _quantize_in_place(model, suffixes=("q_proj", "k_proj")):
    """Replace matching ``Linear.weight`` tensors with 4-bit doubles."""
    doubles = []
    for name, module in model.named_modules():
        if any(name.endswith(suffix) for suffix in suffixes) and hasattr(module, "weight"):
            double = _make_params4bit(module.weight.detach().clone())
            module._parameters["weight"] = double
            doubles.append((f"{name}.weight", double))
    assert doubles, "no projection matched; the stand-in would be vacuous"
    return doubles


def _frozen_quantization_config(model):
    """``config.quantization_config`` shaped exactly like a real nf4 load."""
    model.config.quantization_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float32",
        "bnb_4bit_use_double_quant": True,
        "quant_method": "bitsandbytes",
    }
    return model


def _tiny_llama(vocab: int = 64):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


def _cfg(targets=("q_proj", "k_proj"), r: int = 2):
    from lifemate.config import load_config

    cfg = load_config()
    cfg.lora.target_modules = list(targets)
    cfg.lora.r = r
    cfg.lora.alpha = 2 * r
    cfg.lora.dropout = 0.0
    cfg.lora.bias = "none"
    cfg.training.gradient_checkpointing = False
    return cfg


def _token_ids(n: int = 8, batch: int = 2, vocab: int = 64):
    return torch.randint(0, vocab, (batch, n))


# --------------------------------------------------------------------------- #
# decisions (the three concerns, separately)
# --------------------------------------------------------------------------- #
def test_base_decision_per_tuning_mode() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora base-decision test: torch/peft not installed")
        return
    from lifemate.models.loader import base_requires_grad

    model = _tiny_llama()
    assert base_requires_grad(model, tuning_mode="qlora", training=True) is False
    assert base_requires_grad(model, tuning_mode="lora", training=True) is False
    assert base_requires_grad(model, tuning_mode="full", training=True) is True
    assert base_requires_grad(model, tuning_mode="full", training=False) is False
    print("  base decision -> qlora False / lora False / full True / inference False")


def test_adapter_decision_training_only() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora adapter-decision test: torch/peft not installed")
        return
    from lifemate.models.loader import adapter_requires_grad

    model = _tiny_llama()
    assert adapter_requires_grad(model, training=True) is True
    assert adapter_requires_grad(model, training=False) is False
    print("  adapter decision -> True while training, False at inference")


# --------------------------------------------------------------------------- #
# LoRA: only lora_ trains, gradients flow, the step moves a weight
# --------------------------------------------------------------------------- #
def test_configure_lora_trains_only_adapters() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora configure_lora test: torch/peft not installed")
        return
    from lifemate.training.lora_finetuning import configure_lora

    report = configure_lora(_tiny_llama(), _cfg())
    model = report["model"]
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable, "no adapter parameter is trainable"
    assert all("lora_" in name for name in trainable), trainable
    assert report["trainable_params"] == sum(
        param.numel() for _, param in model.named_parameters() if param.requires_grad
    )
    print(f"  configure_lora -> {report['trainable_params']:,} trainable "
          f"({report['trainable_percentage']}%), all 'lora_' names")


def test_configure_qlora_freezes_the_quantized_base() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora configure_qlora test: torch/peft not installed")
        return
    from lifemate.training.lora_finetuning import configure_qlora

    model = _frozen_quantization_config(_tiny_llama())
    doubles = _quantize_in_place(model)
    report = configure_qlora(model, None, _cfg())
    peft_model = report["model"]

    trainable = [name for name, param in peft_model.named_parameters() if param.requires_grad]
    assert trainable and all("lora_" in name for name in trainable), trainable
    assert report["base_trainable_params"] == 0, report["base_trainable_params"]
    assert report["adapter_trainable_params"] == sum(
        param.numel() for name, param in peft_model.named_parameters() if "lora_" in name
    )
    assert report["adapter_trainable_params"] > 0
    assert report["n_quantized_parameters"] == len(doubles), report["n_quantized_parameters"]
    # the doubles were only ever asked to freeze
    for name, double in doubles:
        assert True not in double.journal, f"{name} was asked to unfreeze: {double.journal}"
    assert report["quantization"]["bnb_4bit_quant_type"] == "nf4"
    print(
        f"  configure_qlora -> base_trainable_params=0, "
        f"adapter_trainable_params={report['adapter_trainable_params']:,}, "
        f"n_quantized_parameters={report['n_quantized_parameters']}, "
        f"dtypes={report['quantized_dtypes']}, doubles only frozen"
    )


def test_configure_qlora_refuses_an_unquantized_model() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora unquantized-guard test: torch/peft not installed")
        return
    from lifemate.errors import LifeMateError
    from lifemate.training.lora_finetuning import configure_qlora

    try:
        configure_qlora(_tiny_llama(), None, _cfg())
    except LifeMateError as exc:
        assert "quantized" in str(exc), exc
        print(f"  configure_qlora -> 16-bit model refused: {str(exc)[:60]}...")
        return
    raise AssertionError("configure_qlora must refuse a 16-bit base")


def test_gradients_and_one_optimizer_step_move_the_adapter() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora gradient/step test: torch/peft not installed")
        return
    from lifemate.training.lora_finetuning import configure_qlora

    model = _frozen_quantization_config(_tiny_llama())
    _quantize_in_place(model)
    peft_model = configure_qlora(model, None, _cfg())["model"]

    adapter = [(name, param) for name, param in peft_model.named_parameters() if "lora_" in name]
    before = {name: param.detach().clone() for name, param in adapter}

    ids = _token_ids()
    out = peft_model(input_ids=ids, labels=ids)
    loss = out.loss
    assert loss is not None and torch.isfinite(loss), loss
    loss.backward()

    norms = {}
    for name, param in adapter:
        grad = param.grad
        assert grad is not None, f"adapter {name} got no gradient"
        assert torch.isfinite(grad).all(), f"adapter {name} gradient is not finite"
        norms[name] = float(grad.norm())
    assert any(value > 0 for value in norms.values()), "every adapter gradient is zero"
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            assert param.grad is None, f"frozen base parameter {name} got a gradient"

    optimizer = torch.optim.AdamW([p for _, p in adapter], lr=1e-2)
    optimizer.step()
    deltas = {name: float((param.detach() - before[name]).abs().max()) for name, param in adapter}
    assert max(deltas.values()) > 0.0, "one optimizer step did not change any adapter weight"
    print(f"  gradients -> {len(norms)} adapter grads finite, max norm "
          f"{max(norms.values()):.3e}; base grads all None; one step moved a weight by "
          f"{max(deltas.values()):.3e}")


def test_adapter_save_and_reload_is_bitwise_equal() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora save/reload test: torch/peft not installed")
        return
    from lifemate.training.lora_finetuning import configure_qlora, load_adapter, save_adapter

    model = _frozen_quantization_config(_tiny_llama())
    _quantize_in_place(model)
    peft_model = configure_qlora(model, None, _cfg())["model"]
    # make the adapter non-default so an equality check means something
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.add_(0.01)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_adapter(peft_model, Path(tmp) / "adapter")
        fresh = _frozen_quantization_config(_tiny_llama())
        _quantize_in_place(fresh)
        reloaded = load_adapter(fresh, path)

    original = {n: p.detach().clone() for n, p in peft_model.named_parameters() if "lora_" in n}
    again = {n: p.detach().clone() for n, p in reloaded.named_parameters() if "lora_" in n}
    assert original and set(original) == set(again), (len(original), len(again))
    worst = max(float((original[name] - again[name]).abs().max()) for name in original)
    assert worst == 0.0, f"adapter reload differs by {worst}"
    print(f"  adapter round-trip -> {len(original)} tensors, max abs diff {worst}")


# --------------------------------------------------------------------------- #
# negative control: the guard must exist
# --------------------------------------------------------------------------- #
def test_trainable_quantized_parameter_raises_with_its_name() -> None:
    if not HAVE_DEPS:
        print("  SKIP qlora quantized-guard test: torch/peft not installed")
        return
    from lifemate.errors import LifeMateError
    from lifemate.models.loader import apply_base_requires_grad, assert_no_quantized_trainable

    model = _frozen_quantization_config(_tiny_llama())
    doubles = _quantize_in_place(model)
    name = doubles[0][0]

    # (a) a 4-bit base can never be unfrozen, even for full fine-tuning
    try:
        apply_base_requires_grad(model, tuning_mode="full", training=True, quantization="none")
    except LifeMateError as exc:
        assert name in str(exc), str(exc)
        assert "requires_grad" in str(exc) or "quantized" in str(exc), str(exc)
    else:
        raise AssertionError("a quantized parameter must never become trainable")

    # (b) the post-condition catches it even when something else set the flag
    double = doubles[0][1]
    double.requires_grad = True
    try:
        assert_no_quantized_trainable(model)
    except LifeMateError as exc:
        assert name in str(exc), str(exc)
        print(f"  quantized guard -> raised naming {name}")
        return
    finally:
        double.requires_grad = False
    raise AssertionError("a trainable quantized parameter must raise (guard removed?)")


def test_qlora_load_freezes_everything_at_inference() -> None:
    """End-to-end: a lora checkpoint loaded for inference has zero trainable params."""
    if not HAVE_DEPS:
        print("  SKIP qlora inference-freeze test: torch/peft not installed")
        return
    from lifemate.models.loader import load_model_and_tokenizer
    from lifemate.training.lora_finetuning import configure_lora, save_adapter

    try:
        from tests.test_tiny_e2e import make_tiny_model_and_tokenizer
    except ImportError:  # stdlib runner imports this file as a bare module
        print("  SKIP qlora inference-freeze test: tiny fixture unimportable")
        return

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = make_tiny_model_and_tokenizer(Path(tmp) / "base")
        bundle = load_model_and_tokenizer(
            str(model_dir), device="cpu", torch_dtype="float32", tuning_mode="lora"
        )
        cfg = _cfg(targets=("q_proj", "v_proj"))
        adapter_dir = save_adapter(configure_lora(bundle.model, cfg)["model"], Path(tmp) / "adapter")

        inference = load_model_and_tokenizer(
            str(model_dir),
            device="cpu",
            torch_dtype="float32",
            tuning_mode="lora",
            checkpoint=adapter_dir,
            trainable=False,
        )
    assert inference.model_mode == "lora"
    trainable = [n for n, p in inference.model.named_parameters() if p.requires_grad]
    assert not trainable, f"inference left {len(trainable)} trainable parameter(s): {trainable[:3]}"
    assert inference.meta["adapter_trainable"] is False
    assert inference.meta["base_trainable"] is False
    assert inference.model.training is False
    print("  inference load -> 0 trainable parameters including adapters, model.eval()")


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            print(f"== {_name}")
            _fn()
    print("HAVE_DEPS:", HAVE_DEPS)
