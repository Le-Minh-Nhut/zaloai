"""Model loading / checkpoint tests (stdlib only, no torch required).

Run with: python -m pytest tests/test_model_loading.py -q

Torch-dependent checks print ``SKIP: <reason>`` and return instead of failing,
so a CPU-only box still exercises everything that can be exercised. No network
or weight download is ever attempted here.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lifemate.errors import LifeMateError  # noqa: E402
from lifemate.models import checkpoint as ckpt  # noqa: E402
from lifemate.models import loader  # noqa: E402

NEEDS_GPU: list[str] = []

try:  # pragma: no cover - environment dependent
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# importability without torch
# --------------------------------------------------------------------------- #
def test_modules_import_without_torch_at_top_level():
    for module in ("lifemate.agent.prompt_builder", "lifemate.agent.agent", "lifemate.models.loader",
                   "lifemate.models.generation", "lifemate.models.checkpoint"):
        __import__(module)
    for module_name in ("lifemate.agent.agent", "lifemate.agent.prompt_builder", "lifemate.models.loader",
                        "lifemate.models.generation", "lifemate.models.checkpoint"):
        assert _module_level_heavy_imports(module_name) == [], (
            f"{module_name} imports torch/transformers/peft/wandb at module top level"
        )


def test_count_parameters_on_tiny_module():
    if not HAVE_TORCH:
        print("SKIP: torch not installed - cannot build a nn.Module")
        NEEDS_GPU.append("count_parameters")
        return
    import torch

    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for param in module[1].parameters():
        param.requires_grad_(False)
    counts = loader.count_parameters(module)
    assert counts["total"] == 4 * 3 + 3 + 3 * 2 + 2
    assert counts["trainable"] == 4 * 3 + 3
    assert counts["frozen"] == 3 * 2 + 2


def test_count_parameters_without_parameters_method():
    assert loader.count_parameters(object()) == {"total": 0, "trainable": 0, "frozen": 0}


# --------------------------------------------------------------------------- #
# offline / local_files_only logic (no network)
# --------------------------------------------------------------------------- #
def test_is_local_or_offline():
    with tempfile.TemporaryDirectory() as tmp:
        assert loader.is_local_or_offline(tmp) is True, "existing path must stay offline"
    assert loader.is_local_or_offline("Qwen/Qwen3.5-4B-Instruct", env={}) is False
    assert loader.is_local_or_offline("Qwen/Qwen3.5-4B-Instruct", env={"HF_HUB_OFFLINE": "1"}) is True
    assert loader.is_local_or_offline("Qwen/Qwen3.5-4B-Instruct", env={"TRANSFORMERS_OFFLINE": "true"}) is True
    assert loader.is_local_or_offline("Qwen/Qwen3.5-4B-Instruct", env={"HF_HUB_OFFLINE": "0"}) is False
    assert loader.is_local_or_offline("") is False, "empty id must not resolve to cwd"


def test_resolve_model_mode():
    assert loader.resolve_model_mode("full", None) == "base"
    assert loader.resolve_model_mode("full", "outputs/ckpt-100") == "full"
    assert loader.resolve_model_mode("lora", "outputs/ckpt-100") == "lora"
    # qlora is a supported tuning mode: its checkpoints are ordinary LoRA
    # adapters, so the resolved mode is "lora".
    assert loader.resolve_model_mode("qlora", "outputs/ckpt-100") == "lora"
    assert loader.resolve_model_mode("qlora", None) == "base"
    for bad in ("adapter", ""):
        try:
            loader.resolve_model_mode(bad, "x")
        except LifeMateError as exc:
            assert "tuning_mode" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"tuning_mode {bad!r} should be rejected")


def test_train_mode_touches_only_mode():
    """``train_mode`` flips train/eval and never touches requires_grad."""

    class _Recorder:
        def __init__(self):
            self.training = None
            self.requires_grad_calls = []

        def train(self):
            self.training = True
            return self

        def eval(self):
            self.training = False
            return self

        def parameters(self):
            raise AssertionError("train_mode must not iterate parameters")

        def requires_grad_(self):
            raise AssertionError("train_mode must not touch requires_grad")

    model = _Recorder()
    assert loader.train_mode(model, training=True) is model and model.training is True
    assert loader.train_mode(model, training=False) is model and model.training is False

def test_base_requires_grad_is_full_tuning_only_while_training():
    """Full → True; lora/qlora/inference → False, regardless of the model."""
    model = object()
    assert loader.base_requires_grad(model, tuning_mode="full", training=True) is True
    assert loader.base_requires_grad(model, tuning_mode=None, training=True) is True
    for mode in ("lora", "qlora", "LoRA"):
        assert loader.base_requires_grad(model, tuning_mode=mode, training=True) is False, mode
    for mode in ("full", "lora", "qlora"):
        assert loader.base_requires_grad(model, tuning_mode=mode, training=False) is False, mode

def test_adapter_requires_grad_only_while_training():
    assert loader.adapter_requires_grad(object(), training=True) is True
    assert loader.adapter_requires_grad(object(), training=False) is False

def test_is_quantized_parameter_detects_integer_and_bnb_classes():
    """Quantized = integer dtype OR a bitsandbytes parameter class name."""

    class _Dtype:
        """Mirrors ``torch.dtype``: ``is_floating_point`` is a plain bool."""

        def __init__(self, is_floating_point):
            self.is_floating_point = is_floating_point

    class _Param:
        def __init__(self, dtype, is_floating_point=None):
            self.dtype = dtype
            if is_floating_point is not None:
                self.is_floating_point = lambda: is_floating_point

    class Params4bit:  # class name is what the loader matches
        dtype = _Dtype(False)

    uint8, bf16 = _Dtype(False), _Dtype(True)
    assert loader.is_quantized_parameter(_Param(uint8)) is True
    assert loader.is_quantized_parameter(_Param(bf16)) is False
    assert loader.is_quantized_parameter(_Param(uint8, is_floating_point=False)) is True
    assert loader.is_quantized_parameter(_Param(bf16, is_floating_point=True)) is False
    assert loader.is_quantized_parameter(Params4bit()) is True
    assert loader.is_quantized_parameter(object()) is False

def test_assert_no_quantized_trainable_names_the_parameter():
    """A trainable integer parameter raises and names it; frozen ones do not."""

    class _Q:
        dtype = None

        def __init__(self, n):
            self.n = n
            self.requires_grad = False

        def numel(self):
            return self.n

        def is_floating_point(self):
            return False

    class _Model:
        def __init__(self, params):
            self._params = params

        def named_parameters(self):
            return list(self._params)

        def parameters(self):
            return [p for _, p in self._params]

    frozen = _Q(8)
    loader.assert_no_quantized_trainable(_Model([("layers.0.q_proj.weight", frozen)]))
    loader.assert_no_quantized_trainable(_Model([("layers.0.q_proj.weight", frozen)]))
    assert loader.quantized_parameters(_Model([("a", frozen)]))[0][0] == "a"

    offender = _Q(8)
    offender.requires_grad = True
    try:
        loader.assert_no_quantized_trainable(_Model([("layers.3.v_proj.weight", offender)]))
    except LifeMateError as exc:
        assert "layers.3.v_proj.weight" in str(exc), str(exc)
        assert "requires_grad=True" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a trainable quantized parameter must raise")
    print("  quantized guard -> refuses layers.3.v_proj.weight, accepts the frozen base")

def test_adapter_parameter_name_detection():
    assert loader.is_adapter_parameter_name("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight")
    assert loader.is_adapter_parameter_name("base_model.modules_to_save.default.weight")
    assert not loader.is_adapter_parameter_name("base_model.model.layers.0.self_attn.q_proj.base_layer.weight")

def test_resolve_dtype_maps_strings():
    if not HAVE_TORCH:
        print("SKIP: torch not installed - torch.dtype mapping unavailable")
        NEEDS_GPU.append("resolve_dtype")
        return
    import torch

    assert loader.resolve_dtype("bf16") is torch.bfloat16
    assert loader.resolve_dtype("float16") is torch.float16
    assert loader.resolve_dtype(torch.float32) is torch.float32
    try:
        loader.resolve_dtype("int7")
    except LifeMateError as exc:
        assert "torch_dtype" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown dtype must raise LifeMateError")


def test_bf16_note_on_cpu():
    dtype, note = loader.bf16_note("bfloat16", "cpu")
    assert dtype == "bfloat16" and note is None
    if HAVE_TORCH and not torch.cuda.is_available():
        dtype, note = loader.bf16_note("bfloat16", "cuda")
        assert dtype == "float16" and note and "CUDA unavailable" in note
    else:
        print("SKIP: CUDA state unknown - bf16 fallback note not exercised")


def test_no_download_attempt_when_path_local():
    """A nonexistent local path must fail loudly, not hit the network."""
    if not HAVE_TORCH or not _have("transformers"):
        print("SKIP: torch/transformers absent - real loader path not exercised")
        NEEDS_GPU.append("load_model_and_tokenizer")
        return
    with tempfile.TemporaryDirectory() as tmp:
        missing = pathlib.Path(tmp) / "nope-model"
        try:
            loader.load_model_and_tokenizer(str(missing), device="cpu", torch_dtype="float32")
        except LifeMateError as exc:
            assert "local_files_only=True" in str(exc) or str(missing) in str(exc)
        else:  # pragma: no cover
            raise AssertionError("missing local model dir must raise LifeMateError")


# --------------------------------------------------------------------------- #
# checkpoints (stdlib)
# --------------------------------------------------------------------------- #
def test_versioned_dir_never_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp) / "ckpt"
        base.mkdir()
        assert ckpt.versioned_dir(base) == base, "empty dir is reusable"
        (base / "model.safetensors").write_text("x")
        assert ckpt.versioned_dir(base) == base.with_name("ckpt-v2")
        (base.with_name("ckpt-v2")).mkdir()
        (base.with_name("ckpt-v2") / "model.safetensors").write_text("x")
        assert ckpt.versioned_dir(base) == base.with_name("ckpt-v3"), "occupied -v2 must be skipped"
        assert ckpt.versioned_dir(base, force=True) == base


def test_save_checkpoint_and_meta_roundtrip():
    class _Tok:
        def save_pretrained(self, path):
            pathlib.Path(path, "tokenizer_config.json").write_text("{}")

    class _Model:
        def save_pretrained(self, path):
            pathlib.Path(path, "config.json").write_text('{"model_type": "qwen"}')

    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "ckpt"
        saved = ckpt.save_checkpoint(
            _Model(),
            _Tok(),
            target,
            meta={"model_name": "Qwen/Qwen3.5-4B-Instruct", "config": {"lr": 1e-4}, "seed": 42},
            is_lora=False,
            step=750,
        )
        assert saved == target
        meta = ckpt.load_checkpoint_meta(saved)
        assert meta["model_name"] == "Qwen/Qwen3.5-4B-Instruct"
        assert meta["step"] == 750 and meta["is_lora"] is False
        assert meta["config"] == {"lr": 1e-4} and meta["timestamp"]
        assert (saved / ckpt.META_FILENAME).exists()

        # Force must still never destroy data silently: the old tree is moved aside.
        forced = ckpt.save_checkpoint(
            _Model(), _Tok(), target, meta={"model_name": "forced"}, is_lora=False, force=True
        )
        assert forced == target
        assert ckpt.load_checkpoint_meta(target)["model_name"] == "forced"
        backups = sorted(p.name for p in pathlib.Path(tmp).glob("ckpt-replaced*"))
        assert backups == ["ckpt-replaced"], backups
        assert ckpt.load_checkpoint_meta(pathlib.Path(tmp) / backups[0])["step"] == 750

        # Second save must not clobber the first.
        again = ckpt.save_checkpoint(_Model(), _Tok(), target, meta={"model_name": "x"}, is_lora=True, step=1)
        assert again == target.with_name("ckpt-v2")
        assert ckpt.load_checkpoint_meta(target)["model_name"] == "forced"
        assert ckpt.load_checkpoint_meta(again)["step"] == 1
        assert ckpt.load_checkpoint_meta(again)["is_lora"] is True


def test_load_checkpoint_meta_missing_and_corrupt():
    with tempfile.TemporaryDirectory() as tmp:
        assert ckpt.load_checkpoint_meta(tmp) == {}
        bad = pathlib.Path(tmp, "lifemate_meta.json")
        bad.write_text("{not json")
        try:
            ckpt.load_checkpoint_meta(tmp)
        except LifeMateError as exc:
            assert "not valid JSON" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("corrupt metadata must raise LifeMateError")


def test_resolve_checkpoint_kinds():
    with tempfile.TemporaryDirectory() as tmp:
        full = pathlib.Path(tmp) / "checkpoint-500"
        full.mkdir()
        (full / "config.json").write_text("{}")
        (full / "model.safetensors").write_text("w")
        info = ckpt.resolve_checkpoint(full)
        assert info.kind == "full" and info.step == 500

        lora = pathlib.Path(tmp) / "lora-300"
        lora.mkdir()
        (lora / "adapter_config.json").write_text("{}")
        (lora / "adapter_model.safetensors").write_text("w")
        info = ckpt.resolve_checkpoint(lora)
        assert info.kind == "lora" and info.step == 300

        empty = pathlib.Path(tmp) / "empty"
        empty.mkdir()
        try:
            ckpt.resolve_checkpoint(empty)
        except LifeMateError as exc:
            assert "cannot classify" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("unclassifiable dir must raise LifeMateError")

        try:
            ckpt.resolve_checkpoint(pathlib.Path(tmp) / "ghost")
        except LifeMateError as exc:
            assert "does not exist" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("missing dir must raise LifeMateError")


def test_resolve_checkpoint_uses_meta_step():
    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "final"
        target.mkdir()
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_text("w")
        (target / ckpt.META_FILENAME).write_text(json.dumps({"step": 1234, "model_mode": "full"}))
        info = ckpt.resolve_checkpoint(target)
        assert info.step == 1234 and info.meta["model_mode"] == "full"
        assert json.loads(json.dumps(info.to_dict()))["kind"] == "full"


def test_lora_plumbing_requires_peft():
    if _have("peft"):
        print("SKIP: peft installed - missing-peft error path not exercised")
        return
    with tempfile.TemporaryDirectory() as tmp:
        adapter = pathlib.Path(tmp) / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        try:
            loader.load_model_and_tokenizer(
                "Qwen/Qwen3.5-4B-vanilla",
                tuning_mode="lora",
                checkpoint=adapter,
                device="cpu",
            )
        except LifeMateError as exc:
            message = str(exc)
            # Without torch we fail earlier with an actionable install hint.
            assert "peft" in message or "torch" in message, message
        else:  # pragma: no cover
            raise AssertionError("lora load without peft/torch must raise LifeMateError")

        # merge_lora_adapter shares the same requirement.
        try:
            ckpt.merge_lora_adapter("Qwen/Qwen3.5-4B-vanilla", adapter, pathlib.Path(tmp) / "merged")
        except LifeMateError as exc:
            assert "peft" in str(exc) or "transformers" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("merge without peft must raise LifeMateError")


def test_incomplete_checkpoint_raises_instead_of_random_init():
    """A checkpoint missing weight files must not load as partly random weights.

    ``from_pretrained`` only prints a MISSING table, so a truncated download (or
    a ``config.json`` claiming more layers than the shards hold) silently leaves
    those parameters at their random initialisation. Measured on a 3-layer config
    over 2-layer weights: layer 2 was initialised at std 0.0198 and the run
    started anyway. The loader now raises.
    """
    if not HAVE_TORCH or not _have("transformers"):
        print("SKIP: torch/transformers absent - incomplete-checkpoint guard not exercised")
        return
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    text = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, tie_word_embeddings=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "partial"
        torch.manual_seed(0)
        Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], **text)
        ).save_pretrained(target)
        # The config now describes one more layer than the weight file contains.
        config = json.loads((target / "config.json").read_text(encoding="utf-8"))
        config["num_hidden_layers"] = 3
        config["layer_types"] = ["linear_attention", "full_attention", "linear_attention"]
        (target / "config.json").write_text(json.dumps(config), encoding="utf-8")

        try:
            loader.load_model_and_tokenizer(
                str(target), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True
            )
        except LifeMateError as exc:
            message = str(exc)
            assert "incomplete" in message, message
            assert "layers.2" in message, message
            print(f"  incomplete checkpoint -> refused: {message[:80]}...")
        else:  # pragma: no cover
            raise AssertionError("an incomplete checkpoint must not load (silent random init)")

        # Tied weights must not be mistaken for missing ones: a clean,
        # tie_word_embeddings checkpoint (no lm_head in the shard) still loads.
        clean = pathlib.Path(tmp) / "clean"
        Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], **text)
        ).save_pretrained(clean)
        bundle = loader.load_model_and_tokenizer(
            str(clean), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True
        )
        assert bundle.meta["n_parameters"]["total"] > 0
        print("  tied lm_head -> still loads (not reported as missing)")

def test_loader_decisions_and_meta_keys():
    """The three decisions are visible in ``meta`` and match ``requires_grad``.

    The QLoRA-training case that used to hit the unfreeze branch: ``checkpoint``
    is None, so ``resolve_model_mode`` returns ``"base"`` — the tuning mode, not
    the resolved mode, is what must decide.
    """
    if not HAVE_TORCH or not _have("transformers"):
        print("SKIP: torch/transformers absent - loader decisions not exercised")
        NEEDS_GPU.append("load_model_and_tokenizer")
        return
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tiny = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = pathlib.Path(tmp) / "tiny"
        tiny.save_pretrained(model_dir)
        from tests.test_tiny_e2e import make_tiny_model_and_tokenizer

        make_tiny_model_and_tokenizer(model_dir)  # adds tokenizer artifacts

        def load(**kwargs):
            return loader.load_model_and_tokenizer(
                str(model_dir), device="cpu", torch_dtype="float32", **kwargs
            )

        trained = load(tuning_mode="full", for_training=True)
        assert trained.meta["base_trainable"] is True
        assert trained.meta["adapter_trainable"] is False
        assert trained.meta["n_quantized_parameters"] == 0
        assert trained.meta["n_parameters"]["frozen"] == 0, trained.meta["n_parameters"]
        assert trained.model.training is True

        # lora *training* with no checkpoint: resolved mode "base", and the base
        # must stay frozen because the adapter is what trains.
        lora = load(tuning_mode="lora", trainable=True, for_training=True)
        assert lora.model_mode == "base", lora.model_mode
        assert lora.meta["base_trainable"] is False
        assert lora.meta["adapter_trainable"] is False, "no adapter was attached"
        assert lora.meta["n_parameters"]["trainable"] == 0, lora.meta["n_parameters"]

        inference = load(tuning_mode="full", for_training=False)
        assert inference.meta["base_trainable"] is False
        assert inference.meta["n_parameters"]["trainable"] == 0
        assert inference.model.training is False
        assert all(not p.requires_grad for p in inference.model.parameters())
    print("  loader decisions -> full/base=True, lora-base=False, inference all frozen; "
          "meta keys base_trainable/adapter_trainable/n_quantized_parameters present")


def test_qwen3_5_loads_through_the_plain_auto_path():
    """Phase 10: the released multimodal ``qwen3_5`` layout needs no override.

    Pins the introspection this repository relies on: transformers maps
    ``qwen3_5`` → ``Qwen3_5ForCausalLM`` for causal LM, so the loader keeps ONE
    interface. The fixture is a real saved model whose shard is rewritten into
    the released layout (``model.language_model.*`` keys, an extra
    ``model.visual.*`` tensor, no ``lm_head.weight``).
    """
    if not HAVE_TORCH or not _have("transformers"):
        print("SKIP: torch/transformers absent - qwen3_5 layout not exercised")
        return
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    if "qwen3_5" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        print(f"BLOCKED: this transformers build has no qwen3_5 causal-LM mapping "
              f"({_transformers_version()}); Qwen3.5-4B loading stays BLOCKED")
        return
    assert MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["qwen3_5"] == "Qwen3_5ForCausalLM"
    assert CONFIG_MAPPING_NAMES["qwen3_5"] == "Qwen3_5Config"

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5TextConfig

    text = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_attention_heads=2,
        num_key_value_heads=2, num_hidden_layers=1, max_position_embeddings=64,
        layer_types=["full_attention"], linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    top = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(**text), architectures=["Qwen3_5ForConditionalGeneration"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "released"
        target.mkdir()
        reference = pathlib.Path(tmp) / "reference"
        AutoModelForCausalLM.from_config(top).save_pretrained(reference)
        top.save_pretrained(target)
        shard = load_file(reference / "model.safetensors")
        released = {
            key.replace("model.", "model.language_model.", 1): value
            for key, value in shard.items()
            if not key.startswith("lm_head")
        }
        released["model.visual.blocks.0.weight"] = torch.zeros(4, 4)
        save_file(released, target / "model.safetensors")
        from tests.test_tiny_e2e import make_tiny_model_and_tokenizer

        tokenizer_dir = make_tiny_model_and_tokenizer(pathlib.Path(tmp) / "tok")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            source = tokenizer_dir / name
            if source.exists():
                (target / name).write_text(source.read_text(encoding="utf-8"))

        bundle = loader.load_model_and_tokenizer(
            str(target), device="cpu", torch_dtype="float32", tuning_mode="full", for_training=True
        )
    assert type(bundle.model).__name__ == "Qwen3_5ForCausalLM", type(bundle.model).__name__
    assert bundle.meta["base_trainable"] is True
    assert bundle.meta["n_quantized_parameters"] == 0
    logits = bundle.model(input_ids=torch.randint(0, 64, (1, 4))).logits
    assert tuple(logits.shape) == (1, 4, 64), logits.shape
    print("  qwen3_5 -> MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3_5'] == Qwen3_5ForCausalLM; "
          "released layout (model.language_model.*, model.visual.*) loads on the plain Auto path")


def _transformers_version():
    import transformers

    return transformers.__version__


def test_generation_module_imports_and_truncates():
    from lifemate.models import generation

    text, hit = generation.truncate_at_marker('abc<action>{"type": "finish"}</action>tail')
    assert hit is True and text.endswith("</action>") and "tail" not in text
    text, hit = generation.truncate_at_marker("no marker here")
    assert hit is False

    class _Tok:
        pad_token_id = None
        eos_token_id = 1
        eos_token = "</s>"

        def __init__(self):
            self.pad_token = None

    # HFGenerator must reject a tokenizer without a chat template, loudly.
    try:
        generation.HFGenerator(model=object(), tokenizer=_Tok())
    except LifeMateError as exc:
        assert "apply_chat_template" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tokenizer without chat template must raise LifeMateError")


def test_callable_generator_captures_messages():
    from lifemate.models.generation import CallableGenerator

    generator = CallableGenerator(lambda messages: "<action>{\"type\": \"finish\"}</action>")
    out = generator.generate([{"role": "user", "content": "hi"}])
    assert out.endswith("</action>")
    assert generator.calls[0][0]["content"] == "hi"
    assert generator.stats.n_calls == 1


def test_needs_gpu_report():
    if NEEDS_GPU:
        print(f"NEEDS_GPU: {sorted(set(NEEDS_GPU))} (torch/transformers/peft absent; nothing fabricated)")
    assert all(isinstance(item, str) for item in NEEDS_GPU)
    if NEEDS_GPU:
        missing = [name for name in ("torch", "transformers", "peft") if not _have(name)]
        assert missing, f"NEEDS_GPU recorded but all heavy deps are installed: {NEEDS_GPU}"


def _module_level_heavy_imports(module_name: str) -> list[str]:
    """AST walk: heavy imports that are NOT inside a function/class body."""
    import ast
    import importlib

    module = importlib.import_module(module_name)
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    heavy = {"torch", "transformers", "peft", "wandb"}
    found: list[str] = []
    for node in tree.body:  # module level only; nested defs are lazy on purpose
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        found.extend(n for n in names if n in heavy)
    return found


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("NEEDS_GPU:", sorted(set(NEEDS_GPU)) or "none")
