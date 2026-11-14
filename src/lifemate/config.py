"""Dataclass configuration with YAML loading, dotted overrides, safe defaults.

Priority (low to high): dataclass defaults -> YAML file -> ``overrides`` args.

PyYAML is optional. When it is missing, a tiny YAML subset loader/dumper is
used that covers this repo's config files (nested mappings, scalars, inline
and block lists, ``#`` comments). ``load_config`` also accepts a JSON document
in a ``*.yaml`` file, because :func:`dump_config` falls back to JSON.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

__all__ = [
    "AppConfig",
    "DataConfig",
    "ModelConfig",
    "LoraConfig",
    "TrainingConfig",
    "AsrConfig",
    "AgentConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "WandbConfig",
    "CloudConfig",
    "load_config",
    "dump_config",
    "TUNING_MODES",
    "QUANTIZATIONS",
    "ASR_BACKENDS",
    "ASR_QUANTIZATIONS",
    "ASR_RUNTIMES",
    "ASR_DECODINGS",
]

TUNING_MODES = ("full", "lora", "qlora")
#: Quantization applied to the frozen base model (``nf4`` = QLoRA only).
QUANTIZATIONS = ("none", "nf4")
ASR_BACKENDS = ("gipformer", "whisper", "oracle", "none")
ASR_QUANTIZATIONS = ("int8", "fp32")
ASR_RUNTIMES = ("onnx", "pytorch")
ASR_DECODINGS = ("greedy_search", "modified_beam_search")


@dataclass
class DataConfig:
    """Where the dataset lives and how much of it to read."""

    root: str = "./data"
    train_split: str = "train"
    max_samples: int | None = None


@dataclass
class ModelConfig:
    """Base model / tokenizer identity and load options."""

    #: DEFAULT: MiniCPM5-2B. Qwen/Qwen3.5-4B stays a supported alternative
    #: (``--model Qwen/Qwen3.5-4B`` or configs/training_lora_qwen.yaml).
    name: str = "openbmb/MiniCPM5-2B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    device: str = "cuda"
    tokenizer_name: str | None = None


@dataclass
class LoraConfig:
    """LoRA hyper-parameters (used only when ``training.tuning_mode == 'lora'``)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"


@dataclass
class TrainingConfig:
    """SFT training loop settings."""

    method: str = "sft"
    tuning_mode: str = "lora"
    #: "none" | "nf4": 4-bit frozen base (QLoRA). Requires bitsandbytes on CUDA.
    quantization: str = "none"
    output_dir: str = "outputs/checkpoints"
    precision: str = "bfloat16"
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    num_train_epochs: float = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    gradient_clip_norm: float = 1.0
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    #: Step cadence for ``save_strategy: steps`` / ``eval_strategy: steps``.
    #: ``0`` falls back to ``logging_steps`` (the pre-existing behaviour), so a
    #: config that never sets them keeps working while the two cadences become
    #: independent knobs. Cadence is bookkeeping, not maths: excluded from
    #: ``config_fingerprint``.
    save_steps: int = 0
    eval_steps: int = 0
    seed: int = 42
    max_seq_length: int = 8192
    validation_ratio: float = 0.1
    optimizer: str = "adamw_torch"
    #: Cross-entropy chunk width over the sequence dimension (0 = the model's own
    #: single-shot loss). transformers upcasts the whole [seq, vocab] logits
    #: tensor to fp32, which is 3.98 GiB at seq 8192 x vocab 130560 (measured OOM
    #: on a 16 GB card); chunking caps that temporary at chunk*vocab*4 bytes.
    loss_chunk_tokens: int = 1024
    resume_from_checkpoint: str | None = None
    dataloader_num_workers: int = 0
    save_total_limit: int = 2
    report_to: list[str] = field(default_factory=list)


@dataclass
class AsrConfig:
    """Speech-to-text settings (default: Gipformer-68M-RNNT ONNX)."""

    backend: str = "gipformer"
    model_id: str = "g-group-ai-lab/gipformer-68M-rnnt"
    revision: str = "318322b009200f8e77a3751fa64638e5cc85a7dc"
    model_dir: str | None = None
    runtime: str = "onnx"
    device: str = "cpu"
    quantization: str = "int8"
    offline: bool = True
    num_threads: int = 4
    decoding_method: str = "modified_beam_search"
    model_name: str = "vinai/PhoWhisper-small"
    language: str = "vi"
    sample_rate: int = 16000
    max_audio_seconds: int = 60

@dataclass
class AgentConfig:
    """Closed-loop agent budget and decoding settings."""

    max_steps: int = 8
    time_budget_seconds: float = 120.0
    allow_ask_missing_info: bool = True
    temperature: float = 0.0
    max_new_tokens: int = 512
    stop_on_finish: bool = True


@dataclass
class InferenceConfig:
    """Prediction job settings."""

    model_mode: str = "base"
    checkpoint: str | None = None
    split: str = "public_test"
    output: str = "outputs/submission.jsonl"
    batch_size: int = 1
    use_mock_env: bool = False
    limit: int | None = None
    #: ASR transcripts of history user turns (JSONL). Required for a test split,
    #: whose history carries audio only; never auto-discovered inside data/.
    history_transcripts: str | None = None


@dataclass
class EvaluationConfig:
    """Offline scoring switches; all metric toggles are independent."""

    output_dir: str = "outputs/evaluation"
    teacher_forced: bool = True
    closed_loop: bool = True
    tool_call_exact_match: bool = True
    tool_name_accuracy: bool = True
    argument_accuracy: bool = True
    final_state_match: bool = True
    per_category_breakdown: bool = True


@dataclass
class WandbConfig:
    """Weights & Biases logging."""

    enabled: bool = False
    project: str = "lifemate"
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"
    log_model: bool = False
    upload_artifacts: bool = False


@dataclass
class CloudConfig:
    """Remote GPU job description (documented, not executed here)."""

    provider: str = "wandb_sandbox"
    project: str | None = None
    entity: str | None = None
    gpu_type: str | None = None
    checkpoint_remote: str | None = None
    artifact_name: str = "lifemate-full"


def _default_feature_config() -> Any:
    """``FeatureConfig()`` when the optional features slice is importable, else ``None``.

    Kept lazy so ``lifemate.config`` has no hard dependency on
    ``lifemate.features``: a checkout without that package keeps loading every
    config, with all experimental switches implicitly off.
    """
    try:  # pragma: no cover - depends on which slice the checkout carries
        from .features.config import FeatureConfig  # noqa: PLC0415

        return FeatureConfig()
    except ImportError:  # pragma: no cover - features slice absent
        return None

@dataclass
class AppConfig:
    """Root config object: ``data``, ``model``, ``lora``, ... sections."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    #: Optional experimental-feature switches (``src/lifemate/features``). Placed
    #: LAST so positional construction of the sections above cannot shift. The
    #: slice is optional: when the package is absent the field is ``None`` and
    #: every feature is implicitly off.
    features: Any = field(default_factory=_default_feature_config)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check cross-field invariants; raises :class:`ConfigError`."""
        if self.training.tuning_mode not in TUNING_MODES:
            raise ConfigError(
                f"training.tuning_mode must be one of {TUNING_MODES}, got {self.training.tuning_mode!r}"
            )
        if self.training.quantization not in QUANTIZATIONS:
            raise ConfigError(
                f"training.quantization must be one of {QUANTIZATIONS}, got {self.training.quantization!r}"
            )
        if self.training.tuning_mode == "qlora" and self.training.quantization == "none":
            raise ConfigError(
                "tuning_mode='qlora' requires training.quantization='nf4'; "
                "a qlora run that silently loads the base in 16-bit is not QLoRA"
            )
        if self.training.tuning_mode != "qlora" and self.training.quantization != "none":
            raise ConfigError(
                f"training.quantization={self.training.quantization!r} requires "
                "tuning_mode='qlora' (4-bit weights cannot be fully fine-tuned)"
            )
        if self.training.validation_ratio < 0 or self.training.validation_ratio >= 1:
            raise ConfigError("training.validation_ratio must be in [0, 1)")
        if self.asr.sample_rate <= 0:
            raise ConfigError("asr.sample_rate must be positive")
        if self.asr.backend not in ASR_BACKENDS:
            raise ConfigError(
                f"asr.backend must be one of {ASR_BACKENDS}, got {self.asr.backend!r}"
            )
        if self.asr.quantization not in ASR_QUANTIZATIONS:
            raise ConfigError(
                f"asr.quantization must be one of {ASR_QUANTIZATIONS}, got {self.asr.quantization!r}"
            )
        if self.asr.runtime not in ASR_RUNTIMES:
            raise ConfigError(
                f"asr.runtime must be one of {ASR_RUNTIMES}, got {self.asr.runtime!r}"
            )
        if self.asr.decoding_method not in ASR_DECODINGS:
            raise ConfigError(
                f"asr.decoding_method must be one of {ASR_DECODINGS}, got {self.asr.decoding_method!r}"
            )
        if self.asr.backend == "gipformer" and self.asr.runtime == "pytorch":
            raise ConfigError("pytorch runtime not implemented; use onnx")
        if self.asr.backend == "gipformer" and self.asr.model_id:
            # Spec-aware, in place of the old "1.5 is rejected because it is a
            # different model" substring rule. A KNOWN id is accepted here — which
            # artifact set it needs, and whether this machine has it, is a
            # load-time question answered by lifemate.asr.model_specs — so an
            # optional 1.5 checkpoint can be selected without a config edit. An
            # unknown gipformer id stays a config error listing the known ones.
            # Stdlib-only import: validate() must not need sherpa_onnx.
            from .asr.model_specs import KNOWN_MODEL_IDS, spec_key  # noqa: PLC0415

            if spec_key(self.asr.model_id) is None:
                raise ConfigError(
                    f"unknown gipformer model_id {self.asr.model_id!r}; known ids: "
                    f"{', '.join(KNOWN_MODEL_IDS)}. Add a GipformerSpec to "
                    "src/lifemate/asr/model_specs.py to support another checkpoint "
                    f"(the default is {KNOWN_MODEL_IDS[0]})"
                )
        if self.agent.max_steps < 1:
            raise ConfigError("agent.max_steps must be >= 1")
        self._validate_features()

    def _validate_features(self) -> None:
        """Apply the feature registry's own rules (dependencies, conflicts, values).

        Silent no-op when the optional slice is absent; a registry error becomes a
        :class:`ConfigError` so every entry point reports a config problem the same
        way.
        """
        if self.features is None:
            return
        try:  # pragma: no cover - depends on which slice the checkout carries
            from .features.registry import REGISTRY  # noqa: PLC0415
        except ImportError:  # pragma: no cover - features slice absent
            return
        validate = getattr(REGISTRY, "validate_or_raise", None)
        if not callable(validate):  # pragma: no cover - older registry
            return
        try:
            validate(self.features)
        except Exception as exc:
            raise ConfigError(f"features: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Plain nested ``dict`` copy of the config."""
        d = dataclasses.asdict(self)
        # ``dataclasses.asdict`` recurses into FeatureConfig and leaks its
        # bookkeeping ``provenance`` mapping; FeatureConfig.to_dict drops it.
        if hasattr(self.features, "to_dict"):
            d["features"] = self.features.to_dict()
        return d


# --------------------------------------------------------------------------- #
# YAML handling (PyYAML optional)
# --------------------------------------------------------------------------- #
def _yaml_module():
    try:  # pragma: no cover - depends on environment
        import yaml  # type: ignore

        return yaml
    except Exception:  # pragma: no cover
        return None


def _coerce_scalar(text: str) -> Any:
    """Best-effort YAML scalar parse for the fallback loader."""
    raw = text.strip()
    if raw == "" or raw in {"~", "null", "Null", "NULL"}:
        return None
    if raw in {"true", "True", "TRUE"}:
        return True
    if raw in {"false", "False", "FALSE"}:
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(part) for part in _split_inline(inner)]
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    unquoted = raw
    if len(unquoted) >= 2 and unquoted[0] == unquoted[-1] and unquoted[0] in {"'", '"'}:
        return unquoted[1:-1]
    try:
        return int(unquoted)
    except ValueError:
        pass
    try:
        return float(unquoted)
    except ValueError:
        pass
    return unquoted


def _split_inline(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    quote: str | None = None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p for p in (part.strip() for part in parts) if p != ""]


def _strip_comment(line: str) -> str:
    out: list[str] = []
    quote: str | None = None
    for idx, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
        elif ch == "#" and (idx == 0 or line[idx - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _child_indent(lines: list[tuple[int, str]], index: int, parent: int) -> int:
    """Indent of the block nested under ``parent`` (falls back for empty blocks)."""
    if index < len(lines) and lines[index][0] > parent:
        return lines[index][0]
    return parent + 1


def _parse_yaml_subset(text: str) -> Any:
    """Parse a nested-mapping YAML subset (indent based, no anchors/aliases)."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        is_list = lines[index][1].startswith("- ")
        container: Any = [] if is_list else {}
        while index < len(lines):
            cur_indent, content = lines[index]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                break
            if is_list and not content.startswith("- "):
                break
            if not is_list:
                if content.startswith("- "):
                    break
                if ":" not in content:
                    break
                key, _, rest = content.partition(":")
                key = key.strip().strip("'\"")
                rest = rest.strip()
                if rest:
                    container[key] = _coerce_scalar(rest)
                    index += 1
                else:
                    index += 1
                    child, index = parse_block(index, _child_indent(lines, index, cur_indent))
                    container[key] = child
            else:
                item_text = content[2:].strip()
                index += 1
                if not item_text:
                    child, index = parse_block(index, _child_indent(lines, index, cur_indent))
                    container.append(child)
                elif ":" in item_text and not item_text.startswith(("'", '"')):
                    key, _, rest = item_text.partition(":")
                    entry: dict[str, Any] = {}
                    if rest.strip():
                        entry[key.strip()] = _coerce_scalar(rest)
                    else:
                        child, index = parse_block(index, _child_indent(lines, index, cur_indent))
                        entry[key.strip()] = child
                    container.append(entry)
                else:
                    container.append(_coerce_scalar(item_text))
        return container, index

    value, _ = parse_block(0, lines[0][0])
    return value


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        yaml = _yaml_module()
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = _parse_yaml_subset(text)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return dict(data)


def _dump_yaml_subset(data: Any, indent: int = 0) -> str:
    pad = " " * indent
    out: list[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            if isinstance(value, (Mapping, list)):
                out.append(f"{pad}{key}:")
                out.append(_dump_yaml_subset(value, indent + 2).rstrip("\n"))
            else:
                out.append(f"{pad}{key}: {_scalar_text(value)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (Mapping, list)):
                nested = _dump_yaml_subset(item, indent + 2).rstrip("\n")
                out.append(f"{pad}-")
                out.append(nested)
            else:
                out.append(f"{pad}- {_scalar_text(item)}")
    else:
        out.append(f"{pad}{_scalar_text(data)}")
    return "\n".join(out) + "\n"


def _scalar_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#{}[],&*!|>'\"%@`") or text.strip() != text:
        return json.dumps(text)
    return text


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def _coerce_to_type(value: Any, current: Any, key: str) -> Any:
    """Coerce an override value to the type of the existing field value."""
    if value is None:
        return None
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            raise ConfigError(f"override {key}: cannot read {value!r} as bool")
        raise ConfigError(f"override {key}: cannot read {value!r} as bool")
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool):
            raise ConfigError(f"override {key}: expected int, got bool")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"override {key}: expected int, got {value!r}") from exc
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"override {key}: expected float, got {value!r}") from exc
    if isinstance(current, str):
        return str(value)
    if isinstance(current, list):
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ConfigError(f"override {key}: expected list, got {value!r}")
    # current is None (e.g. attn_implementation) -> infer from the value
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    raise ConfigError(f"override {key}: unsupported value {value!r}")


def _set_dotted(target: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    owner = target
    for part in parts[:-1]:
        if not hasattr(owner, part):
            raise ConfigError(f"unknown override section: {part!r} in {dotted!r}")
        owner = getattr(owner, part)
        if not dataclasses.is_dataclass(owner):
            raise ConfigError(f"override path not a section: {dotted!r}")
    leaf = parts[-1]
    if not hasattr(owner, leaf):
        raise ConfigError(f"unknown override key: {dotted!r}")
    current = getattr(owner, leaf)
    setattr(owner, leaf, _coerce_to_type(value, current, dotted))


def _apply_section(instance: Any, payload: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in payload.items():
        if not hasattr(instance, key):
            raise ConfigError(f"unknown config key: {prefix}{key}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            _apply_section(current, value, f"{prefix}{key}.")
        elif dataclasses.is_dataclass(current) and not isinstance(value, Mapping):
            raise ConfigError(f"config key {prefix}{key} expects a mapping")
        else:
            setattr(instance, key, _coerce_to_type(value, current, f"{prefix}{key}"))


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from defaults, an optional YAML file, overrides.

    No arguments and no file present yields a fully valid default config.
    """
    cfg = AppConfig()
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise ConfigError(f"config file not found: {cfg_path}")
        payload = _load_document(cfg_path)
        _apply_section(cfg, payload)
    for dotted, value in (overrides or {}).items():
        _set_dotted(cfg, dotted, value)
    cfg.validate()
    return cfg


def dump_config(cfg: AppConfig, path: str | Path) -> Path:
    """Write ``cfg`` to ``path``; YAML when PyYAML exists, else JSON text."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.to_dict()
    yaml = _yaml_module()
    if yaml is not None:
        out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:  # pragma: no cover - exercised only without PyYAML
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
