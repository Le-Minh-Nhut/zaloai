"""The single authoritative feature-flag object.

Every optional LifeMate technique is a *flag plus its parameters* held here.
Nothing in this package enables a technique: the flags are the switchboard, the
owning workstream reads them.

Rules this module enforces by construction
------------------------------------------
* every flag defaults to ``False`` and every default is written out explicitly;
* a numeric/string parameter is **inert** unless its owning flag is ``True`` —
  ``nbest_size: 5`` with ``nbest_enabled: False`` changes nothing;
* the object is a plain nested dataclass, so the existing dot-path override
  engine in :mod:`lifemate.config` (``_set_dotted``) can set
  ``features.asr.nbest_enabled``; ``AppConfig.features`` exists and every entry
  point resolves it through :mod:`lifemate.features.overrides`. No
  ``frozen=True``: an override must be settable.

Fingerprints
------------
:func:`feature_fingerprint` hashes the *effective* state, so two runs can be
compared without diffing YAML. :func:`training_feature_fingerprint` hashes the
subset that can change the supervised data (specs whose ``scope`` is
``training`` or ``both``), for checkpoint/Colab provenance.

Deliberate exclusions, identical in both fingerprints (see ``FINGERPRINT_EXCLUDED``):

* ``audio.vad_min_silence_ms`` and ``audio.vad_padding_ms`` — run-local tuning of
  the same VAD model; moving them changes latency and edge trimming, not the
  technique set. Including them would make every latency tweak look like a new
  experiment.
* ``audio.rms_target_dbfs`` — a gain target; it changes loudness, not content.
* ``training.augmentation_seed`` — an RNG seed: a different draw from the same
  distribution, not a different distribution.

Included on purpose: ``asr.nbest_size``, ``asr.multi_asr_*``,
``training.noisy_asr_ratio``, ``training.tool_dropout_max_tools``,
``decoding.constrained_backend``, ``postprocessing.guardrail_action`` and
``asr.secondary_model_id`` — each one changes the sampled output or the data.
Parameters of a *disabled* feature are excluded as well: they are inert by
definition, so editing them must not move the fingerprint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..errors import FeatureConfigurationError

__all__ = [
    "AudioFeatureConfig",
    "AsrFeatureConfig",
    "PromptFeatureConfig",
    "TrainingFeatureConfig",
    "DecodingFeatureConfig",
    "PostProcessingFeatureConfig",
    "FeatureConfig",
    "FINGERPRINT_EXCLUDED",
    "TRAINING_SCOPES",
    "feature_fingerprint",
    "training_feature_fingerprint",
    "enabled_features",
]

#: Config paths excluded from both fingerprints (documented in the module docstring).
FINGERPRINT_EXCLUDED = frozenset(
    {
        "audio.vad_min_silence_ms",
        "audio.vad_padding_ms",
        "audio.rms_target_dbfs",
        "training.augmentation_seed",
    }
)

#: ``scope`` values whose specs feed :func:`training_feature_fingerprint`.
TRAINING_SCOPES = frozenset({"training", "both"})

_TRUE_TEXT = frozenset({"true", "1", "yes", "on"})
_FALSE_TEXT = frozenset({"false", "0", "no", "off"})

@dataclass
class AudioFeatureConfig:
    """Front-end audio knobs (VAD, level normalisation)."""

    vad_enabled: bool = False
    vad_min_silence_ms: int = 300
    vad_padding_ms: int = 200
    rms_normalization_enabled: bool = False
    rms_target_dbfs: float = -23.0

@dataclass
class AsrFeatureConfig:
    """ASR-side switches: model version, N-best, secondary ASR."""

    gipformer_v15_enabled: bool = False
    nbest_enabled: bool = False
    nbest_size: int = 3
    multi_asr_enabled: bool = False
    secondary_model_id: str = ""
    multi_asr_policy: str = "primary_topk_secondary_top1"
    multi_asr_max_hypotheses: int = 3
    multi_asr_failure_policy: str = "primary_only"

@dataclass
class PromptFeatureConfig:
    """Prompt construction switches: ITN, N-best-aware formatting."""

    itn_enabled: bool = False
    nbest_prompt_enabled: bool = False

@dataclass
class TrainingFeatureConfig:
    """Training switches that change the supervised data."""

    noisy_asr_training_enabled: bool = False
    noisy_asr_ratio: float = 0.25
    nbest_training_format_enabled: bool = False
    slot_masking_enabled: bool = False
    tool_dropout_enabled: bool = False
    tool_dropout_max_tools: int = 1
    augmentation_seed: int = 42

@dataclass
class DecodingFeatureConfig:
    """Constrained-decoding switches."""

    constrained_decoding_enabled: bool = False
    constrained_backend: str = "auto"

@dataclass
class PostProcessingFeatureConfig:
    """Output post-processing switches."""

    enhanced_schema_guardrail_enabled: bool = False
    guardrail_action: str = "reject"

def _section_fields(obj: object) -> dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(obj)}

def _coerce(value: Any, current: Any, key: str) -> Any:
    """Read ``value`` as the type of ``current`` (YAML/env/CLI text welcome)."""
    if current is None:
        return value
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        raise FeatureConfigurationError(
            f"{key}: cannot read {value!r} as bool; use true/false/1/0/yes/no"
        )
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool):
            raise FeatureConfigurationError(f"{key}: expected int, got bool {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise FeatureConfigurationError(f"{key}: expected int, got {value!r}") from exc
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise FeatureConfigurationError(f"{key}: expected float, got {value!r}") from exc
    return str(value)

def _walk(cfg: object, path: Iterable[str]) -> Any:
    target: Any = cfg
    for part in path:
        target = getattr(target, part)
    return target

def apply_mapping(cfg: "FeatureConfig", mapping: Mapping[str, Any] | None) -> list[str]:
    """Copy ``mapping`` (a ``features:`` YAML section) onto ``cfg``.

    Returns the unknown keys it skipped — ``[]`` when everything matched. Values
    are coerced to the field's existing type, so ``"true"``/``1`` both work.
    """
    unknown: list[str] = []
    for section, payload in (mapping or {}).items():
        target = getattr(cfg, str(section), None)
        if target is None or not dataclasses.is_dataclass(target):
            unknown.append(str(section))
            continue
        if not isinstance(payload, Mapping):
            unknown.append(str(section))
            continue
        known = _section_fields(target)
        for key, value in payload.items():
            name = str(key)
            if name not in known:
                unknown.append(f"{section}.{name}")
                continue
            current = getattr(target, name)
            setattr(target, name, _coerce(value, current, f"features.{section}.{name}"))
    return unknown

def _as_plain(obj: Any) -> Any:
    """Nested plain copy, skipping non-init bookkeeping fields (``provenance``)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _as_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init
        }
    return obj

@dataclass
class FeatureConfig:
    """Root feature object. Reachable as ``AppConfig.features`` after wiring."""

    audio: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)
    asr: AsrFeatureConfig = field(default_factory=AsrFeatureConfig)
    prompt: PromptFeatureConfig = field(default_factory=PromptFeatureConfig)
    training: TrainingFeatureConfig = field(default_factory=TrainingFeatureConfig)
    decoding: DecodingFeatureConfig = field(default_factory=DecodingFeatureConfig)
    postprocessing: PostProcessingFeatureConfig = field(
        default_factory=PostProcessingFeatureConfig
    )

    #: ``feature id -> "cli"|"env"|"yaml"|"default"``. Deliberately NOT a
    #: dataclass field: provenance is bookkeeping, never part of the feature
    #: state, the fingerprints, or a ``dump_config`` round trip.
    provenance: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested ``dict`` (``provenance`` excluded)."""
        return _as_plain(self)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any] | None) -> tuple["FeatureConfig", list[str]]:
        """Build a config from a ``features:``-shaped mapping.

        Tolerant on purpose: unknown sections/keys are *returned*, never dropped
        silently — ``(cfg, unknown_keys)``. Missing sections/``None`` are fine
        (defaults are used). Callers decide whether an unknown key is an error
        (:func:`lifemate.features.overrides.apply_overrides` does raise).
        """
        cfg = cls()
        return cfg, apply_mapping(cfg, mapping)

def _state(
    cfg: FeatureConfig,
    *,
    scopes: frozenset[str] | None = None,
    exclude: frozenset[str] = FINGERPRINT_EXCLUDED,
) -> dict[str, Any]:
    """Effective feature state, one entry per feature id, disabled params dropped."""
    from .registry import FEATURES  # local import: registry imports this module

    state: dict[str, Any] = {}
    for spec in FEATURES:
        if scopes is not None and spec.scope not in scopes:
            continue
        entry: dict[str, Any] = {"enabled": bool(_walk(cfg, spec.config_path))}
        if entry["enabled"]:
            for name in sorted(spec.params):
                key = f"{spec.category}.{name}"
                if key in exclude:
                    continue
                entry[name] = _walk(cfg, (spec.category, name))
        state[spec.id] = entry
    return state

def _digest(state: Mapping[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def feature_fingerprint(cfg: FeatureConfig) -> str:
    """16 hex chars over the sorted effective feature state.

    Excludes :data:`FINGERPRINT_EXCLUDED` (documented in the module docstring)
    and any parameter of a disabled feature (inert by definition).
    """
    return _digest(_state(cfg))

def training_feature_fingerprint(cfg: FeatureConfig) -> str:
    """16 hex chars over the subset that can change the supervised data.

    Derived from each spec's ``scope``: only ``training`` and ``both`` specs are
    hashed, so inference-only ids (``asr.gipformer_v15``, ``asr.multi_asr``,
    ``decoding.constrained``, ``postprocessing.enhanced_schema_guardrail``)
    never move it, while ``training.*``, ``prompt.nbest_format``,
    ``prompt.itn``, ``asr.nbest`` and ``audio.*`` do.
    """
    return _digest(_state(cfg, scopes=TRAINING_SCOPES))

def enabled_features(cfg: FeatureConfig) -> list[str]:
    """Sorted ids of every enabled feature (registry order-independent)."""
    from .registry import FEATURES  # local import: registry imports this module

    return sorted(spec.id for spec in FEATURES if bool(_walk(cfg, spec.config_path)))
