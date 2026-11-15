"""Feature registry: the one list of switches, plus validation.

The registry owns the **feature-id spelling** (``asr.nbest``) and the mapping
from an id to its ``features.<section>.<field>`` config path. No other module
may hardcode that mapping.

Availability is lazy: a *disabled* feature is never probed, so importing
:mod:`lifemate.features` and validating a default config touches no optional
dependency. Dotted names in ``requires`` are an **any-of** list (a feature with
three candidate backends needs one of them); non-dotted entries read like
artifacts (``"silero-vad weights"``) and are reported, never used to fail
validation — this package cannot prove an artifact is absent. Pass a custom
``available`` callable for an all-of or on-disk rule.
"""

from __future__ import annotations

import difflib
import importlib.util
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..errors import FeatureConfigurationError
from .config import FeatureConfig

__all__ = [
    "FeatureSpec",
    "FeatureRegistry",
    "FEATURES",
    "REGISTRY",
    "PARAM_RULES",
    "SCOPES",
    "suggest_feature_ids",
    "unknown_feature_error",
]

SCOPES = ("inference", "training", "both")

_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")

def _looks_like_module(name: str) -> bool:
    return bool(_MODULE_RE.match(name))

def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # missing parent package, odd spec, or a hostile import hook
        # A probe must never crash validation: "not importable" is the safe answer.
        return False

def _probe_requires(requires: Sequence[str]) -> tuple[bool, str]:
    """``(available, reason)`` for a spec's ``requires`` entries."""
    modules = [item for item in requires if _looks_like_module(item)]
    artifacts = [item for item in requires if not _looks_like_module(item)]
    if modules and not any(_module_present(name) for name in modules):
        return False, (
            f"none of {', '.join(modules)} is importable; install one or disable the feature"
        )
    reasons = [f"importable: {name}" for name in modules if _module_present(name)]
    reasons += [f"artifact assumed present: {item}" for item in artifacts]
    return True, "; ".join(reasons) or "no external requirement"

@dataclass(frozen=True)
class FeatureSpec:
    """One switch: where it lives, its default, what it needs.

    ``params`` maps a parameter name to ``(kind, default)``. A parameter value is
    inert while the flag is ``False``.

    :raises ValueError: at construction when ``default`` is True or a
        self-reference appears — no feature may ship enabled, and that must fail
        before anything reads the registry.
    """

    id: str
    category: str
    description: str
    default: bool
    config_path: tuple[str, str]
    params: Mapping[str, tuple[Any, Any]] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    scope: str = "both"
    requires: tuple[str, ...] = ()
    available: Callable[[], tuple[bool, str]] | None = None

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"{self.id}: scope must be one of {SCOPES}, got {self.scope!r}")
        if self.config_path[0] != self.category:
            raise ValueError(
                f"{self.id}: config_path section {self.config_path[0]!r} "
                f"must equal category {self.category!r}"
            )
        if self.default:
            raise ValueError(f"{self.id}: feature flags must default to False")
        for other in tuple(self.dependencies) + tuple(self.conflicts):
            if other == self.id:
                raise ValueError(f"{self.id}: cannot depend on or conflict with itself")

#: Every feature, in report order. ``requires`` is evaluated lazily.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        id="audio.vad",
        category="audio",
        description="Voice-activity detection trims silence (and re-pads) before ASR.",
        default=False,
        config_path=("audio", "vad_enabled"),
        params={"vad_min_silence_ms": (int, 300), "vad_padding_ms": (int, 200)},
        scope="both",
        requires=("silero-vad model weights, or the webrtcvad package",),
    ),
    FeatureSpec(
        id="audio.rms_normalization",
        category="audio",
        description="RMS level normalisation to a target dBFS before ASR.",
        default=False,
        config_path=("audio", "rms_normalization_enabled"),
        params={"rms_target_dbfs": (float, -23.0)},
        scope="both",
    ),
    FeatureSpec(
        id="asr.gipformer_v15",
        category="asr",
        description="Decode with the Gipformer-1.5 checkpoint instead of gipformer-68M-rnnt.",
        default=False,
        config_path=("asr", "gipformer_v15_enabled"),
        scope="inference",
        requires=("gipformer-1.5 checkpoint files, downloaded locally",),
    ),
    FeatureSpec(
        id="asr.nbest",
        category="asr",
        description="Beam decoding keeps N-best hypotheses for prompting and scoring.",
        default=False,
        config_path=("asr", "nbest_enabled"),
        params={"nbest_size": (int, 3)},
        scope="both",
    ),
    FeatureSpec(
        id="asr.multi_asr",
        category="asr",
        description="Run a secondary ASR model and merge hypotheses by a fixed policy.",
        default=False,
        config_path=("asr", "multi_asr_enabled"),
        params={
            "secondary_model_id": (str, ""),
            "secondary_backend": (str, "whisper"),
            "multi_asr_policy": (str, "primary_topk_secondary_top1"),
            "multi_asr_max_hypotheses": (int, 3),
            "multi_asr_failure_policy": (str, "primary_only"),
            "multi_asr_models": (list, []),
            "multi_asr_budget_seconds": (float, 30.0),
        },
        scope="inference",
    ),
    FeatureSpec(
        id="prompt.itn",
        category="prompt",
        description="Inverse text normalisation of ASR text before prompting.",
        default=False,
        config_path=("prompt", "itn_enabled"),
        scope="both",
    ),
    FeatureSpec(
        id="prompt.nbest_format",
        category="prompt",
        description="Render the N-best list into the prompt instead of the top-1 transcript.",
        default=False,
        config_path=("prompt", "nbest_prompt_enabled"),
        scope="both",
        dependencies=(),
    ),
    FeatureSpec(
        id="training.noisy_asr",
        category="training",
        description="Replace a ratio of gold transcripts with ASR output during training.",
        default=False,
        config_path=("training", "noisy_asr_training_enabled"),
        params={"noisy_asr_ratio": (float, 0.25), "noisy_asr_mode": (str, "augment")},
        scope="training",
    ),
    FeatureSpec(
        id="training.slot_masking",
        category="training",
        description="Mask slot values in the prompt so the model must copy them from context.",
        default=False,
        config_path=("training", "slot_masking_enabled"),
        scope="training",
    ),
    FeatureSpec(
        id="training.tool_dropout",
        category="training",
        description="Randomly hide tool definitions from the prompt during training.",
        default=False,
        config_path=("training", "tool_dropout_enabled"),
        params={"tool_dropout_max_tools": (int, 1)},
        scope="training",
    ),
    FeatureSpec(
        id="decoding.constrained",
        category="decoding",
        description="Constrain generation to the tool-call grammar.",
        default=False,
        config_path=("decoding", "constrained_decoding_enabled"),
        params={"constrained_backend": (str, "auto")},
        scope="inference",
        # Nothing beyond the inference stack: the implementation is a plain
        # transformers.LogitsProcessor over a prefix trie (models/constrained.py),
        # which is what HFGenerator already imports. outlines/llguidance/xgrammar
        # are NOT used and must not gate this flag. transformers is named as an
        # artifact (not a dotted module) so availability reads as the already
        # installed stack instead of a new dependency to install.
        requires=("transformers LogitsProcessor (already required for inference)",),
    ),
    FeatureSpec(
        id="postprocessing.enhanced_schema_guardrail",
        category="postprocessing",
        description="Validate tool-call arguments against the catalog schema; reject or repair.",
        default=False,
        config_path=("postprocessing", "enhanced_schema_guardrail_enabled"),
        params={"guardrail_action": (str, "reject")},
        scope="inference",
    ),
)

def suggest_feature_ids(name: str, known: Iterable[str], n: int = 3) -> list[str]:
    """Closest known feature ids to ``name``, best first (may be empty)."""
    return difflib.get_close_matches(str(name), sorted(known), n=n, cutoff=0.5)

def unknown_feature_error(name: str, known: Iterable[str]) -> FeatureConfigurationError:
    """Error for an unknown feature id, naming the offender and close matches."""
    known = sorted(known)
    close = suggest_feature_ids(name, known)
    hint = f"did you mean {', '.join(close)}?" if close else "no close match"
    return FeatureConfigurationError(
        f"unknown feature id {name!r}; {hint} Known ids: {', '.join(known)}"
    )

def _nbest_rules(cfg: FeatureConfig) -> list[str]:
    size = cfg.asr.nbest_size
    if not 1 < size <= 10:
        return [
            f"asr.nbest is enabled but features.asr.nbest_size={size} is outside 2..10; "
            "set nbest_size to a beam width between 2 and 10, or disable asr.nbest"
        ]
    return []

def _multi_asr_rules(cfg: FeatureConfig) -> list[str]:
    problems: list[str] = []
    if not cfg.asr.multi_asr_models and not str(cfg.asr.secondary_model_id).strip():
        problems.append(
            "asr.multi_asr is enabled but features.asr.secondary_model_id is empty; "
            "set it to a secondary ASR model id (e.g. 'vinai/PhoWhisper-small') "
            "or disable asr.multi_asr"
        )
    if cfg.asr.multi_asr_max_hypotheses < 1:
        problems.append(
            "features.asr.multi_asr_max_hypotheses must be >= 1; it caps how many "
            "hypotheses reach the prompt"
        )
    return problems

def _noisy_asr_rules(cfg: FeatureConfig) -> list[str]:
    ratio = cfg.training.noisy_asr_ratio
    mode = str(cfg.training.noisy_asr_mode)
    problems: list[str] = []
    if not 0.0 < ratio <= 1.0:
        problems.append(
            f"training.noisy_asr is enabled but features.training.noisy_asr_ratio={ratio} "
            "is outside (0, 1]; set 0 < ratio <= 1 (the share of samples replaced by ASR "
            "text), or disable training.noisy_asr"
        )
    if mode not in ("augment", "replace"):
        problems.append(
            f"training.noisy_asr is enabled but features.training.noisy_asr_mode={mode!r} "
            "is not one of ('augment', 'replace'); 'replace' keeps the record count and "
            "makes the ratio a real noisy share, 'augment' appends variants and makes it "
            "ratio/(1+ratio)"
        )
    return problems

def _tool_dropout_rules(cfg: FeatureConfig) -> list[str]:
    if cfg.training.tool_dropout_max_tools < 1:
        return [
            "features.training.tool_dropout_max_tools must be >= 1; it caps how many "
            "tool definitions are hidden per sample"
        ]
    return []
def _nbest_format_rules(cfg: FeatureConfig) -> list[str]:
    if not (cfg.asr.nbest_enabled or cfg.asr.multi_asr_enabled):
        return [
            "prompt.nbest_format is enabled but its dependency asr.nbest is not; "
            "enable asr.nbest as well, or disable prompt.nbest_format"
        ]
    return []


#: Per-id parameter checks. Each runs ONLY while that feature is enabled: a
#: numeric parameter with its flag off is inert and must never raise.
PARAM_RULES: Mapping[str, Callable[[FeatureConfig], list[str]]] = {
    "asr.nbest": _nbest_rules,
    "asr.multi_asr": _multi_asr_rules,
    "training.noisy_asr": _noisy_asr_rules,
    "training.tool_dropout": _tool_dropout_rules,
    "prompt.nbest_format": _nbest_format_rules,
}

class FeatureRegistry:
    """Lookup + validation over an ordered tuple of :class:`FeatureSpec`."""

    def __init__(self, specs: Sequence[FeatureSpec] = FEATURES) -> None:
        self.specs: tuple[FeatureSpec, ...] = tuple(specs)
        self._by_id: dict[str, FeatureSpec] = {spec.id: spec for spec in self.specs}
        if len(self._by_id) != len(self.specs):
            raise ValueError("duplicate feature id in registry")
        self._availability: dict[str, tuple[bool, str]] = {}

    # -- lookup --------------------------------------------------------- #
    @property
    def ids(self) -> list[str]:
        """Registered ids, in declaration order."""
        return [spec.id for spec in self.specs]

    def spec(self, feature_id: str) -> FeatureSpec:
        """Spec for ``feature_id``; unknown ids raise with close matches."""
        try:
            return self._by_id[feature_id]
        except KeyError:
            raise unknown_feature_error(feature_id, self._by_id) from None

    def value(self, feature_id: str, cfg: FeatureConfig) -> bool:
        """Current on/off value of ``feature_id`` in ``cfg``."""
        section, name = self.spec(feature_id).config_path
        return bool(getattr(getattr(cfg, section), name))

    def is_enabled(self, feature_id: str, cfg: FeatureConfig) -> bool:
        """Whether ``feature_id`` is enabled in ``cfg``."""
        return self.value(feature_id, cfg)

    def set_enabled(self, feature_id: str, cfg: FeatureConfig, value: bool) -> FeatureConfig:
        """Set ``feature_id`` in ``cfg`` in place (``apply_overrides`` clones first)."""
        section, name = self.spec(feature_id).config_path
        setattr(getattr(cfg, section), name, bool(value))
        return cfg

    def list_enabled(self, cfg: FeatureConfig) -> list[str]:
        """Sorted ids of the enabled features."""
        return sorted(spec.id for spec in self.specs if self.value(spec.id, cfg))

    # -- availability --------------------------------------------------- #
    def availability(self, feature_id: str) -> tuple[bool, str]:
        """Cached ``(available, reason)`` for one feature (probed on first use)."""
        spec = self.spec(feature_id)
        if feature_id not in self._availability:
            self._availability[feature_id] = (
                spec.available() if spec.available is not None else _probe_requires(spec.requires)
            )
        return self._availability[feature_id]

    def resolve_availability(self) -> dict[str, dict[str, Any]]:
        """Probe EVERY spec once: ``id -> {available, reason, requires}``.

        Validation never calls this — it probes only the enabled features.
        """
        return {
            spec.id: {
                "available": self.availability(spec.id)[0],
                "reason": self.availability(spec.id)[1],
                "requires": list(spec.requires),
            }
            for spec in self.specs
        }

    # -- validation ----------------------------------------------------- #
    def validate(self, cfg: FeatureConfig, unknown_ids: Sequence[str] = ()) -> list[str]:
        """Every problem with ``cfg`` as human-readable strings; raises nothing.

        ``unknown_ids`` (typos the override layer collected) are reported with
        close-match suggestions.
        """
        problems = [str(unknown_feature_error(name, self._by_id)) for name in unknown_ids]
        enabled = set(self.list_enabled(cfg))
        for spec in self.specs:
            if spec.id not in enabled:
                continue
            for dependency in spec.dependencies:
                if dependency not in enabled:
                    problems.append(
                        f"{spec.id} is enabled but its dependency {dependency} is not; "
                        f"enable {dependency} as well, or disable {spec.id}"
                    )
            for other in spec.conflicts:
                if other in enabled:
                    other_path = self._by_id[other].config_path
                    problems.append(
                        f"{spec.id} conflicts with {other}; enable only one of them "
                        f"(set features.{other_path[0]}.{other_path[1]}=false)"
                    )
            available, reason = self.availability(spec.id)
            if not available:
                problems.append(f"{spec.id} is enabled but unavailable: {reason}")
            rule = PARAM_RULES.get(spec.id)
            if rule is not None:
                problems.extend(rule(cfg))
        return problems

    def validate_or_raise(self, cfg: FeatureConfig, unknown_ids: Sequence[str] = ()) -> FeatureConfig:
        """Validate ``cfg``; raise :class:`FeatureConfigurationError` on problems."""
        problems = self.validate(cfg, unknown_ids)
        if problems:
            joined = "\n".join(f"  - {problem}" for problem in problems)
            raise FeatureConfigurationError(
                f"invalid feature configuration ({len(problems)} problem(s)):\n{joined}"
            )
        return cfg

    # -- introspection -------------------------------------------------- #
    def describe(self) -> list[dict[str, Any]]:
        """JSON-ready description of every feature."""
        return [
            {
                "id": spec.id,
                "category": spec.category,
                "description": spec.description,
                "default": spec.default,
                "config_path": ".".join(spec.config_path),
                "params": {name: kind.__name__ for name, (kind, _) in sorted(spec.params.items())},
                "dependencies": list(spec.dependencies),
                "conflicts": list(spec.conflicts),
                "scope": spec.scope,
                "requires": list(spec.requires),
            }
            for spec in self.specs
        ]

#: The registry the rest of the package uses.
REGISTRY = FeatureRegistry()
