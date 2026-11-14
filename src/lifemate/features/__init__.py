"""Feature flags: one authoritative object, one registry, one resolution chain.

Nothing here implements a technique. This package is the switchboard other
workstreams read:

* :mod:`lifemate.features.config` — the ``FeatureConfig`` dataclass tree, its
  fingerprints, and ``enabled_features``;
* :mod:`lifemate.features.registry` — ``FEATURES``, validation, lazy
  availability;
* :mod:`lifemate.features.overrides` — YAML/env/CLI precedence;
* :mod:`lifemate.features.report` — the effective-configuration report.

Every flag defaults to ``False``. See ``docs/FEATURES.md``.
"""

from __future__ import annotations

from .config import (
    AsrFeatureConfig,
    AudioFeatureConfig,
    DecodingFeatureConfig,
    FeatureConfig,
    FINGERPRINT_EXCLUDED,
    PostProcessingFeatureConfig,
    PromptFeatureConfig,
    TrainingFeatureConfig,
    enabled_features,
    feature_fingerprint,
    training_feature_fingerprint,
)
from .overrides import (
    add_feature_arguments,
    apply_overrides,
    build_feature_config,
    env_name_for,
    parse_env_overrides,
    parse_feature_overrides,
)
from .registry import FEATURES, REGISTRY, FeatureRegistry, FeatureSpec
from .report import feature_report_dict, format_feature_report

__all__ = [
    "AudioFeatureConfig",
    "AsrFeatureConfig",
    "PromptFeatureConfig",
    "TrainingFeatureConfig",
    "DecodingFeatureConfig",
    "PostProcessingFeatureConfig",
    "FeatureConfig",
    "FINGERPRINT_EXCLUDED",
    "FEATURES",
    "REGISTRY",
    "FeatureRegistry",
    "FeatureSpec",
    "add_feature_arguments",
    "apply_overrides",
    "build_feature_config",
    "enabled_features",
    "env_name_for",
    "feature_fingerprint",
    "feature_report_dict",
    "format_feature_report",
    "parse_env_overrides",
    "parse_feature_overrides",
    "training_feature_fingerprint",
]
