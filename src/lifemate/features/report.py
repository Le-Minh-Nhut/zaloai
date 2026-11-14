"""Human and JSON reports of the effective feature configuration.

``format_feature_report`` is the shape printed by the scripts before a run;
``feature_report_dict`` is the same data for ``report.json`` / checkpoint
metadata. Both derive ``mode`` from ``enabled``: ``BASELINE`` iff nothing is on.
"""

from __future__ import annotations

from typing import Any, Mapping

from .config import FeatureConfig, enabled_features, feature_fingerprint, training_feature_fingerprint
from .registry import REGISTRY

__all__ = ["format_feature_report", "feature_report_dict", "report_mode"]

_RULE = "=" * 31

def report_mode(enabled: list[str]) -> str:
    """``BASELINE`` when no feature is enabled, else ``EXPERIMENTAL``."""
    return "EXPERIMENTAL" if enabled else "BASELINE"

def format_feature_report(
    cfg: FeatureConfig,
    *,
    model_id: str,
    asr_id: str,
    registry: Any = REGISTRY,
) -> str:
    """Render the effective configuration; labels are contractual."""
    if getattr(getattr(cfg, "asr", None), "gipformer_v15_enabled", False):
        asr_id = "g-group-ai-lab/gipformer1.5-68M-rnnt (requested)"
    enabled = enabled_features(cfg)
    known = registry.ids
    disabled = sorted(set(known) - set(enabled))
    lines = [
        _RULE,
        "LifeMate Effective Configuration",
        _RULE,
        "",
        "Model:",
        f"  {model_id}",
        "",
        "ASR:",
        f"  {asr_id}",
        "",
        "Enabled features:",
    ]
    lines += [f"  {feature_id}" for feature_id in enabled] or ["  NONE"]
    lines += ["", "Disabled features:"]
    lines += [f"  {feature_id}" for feature_id in disabled] or ["  NONE"]
    lines += ["", "Mode:", f"  {report_mode(enabled)}"]
    return "\n".join(lines) + "\n"

def feature_report_dict(
    cfg: FeatureConfig,
    *,
    model_id: str,
    asr_id: str,
    registry: Any = REGISTRY,
) -> dict[str, Any]:
    """JSON-ready feature report (checkpoint metadata / ``report.json``)."""
    if getattr(getattr(cfg, "asr", None), "gipformer_v15_enabled", False):
        asr_id = "g-group-ai-lab/gipformer1.5-68M-rnnt"
    enabled = enabled_features(cfg)
    provenance: Mapping[str, str] = getattr(cfg, "provenance", {}) or {}
    return {
        "model": model_id,
        "asr": asr_id,
        "enabled": enabled,
        "disabled": sorted(set(registry.ids) - set(enabled)),
        "mode": report_mode(enabled),
        "fingerprint": feature_fingerprint(cfg),
        "training_fingerprint": training_feature_fingerprint(cfg),
        "provenance": {feature_id: provenance.get(feature_id, "default") for feature_id in registry.ids},
    }
