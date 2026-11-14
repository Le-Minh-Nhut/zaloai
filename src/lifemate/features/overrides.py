"""Feature override resolution: one chain, one precedence.

    CLI  >  env  >  YAML  >  defaults

Every level speaks the same vocabulary — a **feature id** (``asr.nbest``), never
a config path. The id -> ``features.<section>.<field>`` mapping lives only in
:mod:`lifemate.features.registry`.

Provenance is recorded per enabled-or-explicitly-set id so a report can say WHY
a feature is on (``"cli"``/``"env"``/``"yaml"``), while untouched ids read
``"default"``.

Typo policy: an unknown id at any level raises
:class:`~lifemate.errors.FeatureConfigurationError` naming the offending string
and the closest known ids. Nothing is silently ignored.
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Iterable, Mapping, Sequence

from ..errors import FeatureConfigurationError
from .config import FeatureConfig, apply_mapping
from .registry import REGISTRY, FeatureRegistry, unknown_feature_error

__all__ = [
    "ENV_PREFIX",
    "TRUE_TEXT",
    "FALSE_TEXT",
    "env_name_for",
    "parse_feature_overrides",
    "parse_env_overrides",
    "load_yaml_feature_overrides",
    "apply_overrides",
    "add_feature_arguments",
    "build_feature_config",
    "feature_config_from_yaml",
]

#: ``LIFEMATE_FEATURE_<ID_UPPER_SNAKE>``
ENV_PREFIX = "LIFEMATE_FEATURE_"

TRUE_TEXT = frozenset({"1", "true", "yes", "on"})
FALSE_TEXT = frozenset({"0", "false", "no", "off"})

def _read_bool(text: str, *, where: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in TRUE_TEXT:
        return True
    if lowered in FALSE_TEXT:
        return False
    raise FeatureConfigurationError(
        f"{where}: cannot read {text!r} as bool; use 1/0, true/false, yes/no or on/off"
    )

def env_name_for(feature_id: str) -> str:
    """``asr.nbest`` -> ``LIFEMATE_FEATURE_ASR_NBEST``."""
    return ENV_PREFIX + feature_id.replace(".", "_").replace("-", "_").upper()

def parse_feature_overrides(pairs: Sequence[str], registry: FeatureRegistry = REGISTRY) -> dict[str, bool]:
    """Parse ``--enable-feature``-style items: ``id`` or ``id=true|false|1|0``.

    A bare ``id`` means ``True``. Unknown ids raise.
    """
    parsed: dict[str, bool] = {}
    for raw in pairs:
        item = str(raw).strip()
        if not item:
            raise FeatureConfigurationError("empty feature id; pass e.g. --enable-feature asr.nbest")
        name, sep, value = item.partition("=")
        name = name.strip()
        if not name:
            raise FeatureConfigurationError(f"missing feature id in {raw!r}")
        if name not in registry.ids:
            raise unknown_feature_error(name, registry.ids)
        parsed[name] = _read_bool(value, where=f"feature override {item!r}") if sep else True
    return parsed

def parse_env_overrides(
    environ: Mapping[str, str] | None = None,
    registry: FeatureRegistry = REGISTRY,
) -> dict[str, bool]:
    """Read every ``LIFEMATE_FEATURE_*`` variable in ``environ``.

    Round-trips :func:`env_name_for`; a name under the prefix that no feature
    maps to raises instead of being ignored.
    """
    source = os.environ if environ is None else environ
    by_env = {env_name_for(feature_id): feature_id for feature_id in registry.ids}
    parsed: dict[str, bool] = {}
    for name, value in source.items():
        if not str(name).startswith(ENV_PREFIX):
            continue
        feature_id = by_env.get(str(name))
        if feature_id is None:
            guesses = sorted(by_env)
            from difflib import get_close_matches

            close = get_close_matches(str(name), guesses, n=3, cutoff=0.5)
            hint = f"did you mean {', '.join(close)}?" if close else "no close match"
            raise FeatureConfigurationError(
                f"unknown feature environment variable {name!r}; {hint} "
                f"(feature ids are spelled {ENV_PREFIX}<ID_UPPER_SNAKE>, e.g. "
                f"{env_name_for('asr.nbest')})"
            )
        parsed[feature_id] = _read_bool(value, where=f"env {name}")
    return parsed

def load_yaml_feature_overrides(
    path: str | os.PathLike[str],
    registry: FeatureRegistry = REGISTRY,
) -> tuple[dict[str, bool], dict[str, Any], list[str]]:
    """Read the ``features:`` section of a YAML file.

    Returns ``(flags, params, unknown_keys)``:

    * ``flags`` — ``feature id -> bool`` derived from the file's paths;
    * ``params`` — the raw ``features:`` mapping, applied verbatim;
    * ``unknown_keys`` — dotted paths the config object does not have.

    Parsing uses :mod:`yaml` (present in every supported interpreter) so this
    helper never depends on the repo loader learning about ``features:``.
    """
    import yaml  # local: keeps module import cheap

    from ..compliance import assert_path_allowed

    payload = yaml.safe_load(assert_path_allowed(path).read_text(encoding="utf-8")) or {}
    section = payload.get("features") or {} if isinstance(payload, Mapping) else {}
    if not isinstance(section, Mapping):
        raise FeatureConfigurationError(f"{path}: 'features' must be a mapping, got {type(section).__name__}")
    flags: dict[str, bool] = {}
    for spec in registry.specs:
        target = section.get(spec.category)
        if not isinstance(target, Mapping) or spec.config_path[1] not in target:
            continue
        value = _read_bool(
            str(target[spec.config_path[1]]), where=f"{path}: features.{'.'.join(spec.config_path)}"
        )
        flags[spec.id] = value
    scratch = FeatureConfig()
    unknown = apply_mapping(scratch, section)
    params = {key: value for key, value in section.items() if isinstance(value, Mapping)}
    return flags, params, unknown

def apply_overrides(
    cfg: FeatureConfig,
    *,
    yaml: Mapping[str, bool] | None = None,
    env: Mapping[str, bool] | None = None,
    cli: Mapping[str, bool] | None = None,
    params: Mapping[str, Any] | None = None,
    unknown: Iterable[str] = (),
    registry: FeatureRegistry = REGISTRY,
) -> tuple[FeatureConfig, dict[str, str]]:
    """Resolve the chain into a NEW :class:`FeatureConfig`.

    ``cfg`` is never mutated. Returns ``(new_cfg, provenance)`` and attaches the
    same mapping as ``new_cfg.provenance``; provenance values are
    ``"cli"``/``"env"``/``"yaml"``/``"default"``.

    ``params`` (raw ``features:`` mappings) are applied first, then the three
    flag layers in precedence order, so a flag always wins over the file value of
    its own parameters.

    :raises FeatureConfigurationError: unknown ids at any level, or unknown
        ``features:`` keys (``unknown``) — collected, then reported together.
    """
    resolved = copy.deepcopy(cfg)
    problems: list[str] = []
    if params:
        problems.extend(
            f"unknown feature config key 'features.{key}'; run "
            f"`python3 -c \"from lifemate.features.registry import REGISTRY; "
            f"import json; print(json.dumps(REGISTRY.describe(), indent=2))\"` for the valid keys"
            for key in apply_mapping(resolved, params)
        )
    problems.extend(str(unknown_feature_error(name, registry.ids)) for name in unknown)

    provenance: dict[str, str] = {}
    for label, layer in (("yaml", yaml), ("env", env), ("cli", cli)):
        for feature_id, enabled in (layer or {}).items():
            if feature_id not in registry.ids:
                problems.append(str(unknown_feature_error(feature_id, registry.ids)))
                continue
            registry.set_enabled(feature_id, resolved, bool(enabled))
            provenance[feature_id] = label

    for feature_id in registry.ids:
        provenance.setdefault(feature_id, "default")
    resolved.provenance = provenance

    if problems:
        joined = "\n".join(f"  - {problem}" for problem in problems)
        raise FeatureConfigurationError(f"feature overrides rejected:\n{joined}")
    return resolved, provenance

def add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--enable-feature ID`` / ``--disable-feature ID`` to ``parser``.

    Wiring into a script is three lines; see ``docs/FEATURES.md``.
    """
    group = parser.add_argument_group("features")
    group.add_argument(
        "--enable-feature",
        metavar="ID",
        action="append",
        default=None,
        help="feature id to enable, repeatable (e.g. asr.nbest; ID=false also works)",
    )
    group.add_argument(
        "--disable-feature",
        metavar="ID",
        action="append",
        default=None,
        help="feature id to force off, repeatable (overrides YAML and env)",
    )

def build_feature_config(
    base_cfg: Any,
    args: argparse.Namespace | None,
    environ: Mapping[str, str] | None = None,
    *,
    yaml_section: Mapping[str, Any] | None = None,
    yaml_flags: Mapping[str, bool] | None = None,
    unknown: Iterable[str] = (),
    registry: FeatureRegistry = REGISTRY,
) -> tuple[FeatureConfig, dict[str, str]]:
    """Run the whole chain from an argparse namespace.

    ``base_cfg`` may be a :class:`FeatureConfig` or any object carrying one as
    ``.features`` (e.g. ``AppConfig`` after wiring). Returns
    ``(new_feature_cfg, provenance)`` without mutating ``base_cfg``.
    """
    base = getattr(base_cfg, "features", base_cfg)
    if not isinstance(base, FeatureConfig):
        raise FeatureConfigurationError(
            f"build_feature_config: {type(base_cfg).__name__} has no FeatureConfig in .features"
        )
    enable = list(getattr(args, "enable_feature", None) or [])
    disable = list(getattr(args, "disable_feature", None) or [])
    cli = parse_feature_overrides(enable, registry) if enable else {}
    for feature_id in parse_feature_overrides(disable, registry) if disable else {}:
        cli[feature_id] = False
    return apply_overrides(
        base,
        yaml=yaml_flags,
        env=parse_env_overrides(environ, registry),
        cli=cli,
        params=yaml_section,
        unknown=unknown,
        registry=registry,
    )

def feature_config_from_yaml(
    path: str | os.PathLike[str],
    registry: FeatureRegistry = REGISTRY,
) -> tuple[FeatureConfig, dict[str, str]]:
    """Convenience: defaults + one experiment YAML's ``features:`` section."""
    flags, params, unknown = load_yaml_feature_overrides(path, registry)
    return apply_overrides(FeatureConfig(), yaml=flags, params=params, unknown=unknown, registry=registry)
