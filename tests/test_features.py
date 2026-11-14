"""Feature-flag infrastructure: defaults, registry, overrides, fingerprints, report.

Stdlib-only (plus PyYAML, which every supported interpreter ships). No pytest
fixtures: every test takes zero arguments so ``python3 tests/run_tests.py`` runs
them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from lifemate.errors import (  # noqa: E402
    ConfigError,
    FeatureConfigurationError,
    LifeMateError,
    ProhibitedAccessError,
)
from lifemate.features.config import (  # noqa: E402
    FINGERPRINT_EXCLUDED,
    FeatureConfig,
    enabled_features,
    feature_fingerprint,
    training_feature_fingerprint,
)
from lifemate.features.overrides import (  # noqa: E402
    add_feature_arguments,
    apply_overrides,
    build_feature_config,
    env_name_for,
    feature_config_from_yaml,
    load_yaml_feature_overrides,
    parse_env_overrides,
    parse_feature_overrides,
)
from lifemate.features.registry import (  # noqa: E402
    FEATURES,
    REGISTRY,
    FeatureRegistry,
    FeatureSpec,
)
from lifemate.features.report import (  # noqa: E402
    feature_report_dict,
    format_feature_report,
    report_mode,
)

EXPERIMENTS = ROOT / "configs" / "experiments"

#: Every experiment overlay -> the exact set of ids it must enable.
EXPECTED_FLAGS = {
    "baseline.yaml": set(),
    "gipformer_v15.yaml": {"asr.gipformer_v15"},
    "asr_nbest.yaml": {"asr.nbest"},
    "itn.yaml": {"prompt.itn"},
    "enhanced_guardrail.yaml": {"postprocessing.enhanced_schema_guardrail"},
    "noisy_asr_training.yaml": {"training.noisy_asr"},
    "advanced_combined.yaml": {
        "asr.gipformer_v15",
        "asr.nbest",
        "prompt.itn",
        "postprocessing.enhanced_schema_guardrail",
    },
}

#: Inference-only ids: they must never move the training fingerprint.
INFERENCE_ONLY = (
    "asr.gipformer_v15",
    "asr.multi_asr",
    "decoding.constrained",
    "postprocessing.enhanced_schema_guardrail",
)

def _enable(feature_id: str, **params) -> FeatureConfig:
    cfg = FeatureConfig()
    REGISTRY.set_enabled(feature_id, cfg, True)
    section = feature_id.split(".", 1)[0]
    for name, value in params.items():
        setattr(getattr(cfg, section), name, value)
    return cfg

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_every_registered_feature_defaults_to_false():
    """Iterate the registry: a newly added flag cannot ship enabled."""
    cfg = FeatureConfig()
    for spec in FEATURES:
        assert spec.default is False, spec.id
        assert REGISTRY.is_enabled(spec.id, cfg) is False, spec.id
        section, name = spec.config_path
        assert getattr(getattr(cfg, section), name) is False, spec.id
    assert enabled_features(cfg) == []
    assert REGISTRY.list_enabled(cfg) == []

def test_registry_ids_are_the_contracted_twelve():
    assert REGISTRY.ids == [
        "audio.vad",
        "audio.rms_normalization",
        "asr.gipformer_v15",
        "asr.nbest",
        "asr.multi_asr",
        "prompt.itn",
        "prompt.nbest_format",
        "training.noisy_asr",
        "training.slot_masking",
        "training.tool_dropout",
        "decoding.constrained",
        "postprocessing.enhanced_schema_guardrail",
    ]

def test_feature_spec_refuses_an_enabled_default():
    try:
        FeatureSpec(
            id="x.bad",
            category="x",
            description="bad",
            default=True,
            config_path=("x", "bad_enabled"),
        )
    except ValueError as exc:
        assert "must default to False" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("FeatureSpec accepted default=True")

def test_describe_covers_every_feature_with_its_scope():
    described = {item["id"]: item for item in REGISTRY.describe()}
    assert set(described) == set(REGISTRY.ids)
    assert described["asr.nbest"]["scope"] == "both"
    assert described["training.slot_masking"]["scope"] == "training"
    assert described["decoding.constrained"]["scope"] == "inference"
    assert described["asr.nbest"]["config_path"] == "asr.nbest_enabled"
    json.dumps(REGISTRY.describe())  # JSON-ready

# --------------------------------------------------------------------------- #
# YAML overlays
# --------------------------------------------------------------------------- #
def test_every_experiment_yaml_declares_exactly_its_flags_and_parses():
    assert sorted(p.name for p in EXPERIMENTS.glob("*.yaml")) == sorted(EXPECTED_FLAGS)
    for name, expected in EXPECTED_FLAGS.items():
        path = EXPERIMENTS / name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(payload) == {"features"}, (name, payload.keys())
        cfg, provenance = feature_config_from_yaml(path)
        assert set(enabled_features(cfg)) == expected, (name, enabled_features(cfg))
        for feature_id in expected:
            assert provenance[feature_id] == "yaml", (name, feature_id)
        assert REGISTRY.validate(cfg) == [], (name, REGISTRY.validate(cfg))

def test_baseline_yaml_is_explicit_and_reports_baseline_mode():
    payload = yaml.safe_load((EXPERIMENTS / "baseline.yaml").read_text(encoding="utf-8"))
    flattened = {(section, key): value for section, body in payload["features"].items() for key, value in body.items()}
    for spec in FEATURES:
        assert flattened[spec.config_path] is False, spec.id
    cfg, _ = feature_config_from_yaml(EXPERIMENTS / "baseline.yaml")
    assert enabled_features(cfg) == []
    assert report_mode(enabled_features(cfg)) == "BASELINE"

def test_every_experiment_overlay_loads_through_the_real_loader():
    """``AppConfig.features`` is wired: each overlay loads and enables exactly its ids.

    This is the inverse of the earlier placeholder (which pinned
    ``ConfigError: unknown config key: features``). Exact set equality, not
    "contains": an overlay that quietly enables a second flag fails here.
    """
    from lifemate.config import load_config

    for name, expected in EXPECTED_FLAGS.items():
        cfg = load_config(EXPERIMENTS / name)
        assert cfg.features is not None, name
        assert isinstance(cfg.features, FeatureConfig), (name, type(cfg.features).__name__)
        assert set(enabled_features(cfg.features)) == expected, (name, enabled_features(cfg.features))
        assert REGISTRY.validate(cfg.features) == [], (name, REGISTRY.validate(cfg.features))
        # The overlay is still a partial config: absent sections keep defaults.
        assert cfg.training.tuning_mode == "lora", name
        assert cfg.model.name == "openbmb/MiniCPM5-2B", name
        assert cfg.asr.model_id == "g-group-ai-lab/gipformer-68M-rnnt", name

def test_loader_and_override_layer_agree_on_every_overlay():
    """Two readers of the same file must not disagree about which flags are on."""
    from lifemate.config import load_config

    for name, expected in EXPECTED_FLAGS.items():
        path = EXPERIMENTS / name
        loaded = load_config(path)
        from_yaml, _provenance = feature_config_from_yaml(path)
        assert enabled_features(loaded.features) == enabled_features(from_yaml), name
        assert enabled_features(loaded.features) == sorted(expected), name
        assert feature_fingerprint(loaded.features) == feature_fingerprint(from_yaml), name
        assert training_feature_fingerprint(loaded.features) == training_feature_fingerprint(from_yaml), name

def test_a_typo_in_a_features_key_is_still_rejected_by_the_loader():
    """Inverting the placeholder must not have removed the unknown-key guard."""
    from lifemate.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        typo = Path(tmp) / "typo.yaml"
        typo.write_text("features:\n  asr:\n    nbest_enable: true\n", encoding="utf-8")
        try:
            load_config(typo)
        except ConfigError as exc:
            assert "nbest_enable" in str(exc), str(exc)
            assert "unknown config key" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("a typo in a features key was accepted")

        unknown_section = Path(tmp) / "section.yaml"
        unknown_section.write_text("features:\n  teleport:\n    on: true\n", encoding="utf-8")
        try:
            load_config(unknown_section)
        except ConfigError as exc:
            assert "teleport" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("an unknown features section was accepted")

def test_an_invalid_feature_combination_is_a_config_error_from_the_loader():
    """Registry rules surface as ``ConfigError`` (the scripts' existing contract)."""
    from lifemate.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.yaml"
        path.write_text(
            "features:\n  prompt:\n    nbest_prompt_enabled: true\n", encoding="utf-8"
        )
        try:
            load_config(path)
        except ConfigError as exc:
            message = str(exc)
            assert "prompt.nbest_format" in message and "asr.nbest" in message, message
        else:  # pragma: no cover
            raise AssertionError("a dependency violation was accepted by load_config")

def test_config_fingerprint_does_not_include_the_features_section():
    """Checkpoint identity must not move when only a feature flag flips.

    ``CONFIG_HASHED_SECTIONS`` whitelists ``training`` + ``model`` (and ``lora``
    in LoRA mode); ``features`` is excluded by construction. The pinned digest
    below is the PRE-feature value: it is identical on ``HEAD`` (before the
    section existed) and in the worktree, so the block provably did not leak into
    the hash.
    """
    from lifemate.config import load_config
    from lifemate.training.trainer import CONFIG_HASHED_SECTIONS, config_fingerprint

    assert "features" not in CONFIG_HASHED_SECTIONS
    assert set(CONFIG_HASHED_SECTIONS) == {"training", "model"}

    cfg = load_config(ROOT / "configs" / "training_lora.yaml")
    pinned = "6c6405d8f07cb105"
    assert config_fingerprint(cfg) == pinned, config_fingerprint(cfg)

    all_on = load_config(ROOT / "configs" / "training_lora.yaml")
    for spec in FEATURES:
        REGISTRY.set_enabled(spec.id, all_on.features, True)
    all_on.features.asr.nbest_size = 9
    all_on.features.training.noisy_asr_ratio = 1.0
    all_on.features.decoding.constrained_backend = "xgrammar"
    assert enabled_features(all_on.features) == sorted(REGISTRY.ids)
    assert config_fingerprint(all_on) == pinned

    absent = load_config(ROOT / "configs" / "training_lora.yaml")
    absent.features = None
    assert config_fingerprint(absent) == pinned

def test_feature_config_survives_asdict_dump_and_reload():
    """``to_dict`` -> ``dump_config`` -> ``load_config`` keeps the flags and params."""
    from lifemate.config import dump_config, load_config

    cfg = load_config(EXPERIMENTS / "advanced_combined.yaml")
    before = enabled_features(cfg.features)
    assert before, "the round-trip test needs a non-empty feature set"
    cfg.features.asr.nbest_size = 7
    cfg.features.training.noisy_asr_ratio = 0.75

    with tempfile.TemporaryDirectory() as tmp:
        dumped = dump_config(cfg, Path(tmp) / "roundtrip.yaml")
        text = dumped.read_text(encoding="utf-8")
        assert "features:" in text
        reloaded = load_config(dumped)

    assert enabled_features(reloaded.features) == before
    assert reloaded.features.asr.nbest_size == 7
    assert reloaded.features.training.noisy_asr_ratio == 0.75
    assert reloaded.to_dict() == cfg.to_dict()
    assert feature_fingerprint(reloaded.features) == feature_fingerprint(cfg.features)
    assert training_feature_fingerprint(reloaded.features) == training_feature_fingerprint(cfg.features)
    # provenance is bookkeeping, never serialised state
    assert "provenance" not in cfg.to_dict()
    assert "provenance" not in reloaded.to_dict()

# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #
def test_precedence_default_when_nothing_is_set():
    cfg, provenance = apply_overrides(FeatureConfig(), yaml={}, env={}, cli={})
    assert REGISTRY.is_enabled("asr.nbest", cfg) is False
    assert provenance["asr.nbest"] == "default"
    assert set(provenance) == set(REGISTRY.ids)

def test_precedence_yaml_enables_when_env_and_cli_are_silent():
    cfg, provenance = apply_overrides(FeatureConfig(), yaml={"asr.nbest": True})
    assert REGISTRY.is_enabled("asr.nbest", cfg) is True
    assert provenance["asr.nbest"] == "yaml"

def test_precedence_env_beats_yaml():
    cfg, provenance = apply_overrides(FeatureConfig(), yaml={"asr.nbest": False}, env={"asr.nbest": True})
    assert REGISTRY.is_enabled("asr.nbest", cfg) is True
    assert provenance["asr.nbest"] == "env"

def test_precedence_cli_beats_env_and_yaml():
    cfg, provenance = apply_overrides(
        FeatureConfig(),
        yaml={"asr.nbest": True},
        env={"asr.nbest": True},
        cli={"asr.nbest": False},
    )
    assert REGISTRY.is_enabled("asr.nbest", cfg) is False
    assert provenance["asr.nbest"] == "cli"

def test_apply_overrides_never_mutates_the_input_config():
    base = FeatureConfig()
    snapshot = base.to_dict()
    cfg, _ = apply_overrides(base, yaml={"asr.nbest": True}, cli={"training.slot_masking": True})
    assert base.to_dict() == snapshot
    assert base.provenance == {}
    assert REGISTRY.is_enabled("asr.nbest", cfg) is True
    assert REGISTRY.is_enabled("asr.nbest", base) is False
    assert cfg is not base and cfg.asr is not base.asr

def test_parse_feature_overrides_accepts_bare_and_explicit_values():
    assert parse_feature_overrides(["asr.nbest"]) == {"asr.nbest": True}
    assert parse_feature_overrides(["asr.nbest=true"]) == {"asr.nbest": True}
    assert parse_feature_overrides(["asr.nbest=1"]) == {"asr.nbest": True}
    assert parse_feature_overrides(["asr.nbest=yes"]) == {"asr.nbest": True}
    assert parse_feature_overrides(["asr.nbest=false"]) == {"asr.nbest": False}
    assert parse_feature_overrides(["asr.nbest=0"]) == {"asr.nbest": False}
    assert parse_feature_overrides(["asr.nbest=no"]) == {"asr.nbest": False}
    assert parse_feature_overrides([]) == {}

def test_parse_feature_overrides_rejects_a_bad_bool():
    try:
        parse_feature_overrides(["asr.nbest=maybe"])
    except FeatureConfigurationError as exc:
        assert "maybe" in str(exc) and "bool" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bad bool accepted")

def test_env_names_follow_the_contract():
    assert env_name_for("asr.nbest") == "LIFEMATE_FEATURE_ASR_NBEST"
    assert (
        env_name_for("postprocessing.enhanced_schema_guardrail")
        == "LIFEMATE_FEATURE_POSTPROCESSING_ENHANCED_SCHEMA_GUARDRAIL"
    )
    assert parse_env_overrides({"LIFEMATE_FEATURE_ASR_NBEST": "1"}) == {"asr.nbest": True}
    assert parse_env_overrides({"LIFEMATE_FEATURE_ASR_NBEST": "false"}) == {"asr.nbest": False}
    assert parse_env_overrides({"UNRELATED": "1"}) == {}

def test_parse_env_overrides_reads_the_process_environment_by_default():
    name = "LIFEMATE_FEATURE_PROMPT_ITN"
    previous = os.environ.get(name)
    os.environ[name] = "true"
    try:
        assert parse_env_overrides()["prompt.itn"] is True
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous

def test_unknown_cli_id_raises_naming_the_offender_and_a_close_match():
    try:
        parse_feature_overrides(["asr.nbes"])
    except FeatureConfigurationError as exc:
        message = str(exc)
        assert "asr.nbes" in message
        assert "asr.nbest" in message, message
    else:  # pragma: no cover
        raise AssertionError("unknown CLI id accepted")

def test_unknown_env_name_raises_naming_the_offender_and_a_close_match():
    try:
        parse_env_overrides({"LIFEMATE_FEATURE_ASR_NBEST_SIZE": "1"})
    except FeatureConfigurationError as exc:
        message = str(exc)
        assert "LIFEMATE_FEATURE_ASR_NBEST_SIZE" in message
        assert "LIFEMATE_FEATURE_ASR_NBEST" in message, message
    else:  # pragma: no cover
        raise AssertionError("unknown env name accepted")

def test_unknown_yaml_feature_key_is_reported_not_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "typo.yaml"
        path.write_text("features:\n  asr:\n    nbest_enabled: true\n    nbest_sise: 5\n", encoding="utf-8")
        _, _, unknown = load_yaml_feature_overrides(path)
        assert unknown == ["asr.nbest_sise"]
        try:
            apply_overrides(FeatureConfig(), yaml={"asr.nbest": True}, unknown=unknown)
        except FeatureConfigurationError as exc:
            assert "asr.nbest_sise" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("unknown YAML key silently accepted")

def test_from_dict_returns_unknown_keys_instead_of_dropping_them():
    cfg, unknown = FeatureConfig.from_dict(
        {"asr": {"nbest_enabled": "true", "nbest_sise": 5}, "nope": {"x": 1}}
    )
    assert cfg.asr.nbest_enabled is True
    assert unknown == ["asr.nbest_sise", "nope"]

def test_from_dict_accepts_none_and_missing_sections():
    cfg, unknown = FeatureConfig.from_dict(None)
    assert unknown == [] and enabled_features(cfg) == []
    cfg, unknown = FeatureConfig.from_dict({"prompt": {"itn_enabled": True}})
    assert unknown == [] and enabled_features(cfg) == ["prompt.itn"]

def test_override_loader_refuses_a_prohibited_environment_path():
    """The switchboard obeys the same *_environment guard as every other reader."""
    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / ("train" + "_environment")  # compliance-scan: allow
        env_dir.mkdir()
        path = env_dir / "probe.yaml"
        path.write_text("features:\n  asr:\n    nbest_enabled: true\n", encoding="utf-8")
        try:
            load_yaml_feature_overrides(path)
        except ProhibitedAccessError as exc:
            assert "prohibited" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("a prohibited path was read")

def test_build_feature_config_wires_argparse_env_and_yaml():
    parser = argparse.ArgumentParser(prog="probe")
    add_feature_arguments(parser)
    args = parser.parse_args(["--enable-feature", "asr.nbest", "--disable-feature", "prompt.itn"])
    cfg, provenance = build_feature_config(
        FeatureConfig(),
        args,
        environ={"LIFEMATE_FEATURE_PROMPT_ITN": "1", "LIFEMATE_FEATURE_TRAINING_SLOT_MASKING": "1"},
        yaml_flags={"asr.nbest": False, "training.tool_dropout": True},
        yaml_section={"asr": {"nbest_enabled": False}},
    )
    assert enabled_features(cfg) == [
        "asr.nbest",
        "training.slot_masking",
        "training.tool_dropout",
    ], enabled_features(cfg)
    assert provenance["asr.nbest"] == "cli"
    assert provenance["prompt.itn"] == "cli"
    assert provenance["training.slot_masking"] == "env"
    assert provenance["training.tool_dropout"] == "yaml"

def test_build_feature_config_accepts_a_carrier_object_without_mutating_it():
    class Carrier:
        features = FeatureConfig()

    args = argparse.Namespace(enable_feature=["prompt.itn"], disable_feature=None)
    cfg, _ = build_feature_config(Carrier(), args, environ={})
    assert enabled_features(cfg) == ["prompt.itn"]
    assert enabled_features(Carrier.features) == []
    assert cfg is not Carrier.features

# --------------------------------------------------------------------------- #
# Inert parameters
# --------------------------------------------------------------------------- #
def test_numeric_param_alone_never_enables_a_feature():
    cfg = FeatureConfig()
    cfg.asr.nbest_size = 5
    assert enabled_features(cfg) == []
    assert REGISTRY.validate(cfg) == []
    REGISTRY.validate_or_raise(cfg)

def test_noisy_asr_ratio_alone_is_inert_and_does_not_raise():
    cfg = FeatureConfig()
    cfg.training.noisy_asr_ratio = 0.5
    assert enabled_features(cfg) == []
    assert REGISTRY.validate(cfg) == []
    REGISTRY.validate_or_raise(cfg)

def test_out_of_range_params_are_inert_while_the_flag_is_off():
    cfg = FeatureConfig()
    cfg.asr.nbest_size = 99
    cfg.training.noisy_asr_ratio = 7.0
    cfg.asr.secondary_model_id = ""
    assert REGISTRY.validate(cfg) == []

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_dependency_violation_names_both_features_and_the_fix():
    cfg = _enable("prompt.nbest_format")
    problems = REGISTRY.validate(cfg)
    assert len(problems) == 1, problems
    assert "prompt.nbest_format" in problems[0] and "asr.nbest" in problems[0]

def test_dependency_satisfied_is_clean():
    cfg = _enable("prompt.nbest_format")
    REGISTRY.set_enabled("asr.nbest", cfg, True)
    assert REGISTRY.validate(cfg) == []

def test_declared_conflict_both_enabled_is_an_error():
    specs = (
        FeatureSpec(
            id="a.one",
            category="audio",
            description="first",
            default=False,
            config_path=("audio", "vad_enabled"),
            conflicts=("b.two",),
        ),
        FeatureSpec(
            id="b.two",
            category="audio",
            description="second",
            default=False,
            config_path=("audio", "rms_normalization_enabled"),
        ),
    )
    registry = FeatureRegistry(specs)
    cfg = FeatureConfig()
    registry.set_enabled("a.one", cfg, True)
    registry.set_enabled("b.two", cfg, True)
    problems = registry.validate(cfg)
    assert len(problems) == 1, problems  # reported once, by the spec that declares it
    assert "a.one conflicts with b.two" in problems[0]
    assert "features.audio.rms_normalization_enabled=false" in problems[0]

def test_itn_and_nbest_format_declare_no_conflict():
    itn = next(spec for spec in FEATURES if spec.id == "prompt.itn")
    nbest_format = next(spec for spec in FEATURES if spec.id == "prompt.nbest_format")
    assert "prompt.nbest_format" not in itn.conflicts
    assert "prompt.itn" not in nbest_format.conflicts
    cfg = _enable("prompt.itn")
    REGISTRY.set_enabled("prompt.nbest_format", cfg, True)
    REGISTRY.set_enabled("asr.nbest", cfg, True)
    assert REGISTRY.validate(cfg) == []

def test_nbest_size_below_two_is_an_error():
    cfg = _enable("asr.nbest", nbest_size=1)
    problems = REGISTRY.validate(cfg)
    assert problems and "asr.nbest_size=1" in problems[0]
    assert "asr.nbest" in problems[0]

def test_nbest_size_above_ten_is_an_error():
    cfg = _enable("asr.nbest", nbest_size=11)
    problems = REGISTRY.validate(cfg)
    assert problems and "asr.nbest_size=11" in problems[0]

def test_nbest_size_boundaries_are_accepted():
    for size in (2, 10):
        cfg = _enable("asr.nbest", nbest_size=size)
        assert REGISTRY.validate(cfg) == [], (size, REGISTRY.validate(cfg))

def test_multi_asr_without_secondary_model_id_is_an_error():
    cfg = _enable("asr.multi_asr")
    problems = REGISTRY.validate(cfg)
    assert problems and "features.asr.secondary_model_id" in problems[0]
    assert "asr.multi_asr" in problems[0]
    cfg.asr.secondary_model_id = "vinai/PhoWhisper-small"
    assert REGISTRY.validate(cfg) == []

def test_noisy_asr_ratio_outside_the_open_unit_interval_is_an_error():
    for bad in (0.0, -0.1, 1.5):
        cfg = _enable("training.noisy_asr", noisy_asr_ratio=bad)
        problems = REGISTRY.validate(cfg)
        assert problems and "features.training.noisy_asr_ratio" in problems[0], (bad, problems)
    for good in (1e-6, 0.25, 1.0):
        cfg = _enable("training.noisy_asr", noisy_asr_ratio=good)
        assert REGISTRY.validate(cfg) == [], (good, REGISTRY.validate(cfg))

def test_tool_dropout_max_tools_must_be_positive():
    cfg = _enable("training.tool_dropout", tool_dropout_max_tools=0)
    problems = REGISTRY.validate(cfg)
    assert problems and "features.training.tool_dropout_max_tools" in problems[0]

def test_validate_or_raise_raises_a_feature_configuration_error_under_lifemate_error():
    cfg = _enable("prompt.nbest_format")
    try:
        REGISTRY.validate_or_raise(cfg)
    except FeatureConfigurationError as exc:
        assert isinstance(exc, LifeMateError)
        assert "prompt.nbest_format" in str(exc)
        assert "asr.nbest" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validate_or_raise did not raise")

def test_validate_reports_unknown_override_ids_with_suggestions():
    problems = REGISTRY.validate(FeatureConfig(), unknown_ids=["asr.nbes"])
    assert problems and "asr.nbes" in problems[0] and "asr.nbest" in problems[0]

def test_unknown_feature_id_lookup_suggests_the_close_match():
    try:
        REGISTRY.spec("training.noisy_asr_ratio")
    except FeatureConfigurationError as exc:
        assert "training.noisy_asr" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown id resolved")

# --------------------------------------------------------------------------- #
# Lazy availability
# --------------------------------------------------------------------------- #
PROBE_SCRIPT = textwrap.dedent(
    """
    import sys

    BLOCKED = {"outlines", "llguidance", "xgrammar"}
    CALLS = []

    class Tripwire:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root in BLOCKED:
                CALLS.append(name)
                raise AssertionError("optional module probed while disabled: " + name)
            return None

    sys.meta_path.insert(0, Tripwire())

    from lifemate.features.config import FeatureConfig, enabled_features
    from lifemate.features.registry import REGISTRY

    cfg = FeatureConfig()
    assert enabled_features(cfg) == []
    assert REGISTRY.validate(cfg) == []
    REGISTRY.validate_or_raise(cfg)
    assert CALLS == [], CALLS
    print("clean")

    # ``decoding.constrained`` needs nothing beyond the inference stack the
    # generator already imports (it is a transformers LogitsProcessor + prefix
    # trie, see src/lifemate/models/constrained.py), so enabling it must validate
    # clean AND must not probe any third-party grammar engine.
    REGISTRY.set_enabled("decoding.constrained", cfg, True)
    assert REGISTRY.validate(cfg) == []
    assert CALLS == [], CALLS
    assert REGISTRY.availability("decoding.constrained")[0] is True
    print("probed")
    """
)

def test_disabled_feature_never_imports_or_probes_its_optional_module():
    """Import hook proves: default config touches no optional dependency."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "probe.py"
        script.write_text(PROBE_SCRIPT, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
            env=env,
        )
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        lines = [line for line in completed.stdout.split() if line in {"clean", "probed"}]
        assert lines == ["clean", "probed"], completed.stdout

def test_resolve_availability_reports_every_feature():
    report = FeatureRegistry().resolve_availability()
    assert set(report) == set(REGISTRY.ids)
    assert all({"available", "reason", "requires"} <= set(item) for item in report.values())
    # No third-party constrained-decoding engine: the feature is available with
    # the stack already installed, which is what its reason must say.
    assert report["decoding.constrained"]["available"] is True
    assert not any(name in report["decoding.constrained"]["reason"]
                   for name in ("outlines", "llguidance", "xgrammar"))
    assert report["audio.rms_normalization"]["available"] is True

# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #
def test_fingerprints_are_16_hex_chars():
    cfg = FeatureConfig()
    for digest in (feature_fingerprint(cfg), training_feature_fingerprint(cfg)):
        assert len(digest) == 16 and all(char in "0123456789abcdef" for char in digest)

def test_fingerprint_changes_when_a_behaviour_flag_flips():
    base = feature_fingerprint(FeatureConfig())
    assert feature_fingerprint(_enable("asr.nbest")) != base
    assert feature_fingerprint(_enable("prompt.itn")) != base
    assert feature_fingerprint(_enable("decoding.constrained")) != base

def test_fingerprint_ignores_vad_padding_and_the_other_tuning_knobs():
    base = feature_fingerprint(FeatureConfig())
    for knob, value in (("vad_padding_ms", 40), ("vad_min_silence_ms", 90)):
        cfg = FeatureConfig()
        setattr(cfg.audio, knob, value)
        assert feature_fingerprint(cfg) == base, knob
    cfg = FeatureConfig()
    cfg.audio.rms_target_dbfs = -18.0
    cfg.training.augmentation_seed = 7
    assert feature_fingerprint(cfg) == base
    assert {"audio.vad_padding_ms", "audio.vad_min_silence_ms", "audio.rms_target_dbfs", "training.augmentation_seed"} == set(
        FINGERPRINT_EXCLUDED
    )

def test_fingerprint_includes_nbest_size_only_while_enabled():
    base = feature_fingerprint(FeatureConfig())
    inert = FeatureConfig()
    inert.asr.nbest_size = 9
    assert feature_fingerprint(inert) == base
    enabled = _enable("asr.nbest", nbest_size=3)
    wider = _enable("asr.nbest", nbest_size=9)
    assert feature_fingerprint(enabled) != feature_fingerprint(wider)
    assert feature_fingerprint(enabled) != base

def test_training_fingerprint_ignores_inference_only_features():
    base = training_feature_fingerprint(FeatureConfig())
    for feature_id in INFERENCE_ONLY:
        cfg = FeatureConfig()
        REGISTRY.set_enabled(feature_id, cfg, True)
        if feature_id == "asr.multi_asr":
            cfg.asr.secondary_model_id = "vinai/PhoWhisper-small"
        assert training_feature_fingerprint(cfg) == base, feature_id
        assert feature_fingerprint(cfg) != feature_fingerprint(FeatureConfig()), feature_id

def test_training_fingerprint_changes_on_training_prompt_and_audio_features():
    base = training_feature_fingerprint(FeatureConfig())
    for feature_id in ("training.slot_masking", "training.tool_dropout", "training.noisy_asr", "prompt.nbest_format", "audio.vad", "audio.rms_normalization", "prompt.itn", "asr.nbest"):
        cfg = FeatureConfig()
        REGISTRY.set_enabled(feature_id, cfg, True)
        assert training_feature_fingerprint(cfg) != base, feature_id

def test_training_fingerprint_matches_the_registry_scopes():
    """The rule is derived from ``scope``, not a hand-written id list."""
    cfg = FeatureConfig()
    for spec in FEATURES:
        REGISTRY.set_enabled(spec.id, cfg, True)
        moved = training_feature_fingerprint(cfg) != training_feature_fingerprint(FeatureConfig())
        assert moved is (spec.scope in {"training", "both"}), (spec.id, spec.scope, moved)
        REGISTRY.set_enabled(spec.id, cfg, False)

def test_training_fingerprint_ignores_tuning_knobs_too():
    base = training_feature_fingerprint(_enable("audio.vad"))
    cfg = _enable("audio.vad", vad_padding_ms=999)
    assert training_feature_fingerprint(cfg) == base

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def test_report_contains_every_label_and_says_baseline():
    text = format_feature_report(
        FeatureConfig(),
        model_id="openbmb/MiniCPM5-2B",
        asr_id="g-group-ai-lab/gipformer-68M-rnnt",
    )
    for label in (
        "LifeMate Effective Configuration",
        "Model:",
        "openbmb/MiniCPM5-2B",
        "ASR:",
        "g-group-ai-lab/gipformer-68M-rnnt",
        "Enabled features:",
        "Disabled features:",
        "Mode:",
    ):
        assert label in text, label
    lines = [line.strip() for line in text.splitlines()]
    assert "BASELINE" in lines
    assert "NONE" in lines
    assert "Mode:\n  BASELINE" in text
    for spec in FEATURES:
        assert f"  {spec.id}" in text, spec.id
    assert text.startswith("===============================\nLifeMate Effective Configuration")

def test_report_switches_to_experimental_and_lists_the_flag():
    text = format_feature_report(
        _enable("asr.nbest"),
        model_id="openbmb/MiniCPM5-2B",
        asr_id="g-group-ai-lab/gipformer-68M-rnnt",
    )
    assert "Mode:\n  EXPERIMENTAL" in text
    enabled_block = text.split("Enabled features:", 1)[1].split("Disabled features:", 1)[0]
    assert "asr.nbest" in enabled_block
    disabled_block = text.split("Disabled features:", 1)[1]
    assert "asr.nbest" not in disabled_block

def test_report_dict_is_json_ready_and_carries_the_fingerprints():
    cfg, provenance = apply_overrides(FeatureConfig(), cli={"asr.nbest": True})
    payload = feature_report_dict(cfg, model_id="openbmb/MiniCPM5-2B", asr_id="g-group-ai-lab/gipformer-68M-rnnt")
    assert payload["model"] == "openbmb/MiniCPM5-2B"
    assert payload["asr"] == "g-group-ai-lab/gipformer-68M-rnnt"
    assert payload["enabled"] == ["asr.nbest"]
    assert payload["mode"] == "EXPERIMENTAL"
    assert payload["fingerprint"] == feature_fingerprint(cfg)
    assert payload["training_fingerprint"] == training_feature_fingerprint(cfg)
    assert set(payload["disabled"]) == set(REGISTRY.ids) - {"asr.nbest"}
    assert payload["provenance"]["asr.nbest"] == "cli"
    assert payload["provenance"]["prompt.itn"] == "default"
    assert set(payload["provenance"]) == set(REGISTRY.ids)
    json.dumps(payload)

def test_report_dict_baseline_mode_when_nothing_is_enabled():
    payload = feature_report_dict(FeatureConfig(), model_id="m", asr_id="a")
    assert payload["enabled"] == [] and payload["mode"] == "BASELINE"

def test_module_is_importable_without_importing_optional_backends():
    """The package itself must stay torch/outlines-free (gate B imports it)."""
    code = (
        "import sys, lifemate.features\n"
        "banned = {'torch','transformers','outlines','onnxruntime','numpy'} & set(sys.modules)\n"
        "print(sorted(banned))\n"
        "sys.exit(1 if banned else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
        env=env,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)


# --------------------------------------------------------------------------- #
# scripts/feature_ablation.py (gate F23)
# --------------------------------------------------------------------------- #
ABLATION_SCRIPT = ROOT / "scripts" / "feature_ablation.py"

def _ablation_module():
    """Import ``scripts/feature_ablation.py`` without running ``main``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_feature_ablation_probe", ABLATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _run_ablation(argv: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["WANDB_BASE_URL"] = "http://127.0.0.1:9"
    return subprocess.run(
        [sys.executable, str(ABLATION_SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
        env=env,
    )

def test_ablation_dry_run_executes_nothing_and_prints_the_commands():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ablation.json"
        completed = _run_ablation(
            [
                "--arms",
                "configs/experiments/baseline.yaml",
                "configs/experiments/asr_nbest.yaml",
                "--limit",
                "20",
                "--out",
                str(out),
                "--dry-run",
            ]
        )
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert "DRY-RUN" in completed.stdout
        assert "dry run: nothing was executed." in completed.stdout
        # The exact command is printed, one per arm, and no arm output dir exists.
        printed = [
            line for line in completed.stdout.splitlines() if line.strip().startswith("command:")
        ]
        assert len(printed) == 2, printed
        assert all("--config" in line for line in printed)
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["ablation.json"], list(Path(tmp).iterdir())
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        for arm in payload["arms"]:
            assert arm["run_status"] == "DRY-RUN"
            assert arm["metrics"] is None
            assert "returncode" not in arm
            assert not Path(arm["output_dir"]).exists(), arm["output_dir"]

def test_ablation_dry_run_names_the_constants_it_holds_fixed():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ablation.json"
        completed = _run_ablation(
            ["--arms", "configs/experiments/baseline.yaml", "--out", str(out), "--dry-run"]
        )
        assert completed.returncode == 0, completed.stderr
        for label in (
            "dataset path",
            "split",
            "model id",
            "checkpoint",
            "seed",
            "agent.max_steps",
            "time budget (s)",
            "generation temperature",
            "generation token budget",
        ):
            assert label in completed.stdout, label
        fixed = json.loads(out.read_text(encoding="utf-8"))["fixed"]
        assert fixed["model id"] == "openbmb/MiniCPM5-2B"
        assert fixed["ASR model"] == "g-group-ai-lab/gipformer-68M-rnnt"
        assert fixed["agent.max_steps"] == 8
        assert fixed["seed"] == 42

def test_ablation_refuses_an_invalid_arm_before_running_anything():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.yaml"
        bad.write_text("features:\n  asr:\n    nbest_enabled: true\n    nbest_size: 1\n", encoding="utf-8")
        out = Path(tmp) / "ablation.json"
        completed = _run_ablation(
            ["--arms", "configs/experiments/baseline.yaml", str(bad), "--out", str(out), "--dry-run"]
        )
        assert completed.returncode != 0
        assert "refusing to start" in completed.stderr
        assert "nbest_size" in completed.stderr
        assert str(bad) in completed.stderr
        assert not out.exists(), "an invalid arm must not produce a report"

def test_ablation_refuses_a_typo_in_an_arm():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "typo.yaml"
        bad.write_text("features:\n  asr:\n    nbest_enable: true\n", encoding="utf-8")
        completed = _run_ablation(["--arms", str(bad), "--out", str(Path(tmp) / "x.json"), "--dry-run"])
        assert completed.returncode != 0
        assert "nbest_enable" in completed.stderr
        assert "refusing to start" in completed.stderr

def test_ablation_marks_mock_results_as_not_competition_scores():
    module = _ablation_module()
    with tempfile.TemporaryDirectory() as tmp:
        report = module.build_report(
            [ROOT / "configs" / "experiments" / "baseline.yaml"],
            split="train",
            data_root=str(ROOT / "tests" / "fixtures" / "synthetic_lifemate"),
            limit=1,
            out=Path(tmp) / "ablation.json",
            mock_env=True,
            dry_run=True,
        )
    assert report["mock_env"] is True
    assert report["competition_score"] is False
    assert "NOT competition scores" in report["mock_note"]
    assert all(arm["mock_env"] is True for arm in report["arms"])

def test_ablation_deltas_are_zero_for_a_self_comparison():
    module = _ablation_module()
    reference = {"tool_call_exact_match": 0.25, "sequence_f1": 0.5, "final_state_correctness": {"available": False}}
    assert module._deltas(reference, dict(reference)) == {
        "tool_call_exact_match": 0.0,
        "tool_name_accuracy": "UNAVAILABLE",
        "argument_accuracy": "UNAVAILABLE",
        "sequence_f1": 0.0,
        "final_state_correctness": "UNAVAILABLE",
    }
    worse = {"tool_call_exact_match": 0.15, "sequence_f1": 0.5, "final_state_correctness": {"available": False}}
    deltas = module._deltas(reference, worse)
    assert deltas["tool_call_exact_match"] == -0.1
    assert deltas["sequence_f1"] == 0.0
    assert deltas["final_state_correctness"] == "UNAVAILABLE"
    # a number vs an unavailable metric is incomparable, never a fake 0 difference
    only_one = module._deltas(reference, {"tool_call_exact_match": {"available": False}})
    assert only_one["tool_call_exact_match"] == "UNAVAILABLE"
def test_ablation_blocks_rather_than_inventing_metrics_when_the_output_is_missing():
    module = _ablation_module()
    missing = "nope"
    assert module._metrics_from({}) == {
        key: {"available": False, "reason": "metric absent from this run"}
        for key in module.COMPARED_METRICS
    }
    del missing
    assert module._metrics_from({"metrics": {"sequence_f1": 0.5}})["sequence_f1"] == 0.5
