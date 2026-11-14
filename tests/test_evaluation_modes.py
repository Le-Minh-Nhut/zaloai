"""Tests for ASR feature ablation evaluation modes, transcript caching, and fairness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from lifemate.config import load_config
from lifemate.errors import FeatureConfigurationError
from lifemate.evaluation import evaluator
from lifemate.evaluation.transcripts import load_cache, save_cache, transcript_cache_key
from lifemate.features.config import FeatureConfig

ROOT = Path(__file__).resolve().parents[1]


class _DummySample:
    def __init__(self, sample_id: str, text: str = "xin chào") -> None:
        self.id = sample_id
        self.audio_text = text


def test_transcript_cache_key_deterministic():
    cfg = load_config()
    feats = FeatureConfig()
    key1 = transcript_cache_key(cfg, feats, split="val", sample_ids=["s1", "s2"])
    key2 = transcript_cache_key(cfg, feats, split="val", sample_ids=["s2", "s1"])
    assert key1 == key2
    assert len(key1) == 16

    # Different split changes key
    key3 = transcript_cache_key(cfg, feats, split="train", sample_ids=["s1", "s2"])
    assert key1 != key3

    # Enabled feature changes key
    feats.audio.vad_enabled = True
    key4 = transcript_cache_key(cfg, feats, split="val", sample_ids=["s1", "s2"])
    assert key1 != key4


def test_transcript_cache_load_and_save():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "cache.json"
        transcripts = {"s1": "xin chào", "s2": "tạm biệt"}
        key = "abcdef0123456789"
        save_cache(cache_path, key, transcripts)
        loaded = load_cache(cache_path)
        assert key in loaded
        assert loaded[key] == transcripts


def test_evaluate_split_text_to_action_mode():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "eval"
        payload = evaluator.evaluate_split(
            cfg=load_config(),
            split="train",
            output_dir=directory,
            evaluation_mode="text_to_action",
            teacher_forced={
                "n_records": 1,
                "metrics": {"tool_call_exact_match": 1.0},
            },
            data_root=ROOT / "tests" / "fixtures" / "synthetic_lifemate",
            mock_env=True,
        )
        assert payload["evaluation_mode"] == "text_to_action"
        assert payload["provenance"]["evaluation_mode"] == "text_to_action"
        # Notes confirm LLM evaluated on gold/fixed transcripts
        assert any("text_to_action" in note for note in payload["comparison_notes"])
        assert any("not an ASR evaluation" in note for note in payload["comparison_notes"])


def test_evaluate_split_audio_to_action_mode():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "eval"
        payload = evaluator.evaluate_split(
            cfg=load_config(),
            split="train",
            output_dir=directory,
            evaluation_mode="audio_to_action",
            closed_loop={
                "n_run": 2,
                "status_counts": {"finished": 2},
                "evaluation_mode": "audio_to_action",
                "decision_metrics": {"tool_call_exact_match": 0.5},
            },
            data_root=ROOT / "tests" / "fixtures" / "synthetic_lifemate",
            mock_env=True,
        )
        assert payload["evaluation_mode"] == "audio_to_action"
        assert payload["provenance"]["evaluation_mode"] == "audio_to_action"
        assert any("audio_to_action" in note for note in payload["comparison_notes"])


def test_missing_metric_reports_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "eval"
        payload = evaluator.evaluate_split(
            cfg=load_config(),
            split="train",
            output_dir=directory,
            closed_loop={
                "n_run": 1,
                "status_counts": {"finished": 1},
                "final_state_correctness": {"available": False, "reason": "tooling absent"},
            },
            data_root=ROOT / "tests" / "fixtures" / "synthetic_lifemate",
            mock_env=True,
        )
        md_text = (directory / "metrics.md").read_text(encoding="utf-8")
        assert "UNAVAILABLE" in md_text
        # final_state_correctness value is UNAVAILABLE in markdown
        table_lines = [line for line in md_text.splitlines() if line.startswith("| final_state_correctness")]
        assert len(table_lines) == 1
        assert "UNAVAILABLE" in table_lines[0]


def test_ablation_fairness_rejects_mismatch():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lifemate_ablation_mod", ROOT / "scripts" / "feature_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        arm1 = Path(tmp) / "arm1.yaml"
        arm2 = Path(tmp) / "arm2.yaml"
        arm1.write_text("model:\n  name: openbmb/MiniCPM5-2B\ntraining:\n  seed: 42\n", encoding="utf-8")
        arm2.write_text("model:\n  name: different/Model\ntraining:\n  seed: 42\n", encoding="utf-8")

        # Mismatched model.name without declared variable must raise
        with pytest.raises(FeatureConfigurationError) as exc_info:
            module._validate_fairness([arm1, arm2])
        assert "Ablation fairness violation" in str(exc_info.value)
        assert "model id" in str(exc_info.value)

        # Explicitly declaring the experimental variable allows it
        module._validate_fairness([arm1, arm2], experimental_variables=["model.name"])
        module._validate_fairness([arm1, arm2], experimental_variables=["model"])


def test_ablation_per_arm_report_keys():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lifemate_ablation_mod", ROOT / "scripts" / "feature_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        arm = ROOT / "configs" / "experiments" / "baseline.yaml"
        out = Path(tmp) / "report.json"
        report = module.build_report(
            [arm],
            split="train",
            data_root=str(ROOT / "tests" / "fixtures" / "synthetic_lifemate"),
            limit=5,
            out=out,
            mock_env=True,
            dry_run=True,
            evaluation_mode="text_to_action",
        )
        arm_entry = report["arms"][0]
        required_keys = (
            "model",
            "checkpoint",
            "dataset",
            "n_samples",
            "n_successful",
            "n_failed",
            "enabled_features",
            "effective_features",
            "degraded_features",
            "evaluation_mode",
            "metrics",
        )
        for key in required_keys:
            assert key in arm_entry, f"missing {key} in arm report"
        assert arm_entry["evaluation_mode"] == "text_to_action"
