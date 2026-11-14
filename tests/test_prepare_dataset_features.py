"""`prepare_dataset.py` feature wiring: cfg.features authoritative, aliases kept.

Stdlib only (subprocess against the script, real fixtures + a temp transcript
file). The pinned digests are the ones already proven for the all-off build.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_dataset.py"
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"
DATA = ROOT / "data"

#: The all-off build of ``tests/fixtures/synthetic_lifemate`` (proven earlier).
PINNED_ALL_OFF = {
    "train.jsonl": "64b3bdf7fc252a04f415746f951c69ae395d40f9fdebd38a9b503ae94ca85d9f",
    "val.jsonl": "b050077ecf2dd2ba82951c05296faa1a3f1d9a7cf6d2fe8c65d4a0e610d98f65",
    "manifest.json": "a70536eb9f6c1dd6a2aa4592724a4265dd7375109afba6b05c19898e493d34f5",
    "stats.json": "8b1e32de4714f1bc68355117be558d948b3031b4bdae3c714f6f689d45098f86",
}

def _run(out: Path, *extra: str) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(SCRIPT),
        "--data-root",
        str(FIXTURES),
        "--split",
        "train",
        "--output",
        str(out),
        *extra,
    ]
    return subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT), timeout=300)

def _digests(out: Path) -> dict[str, str]:
    split = out / "train"
    result = {}
    for name in ("train.jsonl", "val.jsonl", "manifest.json", "stats.json"):
        content = (split / name).read_bytes()
        if name == "manifest.json":
            # The manifest intentionally records the absolute source path. Keep
            # checking that path, then normalize it for the pinned digest so
            # the same semantic manifest can be checked in any worktree.
            manifest = json.loads(content)
            assert manifest["data_root"] == str(FIXTURES.resolve())
            manifest["data_root"] = "<FIXTURE_ROOT>"
            content = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
        result[name] = hashlib.sha256(content).hexdigest()
    return result

def _transcripts(tmp: Path, sample_ids: list[str], ratio_texts: tuple[str, str]) -> Path:
    """A 2-hypothesis transcript file for ``sample_ids`` (best first)."""
    path = tmp / "transcripts.jsonl"
    rows = [
        {
            "sample_id": sample_id,
            "hypotheses": [
                {"text": ratio_texts[0], "rank": 0, "source": "asr"},
                {"text": f"{ratio_texts[1]} {sample_id}", "rank": 1, "source": "asr:alt"},
            ],
            "asr_model": "gipformer-68M",
            "asr_revision": "rev1",
            "asr_config": {"backend": "gipformer"},
        }
        for sample_id in sample_ids
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path

def _fixture_ids() -> list[str]:
    rows = [json.loads(line) for line in (FIXTURES / "train" / "train.jsonl").read_text().splitlines() if line]
    return [str(row["id"]) for row in rows]

def _train_records(out: Path) -> list[dict]:
    path = out / "train" / "train.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]

# --------------------------------------------------------------------------- #
# All-off: byte identity and no feature keys
# --------------------------------------------------------------------------- #
def test_every_flag_off_matches_the_pinned_digests_and_has_no_feature_keys():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "off"
        probe = _run(out)
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        assert _digests(out) == PINNED_ALL_OFF, _digests(out)
        manifest = json.loads((out / "train" / "manifest.json").read_text())
        assert "augmentation" not in manifest
        assert "features" not in manifest
        assert "training_feature_fingerprint" not in manifest
        assert "augmentation set    : NONE" in probe.stdout

# --------------------------------------------------------------------------- #
# Feature path vs alias path: identical dataset bytes
# --------------------------------------------------------------------------- #
def test_feature_flag_and_deprecated_alias_produce_identical_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("nhiễu một", "nhiễu hai"))
        feature_run = base / "feature"
        alias_run = base / "alias"
        common = ("--noisy-asr-transcripts", str(transcripts), "--noisy-asr-ratio", "1.0")
        first = _run(feature_run, "--enable-feature", "training.noisy_asr", *common)
        second = _run(alias_run, "--noisy-asr-enabled", *common)
        assert first.returncode == 0, (first.stdout, first.stderr)
        assert second.returncode == 0, (second.stdout, second.stderr)
        assert _digests(feature_run) == _digests(alias_run)
        variants = [r for r in _train_records(feature_run) if r["meta"].get("augmentation") == "noisy_asr"]
        assert variants, "the feature flag alone must produce noisy-ASR variants"
        assert "deprecated alias" in second.stdout
        assert "deprecated alias" not in first.stdout
        manifest = json.loads((feature_run / "train" / "manifest.json").read_text())
        assert "training.noisy_asr" in manifest["features"]["enabled"]
        assert manifest["augmentation"]["n_variants"] == len(variants)

def test_feature_flag_alone_needs_no_bespoke_boolean():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("a", "b"))
        out = base / "out"
        probe = _run(
            out,
            "--enable-feature",
            "training.slot_masking",
            "--enable-feature",
            "training.tool_dropout",
        )
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        kinds = {
            record["meta"].get("augmentation")
            for record in _train_records(out)
            if record["meta"].get("augmentation")
        }
        assert "tool_dropout" in kinds, kinds
        assert "deprecated alias" not in probe.stdout
        # And the transcripts flag alone changes nothing (unused here).
        assert transcripts.is_file()

def test_yaml_experiment_enables_the_feature_with_no_feature_flag():
    """`--config configs/experiments/noisy_asr_training.yaml` alone must bite."""
    overlay = ROOT / "configs" / "experiments" / "noisy_asr_training.yaml"
    if not overlay.is_file():  # pragma: no cover - overlay deleted
        return
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("yaml một", "yaml hai"))
        out = base / "yaml"
        probe = _run(
            out,
            "--config",
            str(overlay),
            "--noisy-asr-transcripts",
            str(transcripts),
            "--noisy-asr-ratio",
            "1.0",
        )
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        variants = [r for r in _train_records(out) if r["meta"].get("augmentation") == "noisy_asr"]
        assert variants, "a YAML features: block must enable the augmentation"
        manifest = json.loads((out / "train" / "manifest.json").read_text())
        assert "training.noisy_asr" in manifest["features"]["enabled"]
        assert manifest["augmentation"]["noisy_asr_ratio"] == 1.0

# --------------------------------------------------------------------------- #
# Validation happens before any dataset work
# --------------------------------------------------------------------------- #
def test_ratio_out_of_range_fails_before_any_dataset_work():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("a", "b"))
        out = base / "invalid"
        probe = _run(
            out,
            "--enable-feature",
            "training.noisy_asr",
            "--noisy-asr-ratio",
            "1.5",
            "--noisy-asr-transcripts",
            str(transcripts),
        )
        assert probe.returncode != 0
        assert "outside (0, 1]" in probe.stdout + probe.stderr
        assert not (out / "train" / "train.jsonl").exists(), "no dataset may be written"

def test_nbest_format_without_asr_nbest_names_both_ids():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("a", "b"))
        out = base / "nbest"
        probe = _run(out, "--nbest-training-format", "--nbest-transcripts", str(transcripts))
        assert probe.returncode != 0
        combined = probe.stdout + probe.stderr
        assert "prompt.nbest_format" in combined
        assert "asr.nbest" in combined
        assert not (out / "train" / "train.jsonl").exists()

# --------------------------------------------------------------------------- #
# Inert numeric parameter
# --------------------------------------------------------------------------- #
def test_yaml_ratio_with_every_flag_off_is_inert():
    """`noisy_asr_ratio: 0.25` in YAML with the feature off changes nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        out = base / "inert"
        probe = _run(out, "--disable-feature", "training.noisy_asr")
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        assert _digests(out) == PINNED_ALL_OFF
        manifest = json.loads((out / "train" / "manifest.json").read_text())
        assert "augmentation" not in manifest
        assert "training.noisy_asr" not in manifest.get("features", {}).get("enabled", [])
        assert "augmentation set    : NONE" in probe.stdout

def test_yaml_ratio_with_the_flag_off_is_inert():
    """`features.training.noisy_asr_ratio: 0.5` with the flag off: no variants, no error."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        overlay = base / "ratio_only.yaml"
        overlay.write_text(
            "features:\n  training:\n    noisy_asr_ratio: 0.5\n", encoding="utf-8"
        )
        out = base / "inert"
        probe = _run(out, "--config", str(overlay))
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        assert _digests(out) == PINNED_ALL_OFF
        manifest = json.loads((out / "train" / "manifest.json").read_text())
        assert "augmentation" not in manifest

def test_disable_feature_wins_over_yaml():
    overlay = ROOT / "configs" / "experiments" / "noisy_asr_training.yaml"
    if not overlay.is_file():  # pragma: no cover - overlay deleted
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "disabled"
        probe = _run(out, "--config", str(overlay), "--disable-feature", "training.noisy_asr")
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        manifest = json.loads((out / "train" / "manifest.json").read_text())
        assert "augmentation" not in manifest
        assert "training.noisy_asr" not in manifest.get("features", {}).get("enabled", [])

# --------------------------------------------------------------------------- #
# Manifest provenance
# --------------------------------------------------------------------------- #
def test_manifest_records_the_feature_report_and_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("một", "hai"))
        out = base / "out"
        probe = _run(
            out,
            "--enable-feature",
            "training.noisy_asr",
            "--noisy-asr-transcripts",
            str(transcripts),
            "--noisy-asr-ratio",
            "1.0",
        )
        assert probe.returncode == 0, (probe.stdout, probe.stderr)
        manifest = json.loads((out / "train" / "manifest.json").read_text())
        report = manifest["features"]
        assert report["mode"] == "EXPERIMENTAL"
        assert report["asr"] and report["model"]
        assert isinstance(manifest["training_feature_fingerprint"], str)
        assert len(manifest["training_feature_fingerprint"]) == 16
        assert report["training_fingerprint"] == manifest["training_feature_fingerprint"]
        assert manifest["augmentation"]["val_augmented"] is False
        val = [json.loads(line) for line in (out / "train" / "val.jsonl").read_text().splitlines() if line]
        assert all(not record["meta"].get("augmentation") for record in val)

def test_real_split_is_not_required_but_available():
    """The real train split exists; this test only pins the data root's presence."""
    assert (DATA / "train" / "train.jsonl").is_file()

def test_prepare_dataset_noisy_asr_mode_replace_vs_augment():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        transcripts = _transcripts(base, _fixture_ids(), ("một", "hai"))
        
        out_rep = base / "out_replace"
        probe_rep = _run(
            out_rep,
            "--enable-feature",
            "training.noisy_asr",
            "--noisy-asr-transcripts",
            str(transcripts),
            "--noisy-asr-ratio",
            "1.0",
            "--noisy-asr-mode",
            "replace",
        )
        assert probe_rep.returncode == 0, (probe_rep.stdout, probe_rep.stderr)
        records_rep = _train_records(out_rep)
        manifest_rep = json.loads((out_rep / "train" / "manifest.json").read_text())
        assert manifest_rep["augmentation"]["noisy_asr_mode"] == "replace"
        assert manifest_rep["augmentation"]["reports"]["noisy_asr"]["augmentation_mode"] == "replace"
        
        out_aug = base / "out_augment"
        probe_aug = _run(
            out_aug,
            "--enable-feature",
            "training.noisy_asr",
            "--noisy-asr-transcripts",
            str(transcripts),
            "--noisy-asr-ratio",
            "1.0",
            "--noisy-asr-mode",
            "augment",
        )
        assert probe_aug.returncode == 0, (probe_aug.stdout, probe_aug.stderr)
        records_aug = _train_records(out_aug)
        manifest_aug = json.loads((out_aug / "train" / "manifest.json").read_text())
        assert manifest_aug["augmentation"]["noisy_asr_mode"] == "augment"
        
        # Replace keeps original record count; augment appends variants
        assert len(records_rep) < len(records_aug)
        assert len(records_aug) == len(records_rep) * 2
