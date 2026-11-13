"""EDA tests. Stdlib-only: matplotlib/numpy are absent in CI and must not matter."""

from __future__ import annotations

import json
import tempfile
import wave
from pathlib import Path

from lifemate.data.catalog import ToolCatalog
from lifemate.data.discovery import discover_splits
from lifemate.data.loader import load_catalog, load_raw_split, load_samples
from lifemate.data.schemas import Context, Label, Sample, ToolCall, ToolResult
from lifemate.eda import (
    build_analysis_bundle,
    build_report,
    describe,
    make_charts,
    matplotlib_available,
)
from lifemate.eda.audio_analysis import analyze_split_audio, analyze_wav
from lifemate.eda.report import SYNTHETIC_BANNER
from lifemate.eda.tool_analysis import (
    analyze_catalog,
    render_dependency_graph_ascii,
    render_dependency_graph_mermaid,
)
from lifemate.eda.trajectory_analysis import alignment_report, analyze_trajectories

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic_lifemate"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _train_samples(limit: int | None = None):
    return load_samples(FIXTURES, "train", limit=limit)


def _train_catalog():
    return load_catalog(FIXTURES, "train")


def _write_wav(path: Path, samples, rate: int = 16000, channels: int = 1, width: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        frames = bytearray()
        for value in samples:
            for _ in range(channels):
                frames += int(value).to_bytes(width, "little", signed=True)
        wf.writeframes(bytes(frames))
    return path


def _silent_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    return _write_wav(path, [0] * int(seconds * rate), rate=rate)


def _make_sample(sid, calls=None, results=None, text="Dat bao thuc", apps=None, history=None):
    return Sample(
        id=sid,
        split="train",
        audio_path=None,
        audio_text=text,
        context=Context(installed_apps=list(apps or ["alarm"]), current_time="2026-11-07T09:30:00+07:00"),
        history=list(history or []),
        label=Label(tool_calls=list(calls or []), tool_results=list(results or [])),
    )


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------
def test_describe_empty_and_single():
    empty = describe([])
    assert empty["count"] == 0
    assert all(empty[k] is None for k in ("mean", "median", "stdev", "min", "max", "p95"))

    one = describe([7])
    assert one["count"] == 1
    assert one["mean"] == 7 and one["median"] == 7
    assert one["stdev"] == 0.0
    assert one["min"] == one["max"] == 7
    assert one["p5"] == 7 and one["p99"] == 7


def test_describe_ignores_non_finite_and_non_numeric():
    stats = describe([1, 2, 3, None, "x", float("nan"), float("inf")])
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["p95"] is not None


def test_describe_percentiles_monotone():
    stats = describe(list(range(1, 101)))
    assert stats["p5"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"] <= stats["p99"]


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------
def test_analyze_dataset_on_synthetic_samples():
    samples = _train_samples()
    catalog = _train_catalog()
    raw = load_raw_split(FIXTURES, "train")
    bundle = build_analysis_bundle(samples, raw_records=raw, catalog=catalog, synthetic=True)
    ds = bundle["dataset"]

    assert ds["n_samples"] == len(samples) == 14
    assert ds["n_duplicate_ids"] == 0
    assert ds["n_with_labels"] == 14
    assert ds["history_length"]["stats"]["max"] >= 14  # long_history fixture
    assert ds["input_text"]["words"]["stats"]["count"] == 14
    assert ds["installed_apps"]["n_distinct_apps"] >= 4
    assert ds["catalog"]["n_tools"] == len(catalog.tools)
    # Context time parses (all fixtures use ISO with +07:00).
    # 13 of 14 samples carry current_time; the one without is a *missing* field,
    # not a parse failure, so failures stay at 0.
    assert ds["context_time"]["n_present"] == 13
    assert ds["context_time"]["n_parsed"] == 13
    assert ds["context_time"]["n_parse_failures"] == 0
    assert ds["context_time"]["hour_histogram"]["09"] == 13
    assert ds["missing_fields"]["context.current_time"] == 1


def test_dataset_never_leaks_profile_values():
    samples = _train_samples()
    result = __import__("lifemate.eda.dataset_analysis", fromlist=["analyze_dataset"]).analyze_dataset(samples)
    blob = json.dumps(result, ensure_ascii=False)
    assert "Test User" not in blob
    assert "+84900000000" not in blob
    # ...but key names are reported.
    keys = {row["key"] for row in result["user_profile"]["key_frequency"]}
    assert {"name", "phone"} <= keys


def test_duplicate_ids_detected():
    samples = _train_samples()
    samples.append(samples[0])
    result = __import__("lifemate.eda.dataset_analysis", fromlist=["analyze_dataset"]).analyze_dataset(samples)
    assert result["n_samples"] == 15
    assert result["n_duplicate_ids"] == 1
    assert result["duplicate_ids"][samples[0].id] == 2


def test_missing_fields_tolerated():
    bare = Sample(
        id="bare",
        split="train",
        audio_path=None,
        audio_text="",
        context=Context(),  # everything empty
        history=[],
        label=None,
    )
    result = __import__("lifemate.eda.dataset_analysis", fromlist=["analyze_dataset"]).analyze_dataset([bare])
    assert result["n_samples"] == 1
    assert result["n_with_labels"] == 0
    assert result["missing_fields"]["context.current_time"] == 1
    assert result["missing_fields"]["label"] == 1
    assert result["history_length"]["zero_turn_samples"] == 1


def test_tokenizer_absent_marker_and_injected_tokenizer():
    module = __import__("lifemate.eda.dataset_analysis", fromlist=["analyze_dataset"])
    samples = _train_samples()
    assert module.analyze_dataset(samples)["tokens"]["tokenizer"] == "unavailable"

    injected = module.analyze_dataset(samples, tokenizer=lambda text: text.split())
    assert injected["tokens"]["tokenizer"] == "available"
    assert injected["tokens"]["stats"]["count"] == 14

def test_hf_style_tokenizer_uses_encode_not_call():
    """A real HF tokenizer is callable *and* has ``.encode``; ``len()`` of a call is 2.

    Measured on ``Qwen/Qwen3.5-4B``: ``len(tok.encode(text))`` is 11..19 for real
    requests while ``len(tok(text))`` is 2 (number of BatchEncoding fields), so
    the wrong branch reports a constant 2 tokens for every sample.
    """
    module = __import__("lifemate.eda.dataset_analysis", fromlist=["analyze_dataset"])

    class FakeEncoding:
        def __len__(self) -> int:
            return 2  # input_ids + attention_mask: what the old adapter measured

    class HFTokenizer:
        name_or_path = "fake-hf"

        def encode(self, text):  # noqa: D401 - duck-typed HF contract
            return [0] * max(1, len(text.split()))

        def __call__(self, text):  # HF tokenizers are callable too
            return FakeEncoding()

    result = module.analyze_dataset(_train_samples(), tokenizer=HFTokenizer())
    stats = result["tokens"]["stats"]
    assert result["tokens"]["tokenizer"] == "available"
    assert stats["min"] > 2, stats
    assert stats["max"] > stats["min"], stats


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------
def test_analyze_wav_sane_values():
    sample = _train_samples()[0]
    result = analyze_wav(sample.audio_path)
    assert result["ok"] is True
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["sample_width_bits"] == 16
    assert 0.4 < result["duration_s"] < 10
    assert result["peak_amplitude"] > 0
    assert result["silence_threshold"] == 500
    assert 0.0 <= result["silence_ratio"] <= 1.0
    assert result["flags"] == [] or "unexpected_sample_rate" not in result["flags"]


def test_corrupt_file_flagged_not_crashing():
    result = analyze_wav(FIXTURES / "broken" / "zero_length.wav")
    assert result["ok"] is False
    assert result["error"]
    assert "corrupt" in result["flags"]
    assert result["duration_s"] is None

    missing = analyze_wav(FIXTURES / "does_not_exist.wav")
    assert missing["ok"] is False


def test_silence_detection_on_generated_wav():
    with tempfile.TemporaryDirectory() as tmp:
        path = _silent_wav(Path(tmp) / "silent.wav", seconds=1.0)
        result = analyze_wav(path)
        assert result["ok"] is True
        assert result["silence_ratio"] == 1.0
        assert result["rms"] == 0.0
        assert "silent" in result["flags"]

        loud = _write_wav(Path(tmp) / "loud.wav", [12000] * 16000)
        loud_result = analyze_wav(loud)
        assert loud_result["silence_ratio"] == 0.0
        assert "silent" not in loud_result["flags"]


def test_clipping_flag_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_wav(Path(tmp) / "clip.wav", [32767, -32768] * 8000)
        result = analyze_wav(path)
        assert "clipping" in result["flags"]
        assert result["clipping_ratio"] > 0.9


def test_very_short_and_unexpected_rate_flags():
    with tempfile.TemporaryDirectory() as tmp:
        short = _write_wav(Path(tmp) / "short.wav", [1000] * 100, rate=8000)
        result = analyze_wav(short)
        assert "very_short" in result["flags"]
        assert "unexpected_sample_rate" in result["flags"]


def test_analyze_split_audio_aggregate():
    samples = _train_samples()
    result = analyze_split_audio(samples, synthetic=True)
    assert result["synthetic"] is True
    assert result["n_files_analyzed"] == 14
    assert result["n_files_ok"] == 14
    assert result["duration_stats"]["count"] == 14
    assert result["sample_rate_distribution"] == {"16000": 14}
    assert result["total_duration_seconds"] > 0
    corr = result["duration_vs_transcript_words"]
    assert corr["available"] is True and -1.0 <= corr["r"] <= 1.0


def test_analyze_split_audio_survives_bad_file():
    bad = _make_sample("bad")
    bad.audio_path = Path("/nonexistent/nope.wav")
    result = analyze_split_audio([bad], synthetic=True)
    assert result["n_files_failed"] == 1
    assert result["flag_counts"].get("corrupt") == 1


def test_pearson_needs_three_points():
    from lifemate.eda.audio_analysis import pearson

    assert pearson([1, 2], [1, 2])["available"] is False
    exact = pearson([1, 2, 3], [2, 4, 6])
    assert exact["available"] is True and abs(exact["r"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Catalog analysis
# ---------------------------------------------------------------------------
def test_analyze_catalog_structure():
    catalog = _train_catalog()
    result = analyze_catalog(catalog)
    assert result["available"] is True
    assert result["n_tools"] == len(catalog.tools) == 13
    assert result["n_apps"] == 5
    assert sum(result["kind_distribution"].values()) == 13
    assert result["kind_distribution"].get("meta") == 1
    assert result["x_ref"]["n_refs"] >= 1
    assert result["has_meta_tool"] is True
    assert len(result["table_rows"]) == 13
    assert result["dependency_graph"]["n_edges"] >= 1


def test_analyze_catalog_absent_is_reported_not_raised():
    result = analyze_catalog(None)
    assert result["available"] is False
    assert result["n_tools"] == 0


def test_dependency_graph_renders_both_tool_names():
    edges = catalog_edges = _train_catalog().dependency_edges()
    source, target, field = catalog_edges[0]
    mermaid = render_dependency_graph_mermaid(edges)
    ascii_art = render_dependency_graph_ascii(edges)
    assert mermaid.startswith("```mermaid")
    assert "graph TD" in mermaid and mermaid.rstrip().endswith("```")
    assert source in mermaid and target in mermaid
    assert source in ascii_art and target in ascii_art
    assert field.split(" ")[0] in mermaid


def test_dependency_graph_empty_is_valid():
    mermaid = render_dependency_graph_mermaid([])
    assert "no_dependencies" in mermaid
    assert "no x-ref" in render_dependency_graph_ascii([])


def test_catalog_from_dict_minimal():
    catalog = ToolCatalog.from_dict(
        {
            "apps": [
                {
                    "app_name": "notes",
                    "tools": [
                        {
                            "name": "notes.create",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["title"],
                            },
                        }
                    ],
                }
            ]
        }
    )
    result = analyze_catalog(catalog)
    assert result["n_tools"] == 1
    assert result["args_per_tool"]["required"] == {1: 1}
    assert result["parameter_type_frequency"]["string"] == 1
    assert result["parameter_type_frequency"]["array"] == 1


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------
def test_trajectory_counts_and_flags():
    result = analyze_trajectories(_train_samples(), catalog=_train_catalog())
    assert result["zero_calls"] == 2
    assert result["single_call"] == 7
    assert result["multi_call"] == 5
    assert result["missing_info"]["count"] == 1
    assert result["out_of_scope"]["count"] == 2
    assert "train_out_of_scope" in result["out_of_scope"]["ids"]
    assert result["failed_tool_results"]["count"] >= 1
    assert result["transitions"]["2"]["n_distinct"] >= 1
    assert result["top_chains"]
    assert result["challenges"]["counts"]["missing_information"] == 1
    assert result["challenges"]["counts"]["long_history"] == 1


def test_read_before_write_uses_history():
    result = analyze_trajectories(_train_samples(), catalog=_train_catalog())
    rbw = result["read_before_write"]
    assert rbw["samples_with_write"] >= 1
    assert rbw["samples_satisfied"] >= 1
    # train_read_before_write reads smart_home.list_devices in the history first.
    unsatisfied_ids = {row["id"] for row in rbw["unsatisfied"]}
    assert "train_read_before_write" not in unsatisfied_ids


def test_alignment_report_flags_mismatch():
    ok = _make_sample(
        "ok", calls=[ToolCall("alarm.set", {"time": "06:30"})], results=[ToolResult("alarm.set", True, {})]
    )
    mismatched = _make_sample(
        "mismatch", calls=[ToolCall("alarm.set", {"time": "06:30"})], results=[]
    )
    unlabeled = Sample(
        id="unlabeled", split="train", audio_path=None, audio_text="x", context=Context(), history=[], label=None
    )
    report = alignment_report([ok, mismatched, unlabeled])
    assert report["n_samples"] == 3
    assert report["n_missing_labels"] == 1
    assert report["n_call_result_mismatch"] == 1
    assert report["call_result_mismatches"][0]["id"] == "mismatch"
    assert report["has_limitations"] is True


def test_alignment_report_detects_result_before_call():
    from lifemate.data.schemas import HistoryTurn

    history = [
        HistoryTurn.from_raw({"role": "tool", "name": "alarm.list", "result": {"ok": True}}),
    ]
    sample = _make_sample("early_result", history=history)
    report = alignment_report([sample])
    assert report["n_results_before_calls"] >= 1


def test_final_state_counts_only():
    result = analyze_trajectories(_train_samples(), catalog=_train_catalog())
    fs = result["final_state"]
    assert fs["n_labels_with_final_state"] >= 1
    assert fs["note"].startswith("counts only")
    assert "apps" in {row["app"] for row in fs["apps_named"]}


# ---------------------------------------------------------------------------
# Report + determinism
# ---------------------------------------------------------------------------
def test_build_report_synthetic_banner_and_sections():
    samples = _train_samples()
    bundle = build_analysis_bundle(
        samples,
        raw_records=load_raw_split(FIXTURES, "train"),
        catalog=_train_catalog(),
        synthetic=True,
    )
    bundle["asr"] = {"available": False, "reason": "no predictions", "backend": None, "n": 0}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eda"
        info = build_report(bundle, split="train", data_root=str(FIXTURES), synthetic=True, out_dir=out)

        report = (out / "report.md").read_text(encoding="utf-8")
        first_two = report.splitlines()[:2]
        assert first_two == SYNTHETIC_BANNER.splitlines()
        assert "SYNTHETIC TEST DATA" in first_two[0]
        assert "NOT OFFICIAL COMPETITION STATISTICS" in first_two[1]

        for section in (
            "## Data quality issues",
            "## Missing fields",
            "## Data visibility limitations",
            "## Training implications",
            "## Preprocessing recommendations",
            "## Imbalance",
            "Token statistics: unavailable (tokenizer not loaded)",
            "ASR evaluation framework present; no predictions available ⇒ WER/CER not reported",
        ):
            assert section in report, section

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["meta"]["split"] == "train"
        assert summary["meta"]["synthetic"] is True
        assert summary["dataset"]["n_samples"] == 14
        assert summary["trajectory"]["n_samples"] == 14

        tables = sorted(p.name for p in (out / "tables").glob("*.csv"))
        assert tables == [
            "audio_per_file.csv",
            "per_app.csv",
            "tool_usage.csv",
            "tools.csv",
            "transitions.csv",
        ]
        assert info["synthetic"] is True
        # matplotlib is absent in this environment -> graceful degradation note.
        if not matplotlib_available():
            assert "charts: skipped (matplotlib unavailable)" in report
            manifest = json.loads((out / "charts" / "charts_manifest.json").read_text())
            assert manifest["available"] is False


def test_no_synthetic_banner_when_real():
    bundle = build_analysis_bundle(_train_samples(), catalog=_train_catalog(), synthetic=False)
    with tempfile.TemporaryDirectory() as tmp:
        build_report(bundle, split="train", data_root="data", synthetic=False, out_dir=tmp)
        report = (Path(tmp) / "report.md").read_text(encoding="utf-8")
        assert not report.startswith("> **SYNTHETIC")


def test_report_is_deterministic():
    """Two runs must produce byte-identical artifacts (no timestamps, no randomness)."""
    samples = _train_samples()
    catalog = _train_catalog()
    raw = load_raw_split(FIXTURES, "train")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in (1, 2):
            bundle = build_analysis_bundle(samples, raw_records=raw, catalog=catalog, synthetic=True)
            out = Path(tmp) / f"run{i}"
            build_report(bundle, split="train", data_root=str(FIXTURES), synthetic=True, out_dir=out)
            paths.append(out)
        first = (paths[0] / "summary.json").read_text(encoding="utf-8")
        second = (paths[1] / "summary.json").read_text(encoding="utf-8")
        assert first == second
        assert (paths[0] / "report.md").read_text() == (paths[1] / "report.md").read_text()
        for name in ("tools.csv", "tool_usage.csv", "transitions.csv", "per_app.csv"):
            assert (paths[0] / "tables" / name).read_text() == (paths[1] / "tables" / name).read_text()


def test_charts_manifest_and_degradation():
    bundle = build_analysis_bundle(_train_samples(), catalog=_train_catalog(), synthetic=True)
    with tempfile.TemporaryDirectory() as tmp:
        written = make_charts(bundle, tmp)
        manifest = json.loads((Path(tmp) / "charts_manifest.json").read_text())
        if matplotlib_available():
            assert len(written) == 10
            assert manifest["available"] is True
            assert len(manifest["charts"]) == 10
            for row in manifest["charts"]:
                assert Path(row["path"]).is_file()
                assert row["data_hash"]
                assert row["series"] is not None
        else:
            print("SKIP: matplotlib unavailable -> verifying graceful degradation")
            assert written == []
            assert manifest["available"] is False
            assert manifest["reason"] == "matplotlib unavailable"


def test_charts_never_fabricate():
    """Empty bundle -> placeholder charts, and no crash on missing keys."""
    empty_bundle = {"dataset": {}, "audio": {"per_file": []}, "tools": {}, "trajectory": {}}
    with tempfile.TemporaryDirectory() as tmp:
        written = make_charts(empty_bundle, tmp)
        if matplotlib_available():
            assert len(written) == len(__import__("lifemate.eda.visualization", fromlist=["CHART_SPECS"]).CHART_SPECS)
            for path in written:
                assert Path(path).stat().st_size > 0
        else:
            print("SKIP: matplotlib unavailable")


# ---------------------------------------------------------------------------
# Compliance / static checks
# ---------------------------------------------------------------------------
def test_eda_source_never_references_environment_dirs():
    """No EDA module or the runner may name a prohibited simulator directory."""
    targets = sorted((REPO_ROOT / "src" / "lifemate" / "eda").glob("*.py"))
    targets.append(REPO_ROOT / "scripts" / "run_eda.py")
    offenders = []
    for path in targets:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "_environment" in line:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert offenders == [], offenders


def test_discovery_finds_fixture_splits():
    splits = discover_splits(FIXTURES)
    assert {"train", "public_test"} <= set(splits)
    assert splits["train"].has_environment_dir is False

# ---------------------------------------------------------------------------
# ASR evaluation reachability + doc drift
# ---------------------------------------------------------------------------
def _prediction_rows() -> list[dict]:
    """Synthetic predictions derived from the fixture references (clearly labelled)."""
    rows = []
    for sample in _train_samples():
        rows.append(
            {
                "id": str(sample.id),
                "reference": sample.audio_text,
                "hypothesis": sample.audio_text,
                "latency_s": 0.5,
            }
        )
    return rows

def test_eda_asr_predictions_reach_the_report():
    """run_eda --asr-predictions feeds evaluate_asr + diagnostics into the report."""
    import subprocess
    import sys as _sys

    rows = _prediction_rows()
    rows[0]["hypothesis"] = "SYNTHETIC hypothesis carrying a magnitude change 5.000.000d"
    with tempfile.TemporaryDirectory() as tmp:
        preds = Path(tmp) / "preds.jsonl"
        preds.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        out = Path(tmp) / "eda"
        probe = subprocess.run(
            [_sys.executable, str(REPO_ROOT / "scripts" / "run_eda.py"),
             "--data-root", str(FIXTURES), "--split", "train",
             "--asr-predictions", str(preds), "--output", str(out)],
            capture_output=True, text=True, timeout=300,
        )
        assert probe.returncode == 0, probe.stderr
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        asr = summary["asr"]
        assert asr["available"] is True
        assert asr["n"] == len(rows)
        assert asr["raw"] and asr["normalized"]
        assert set(asr["critical_tokens"]["classes"]) == {
            "numbers", "times", "money", "entities", "addresses", "english_loanwords"
        }
        report = (out / "report.md").read_text(encoding="utf-8")
        assert f"ASR evaluation: {asr['backend']}, n={asr['n']}" in report
        assert "file-supplied predictions (not a benchmark result)" in report
        assert "Critical token diagnostics" in report
        assert "magnitude errors" in report.lower()

def test_eda_asr_predictions_unavailable_path_is_explicit():
    """No --asr-predictions => the explicit unavailable sentence, never zero-filled scores."""
    import subprocess
    import sys as _sys

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eda"
        probe = subprocess.run(
            [_sys.executable, str(REPO_ROOT / "scripts" / "run_eda.py"),
             "--data-root", str(FIXTURES), "--split", "train", "--output", str(out)],
            capture_output=True, text=True, timeout=300,
        )
        assert probe.returncode == 0, probe.stderr
        asr = json.loads((out / "summary.json").read_text(encoding="utf-8"))["asr"]
        assert asr["available"] is False
        assert asr["reason"] == "no ASR predictions supplied to run_eda.py"
        assert "raw" not in asr and "normalized" not in asr
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "no predictions available ⇒ WER/CER not reported" in report

def test_eda_asr_predictions_missing_file_is_not_zero_filled():
    """A missing predictions file exits 2 and leaves the ASR block unavailable."""
    import subprocess
    import sys as _sys

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eda"
        probe = subprocess.run(
            [_sys.executable, str(REPO_ROOT / "scripts" / "run_eda.py"),
             "--data-root", str(FIXTURES), "--split", "train",
             "--asr-predictions", str(Path(tmp) / "absent.jsonl"), "--output", str(out)],
            capture_output=True, text=True, timeout=300,
        )
        assert probe.returncode == 2, (probe.returncode, probe.stderr)
        asr = json.loads((out / "summary.json").read_text(encoding="utf-8"))["asr"]
        assert asr["available"] is False
        assert "not found" in asr["reason"]

def _fenced_blocks(path: Path) -> list[tuple[int, str]]:
    """(line number, block text) for every fenced code block."""
    blocks = []
    current: list[str] = []
    start = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().startswith("```"):
            if start:
                blocks.append((start, "\n".join(current)))
                current, start = [], 0
            else:
                start = lineno
            continue
        if start:
            current.append(line)
    return blocks

def test_docs_never_present_whisper_as_primary_asr():
    """--asr whisper / ASR=whisper only inside a block labelled as a comparison."""
    offenders = []
    for name in ("docs/INFERENCE_GUIDE.md", "README.md"):
        path = REPO_ROOT / name
        lines = path.read_text(encoding="utf-8").splitlines()
        for start, block in _fenced_blocks(path):
            if "--asr whisper" not in block and "ASR=whisper" not in block:
                continue
            context = lines[max(0, start - 7): start + len(block.splitlines())]
            haystack = "\n".join(context).lower()
            if "comparison" not in haystack and "so sánh" not in haystack:
                offenders.append(f"{name}:{start}\n{block}")
    assert offenders == [], "\n\n".join(offenders)

def test_primary_inference_example_uses_gipformer_default():
    path = REPO_ROOT / "docs/INFERENCE_GUIDE.md"
    text = path.read_text(encoding="utf-8")
    blocks = _fenced_blocks(path)
    primary = next((b for _, b in blocks if "scripts/infer.py" in b), "")
    assert primary, "no primary infer.py example found"
    assert "--asr whisper" not in primary
    assert "gipformer" in text.lower()

def _help(script: str) -> str:
    import subprocess
    import sys as _sys

    probe = subprocess.run(
        [_sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert probe.returncode == 0, (script, probe.stderr)
    return probe.stdout + probe.stderr

def test_cli_help_reaches_asr_evaluation():
    transcribe = _help("transcribe_split.py")
    for flag in ("--split", "--output", "--quantization", "--limit"):
        assert flag in transcribe, f"transcribe_split.py --help lacks {flag}"
    assert "--asr-predictions" in _help("run_eda.py")
    assert "--asr-predictions" in _help("evaluate.py")

def test_docs_do_not_claim_backend_side_llm_normalization():
    """to_llm_text lives in scripts/infer.py, not inside GipformerBackend.

    A window of +-3 lines containing both names is a stale claim unless it also
    points at the real, backend-agnostic location.
    """
    import re

    pattern = re.compile(
        r"GipformerBackend[\s\S]{0,400}?to_llm_text|to_llm_text[\s\S]{0,400}?GipformerBackend"
    )
    offenders = []
    for name in ("docs/GIPFORMER_INTEGRATION_AUDIT.md", "docs/IMPLEMENTATION_READINESS_REPORT.md"):
        lines = (REPO_ROOT / name).read_text(encoding="utf-8").splitlines()
        for i in range(len(lines)):
            window = "\n".join(lines[max(0, i - 3): i + 4])
            if not pattern.search(window):
                continue
            if "scripts/infer.py" in window or "backend-agnostic" in window:
                continue
            offenders.append(f"{name}:{i + 1}: {lines[i].strip()}")
    assert offenders == [], offenders

def _load_script(name: str):
    """Import scripts/<name>.py by path (scripts/ is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"lifemate_script_{name.replace('.py', '')}", REPO_ROOT / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class _StubBackend:
    """Synthetic backend: echoes a fixed transcript, no audio decoding."""

    name = "stub"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def transcribe(self, audio_path, *, sample_id=None, lenient=False):
        from lifemate.asr.interface import TranscriptResult

        if self.fail:
            return TranscriptResult(text="", error="SYNTHETIC stub failure", backend=self.name)
        return TranscriptResult(
            text="SYNTHETIC transcript", raw_text="SYNTHETIC transcript",
            duration_s=1.0, latency_s=0.25, backend=self.name, model_name="stub",
        )

def test_transcribe_split_writes_predictions_and_metrics():
    """transcribe_split.py: exit 0, JSONL rows + <output>.metrics.json (SYNTHETIC backend)."""
    import lifemate.asr as asr_pkg

    script = _load_script("transcribe_split.py")
    original = asr_pkg.create_asr_backend
    asr_pkg.create_asr_backend = lambda cfg, *a, **k: _StubBackend()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preds.jsonl"
            code = script.main([
                "--split", "train", "--data-root", str(FIXTURES), "--output", str(out), "--limit", "3",
            ])
            assert code == 0
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 3
            for row in rows:
                assert set(row) == {
                    "id", "split", "reference", "hypothesis", "raw_hypothesis",
                    "latency_s", "duration_s", "error",
                }
                assert row["hypothesis"] == "SYNTHETIC transcript"
                assert row["reference"]  # gold audio_text from the fixture label
            metrics = json.loads((out.with_name(out.name + ".metrics.json")).read_text(encoding="utf-8"))
            assert metrics["available"] is True
            assert metrics["n"] == 3 and metrics["backend"] == "stub"
            assert set(metrics["critical_tokens"]["classes"]) == {
                "numbers", "times", "money", "entities", "addresses", "english_loanwords"
            }
            assert metrics["run"]["n_failed"] == 0
            assert metrics["run"]["rtf"] is not None
    finally:
        asr_pkg.create_asr_backend = original

def test_transcribe_split_all_failed_exits_3():
    """Every transcription failing => exit 3, counted, and no invented text."""
    import lifemate.asr as asr_pkg

    script = _load_script("transcribe_split.py")
    original = asr_pkg.create_asr_backend
    asr_pkg.create_asr_backend = lambda cfg, *a, **k: _StubBackend(fail=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preds.jsonl"
            code = script.main([
                "--split", "train", "--data-root", str(FIXTURES), "--output", str(out), "--limit", "2",
            ])
            assert code == 3
    finally:
        asr_pkg.create_asr_backend = original

def test_evaluate_asr_block_is_independent_of_final_state():
    """evaluate.py --asr-predictions appends an asr block; final_state stays unavailable."""
    script = _load_script("evaluate.py")
    rows = _prediction_rows()[:3]
    with tempfile.TemporaryDirectory() as tmp:
        preds = Path(tmp) / "preds.jsonl"
        preds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        block = script._asr_block(str(preds))
        assert block["available"] is True and block["n"] == 3
        assert block["source"] == "file"
        section = script._asr_section_md(block)
        assert "independent of final-state correctness" in section
        assert "final-state" in section
        missing = script._asr_block(str(Path(tmp) / "absent.jsonl"))
        assert missing["available"] is False and "not found" in missing["reason"]
