"""Exploratory data analysis for the LifeMate dataset.

Every submodule is import-safe without numpy / matplotlib / torch. Heavy
dependencies are imported lazily inside the functions that need them.

Contract:
    analyze_dataset(...)      -> dataset-level statistics
    analyze_split_audio(...)  -> audio statistics (stdlib ``wave``)
    analyze_catalog(...)      -> tool-catalog statistics + dependency graph
    analyze_trajectories(...) -> gold-trajectory statistics
    alignment_report(...)     -> data visibility limitations
    make_charts(...)          -> PNGs (or a note when matplotlib is missing)
    build_report(...)         -> summary.json / report.md / charts / tables
"""

from __future__ import annotations

from .audio_analysis import analyze_split_audio, analyze_wav, pearson
from .dataset_analysis import analyze_dataset, describe, parse_iso_like
from .report import SYNTHETIC_BANNER, build_report
from .tool_analysis import (
    analyze_catalog,
    render_dependency_graph_ascii,
    render_dependency_graph_mermaid,
)
from .trajectory_analysis import alignment_report, analyze_trajectories, classify_challenge
from .visualization import CHART_SPECS, make_charts, matplotlib_available

__all__ = [
    "analyze_dataset",
    "describe",
    "parse_iso_like",
    "analyze_split_audio",
    "analyze_wav",
    "pearson",
    "analyze_catalog",
    "render_dependency_graph_ascii",
    "render_dependency_graph_mermaid",
    "analyze_trajectories",
    "alignment_report",
    "classify_challenge",
    "make_charts",
    "matplotlib_available",
    "CHART_SPECS",
    "build_report",
    "SYNTHETIC_BANNER",
    "build_analysis_bundle",
]


def build_analysis_bundle(
    samples,
    *,
    raw_records=None,
    catalog=None,
    tokenizer=None,
    max_audio_files=None,
    synthetic: bool = False,
) -> dict:
    """Run every analysis once and return the bundle consumed by charts/report."""
    return {
        "dataset": analyze_dataset(
            samples, raw_records=raw_records, catalog=catalog, tokenizer=tokenizer
        ),
        "audio": analyze_split_audio(
            samples, max_files=max_audio_files, synthetic=synthetic
        ),
        "tools": analyze_catalog(catalog),
        "trajectory": analyze_trajectories(samples, catalog=catalog),
    }
