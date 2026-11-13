"""Chart generation with matplotlib (Agg) — optional dependency.

Import-safe without matplotlib: :func:`make_charts` returns ``[]`` and a note
when it is unavailable, and never fabricates data. Every chart is derived from
the passed analysis dict only; an empty series produces an explicit "no data"
placeholder chart instead of invented values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["make_charts", "CHART_SPECS", "data_hash", "matplotlib_available"]

#: (filename, human title in Vietnamese ASCII-safe text, bundle key path)
CHART_SPECS: list[tuple[str, str, str]] = [
    ("01_audio_duration_hist.png", "Phan bo thoi luong audio (giay)", "audio.duration"),
    ("02_token_length_hist.png", "Phan bo do dai token", "dataset.tokens"),
    ("03_tool_call_count_bar.png", "So luong tool call moi mau", "trajectory.n_gold_calls"),
    ("04_top_tools_bar.png", "Cac tool duoc dung nhieu nhat", "trajectory.tool_usage"),
    ("05_tool_kind_bar.png", "Phan loai tool (read/write/meta)", "tools.kind_distribution"),
    ("06_installed_apps_bar.png", "So app duoc cai dat moi mau", "dataset.installed_apps"),
    ("07_apps_changed_bar.png", "So app thay doi trang thai", "trajectory.final_state"),
    ("08_read_write_sequences.png", "Trinh tu read/write trong trajectory", "trajectory.top_chains"),
    ("09_missing_info_out_of_scope.png", "Ty le thieu thong tin / ngoai pham vi", "trajectory.missing_info"),
    ("10_history_length_hist.png", "Phan bo do dai hoi thoai (so luot)", "dataset.history_length"),
]


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def data_hash(obj: Any) -> str:
    """Stable short hash of a JSON-serialisable object (for manifest provenance)."""
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _dig(bundle: dict[str, Any], dotted: str) -> Any:
    node: Any = bundle
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _placeholder(ax: Any, reason: str) -> None:
    ax.text(
        0.5,
        0.5,
        "no data\n" + reason,
        ha="center",
        va="center",
        fontsize=11,
        color="#888888",
        transform=ax.transAxes,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _hist(ax: Any, values: list[float], title: str, xlabel: str) -> None:
    if not values:
        _placeholder(ax, "empty series")
        return
    unique = len(set(values))
    bins = min(40, max(5, unique)) if unique > 1 else 1
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("So mau")


def _bar(ax: Any, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    if not labels or not values:
        _placeholder(ax, "empty series")
        return
    order = sorted(range(len(values)), key=lambda i: values[i])
    ax.barh([labels[i] for i in order], [values[i] for i in order], color="#F58518")
    ax.set_title(title)
    ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------
# Per-chart builders (each returns the series that was plotted)
# ---------------------------------------------------------------------------
def _chart_audio_duration(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "audio") or {}
    values = [
        row["duration_s"]
        for row in (node.get("per_file") or [])
        if isinstance(row.get("duration_s"), (int, float))
    ]
    _hist(ax, values, "Phan bo thoi luong audio (giay)", "Thoi luong (giay)")
    return {"per_file_durations": len(values)}


def _chart_token_length(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "dataset.tokens") or {}
    if node.get("tokenizer") != "available":
        _placeholder(ax, "tokenizer unavailable")
        ax.set_title("Phan bo do dai token")
        return {"tokenizer": node.get("tokenizer", "unavailable")}
    stats = node.get("stats") or {}
    # Only aggregate stats are available (no per-sample list) -> plot the
    # quantile curve we actually know, never a synthesised histogram.
    labels = ["mean", "median", "p95", "p99", "max"]
    values = [stats.get(k) for k in ("mean", "median", "p95", "p99", "max")]
    pairs = [(lab, float(v)) for lab, v in zip(labels, values) if isinstance(v, (int, float))]
    if not pairs:
        _placeholder(ax, "no token stats")
        ax.set_title("Phan bo do dai token")
        return {}
    ax.bar([p[0] for p in pairs], [p[1] for p in pairs], color="#54A24B")
    ax.set_title("Thong ke do dai token")
    ax.set_ylabel("Token")
    return {"aggregate_stats": dict(pairs)}


def _chart_call_count(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "trajectory") or {}
    values = (node.get("n_gold_calls") or {}).get("values") or []
    _hist(ax, [float(v) for v in values], "So luong tool call moi mau", "So tool call")
    return {"n_samples": len(values)}


def _chart_top_tools(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "trajectory.tool_usage") or []
    rows = [r for r in node if isinstance(r, dict)][:20]
    _bar(
        ax,
        [str(r.get("tool")) for r in rows],
        [float(r.get("count") or 0) for r in rows],
        "Cac tool duoc dung nhieu nhat (top 20)",
        "So lan xuat hien",
    )
    return {"top": [str(r.get("tool")) for r in rows]}


def _chart_tool_kind(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "tools.kind_distribution") or {}
    items = sorted(node.items())
    if not items:
        _placeholder(ax, "no catalog")
        ax.set_title("Phan loai tool (read/write/meta)")
        return {}
    ax.bar([k for k, _ in items], [v for _, v in items], color="#B279A2")
    ax.set_title("Phan loai tool (read/write/meta)")
    ax.set_ylabel("So tool")
    return dict(items)


def _chart_installed_apps(ax: Any, bundle: dict[str, Any]) -> Any:
    node = (_dig(bundle, "dataset.installed_apps") or {}).get("count_distribution") or {}
    items = sorted(((int(k), v) for k, v in node.items()), key=lambda kv: kv[0])
    if not items:
        _placeholder(ax, "no context data")
        ax.set_title("So app duoc cai dat moi mau")
        return {}
    ax.bar([str(k) for k, _ in items], [v for _, v in items], color="#72B7B2")
    ax.set_title("So app duoc cai dat moi mau")
    ax.set_xlabel("So app")
    ax.set_ylabel("So mau")
    return {"buckets": {str(k): v for k, v in items}}


def _chart_apps_changed(ax: Any, bundle: dict[str, Any]) -> Any:
    node = (_dig(bundle, "trajectory.final_state") or {}).get(
        "apps_changed_distribution"
    ) or {}
    items = sorted(((int(k), v) for k, v in node.items()), key=lambda kv: kv[0])
    if not items:
        _placeholder(ax, "no final_state labels")
        ax.set_title("So app thay doi trang thai")
        return {}
    ax.bar([str(k) for k, _ in items], [v for _, v in items], color="#E45756")
    ax.set_title("So app thay doi trang thai")
    ax.set_xlabel("So app thay doi")
    ax.set_ylabel("So mau")
    return {"buckets": {str(k): v for k, v in items}}


def _chart_read_write(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "trajectory.top_chains") or []
    rows = [r for r in node if isinstance(r, dict)][:10]
    if not rows:
        _placeholder(ax, "no multi-step chains")
        ax.set_title("Trinh tu read/write trong trajectory")
        return {}
    labels = [" > ".join(str(s) for s in (r.get("sequence") or [])) for r in rows]
    values = [float(r.get("count") or 0) for r in rows]
    ax.barh(labels[::-1], values[::-1], color="#4C78A8")
    ax.set_title("Trinh tu read/write trong trajectory (top 10)")
    ax.set_xlabel("So lan")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(7)
    return {"chains": [{"sequence": r.get("sequence"), "count": r.get("count")} for r in rows]}


def _chart_missing_out_of_scope(ax: Any, bundle: dict[str, Any]) -> Any:
    traj = _dig(bundle, "trajectory") or {}
    n = traj.get("n_samples") or 0
    missing = (traj.get("missing_info") or {}).get("count")
    oos = (traj.get("out_of_scope") or {}).get("count")
    if not n or missing is None or oos is None:
        _placeholder(ax, "no trajectory data")
        ax.set_title("Ty le thieu thong tin / ngoai pham vi")
        return {}
    labels = ["Thieu thong tin\n(ask_missing_info)", "Ngoai pham vi\n(0 call)", "Con lai"]
    remaining = max(0, n - (missing or 0) - (oos or 0))
    values = [missing or 0, oos or 0, remaining]
    ax.bar(labels, values, color=["#54A24B", "#E45756", "#B0B0B0"])
    ax.set_title(f"Ty le thieu thong tin / ngoai pham vi (n={n})")
    ax.set_ylabel("So mau")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
    return {"missing_info": missing, "out_of_scope": oos, "remaining": remaining}


def _chart_history_length(ax: Any, bundle: dict[str, Any]) -> Any:
    node = _dig(bundle, "dataset.history_length") or {}
    values = node.get("values") or []
    _hist(ax, [float(v) for v in values], "Phan bo do dai hoi thoai (so luot)", "So luot")
    return {"n_samples": len(values)}


_BUILDERS = {
    "01_audio_duration_hist.png": _chart_audio_duration,
    "02_token_length_hist.png": _chart_token_length,
    "03_tool_call_count_bar.png": _chart_call_count,
    "04_top_tools_bar.png": _chart_top_tools,
    "05_tool_kind_bar.png": _chart_tool_kind,
    "06_installed_apps_bar.png": _chart_installed_apps,
    "07_apps_changed_bar.png": _chart_apps_changed,
    "08_read_write_sequences.png": _chart_read_write,
    "09_missing_info_out_of_scope.png": _chart_missing_out_of_scope,
    "10_history_length_hist.png": _chart_history_length,
}


def make_charts(analysis_bundle: dict[str, Any], out_dir: str | Path) -> list[str]:
    """Write PNGs + ``charts_manifest.json``; returns written paths (``[]`` if no matplotlib)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not matplotlib_available():
        manifest = {
            "available": False,
            "reason": "matplotlib unavailable",
            "charts": [],
            "note": "install matplotlib to generate charts",
        }
        (out / "charts_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return []

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    written: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for filename, title, source_key in CHART_SPECS:
        builder = _BUILDERS[filename]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        try:
            series = builder(ax, analysis_bundle)
            ax.set_title(title if not ax.get_title() else ax.get_title())
            fig.tight_layout()
            path = out / filename
            fig.savefig(path, dpi=110)
            written.append(str(path))
            manifest_rows.append(
                {
                    "file": filename,
                    "path": str(path),
                    "title": title,
                    "source": source_key,
                    "series": series,
                    "data_hash": data_hash(series),
                }
            )
        finally:
            plt.close(fig)

    manifest = {
        "available": True,
        "backend": "matplotlib Agg",
        "n_charts": len(written),
        "charts": manifest_rows,
        "data_hash": data_hash([row["data_hash"] for row in manifest_rows]),
    }
    (out / "charts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return written
