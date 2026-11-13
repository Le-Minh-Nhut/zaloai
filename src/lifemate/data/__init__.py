"""Data access layer: discovery, tolerant schemas, loading, validation."""

from __future__ import annotations

from .catalog import ToolCatalog, ToolSpec
from .discovery import (
    SplitPaths,
    available_splits,
    discover_splits,
    require_split,
)
from .loader import (
    iter_raw_records,
    load_catalog,
    load_raw_split,
    load_samples,
    read_sample_audio,
)
from .schemas import (
    Context,
    HistoryTurn,
    Label,
    Sample,
    ToolCall,
    ToolResult,
    parse_context,
    parse_history,
    parse_tool_calls,
    parse_tool_results,
)
from .validator import validate_split

__all__ = [
    "ToolCatalog",
    "ToolSpec",
    "SplitPaths",
    "available_splits",
    "discover_splits",
    "require_split",
    "iter_raw_records",
    "load_catalog",
    "load_raw_split",
    "load_samples",
    "read_sample_audio",
    "Context",
    "HistoryTurn",
    "Label",
    "Sample",
    "ToolCall",
    "ToolResult",
    "parse_context",
    "parse_history",
    "parse_tool_calls",
    "parse_tool_results",
    "validate_split",
]
