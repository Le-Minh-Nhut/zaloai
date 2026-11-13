"""Tool-catalog EDA: shape, parameter complexity, and dependency graph.

Pure stdlib; works with any object exposing the ``ToolCatalog`` contract
(``.apps``, ``.tools``, ``.dependency_edges()``). Missing attributes degrade to
an explicit ``available: False`` block instead of an exception.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

__all__ = [
    "analyze_catalog",
    "render_dependency_graph_mermaid",
    "render_dependency_graph_ascii",
    "tool_table_rows",
    "META_TOOL",
]

#: Tool the agent calls when the request lacks required information.
META_TOOL = "sys.ask_missing_info"

_JSON_TYPE_ORDER = ("string", "integer", "number", "boolean", "array", "object", "null")


def _is_mapping(value: Any) -> bool:
    return hasattr(value, "get") and hasattr(value, "items")


def _tool_parameters(spec: Any) -> dict[str, Any]:
    params = getattr(spec, "parameters", None)
    return dict(params) if _is_mapping(params) else {}


def _properties(spec: Any) -> dict[str, Any]:
    props = getattr(spec, "properties", None)
    if _is_mapping(props):
        return dict(props)
    props = _tool_parameters(spec).get("properties", {})
    return dict(props) if _is_mapping(props) else {}


def _required(spec: Any) -> list[str]:
    req = getattr(spec, "required", None)
    if isinstance(req, (list, tuple)):
        return [str(x) for x in req]
    req = _tool_parameters(spec).get("required", [])
    return [str(x) for x in req] if isinstance(req, (list, tuple)) else []


def _ref_of(schema: Any) -> tuple[str | None, str | None]:
    """Return ``(kind, target)`` for an x-ref / x-match annotation."""
    if not _is_mapping(schema):
        return None, None
    for key in ("x-ref", "x_ref", "ref"):
        value = schema.get(key)
        if isinstance(value, str) and value:
            return "x-ref", value
    for key in ("x-match", "x_match"):
        value = schema.get(key)
        if isinstance(value, str) and value:
            return "x-match", value
    return None, None


def _one_of_of(schema: Any) -> Any:
    if not _is_mapping(schema):
        return None
    return schema.get("x-one-of", schema.get("oneOf"))


# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------
def _graph_nodes(edges: Iterable[tuple[str, str, str]]) -> list[str]:
    nodes: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        for name in (edge[0], edge[1]):
            if name and name not in seen:
                seen.add(name)
                nodes.append(name)
    return nodes


def _mermaid_id(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not safe or safe[0].isdigit():
        safe = "n_" + safe
    return safe


def render_dependency_graph_mermaid(
    edges: Iterable[tuple[str, str, str]], title: str = "Tool dependency graph"
) -> str:
    """Render ``x-ref``/``x-match`` edges as a fenced mermaid graph.

    Always returns a valid fenced block; with no edges it renders a single
    explicit "no dependencies" node rather than an empty diagram.
    """
    edges = [tuple(e) for e in edges or []]
    lines = ["```mermaid", "graph TD", f"    %% {title}"]
    if not edges:
        lines.append('    no_dependencies["No x-ref/x-match dependencies"]')
        lines.append("```")
        return "\n".join(lines)
    for name in _graph_nodes(edges):
        lines.append(f'    {_mermaid_id(name)}["{name}"]')
    kind_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, target, field in edges:
        kind_seen[(source, target)].add(str(field))
    for (source, target), fields in kind_seen.items():
        label = ",".join(sorted(fields))
        lines.append(f"    {_mermaid_id(source)} -->|{label}| {_mermaid_id(target)}")
    lines.append("```")
    return "\n".join(lines)


def render_dependency_graph_ascii(edges: Iterable[tuple[str, str, str]]) -> str:
    """Plain-text adjacency listing (terminal/report friendly, no Graphviz)."""
    edges = [tuple(e) for e in edges or []]
    if not edges:
        return "Tool dependency graph: no x-ref/x-match dependencies"
    children: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, target, field in edges:
        children[source].append((target, str(field)))
    roots = sorted(children)
    width = max(len(name) for name in _graph_nodes(edges))
    lines: list[str] = ["Tool dependency graph (x-ref / x-match):"]
    for source in roots:
        for i, (target, field) in enumerate(sorted(children[source])):
            head = source.ljust(width) if i == 0 else " " * width
            lines.append(f"{head}  -> {target.ljust(width)}  [{field}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Catalog analysis
# ---------------------------------------------------------------------------
def analyze_catalog(catalog: Any) -> dict[str, Any]:
    """Analyze a tool catalog. Returns ``{"available": False}`` when absent."""
    if catalog is None:
        return {
            "available": False,
            "reason": "no catalog provided (split has no tools catalog file)",
            "n_apps": 0,
            "n_tools": 0,
        }
    tools = getattr(catalog, "tools", None)
    if not _is_mapping(tools):
        return {
            "available": False,
            "reason": f"object {type(catalog).__name__} has no .tools mapping",
            "n_apps": 0,
            "n_tools": 0,
        }
    tools = dict(tools)
    apps = list(getattr(catalog, "apps", []) or [])

    kind_counts: Counter[str] = Counter()
    tools_per_app: Counter[str] = Counter()
    required_counts: Counter[int] = Counter()
    optional_counts: Counter[int] = Counter()
    type_freq: Counter[str] = Counter()
    enum_freq: Counter[int] = Counter()
    enum_tools: list[dict[str, Any]] = []
    ref_freq: Counter[str] = Counter()
    ref_fields: Counter[str] = Counter()
    one_of_count = 0
    one_of_tools: list[str] = []
    match_fields: Counter[str] = Counter()
    n_params_total = 0
    rows: list[dict[str, Any]] = []

    for name in sorted(tools):
        spec = tools[name]
        kind = str(getattr(spec, "kind", "unknown") or "unknown")
        kind_counts[kind] += 1
        app = str(getattr(spec, "app_name", "") or "")
        tools_per_app[app or "(none)"] += 1
        props = _properties(spec)
        required = _required(spec)
        optional = [k for k in props if k not in required]
        required_counts[len(required)] += 1
        optional_counts[len(optional)] += 1
        n_params_total += len(props)

        tool_enums: list[str] = []
        tool_refs: list[str] = []
        has_one_of = _one_of_of(_tool_parameters(spec)) is not None
        for param, schema in sorted(props.items()):
            if not _is_mapping(schema):
                type_freq["(untyped)"] += 1
                continue
            declared = schema.get("type")
            if isinstance(declared, list):
                for t in declared:
                    type_freq[str(t)] += 1
            elif declared is None:
                type_freq["(untyped)"] += 1
            else:
                type_freq[str(declared)] += 1
            enum = schema.get("enum")
            if isinstance(enum, (list, tuple)) and enum:
                enum_freq[len(enum)] += 1
                tool_enums.append(param)
            kind_ref, target = _ref_of(schema)
            if kind_ref:
                ref_freq[kind_ref] += 1
                ref_fields[param] += 1
                tool_refs.append(f"{param}->{target}")
            if kind_ref == "x-match":
                match_fields[param] += 1
            if _one_of_of(schema) is not None:
                has_one_of = True
        if tool_enums:
            enum_tools.append({"tool": name, "fields": tool_enums})
        if has_one_of:
            one_of_count += 1
            one_of_tools.append(name)

        rows.append(
            {
                "tool": name,
                "app": app,
                "kind": kind,
                "n_parameters": len(props),
                "n_required": len(required),
                "n_optional": len(optional),
                "required": "|".join(required),
                "params_with_enum": "|".join(tool_enums),
                "params_with_ref": "|".join(tool_refs),
                "has_one_of": has_one_of,
                "description_chars": len(str(getattr(spec, "description", "") or "")),
            }
        )

    edges: list[tuple[str, str, str]] = []
    edges_source = getattr(catalog, "dependency_edges", None)
    if callable(edges_source):
        try:
            edges = [tuple(e) for e in edges_source() if len(e) >= 3]
        except Exception:  # noqa: BLE001 - a broken catalog must not kill EDA
            edges = []
    if not edges:
        # Reconstruct from parameter annotations when the catalog has no helper.
        for name in sorted(tools):
            for param, schema in _properties(tools[name]).items():
                kind_ref, target = _ref_of(schema)
                if kind_ref and target:
                    edges.append((name, target, f"{param} ({kind_ref})"))

    adjacency: dict[str, list[str]] = {}
    for source, target, field in edges:
        adjacency.setdefault(source, [])
        if f"{target} [{field}]" not in adjacency[source]:
            adjacency[source].append(f"{target} [{field}]")

    known = set(tools)
    unresolved = sorted({t for _, t, _ in edges if t not in known and not t.startswith("#")})

    return {
        "available": True,
        "reason": None,
        "n_apps": len(apps),
        "n_tools": len(tools),
        "apps": apps,
        "kind_distribution": dict(sorted(kind_counts.items())),
        "tools_per_app": dict(sorted(tools_per_app.items(), key=lambda kv: (-kv[1], kv[0]))),
        "n_params_total": n_params_total,
        "args_per_tool": {
            "required": dict(sorted(required_counts.items())),
            "optional": dict(sorted(optional_counts.items())),
            "mean_required": (sum(k * v for k, v in required_counts.items()) / len(tools))
            if tools
            else None,
        },
        "parameter_type_frequency": dict(
            sorted(type_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "type_order_expected": list(_JSON_TYPE_ORDER),
        "enum": {
            "n_enum_params": sum(enum_freq.values()),
            "option_count_distribution": dict(sorted(enum_freq.items())),
            "tools_with_enums": enum_tools,
        },
        "x_ref": {
            "n_refs": sum(ref_freq.values()),
            "by_kind": dict(sorted(ref_freq.items())),
            "fields": dict(sorted(ref_fields.items(), key=lambda kv: (-kv[1], kv[0]))),
            "n_unresolved_targets": len(unresolved),
            "unresolved_targets": unresolved[:20],
        },
        "x_one_of": {"n_tools": one_of_count, "tools": one_of_tools},
        "x_match": {
            "n": sum(match_fields.values()),
            "fields": dict(sorted(match_fields.items())),
        },
        "dependency_graph": {
            "n_edges": len(edges),
            "edges": [{"source": s, "target": t, "field": f} for s, t, f in edges],
            "adjacency": dict(sorted(adjacency.items())),
            "n_nodes": len(_graph_nodes(edges)),
            "mermaid": render_dependency_graph_mermaid(edges),
            "ascii": render_dependency_graph_ascii(edges),
        },
        "has_meta_tool": META_TOOL in tools,
        "table_rows": rows,
    }


def tool_table_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-tool CSV-ready rows (helper for the report writer)."""
    return list(analysis.get("table_rows", []) or [])
