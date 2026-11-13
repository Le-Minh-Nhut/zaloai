"""Tolerant tool-catalog model built from organizer JSON.

The real catalog is not available; this module detects several plausible
shapes and records what it saw instead of failing. Extensions understood:
``x-ref``, ``x-match``, ``x-one-of`` (and ``oneOf``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import SchemaError

__all__ = ["ToolSpec", "ToolCatalog", "READ_VERBS", "META_TOOL_NAMES"]

META_TOOL_NAMES = frozenset({"sys.ask_missing_info", "ask_missing_info", "ask_missing"})
READ_VERBS = ("get", "list", "search", "read", "find", "query", "status", "check", "lookup", "retrieve")

_JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass
class ToolSpec:
    """One callable tool from the catalog."""

    name: str
    app_name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)
    kind: str = "unknown"
    annotations: dict = field(default_factory=dict)

    @property
    def required(self) -> list[str]:
        """Required parameter names (JSON Schema ``required``)."""
        req = self.parameters.get("required", [])
        return [str(x) for x in req] if isinstance(req, (list, tuple)) else []

    @property
    def properties(self) -> dict:
        """Parameter schemas by name."""
        props = self.parameters.get("properties", {})
        return dict(props) if isinstance(props, Mapping) else {}

    def to_dict(self) -> dict:
        """Compact prompt-friendly description."""
        return {
            "name": self.name,
            "app": self.app_name,
            "description": self.description,
            "kind": self.kind,
            "parameters": self.parameters,
        }


def _infer_kind(
    name: str, description: str, annotations: Mapping[str, Any], declared: Any = None
) -> str:
    """The tool's ``kind``: the catalog's own declaration wins when present.

    Every official tool carries ``"kind": read|write|meta`` (catalog
    ``schema_version 2.7.0``, documented in its ``note``). Inference by verb
    disagrees with it on 11 of 147 real train tools (e.g. ``calculator.calculate``
    and ``tv.guide`` are declared ``read`` but named like writes), so a declared
    value is authoritative and the heuristics only cover catalogs without one.
    """
    declared_kind = str(declared or "").strip().lower()
    if declared_kind in {"read", "write", "meta"}:
        return declared_kind
    lowered = f"{name} {description}".lower()
    if name in META_TOOL_NAMES or name.startswith("sys.") or "ask_missing" in lowered:
        return "meta"
    if annotations.get("readOnlyHint") is True:
        return "read"
    if annotations.get("destructiveHint") is True:
        return "write"
    leaf = name.split(".")[-1].lower()
    head = leaf.split("_")[0]
    if head in READ_VERBS or leaf in READ_VERBS:
        return "read"
    return "write"


def _split_tool_name(name: str, raw: Mapping[str, Any]) -> tuple[str, str]:
    """Derive ``(app_name, full_name)`` from a raw tool entry."""
    app = raw.get("app_name") or raw.get("app") or raw.get("appName") or raw.get("namespace")
    if not app and "." in name:
        app = name.split(".")[0]
    return str(app or ""), str(name)


class ToolCatalog:
    """All tools plus the apps they belong to."""

    def __init__(self, tools: dict[str, ToolSpec] | None = None, apps: Iterable[str] | None = None, raw: Any = None) -> None:
        self.tools: dict[str, ToolSpec] = dict(tools or {})
        discovered = {spec.app_name for spec in self.tools.values() if spec.app_name}
        self.apps: list[str] = sorted({str(a) for a in (apps or [])} | discovered)
        self.raw = raw

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Any) -> "ToolCatalog":
        """Build a catalog from any supported JSON shape."""
        if not isinstance(data, Mapping):
            raise SchemaError("tool catalog must be a JSON object")
        apps, entries = _extract_entries(data)
        tools: dict[str, ToolSpec] = {}
        for entry in entries:
            spec = _spec_from_entry(entry)
            if spec is None:
                continue
            tools[spec.name] = spec
        return cls(tools=tools, apps=apps, raw=data)

    @classmethod
    def from_file(cls, path: str | Path) -> "ToolCatalog":
        """Load a catalog JSON file (through the compliance guard)."""
        from ..compliance import safe_open

        with safe_open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- queries ----------------------------------------------------------- #
    def get(self, name: str) -> ToolSpec | None:
        """Tool by exact name."""
        return self.tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def tools_for_apps(self, apps: Iterable[str], include_meta: bool = True) -> list[ToolSpec]:
        """Tools of the installed apps, plus ``sys.ask_missing_info`` when present."""
        wanted = {str(a) for a in (apps or [])}
        selected = [
            spec
            for name, spec in sorted(self.tools.items())
            if spec.app_name in wanted and not (spec.kind == "meta" and not include_meta)
        ]
        if include_meta:
            for meta_name in sorted(META_TOOL_NAMES):
                spec = self.tools.get(meta_name)
                if spec is not None and spec.name not in {s.name for s in selected}:
                    selected.append(spec)
                    break
        return selected

    def to_schema_prompt(self, apps: Iterable[str] | None = None, include_meta: bool = True) -> str:
        """Render a compact tool list for prompting."""
        specs = self.tools_for_apps(apps, include_meta) if apps is not None else [self.tools[k] for k in sorted(self.tools)]
        lines = ["Available tools:"]
        for spec in specs:
            required = spec.required
            optional = [k for k in spec.properties if k not in required]
            params = ", ".join(
                [f"{k} (required)" for k in required] + [f"{k} (optional)" for k in optional]
            )
            lines.append(f"- {spec.name} [{spec.kind}] params: {params or 'none'}: {spec.description}")
        return "\n".join(lines)

    # -- validation -------------------------------------------------------- #
    def validate_call(self, name: str, args: Mapping[str, Any] | None = None) -> dict:
        """Validate one call; returns ``{"errors": [...], "warnings": [...]}``."""
        errors: list[str] = []
        warnings: list[str] = []
        arguments = dict(args or {})
        spec = self.tools.get(name)
        if spec is None:
            return {"errors": [f"unknown tool: {name}"], "warnings": []}
        props = spec.properties
        for req in spec.required:
            if _is_empty_arg(arguments.get(req)):
                errors.append(f"{name}: missing required argument '{req}'")
        for key, value in arguments.items():
            if key not in props:
                warnings.append(f"{name}: unknown argument '{key}'")
                continue
            if _is_empty_arg(value):
                continue  # an empty optional value is simply "not given" (official rule)
            errors.extend(self._check_value(name, key, value, props[key], warnings))
        errors.extend(self._check_one_of(name, arguments, spec))
        return {"errors": errors, "warnings": warnings}

    def _check_value(
        self, tool: str, key: str, value: Any, schema: Mapping[str, Any], warnings: list[str]
    ) -> list[str]:
        problems: list[str] = []
        expected = schema.get("type")
        if isinstance(expected, list):
            expected_types = [str(t) for t in expected]
        elif expected is None:
            expected_types = []
        else:
            expected_types = [str(expected)]
        if expected_types and not any(_matches_type(value, t) for t in expected_types):
            problems.append(
                f"{tool}: argument '{key}' expected type {'/'.join(expected_types)}, got {type(value).__name__}"
            )
        enum = schema.get("enum")
        if isinstance(enum, (list, tuple)) and value not in enum:
            problems.append(f"{tool}: argument '{key}'={value!r} not in enum {list(enum)}")
        if any(
            t in {"integer", "number"} for t in expected_types
        ) and isinstance(value, (int, float)) and not isinstance(value, bool):
            problems.extend(_number_bound_problems(tool, key, value, schema))
        fmt = schema.get("format")
        if isinstance(fmt, str) and fmt:
            if isinstance(value, str) and not _format_ok(value, fmt):
                problems.append(
                    f"{tool}: argument '{key}'={value!r} is not a valid '{fmt}'"
                )
        items = schema.get("items")
        if isinstance(value, list) and isinstance(items, Mapping):
            for index, element in enumerate(value):
                problems.extend(
                    self._check_value(f"{tool}", f"{key}[{index}]", element, items, warnings)
                )
        ref = schema.get("x-ref") or schema.get("x_ref") or schema.get("ref")
        if isinstance(ref, str) and ref and not _resolve_ref(self, ref):
            warnings.append(f"{tool}: unresolvable x-ref '{ref}' for argument '{key}' (warning)")
        one_of = schema.get("x-one-of") or schema.get("oneOf")
        if one_of is not None:
            problems.extend(_check_one_of_value(tool, key, value, one_of))
        return problems

    def _check_one_of(self, tool: str, arguments: Mapping[str, Any], spec: ToolSpec) -> list[str]:
        group = spec.parameters.get("x-one-of") or spec.parameters.get("oneOf")
        if group is None:
            return []
        return _check_one_of_group(tool, arguments, group)

    # -- dependencies ------------------------------------------------------ #
    def dependency_edges(self) -> list[tuple[str, str, str]]:
        """``(tool, referenced_tool_or_ref, field)`` edges from x-ref/x-match."""
        edges: list[tuple[str, str, str]] = []
        for name, spec in sorted(self.tools.items()):
            for param, schema in spec.properties.items():
                if not isinstance(schema, Mapping):
                    continue
                for key in ("x-ref", "x_ref", "ref", "x-match", "x_match"):
                    ref = schema.get(key)
                    if isinstance(ref, str) and ref:
                        edges.append((name, ref, param))
                        break
        return edges


#: Value checks the official scorer applies. Kept byte-compatible in *semantics*
#: with ``zaic_tools.validate`` but implemented here in stdlib: this module is the
#: fallback when the SDK package is absent, so it may not import it.
#:
#: ``format`` is matched by regex, never ``datetime.fromisoformat``: the official
#: validator avoids it because Python 3.10 and 3.11 accept different inputs.
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATETIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def _format_ok(value: Any, fmt: str) -> bool:
    """True when ``value`` matches the JSON-Schema ``format`` the catalog declares.

    Mirrors the official ``format_ok``: regex first, then a real calendar check for
    ``date``/``date-time`` and hour/minute ranges for all three formats.
    """
    if not isinstance(value, str):
        return False
    if fmt == "date":
        match = _DATE_RE.match(value)
    elif fmt == "date-time":
        match = _DATETIME_RE.match(value)
    elif fmt == "time":
        if not _TIME_RE.match(value):
            return False
        hour, minute = value.split(":")[0], value.split(":")[1]
        return int(hour) < 24 and int(minute) < 60
    else:
        return True  # unknown format: never fail the caller
    if not match:
        return False
    try:
        date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return False  # 2026-02-30 is shaped like a date but is not one
    if fmt == "date-time":
        return int(value[11:13]) < 24 and int(value[14:16]) < 60
    return True


def _number_bound_problems(tool: str, key: str, value: Any, schema: Mapping[str, Any]) -> list[str]:
    """``minimum``/``maximum`` (inclusive) and the exclusive forms (strict)."""
    problems: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(f"{tool}: argument '{key}' must be a finite number")
        return problems
    for name, compare, symbol in (
        ("minimum", lambda v, b: v < b, ">="),
        ("exclusiveMinimum", lambda v, b: v <= b, ">"),
        ("maximum", lambda v, b: v > b, "<="),
        ("exclusiveMaximum", lambda v, b: v >= b, "<"),
    ):
        if name not in schema:
            continue
        bound = schema[name]
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            continue  # malformed catalog entry: do not invent a verdict
        try:
            violated = compare(value, bound)
        except TypeError:
            continue
        if violated:
            problems.append(f"{tool}: argument '{key}' must be {symbol} {bound}, got {value!r}")
    return problems


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    target = _JSON_TYPES.get(expected)
    if target is None:
        return True  # unknown type keyword: don't fail the caller
    if expected in {"number", "integer"} and isinstance(value, bool):
        return False
    return isinstance(value, target)


def _resolve_ref(catalog: ToolCatalog, ref: str) -> Any:
    """Very small ``#/...`` JSON-pointer resolver over the raw catalog."""
    if not ref:
        return None
    if not ref.startswith("#"):
        # Bare token: treat as a tool or app name.
        if ref in catalog.tools:
            return catalog.tools[ref]
        return None
    node: Any = catalog.raw
    for part in ref.lstrip("#/").split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, (list, tuple)) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, (list, tuple)):
            # Named lookup: "#/apps/smart_home/tools/device_set" over a list of apps.
            match = next(
                (
                    item
                    for item in node
                    if isinstance(item, Mapping)
                    and part in {str(item.get("app_name")), str(item.get("name")), str(item.get("id"))}
                ),
                None,
            )
            if match is None:
                return None
            node = match
        else:
            return None
    return node


def _check_one_of_value(tool: str, key: str, value: Any, group: Any) -> list[str]:
    options = group if isinstance(group, (list, tuple)) else [group]
    simple: list[Any] = []
    for option in options:
        if isinstance(option, (str, int, float, bool)) or option is None:
            simple.append(option)
    if simple and value not in simple:
        return [f"{tool}: argument '{key}'={value!r} violates x-one-of {simple}"]
    return []


def _is_empty_arg(value: Any) -> bool:
    """The official SDK's EMPTY rule: ``None``, ``""``, ``[]`` and ``{}`` count as absent."""
    return value is None or value == "" or value == [] or value == {}


def _check_one_of_group(tool: str, arguments: Mapping[str, Any], group: Any) -> list[str]:
    """Reject only when no ``x-one-of`` group is fully present.

    Mirrors ``zaic_tools.validate.schema_error`` exactly: the parameter-level
    ``x-one-of`` is a list of groups (``[["a"], ["b"]]``), and a *group* must be
    complete — extra members from other groups are fine. A flat list
    (``["a", "b"]``) is one group, so both members are required. (No official
    catalog uses the flat shape; it is kept for the synthetic fixture.)
    """
    groups = list(group) if isinstance(group, (list, tuple)) else [group]
    if not groups:
        return []
    names_groups: list[list[str]] = []
    for entry in groups:
        if isinstance(entry, str):
            names_groups.append([entry])
        elif isinstance(entry, (list, tuple)):
            members = [str(name) for name in entry if isinstance(name, str)]
            if members:
                names_groups.append(members)
    if not names_groups:
        return []
    if any(all(not _is_empty_arg(arguments.get(name)) for name in members) for members in names_groups):
        return []
    rendered = " / ".join(" + ".join(members) for members in names_groups)
    return [f"{tool}: cần một trong: {rendered}"]


def _extract_entries(data: Mapping[str, Any]) -> tuple[list[str], list[Mapping[str, Any]]]:
    """Find the apps/tools container in a loosely shaped catalog document."""
    apps: list[str] = []
    entries: list[Mapping[str, Any]] = []

    def consume(node: Any, app_hint: str = "") -> None:
        if isinstance(node, Mapping):
            if "tools" in node and isinstance(node["tools"], (list, tuple, Mapping)):
                app_name = node.get("app_name") or node.get("app") or node.get("name") or app_hint
                if app_name:
                    apps.append(str(app_name))
                for tool in _as_entry_list(node["tools"]):
                    if app_hint and "app_name" not in tool and "app" not in tool:
                        tool = {**tool, "app_name": str(app_name)}
                    entries.append(tool)
                return
            for key, value in node.items():
                if isinstance(value, (Mapping, list, tuple)):
                    consume(value, app_hint or str(key))
        elif isinstance(node, (list, tuple)):
            for item in node:
                consume(item, app_hint)

    for key in ("apps", "applications", "tool_apps", "namespaces"):
        if key in data and isinstance(data[key], (list, tuple, Mapping)):
            consume(data[key])
    if "tools" in data:
        for tool in _as_entry_list(data["tools"]):
            entries.append(tool)
    if not entries:
        consume(data)
    explicit_apps = data.get("app_list") or data.get("available_apps")
    if isinstance(explicit_apps, (list, tuple)):
        apps.extend(str(a) for a in explicit_apps)
    return apps, entries


def _as_entry_list(node: Any) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    if isinstance(node, Mapping):
        if "name" in node or "tool_name" in node:
            out.append(dict(node))
        else:
            for key, value in node.items():
                if isinstance(value, Mapping):
                    out.append({**dict(value), "name": value.get("name", key)})
                elif isinstance(value, str):
                    out.append({"name": key, "description": value})
        return out
    if isinstance(node, (list, tuple)):
        for item in node:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif isinstance(item, str):
                out.append({"name": item})
    return out


def _spec_from_entry(entry: Mapping[str, Any]) -> ToolSpec | None:
    if "function" in entry and isinstance(entry["function"], Mapping):
        fn = dict(entry["function"])
        entry = {**entry, **fn}
    name = entry.get("name") or entry.get("tool_name") or entry.get("id")
    if not name:
        return None
    app_name, full_name = _split_tool_name(str(name), entry)
    parameters = entry.get("parameters") or entry.get("input_schema") or entry.get("schema") or {}
    if not isinstance(parameters, Mapping):
        parameters = {}
    parameters = dict(parameters)
    # Normalize OpenAI-style {"type": "object", "properties": ...} at top level.
    if "properties" not in parameters and isinstance(parameters.get("params"), Mapping):
        parameters = {"type": "object", "properties": dict(parameters["params"])}
    annotations = entry.get("annotations") or {}
    if not isinstance(annotations, Mapping):
        annotations = {}
    description = str(entry.get("description") or entry.get("desc") or "")
    return ToolSpec(
        name=full_name,
        app_name=app_name,
        description=description,
        parameters=parameters,
        kind=_infer_kind(full_name, description, annotations, entry.get("kind")),
        annotations=dict(annotations),
    )
