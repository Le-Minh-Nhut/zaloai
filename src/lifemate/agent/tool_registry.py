"""Tool visibility registry: what the agent is allowed to call for one sample.

Wraps a ``lifemate.data.catalog.ToolCatalog`` and narrows it to the apps
installed in a sample's ``Context``. Tools of apps that are not installed MUST
never reach the prompt or the parser — that leak is the single easiest way to
lose points in this competition.

The catalog is used duck-typed so this module stays importable (and testable)
without the data layer being present.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Meta tool always exposed so the agent can ask for missing information.
META_TOOL_NAME = "sys.ask_missing_info"


def _spec_payload(spec: Any) -> dict[str, Any]:
    """Raw schema dict of a ``ToolSpec`` (falls back to its public fields)."""
    raw = getattr(spec, "raw", None)
    if isinstance(raw, dict) and raw:
        return dict(raw)
    return {
        "name": getattr(spec, "name", None),
        "app_name": getattr(spec, "app_name", None),
        "description": getattr(spec, "description", ""),
        "parameters": getattr(spec, "parameters", {}) or {},
        "kind": getattr(spec, "kind", "unknown"),
    }


class ToolRegistry:
    """Filtered, read-only view over a tool catalog."""

    def __init__(self, catalog: Any, apps: Sequence[str] | None = None) -> None:
        if catalog is None:
            raise ValueError("ToolRegistry requires a ToolCatalog instance")
        self.catalog = catalog
        self.apps: list[str] = list(apps) if apps is not None else list(
            getattr(catalog, "apps", []) or []
        )
        self.specs: list[Any] = self._collect_specs()

    # ------------------------------------------------------------------ build
    def _collect_specs(self) -> list[Any]:
        """Tools of the installed apps plus the meta tool when the catalog has it."""
        tools: dict[str, Any] = getattr(self.catalog, "tools", {}) or {}
        selected: list[Any] = []
        for name, spec in tools.items():
            if name == META_TOOL_NAME or getattr(spec, "kind", None) == "meta":
                selected.append(spec)
            elif getattr(spec, "app_name", None) in self.apps:
                selected.append(spec)
        return sorted(selected, key=lambda s: str(getattr(s, "name", "")))

    def for_sample(self, sample: Any) -> "ToolRegistry":
        """Registry filtered to ``sample.context.installed_apps`` (+ meta tool)."""
        context = sample.get("context") if isinstance(sample, dict) else getattr(sample, "context", None)
        apps = (
            context.get("installed_apps")
            if isinstance(context, dict)
            else getattr(context, "installed_apps", None)
        )
        return ToolRegistry(self.catalog, apps=list(apps or []))

    # ------------------------------------------------------------------- query
    def tool_names(self) -> list[str]:
        """Names of the visible tools (installed apps + meta)."""
        return [str(getattr(spec, "name", "")) for spec in self.specs]

    @property
    def documents(self) -> list[dict[str, Any]]:
        """Raw schema dicts, safe to embed in a prompt or a tool-calling payload."""
        return [_spec_payload(spec) for spec in self.specs]

    def get(self, name: str) -> Any | None:
        """Visible ``ToolSpec`` by name, or ``None``."""
        for spec in self.specs:
            if getattr(spec, "name", None) == name:
                return spec
        return None

    def is_available(self, name: str) -> bool:
        """Whether ``name`` may be called for this sample."""
        return isinstance(name, str) and name in self.tool_names()

    def validate(self, name: str, args: dict[str, Any]) -> dict[str, list[str]]:
        """Delegate argument validation to the catalog (no side effects)."""
        if not self.is_available(name):
            return {"errors": [f"tool {name!r} is not available for this sample"], "warnings": []}
        outcome = self.catalog.validate_call(name, args if isinstance(args, dict) else {})
        if not isinstance(outcome, dict):
            return {"errors": [], "warnings": []}
        return {
            "errors": list(outcome.get("errors") or []),
            "warnings": list(outcome.get("warnings") or []),
        }

    def render(self) -> str:
        """Schema prompt text, catalog-rendered but scoped to the installed apps."""
        return str(self.catalog.to_schema_prompt(apps=self.apps, include_meta=True))

    def as_set(self) -> set[str]:
        """Visible tool names as a set (membership checks in hot paths)."""
        return set(self.tool_names())

    def __len__(self) -> int:
        return len(self.specs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolRegistry(apps={self.apps!r}, tools={self.tool_names()!r})"


def registry_for_sample(catalog: Any, sample: Any) -> ToolRegistry:
    """Convenience: build the per-sample registry in one call."""
    return ToolRegistry(catalog).for_sample(sample)


__all__ = ["META_TOOL_NAME", "ToolRegistry", "registry_for_sample"]
