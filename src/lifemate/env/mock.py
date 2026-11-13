"""Deterministic in-memory tool environment for tests and dry runs.

State comes ONLY from the synthetic dict passed by the caller. This class
never reads ``*_environment`` directories — the competition simulator is
off-limits during development.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from ..data.catalog import ToolCatalog
from .base import ValidationOutcome

__all__ = ["MockToolEnv"]

META_TOOLS = frozenset({"sys.ask_missing_info", "ask_missing_info", "ask_missing"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class MockToolEnv:
    """Simulated tool backend with call log and state hashing."""

    def __init__(
        self,
        catalog: ToolCatalog | None = None,
        initial_state: dict | None = None,
        apps: list[str] | None = None,
        sample_id: str = "mock",
        fail_tools: dict[str, int] | None = None,
    ) -> None:
        self.name = "mock"
        self.catalog = catalog
        self.sample_id = sample_id
        self._state: dict = copy.deepcopy(initial_state or {})
        self._state.setdefault("apps", {})
        self._state.setdefault("writes", [])
        self._apps: list[str] = list(
            apps
            if apps is not None
            else (catalog.apps if catalog is not None else list(self._state.get("apps", {}).keys()))
        )
        self._log: list[dict] = []
        self._fail_remaining: dict[str, int] = dict(fail_tools or {})

    # -- interface --------------------------------------------------------- #
    def available_tools(self) -> list[str]:
        """Tool names from the catalog, filtered to installed apps."""
        if self.catalog is None:
            return []
        return [spec.name for spec in self.catalog.tools_for_apps(self._apps)]

    def validate(self, tool_name: str, arguments: dict) -> ValidationOutcome:
        """Schema-check a call without touching state."""
        if self.catalog is None:
            if not isinstance(arguments, Mapping):
                return ValidationOutcome.from_errors([f"{tool_name}: arguments must be an object"])
            return ValidationOutcome()
        report = self.catalog.validate_call(tool_name, arguments)
        return ValidationOutcome.from_errors(list(report["errors"]))

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Execute (or reject) a call and append it to the log verbatim."""
        args = dict(arguments or {}) if isinstance(arguments, Mapping) else {}
        if self.catalog is not None and tool_name not in self.catalog.tools:
            return self._record(tool_name, args, ok=False, error=f"unknown tool: {tool_name}")
        outcome = self.validate(tool_name, args)
        if not outcome.ok:
            return self._record(tool_name, args, ok=False, error="; ".join(outcome.errors))
        remaining = self._fail_remaining.get(tool_name, 0)
        if remaining > 0:
            self._fail_remaining[tool_name] = remaining - 1
            return self._record(
                tool_name, args, ok=False, error=f"simulated failure for {tool_name} ({remaining} remaining)"
            )
        spec = None if self.catalog is None else self.catalog.get(tool_name)
        kind = "unknown" if spec is None else spec.kind
        if tool_name in META_TOOLS or kind == "meta":
            return self._record(tool_name, args, ok=True, result={"question": self._missing_hint(args)})
        if kind == "read":
            return self._record(tool_name, args, ok=True, result=self._read(tool_name, args, spec))
        result = self._write(tool_name, args, spec)
        return self._record(tool_name, args, ok=True, result=result)

    def get_tool_calls(self) -> list[dict]:
        """Log of executed calls (copies; safe to mutate)."""
        return copy.deepcopy(self._log)

    def get_state_hash(self) -> str:
        """sha256 of canonical JSON state (excluding the write transcript)."""
        payload = {key: value for key, value in self._state.items() if key != "writes"}
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    # -- helpers ----------------------------------------------------------- #
    @property
    def state(self) -> dict:
        """Copy of the current world state."""
        return copy.deepcopy(self._state)

    def _record(self, name: str, args: dict, *, ok: bool, result: Any = None, error: str | None = None) -> dict:
        entry: dict[str, Any] = {"name": name, "arguments": copy.deepcopy(args), "ok": ok}
        if ok:
            entry["result"] = copy.deepcopy(result)
        else:
            entry["error"] = error
        self._log.append(entry)
        return copy.deepcopy(entry)

    def _app_of(self, tool_name: str) -> str:
        if "." in tool_name:
            return tool_name.split(".")[0]
        return tool_name

    def _read(self, tool_name: str, args: Mapping[str, Any], spec: Any) -> Any:
        app = "" if spec is None else spec.app_name or self._app_of(tool_name)
        bucket = self._state.get("apps", {}).get(app, {})
        if not isinstance(bucket, Mapping):
            return bucket
        identifiers = [str(value) for value in args.values() if isinstance(value, (str, int))]
        if identifiers:
            for key in identifiers:
                if key in bucket:
                    return copy.deepcopy(bucket[key])
        return copy.deepcopy(dict(bucket))

    def _write(self, tool_name: str, args: Mapping[str, Any], spec: Any) -> Any:
        app = "" if spec is None else spec.app_name or self._app_of(tool_name)
        apps = self._state.setdefault("apps", {})
        bucket = apps.setdefault(app, {})
        if not isinstance(bucket, dict):  # keep the state hashable/serialisable
            bucket = {"value": bucket}
            apps[app] = bucket
        key = self._write_key(tool_name, args)
        bucket[key] = copy.deepcopy(dict(args))
        self._state.setdefault("writes", []).append(
            {"tool": tool_name, "app": app, "key": key, "arguments": copy.deepcopy(dict(args))}
        )
        return {"ok": True, "app": app, "key": key, "arguments": copy.deepcopy(dict(args))}

    def _write_key(self, tool_name: str, args: Mapping[str, Any]) -> str:
        for candidate in ("id", "device_id", "event_id", "alarm_id", "message_id", "contact_id", "name", "title"):
            value = args.get(candidate)
            if isinstance(value, (str, int)) and str(value):
                return str(value)
        return tool_name.split(".")[-1]

    def _missing_hint(self, args: Mapping[str, Any]) -> str:
        missing = args.get("missing_fields") or args.get("fields") or args.get("question")
        if missing:
            return f"Please provide: {missing}"
        return "Please provide the missing information."
