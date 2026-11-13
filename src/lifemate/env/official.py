"""Adapter over the organizers' ``zaic_tools`` simulator.

The dataset ships ONE COPY PER SPLIT (``<root>/<split>/zaic_tools``) and their
``apps/`` differ, so the import must come from the requested split's directory.
``sys.path`` is prepended with ``<root>/<split>`` and the imported module's
origin is verified; a module left in ``sys.modules`` by an earlier split is
evicted before re-importing, because Python caches the first ``zaic_tools``.

There is deliberately NO fallback to :class:`~lifemate.env.mock.MockToolEnv`:
competition inference must fail loudly rather than silently score against a
simulator that is not the official one.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from ..data.catalog import ToolCatalog
from ..errors import LifeMateError
from .base import ValidationOutcome

__all__ = ["OfficialToolEnv", "ZAIC_MODULE_PREFIX", "check_official_sdk"]

ZAIC_MODULE_PREFIX = "zaic_tools"
MISSING_SDK_MESSAGE = (
    "zaic_tools not installed: the official LifeMate simulator SDK is required for "
    "real evaluation. It ships per split at <data_root>/<split>/zaic_tools; pass a "
    "data_root that contains the split directory, or use_mock=True for a synthetic "
    "dry run (never for a competition submission)."
)

def check_official_sdk(split: str, data_root: Any) -> None:
    """Preflight: the requested split's ``zaic_tools`` is importable. Creates no session.

    ``ToolEnv`` may be constructed only ONCE per sample (organizers' rule), so a
    caller that merely wants to fail fast must use this, never a throwaway env.
    """
    _import_split_sdk(split, data_root)


def _import_split_sdk(split: str, data_root: Any):
    """Import ``zaic_tools`` from ``<data_root>/<split>/zaic_tools`` and verify it."""
    split_dir = Path(data_root) / split
    expected = (split_dir / ZAIC_MODULE_PREFIX).resolve()
    if not split_dir.is_dir() or not expected.is_dir():
        # Exact message identity matters: callers/tests match on MISSING_SDK_MESSAGE.
        raise LifeMateError(MISSING_SDK_MESSAGE)
    start = str(split_dir)
    # Hoist (not just append): a previous split's directory is already on the
    # path and would win the import.
    while start in sys.path:
        sys.path.remove(start)
    sys.path.insert(0, start)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(ZAIC_MODULE_PREFIX)
    except ImportError as exc:
        raise LifeMateError(MISSING_SDK_MESSAGE) from exc
    origin = Path(getattr(module, "__file__", "") or "").resolve().parent
    if origin != expected:
        # Python cached another split's copy: drop it (and its submodules) and
        # retry, so processing train then public_test in one process is correct.
        for name in [n for n in sys.modules if n == ZAIC_MODULE_PREFIX or n.startswith(ZAIC_MODULE_PREFIX + ".")]:
            del sys.modules[name]
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(ZAIC_MODULE_PREFIX)
        except ImportError as exc:
            raise LifeMateError(
                f"{MISSING_SDK_MESSAGE} Expected the split's copy at {expected}."
            ) from exc
        origin = Path(getattr(module, "__file__", "") or "").resolve().parent
    if origin != expected:
        raise LifeMateError(
            f"zaic_tools was imported from {origin} but split {split!r} needs {expected}; "
            "the app sets differ per split, so refusing to run with the wrong simulator copy"
        )
    return module


class OfficialToolEnv:
    """Thin delegating wrapper around ``zaic_tools.ToolEnv``."""

    name = "official"

    def __init__(
        self,
        split: str,
        sample_id: str,
        data_root: str | Any,
        catalog: ToolCatalog | None = None,
    ) -> None:
        zaic_tools = _import_split_sdk(split, data_root)
        tool_env_cls = getattr(zaic_tools, "ToolEnv", None)
        if tool_env_cls is None:  # pragma: no cover - only with a broken SDK
            raise LifeMateError("zaic_tools is installed but exposes no ToolEnv class")
        self.split = split
        self.sample_id = sample_id
        self.data_root = data_root
        self.catalog = catalog
        self._env = tool_env_cls(split, sample_id, root=data_root)
        module = type(self._env).__module__ or ""
        if not module.startswith(ZAIC_MODULE_PREFIX):  # pragma: no cover
            raise LifeMateError(
                f"OfficialToolEnv got an unexpected backend {type(self._env)!r} from module {module!r}; "
                "refusing to run with a non-official simulator"
            )

    # -- interface --------------------------------------------------------- #
    def validate(self, tool_name: str, arguments: dict) -> ValidationOutcome:
        """Delegate to the SDK's validator when present, else trust the catalog."""
        validate = getattr(self._env, "validate", None)
        if callable(validate):
            raw = validate(tool_name, arguments)
            if isinstance(raw, ValidationOutcome):
                return raw
            if isinstance(raw, dict):
                errors = list(raw.get("errors") or [])
                # The official ``validate`` returns {"ok": bool, "error": str|None}
                # with no "errors" list; keep its message so ok=False is never
                # reported with an empty explanation.
                message = raw.get("error")
                if message:
                    errors.append(str(message))
                return ValidationOutcome(ok=bool(raw.get("ok", not errors)), errors=errors)
            if isinstance(raw, (list, tuple)):
                return ValidationOutcome.from_errors([str(item) for item in raw])
            if isinstance(raw, bool):
                return ValidationOutcome(ok=raw, errors=[] if raw else ["invalid arguments"])
        if self.catalog is not None:
            report = self.catalog.validate_call(tool_name, arguments)
            return ValidationOutcome.from_errors(list(report["errors"]))
        return ValidationOutcome()

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Delegate one call to the SDK."""
        return self._env.call(tool_name, arguments)

    def get_tool_calls(self) -> list[dict]:
        """Delegate the official call transcript."""
        return list(self._env.get_tool_calls())

    def get_state_hash(self) -> str:
        """Delegate the official state hash."""
        return str(self._env.get_state_hash())

    def get_state(self) -> dict:
        """Copy of the official per-app state (train-time verification only)."""
        getter = getattr(self._env, "get_state", None)
        return getter() if callable(getter) else {}

    def available_tools(self) -> list[str]:
        """Tool names this sample may call (the session's installed apps)."""
        getter = getattr(self._env, "available_tools", None)
        if callable(getter):  # pragma: no cover - depends on SDK
            return [str(name) for name in getter()]
        allowed = getattr(self._env, "allowed", None)
        if allowed:  # zaic_tools Session exposes the per-sample allow-list
            return sorted(str(name) for name in allowed)
        if self.catalog is not None:
            return sorted(self.catalog.tools)
        return []
