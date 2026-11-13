"""Tool environments: official simulator adapter and deterministic mock."""

from __future__ import annotations

from .base import ToolEnvironment, ValidationOutcome
from .factory import create_environment
from .mock import MockToolEnv

__all__ = [
    "ToolEnvironment",
    "ValidationOutcome",
    "MockToolEnv",
    "create_environment",
]


def __getattr__(name: str):
    """Lazily expose ``OfficialToolEnv`` without importing the SDK eagerly."""
    if name == "OfficialToolEnv":
        from .official import OfficialToolEnv

        return OfficialToolEnv
    raise AttributeError(name)
