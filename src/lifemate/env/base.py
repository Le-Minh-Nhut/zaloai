"""Environment interface shared by the mock and official tool environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["ValidationOutcome", "ToolEnvironment"]


@dataclass
class ValidationOutcome:
    """Result of pre-flight argument checking for one call."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_errors(cls, errors: list[str]) -> "ValidationOutcome":
        """Build an outcome from an error list (``ok`` is ``not errors``)."""
        return cls(ok=not errors, errors=list(errors))

    def to_dict(self) -> dict:
        """Plain dict form."""
        return {"ok": self.ok, "errors": list(self.errors)}


@runtime_checkable
class ToolEnvironment(Protocol):
    """What the agent loop needs from a tool backend.

    Implementations must be deterministic given the same call sequence so
    that scoring is reproducible.
    """

    name: str

    def validate(self, tool_name: str, arguments: dict) -> ValidationOutcome:
        """Check arguments against the tool schema without side effects."""

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Execute a call; returns ``{"ok": True, "result": ...}`` or ``{"ok": False, "error": ...}``."""

    def get_tool_calls(self) -> list[dict]:
        """Full transcript of executed calls, in order."""

    def get_state_hash(self) -> str:
        """Stable hash of the environment's world state."""

    def available_tools(self) -> list[str]:
        """Tool names this environment can execute."""
