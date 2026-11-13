"""LifeMate Track 1 reference implementation (voice -> tool-calling agent).

Heavy dependencies (torch, transformers, peft, soundfile) are imported lazily
inside functions so that importing this package never requires them, and so
CPU-only tests run on a bare ``python3``.
"""

from __future__ import annotations

from .errors import (
    ConfigError,
    DatasetNotFoundError,
    LifeMateError,
    ProhibitedAccessError,
    SchemaError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LifeMateError",
    "DatasetNotFoundError",
    "ProhibitedAccessError",
    "ConfigError",
    "SchemaError",
]


def __getattr__(name: str):
    """Lazy access to submodule symbols (keeps import cost near zero)."""
    if name == "load_config":
        from .config import load_config

        return load_config
    raise AttributeError(name)
