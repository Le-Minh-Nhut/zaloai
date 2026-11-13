"""Small logging helpers: one configured root logger, deterministic format."""

from __future__ import annotations

import logging
import os

__all__ = ["get_logger", "configure_logging", "DEFAULT_FORMAT", "LOG_LEVEL_ENV"]

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_LEVEL_ENV = "LIFEMATE_LOG_LEVEL"

_CONFIGURED = False


def configure_logging(level: str | int | None = None, fmt: str = DEFAULT_FORMAT) -> logging.Logger:
    """Configure the package root logger once; safe to call repeatedly."""
    global _CONFIGURED
    log = logging.getLogger("lifemate")
    if level is None:
        level = os.environ.get(LOG_LEVEL_ENV, "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    log.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(handler)
        log.propagate = False
        _CONFIGURED = True
    return log


def get_logger(name: str = "lifemate") -> logging.Logger:
    """Return a logger under the ``lifemate`` namespace."""
    if not name.startswith("lifemate"):
        name = f"lifemate.{name}"
    return logging.getLogger(name)
