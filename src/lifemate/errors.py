"""Exception hierarchy for LifeMate.

Every error raised by this package derives from :class:`LifeMateError` so
callers (scripts, notebooks, workers) can catch a single base type.
"""

from __future__ import annotations

__all__ = [
    "LifeMateError",
    "DatasetNotFoundError",
    "ProhibitedAccessError",
    "ConfigError",
    "SchemaError",
    "FeatureConfigurationError",
]


class LifeMateError(Exception):
    """Base class for all LifeMate errors."""


_NOT_FOUND_TEMPLATE = (
    "Dataset not found.\n"
    "\n"
    "Expected directory:\n"
    "{path}\n"
    "\n"
    "Please extract the official LifeMate dataset\n"
    "into the configured data directory.\n"
    "\n"
    "Implementation is available.\n"
    "Real-data validation has not yet been performed."
)


class DatasetNotFoundError(LifeMateError):
    """Raised when a requested split directory does not exist.

    The message is fixed text (deliberately, so the same wording appears in
    every CLI/report); it never contains a traceback or a guess at the data.
    """

    def __init__(self, split: str, expected_path: str) -> None:
        self.split = split
        self.expected_path = str(expected_path)
        super().__init__(self.__str__())

    def __str__(self) -> str:  # noqa: D105 - exact wording is contractual
        return _NOT_FOUND_TEMPLATE.format(path=self.expected_path)


class ProhibitedAccessError(LifeMateError):
    """Raised when code tries to touch a ``*_environment`` path.

    The competition dataset ships a per-split simulator directory. Reading it
    at inference/training time is prohibited; this error is the guard.
    """

    def __init__(self, path: object, reason: str = "prohibited *_environment path") -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(f"{reason}: {self.path}")


class ConfigError(LifeMateError):
    """Raised for invalid configuration values or unknown override keys."""


class SchemaError(LifeMateError):
    """Raised when a record cannot be interpreted even tolerantly."""


class FeatureConfigurationError(LifeMateError):
    """Raised for an invalid feature-flag combination or override.

    Subclasses :class:`LifeMateError`, so the scripts' existing
    ``except LifeMateError`` handlers already turn it into their normal failure
    exit code; :func:`lifemate.features.registry.validate_or_raise` only adds
    the actionable, flag-naming message.
    """
