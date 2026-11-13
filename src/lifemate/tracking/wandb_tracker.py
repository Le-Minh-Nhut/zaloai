"""Weights & Biases logging with redaction and hard offline safety.

``wandb`` is imported lazily and *optionally*: when the package is missing, the
run is disabled, or the network is unavailable, every call degrades to a no-op
that logs a warning once. Logging must never break a training/eval run.

PII policy: keys matching ``api_key|token|secret|password`` are replaced with
``"[REDACTED]"``, and ``user_profile`` / ``messages`` / ``audio_text`` (and the
other keys in :data:`PII_KEYS`) are dropped outright unless
``allow_pii=True`` is passed explicitly.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
from typing import Any, Mapping

__all__ = [
    "PII_KEYS",
    "REDACTED",
    "SECRET_PATTERNS",
    "WandbTracker",
    "describe_environment",
    "sanitize",
    "wandb_available",
]

log = logging.getLogger("lifemate.tracking")

REDACTED = "[REDACTED]"

#: Regexes for keys that must never reach a log line.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"passwd", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
)

#: Dataset text keys that carry user content; dropped unless explicitly allowed.
PII_KEYS = frozenset(
    {
        "user_profile",
        "messages",
        "audio_text",
        "transcript",
        "transcript_text",
        "conversation",
        "history",
        "content",
        "raw_text",
        "phone",
        "email",
        "address",
        "full_name",
        "name",
    }
)


def _is_secret_key(key: str) -> bool:
    return any(pattern.search(str(key)) for pattern in SECRET_PATTERNS)


def sanitize(payload: Any, *, allow_pii: bool = False) -> Any:
    """Recursively redact secrets and drop PII keys from a log payload."""
    if isinstance(payload, Mapping):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            name = str(key)
            if allow_pii is False and name in PII_KEYS:
                out[name] = "[DROPPED_PII]"
                continue
            if _is_secret_key(name):
                out[name] = REDACTED
                continue
            out[name] = sanitize(value, allow_pii=allow_pii)
        return out
    if isinstance(payload, (list, tuple)):
        return [sanitize(item, allow_pii=allow_pii) for item in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        if isinstance(payload, str) and len(payload) > 4000:  # keep log lines small
            return payload[:4000] + "...[truncated]"
        return payload
    return str(payload)[:500]


def wandb_available() -> bool:
    """True when ``import wandb`` succeeds (no network probe, no login)."""
    try:
        import wandb  # type: ignore  # noqa: F401

        # A local ``wandb/`` run directory is importable as an empty namespace
        # package; require the real API so that does not look like the library.
        return hasattr(wandb, "init")
    except Exception:
        return False


def describe_environment() -> dict[str, Any]:
    """Dependency/toolchain facts for a run. No host, no user, no machine model.

    An allowlist, not a redaction pass: only keys that cannot identify the
    operator or the machine leave this function. ``hostname``/``user`` are
    deliberately absent (a workstation hostname frequently contains its owner's
    name), ``platform`` is reduced to the OS family, and ``gpu`` is the compute
    capability class (``sm_120``) rather than the card's marketing name.
    """
    info: dict[str, Any] = {
        "platform": sys.platform,
        "python": platform.python_version(),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        info["git_commit"] = commit or "unavailable"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        info["git_dirty"] = bool(status)
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = "unavailable"
        info["git_dirty"] = None
    try:
        import torch  # type: ignore

        info["torch"] = getattr(torch, "__version__", "unavailable")
        info["cuda"] = getattr(getattr(torch, "version", None), "cuda", None) or "unavailable"
        available = bool(torch.cuda.is_available())
        info["cuda_available"] = available
        if available:
            major, minor = torch.cuda.get_device_capability(0)
            info["gpu"] = f"cuda-sm_{major}{minor}"
        else:
            info["gpu"] = "unavailable"
    except Exception:
        info["torch"] = "unavailable"
        info["cuda"] = "unavailable"
        info["cuda_available"] = False
        info["gpu"] = "unavailable"
    try:
        import transformers  # type: ignore

        info["transformers"] = getattr(transformers, "__version__", "unavailable")
    except Exception:
        info["transformers"] = "unavailable"
    return info


class WandbTracker:
    """Thin W&B wrapper whose disabled form has the exact same interface.

    ``enabled`` comes from ``cfg.wandb.enabled`` and is additionally forced off
    when ``wandb`` is not importable or ``WANDB_MODE=disabled`` is set.
    """

    def __init__(
        self,
        cfg: Any = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
        *,
        allow_pii: bool = False,
        config: Mapping[str, Any] | None = None,
        **init_kwargs: Any,
    ) -> None:
        wandb_cfg = getattr(cfg, "wandb", None)
        self.requested = bool(getattr(wandb_cfg, "enabled", False))
        self.project = getattr(wandb_cfg, "project", None) or "lifemate"
        self.entity = getattr(wandb_cfg, "entity", None)
        # Unset => let ``wandb`` pick its own default (never force offline silently).
        self.mode = os.environ.get("WANDB_MODE") or getattr(wandb_cfg, "mode", None)
        self.run_name = run_name or getattr(wandb_cfg, "run_name", None)
        self.tags = list(tags or [])
        self.allow_pii = bool(allow_pii)
        self.dir = getattr(wandb_cfg, "dir", None)
        self._run: Any = None
        self._warned = False
        self.disabled_reason: str | None = None
        self.summary: dict[str, Any] = {}

        if not self.requested:
            self.enabled = False
            self.disabled_reason = "wandb.enabled is false (no-op tracker)"
            return
        if not wandb_available():
            self.enabled = False
            self.disabled_reason = "wandb package not installed"
            log.warning("wandb logging requested but the package is not installed; continuing without it")
            return
        if self.mode == "disabled":
            self.enabled = False
            self.disabled_reason = "WANDB_MODE=disabled"
            return

        try:
            import wandb  # type: ignore

            payload = sanitize(dict(config or {}), allow_pii=self.allow_pii)
            payload.setdefault("environment", describe_environment())
            init_kwargs.setdefault("project", self.project)
            if self.entity:
                init_kwargs.setdefault("entity", self.entity)
            if self.run_name:
                init_kwargs.setdefault("name", self.run_name)
            if self.tags:
                init_kwargs.setdefault("tags", self.tags)
            if self.mode:
                init_kwargs.setdefault("mode", self.mode)
            if self.dir:
                init_kwargs.setdefault("dir", self.dir)
            self._run = wandb.init(config=payload, **init_kwargs)
            self.enabled = True
        except Exception as exc:  # offline, bad key, proxy failure ...
            self.enabled = False
            self.disabled_reason = f"wandb.init failed: {type(exc).__name__}: {exc}"[:300]
            log.warning("wandb unavailable, continuing without logging: %s", self.disabled_reason)

    # -- interface --------------------------------------------------------- #
    @property
    def url(self) -> str | None:
        """Run URL, or ``None`` when logging is disabled."""
        if not self.enabled or self._run is None:
            return None
        return getattr(self._run, "url", None) or getattr(self._run, "get_url", lambda: None)()

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        """Log one dict of scalars (sanitised)."""
        if not self.enabled or self._run is None:
            self._remember(data)
            return
        try:
            self._run.log(sanitize(dict(data), allow_pii=self.allow_pii), step=step)
        except Exception as exc:
            self._fail(exc)

    def log_metrics(self, data: Mapping[str, Any], step: int | None = None) -> None:
        """Alias for :meth:`log` (explicit name for metric dicts)."""
        self.log(data, step=step)

    def log_summary(self, data: Mapping[str, Any]) -> None:
        """Write final values into the run summary."""
        clean = sanitize(dict(data), allow_pii=self.allow_pii)
        self.summary.update(clean)
        if not self.enabled or self._run is None:
            return
        try:
            for key, value in clean.items():
                self._run.summary[key] = value
        except Exception as exc:
            self._fail(exc)

    def log_config(self, data: Mapping[str, Any]) -> None:
        """Merge extra config (redacted) into the run."""
        clean = sanitize(dict(data), allow_pii=self.allow_pii)
        if not self.enabled or self._run is None:
            return
        try:
            config = getattr(self._run, "config", None)
            if config is not None:
                for key, value in clean.items():
                    config[key] = value
        except Exception as exc:
            self._fail(exc)

    def finish(self) -> None:
        """Close the run; never raises."""
        if not self.enabled:
            return
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as exc:  # pragma: no cover - depends on wandb
            self._fail(exc)
        finally:
            self.enabled = False

    def __enter__(self) -> "WandbTracker":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.finish()

    # -- internals --------------------------------------------------------- #
    def _remember(self, data: Mapping[str, Any]) -> None:
        """Keep a local copy of what would have been logged (auditable)."""
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                self.summary[str(key)] = sanitize(value, allow_pii=self.allow_pii)

    def _fail(self, exc: Exception) -> None:
        if not self._warned:
            log.warning("wandb call failed, disabling logging: %s: %s", type(exc).__name__, exc)
            self._warned = True
        self.enabled = False
        self.disabled_reason = f"{type(exc).__name__}: {exc}"[:300]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "enabled" if self.enabled else f"disabled ({self.disabled_reason})"
        return f"WandbTracker({state}, project={self.project!r})"
