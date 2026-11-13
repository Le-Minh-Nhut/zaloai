"""Small training callbacks: interrupt handling and tracker plumbing.

Kept separate from :mod:`lifemate.training.trainer` so the trainer loop stays
readable and the interrupt path is testable without torch.

``StopController`` gives "stop at the next safe point" semantics: a signal sets
a flag, the trainer finishes the optimizer step it is in, checkpoints, and then
exits. Nothing is killed mid-step, so a checkpoint is never a half-updated
optimizer state.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["JsonlTracker", "LoggingCallback", "NullTracker", "StopController"]

class StopController:
    """Cooperative SIGINT/SIGTERM handling.

    ``install()`` registers handlers for SIGINT and SIGTERM and returns the
    controller. ``requested`` is a plain flag (a lock guards the write, so it is
    also safe when a handler fires on another thread). ``uninstall()`` restores
    the previous handlers — call it in a ``finally`` block.

    Only valid in the main thread; ``install()`` is a no-op elsewhere because
    ``signal.signal`` raises there.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._previous: dict[int, Any] = {}
        self._installed = False
        self.received_signal: int | None = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, signum: int | None = None) -> None:
        """Set the flag (public so tests and non-signal callers can trigger it)."""
        with self._lock:
            if signum is not None:
                self.received_signal = int(signum)
            self._event.set()

    def clear(self) -> None:
        with self._lock:
            self.received_signal = None
            self._event.clear()

    def install(self) -> "StopController":
        """Register handlers for SIGINT/SIGTERM (no-op off the main thread)."""
        if self._installed or threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (ValueError, OSError):  # pragma: no cover - restricted envs
                continue
        self._installed = True
        return self

    def _handle(self, signum: int, _frame: Any) -> None:  # pragma: no cover - signal path
        self.request(signum)

    def uninstall(self) -> None:
        """Restore the handlers that were active before ``install()``."""
        if not self._installed:
            return
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):  # pragma: no cover
                continue
        self._previous.clear()
        self._installed = False

    def __enter__(self) -> "StopController":
        return self.install()

    def __exit__(self, *_exc: Any) -> None:
        self.uninstall()

class NullTracker:
    """Tracker that swallows metrics (default when W&B is disabled)."""

    def log(self, metrics: Mapping[str, Any]) -> None:
        """Accept and discard a metrics dict."""

    def finish(self) -> None:
        """No-op (kept for API symmetry with real trackers)."""

class JsonlTracker:
    """Append-only JSONL metrics logger used when W&B is unavailable.

    Fallback of last resort for ``--report-to wandb`` runs where the tracking
    package cannot be imported: metrics are appended to
    ``<output_dir>/<run_name>/metrics.jsonl`` instead of being dropped. Payloads
    are redacted with :func:`lifemate.tracking.sanitize` when that module is
    importable; otherwise only scalar values are kept (no nested config or text
    can leak through a metrics line).
    """

    def __init__(self, output_dir: str | Path, run_name: str = "run") -> None:
        self.path = Path(str(output_dir)) / str(run_name) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._warned = False

    @staticmethod
    def _scrub(data: Mapping[str, Any]) -> dict[str, Any]:
        """Redact secrets/PII; fall back to scalars only when tracking is absent."""
        try:
            from lifemate.tracking import sanitize  # noqa: PLC0415 - optional

            cleaned = sanitize(dict(data))
            return dict(cleaned) if isinstance(cleaned, Mapping) else {}
        except Exception:
            return {
                key: value
                for key, value in data.items()
                if isinstance(value, (int, float, bool, str)) or value is None
            }

    def log(self, metrics: Mapping[str, Any]) -> None:
        """Append one redacted metrics line (flushed + fsynced); warns once on error."""
        line = json.dumps(self._scrub(metrics), ensure_ascii=False, sort_keys=True, default=str) + "\n"
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            if not self._warned:
                self._warned = True
                print(f"warning: could not append metrics to {self.path}: {exc}")

    def finish(self) -> None:
        """No-op: every record is already durable in the JSONL file."""


class LoggingCallback:
    """Print metrics every ``every_n_steps``; injectable ``write`` for tests."""

    def __init__(self, every_n_steps: int = 10, write: Callable[[str], Any] = print) -> None:
        self.every_n_steps = max(1, int(every_n_steps))
        self.write = write

    def on_log(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Print one compact line when ``step`` is a logging step."""
        if step % self.every_n_steps != 0:
            return
        parts = " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
        self.write(f"[step {step}] {parts}")

    def on_train_end(self, result: Any) -> None:
        """Print the final summary line."""
        self.write(
            f"[done] steps={getattr(result, 'steps', '?')} "
            f"train_loss={getattr(result, 'final_loss', '?')} "
            f"val_loss={getattr(result, 'val_loss', '?')} "
            f"runtime_s={getattr(result, 'runtime_s', '?')} "
            f"checkpoint={getattr(result, 'checkpoint_dir', None)}"
        )
