"""Regression: the cloud probe must not claim a sandbox launcher that is absent.

``_has_sandbox_sdk`` used to accept ``wandb.run`` (the *current run object*,
``None`` outside a run) and ``wandb.sweep`` (a hyperparameter scheduler that runs
on compute you already own). On wandb 0.31.0 that made preflight report
``sandbox_sdk_present: true`` and let ``submit --no-dry-run`` reach a bogus
``wandb.sandbox.run(...)`` call, so the "unsupported -> clear error" promise was
dead code.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cloud.sandbox_runner import (  # noqa: E402
    _has_sandbox_sdk,
    _sandbox_launcher,
    preflight,
)


class _FakeWandb:
    """Mimics wandb 0.31.0 outside a run: run is None, sweep/runs exist."""

    __version__ = "0.31.0"
    run = None

    @staticmethod
    def sweep(sweep, entity=None, project=None, prior_runs=None):  # noqa: ANN001, ANN205
        return "sweep-id"


def test_run_and_sweep_are_not_launchers() -> None:
    assert not _has_sandbox_sdk(_FakeWandb()), "wandb.run/wandb.sweep must not count as a sandbox SDK"
    assert _sandbox_launcher(None) is None
    print("  probe -> wandb.run/wandb.sweep do not pass the launcher gate")


def test_installed_wandb_without_cwsandbox_is_unsupported() -> None:
    """The real installed wandb reports False unless the hardware client exists."""
    try:
        import wandb  # noqa: PLC0415
    except Exception:
        print("  probe -> wandb not importable, nothing to check")
        return
    import importlib.util

    expected = importlib.util.find_spec("cwsandbox") is not None
    assert _has_sandbox_sdk(wandb) is expected, (wandb.__version__, expected)
    report = preflight(None)
    assert report["sandbox_sdk_present"] is expected, report["sandbox_sdk_present"]
    if not expected:
        assert any("cannot launch sandbox hardware" in w for w in report["warnings"]), report["warnings"]
    print(f"  probe -> wandb {wandb.__version__}: sandbox_sdk_present={expected} (cwsandbox installed={expected})")


def test_launcher_gate_accepts_a_real_sandbox_run_callable() -> None:
    """A stub ``cwsandbox`` module with ``Sandbox.run`` must satisfy the gate."""
    import importlib.util

    if importlib.util.find_spec("cwsandbox") is not None:
        print("  probe -> real cwsandbox installed; stub path not exercised")
        return

    class _Sandbox:
        @classmethod
        def run(cls, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

    fake_cw = types.ModuleType("cwsandbox")
    fake_cw.__spec__ = importlib.machinery.ModuleSpec("cwsandbox", loader=None)
    fake_cw.Sandbox = _Sandbox
    fake_sandbox = types.ModuleType("wandb.sandbox")
    fake_sandbox.Sandbox = _Sandbox
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__version__ = "0.31.0"
    fake_wandb.run = None
    fake_wandb.sandbox = fake_sandbox

    saved_cw, saved_sandbox, saved_wandb = (
        sys.modules.get("cwsandbox"),
        sys.modules.get("wandb.sandbox"),
        sys.modules.get("wandb"),
    )
    sys.modules.update({"cwsandbox": fake_cw, "wandb.sandbox": fake_sandbox, "wandb": fake_wandb})
    try:
        assert _has_sandbox_sdk(fake_wandb), "a callable Sandbox.run must pass the gate"
        assert _sandbox_launcher(fake_wandb) is not None
    finally:
        for name, value in (("cwsandbox", saved_cw), ("wandb.sandbox", saved_sandbox), ("wandb", saved_wandb)):
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    print("  probe -> stub cwsandbox.Sandbox.run satisfies the gate")


def main() -> int:
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - test harness
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
