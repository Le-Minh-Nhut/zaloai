"""W&B tracker: offline run creation + secret/PII redaction. No network, no torch.

The tracker is the only code path that can put user content on a third-party
service, so two things are pinned here:

1. ``sanitize`` (the policy every payload passes through) redacts secrets and
   drops PII keys;
2. a real ``WandbTracker`` in offline mode writes a run to disk and the bytes it
   wrote contain none of the secret/PII values that were logged.

Skipped cleanly when ``wandb`` is not importable (stdlib runner: prints SKIP,
pytest: real skip), which is the normal state of the system interpreter.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:
    class _PytestShim:
        @staticmethod
        def skip(reason: str = "") -> None:
            """Stdlib runner: skipping is a return, not an exception."""

    pytest = _PytestShim()  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SECRETS = ("sk-live-SHOULD-NOT-LEAK", "hunter2", "+84900000000", "Nguyen Van A", "Dat bao thuc luc 6 gio")


def _wandb_available() -> bool:
    """Delegate to the library's own check (a stray ``wandb/`` run dir is not the package)."""
    from lifemate.tracking import wandb_available  # noqa: PLC0415

    return wandb_available()


def test_environment_payload_carries_no_machine_identity() -> None:
    """``describe_environment`` leaks no host, user or machine model.

    Asserted by construction: the payload's key set is exactly an allowlist, so
    an identifier cannot be added without failing the set comparison. A machine
    model name, a hostname or a username would each be a *value* under a key not
    in that allowlist.
    """
    import getpass  # noqa: PLC0415
    import socket  # noqa: PLC0415

    from lifemate.tracking import describe_environment  # noqa: PLC0415

    env = describe_environment()
    allowed = {
        "git_commit",
        "git_dirty",
        "python",
        "platform",
        "torch",
        "cuda",
        "cuda_available",
        "transformers",
        "gpu",
    }
    assert set(env) == allowed, f"unexpected environment keys: {sorted(set(env) - allowed)}"
    # The two identifiers the leak was reported for are not even collected.
    assert "hostname" not in env and "user" not in env
    flat = repr(env)
    assert socket.gethostname() not in flat, "hostname reached the environment payload"
    assert getpass.getuser() not in flat, "username reached the environment payload"
    # platform is the OS family only ("linux"), not platform.platform()'s
    # "Linux-6.x-...#N SMP ..." string, which carries kernel and host details.
    assert env["platform"] in {"linux", "darwin", "win32", "cygwin"}, env["platform"]
    # gpu is a capability class, not the card's marketing name.
    assert env["gpu"] == "unavailable" or env["gpu"].startswith("cuda-sm_"), env["gpu"]
    print("  env payload -> allowlist only; no hostname, username or device name")


def test_sanitize_redacts_secrets_and_drops_pii() -> None:
    """Policy-level check: runs anywhere, no wandb import needed."""
    from lifemate.tracking import REDACTED, sanitize  # noqa: PLC0415

    clean = sanitize(
        {
            "api_key": "sk-live-SHOULD-NOT-LEAK",
            "password": "hunter2",
            "user_profile": {"phone": "+84900000000", "name": "Nguyen Van A"},
            "audio_text": "Dat bao thuc luc 6 gio",
            "learning_rate": 5e-6,
            "nested": {"access_token": "tok-123", "step": 3},
        }
    )
    assert clean["api_key"] == REDACTED
    assert clean["password"] == REDACTED
    assert clean["user_profile"] == "[DROPPED_PII]"
    assert clean["audio_text"] == "[DROPPED_PII]"
    assert clean["nested"]["access_token"] == REDACTED
    # metrics must survive untouched
    assert clean["learning_rate"] == 5e-6 and clean["nested"]["step"] == 3
    flat = json.dumps(clean, ensure_ascii=False)
    for secret in SECRETS:
        assert secret not in flat, f"{secret!r} survived sanitize"


def test_wandb_tracker_offline_run_redacts_on_disk() -> None:
    """A real offline run: files are written, and no logged secret reaches them."""
    saved = {key: os.environ.get(key) for key in ("WANDB_MODE", "WANDB_DIR", "WANDB_SILENT")}
    tmp = tempfile.mkdtemp(prefix="lifemate-wandb-test-")
    # Must be set before the first ``import wandb``: settings are cached at import.
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = tmp
    os.environ["WANDB_SILENT"] = "true"
    if not _wandb_available():
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        print("  SKIP wandb tracker test: wandb not installed")
        pytest.skip("wandb not installed")
        return
    from lifemate.config import load_config  # noqa: PLC0415
    from lifemate.tracking import WandbTracker  # noqa: PLC0415

    try:
        cfg = load_config(ROOT / "configs" / "training_lora_wandb.yaml")
        assert cfg.training.report_to == ["wandb"], cfg.training.report_to
        # ``dir`` is passed explicitly: WANDB_DIR is cached at wandb's first
        # import, which another test module may already have triggered.
        tracker = WandbTracker(cfg, run_name="pytest-offline", config=cfg.to_dict(), dir=tmp)
        assert tracker.enabled, tracker.disabled_reason
        assert tracker.mode == "offline"
        for step in (1, 2):
            tracker.log(
                {
                    "step": step,
                    "train_loss": 1.0 / step,
                    "api_key": "sk-live-SHOULD-NOT-LEAK",
                    "password": "hunter2",
                    "user_profile": {"phone": "+84900000000", "name": "Nguyen Van A"},
                    "audio_text": "Dat bao thuc luc 6 gio",
                }
            )
        tracker.finish()

        runs = list(Path(tmp).glob("wandb/offline-run-*"))
        assert runs, f"no offline-run directory under {tmp}: {list(Path(tmp).iterdir())}"
        blob = b"".join(p.read_bytes() for p in runs[0].rglob("*") if p.is_file())
        assert blob, "the offline run wrote no bytes"
        leaked = [secret for secret in SECRETS if secret.encode() in blob]
        assert not leaked, f"secret/PII reached the offline run files: {leaked}"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("  wandb -> offline run created, 2 steps logged, no secret/PII on disk")
