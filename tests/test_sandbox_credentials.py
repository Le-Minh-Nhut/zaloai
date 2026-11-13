"""Regression: the submit gate must accept a ``~/.netrc`` credential.

``submit`` used to require ``WANDB_API_KEY`` in the environment. On this machine
that variable is unset and authentication nonetheless succeeds from
``~/.netrc`` (``machine api.wandb.ai``), which wandb maps to the Forge instance
via ``FORGE_HOSTS`` and which ``cwsandbox`` 1.28.0 reads directly. The env-only
gate therefore blocked the only working credential path.

Also covers the actionable refusal that replaced the dead-end
``no container_image`` message: it must still refuse, and name the three
undeliverable inputs plus the GPU entitlement gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import os  # noqa: E402

from lifemate.errors import LifeMateError  # noqa: E402
from cloud.sandbox_runner import resolve_credentials, submit, wandb_host  # noqa: E402

NETRC = "machine api.wandb.ai\n  login probe-user\n  password probe-secret-value\n"


def _isolate(tmp_path: Path, *, base_url: str | None = None, api_key: str | None = None):
    """Point the resolver at a temp HOME and a controlled environment."""
    saved = {k: os.environ.get(k) for k in ("HOME", "WANDB_API_KEY", "WANDB_BASE_URL")}
    saved_netrc = None
    import cloud.sandbox_runner as mod

    saved_netrc = mod.NETRC_PATH
    os.environ["HOME"] = str(tmp_path)
    mod.NETRC_PATH = tmp_path / ".netrc"
    for key, value in (("WANDB_API_KEY", api_key), ("WANDB_BASE_URL", base_url)):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def restore() -> None:
        mod.NETRC_PATH = saved_netrc
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return restore


def test_host_derivation() -> None:
    assert wandb_host("https://api.wandb.ai") == "api.wandb.ai"
    assert wandb_host("api.wandb.ai") == "api.wandb.ai"
    assert wandb_host("https://api.wandb.ai/base") == "api.wandb.ai"
    print("  gate -> host derived from base URL, path ignored")


def test_netrc_entry_authorises_without_env(tmp_path: Path) -> None:
    (tmp_path / ".netrc").write_text(NETRC)
    restore = _isolate(tmp_path)
    try:
        creds = resolve_credentials()
        assert creds["source"] == "netrc", creds
        assert creds["netrc_machine"] == "api.wandb.ai", creds
        assert "probe-secret-value" not in repr(creds), "credential value leaked into the report"
    finally:
        restore()
    print("  gate -> netrc entry satisfies the credential gate, value never reported")


def test_env_var_still_authorises(tmp_path: Path) -> None:
    restore = _isolate(tmp_path, api_key="env-value")
    try:
        creds = resolve_credentials()
        assert creds["source"] == "env:WANDB_API_KEY", creds
        assert "env-value" not in repr(creds)
    finally:
        restore()
    print("  gate -> WANDB_API_KEY path still accepted")


def test_missing_both_is_absent(tmp_path: Path) -> None:
    restore = _isolate(tmp_path, base_url="https://api.example.invalid")
    try:
        assert resolve_credentials()["source"] == "absent"
    finally:
        restore()
    print("  gate -> no env var, no matching netrc machine -> absent")


def test_netrc_for_a_different_host_does_not_authorise(tmp_path: Path) -> None:
    (tmp_path / ".netrc").write_text(NETRC)
    restore = _isolate(tmp_path, base_url="https://api.example.invalid")
    try:
        assert resolve_credentials()["source"] == "absent", "a netrc entry for another host must not count"
    finally:
        restore()
    print("  gate -> netrc entry for a different host is rejected")


def test_submit_refusal_is_actionable(tmp_path: Path) -> None:
    (tmp_path / ".netrc").write_text(NETRC)
    restore = _isolate(tmp_path)
    import cloud.sandbox_runner as mod

    saved = (mod._wandb, mod._sandbox_launcher)
    mod._wandb = lambda: type("W", (), {"__version__": "0.31.0"})()
    mod._sandbox_launcher = lambda _w: (lambda **kwargs: None)
    try:
        try:
            submit({"command": ["python", "-m", "lifemate.train"]}, dry_run=False, confirmed=True, cfg=None)
        except LifeMateError as exc:
            message = str(exc)
        else:  # pragma: no cover - the refusal is the contract
            raise AssertionError("submit must still refuse: no launch path exists")
    finally:
        mod._wandb, mod._sandbox_launcher = saved
        restore()
    assert "container_image" in message and "mounted_files" in message, message
    for needle in ("1.1 GB", "8.68 GiB", "not entitled to create GPU sandboxes", "mounted_files"):
        assert needle in message, f"refusal must name {needle!r}: {message}"
    assert "probe-secret-value" not in message
    print("  gate -> refusal names code/dataset/weights/entitlement, not just 'no image'")


def test_submit_still_refuses_without_credentials(tmp_path: Path) -> None:
    restore = _isolate(tmp_path, base_url="https://api.example.invalid")
    try:
        try:
            submit({"command": ["python", "-m", "x"]}, dry_run=False, confirmed=True, cfg=None)
        except LifeMateError as exc:
            assert "no wandb credential" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError("submit must refuse without credentials")
    finally:
        restore()
    print("  gate -> refuse-without-credentials behaviour preserved")


def test_dry_run_never_needs_credentials(tmp_path: Path) -> None:
    restore = _isolate(tmp_path, base_url="https://api.example.invalid")
    try:
        result = submit({"command": ["python", "-m", "x"]}, dry_run=True)
        assert result["dry_run"] is True and result["submitted"] is False
    finally:
        restore()
    print("  gate -> dry run stays free and credential-free")


def main() -> int:
    import tempfile

    failures = 0
    for name, func in sorted(globals().items()):
        if not (name.startswith("test_") and callable(func)):
            continue
        try:
            if func.__code__.co_argcount and "tmp_path" in func.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    func(Path(tmp))
            else:
                func()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - test harness
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
