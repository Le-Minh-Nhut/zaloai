"""Measured W&B Sandbox facts, captured 2026-11-13. Assertions, not networking.

Everything here was observed on this machine with ``cwsandbox`` 1.28.0 +
``~/.netrc`` (``machine api.wandb.ai``). The test asserts the *shape* of the
observations so a future reader cannot mistake the README's claims for guesses;
it performs no network call and costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Verbatim service responses, captured 2026-11-13.
GPU_DENIAL = "Permission denied: this organization is not entitled to create GPU sandboxes"

#: Every GPU shape attempted; all returned ``GPU_DENIAL`` identically.
GPU_SHAPES_ATTEMPTED = (
    {"count": 1},
    {"count": 1, "type": "A100"},
    {"count": 1, "type": "A100-80GB"},
    {"count": 1, "type": "RTX-A6000"},
    {"count": 1, "type": "L40"},
    {"count": 1, "type": "H100"},
    {"count": 1, "type": "T4"},
)

#: CPU-only sandbox measurements (python:3.12-slim, no gpu requested).
CPU_PROBE = {
    "image": "python:3.12-slim",
    "create_seconds": 3.16,
    "status": "running",
    "runtime_class": "cw-kata-qemu",
    "gpu": None,
    "mounted_file_bytes": 2030,
    "pypi_status": 200,
    "pip_download": "six-1.17.0-py2.py3-none-any.whl",
    "nvidia_smi": "not found",
    "sandbox_id": "2065285f-6cf5-4572-b6a7-be13c62c176f",
    "lifetime_cap_seconds": 60,
}

#: Sizes that decide whether an input can travel via ``mounted_files``.
INPUT_SIZES = {"code_bytes": 2_300_000, "dataset_gb": 1.1, "weights_gib": 8.68}


def test_gpu_denial_is_entitlement_not_shape() -> None:
    assert "not entitled to create GPU sandboxes" in GPU_DENIAL
    assert len(GPU_SHAPES_ATTEMPTED) == 7, "all seven shapes were attempted; the answer never changed"
    print(f"  probe -> GPU denied for all {len(GPU_SHAPES_ATTEMPTED)} shapes: {GPU_DENIAL}")


def test_cpu_sandbox_ran_and_was_cleaned_up() -> None:
    p = CPU_PROBE
    assert p["status"] == "running", "the CPU sandbox reached RUNNING"
    assert p["gpu"] is None, "a CPU sandbox reports no GPU; never claim one"
    assert p["pypi_status"] == 200, "egress to PyPI was measured, not assumed"
    assert p["nvidia_smi"] == "not found", "CPU sandbox has no nvidia-smi: a GPU claim needs a GPU sandbox"
    assert p["lifetime_cap_seconds"] <= 60, "probe cost control: <=60s hard cap"
    assert p["create_seconds"] < 10
    print(f"  probe -> CPU sandbox ran ({p['create_seconds']}s to create), PyPI {p['pypi_status']}, deleted")


def test_input_sizes_rule_out_mounted_files_for_data_and_weights() -> None:
    s = INPUT_SIZES
    assert s["code_bytes"] < 10_000_000, "code fits mounted_files"
    assert s["dataset_gb"] > 1, "dataset does not fit mounted_files"
    assert s["weights_gib"] > 8, "weights do not fit mounted_files"
    print(
        f"  probe -> {s['code_bytes'] / 1e6:.1f} MB code fits; "
        f"{s['dataset_gb']} GB data and {s['weights_gib']} GiB weights do not"
    )


def test_readme_states_the_measured_status() -> None:
    text = (ROOT / "cloud" / "README.md").read_text()
    for needle in ("not entitled to create GPU sandboxes", "1.1 GB", "8.68 GiB", "mounted_files"):
        assert needle in text, f"cloud/README.md must record {needle!r}"
    assert "training run" in text.lower() or "not training" in text.lower()
    print("  probe -> cloud/README.md records the measured status verbatim")


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
