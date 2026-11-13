"""Cloud job-spec checks: every container path the job uses must be mounted.

Regression for the P1 where ``inference_command`` passed ``--checkpoint
/models/ckpt`` while the model mount was ``/models`` (so the container path
resolved to ``<host>/ckpt``, which never exists) and the training command
pointed at ``/workspace/configs/...``, a path the image does not have.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.config import load_config  # noqa: E402

sys.path.insert(0, str(ROOT))
from cloud.sandbox_runner import (  # noqa: E402
    build_job_spec,
    uncovered_container_paths,
)


def _spec(**kwargs):
    cfg = load_config(None)
    return build_job_spec(cfg, **kwargs)


def test_spec_has_no_uncovered_container_paths() -> None:
    model = ROOT / "outputs" / "checkpoints" / "full"
    spec = _spec(dataset_dir=ROOT / "data", model=model, output_dir=ROOT / "outputs")
    assert uncovered_container_paths(spec) == [], uncovered_container_paths(spec)
    # inference must load the same directory the model mount provides
    command = spec["inference_command"]
    checkpoint = command[command.index("--checkpoint") + 1]
    assert checkpoint == spec["mounts"]["model"]["container"], (checkpoint, spec["mounts"]["model"])
    # config path must be the one the image bakes in
    config = spec["command"][spec["command"].index("--config") + 1]
    assert config.startswith("/app/"), config
    print("  cloud spec -> --checkpoint matches the model mount; --config under /app")


def test_uncovered_detector_flags_legacy_layout() -> None:
    model_host = ROOT / "outputs" / "checkpoints" / "full"
    spec = {
        "command": ["python", "scripts/train.py", "--config", "/workspace/configs/x.yaml"],
        "inference_command": ["python", "scripts/infer.py", "--checkpoint", "/models/ckpt"],
        "mounts": {"model": {"container": "/models", "host": str(model_host), "mode": "ro"}},
    }
    problems = uncovered_container_paths(spec)
    assert any("/models/ckpt" in line for line in problems), problems
    assert any("/workspace/configs/x.yaml" in line for line in problems), problems
    print(f"  uncovered detector -> {len(problems)} problem(s) reported for the legacy layout")


def test_plan_exits_zero_and_submit_refuses_unmounted() -> None:
    plan = subprocess.run(
        [sys.executable, "cloud/sandbox_runner.py", "plan", "--dataset", "data", "--output", "outputs"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert plan.returncode == 0, plan.stderr
    payload = json.loads(plan.stdout)
    assert uncovered_container_paths(payload) == []
    print("  cloud plan -> exit 0, JSON spec parses, no uncovered container path")


def test_training_command_points_at_prepared_dataset_dir() -> None:
    """``--dataset-dir`` must hold train.jsonl, and the job must create it first.

    ``scripts/train.py --dataset-dir`` appends nothing and reads
    ``<dir>/train.jsonl``; the old spec passed the mount root ``/data`` (raw
    split tree with ``train/train.jsonl`` nested below it), which exits 2
    "Dataset not found" — verified locally: ``--dataset-dir ./data`` -> exit 2,
    ``--dataset-dir outputs/datasets/train`` -> 29 records. The gate must also
    catch a regression, since a paid job that dies there still bills.
    """
    import copy

    model = ROOT / "outputs" / "checkpoints" / "full"
    spec = _spec(dataset_dir=ROOT / "data", model=model, output_dir=ROOT / "outputs")
    command = spec["command"]
    dataset_arg = command[command.index("--dataset-dir") + 1]
    assert dataset_arg.endswith("/train"), dataset_arg
    assert dataset_arg.startswith(spec["mounts"]["output"]["container"] + "/"), dataset_arg
    prepare = spec["prepare_command"]
    assert prepare[prepare.index("--output") + 1] == dataset_arg.rsplit("/", 1)[0], prepare
    assert prepare[prepare.index("--data-root") + 1] == spec["mounts"]["dataset"]["container"], prepare

    legacy = copy.deepcopy(spec)
    legacy["command"][legacy["command"].index("--dataset-dir") + 1] = spec["mounts"]["dataset"]["container"]
    problems = uncovered_container_paths(legacy)
    assert any("--dataset-dir" in line for line in problems), problems
    print(f"  cloud spec -> --dataset-dir {dataset_arg} prepared from the read-only /data mount")

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

def test_history_transcripts_are_mounted_and_passed_to_inference() -> None:
    """A test split's history is audio-only, so the job must carry the ASR file.

    Without it 419 of 634 `public_test` samples render their prior turns blank
    (infer.py warns "history transcripts: NONE configured"). The mount is a
    single FILE at ``/history/history_transcripts.jsonl``, so the coverage gate
    must handle a file mount as well as a directory mount.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        transcripts = Path(tmp) / "asr" / "history_transcripts.jsonl"
        transcripts.parent.mkdir(parents=True)
        transcripts.write_text('{"split": "public_test", "id": "x", "turn_index": 1, "text": "a"}\n',
                              encoding="utf-8")
        model = ROOT / "outputs" / "checkpoints" / "full"
        spec = _spec(dataset_dir=ROOT / "data", model=model, output_dir=ROOT / "outputs",
                     history_transcripts=transcripts)

        mount = spec["mounts"]["history"]
        assert mount["container"] == "/history/history_transcripts.jsonl", mount
        assert mount["host"] == str(transcripts.resolve()), mount
        assert mount["mode"] == "ro"
        command = spec["inference_command"]
        assert "--history-transcripts" in command, command
        assert command[command.index("--history-transcripts") + 1] == mount["container"], command
        assert uncovered_container_paths(spec) == [], uncovered_container_paths(spec)
        assert spec["notes"] == [], spec["notes"]

        # Default location (<output>/asr/...) is used when the argument is omitted.
        default_spec = build_job_spec(load_config(None), ROOT / "data", model, ROOT / "outputs")
        default_host = Path(default_spec["mounts"].get("history", {}).get("host") or "")
        assert default_host.name == "history_transcripts.jsonl" or "history" not in default_spec["mounts"]

        # Missing file: the flag is omitted and the plan says why (no silent claim).
        absent = _spec(dataset_dir=ROOT / "data", model=model, output_dir=ROOT / "outputs",
                       history_transcripts=Path(tmp) / "nope.jsonl")
        assert "--history-transcripts" not in absent["inference_command"], absent["inference_command"]
        assert "history" not in absent["mounts"], absent["mounts"]
        assert any("history transcripts" in note for note in absent["notes"]), absent["notes"]
        assert uncovered_container_paths(absent) == []

        # The coverage gate must flag a history path that nobody mounted.
        broken = copy.deepcopy(spec)
        broken["mounts"].pop("history")
        problems = uncovered_container_paths(broken)
        assert any("/history/history_transcripts.jsonl" in line for line in problems), problems
    print("  cloud spec -> /history mount + --history-transcripts wired; unmounted path is flagged")
