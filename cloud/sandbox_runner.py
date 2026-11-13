#!/usr/bin/env python3
"""Describe (and optionally submit) a W&B Sandbox GPU job for LifeMate.

Nothing billable happens by default: ``submit`` runs with ``dry_run=True`` and
prints the exact command. A real submit requires ``--no-dry-run``/``--submit``
AND a wandb credential (``WANDB_API_KEY`` in the environment, or a ``~/.netrc``
entry for the resolved wandb host) AND ``--yes``. Even then ``submit`` refuses:
no launch path exists until code, data and the 8.68 GiB checkpoint have a
delivery route and the org is entitled to GPU sandboxes (see cloud/README.md).

Gated SDK usage: ``wandb.sandbox`` is only touched through ``hasattr`` probes, so
an unsupported wandb version produces a clear error instead of a guess.

Usage::

    python cloud/sandbox_runner.py preflight
    python cloud/sandbox_runner.py plan --dataset ./data --output ./outputs
    python cloud/sandbox_runner.py submit --dataset ./data --output ./outputs      # dry run
    python cloud/sandbox_runner.py submit --dataset ./data --output ./outputs --no-dry-run --yes

Exit codes: 0 ok, 2 preflight found a blocker, 3 requested action unsupported.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import netrc
import os
import posixpath
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402

EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_UNSUPPORTED = 3

#: Environment variables the job needs. Names only: values are never printed.
CREDENTIAL_ENV_VARS = ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")

#: Host whose ``~/.netrc`` entry authorises ``wandb`` and ``cwsandbox`` when no
#: ``WANDB_API_KEY`` is exported. ``cwsandbox`` 1.28.0 reads exactly this via
#: wandb's ``FORGE_HOSTS`` mapping, so netrc is a first-class credential path.
DEFAULT_WANDB_BASE_URL = "https://api.wandb.ai"
NETRC_PATH = Path("~/.netrc").expanduser()

def wandb_host(base_url: str | None = None) -> str:
    """Hostname that must appear in ``~/.netrc`` to authorise the wandb API."""
    raw = base_url or os.environ.get("WANDB_BASE_URL") or DEFAULT_WANDB_BASE_URL
    return urlparse(raw if "//" in raw else f"//{raw}", scheme="https").hostname or "api.wandb.ai"

def resolve_credentials(base_url: str | None = None) -> dict[str, Any]:
    """Where the wandb credential comes from. Reports provenance, never a value."""
    host = wandb_host(base_url)
    report: dict[str, Any] = {
        "source": "absent",
        "base_url_host": host,
        "netrc_path": str(NETRC_PATH),
        "netrc_machine": None,
    }
    if os.environ.get("WANDB_API_KEY"):
        report["source"] = "env:WANDB_API_KEY"
        return report
    try:
        parsed = netrc.netrc(str(NETRC_PATH))
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return report
    entry = parsed.hosts.get(host)
    if entry is not None:
        report["netrc_machine"] = host
    else:
        entry = getattr(parsed, "default", None)
        if entry is not None:
            report["netrc_machine"] = "default"
    if entry and (entry[0] or entry[2]):
        report["source"] = "netrc"
    return report

#: The package that actually provisions the hardware. In wandb 0.31.0 the
#: ``wandb.sandbox`` module is a deprecated shim that re-exports this client and
#: raises ``ImportError`` when it is missing. ``wandb.run`` is the current run
#: object (``None`` outside a run) and ``wandb.sweep`` schedules hyperparameters
#: on a machine *you* provide: neither rents a GPU.
SANDBOX_BACKEND = "cwsandbox"


def _wandb() -> Any:
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def _sandbox_launcher(wandb_module: Any) -> Any:
    """Return ``wandb.sandbox.Sandbox.run`` when the real backend is installed.

    That classmethod is the only entry point in wandb 0.31.0 that asks the
    service for a container, and it accepts ``resources={"gpu": {"count": n}}``.
    Everything else (``wandb.run``, ``wandb.sweep``, ``wandb.sdk.launch.launch``)
    either reports a local run or schedules work onto compute the caller owns.
    """
    if wandb_module is None or importlib.util.find_spec(SANDBOX_BACKEND) is None:
        return None
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # deprecation notice
            from wandb.sandbox import Sandbox  # noqa: PLC0415
    except Exception:
        return None
    run = getattr(Sandbox, "run", None)
    return run if callable(run) else None


def _has_sandbox_sdk(wandb_module: Any) -> bool:
    """True only when a callable that provisions sandbox hardware exists."""
    return _sandbox_launcher(wandb_module) is not None


def preflight(cfg: Any = None) -> dict[str, Any]:
    """Local, free checks. No network call, no GPU claim beyond what is provable."""
    wandb_module = _wandb()
    report: dict[str, Any] = {
        "provider": getattr(getattr(cfg, "cloud", None), "provider", "wandb_sandbox"),
        "wandb_installed": wandb_module is not None,
        "wandb_version": getattr(wandb_module, "__version__", "unavailable") if wandb_module else "unavailable",
        "sandbox_sdk_present": _has_sandbox_sdk(wandb_module),
        "credentials": {},
        "gpu_type_availability": "unverified",
        "dataset_present": None,
        "blockers": [],
        "warnings": [],
    }
    for name in CREDENTIAL_ENV_VARS:
        report["credentials"][name] = "set" if os.environ.get(name) else "absent"
    report["credential_source"] = resolve_credentials()
    report["credentials"]["WANDB_CREDENTIAL"] = (
        report["credential_source"]["source"]
    )
    if not report["wandb_installed"]:
        report["warnings"].append("wandb is not installed: pip install wandb (sandbox submission needs it)")
    if not _has_sandbox_sdk(wandb_module):
        report["warnings"].append(
            "this wandb build cannot launch sandbox hardware: no importable "
            "wandb.sandbox.Sandbox.run with the 'cwsandbox' backend installed "
            "(wandb.run/wandb.sweep are not job launchers); cloud submission is unsupported here"
        )
    if report["credential_source"]["source"] == "absent":
        report["blockers"].append(
            f"no wandb credential: WANDB_API_KEY is unset and {report['credential_source']['netrc_path']} "
            f"has no entry for {report['credential_source']['base_url_host']}"
        )
    gpu_type = getattr(getattr(cfg, "cloud", None), "gpu_type", None)
    report["requested_gpu"] = gpu_type or "unspecified"
    report["gpu_type_availability"] = "unverified (no SDK query performed; never claimed as available)"
    report["free_actions_only"] = True
    data_root = getattr(getattr(cfg, "data", None), "root", None)
    if data_root:
        try:
            from lifemate.data.discovery import available_splits

            splits = available_splits(data_root)
            report["dataset_present"] = bool(splits)
            report["dataset_splits"] = splits
        except Exception:
            report["dataset_present"] = False
    return report


#: Paths baked into the image by ``docker/Dockerfile`` rather than mounted.
BAKED_PREFIXES = ("/app/",)

#: Where the job writes SFT records (``<outputs>/datasets`` on the host) and the
#: per-split directory ``scripts/train.py --dataset-dir`` must be pointed at.
PREPARED_DATASETS_CONTAINER = "/outputs/datasets"
PREPARED_TRAIN_CONTAINER = f"{PREPARED_DATASETS_CONTAINER}/train"

#: History-turn ASR transcripts, mounted as a single FILE. A test split carries
#: history audio only, so without this flag 419 of 634 public_test samples lose
#: their prior turns (measured; `scripts/infer.py` warns "history transcripts:
#: NONE configured"). Mounted rather than baked: the file is regenerated per
#: dataset release.
HISTORY_TRANSCRIPTS_CONTAINER = "/history/history_transcripts.jsonl"


def container_path_is_baked(path: str, mounts: Mapping[str, Mapping[str, Any]] | None = None) -> bool:
    """Whether a container path is served by the image rather than a mount.

    ``/app`` is where ``docker/Dockerfile`` copies ``src``, ``scripts`` and
    ``configs``; anything else must be covered by a mount, otherwise the job
    reads a path that does not exist in the container.
    """
    return str(path).startswith(BAKED_PREFIXES)


def uncovered_container_paths(spec: Mapping[str, Any]) -> list[str]:
    """Absolute paths used by the job's commands that no mount (or the image) provides.

    Cheap, static check that catches the whole class of "the command points at a
    container path nobody mounted" bugs — the dry run then fails loudly instead
    of the paid GPU job dying on a missing model directory.
    """
    mounts = spec.get("mounts") or {}
    prefixes: list[tuple[str, str]] = [
        (
            posixpath.normpath(str(entry.get("container"))).rstrip("/") or "/",
            str(entry.get("host") or "") if str(entry.get("mode", "")) == "ro" else "",
        )
        for entry in mounts.values()
        if isinstance(entry, Mapping) and entry.get("container")
    ]
    missing: list[str] = []
    for key in ("prepare_command", "command", "inference_command"):
        for token in spec.get(key) or []:
            token = str(token)
            if not token.startswith("/"):
                continue
            token = posixpath.normpath(token)
            if container_path_is_baked(token):
                continue
            owner = next(
                (
                    (prefix, host)
                    for prefix, host in prefixes
                    if token == prefix or token.startswith(prefix + "/")
                ),
                None,
            )
            if owner is None:
                missing.append(f"{key}: {token} (not mounted; mounted: {[p for p, _ in prefixes]})")
                continue
            # The mount prefix is covered, but a read-only mount's host side may be
            # a shallow directory: /models/ckpt works only if <host>/ckpt exists.
            # Read-write mounts are skipped: their files are written by the job.
            prefix, host = owner
            if host:
                suffix = token[len(prefix) :].lstrip("/")
                if suffix and not Path(host, suffix).exists():
                    missing.append(f"{key}: {token} (host {Path(host, suffix)} does not exist)")
                elif token == prefix and not Path(host).exists():
                    # Single-file mounts (e.g. /history/history_transcripts.jsonl) map
                    # the container path straight onto the host file.
                    missing.append(f"{key}: {token} (host {host} does not exist)")
                elif token == prefix and Path(host).is_file() and suffix:
                    missing.append(f"{key}: {token} (mount is a file, cannot serve {suffix})")
        # ``scripts/train.py --dataset-dir X`` reads ``X/train.jsonl``: passing the
        # mount root instead of the prepared split dir is a silent exit-2 at
        # submit time (verified: ``--dataset-dir ./data`` -> "Dataset not found";
        # ``--dataset-dir outputs/datasets/train`` -> 29 records).
        tokens = [str(token) for token in (spec.get(key) or [])]
        if "--dataset-dir" in tokens:
            dataset_arg = tokens[tokens.index("--dataset-dir") + 1]
            if not dataset_arg.startswith(PREPARED_DATASETS_CONTAINER + "/"):
                missing.append(
                    f"{key}: --dataset-dir {dataset_arg} (must be "
                    f"{PREPARED_TRAIN_CONTAINER}, holding train.jsonl/val.jsonl)"
                )
    return missing


def build_job_spec(
    cfg: Any,
    dataset_dir: str | Path,
    model: str | Path,
    output_dir: str | Path,
    history_transcripts: str | Path | None = None,
) -> dict[str, Any]:
    """Describe the job: command, mounts, env and artifact paths.

    ``history_transcripts`` defaults to ``<output_dir>/asr/history_transcripts.jsonl``
    (where ``scripts/transcribe_split.py`` writes it). When that file exists it is
    mounted read-only and passed to ``scripts/infer.py --history-transcripts``;
    when it does not, the command omits the flag and a note says why.
    """
    dataset = Path(dataset_dir).resolve()
    model_path = Path(model).resolve()
    output = Path(output_dir).resolve()
    history_path = Path(
        history_transcripts if history_transcripts is not None else output / "asr" / "history_transcripts.jsonl"
    ).resolve()
    history_mount = history_path.is_file()
    cloud = getattr(cfg, "cloud", None)
    project = getattr(cloud, "project", None) or getattr(getattr(cfg, "wandb", None), "project", "lifemate")
    entity = getattr(cloud, "entity", None) or getattr(getattr(cfg, "wandb", None), "entity", None)
    # ``infer.py --checkpoint`` is handed straight to ``from_pretrained``
    # (loader.py: ``base_source = str(checkpoint)``), so it must be a directory
    # holding config.json + weights — i.e. a ``step-<n>`` dir, not the run root
    # that ``scripts/train.py`` writes under ``<output-dir>/<run-name>/``.
    checkpoint_container = "/models/ckpt"
    model_host = str(model_path)
    model_note = None
    if model_path.exists():
        try:
            from lifemate.models.checkpoint import resolve_checkpoint  # noqa: PLC0415

            model_host = resolve_checkpoint(model_path).path
        except LifeMateError as exc:
            model_note = f"model path {model_path} is not a loadable checkpoint: {exc}"
    else:
        model_note = f"model path {model_path} does not exist on this host (dry run only)"
    history_note = None
    if not history_mount:
        history_note = (
            f"history transcripts {history_path} not found: the job runs without "
            "--history-transcripts, so public_test prior turns render blank (419/634 "
            "samples affected). Generate it with scripts/transcribe_split.py."
        )
    mounts: dict[str, Any] = {
        "dataset": {"host": str(dataset), "container": "/data", "mode": "ro"},
        "model": {"host": model_host, "container": checkpoint_container, "mode": "ro"},
        "output": {"host": str(output), "container": "/outputs", "mode": "rw"},
    }
    if history_mount:
        mounts["history"] = {
            "host": str(history_path),
            "container": HISTORY_TRANSCRIPTS_CONTAINER,
            "mode": "ro",
        }
    inference_command = [
        "python",
        "scripts/infer.py",
        "--split",
        "public_test",
        "--data-root",
        "/data",
        "--model-mode",
        "full",
        "--checkpoint",
        checkpoint_container,
        "--output",
        "/outputs/submission.jsonl",
    ]
    if history_mount:
        inference_command += ["--history-transcripts", HISTORY_TRANSCRIPTS_CONTAINER]
    return {
        "provider": getattr(cloud, "provider", "wandb_sandbox"),
        "project": project,
        "entity": entity,
        "gpu_type": getattr(cloud, "gpu_type", None),
        # ``scripts/train.py --dataset-dir`` wants the directory *holding*
        # train.jsonl/val.jsonl (``outputs/datasets/<split>``), while
        # ``scripts/infer.py --data-root`` wants the raw split tree. The job
        # therefore prepares the SFT records from the read-only ``/data`` mount
        # into the writable ``/outputs`` mount first; pointing ``--dataset-dir``
        # at ``/data`` (as the plan used to) exits 2 "Dataset not found".
        "prepare_command": [
            "python",
            "scripts/prepare_dataset.py",
            "--data-root",
            "/data",
            "--split",
            "train",
            "--output",
            PREPARED_DATASETS_CONTAINER,
        ],
        "command": [
            "python",
            "scripts/train.py",
            "--config",
            "/app/configs/training_full.yaml",
            "--dataset-dir",
            PREPARED_TRAIN_CONTAINER,
            "--output-dir",
            "/outputs/checkpoints",
        ],
        "inference_command": inference_command,
        "mounts": mounts,
        "notes": [note for note in (model_note, history_note) if note],
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_PROJECT": project,
            "WANDB_MODE": "online",
            "PYTHONUNBUFFERED": "1",
        },
        "env_names_required": ["WANDB_API_KEY"],
        "artifacts": {
            "checkpoint": f"{project}/{'/'.join(filter(None, [entity, getattr(cloud, 'artifact_name', 'lifemate-full')]))}:latest",
            "outputs": [str(output / "checkpoints"), str(output / "evaluation")],
        },
        "estimated_cost_note": (
            "A single-GPU run bills per minute; check your organisation's quota page before submitting. "
            "This tool performs no billable action in dry-run mode (the default)."
        ),
    }


def submit(
    job_spec: Mapping[str, Any],
    dry_run: bool = True,
    *,
    confirmed: bool = False,
    cfg: Any = None,
) -> dict[str, Any]:
    """Print (dry run) or launch the job. Real launches need key + explicit yes."""
    command = list(job_spec.get("command") or [])
    printable = " ".join(command)
    if dry_run:
        return {
            "submitted": False,
            "dry_run": True,
            "command": printable,
            "mounts": job_spec.get("mounts"),
            "env": job_spec.get("env"),
            "message": "DRY RUN: nothing was launched and nothing was billed.",
        }
    credentials = resolve_credentials()
    if credentials["source"] == "absent":
        raise LifeMateError(
            "refusing to submit: no wandb credential. Set WANDB_API_KEY, or add a "
            f"~/.netrc entry for {credentials['base_url_host']} "
            "(machine <host> login <user> password <key>). Checked "
            f"{credentials['netrc_path']}: source=absent."
        )
    if not confirmed:
        raise LifeMateError("refusing to submit: pass --yes to confirm a billable GPU job")

    wandb_module = _wandb()
    if wandb_module is None:
        raise LifeMateError("refusing to submit: wandb is not installed (pip install wandb)")
    launcher = _sandbox_launcher(wandb_module)
    if launcher is None:
        raise LifeMateError(
            "SDK version unsupported: this wandb build exposes no wandb.sandbox.Sandbox.run "
            f"(the 'cwsandbox' backend is not installed). wandb version: "
            f"{getattr(wandb_module, '__version__', 'unknown')}. "
            f"wandb.run/wandb.sweep are not job launchers. Run the command manually instead: {printable}"
        )
    # ponytail: the sandbox SDK binds container_image + mounted_files, not host
    # directories. Launching is one call away once code+data+weights have a
    # delivery route; see cloud/README.md for what is measured vs missing.
    raise LifeMateError(
        "refusing to submit: this spec mounts three host directories "
        "(/data ro, /models/ckpt ro, /outputs rw) but the sandbox API only binds "
        "a container_image plus mounted_files (file bytes, read-only at runtime). "
        "Nothing here can launch it. What a real launch needs:\n"
        "  1. code: container_image with the repo baked in, OR mounted_files "
        "(2.3 MB of src/scripts/configs/docker is within reach; the tree is NOT "
        "mounted by this tool yet)\n"
        "  2. dataset: 1.1 GB under data/ - too big for mounted_files, needs "
        "object storage with a signed URL or a baked image\n"
        "  3. weights: 8.68 GiB Qwen checkpoint - same, needs a signed URL or a "
        "baked image, plus HF egress in the sandbox\n"
        "  4. GPU entitlement: the W&B org this machine authenticates as is not "
        "entitled to create GPU sandboxes "
        "('Permission denied: this organization is not entitled to create GPU "
        "sandboxes') - no image fixes that; ask the org admin.\n"
        "Verified working alternative (probe, not training): a CPU sandbox on "
        "'python:3.12-slim' runs, accepts mounted_files, and reaches PyPI "
        "(HTTP 200). Resolve the credential (done) and the GPU entitlement "
        "before any billable submit."
    )


def resume(ref: str) -> dict[str, Any]:
    """Report how to resume a run (listing is local-only; no launch)."""
    return {
        "resume_ref": ref,
        "action": "not executed",
        "message": (
            "Resuming needs a live SDK handle for the original job. Re-submit with the same "
            "--output directory and set training.resume_from_checkpoint; nothing was launched here."
        ),
    }


def download_checkpoint(artifact: str, dest: str | Path) -> dict[str, Any]:
    """Delegate to the tracking artifact manager (real download if wandb exists)."""
    from lifemate.tracking.artifact_manager import download_artifact

    return download_artifact(artifact, dest)


def _build_cfg(args: argparse.Namespace) -> Any:
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "gpu", None):
        cfg.cloud.gpu_type = args.gpu
    return cfg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface (subcommand first)."""
    parser = argparse.ArgumentParser(description="Inspect or submit a W&B Sandbox job for LifeMate.")
    parser.add_argument("action", nargs="?", default="preflight", choices=("preflight", "plan", "submit", "resume", "download"))
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--dataset", default="./data", help="dataset directory to mount at /data")
    parser.add_argument(
        "--model",
        default="./outputs/checkpoints/full",
        help="checkpoint dir to mount at /models/ckpt; a run root is resolved to its newest step-<n>",
    )
    parser.add_argument("--output", default="./outputs", help="output directory mounted at /outputs")
    parser.add_argument(
        "--history-transcripts",
        default=None,
        metavar="FILE",
        help="history ASR transcripts JSONL to mount read-only at "
        "/history/history_transcripts.jsonl (default: <output>/asr/history_transcripts.jsonl)",
    )
    parser.add_argument("--gpu", default=None, help="GPU type to request (availability stays unverified)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="default: print only")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="allow a real submit")
    parser.add_argument("--yes", action="store_true", help="confirm a billable GPU job")
    parser.add_argument("--resume", default=None, help="run/artifact reference to resume")
    parser.add_argument("--download", default=None, metavar="ARTIFACT", help="artifact reference to download")
    parser.add_argument("--dest", default="outputs/checkpoints", help="download destination")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = _build_cfg(args)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    if args.action == "preflight":
        report = preflight(cfg)
        print(json.dumps(report, indent=2) if args.json else _render_preflight(report))
        return EXIT_BLOCKED if report["blockers"] else EXIT_OK

    if args.action == "plan":
        spec = build_job_spec(
            cfg, args.dataset, args.model, args.output, history_transcripts=args.history_transcripts
        )
        print(json.dumps(spec, indent=2))
        return _report_uncovered(spec)

    if args.action == "resume":
        if not args.resume:
            print("ERROR: --resume REF is required for the resume action", file=sys.stderr)
            return EXIT_BLOCKED
        print(json.dumps(resume(args.resume), indent=2))
        return EXIT_OK

    if args.action == "download":
        if not args.download:
            print("ERROR: --download ARTIFACT is required for the download action", file=sys.stderr)
            return EXIT_BLOCKED
        try:
            result = download_checkpoint(args.download, args.dest)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_BLOCKED
        print(json.dumps(result, indent=2))
        return EXIT_OK

    spec = build_job_spec(
        cfg, args.dataset, args.model, args.output, history_transcripts=args.history_transcripts
    )
    # A paid GPU job that dies on a missing container path still bills; refuse
    # before the SDK is touched.
    uncovered = _report_uncovered(spec)
    if uncovered:
        return uncovered
    try:
        result = submit(spec, dry_run=args.dry_run, confirmed=args.yes, cfg=cfg)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED if "SDK version unsupported" in str(exc) else EXIT_BLOCKED
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 78)
        print("JOB SPEC")
        print("=" * 78)
        print(json.dumps(spec, indent=2))
        print("=" * 78)
        print(result.get("message") or json.dumps(result, indent=2))
    return EXIT_OK


def _report_uncovered(spec: Mapping[str, Any]) -> int:
    """Print any container path the spec references but does not mount."""
    problems = uncovered_container_paths(spec)
    if not problems:
        return EXIT_OK
    print("ERROR: the job references container paths that are not mounted:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return EXIT_BLOCKED


def _render_preflight(report: Mapping[str, Any]) -> str:
    lines = ["Cloud preflight (local checks only, nothing billable)", ""]
    lines.append(f"provider                : {report['provider']}")
    lines.append(f"wandb installed         : {report['wandb_installed']} ({report['wandb_version']})")
    lines.append(f"sandbox SDK present     : {report['sandbox_sdk_present']}")
    lines.append(f"requested GPU           : {report.get('requested_gpu')}")
    lines.append(f"GPU availability        : {report['gpu_type_availability']}")
    lines.append("credentials (names only):")
    for name, status in report["credentials"].items():
        lines.append(f"  {name:20s} {status}")
    source = report.get("credential_source") or {}
    lines.append(
        f"credential resolved from: {source.get('source')} "
        f"(netrc machine: {source.get('netrc_machine')}, host: {source.get('base_url_host')})"
    )
    if report["blockers"]:
        lines.append("")
        lines.append("BLOCKERS:")
        lines += [f"  - {item}" for item in report["blockers"]]
    if report["warnings"]:
        lines.append("")
        lines.append("WARNINGS:")
        lines += [f"  - {item}" for item in report["warnings"]]
    lines.append("")
    lines.append("No download, no GPU launch and no network call was performed by this command.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
