# Cloud (W&B Sandbox) — optional, off by default

`cloud/sandbox_runner.py` inspects and (optionally) submits a W&B Sandbox GPU
job. **Nothing billable happens by default:** `submit` runs with
`dry_run=True` and prints the exact command.

> Status (measured 2026-11-13 on this machine, `cwsandbox` 1.28.0): sandbox
> compute **is reachable and authenticated**. `Sandbox(Session(AuthStrategy.WANDB)).list()`
> succeeds, a CPU sandbox launches in ~3 s, and credentials come from
> `~/.netrc` (`machine api.wandb.ai`), which wandb's `FORGE_HOSTS` maps to the
> Forge instance. **GPU sandboxes are not available to this org:**
>
> ```
> CWSandboxAuthenticationError: Permission denied: this organization is not entitled to create GPU sandboxes
> ```
>
> That error was identical for all seven resource shapes attempted
> (`{"count": 1}` with no `type`, then `A100`, `A100-80GB`, `RTX-A6000`, `L40`,
> `H100`, `T4`). GPU type is therefore **irrelevant** — no `type` value helps;
> the entitlement is missing. So `nvidia-smi` never ran and **no training run
> happened**: the probe below is a CPU probe, and a CPU probe is not training.
>
> Also measured: `wandb.sandbox` is a deprecated shim over the standalone
> `cwsandbox` client (a CoreWeave product). `wandb.launch`, `wandb.jobs` and
> `wandb.inference` are not importable at all; `wandb.run` is the current run
> object (`None` outside a run) and `wandb.sweep` is a hyperparameter scheduler
> that runs agents on compute **you** provide. None of them rents a GPU.

### Probe result (verbatim, one sandbox, 60 s cap, deleted)

| Field | Value |
|---|---|
| image | `python:3.12-slim` |
| resources | none requested (CPU) |
| `max_lifetime_seconds` | 60 |
| sandbox id | `2065285f-6cf5-4572-b6a7-be13c62c176f` |
| create → accepted | 3.16 s |
| status after `wait()` | `running` |
| runtime class | `cw-kata-qemu` |
| `resource_gpu` | `None` |
| `nvidia-smi` | `sh: 1: nvidia-smi: not found` |
| `nvidia-smi --query-gpu=...` | not run (no GPU, no `nvidia-smi` binary) |
| egress to PyPI | `status 200` |
| `pip download --no-deps six` | `Saved /tmp/dl/six-1.17.0-py2.py3-none-any.whl` |
| `mounted_files` | 2030 B of `src/lifemate/errors.py` mounted at `/app/errors.py`, imported OK (`LifeMateError 5`) |
| teardown | `Sandbox.delete(<id>)` → `completed`; `list()` → 0 sandboxes |
| wall time | 11.6 s total (probe wall clock, not billed GPU time) |

`Sandbox.delete` is a **static** method (`Sandbox.delete(sandbox_id, auth=...)`);
`sandbox.delete()` raises `TypeError: missing 1 required positional argument`.

### What a real training job still needs

1. **GPU entitlement** — blocker zero. Without it nothing else matters; ask the
   W&B org admin to enable GPU sandboxes for this org.
2. **Code delivery** — `mounted_files` works and is the approved route: 2.3 MB
   of `src/`, `scripts/`, `configs/`, `docker/`. No registry, no docker daemon.
3. **Dataset** — `data/` is **1.1 GB**. Too big for `mounted_files`; needs
   object storage with a signed URL, a public dataset URL, or a baked image.
   *No such delivery plan exists yet.*
4. **Weights** — the Qwen checkpoint is **8.68 GiB**. Same problem, same missing
   plan, plus HF (or object-storage) egress inside the sandbox. Egress is proven
   for PyPI (HTTP 200), not yet for the HF hub.
5. **Image build** — `docker/Dockerfile` has **never been built** (no container
   runtime on this machine; `docker` is not on PATH).

So: 2.3 MB of code is one step away, 1.1 GB of data and 8.68 GiB of weights are
not, and the GPU entitlement is a control-plane setting, not a code change.

## 1. Credentials (names only, never printed)

Two paths, either is sufficient. The runner reports which one resolved and the
*host* name, never a value:

| Path | Detail |
|---|---|
| `WANDB_API_KEY` in the environment | `export WANDB_API_KEY=...` from https://wandb.ai/authorize |
| `~/.netrc` | a `machine <host>` entry for the resolved base URL (**this is what works on this machine**: `machine api.wandb.ai`) |

`resolve_credentials()` derives the host from `WANDB_BASE_URL` (default
`https://api.wandb.ai`), looks for a matching `~/.netrc` machine entry (or a
`default` entry), and reports `source` as `env:WANDB_API_KEY`, `netrc`, or
`absent`. It reads `login`/`password` only to test that the entry is non-empty;
no value is printed, logged, or returned.

```bash
export WANDB_ENTITY=my-team     # optional, defaults to your user
export WANDB_PROJECT=lifemate   # optional
```

`preflight` prints `WANDB_CREDENTIAL: netrc` (or `env:WANDB_API_KEY`) plus
`credential resolved from: netrc (netrc machine: api.wandb.ai, host: api.wandb.ai)`.
A netrc entry for a *different* host does not authorise: point `WANDB_BASE_URL`
at the host you actually have and the gate reports `absent`.

## 2. Preflight (free, local)

```bash
python cloud/sandbox_runner.py preflight
python cloud/sandbox_runner.py preflight --json
```

Checks: `wandb` importable, sandbox entry point present, credential source
resolved (`env:WANDB_API_KEY` / `netrc` / `absent`), requested GPU recorded. GPU
availability is reported as `unverified` — it is not inferred from a GPU name.
(Measured truth for *this* org is in the status block above: GPU creation is
denied by entitlement.)

Exit `2` when a blocker exists — i.e. when no credential resolves. With the
`~/.netrc` entry present, preflight exits `0`.

## 3. Plan the job (free, local)

```bash
python cloud/sandbox_runner.py plan --dataset ./data --model outputs/checkpoints/full/step-<n> --output ./outputs
```

Prints the job spec: `prepare_command` (build SFT records into `/outputs/datasets`),
`command` (training), `inference_command`, mounts (`/data` ro, `/models/ckpt` ro,
`/outputs` rw), env (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, …) and
artifact paths.

Two path contracts the gate enforces (`uncovered_container_paths`), both verified
locally:

* `--checkpoint` must equal the model mount's container path. `build_job_spec`
  resolves `--model` through `resolve_checkpoint`, so a run root
  (`outputs/checkpoints/full`) becomes its newest `step-<n>` and is mounted at
  `/models/ckpt`.
* `--dataset-dir` must point at a directory that *holds* `train.jsonl`, i.e.
  `/outputs/datasets/train`. The raw mount root `/data` contains
  `train/train.jsonl` instead, and `scripts/train.py --dataset-dir /data` exits 2
  ("Dataset not found") — the job therefore runs `prepare_command` first
  (`--data-root /data --output /outputs/datasets`). A spec that points
  `--dataset-dir` at a mount root is rejected before submit.

### Checkpoint persistence outside ephemeral storage
The host directory passed to `--output` (default `./outputs`) is mounted `rw` to
container `/outputs`. Checkpoints written by the trainer land under
`/outputs/checkpoints` directly on the host volume, surviving container teardown
without depending on ephemeral container disk.
## 4. Dry run (free, default)

```bash
python cloud/sandbox_runner.py submit --dataset ./data --output ./outputs
```

Prints the spec and `DRY RUN: nothing was launched and nothing was billed.`

## 5. Real submit (billable — currently a refusal, by design)

```bash
python cloud/sandbox_runner.py submit --dataset ./data --output ./outputs --no-dry-run --yes
```

The gate now needs: a resolvable credential (`WANDB_API_KEY` **or** `~/.netrc`),
`--no-dry-run`, and `--yes`. Missing any aborts before any SDK call — the
credential check is netrc-aware, so **the gate passes on this machine**.

Passing the gate does **not** launch anything. `submit` still refuses, because
the spec mounts three host directories (`/data` ro, `/models/ckpt` ro,
`/outputs` rw) while the sandbox API binds a `container_image` plus
`mounted_files` (file bytes, read-only). The refusal is actionable and names
exactly what is missing:

```
ERROR: refusing to submit: this spec mounts three host directories (/data ro,
/models/ckpt ro, /outputs rw) but the sandbox API only binds a container_image
plus mounted_files (file bytes, read-only at runtime). Nothing here can launch it.
What a real launch needs:
  1. code: container_image with the repo baked in, OR mounted_files (2.3 MB ...)
  2. dataset: 1.1 GB under data/ - too big for mounted_files, ...
  3. weights: 8.68 GiB Qwen checkpoint - same, ... plus HF egress in the sandbox
  4. GPU entitlement: ... 'Permission denied: this organization is not entitled
     to create GPU sandboxes' - no image fixes that; ask the org admin.
Verified working alternative (probe, not training): a CPU sandbox on
'python:3.12-slim' runs, accepts mounted_files, and reaches PyPI (HTTP 200).
```

Exit `2`. There is deliberately **no launch path** in this tool: inventing one
would either be a fake launch or a paid failure.

> **Cost warning.** A GPU Sandbox bills per minute for as long as it lives. The
> probe in the status block cost **0 GPU-seconds**: the seven GPU creates were
> rejected before allocation, and the single CPU sandbox is not GPU compute.

## 6. Other actions

```bash
python cloud/sandbox_runner.py resume --resume <run-or-artifact-ref>     # reports how to resume
python cloud/sandbox_runner.py download --download lifemate-full:v3 --dest outputs/checkpoints
```

### Resume workflow
To resume an interrupted or completed run:
1. Ensure host `--output` points to the directory containing the saved `step-<n>` checkpoints.
2. Re-submit pointing `training.resume_from_checkpoint` (or `--resume`) at `/outputs/checkpoints/<run_name>/step-<n>`.
3. The trainer verifies step monotonicity, dataset manifest hash, model mode (full vs LoRA), and restores optimizer, scheduler, and RNG states.

`download` delegates to `lifemate.tracking.artifact_manager.download_artifact`
(`wandb.Api().artifact(ref).download(root=dest)`), which raises a clear error if
`wandb` is missing or the reference is malformed.

## 7. Verified locally vs blocked

| Item | Status |
|---|---|
| preflight/plan/dry-run render correct JSON and exit codes | verified locally |
| credential names reported, values never printed | verified locally |
| credential from `~/.netrc` accepted, `WANDB_API_KEY` still accepted | **measured** 2026-11-13 (netrc machine `api.wandb.ai`; preflight exit 0) |
| sandbox launcher (`cwsandbox` 1.28.0 via `AuthStrategy.WANDB`) | **measured: present and authenticated** |
| CPU sandbox create → exec → delete | **measured**: 3.16 s create, `running`, deleted, 0 left |
| egress from sandbox to PyPI | **measured**: HTTP 200; `pip download six` succeeded |
| `mounted_files` (single file) | **measured**: 2030 B `errors.py` mounted + imported |
| arbitrary code exec, `with Sandbox.run(...)` | **measured**: `print('Hello from W&B Sandbox!')` → `returncode 0`, stdout exact, `with`-exit teardown left 0 running |
| default image when `container_image` omitted | **measured**: `python:3.11` (`SandboxDefaults`), silently, **no warning** |
| GPU sandbox create | **DENIED**: `Permission denied: this organization is not entitled to create GPU sandboxes` (all 7 shapes) |
| GPU type availability | **unverified** — no GPU was ever allocated, so no `nvidia-smi` output exists |
| real submit path | **BLOCKED by design**: no launch call; GPU entitlement missing; data/weights have no delivery route |
| training run on sandbox | **never happened** — a CPU probe is not training |
| artifact download | **BLOCKED**: no `wandb` in `.venv` |

## 8. If the SDK differs

`submit()` accepts exactly one shape: `wandb.sandbox.Sandbox.run`, reachable only
when the `cwsandbox` package is installed. If it is not, submit raises:

```
SDK version unsupported: this wandb build exposes no wandb.sandbox.Sandbox.run
(the 'cwsandbox' backend is not installed). wandb version: <version>.
wandb.run/wandb.sweep are not job launchers. Run the command manually instead: <command>
```

Note the two environments: the base interpreter has no `wandb`/`cwsandbox`, so it
reports `SDK version unsupported`. `/tmp/sbxenv/bin/python` has `cwsandbox`
1.28.0 and reaches the service. Run probe commands with that interpreter.

When the backend *is* present, submit still refuses: the spec mounts host
directories (`/data` ro, `/models/ckpt` ro, `/outputs` rw) and specifies no
`container_image`, while the sandbox API takes `container_image` plus
`mounted_files` (file content), not host binds. `docker/Dockerfile` has never
been built here (no container runtime installed: `docker` is not on PATH).

That is deliberate: an unverified method call would fail obscurely, and a bad
submit costs money.

### Gotchas found the hard way (cwsandbox 1.28.0)

- `Sandbox.delete` is a **static** method: `Sandbox.delete(sandbox_id, auth=...)`.
  Calling `sandbox.delete()` raises `TypeError: ... missing 1 required positional
  argument: 'sandbox_id'`.
- `Sandbox.list(...)` returns an `OperationRef`; iterate
  `Sandbox.list(auth=...).result(timeout=...)`. Statuses include terminal ones in
  `show_terminated=True` (a deleted sandbox shows `completed`).
- `Sandbox.run` returns as soon as the backend **accepts** the create; call
  `.wait()` for `RUNNING`. `sb.exec([...])` returns a `Process`; call
  `.result()` for `stdout_bytes`/`stderr_bytes`/`returncode`.
- GPU denial arrives at create time (~1.3 s), before any image pull, so a denied
  attempt costs nothing.
- `exec(...).result()` returns `ProcessResult` with fields
  `stdout`, `stderr`, `returncode`, `stdout_bytes`, `stderr_bytes`, `command`.
  There is **no `exit_code`**; use `returncode`.
- `Sandbox.run()` with no `container_image` defaults to **`python:3.11`** and
  emits **no warning**. `max_lifetime_seconds` is a ceiling, not a duration: the
  `with` block tears the sandbox down on exit in seconds.
- Grant for a default CPU sandbox: `requests={'cpu':'1','memory':'2Gi'}`,
  `limits` identical, `resource_gpu=None`, runtime class `kata-qemu`.
- `cwsandbox` prints `wandb: [cwsandbox] Loaded credentials for <host> from
  ~/.netrc.` on **stdout** while resolving auth. Not from your script.
