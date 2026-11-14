# Docker (inference image)

Offline batch image for `scripts/infer.py`. Nothing is installed at run time and
no port is exposed; the image is meant to be run with `--network none`.

## Model identity: `MODEL_NAME`

The image exposes the base model **identity** as an explicit variable, so a
submission run never depends on an implicit default buried in a YAML file.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `openbmb/MiniCPM5-2B` | Base-model identity: selects the chat template, response terminator, stop ids and default token budget, and is the weight source for `base` (no checkpoint) and `lora` (the frozen base the adapter attaches to). Supported alternative: `Qwen/Qwen3.5-4B`. |
| `MODEL_MODE` | `full` | *How* weights are loaded: `base` · `full` · `lora`. Never a model identity. |
| `CHECKPOINT` | `/models/ckpt` | *Where* a fine-tune lives: a full-finetune directory (`full`) or a PEFT adapter directory (`lora`). |
| `MODEL_ROOT` | `/models` | Mount root; also the anchor for the weights preflight. |
| `HF_HOME` | `/models/hf-cache` | Pre-mounted Hub cache; a hub-id `MODEL_NAME` resolves inside `$HF_HOME/hub`. |

These three are orthogonal — do not conflate them:

* `MODEL_NAME` = **which** model family/weights.
* `MODEL_MODE` = **how** to load it.
* `CHECKPOINT` = **where** the fine-tuned artefact is.

`MODEL_NAME` is forwarded as `--model` (`scripts/infer.py`:
`parser.add_argument("--model", default=None, help="override model.name (base model id or path)")`,
consumed by `cfg.model.name = args.model`). It beats any `model.name` in a
`--config` YAML, and it is what `HFGenerator` uses to pick the family profile —
so the stop ids and token budget follow the *selected* model, not the YAML.

Mode/identity interaction, matching `scripts/infer.py` exactly:

| `MODEL_MODE` | Weights loaded from | Tokenizer loaded from |
|---|---|---|
| `base` | `CHECKPOINT` if that directory exists, else `MODEL_NAME` | `CHECKPOINT` if it carries tokenizer files, else `MODEL_NAME` |
| `full` | `CHECKPOINT` (required) | `CHECKPOINT` if it carries tokenizer files, else `MODEL_NAME` |
| `lora` | `MODEL_NAME` base + the adapter in `CHECKPOINT` (required, needs `adapter_config.json`) | `CHECKPOINT` if it carries tokenizer files, else `MODEL_NAME` |

### Required mounts per model

MiniCPM5-2B (default) — either a plain directory or the Hub cache:
```bash
# a) plain weights directory
-v /path/to/MiniCPM5-2B:/models/MiniCPM5-2B:ro -e MODEL_NAME=/models/MiniCPM5-2B
# b) Hub cache (offline resolution of the hub id)
-v /path/to/hf-cache:/models/hf-cache:ro            # MODEL_NAME=openbmb/MiniCPM5-2B
```
Qwen3.5-4B:
```bash
-v /path/to/Qwen3.5-4B:/models/Qwen3.5-4B:ro -e MODEL_NAME=/models/Qwen3.5-4B
# or: -v /path/to/hf-cache:/models/hf-cache:ro -e MODEL_NAME=Qwen/Qwen3.5-4B
```
The directory/cache entry must hold `config.json` + `*.safetensors` (or
`pytorch_model*.bin`) **and** tokenizer files (`tokenizer_config.json` /
`tokenizer.json`). Fine-tuned artefacts are extra: `MODEL_MODE=full` mounts the
full checkpoint at `/models/ckpt`, `MODEL_MODE=lora` mounts the adapter there
(and still needs the base weights mounted).

## Offline guarantee

* The entrypoint runs a **weights preflight** before inference: it resolves the
  selected model (checkpoint / directory / `$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>`)
  and exits **2** with the exact mount to add when weights, tokenizer or the LoRA
  adapter are absent. Nothing is downloaded at scoring time.
* `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked in and re-asserted by
  the entrypoint; `src/lifemate/models/loader.py::is_local_or_offline` turns either
  into `local_files_only=True`, so `from_pretrained` can only hit local files.
* No `huggingface-cli download`, no `snapshot_download`, no `pip install`, no
  `curl`/`wget`/`git clone` anywhere in the Dockerfile or entrypoint.
* Run with `--network none`; no port is exposed and there is no `HEALTHCHECK`.

## Build

CUDA 12.8 default (`sm_120`-capable: `cu128` torch wheels carry Blackwell kernels):
```bash
docker build -f docker/Dockerfile -t lifemate-infer .
```

CPU-only build (e.g. for testing or CPU hosts):
```bash
docker build -f docker/Dockerfile \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
    --platform linux/amd64 \
    -t lifemate-infer-cpu .
```

Base image: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` (multi-stage — a build
stage compiles/installs Python dependencies via `--index-url "${TORCH_INDEX}"`,
and runtime stage copies the virtualenv).

**The dependency floors are load-bearing, not hygiene.** The image pins the verified
stack `torch>=2.15` / `transformers>=5.22` / `peft>=0.21` (interpreter of record:
`torch 2.15.0+cu132`, `transformers 5.22.2`, `peft 0.21.2`); at `transformers` 4.44–4.5x
`from_pretrained` fails on `model_type qwen3_5` and MiniCPM5-2B's `config.json` declares
`transformers_version: 5.6.2`, so relaxing them back produces an image that builds and
then cannot load either supported model. `bitsandbytes` is deliberately absent: this
image only runs inference and nothing on that path requests 4-bit quantization.

## Run

GPU run (requires a CUDA-enabled torch build in the image):
```bash
docker run --rm --gpus all --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/full/step-<n>":/models/ckpt:ro \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -v "$PWD/models/MiniCPM5-2B":/models/MiniCPM5-2B:ro \
    -v "$PWD/outputs/asr/history_transcripts.jsonl":/history/history_transcripts.jsonl:ro \
    -v "$PWD/outputs":/outputs \
    -e SPLIT=public_test -e MODEL_MODE=full -e ASR=gipformer -e DEVICE=cuda \
    -e MODEL_NAME=/models/MiniCPM5-2B \
    -e HISTORY_TRANSCRIPTS=/history/history_transcripts.jsonl \
    lifemate-infer
```

CPU run (same mounts; Qwen shown as the alternative base identity):
```bash
docker run --rm --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/lora/qwen/step-<n>":/models/ckpt:ro \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -v "$PWD/models/Qwen3.5-4B":/models/Qwen3.5-4B:ro \
    -v "$PWD/outputs/asr/history_transcripts.jsonl":/history/history_transcripts.jsonl:ro \
    -v "$PWD/outputs":/outputs \
    -e SPLIT=public_test -e MODEL_MODE=lora -e ASR=gipformer -e DEVICE=cpu \
    -e MODEL_NAME=/models/Qwen3.5-4B \
    -e HISTORY_TRANSCRIPTS=/history/history_transcripts.jsonl \
    lifemate-infer-cpu
```

Output: `/outputs/submission.jsonl` on the host under `outputs/`.

| Variable | Default | Meaning |
|---|---|---|
| `SPLIT` | `public_test` | split to predict |
| `DATA_ROOT` | `/data` | dataset root inside the container |
| `MODEL_NAME` | `openbmb/MiniCPM5-2B` | base-model identity (or a local path). `Qwen/Qwen3.5-4B` is the supported alternative. Forwarded as `--model`. See [Model identity](#model-identity-model_name). |
| `MODEL_MODE` | `full` | `base` · `full` · `lora` — loading mode only, never a model identity |
| `CHECKPOINT` | `/models/ckpt` | checkpoint/adapter dir. For `base` it is used only when the directory exists, else the `MODEL_NAME` weights load |
| `MODEL_ROOT` | `/models` | mount root and anchor of the weights preflight |
| `HF_HOME` | `/models/hf-cache` | pre-mounted Hub cache (`HF_HUB_CACHE` defaults to `$HF_HOME/hub`) |
| `OUTPUT` | `/outputs/submission.jsonl` | submission path |
| `ASR` | `gipformer` | `gipformer` (default) · `whisper` · `none` · `oracle` (needs `ALLOW_ORACLE=1` + `TRANSCRIPTS`) |
| `DEVICE` | `cuda` | `cuda` · `cpu` (use `cpu` for CPU image builds) |
| `CONFIG` | *(unset)* | optional YAML config path |
| `HISTORY_TRANSCRIPTS` | *(unset)* | history-turn ASR transcripts JSONL. A test split ships history audio only, so **419 of 634 `public_test` samples lose their prior turns** without the file. `scripts/infer.py` now **refuses to run (exit 2)** when history turns would render blank: leave the variable unset or point it at a missing file and the run stops with `ERROR: … would render N history user turn(s) blank`; only `--allow-missing-history` (diagnostic, via `EXTRA_ARGS`) proceeds. A set-but-unreadable path also exits **2**. See the mount below. |
| `HF_HUB_OFFLINE` | `1` | baked into the image |
| `TRANSFORMERS_OFFLINE` | `1` | baked into the image |

The image sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`PYTHONUNBUFFERED=1` and runs as the unprivileged user `lifemate` (uid 10001).
`HEALTHCHECK` is intentionally absent: there is no service to probe and the
container has no network.

## Mounts

| Host | Container | Mode |
|---|---|---|
| dataset root | `/data` | read-only |
| checkpoint dir / LoRA adapter dir | `/models/ckpt` | read-only |
| ASR model dir | `/models/asr` | read-only (e.g. Gipformer ONNX; whisper is optional host extra) |
| base model weights **or** Hub cache | `/models/<name>` or `/models/hf-cache` | read-only — required, selected by `MODEL_NAME` |
| history transcripts file | `/history/history_transcripts.jsonl` | read-only |
| outputs dir | `/outputs` | read-write |

## What has been verified here

* Structural lint: `docker/Dockerfile` references only paths that exist in this repository —
  asserted by `tests/test_eval.py::test_dockerfile_references_only_existing_paths`.
* Syntax check: `docker/entrypoint.sh` passes `bash -n`.
* Default job wiring: the Dockerfile declares
  `ENTRYPOINT ["/app/docker/entrypoint.sh"]`, so a plain `docker run` runs
  inference and a command appended after the image name (`docker run IMAGE python
  scripts/train.py --config /app/configs/training_full.yaml …`, which is what
  `cloud/sandbox_runner.py` submits) replaces it — verified on the host:
  `entrypoint.sh echo args-win` prints `args-win` instead of starting inference.
* Flag propagation: `docker/entrypoint.sh` forwards `SPLIT`, `MODEL_NAME` (as
  `--model`), `MODEL_MODE`, `ASR`, `DEVICE`, `CHECKPOINT`, `OUTPUT`,
  `HISTORY_TRANSCRIPTS` to `scripts/infer.py`. Every forwarded flag was checked
  against the script's real `argparse` block (`--model`, `--model-mode`,
  `--checkpoint`, `--split`, `--data-root`, `--output`, `--asr`, `--device`,
  `--config`, `--override`, `--history-transcripts`, `--allow-oracle`,
  `--transcripts` all exist).
* `MODEL_NAME` wiring, dry-run on the host with a stub `python` on `PATH`:
  default yields `--model openbmb/MiniCPM5-2B`; `-e MODEL_NAME=Qwen/Qwen3.5-4B`
  yields `--model Qwen/Qwen3.5-4B`; `-e MODEL_NAME=/models/MiniCPM5-2B` (a real
  directory) is accepted as a local weight source.
* Weights preflight, dry-run on the host against a scratch `MODEL_ROOT`: exits 2
  with an actionable message for missing hub-cache weights, a directory without
  `config.json`+safetensors, a directory without tokenizer files, `MODEL_MODE=lora`
  without `adapter_config.json`, `MODEL_MODE=full` without a checkpoint, and an
  invalid `MODEL_MODE`. A complete directory (weights + tokenizer) and a populated
  `$HF_HOME/hub/models--…--…/snapshots/<rev>` both pass.
* History-transcripts wiring: `HISTORY_TRANSCRIPTS` appends
  `--history-transcripts <path>` to the generated command; an unreadable path exits
  2 with `ERROR: HISTORY_TRANSCRIPTS set but not readable`. Asserted by
  `tests/test_eval.py::test_entrypoint_wires_history_transcripts`.
* Blank history is a hard failure, not a warning: with the mount absent (variable
  unset, or a stale path) `scripts/infer.py` exits 2 with
  `ERROR: … would render N history user turn(s) blank`, before any weights load.
  `--allow-missing-history` is the only opt-out and is diagnostic-only. Asserted by
  `tests/test_inference.py::test_blank_history_fails_unless_explicitly_allowed` and
  `::test_uncovered_history_turns_from_a_partial_file_are_refused`.
* Container-layout dry run (no daemon): the exact entrypoint argv was executed on
  the host with `/tmp/ctr/{data,history,outputs}` symlinks standing in for the
  mount points — `history transcripts: … filled=1299 existing=0 missing=0
  (samples with history=419, samples receiving text=419)`, i.e. 1299/1299 turns.
  Generation itself was NOT RUN here (`sherpa-onnx` absent, exit 2).
* The inference command itself is exercised in mock mode on the host
  (`scripts/infer.py --use-mock-env --generator … --limit 3`).
* No network path in the image: no `huggingface-cli download`,
  `snapshot_download`, `pip install`, `curl`, `wget` or `git clone` in
  `docker/Dockerfile` or `docker/entrypoint.sh` (grep-verified).

## What has NOT been verified

**Docker build and runtime validation: NOT RUN (never built or run — no docker daemon
on this machine).**
The image has never been built or started in this environment: there is no Docker daemon, the
installed torch build is CPU-only (`2.15.0+cpu`), and there is no network to pull the base image.
Treat the Dockerfile as reviewed code, not as a tested artifact. Model weights *are* present on the
host (`models/MiniCPM5-2B`), but nothing was ever mounted into an image — there is no image.

**MiniCPM5-2B support in the image is UNVERIFIED at runtime.** `MODEL_NAME`
defaults to `openbmb/MiniCPM5-2B` and the family profile (`stop ids`, chat
template, 192-token default budget) is resolved in `src/lifemate/models/profiles.py`.
The image was never built or run, so **MiniCPM support inside the container** —
tokenizer load, weights preflight, generation — has never been executed and stays
unverified. What *was* checked inside the image's logic is static: flag names,
offline loader semantics (`local_files_only=True` under `HF_HUB_OFFLINE=1`) and the
preflight decision table, all on the host with stub directories.

**Separately, the model itself was exercised on the host, outside the image.**
`models/MiniCPM5-2B` is on this machine and was really loaded, forward/backwarded
and generated from on the host GPU (see the root `README.md` status table). That
says nothing about the container: the same weights were never mounted into a built
image, because there is no image.

`shellcheck` was not available on this machine (`shellcheck` not installed, no
`pip` to install it), so the entrypoint is checked by `bash -n` plus the dry-run
scenario matrix only — not by a linter.
