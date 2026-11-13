# Docker (inference image)

Offline batch image for `scripts/infer.py`. Nothing is installed at run time and
no port is exposed; the image is meant to be run with `--network none`.

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

## Run

GPU run (requires a CUDA-enabled torch build in the image):
```bash
docker run --rm --gpus all --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/full/step-<n>":/models/ckpt:ro \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -v "$PWD/outputs/asr/history_transcripts.jsonl":/history/history_transcripts.jsonl:ro \
    -v "$PWD/outputs":/outputs \
    -e SPLIT=public_test -e MODEL_MODE=full -e ASR=gipformer -e DEVICE=cuda \
    -e HISTORY_TRANSCRIPTS=/history/history_transcripts.jsonl \
    lifemate-infer
```

CPU run:
```bash
docker run --rm --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/full/step-<n>":/models/ckpt:ro \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -v "$PWD/outputs/asr/history_transcripts.jsonl":/history/history_transcripts.jsonl:ro \
    -v "$PWD/outputs":/outputs \
    -e SPLIT=public_test -e MODEL_MODE=full -e ASR=gipformer -e DEVICE=cpu \
    -e HISTORY_TRANSCRIPTS=/history/history_transcripts.jsonl \
    lifemate-infer-cpu
```

Output: `/outputs/submission.jsonl` on the host under `outputs/`.

| Variable | Default | Meaning |
|---|---|---|
| `SPLIT` | `public_test` | split to predict |
| `DATA_ROOT` | `/data` | dataset root inside the container |
| `MODEL_MODE` | `full` | `base` · `full` · `lora` |
| `CHECKPOINT` | `/models/ckpt` | checkpoint dir (ignored for `base`) |
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
| checkpoint dir | `/models/ckpt` | read-only |
| ASR model dir | `/models/asr` | read-only (e.g. Gipformer ONNX; whisper is optional host extra) |
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
* Flag propagation: `docker/entrypoint.sh` forwards `SPLIT`, `MODEL_MODE`,
  `ASR`, `DEVICE`, `CHECKPOINT`, `OUTPUT`, `HISTORY_TRANSCRIPTS` to `scripts/infer.py`.
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

## What has NOT been verified

**Docker build and runtime validation: NOT RUN (never built or run — no docker daemon).**
The image has never been built or started in this environment: there is no Docker daemon, the
installed torch build is CPU-only (`2.15.0+cpu`), there are no model weights
and no network to pull the base image. Treat the Dockerfile as reviewed code, not
as a tested artifact.
