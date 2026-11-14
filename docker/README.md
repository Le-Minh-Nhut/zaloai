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
| `ASR_MODEL_ID` | `g-group-ai-lab/gipformer-68M-rnnt` | gipformer checkpoint id, forwarded as `--override asr.model_id=<id>` when it differs from the default. A non-default id **requires** `LIFEMATE_FEATURE_ASR_GIPFORMER_V15` enabled, else the entrypoint exits **2**: picking another checkpoint without the flag would silently move the baseline. |
| `ASR_MODEL_DIR` | `/models/asr` | gipformer ONNX directory, forwarded as `--override asr.model_dir=<dir>` only when it holds `tokens.txt`. The *default* path being absent is normal and just skips the override; an **explicitly set** path without `tokens.txt` exits **2**. |
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

## Optional features

Every feature flag in `src/lifemate/features/registry.py` is **OFF by default** and
stays off unless its variable is set. `python` reads the variables straight from
`os.environ` (`lifemate.features.overrides.parse_env_overrides`); the entrypoint
never enables, rewrites, unsets or case-folds one — it **validates** them and
prints a summary.

Accepted values: `1|0|true|false|yes|no|on|off`, case-insensitive (exactly
`TRUE_TEXT`/`FALSE_TEXT` in `overrides.py`). Any other value, or any other name
under the `LIFEMATE_FEATURE_` prefix, exits **2** before a weight is touched —
`parse_env_overrides` raises on both, so the entrypoint fails early instead of
letting a typo surface as a traceback after model load.

The entrypoint prints one stderr line before `exec`:
`lifemate: features=NONE` when nothing is enabled, else the sorted names, e.g.
`lifemate: features=LIFEMATE_FEATURE_ASR_NBEST,LIFEMATE_FEATURE_PROMPT_ITN`.

```bash
# env — the primary way
docker run … -e LIFEMATE_FEATURE_ASR_NBEST=1 lifemate-infer
# equivalent without touching the environment: the same `features:` path via --override
docker run … -e EXTRA_ARGS="--override features.asr.nbest_enabled=true" lifemate-infer
# or a features-only experiment overlay (configs/experiments/*.yaml)
docker run … -v "$PWD/configs/experiments/asr_nbest.yaml:/app/cfg.yaml:ro" \
    -e EXTRA_ARGS="--config /app/cfg.yaml" lifemate-infer
```

**Artifacts are mounted, never downloaded.** `--network none` plus
`HF_HUB_OFFLINE=1` means an enabled feature that needs files must find them on a
mounted path. No feature needs a package the image does not already install (see
the comment block in `pyproject.toml`), so enabling one never implies a
`pip install`; `--build-arg LIFEMATE_EXTRAS=…` exists only for the pre-existing
`asr`/`asr-whisper`/`onnx` groups.

| Variable | Feature id | What it needs on disk |
|---|---|---|
| `LIFEMATE_FEATURE_AUDIO_VAD` | `audio.vad` | nothing — the repo's VAD is the pure-python energy detector in `src/lifemate/asr/audio_ops.py` (no silero, no webrtcvad package). |
| `LIFEMATE_FEATURE_AUDIO_RMS_NORMALIZATION` | `audio.rms_normalization` | nothing (same module). |
| `LIFEMATE_FEATURE_ASR_NBEST` | `asr.nbest` | nothing extra — `sherpa-onnx` is already installed; keep `nbest_size` in `2..10` or validation fails. |
| `LIFEMATE_FEATURE_ASR_GIPFORMER_V15` | `asr.gipformer_v15` | the **Gipformer-1.5 ONNX directory** at `ASR_MODEL_DIR` (`encoder/decoder/joiner .onnx` + `tokens.txt`). Selecting a non-default `ASR_MODEL_ID` *requires* this flag (exit 2 otherwise). The file names are **UNVERIFIED** — `model_specs.py` marks the 1.5 entry `verified=False` and never observed its file set. |
| `LIFEMATE_FEATURE_ASR_MULTI_ASR` | `asr.multi_asr` | the secondary model's own artifacts **and** `features.asr.secondary_model_id`; an empty id fails validation. The merge layer itself imports only the stdlib. |
| `LIFEMATE_FEATURE_PROMPT_ITN` | `prompt.itn` | nothing (`src/lifemate/asr/normalization.py` is stdlib-only). |
| `LIFEMATE_FEATURE_PROMPT_NBEST_FORMAT` | `prompt.nbest_format` | nothing, but it **depends on** `asr.nbest`: enabling it alone fails validation. |
| `LIFEMATE_FEATURE_DECODING_CONSTRAINED` | `decoding.constrained` | nothing beyond the image's `torch`+`transformers` (a plain `LogitsProcessor` + prefix trie; no outlines/llguidance/xgrammar). |
| `LIFEMATE_FEATURE_POSTPROCESSING_ENHANCED_SCHEMA_GUARDRAIL` | `postprocessing.enhanced_schema_guardrail` | nothing — `src/lifemate/agent/guardrail.py` is stdlib-only; it validates against the tool catalog that ships with the dataset under `/data`. |
| `LIFEMATE_FEATURE_TRAINING_NOISY_ASR` | `training.noisy_asr` | training only: the training split under `/data` **plus** ASR transcripts to substitute (the same kind of file `HISTORY_TRANSCRIPTS` points at). |
| `LIFEMATE_FEATURE_TRAINING_SLOT_MASKING` | `training.slot_masking` | training only: the training split (stdlib-only masking). |
| `LIFEMATE_FEATURE_TRAINING_TOOL_DROPOUT` | `training.tool_dropout` | training only: the training split (stdlib-only masking). |

**Wiring status (UNVERIFIED for inference output).** `docs/FEATURES.md` records
every flag's consuming module as `PENDING`, and `scripts/infer.py` has no
`--enable-feature` / `--disable-feature` argument at all. So today a flag set in
the container is **validated, summarised and carried in `os.environ`**, but it
does not yet change what `scripts/infer.py` generates. The `EXTRA_ARGS` forms
above write the same `features.<section>.<field>` values into the config object
(verified with `lifemate.config.load_config` + `_set_dotted`); whether any
generation path honours them is not verified here. The one exception noted in
`docs/FEATURES.md` is `decoding.constrained`, wired inside
`HFGenerator(constrained=True)` but still without a call site in `infer.py`.

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
* No network path at **run** time: `docker/entrypoint.sh` contains no `pip install`,
  `curl`, `wget`, `git clone`, `snapshot_download` or `huggingface-cli` on any
  executable line (asserted by `tests/test_docker_entrypoint.py::test_entrypoint_never_installs_or_downloads`).
  The Dockerfile *does* `pip install` — at **build** time only, which is what
  produces the image; nothing in it downloads model weights.
* Optional features, entrypoint side (host, stub `python`, no daemon):
  `bash -n docker/entrypoint.sh` exits 0; with the `LIFEMATE_FEATURE_` prefix
  stripped from the environment the stderr summary reads `lifemate: features=NONE`;
  `LIFEMATE_FEATURE_ASR_NBEST=1 LIFEMATE_FEATURE_PROMPT_ITN=true` prints
  `features=LIFEMATE_FEATURE_ASR_NBEST,LIFEMATE_FEATURE_PROMPT_ITN` (sorted, values
  passed through untouched); `LIFEMATE_FEATURE_ASR_NBEST=maybe` and a
  misspelled name under the prefix each exit **2** naming the variable, with no
  `python` invocation at all; `ASR_MODEL_ID=g-group-ai-lab/gipformer1.5-68M-rnnt`
  alone exits **2**, and with `LIFEMATE_FEATURE_ASR_GIPFORMER_V15=1` appends
  `--override asr.model_id=…`. Asserted by `tests/test_docker_entrypoint.py`
  (12 tests, including a negative control that injects a truthy default into a
  copy of the script and requires the default-off check to reject it).
* Registry ↔ deployment parity: the 12 `LIFEMATE_FEATURE_*` names in
  `entrypoint.sh` equal `{env_name_for(spec.id) for spec in lifemate.features.registry.FEATURES}`,
  and the same set appears in this README — so a new flag cannot ship undocumented
  (`tests/test_docker_entrypoint.py::test_entrypoint_advertises_exactly_the_registry_ids`,
  `::test_feature_vars_are_the_same_in_entrypoint_and_readme`).
* Optional-dependency groups: `pip install -e .` resolves to the unchanged
  `dependencies = []`, and **no** group was added for any feature flag; the
  per-flag justification is the comment block in `pyproject.toml`. Verified by
  reading each consumer module's imports (`audio_ops.py`, `multi_backend.py`,
  `constrained.py`, `asr_variants.py`, `augmentation.py`, `guardrail.py`): all
  are stdlib + `lifemate.*`, or need only `torch`+`transformers` from `train`.
* `EXTRA_ARGS` feature equivalents: `--override features.asr.nbest_enabled=true`
  (and the 11 sibling paths) round-trip through `lifemate.config.load_config` +
  `_set_dotted` on the host, and `configs/experiments/*.yaml` all load with
  `load_config`.

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

**Optional features: validated in the container's shell, NOT executed as features.**
* No `docker build` was run, so `ARG LIFEMATE_EXTRAS` and the guarded
  `pip install ".[${LIFEMATE_EXTRAS}]"` line are **UNVERIFIED**: the Dockerfile's
  `RUN` block was never executed. An operator checks it with
  `docker build -f docker/Dockerfile -t lifemate-infer .` (baseline) and
  `docker build -f docker/Dockerfile --build-arg LIFEMATE_EXTRAS=asr-whisper -t lifemate-extra .`.
* No feature flag was ever observed changing inference output. `docs/FEATURES.md`
  lists every consumer module as `PENDING`, and `scripts/infer.py` has no
  `--enable-feature` / `--disable-feature` argument. The entrypoint therefore
  **validates and reports** the flags; whether `scripts/infer.py` honours them is
  unverified here and out of this document's scope.
* The Gipformer-1.5 artifact set is `verified=False` in
  `src/lifemate/asr/model_specs.py`: no weights exist on this machine, so the file
  names for `LIFEMATE_FEATURE_ASR_GIPFORMER_V15` / `ASR_MODEL_ID` are the
  conventional sherpa-onnx role names, never observed.
* `asr.multi_asr` needs `features.asr.secondary_model_id` (empty fails validation)
  and a secondary model's artifacts; no such model was mounted or run.
