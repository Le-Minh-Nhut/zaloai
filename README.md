# LifeMate — Zalo AI Challenge 2026, Track 1

Voice → action agentic benchmark: a spoken Vietnamese request plus a world
context (time, location, installed apps, conversation history) must be turned
into the right sequence of tool calls.

**Status: verified on the real data, no model trained yet.** The official
LifeMate dataset **is** present under `./data` (1545 train / 634 public_test
samples) and drove the EDA, the SFT build (4368 train + 482 val decision
points), ToolEnv replay and the ASR measurement. The organizers' `zaic_tools`
SDK ships with each split and is used through `lifemate.env.official`. Still
absent: the **Qwen3.5-4B weights** — and therefore any training run or score.
Every number, file and record in `tests/fixtures/` is synthetic and labelled as
such.
## Current status vs. verified

| Status item | Verified? | Reason |
| --- | --- | --- |
| CODE COMPLETE | YES | All modules, CLI entrypoints, scripts, and verification gates implemented; gates **17/17 PASS** on the current tree (`scripts/verify_all.py --no-write`, gates A–O, R, P). Gate Q is unmeasured — GPU work is paused by the operator |
| SYNTHETIC TESTED | YES | `.venv/bin/python -m pytest -q` → `485 passed, 7 skipped` (6 torch-gated tiny tests + `test_wandb_tracking.py`, `wandb` absent from `.venv`); stdlib runner `passed=479 failed=0 skipped(needs fixtures)=13` (6 torch-gated tiny tests + 7 tests in `test_sandbox_credentials.py` that take pytest fixtures, which the stdlib runner cannot bind); torch interpreter `/tmp/cudaenv/bin/python -m pytest -q` → `491 passed, 1 skipped` (`tests/test_asr.py:376`). `/tmp/torchenv` was deleted during the 2026-11-13 session, so `/tmp/cudaenv` is the torch interpreter of record and the old `468 / 468 / 473` triple is historical |
| REAL DATASET TESTED | PARTIAL | Real splits present under `./data` (`train` 1545 samples, `public_test` 634) and exercised: `inspect_data.py` / `run_eda.py` / `prepare_dataset.py` (**4850** decision points = 4368 train + 482 val, 0 sample drops, 19 `unobservable_ref_target` point drops) / collator truncation measurement (47 at `max_seq_length` 8192). Model weights and GPU are still absent, so no training or scoring has run on it. |
| REAL MODEL TESTED | NO | Official Qwen3.5-4B weights not downloaded locally |
| GPU TRAINING TESTED | NO | GPU present (RTX 5070 Ti, 16303 MiB = 17.09 GB decimal / 15.92 GiB) and a CUDA-capable build exists locally at `/tmp/cudaenv` (`torch 2.15.0+cu132`, `sm_120` verified) — the blocker is the missing Qwen3.5-4B weights, not the toolchain. The repo venv has no `torch` installed at all; `/tmp/cudaenv` is the only torch interpreter on the host — the CPU-only `/tmp/torchenv` it replaced was deleted on 2026-11-13 |
| ASR TESTED | PARTIAL | Gipformer-68M-RNNT (int8, `sherpa-onnx`, CPU) on all **1545** real train utterances: WER **17.91 %** / CER **11.13 %**, sentence exact match 8.74 %, RTF ~0.011 (82 ms mean latency), 0 errors. The ASR metric layer is measured; the end-task score is not |
| DOCKER BUILT | NO | Docker daemon unavailable in execution container. `docker/Dockerfile` was re-pinned to `nvidia/cuda:12.8.1` + `whl/cu128` because cu124 ships no `sm_120` kernels, but the image was **never built and never run** — status `NOT RUN / UNVERIFIED`, not passing |
| W&B TRACKING | YES (offline) | `configs/training_lora_wandb.yaml` (copy of `training_lora.yaml`, three deltas: `report_to: ["wandb"]`, `wandb.enabled: true`, `wandb.mode: 'offline'`) drives `lifemate.tracking.WandbTracker`. Measured offline: 2 logged steps with 5 secrets/PII values in the payload, 0 hits in the 15,813 B wandb wrote. Needs **both** `report_to: ["wandb"]` and `wandb.enabled: true`; `WANDB_MODE` overrides the config value; `--smoke` bypasses tracking. `describe_environment()` is an allowlist (no `hostname`/`user`) after it leaked both. No run has ever been uploaded — see `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` Phụ lục C |
| CLOUD SUBMIT TESTED | NO | CPU sandbox **reachable and verified** (auth via `~/.netrc`, created, executed code, deleted, `list()` → 0); GPU sandbox **denied at the org entitlement**: `Permission denied: this organization is not entitled to create GPU sandboxes` (identical for all 7 shapes). 0 GPU-seconds consumed, no `nvidia-smi`, no training. Data (1.1 GB) and weights (8.68 GiB) have no delivery route, and the job spec's host bind-mounts cannot exist in a serverless sandbox — `cloud/README.md` |
| COMPETITION VALIDATED | NO | Official leaderboard evaluation not conducted |

Detailed readiness and handoff specifications: `docs/IMPLEMENTATION_READINESS_REPORT.md`.
It is a dated snapshot (2026-11-07): for the figures that changed, see the
superseded-snapshot notice at the top of that file, and the table above for the
current values.

## Install

Stdlib first — the data layer, mock environment and tests run on bare
`python3` (3.10+), with no third-party packages:

```bash
git clone <repo> && cd zaloai
python3 tests/run_tests.py                 # stdlib runner, no pytest needed
```

Optional extras (nothing is required to import `lifemate`):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"        # pytest + pyyaml
pip install -e ".[train]"      # torch, transformers, peft, accelerate, datasets, wandb
pip install -e ".[asr]"        # sherpa-onnx, sherpa-onnx-core, soundfile, huggingface-hub, numpy
pip install -e ".[cloud]"      # wandb
pip install -e ".[eda]"        # matplotlib, numpy
```

`pyyaml` is optional: without it, configs use a built-in YAML-subset parser.

## Dataset placement

Extract the official data so split directories sit under `--data-root`:

```
data/
├── train/
│   ├── train.jsonl
│   ├── train_tools_catalog.json
│   ├── sample_submission.jsonl
│   └── audio/<id>/<id>.wav
└── public_test/
    ├── public_test.jsonl
    ├── public_test_tools_catalog.json
    └── audio/<id>/<id>.wav
```

Split names are discovered dynamically, so new splits (e.g. `private_test`)
work without code changes. `<split>_environment/` directories are **never
read**: `lifemate.compliance` raises `ProhibitedAccessError` on any path whose
component ends in `_environment`.

Missing data produces a clean message and exit code 2:

```
$ python3 scripts/inspect_data.py --data-root ./data
Dataset not found.

Expected directory:
data/train

Please extract the official LifeMate dataset
into the configured data directory.

Implementation is available.
Real-data validation has not yet been performed.
```

That last sentence is the tool's fixed string (`src/lifemate/errors.py:31`), emitted
only when a split is absent; it is out of date as a statement about this repository —
real-data validation **has** been performed (see the status table above).

## Pipeline

The complete end-to-end pipeline (`inspect` → `eda` → `prepare` → `train` → `infer` → `evaluate` → `verify`):

```bash
# 1. Inspect dataset structure and split readiness
python3 scripts/inspect_data.py --data-root tests/fixtures/synthetic_lifemate

# 2. Exploratory Data Analysis (EDA) report and distribution summaries
python3 scripts/run_eda.py --data-root tests/fixtures/synthetic_lifemate --split train --output outputs/eda_synthetic

# 3. Prepare causal SFT dataset (decision points, prompt masking)
python3 scripts/prepare_dataset.py --data-root tests/fixtures/synthetic_lifemate --split train --output outputs/datasets

# 4. Train model (full fine-tuning is default; LoRA is selectable)
python3 scripts/train.py --dry-run --param-count 4000000000 --tuning full
python3 scripts/train.py --dry-run --param-count 4000000000 --tuning lora

# 5. Closed-loop agent inference (synthetic self-test via mock environment)
#    The scripted generator reads a JSON map sample_id -> raw model outputs;
#    write it once (one action per step, then finish):
cat > /tmp/verify_dumpscript.json <<'JSON'
{"test_single_tool": ["<action>{\"type\": \"tool_call\", \"name\": \"smart_home.device_set\", \"arguments\": {\"device_id\": \"light1\", \"on\": false}}</action>", "<action>{\"type\": \"finish\"}</action>"],
 "test_missing_info": ["<action>{\"type\": \"tool_call\", \"name\": \"sys.ask_missing_info\", \"arguments\": {\"missing_fields\": [\"thoi_gian\"]}}</action>", "<action>{\"type\": \"finish\"}</action>"],
 "test_out_of_scope": ["<action>{\"type\": \"finish\"}</action>"]}
JSON
python3 scripts/infer.py --split public_test --data-root tests/fixtures/synthetic_lifemate \
    --use-mock-env --generator dumpscript:/tmp/verify_dumpscript.json --limit 3 \
    --output /tmp/mock_submission.jsonl --asr none

# 6. Evaluation (closed-loop or validation set)
python3 scripts/evaluate.py --closed-loop --split train --use-mock-env \
    --data-root tests/fixtures/synthetic_lifemate --limit 3

# 7. Full acceptance gate verification
python3 scripts/verify_all.py
```

ASR is **Gipformer-68M-RNNT by default** (`asr.backend: gipformer` in
`configs/inference.yaml`), so `scripts/infer.py` needs no `--asr` flag. PhoWhisper
is an optional *comparison* backend only (`--asr whisper`, from
`docs/INFERENCE_GUIDE.md` §4). `--asr none` above is the mock-environment
self-test, where the transcript is irrelevant.

ASR scoring (stdlib, no GPU for the metric layer):

```bash
# transcribe a split, then score the predictions in the EDA report
python3 scripts/transcribe_split.py --split train --data-root ./data
python3 scripts/run_eda.py --data-root ./data --split train \
    --asr-predictions outputs/asr/train_predictions.jsonl --output outputs/eda
python3 scripts/evaluate.py --validation-only --asr-predictions outputs/asr/train_predictions.jsonl
```

ASR WER/CER is independent of final-state correctness, which stays
`available: false` without the official scorer.

## Unified CLI Dispatcher

All tooling is accessible via the package entrypoint:

```bash
python3 -m lifemate --help
```

Supported subcommands (dispatching to `scripts/<name>.py`):
- `inspect-data` (`scripts/inspect_data.py`)
- `eda` (`scripts/run_eda.py`)
- `prepare` (`scripts/prepare_dataset.py`)
- `train` (`scripts/train.py`)
- `infer` (`scripts/infer.py`)
- `evaluate` (`scripts/evaluate.py`)
- `export` (`scripts/export_model.py`)
- `download-checkpoint` (`scripts/download_checkpoint.py`)
- `verify` (`scripts/verify_all.py`)
## Layout

```
src/lifemate/
├── __init__.py             # package initialization and version exports
├── cli.py                  # unified command line interface dispatcher
├── compliance.py           # prohibited path detection and access guard
├── config.py               # configuration schema and YAML loader
├── errors.py               # structured domain exception hierarchy
├── logging_utils.py        # central logging setup and formatters
├── agent/                  # closed-loop multi-step tool agent
│   ├── agent.py            # core agent loop and step coordinator
│   ├── prompt_builder.py   # context and history prompt renderer
│   ├── protocol.py         # action definitions and XML tags
│   ├── state_tracker.py    # multi-turn dialogue state tracker
│   ├── stopping_policy.py  # step budgets and loop termination
│   ├── tool_executor.py    # tool execution and validation dispatcher
│   ├── tool_parser.py      # action parser from generated text
│   └── tool_registry.py    # tool catalog and schema validator
├── asr/                    # speech recognition backends and metrics
│   ├── gipformer_backend.py # Gipformer-68M-RNNT default ASR backend
│   ├── interface.py        # abstract audio backend and wav reader
│   ├── metrics.py          # WER, CER, and Levenshtein distances
│   ├── normalization.py    # Vietnamese text normalization rules
│   ├── oracle_backend.py   # ground-truth transcript reader
│   └── whisper_backend.py  # PhoWhisper speech recognition backend (optional comparison)
├── data/                   # dataset loading and SFT preprocessing
│   ├── catalog.py          # tool catalog schema and loader
│   ├── discovery.py        # split detection and file indexer
│   ├── loader.py           # robust JSONL sample reader
│   ├── preprocessing.py    # token estimators and prompt formatters
│   ├── schemas.py          # dataclasses for samples, calls, turns
│   ├── split_builder.py    # stratified train/val split generator
│   ├── trajectory_builder.py # causal decision point builder
│   └── validator.py        # schema validator and error reporter
├── eda/                    # exploratory data analysis and reports
│   ├── audio_analysis.py   # sample rate and duration statistics
│   ├── dataset_analysis.py # field presence and token distributions
│   ├── report.py           # markdown summary generator with banner
│   ├── tool_analysis.py    # tool invocation frequencies and arguments
│   ├── trajectory_analysis.py # trajectory lengths and success rates
│   └── visualization.py    # distribution plots and charts
├── env/                    # tool execution environments
│   ├── base.py             # abstract tool environment interface
│   ├── factory.py          # environment loader and selector
│   ├── mock.py             # deterministic in-memory mock environment
│   └── official.py         # zaic_tools simulator adapter wrapper
├── evaluation/             # submission scoring and validation
│   ├── error_analysis.py   # error breakdown by tool and failure type
│   ├── evaluator.py        # metrics computation and report generator
│   ├── metrics.py          # tool precision, recall, state accuracy
│   ├── submission.py       # submission JSONL serializer and writer
│   └── submission_validator.py # submission schema and integrity validator
├── models/                 # model loading and inference wrappers
│   ├── checkpoint.py       # save and resume checkpoint utilities
│   ├── generation.py       # greedy and sampled text generation
│   └── loader.py           # tokenizer and causal LM loader
├── tracking/               # experiment tracking and artifact storage
│   ├── artifact_manager.py # directory hashing and artifact packaging
│   └── wandb_tracker.py    # Weights & Biases telemetry client
└── training/               # supervised fine-tuning pipelines
    ├── callbacks.py        # checkpointing and evaluation callbacks
    ├── collator.py         # prompt-masked supervised collator
    ├── full_finetuning.py  # end-to-end full parameter training
    ├── lora_finetuning.py  # PEFT LoRA adapter injection
    ├── memory_preflight.py # parameter count and VRAM estimator
    └── trainer.py          # training loop and optimizer setup
configs/                    # YAML configuration files for pipelines
scripts/                    # standalone entrypoint scripts
cloud/                      # cloud submission and sandbox runner
docker/                     # container build and entrypoint script
docs/                       # architecture and implementation guides
tests/                      # test suite and synthetic fixtures
```

## Ground rules

- No official schema is asserted as fact anywhere; parsers are tolerant and keep
  unknown fields in `.raw`.
- Nothing under a `*_environment` path is opened, by policy and by guard.
- No model or dataset downloads, no GPU requirement for any test.
- Seeds are fixed (`training.seed = 42`); mock environments are deterministic.
