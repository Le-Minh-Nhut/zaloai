# LifeMate — Zalo AI Challenge 2026, Track 1

Voice → action agentic benchmark: a spoken Vietnamese request plus a world
context (time, location, installed apps, conversation history) must be turned
into the right sequence of tool calls.

**Status: the default model is `openbmb/MiniCPM5-2B` and its weights are present
locally; no full training run has happened yet.** The official LifeMate dataset
**is** present under `./data` (1545 train / 634 public_test samples) and drove the
EDA, the SFT build (4368 train + 482 val decision points), ToolEnv replay and the
ASR measurement. The organizers' `zaic_tools` SDK ships with each split and is
used through `lifemate.env.official`. A LoRA forward/backward step on the real
MiniCPM5-2B weights was measured on this machine's GPU (see the status table
below); a full epoch, a full fine-tune and a leaderboard score are still absent.
Every number, file and record in `tests/fixtures/` is synthetic and labelled as
such.

## Default model

`openbmb/MiniCPM5-2B` is the default: `model.name` in
`src/lifemate/config.py`, `configs/base.yaml`, `configs/training_lora.yaml` and
`configs/inference.yaml`. `Qwen/Qwen3.5-4B` stays supported and is one flag or
one config away:

```bash
python3 scripts/train.py --model Qwen/Qwen3.5-4B   # override model.name
python3 scripts/train.py --config configs/training_lora_qwen.yaml
```

Nothing else changes per family: `lifemate.models.profiles` holds the prompt
template kwargs, the supervised terminator, the LoRA projections, the decode stop
ids and the `max_new_tokens` fallback (used only when a caller supplies none; the
shipped scripts pass `agent.max_new_tokens`, 512 for both models). See
`docs/ARCHITECTURE.md`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Le-Minh-Nhut/zaloai/blob/main/notebooks/finetune_qwen35_colab.ipynb)

Fine-tune Qwen3.5-4B (LoRA/QLoRA) on a free Colab GPU — setup, dataset, training, resume, export; see [`notebooks/README_colab.md`](notebooks/README_colab.md) (tiếng Việt).

That notebook is the **Qwen3.5-4B** walkthrough and has **never been executed on
Colab**; its VRAM/time figures are labelled ESTIMATE there. No Colab notebook
exists for the MiniCPM default.

## Current status vs. verified

| Status item | Verified? | Reason |
| --- | --- | --- |
| CODE COMPLETE | YES | All modules, CLI entrypoints, scripts, and verification gates implemented. Gates **A–O, R, P: 17 PASS / 0 FAIL** with the torch interpreter (`.venv`/stdlib run: 16 PASS + R SKIP because no torch is importable there). Gate Q stays unmeasured — no leaderboard run |
| DEFAULT MODEL | YES | `openbmb/MiniCPM5-2B` (`LlamaForCausalLM`, 42 layers, hidden 2048, 16 heads / 2 KV heads, head_dim 128, intermediate 6144, vocab 130560, untied embeddings, rope_theta 5e6, max_position 131072): **2,516,756,480 params**, one 5,033,557,096 B safetensors shard, 381 BF16 tensors (5,033,512,960 B), local copy at `models/MiniCPM5-2B` (10 files, 5,043,806,458 B). `Qwen/Qwen3.5-4B` is the alternative, selected by `--model` or `configs/training_lora_qwen.yaml` |
| QWEN ALTERNATIVE | PARTIAL (tests only) | `Qwen/Qwen3.5-4B` weights are **absent from `models/`** on this machine (the directory holds only `MiniCPM5-2B/` and `gipformer-68M-rnnt/`), so Qwen was **never loaded, forward-passed, trained or generated from** here. What *is* verified, all unit/static: `resolve_profile("Qwen/Qwen3.5-4B")` → `qwen3_5` (`test_profile_resolution_registry`); the prompt-rendering regression `test_qwen_prompt_rendering_is_unchanged` (frozen ChatML + closed-think bytes, rendered through the **MiniCPM** tokenizer, not Qwen weights); `test_qwen_stays_one_config_away` (`configs/training_lora_qwen.yaml` differs from `configs/training_lora.yaml` in exactly `model.name`) and `test_qwen_is_selectable_by_yaml_and_cli`; LoRA targets stay auto-discovered with no pinned list — `QWEN3_5_PROFILE.lora_target_modules == ()`, checked against the released module tree on a **meta-device** build, zero weight bytes (`test_lora_target_modules_exist_in_released_qwen3_5_architecture`, `test_lora_trainable_count_matches_the_shipped_config`); and the Qwen terminator id **248046** (`test_stop_ids_union_every_declared_eos`). The Qwen regression evidence is therefore **static and unit-level, not a run against real Qwen weights** |
| TUNING DEFAULT | YES | `training.tuning_mode: lora` in `configs/base.yaml`, `configs/training_lora.yaml` is the default `--config`, and `python3 scripts/train.py` with no arguments loads it. `--tuning {full,lora,qlora}` overrides. Full FT still exists (`configs/training_full.yaml`) but does not fit a 16 GB card: `--dry-run --tuning full` on the default config (MiniCPM5-2B, 2,516,756,480 params, seq 8192, micro-batch 1, gradient checkpointing on) prints **TOTAL 43.62 GB**, `verdict: does_not_fit` (weights 10.07 + gradients 10.07 + optimizer 20.13 + activations 2.82 + chunked logits 0.53, vocab 130560) — formula output, not a measured peak. The shipped LoRA default estimates **8.54 GB / `fits`**, QLoRA **4.76 GB / `fits`** |
| SYNTHETIC TESTED | YES | `python3 tests/run_tests.py` → 516 passed / 13 skipped / 0 failed; `.venv/bin/python -m pytest -q` → 522 / 7 / 0; `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q` → 528 / 1 / 0. `/tmp/cudaenv` (`torch 2.15.0+cu132`, transformers 5.22.2) is the torch interpreter of record; the repo venv has no `torch` |
| REAL DATASET TESTED | PARTIAL | Real splits present under `./data` (`train` 1545 samples, `public_test` 634) and exercised: `inspect_data.py` / `run_eda.py` / `prepare_dataset.py` (**4850** decision points = 4368 train + 482 val, 0 sample drops, 19 `unobservable_ref_target` point drops) / collator truncation measurement with the **MiniCPM tokenizer**: full-sequence tokens p50 6198, p90 8435, p95 9089, p99 10075, max 11037 (val p99 9309, max 9623); at 8192, **4298 kept / 488 truncated (11.17 %) / 70 dropped `target_value_truncated` (1.60 %)**, val 477 kept / 41 truncated / 5 dropped, and 0 kept prompts lose the tool schema block or the `# Công cụ khả dụng` catalog. No training epoch has run on it |
| REAL MODEL TESTED | YES (forward/backward) | MiniCPM5-2B + LoRA r16 bf16 on the local RTX 5070 Ti (16.57 GB, `sm_120`), gradient checkpointing, micro-batch 1, chunked CE (1024), batch **padded to exactly 8192 tokens**, no `PYTORCH_CUDA_ALLOC_CONF` → **one train-only step: OK, peak allocated 14.37 GB, peak reserved 15.17 GB**, loss 2.911356, 3.78 s wall. A real adapter save + reload and a greedy generation also ran. Quality is not established: the **base** model did not reliably close with `</action>` (it sometimes ended with EOS; the guard raised rather than parsing a truncated decision). With the format spec in the prompt it produced one clean action block in 13 tokens. No fine-tuned checkpoint has been evaluated |
| GPU TRAINING TESTED | PARTIAL | One LoRA **train step + eval step + epoch-end adapter save in the same process** fits at seq **padded to exactly 8192** without `PYTORCH_CUDA_ALLOC_CONF`: 14.37 GB allocated / **15.17 GB reserved** peak, i.e. **1.40 GB reserved headroom** on the 16.57 GB card (eval-only step: 8.34 GB / 9.48 GB, ~1.0 s; save inside the cycle: 245 ms). The train-only step alone also peaks at 14.37 / 15.17 GB; with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the train-only step reserves 14.88 GB instead (optional, saves 0.29 GB reserved and 0 allocated). seq **12288 OOMs** ("Tried to allocate 502.00 MiB; 14.02 GiB allocated, 123.88 MiB free"), and seq 8192 OOMs **without** chunked CE in train (`ForCausalLMLoss` upcasts the whole `[seq, vocab]` logits to fp32, 8192×130560×4 ≈ 3.98 GiB) or in eval before `evaluate` was chunked. No whole epoch, no convergence number, no full fine-tune |
| QLORA | NO (not executed) | Implemented for real — `BitsAndBytesConfig` NF4 with double quant, `prepare_model_for_kbit_training`, no `.to()` on a quantized model, and the CLI refuses to silently downgrade — but **GPU execution was not tested**: `bitsandbytes` is not installed in the torch environment used here |
| COLAB | NO (not executed) | `notebooks/finetune_qwen35_colab.ipynb` (Qwen3.5-4B walkthrough, `notebooks/README_colab.md`) has **never been run on Colab**. Verified only: valid JSON, every code cell `ast.parse`s, every flag it uses exists in the scripts' `--help`. Its VRAM/time table is labelled ESTIMATE |
| ASR TESTED | PARTIAL | Gipformer-68M-RNNT (int8, `sherpa-onnx`, CPU) on all **1545** real train utterances: WER **17.91 %** / CER **11.13 %**, sentence exact match 8.74 %, RTF ~0.011 (82 ms mean latency), 0 errors. The ASR metric layer is measured; the end-task score is not |
| DOCKER BUILT | NO | Docker daemon unavailable in execution container. `docker/Dockerfile` was re-pinned to `nvidia/cuda:12.8.1` + `whl/cu128` because cu124 ships no `sm_120` kernels, but the image was **never built and never run** — status `NOT RUN / UNVERIFIED`, not passing |
| W&B TRACKING | YES (offline) | `configs/training_lora_wandb.yaml` (copy of `training_lora.yaml`, three deltas: `report_to: ["wandb"]`, `wandb.enabled: true`, `wandb.mode: 'offline'`) drives `lifemate.tracking.WandbTracker`. Measured offline: 2 logged steps with 5 secrets/PII values in the payload, 0 hits in the 15,813 B wandb wrote. Needs **both** `report_to: ["wandb"]` and `wandb.enabled: true`; `WANDB_MODE` overrides the config value; `--smoke` bypasses tracking. `describe_environment()` is an allowlist (no `hostname`/`user`) after it leaked both. No run has ever been uploaded — see `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` Phụ lục C |
| CLOUD SUBMIT TESTED | NO | No cloud GPU is needed: everything above ran on the local RTX 5070 Ti. The cloud path stays unrelated and unproven. CPU sandbox **reachable and verified** (auth via `~/.netrc`, created, executed code, deleted, `list()` → 0); GPU sandbox **denied at the org entitlement**: `Permission denied: this organization is not entitled to create GPU sandboxes` (identical for all 7 shapes). 0 GPU-seconds consumed, no `nvidia-smi`, no training. Data (1.1 GB) and the Qwen-era 8.68 GiB weight download have no delivery route, and the job spec's host bind-mounts cannot exist in a serverless sandbox — `cloud/README.md` |
| W&B REQUIRED | NO | W&B is optional and off by default (`training.report_to: []`, `wandb.enabled: false`). Training, evaluation and inference all run without it; metrics fall back to `<output_dir>/<run_name>/metrics.jsonl` |
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

# 4. Train (LoRA on MiniCPM5-2B is the default; full FT is explicit)
python3 scripts/train.py --dry-run          # no weights loaded, preflight only
python3 scripts/train.py                    # configs/training_lora.yaml
python3 scripts/train.py --tuning qlora     # 4-bit base; needs bitsandbytes (untested on GPU here)
python3 scripts/train.py --config configs/training_full.yaml   # ~70 GB, 80 GB-class GPU

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
│   ├── loader.py           # tokenizer and causal LM loader
│   └── profiles.py         # per-family prompt/terminator/stop-ids/LoRA targets
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
