# LifeMate Track 1 — Implementation Readiness & Handoff Report

- **Date**: 2026-11-07
- **Target**: Zalo AI Challenge 2026 Track 1 "LifeMate" (Vietnamese Voice Request → Multi-step Tool Calling Agent)
- **Status Summary**: Complete implementation; data path tested on the real extracted splits, model path synthetic-tested only (no weights; the repo venv's `2.15.0+cpu` has `cuda = None`, but a CUDA build exists at `/tmp/cudaenv` — `torch 2.15.0+cu132`, `sm_120` verified — so the weights are the actual blocker), ready for GPU cutover.

> ## ⚠ SUPERSEDED SNAPSHOT — dated 2026-11-07. The numbers in this document are NOT current.
>
> This file is kept for three things that are **still accurate and safe to rely on**:
>
> - the **§1 module → test-coverage map** (which package owns which behaviour, and which test file covers it);
> - the **§2 architecture and dataflow description** (ASR → decision prompt → action → `zaic_tools` observation → submission);
> - the **package responsibilities** in §1 and the workstream layout in §13.
>
> Everything **quantitative** here — test counters, SFT build size, truncation denominators, LoRA /
> full-FT memory estimates, gate status, per-file test counts — predates this session's 26 fixes, the
> SFT rebuild, the gate re-runs and the LoRA parameter correction. Do not quote those figures; the
> historical values appear in §3, §5, §6 and §12 with no marking, and only this notice marks them stale.
>
> **Live documents (use these for numbers):**
>
> - `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` — experiment readiness, in Vietnamese;
> - `docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md` — dataset and defect audit, in Vietnamese;
> - `README.md` status table — one status line per dimension.
>
> **Figure families that changed (old → new):**
>
> - test suites, stdlib / `.venv` pytest / torch venv: `450 / 450 / 455` with 6 / 6 / 1 skips
>   → `468 / 468 / 473` → **`479 / 485 / 491`** (skips `13 / 7 / 1`, measured 2026-11-13).
>   The `455`/`473` figures came from `/tmp/torchenv`, a CPU venv **deleted during the
>   2026-11-13 session**; `/tmp/cudaenv` (`torch 2.15.0+cu132`, transformers 5.22.2) is the
>   torch interpreter of record now. Both older triples are historical;
> - SFT build: `4385` records → **4850** (**4368** train + **482** val decision points, 0 sample drops);
> - decision points dropped for `unobservable_ref_target`: unstated → **19** (**17** train + **2** val);
> - truncation at `max_seq_length` 8192: `1.05 %` of 4385 → **47** of 4850 (**0.97 %**; 47/4368 = 1.08 % of train, 0/482 val);
> - truncation at 4096: `75.23 % (3299/4385)` → **72.6 % (3520/4850)**;
> - `tool_call` prompts losing their parameter-schema block at 4096: `27.7 % (921/3324)` → **846 of 3305**;
> - schema-block loss at 8192: `3/3324` → **0/3305**; whole catalog-section loss at 8192: `0` → **0/3305**;
> - LoRA trainable parameters: the `11.31 GB` line below is the **0.5 % placeholder** (21,028,756) → actually measured **32,464,896** (12 suffixes over 248 modules, `trainable_source=measured`);
> - LoRA estimate at seq 8192 / microbatch 1 / grad-checkpointing: `11.31 GB` (8.41 + 0.04 + 0.17 + 2.68) → **11.49 GB** (8.48 + 0.06 + 0.26 + 2.68);
> - LoRA share of the 15.45 GB free VRAM: `0.73×` → **0.74× (74.4 %)**; headroom: `4.14 GB` → **3.96 GB**;
> - full fine-tuning: `69.98 GB` → **69.98 GB, unchanged** (re-derived against the real 4,205,751,296-parameter base) and still `does_not_fit` on this card;
> - gates: no gate count is stated in this file → the current tree is **17/17 PASS** (A–O, P, R), with gate **Q unmeasured because GPU work is paused**;
> - per-file test counts in §12: `test_agent.py` 45 → **56**, `test_training.py` 52 → **68**, `test_eval.py` 22 → **31**, `test_submission.py` 11 → **11**, `test_gipformer.py` 22 → **31**;
> - "Official dataset: NOT RUN — no official split on disk" (§12) → both splits **are** on disk and were exercised: `train` **1545** samples, `public_test` **634**;
> - Docker: the `cu124` pin could not run `sm_120` kernels, so `docker/Dockerfile` was re-pinned to `cuda:12.8.1` + `whl/cu128`; the image was still **never built — UNVERIFIED**.

---

## 1. Implemented Modules

| Package | Key Files | Responsibility | Test Coverage File |
|---|---|---|---|
| `lifemate.data` | `discovery.py`, `loader.py`, `schemas.py`, `catalog.py`, `trajectory_builder.py`, `split_builder.py`, `validator.py` | Dataset path discovery, tolerant JSONL parsing, tool catalog extraction, causal decision point generation, stratified train/val splits | `tests/test_data.py`, `tests/test_trajectory.py` |
| `lifemate.env` | `base.py`, `mock.py`, `official.py`, `factory.py` | Environment protocol, deterministic mock simulator, strict official `zaic_tools` adapter wrapper | `tests/test_env_mock.py` |
| `lifemate.agent` | `agent.py`, `protocol.py`, `prompt_builder.py`, `tool_parser.py`, `tool_executor.py`, `tool_registry.py`, `stopping_policy.py`, `state_tracker.py` | Multi-step agent loop, XML action parsing, observation feedback, loop termination, history compaction | `tests/test_agent.py`, `tests/test_tool_registry.py` |
| `lifemate.models` | `loader.py`, `generation.py`, `checkpoint.py` | Causal LM loading (full/LoRA/base), greedy generation, model checkpoint saving and export | `tests/test_model_loading.py` |
| `lifemate.training` | `trainer.py`, `collator.py`, `full_finetuning.py`, `lora_finetuning.py`, `memory_preflight.py`, `callbacks.py` | SFT trainer, prompt loss masking (`labels=-100`), full-parameter optimizer setup, LoRA PEFT injection, VRAM math | `tests/test_training.py` |
| `lifemate.asr` | `interface.py`, `normalization.py`, `metrics.py`, `oracle_backend.py`, `whisper_backend.py`, `gipformer_backend.py` | Audio loading, Vietnamese text normalization (`to_llm_text` vs `for_wer`), WER/CER metrics, critical token diagnostics, oracle reader, Gipformer ONNX transducer (default), PhoWhisper backend (optional comparison) | `tests/test_asr.py` |
| `lifemate.eda` | `dataset_analysis.py`, `audio_analysis.py`, `tool_analysis.py`, `trajectory_analysis.py`, `report.py`, `visualization.py` | Comprehensive dataset statistics, audio properties, tool co-occurrences, markdown reports | `tests/test_eda.py` |
| `lifemate.evaluation` | `evaluator.py`, `metrics.py`, `submission.py`, `submission_validator.py`, `error_analysis.py` | Multi-step accuracy, LCS call alignment, submission JSONL serialization, format and integrity validation | `tests/test_eval.py`, `tests/test_submission.py` |
| `lifemate.tracking` | `wandb_tracker.py`, `artifact_manager.py` | W&B experiment telemetry, secret redaction, SHA-256 artifact hashing and tracking | `tests/test_eval.py` |
| Core utilities | `config.py`, `compliance.py`, `errors.py`, `logging_utils.py`, `cli.py` | Hierarchical config, prohibited path access guard, exception hierarchy, unified CLI dispatcher | `tests/test_compliance.py`, `tests/test_inference.py` |

---

## 2. Architecture & Dataflow

Detailed design specifications are documented in `docs/ARCHITECTURE.md`.

```mermaid
flowchart LR
    A[Audio .wav + Context] --> B[ASR: Gipformer 68M int8]
    B --> C[Agent Decision Prompt]
    C --> D[Qwen3.5-4B SFT Model]
    D --> E{Action Type}
    E -->|tool_call| F[ToolEnv: zaic_tools]
    F -->|Observation| C
    E -->|finish| G[Final State Hash]
    F -->|get_tool_calls & get_state_hash| H[Submission JSONL]
    G --> H
```

- **Data Ingestion**: Audio files (`16kHz mono .wav`) converted by Gipformer-68M-RNNT (default, ~80 MB int8 ONNX) or PhoWhisper (optional comparison) into Vietnamese text transcripts. Backends return recognizer text verbatim; the single LLM-input `to_llm_text()` normalization is applied backend-agnostically in `scripts/infer.py::_transcribe`, so the gold-vs-ASR comparison is not confounded.
- **Decision Cycle**: Current transcript, conversation history, context (time, location, installed apps), and prior observations rendered into causal decision prompt.
- **Model Generation**: Causal LM emits structured `<action>{"type": "tool_call"|"finish", ...}</action>`.
- **Environment Execution**: Official `zaic_tools.ToolEnv` validates and executes calls; results returned as observation.
- **Submission Serialization**: Calls and `state_hash` copied directly from environment calls, never synthesized.

---

## 3. Actual Test Results

Executed on local test host (`Linux 7.0.0-31-generic x86_64`, Python 3.12.3):

### Stdlib Runner (`python3 tests/run_tests.py`)
```
$ python3 tests/run_tests.py
passed=450 failed=0 skipped(needs fixtures)=6
skipped modules: ['test_tiny_e2e.py', 'test_tiny_generation.py']
```
```
$ .venv/bin/python tests/run_tests.py
passed=450 failed=0 skipped(needs fixtures)=6
skipped modules: ['test_tiny_e2e.py', 'test_tiny_generation.py']
```
- **Timestamp**: 2026-11-08T10:56 (system runner), 2026-11-08T10:56 (.venv runner) — both re-measured in the same pass as `outputs/verification/verify_all.json`
- **Environment**: Bare system Python 3.12.3 (stdlib + PyYAML only) and the local `.venv` (Python 3.12.11). Both runs report identical totals. The 6 skipped tests are `test_tiny_e2e.py` (5) and `test_tiny_generation.py` (1), excluded as skipped modules because heavy ML dependencies (`torch`, `transformers`, `peft`) are absent.

### Pytest Runner (`.venv/bin/python -m pytest -q`)
```
$ .venv/bin/python -m pytest -q
450 passed, 6 skipped
```
- **Timestamp**: 2026-11-08T10:56 (pytest runner) — re-measured with the counts below
- **Environment**: Local `.venv` (pytest 9.2.0, numpy, matplotlib, pyyaml). The 6 skips are the torch-gated tests in `tests/test_tiny_e2e.py` (5) and `tests/test_tiny_generation.py` (1); both files carry `pytestmark = pytest.mark.skipif(not HAVE_DEPS, ...)`, so they are reported as skipped rather than passing without asserting.

### Torch-Gated Verification (**historical** — measured on `/tmp/torchenv`, deleted 2026-11-13)
```
455 passed, 1 skipped
```
```
$ PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest tests/test_tiny_e2e.py tests/test_tiny_generation.py -q
6 passed in 1.99s
```
- **Timestamp**: 2026-11-08T10:57 (full suite), then the tiny files
- **Environment**: Throwaway CPU venv at `/tmp/torchenv`, **deleted during the 2026-11-13 session**
  (`ls -d /tmp/torchenv` → `No such file or directory`). `/tmp/cudaenv`
  (`torch 2.15.0+cu132`, `transformers 5.22.2`, `peft 0.21.2`) is the torch interpreter of
  record now; on it the suite is `491 passed, 1 skipped`. Historical versions from dist-info:
  - `torch`: **2.15.0+cpu**
  - `transformers`: **5.22.2**
  - `peft`: **0.21.2**
- **Skip Explanation**: Exactly 1 test skipped: `tests/test_asr.py::test_whisper_backend_without_torch_is_actionable` (`:376`). This is an environment-conditional skip (`pytest.skip` when torch is importable) because the missing-package exception path cannot be reached when torch is installed; it is not a hidden failure.
- **Gradient Anti-Leak Verification**:
  - Test: `tests/test_training.py::test_gradient_independent_of_prompt_tokens PASSED`
  - Output: `masked max_rel_delta=4.96e-08 vs unmasked (control) 1.09e-01`
- **LoRA Trainable & Gradient Verification**:
  - Test: `tests/test_training.py::test_lora_freezes_base_and_trains_only_adapters PASSED` (base frozen, adapter gradients flow, 64 trainable params / 26.2%)
  - Test: `tests/test_training.py::test_lora_discovery_and_empty_set PASSED`
  - Test: `tests/test_tiny_e2e.py::test_tiny_lora_train_and_adapter_change PASSED` (LoRA adapter SHA-256 changes pre/post 2 steps)

## 4. Synthetic EDA Results

Exact statistics extracted from `outputs/eda_synthetic/report.md` over synthetic fixtures (`tests/fixtures/synthetic_lifemate`):

> **SYNTHETIC TEST DATA — NOT OFFICIAL COMPETITION STATISTICS**

- **Samples Inspected**: 14 records, all 14 in the `train` split.
- **Separate Splits Note**: `public_test` (3 records, 1 deliberate missing-audio negative control) is covered by `inspect_data.py`, not by this single-split EDA run.
- **Audio Files Analyzed**: 14 audio files in `train`, 100% verified 16,000 Hz, 16-bit mono PCM.
  - Duration: mean `1.312s`, min `0.500s`, max `2.125s`, total duration `18.375s`.
- **Tool Catalog**: 13 tools across 5 applications (`alarm`: 3, `calendar`: 3, `messaging`: 3, `smart_home`: 3, `sys`: 1).
- **Trajectory Distribution**:
  - Total tool calls: 17 calls across 14 samples (mean `1.214` calls/sample).
  - Call counts: 2 zero-call (out-of-scope), 7 single-call, 5 multi-call.
  - Missing-info meta-tool invocations (`sys.ask_missing_info`): 1 sample (7.1%).
  - Failed tool results preserved in trajectories: 4 / 17 calls (23.5%).
  - Read-before-write dependency satisfaction: 3/9 trajectories with write operations.

---

## 5. Full Fine-Tuning Readiness

- **Default Mode**: Confirmed default `tuning_mode == "full"` in `load_config()`, `configs/base.yaml`, and `scripts/train.py`.
- **Invariants Verified**:
  - `configure_full_finetuning()` unfreezes all parameters and asserts `trainable == total`.
  - Half-frozen models (e.g. loaded LoRA checkpoints) are unfrozen; failure to unfreeze raises `LifeMateError`.
  - Supervised collator masks prompt tokens (`labels[:prompt_len] == -100`) and preserves target supervision.
  - Gradient anti-leak test proves prompt tokens produce zero gradient updates.
  - Optimizer (AdamW), cosine decay learning rate scheduler, gradient clipping, checkpoint resume, and graceful interrupt verified on tiny model in CPU venv.
- **Memory Math (Qwen3.5-4B, Sequence Length 8192, Microbatch 1, estimator only)**:
  - Model weights: 4.2B × 4 bytes = **16.82 GB** (fp32 master — see the precision note below)
  - Gradients: 4.2B × 4 bytes = **16.82 GB**
  - Optimizer state (AdamW fp32 m, v + fp32 master): 4.2B × 8 bytes = **33.65 GB**
  - Heuristic activation memory (gradient checkpointing enabled, `k = 2`): **2.68 GB**
  - Estimated Total VRAM: **69.98 GB**
  - Target Hardware: Requires **A100 (80GB)**, **H100 (80GB)**, or multi-GPU setup with DeepSpeed ZeRO-3 / FSDP.
  - The estimator reports 66.68 GB for the same run at its 4.0B default parameter count; both figures are formula output, not a measured peak.
- **Precision**: parameters stay fp32 in full fine-tuning and bf16 comes from `torch.autocast`. AdamW applies its update to the parameter tensor itself, so a bf16 master rounds a `lr = 5e-6` step away — measured over 20 steps on `N(0, 0.02)` weights, **7.4 %** of elements moved in bf16 versus **100 %** in fp32.
- **Sequence budget**: `max_seq_length: 8192` in both shipped configs (was 4096, change applied). Measured over the 4385-record SFT build with the real tokenizer: 4096 middle-truncated **75.23 %** of records and **27.7 % (921/3324)** of `tool_call` decision points lost the parameter-schema block of the tool they were asked to call (the literal `### <tool> (kind=` string in the decoded, truncated prompt, train+val), with the whole `# Công cụ khả dụng` section lost in **4.0 % (133/3324)**; at 8192 that is **1.05 %** truncated, **0.09 % (3/3324)** schema-block loss and **0** section loss; since the 2026-11-08 collator change (catalog pinned to the tail) schema-block loss is **0/3324**.
- **GPU Status**: `GPU TRAINING: NOT RUN` (GPU present — RTX 5070 Ti, 16303 MiB = 17.09 GB decimal / 15.92 GiB — but the repo venv has no `torch` at all and the CPU-only `/tmp/torchenv` it used to have was deleted on 2026-11-13, and there are no weights). A CUDA-capable build **does** exist at `/tmp/cudaenv` (`torch 2.15.0+cu132`, `sm_120` verified), so the toolchain is no longer the blocker — the **missing Qwen3.5-4B weights** are. `/tmp/cudaenv` is the only torch interpreter on the host.

---

## 6. LoRA Readiness

- **Configuration**: Explicitly selectable via `--tuning lora` or `configs/training_lora.yaml` (`r=16, alpha=32, dropout=0.05`).
- **Invariants Verified**:
  - Base model parameters are completely frozen (`requires_grad=False`).
  - PEFT LoRA adapters attach to target projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`).
  - Gradients flow strictly into adapter weights (`adapter_params > 0`, base gradients are `None`).
  - Adapter weights can be merged into base weights via `scripts/export_model.py --merge-lora`.
  - LoRA dry-run memory preflight estimate: **11.31 GB** total VRAM at seq 8192 / microbatch 1 (8.41 GB bf16 base, 0.04 GB grads, 0.16 GB optimizer state, 2.68 GB activations with gradient checkpointing). That is 0.73x of the 15.45 GB free on this machine's card (17.09 GB total) — 4.14 GB headroom — so it is estimated to fit a 16 GB-class GPU (this card included). Estimator arithmetic, not a measured allocation.

---

## 7. Cloud Training Readiness

- **Tooling**: `cloud/sandbox_runner.py` provides local dry-run planning, schema verification, and a gated submit path.
- **Preflight Check (`python3 cloud/sandbox_runner.py preflight`)**:
  - Output observed: `sandbox SDK present: False` on wandb 0.31.0 (the `wandb.sandbox` shim needs the separate `cwsandbox` client, which is not installed), plus the blocker `WANDB_API_KEY is not set`, exit code 2.
  - Local verification executes without network access, credentials, or billable actions.
- **Dry-Run Specification (`python3 cloud/sandbox_runner.py submit ...`)**:
  - Generates structured JSON job specification containing entrypoint command, container mounts, environment flags (`HF_HUB_OFFLINE=1`, `WANDB_PROJECT=lifemate`), and output artifact paths.
- **Status**: `CLOUD SUBMIT: BLOCKED — never executable here`. No sandbox launcher in this SDK build, no `WANDB_API_KEY`, no launch queue, and no built/pushed `docker/Dockerfile` image; the spec also binds host directories, which the sandbox API cannot do (it takes a registry image plus in-file mounts).

---

## 8. Official ToolEnv Integration Status

- **Adapter Implementation**: `src/lifemate/env/official.py` defines `OfficialToolEnv`, wrapping `zaic_tools.ToolEnv`.
- **Strict Isolation**: No fallback to mock environment. If `zaic_tools` is missing, initialization immediately raises `LifeMateError`.
- **Observed Behavior**:
  - Command: `python3 scripts/infer.py --split public_test --data-root tests/fixtures/synthetic_lifemate --output /tmp/x.jsonl`
  - Exit Code: **2**
  - Stderr output: `ERROR: cannot create the environment: zaic_tools not installed: the official LifeMate simulator SDK is required for real evaluation...`
  - Output file: `/tmp/x.jsonl` was **NOT written**.
- **Status**: `REAL TOOLENV: NOT RUN` (Official `zaic_tools` SDK absent from repository by design).

---

## 9. Competition Compliance

- **Prohibited Directory Access**:
  - Path guard: `src/lifemate/compliance.py` intercepts any access containing `*_environment` (`train_environment`, `public_test_environment`, `private_test_environment`) and raises `ProhibitedAccessError`.
  - Static AST/regex scanner: `scan_source_for_prohibited_access('.')` finds **0** violations across `src`, `scripts`, `cloud`, `tests`, and `docker`.
- **Mock Environment Barriers**:
  - `--use-mock-env` is required for synthetic execution.
  - Refuses to write to official output path (`outputs/submission.jsonl`) when mock is active.
  - Scripted test hook `--generator dumpscript:...` rejected without `--use-mock-env`.
  - Mock runs emit prominent stderr warning banners.
- **ASR Oracle Leak Barrier**:
  - `--asr oracle` strictly forbidden on unlabeled splits (`public_test`, `private_test`).
  - On labeled splits (`train`, `val`), `--allow-oracle` is strictly required.
- **Scanner Limitations**: Scanner inspects literal string concatenations and path joins; dynamic string constructions from external network sources would bypass static regexes and are stopped by runtime path guards.

---

## 10. Missing External Resources

The following external assets are absent from this environment and must be supplied for production runs:

1. **Official Competition Dataset**: Extracted into `./data` (`train/`, `public_test/`, `private_test/`).
2. **Qwen3.5-4B Model Weights**: Downloaded locally to checkpoint directory.
3. **GPU Hardware**: A100 (80GB) or H100 (80GB) for full fine-tuning; RTX 3090/4090 or T4 for LoRA / inference.
4. **Official Simulator SDK**: `zaic_tools` wheel provided by Zalo AI organizers.
5. **Weights & Biases Account / API Key**: For remote experiment tracking and cloud sandbox submissions.

---

## 11. Exact Commands for Real Dataset Execution

When real dataset and GPU are available, execute this sequence:

```bash
# Set environment variables (every command below passes --data-root explicitly;
# this repo reads no data-root environment variable — only LIFEMATE_LOG_LEVEL)
export WANDB_API_KEY="<your-wandb-key>"
export PYTHONPATH="src"

# 1. Inspect real dataset placement and integrity (Expected exit: 0)
python3 scripts/inspect_data.py --data-root ./data --output-dir outputs/real_inspect

# 2. Run EDA on real dataset with real tokenizer (Expected exit: 0)
python3 scripts/run_eda.py --data-root ./data --split train --tokenizer Qwen/Qwen3.5-4B --output outputs/real_eda

# 2b. Transcribe a labeled split with the default ASR (Gipformer-68M-RNNT, offline, int8)
#     Prerequisites: pip install -e ".[asr]" and the pinned ONNX assets
#     (python3 scripts/download_asr_model.py --output-dir models/gipformer-68M-rnnt)
#     Writes <output> plus <output>.metrics.json; exit 2 when the backend is unavailable.
python3 scripts/transcribe_split.py --split train --data-root ./data \
    --output outputs/asr/train_predictions.jsonl

# 2c. ASR evaluation loop: WER/CER + critical-token diagnostics from those predictions.
#     Numbers are FILE-SUPPLIED, not a benchmark result; unusable input exits 2 and
#     leaves the ASR section at available: false.
python3 scripts/run_eda.py --data-root ./data --split train --output outputs/real_eda \
    --asr-predictions outputs/asr/train_predictions.jsonl

# 3. Build causal SFT dataset (Expected exit: 0)
python3 scripts/prepare_dataset.py --data-root ./data --split train --output outputs/datasets \
    --tokenizer Qwen/Qwen3.5-4B --val-ratio 0.1 --seed 42

# 4. Train full fine-tuning model on GPU (Expected exit: 0)
python3 scripts/train.py --config configs/training_full.yaml --dataset-dir outputs/datasets/train \
    --output-dir outputs/checkpoints

# (Optional: Train LoRA adapter)
# python3 scripts/train.py --config configs/training_lora.yaml --dataset-dir outputs/datasets/train \
#     --output-dir outputs/checkpoints --tuning lora

# 5. Closed-loop inference on public_test with official ToolEnv (Expected exit: 0)
python3 scripts/infer.py --split public_test --data-root ./data --checkpoint outputs/checkpoints/full/step-<n> \
    --model-mode full --asr gipformer --history-transcripts outputs/asr/history_transcripts.jsonl \
    --output outputs/submission.jsonl

# 6. Evaluate validation checkpoint (Expected exit: 0)
python3 scripts/evaluate.py --validation-only --model-mode full --checkpoint outputs/checkpoints/full/step-<n> \
    --predictions outputs/datasets/train/val.jsonl --output outputs/evaluation

# 6b. Same run plus the ASR block (file-supplied predictions, same caveat as 2c)
python3 scripts/evaluate.py --validation-only --model-mode full --checkpoint outputs/checkpoints/full/step-<n> \
    --predictions outputs/datasets/train/val.jsonl --output outputs/evaluation \
    --asr-predictions outputs/asr/train_predictions.jsonl

# 7. Final gate verification (Expected exit: 0)
python3 scripts/verify_all.py
```

---

## 12. Verification Matrix

| Status Dimension | Status | Action Required to Move to YES |
|---|---|---|
| **CODE COMPLETE** | **YES** | None; all modules, schemas, CLIs, and runners are fully implemented. |
| **SYNTHETIC TESTED** | **YES** | None; measured 2026-11-13: `.venv/bin/python -m pytest -q` → `485 passed, 7 skipped`; the stdlib runner reports `passed=479 failed=0 skipped(needs fixtures)=13`; `/tmp/cudaenv/bin/python -m pytest -q` → `491 passed, 1 skipped`. The `450`/`455` figures in §3 were measured on `/tmp/torchenv`, deleted 2026-11-13. |
| **Gipformer integration** | **PASS** | None; default `g-group-ai-lab/gipformer-68M-rnnt` @ `318322b009200f8e77a3751fa64638e5cc85a7dc`, sherpa-onnx, int8, CPU, offline, `modified_beam_search`; 22 test functions in `tests/test_gipformer.py` (mock recognizer, synthetic audio). Real weights not loaded. |
| **Checkpoint save+load** | **PASS** | None; resume round-trips **optimizer moments, scheduler position and both RNG streams** and is test-verified: `test_resume_restores_optimizer_and_rng_state`, `test_checkpoint_training_state_has_rng_and_optimizer_files`, `test_resume_rejects_config_hash_mismatch`, `test_resume_tolerates_irrelevant_config_changes` (per-key bookkeeping loop + LoRA-mode contrast), `test_config_fingerprint_is_stable_across_dict_order` (plus the existing versioning/pruning/guard tests) — `tests/test_training.py` 52 passed. Three defects were fixed to make this true: `resume()` left `optimizer=None` so `_load_optimizer_state` returned early (moments silently dropped), `run()` called `set_seed()` unconditionally and discarded the restored stream, and `config_hash` was written but never compared. The resume guard now hashes a whitelist projection (`training.*` minus `resume_from_checkpoint`/`output_dir` and the bookkeeping keys `logging_steps`, `save_strategy`, `eval_strategy`, `save_total_limit`, `report_to`, `dataloader_num_workers`, `method`; `model.*` minus `device`/`tokenizer_name`; `lora.*` only in LoRA mode) instead of the whole config, so `data.root`, `model.device`, `wandb.run_name`, the ASR/inference paths and checkpoint cadence may change across a resume, while no optimisation knob is excluded: lr, weight decay, epochs, batch/accum, warmup, scheduler, clip, seed, `max_seq_length`, precision, `tuning_mode`, model identity and (in LoRA mode) `lora.r` still block it. |
| **Agent execution** | **PASS** | None; `tests/test_agent.py` (45 passed) covers multi-step calls, read-before-write ordering, failure recovery, env-exception recording. |
| **Evaluation correctness** | **PASS** | None; teacher-forced and closed-loop metrics verified against hand-checked inputs; `tests/test_eval.py` 22 passed, `tests/test_submission.py` 11 passed. |
| **ToolEnv compliance** | **NOT RUN** | Official `zaic_tools` SDK absent; `OfficialToolEnv` only reviewable. Install the organizer wheel and run `create_environment('public_test', ...)`. |
| **REAL DATASET TESTED** | **PARTIAL** | Real splits are mounted under `./data` (`train` 1545 samples, `public_test` 634) and were exercised through `scripts/inspect_data.py`, `scripts/run_eda.py`, `scripts/prepare_dataset.py` (4385 SFT records) and the collator; no model weights, so nothing was trained or scored on them. |
| **REAL MODEL TESTED** | **NOT RUN** | Qwen3.5-4B weights not present locally; download and load via `lifemate.models.loader`. |
| **Real Qwen GPU training** | **NOT RUN** | GPU present (RTX 5070 Ti, 17.09 GB decimal / 15.92 GiB); the repo venv's `2.15.0+cpu` (`cuda = None`) is not the obstacle — a CUDA-capable build exists at `/tmp/cudaenv` (`torch 2.15.0+cu132`, `sm_120` verified), so the blocker is the missing **Qwen3.5-4B weights**; use A100/H100 for full FT. |
| **Official dataset** | **NOT RUN** | Same blocker as above: no official split on disk, so all counts here are synthetic-fixture only. |
| **Docker runtime** | **NOT RUN** | No docker daemon and no network for the base image; separately, the pre-existing `cu124` pin could not run `sm_120` kernels, so `docker/Dockerfile` was re-pinned to `cuda:12.8.1` + `whl/cu128` — the image was still **never built** (UNVERIFIED). `docker/Dockerfile` and `entrypoint.sh` are statically linted only (`tests/test_eval.py::test_dockerfile_references_only_existing_paths`, `bash -n`). |
| **Cloud submit** | **BLOCKED** | Measured 2026-11-13 against wandb 0.31.0 in `/tmp/cudaenv`: `wandb.sandbox` is a deprecated shim needing the separate `cwsandbox` client (absent), `wandb.launch`/`wandb.jobs`/`wandb.inference` do not exist, `wandb.run` is the current run object, `wandb.sweep` schedules on your own hardware. Also no `WANDB_API_KEY`, no launch queue, no built/pushed image. Preflight now reports `sandbox_sdk_present: false`; dry-run only. |
| **COMPETITION VALIDATED** | **NOT RUN** | Submit `outputs/submission.jsonl` to the official Zalo AI evaluation server; nothing has been scored by the organizers. |

---

## 13. Implementation Provenance

- **Implementation Workstreams (Five total)**:
  1. `foundation`: Data schemas, catalog, loader, split builder, error hierarchy, config loading, compliance guard (`src/lifemate/data`, `config.py`, `compliance.py`, `errors.py`).
  2. `agent-core`: Agent loop, prompt builder, action XML protocol, tool executor, tool parser, tool registry, stopping policy (`src/lifemate/agent`).
  3. `eda-asr`: EDA dataset and audio analysis, report generator, chart visualization, Vietnamese normalization, WER/CER metrics, ASR backends (`src/lifemate/eda`, `src/lifemate/asr`).
  4. `training`: SFT causal trajectory builder, supervised collator, loss masking, full fine-tuning, LoRA adapter injection, trainer, memory preflight (`src/lifemate/training`, `models`).
  5. `eval-delivery`: Evaluation metrics, submission serialization and validation, W&B telemetry tracker, artifact manager, cloud runner, Dockerfile, CLI dispatcher (`src/lifemate/evaluation`, `tracking`, `cloud`, `docker`, `cli.py`).
- **Independent Audit Pass**:
  - Verification, defect remediation, gate reconciliation (A–P, Q and R), documentation harmonization, and negative self-tests (`tests/`, `scripts/verify_all.py`, handoff reports).
- **ASR Correction & Follow-up Fixes**:
  - The PhoWhisper → Gipformer-68M-RNNT default correction (`configs/inference.yaml`, `base.yaml`, `src/lifemate/config.py`, `src/lifemate/asr/gipformer_backend.py`) and the nine follow-up fixes it required (A1 unknown-backend fallthrough, A2 compliance-guarded audio resolution in all three layers, A3 ASR-failure refusal at exit 3, A4 single `to_llm_text` call site, A5 `modified_beam_search` default, A6 protocol signature, B1 ASR-eval wiring via `transcribe_split.py` and `--asr-predictions`, B2 doc default sweep, B3 doc-drift guard) were performed by worker sessions, each fix landed with a covering test recorded in `docs/VERIFICATION_REPORT.md` (entries 31–40) and `docs/GIPFORMER_INTEGRATION_AUDIT.md`.
- **Model Routing & Substitution Record**:
  - The five implementation workstreams (four initial concurrent sessions + one follow-up delivery session) were routed to `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017` before operator routing directives pinned the primary model.
  - The final independent audit and verification pass was executed directly on `9router/ag/gemini-3.8-flash-high` as instructed.
  - Mid-project the Gemini route became unavailable: the provider returned quota-exhaustion errors, so the remaining ASR correction and follow-up fix work was re-routed to the DeepSeek route to keep the pass moving. No work was dropped and no verification claim changed as a result; every count in this report was re-measured on this host after re-routing.
  - This substitution record is documented factually; all integration claims were independently verified and proven via direct command execution during the audit.
---

## 14. Known Limitations & Unverified Items

| Item | Reason Unverified | Minimum Cost Action to Resolve |
|---|---|---|
| GPU Out-Of-Memory (OOM) | GPU present but no CUDA-enabled torch build in the repo venv (`2.15.0+cpu`, `cuda = None`) — a CUDA build does exist at `/tmp/cudaenv` (`torch 2.15.0+cu132`, `sm_120` verified), so the real gap is the missing Qwen3.5-4B weights plus the operator's GPU pause | Have the weights present and GPU work resumed, then run 1 training step on this RTX 5070 Ti (16303 MiB = 17.09 GB decimal / 15.92 GiB) with `torch.cuda.max_memory_allocated()`, or do the same on a cloud A100 |
| ASR Word Error Rate (Gipformer / PhoWhisper) | No real Vietnamese audio evaluation set with ground truth | Run `lifemate.asr.evaluate_asr()` on 50 real audio samples with Gipformer (default) or PhoWhisper |
| `zaic_tools.ToolEnv` API changes | SDK not publicly available before competition launch | Test `create_environment('public_test', ...)` once organizer wheel is provided |
| Large dataset IO throughput | Synthetic dataset has 14 samples; real dataset has thousands | Benchmark `load_samples()` with Python `multiprocessing` on full dataset |
| Network isolation in container | Docker daemon unavailable in current environment; the pre-existing `cu124` pin also could not run `sm_120` kernels and was changed to `cuda:12.8.1` + `whl/cu128`, but the image was never built (UNVERIFIED) | Run `docker build` then `docker run --network none` sanity test on staging machine |
