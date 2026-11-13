# Training guide — LifeMate action policy (Qwen3.5-4B)

Fine-tunes Qwen3.5-4B to emit one frozen action block per turn:

```
<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": "06:30"}}</action>
<action>{"type": "finish"}</action>
```

**What was actually exercised on this machine, and what was not**

| Item | Status here |
|---|---|
| SFT data construction, alignment, leakage guards, splits | ✅ CPU-verified (`tests/test_trajectory.py`) |
| Collator masking, drop policy, padding, checkpoint policy, preflight math | ✅ CPU-verified (`tests/test_training.py`) |
| `prepare_dataset.py` on the synthetic fixture | ✅ CPU-verified |
| `train.py --dry-run` (full + LoRA) | ✅ CPU-verified |
| `train.py --smoke` (tiny model, 2 steps, CPU) | ✅ verified in a throwaway CPU venv with `torch`+`transformers` (24,736-param model, exit 0, `full/step-2` written) — **BLOCKED** in the repo venv, where the script correctly exits 3 and writes no checkpoint |
| `train.py --smoke --tuning lora` | ✅ same venv (`peft`), 7 target modules discovered, adapter-only checkpoint written |
| Real SIGTERM → `interrupt-<step>` → `--resume` | ✅ same venv: exited 130, `interrupt-8` written, resumed to step 12 |
| Gradient anti-leak control | ✅ same venv: masked max relative grad delta 5e-8 vs 1.1e-1 for an unmasked (leaky) control |
| Full 4B training on GPU | ❌ NOT RUN — GPU present (RTX 5070 Ti, 16303 MiB = 17.09 GB decimal / 15.92 GiB) and a CUDA-capable build **exists** at `/tmp/cudaenv` (`torch 2.15.0+cu132`, `sm_120`, bf16 OK), but no Qwen3.5-4B weights. The repo venv (`2.15.0+cpu`) and `/tmp/torchenv` stay CPU-only. Nothing below is a measured training result. |

No model weights are present locally: the Qwen3.5-4B checkpoint (738 tensors
over 2 shards, 8.68 GiB) was never fetched. The real dataset **is** present (`./data`:
`train` 1545 samples, `public_test` 634) and drove every count below — the
shipped SFT build at `outputs/datasets/train/` is **4368 train + 482 val = 4850**
decision points (0 sample drops, 19 `unobservable_ref_target` point drops:
17 train / 2 val). Every memory number in this guide is an **estimate** from the
formulas in `src/lifemate/training/memory_preflight.py`.

Test and gate status on the current tree (not a training result): the suites are
**479 / 485 / 491** (stdlib `passed=479 skipped=13` / `.venv` pytest `485 passed, 7 skipped` /
`/tmp/cudaenv` pytest `491 passed, 1 skipped`; measured 2026-11-13) and `verify_all.py`
gates are **17/17 PASS**, with **gate Q unmeasured because GPU work is paused**. The
older `468 / 468 / 473` triple is historical: it was measured on `/tmp/torchenv`, a CPU
venv deleted during the 2026-11-13 session — `/tmp/cudaenv` is the torch interpreter of
record now, and four test files (`test_wandb_tracking`, `test_sandbox_probe`,
`test_sandbox_credentials`, `test_sandbox_measured`) have been added since. Docker is
**NOT RUN / UNVERIFIED**: the image was never built, and
the old `cu124` pin could not run `sm_120` kernels, so it was changed to
`cuda:12.8.1` + `whl/cu128`.

---

## 1. Prepare the SFT dataset

`prepare_dataset.py` converts gold trajectories into *causal decision points*: one
(prompt, target action) pair per agent step.

```bash
# real data (needs the official dataset extracted under --data-root)
python3 scripts/prepare_dataset.py \
    --data-root ./data --split train --output outputs/datasets \
    --val-ratio 0.1 --seed 42

# with a real tokenizer for exact token counts (optional; downloads the tokenizer only)
python3 scripts/prepare_dataset.py \
    --data-root ./data --split train --output outputs/datasets \
    --tokenizer Qwen/Qwen3.5-4B --max-seq-length 8192

# synthetic fixture (works offline)
python3 scripts/prepare_dataset.py \
    --data-root tests/fixtures/synthetic_lifemate --split train --output /tmp/sft_synth
```

Writes `outputs/datasets/train/`:

```
train.jsonl   # {"sample_id","step_index","messages":[system,user],"target","meta"}
val.jsonl
manifest.json # counts, seed, ratio, id hashes, group hash, tokenization status
stats.json    # decision-point accounting + drop reasons + token-size summary
```

Prompts are rendered at preparation time and stored verbatim, so training cannot
drift from the inference-time renderer (`agent/prompt_builder.py`).

### Thinking is disabled in the rendered prompt

`training/collator.py::render_prompt_text` and `models/generation.py::HFGenerator.build_prompt`
both call the chat template with `enable_thinking=False`, so every prompt ends with the
closed think block (`collator.NO_THINK_SUFFIX`: think open+close tags, then a blank line).
Qwen3's default (flag absent) ends with an *open* think tag instead, which would supervise
the action inside the reasoning block and leave inference
mid-thought with no cue to stop. A tokenizer that rejects the kwarg is retried without it;
the bare-tokenizer ChatML fallback appends the same closed-think suffix. A tokenizer that
*has* a template and raises is a hard `LifeMateError` on both sides — never a silent
reformat, which would train on prompt bytes inference never produces.

**Read the warnings.** `visibility_limited` means the gold trajectory's final tool
result is missing, so no terminal `finish` example could be built.
`unobservable x-ref` means a write argument's value appears nowhere the model can
see (request, context, rendered history window, earlier action, earlier
observation, or the tool's own enum). Those targets are **dropped** rather than
repaired — the rest of the trajectory is kept — and counted under
`stats.json → dropped_decision_points.unobservable_ref_target`. The gold value is
never injected into a prompt.

### Decision-point rules (why the data looks like this)

| Gold trajectory | Emitted targets |
|---|---|
| N calls, all results present, last call ≠ `sys.ask_missing_info` | `call_0 … call_{N-1}`, then `finish` |
| last call is `sys.ask_missing_info` | `…, ask_missing_info` and **no** `finish` (asking is terminal for that sample) |
| zero calls (out of scope) | a single `finish` |
| last result missing | calls only, **no** `finish`; sample flagged `visibility_limited` |

Prompt *k* contains observations of steps `0..k-1` only. It never contains step
*k*'s own observation, a later observation, or `label.final_state`
(`assert_no_label_leak` hard-fails on violation).

---

## 2. Train — full fine-tuning (DEFAULT)

```bash
python3 scripts/train.py --config configs/training_full.yaml
```

`--tuning` defaults to config setting (`full`); `training.tuning_mode: full` in the config agrees.
Full mode means **every parameter** trains: `configure_full_finetuning` unfreezes
everything and raises `LifeMateError` if `trainable != total`. The run never
silently falls back to LoRA.

### Multimodal towers policy (Qwen3.5 vision/audio towers)
In LifeMate Track 1, audio inputs are processed by upstream decoupled ASR
(Gipformer 68M int8 default, or PhoWhisper optional), and the causal LM operates on text decision prompts.
If a multimodal variant of Qwen3.5 (exposing `visual`, `vision_tower`, `audio_tower`, etc.)
is loaded:
* `everything` (default): full FT trains **all** parameters (`trainable == total`).
* `lm_only`: explicitly freezes vision/audio towers; only LM parameters train.
The policy is printed at startup (`multimodal: policy=everything (towers: ...)`)
and reflected in the trainable parameter count. Silent partial training is forbidden.
```bash
# override the tuning mode explicitly (LoRA)
python3 scripts/train.py --config configs/training_full.yaml --tuning lora
# LoRA config, own hyper-parameters
python3 scripts/train.py --config configs/training_lora.yaml --tuning lora
```

### Other invocations

```bash
# plan only: no weights loaded, no training launched
python3 scripts/train.py --dry-run --param-count 4000000000 --tuning full

# resume: continue from a step-<n> directory (verifies step/hash/tuning mode)
# run_name defaults to the tuning mode, so a full-FT run lands in .../full/step-<n>
python3 scripts/train.py --config configs/training_full.yaml --resume outputs/checkpoints/full/step-1500
# the resume guard compares only optimisation-relevant settings.
# MAY change across a resume: dataset mount (data.root/--dataset-dir), device, W&B names,
#   ASR/inference paths, and checkpoint bookkeeping (logging/save/eval cadence, save_total_limit,
#   report_to, dataloader_num_workers).
# MAY NOT change: lr, weight decay, epochs, per-device batch size, grad accumulation, warmup,
#   lr scheduler, gradient checkpointing, clip norm, seed, max_seq_length, validation_ratio,
#   optimizer, precision, tuning mode, model identity/precision, and (in LoRA mode) LoRA hparams.

# explicit dataset location
python3 scripts/train.py --dataset-dir outputs/datasets/train

# short run
python3 scripts/train.py --max-steps 20
```

Exit codes: `0` ok · `1` config/training error · `2` dataset missing · `3`
environment cannot run the request (no torch/CUDA, model unloadable) · `130`
interrupted after saving `interrupt-<step>`.

---

## 3. Hyper-parameters (starting point, **not tuned**)

No hyper-parameter search has been run — no training run has happened at all (the
weights are absent; the CUDA toolchain is not the blocker, see the table above).
These are conservative literature defaults for 4B SFT.

| Setting | Full | LoRA | Rationale |
|---|---|---|---|
| `learning_rate` | `5e-6` | `2e-4` | Full FT of a 4B model needs a small LR; LoRA tolerates ~40× more because only adapters move. |
| `per_device_train_batch_size` | 1 | 1 | Memory-bound at seq 8192: with the measured LoRA adapter count (32,464,896 params) microbatch 2 estimates 14.17 GB = 91.7 % of the 15.45 GB sample free VRAM on this machine's card (16303 MiB = 17.09 GB decimal / 15.92 GiB; that 15.45 GB is one observed sample, not a constant — free varies 3.03–15.68 GB, see §4.1); microbatch 1 keeps **11.49 GB, 3.96 GB below** that sample. Effective batch is kept at 8 by `gradient_accumulation_steps` (8 full / 8 LoRA). |
| `gradient_accumulation_steps` | 8 | 8 | Effective batch 8 sequences in both cases. |
| `num_train_epochs` | 2 | 3 | Short action-format task; more epochs risk memorising the few hundred trajectories. |
| `max_seq_length` | 8192 | 8192 | On the **current** build (4850 records) 4096 truncates **72.6 % (3520/4850)** and leaves **846 of 3242** kept `tool_call` prompts (26.1 %) without the parameter-schema block of the tool they were asked to call. At 8192 that is **47/4850 (0.97 %)** truncated overall, **47/4368 (1.08 %) train / 0/482 val**, and **0/3305** schema-block loss since the collator pins `# Công cụ khả dụng` to the preserved tail (2026-11-08) — the pre-pinning figure was 3/3324 on the superseded 4869-record build. The block is the literal `### <tool> (kind=` string in the decoded, truncated prompt, counted over every decision point whose target is a `tool_call` in `outputs/datasets/train/{train,val}.jsonl`. The largest record needs 9016 tokens (prompt + target + eos; a target is never truncated). Estimate first (`--dry-run`). |
| `warmup_ratio` | 0.05 | 0.03 | Standard warmup stabilises the first steps. |
| `lr_scheduler_type` | cosine | cosine | `linear`/`constant` also supported. |
| `weight_decay` | 0.01 | 0.01 | Applied to matrices only (biases/norms excluded). |
| `precision` | bfloat16 | bfloat16 | bf16 avoids fp16 loss-scaling failures; CPU runs silently use fp32. This is the **compute** dtype (autocast); full fine-tuning keeps fp32 master weights — see §4 — because a bf16 master quantises a 5e-6 step away (measured: 7.4 % of elements moved in bf16 vs 100 % in fp32 over 20 steps). |
| `gradient_checkpointing` | true | true | ~5× smaller activation term, ~20% slower. |
| `gradient_clip_norm` | 1.0 | 1.0 | Guards occasional spikes in tool-call JSON. |
| `save_strategy` / `eval_strategy` | epoch | epoch | Few epochs; more granular checkpoints add little. |
| `save_total_limit` | 2 | 2 | Keeps disk bounded; older step dirs are pruned. |
| LoRA `r` / `alpha` / `dropout` | — | 16 / 32 / 0.05 | r=16 with alpha=2r is the conventional starting point. |
| LoRA `target_modules` | — | auto (attention + MLP) | Discovery intersects known projection names with the model's real module names. |

---

## 4. Memory plan for 4B full fine-tuning

Formula (decimal GB, `1e9` bytes) from `training/memory_preflight.py`:

```
weights          = params × 4                (fp32 master; AdamW writes the update
                                              into the parameter tensor, so a bf16
                                              master quantises a 5e-6 step away)
gradients        = params × 4                (fp32, matching the master)
optimizer state  = params × 8                (AdamW fp32 m, v + fp32 master)
activations      = layers × seq × microbatch × hidden × k × 2   (bf16 compute)
                   k = 2 with gradient checkpointing, else 10
total            = weights + gradients + optimizer state + activations
```

LoRA is the same formula with `bpp = 2` (bf16 frozen base), and gradients /
optimizer computed over the **adapter** count only, not `params`.

With `params = 4.205751e9`, `layers = 32` (released `text_config`), `hidden = 2560`,
`seq = 8192`, microbatch 1, gradient checkpointing on (`k = 2`):

`trainable_source` decides which LoRA line you are reading. The first preflight
runs before any weights exist (`scripts/train.py:343`) and can only report the
0.5 % placeholder; the second (`:430`, after the adapters are attached) reports
`trainable_source: measured`. The table below is the **measured** path:
32,464,896 trainable parameters (12 suffixes over 248 modules, 0.766 % of the
4,205,751,296-parameter base) — not the old 0.5 % placeholder, whose 21,028,756
must never be quoted as a measurement.

| Term | Full (grad ckpt) | LoRA (grad ckpt) |
|---|---|---|
| weights | 16.82 GB | 8.48 GB (bf16 base, frozen) |
| gradients | 16.82 GB | 0.06 GB (adapter params, measured) |
| optimizer state | 33.65 GB | 0.26 GB |
| activations | 2.68 GB | 2.68 GB |
| **total** | **69.98 GB** | **11.49 GB** |
| checkpoint size | 33.65 GB | 8.54 GB |
| verdict at the 15.45 GB free sample (see §4.1 — free varies 3.03–15.68 GB) | `does_not_fit` (4.09×) | **74.3 %** of that sample — `fits` **iff** free ≥ 12 GB at launch, 3.96 GB headroom at that sample |

Gradient checkpointing **is** modelled: `memory_preflight` reads
`training.gradient_checkpointing` and uses `k = 2` instead of `10`, so leaving the
flag on cuts the activation term ~5x (at 8192, microbatch 1: 13.42 GB → 2.68 GB, so
LoRA totals 22.22 GB without checkpointing vs 11.49 GB with it). It does not touch
weights, gradients or optimizer state — those dominate, which is why full FT is unaffected.

Same estimator at `seq = 4096` for reference: full 69.15 GB, LoRA 10.14 GB
(microbatch 1). Sequence length is therefore *not* the binding constraint for
either mode; the fixed parameter/optimizer cost is.

### 4.1 Free VRAM here is a moving quantity, not a constant

The requirement is fixed: **11.49 GB** for LoRA, **69.98 GB** for full FT. What
moves is how much is free. Free VRAM on this host varies with a foreign tenant
that shares the card, so it is not a property of the machine but of the moment:
observed samples span **3.03 – 15.68 GB** within one day on the same card and
driver, and two of those samples were **78 seconds apart** (7.72 → 3.03 GB,
same foreign pid growing 7582 → 12062 MiB). The four-row timestamped table is in
`docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` §6.1b. Therefore the go/no-go
check is not a documented number but a measurement **immediately before launch**:

```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Rule: **free ≥ 12 GB** (the 11.49 GB requirement plus margin). Below that, wait
for the tenant to exit and measure again — do not panic-reconfigure the run.

**Conclusion (estimator arithmetic, not a measured peak allocation):** 4B full
fine-tuning (69.98 GB) needs an **≥80 GB class GPU** for headroom, and the current
`tuning_mode` default is still `full` — so on this card a default run is a
`does_not_fit` plan, not a run. 2× 40 GB with DeepSpeed ZeRO-3 optimizer sharding
also works. LoRA at **11.49 GB** is estimated to fit this machine's card
(17.09 GB total; 15.45 GB free at the measured sample — see §4.1, free varies
3.03–15.68 GB, so the go/no-go is `fits` **iff** free ≥ 12 GB at launch) at
**74.3 %** of that sample's free VRAM, with **3.96 GB**
headroom; that is an assumption about allocator/devices overhead, not a
measurement — confirm with a short real run and `torch.cuda.max_memory_allocated()`.

bf16 speed in full fine-tuning comes from autocast (`SFTTrainer._autocast`),
not from a bf16 parameter dtype: at `learning_rate: 5e-6` a bf16 parameter
moves for only ~7 % of its elements per 20 steps (measured), i.e. the run is
effectively frozen.

Always confirm on the target machine:

```bash
python3 scripts/train.py --dry-run --param-count 4000000000 --tuning full
```

The report prints the model's real free/total VRAM when CUDA is visible, and
`verdict: unknown` (never a fake measurement) when it is not.

---

## 5. Cloud / GPU run

Local dry run first (this machine), then the real job on a GPU box:

```bash
# 1. this workstation: verify the plan and the dataset
python3 scripts/prepare_dataset.py --data-root ./data --split train --output outputs/datasets
python3 scripts/train.py --dry-run --param-count 4000000000 --tuning full

# 2. GPU box: install the training extras once
pip install -e '.[train]'

# 3. GPU box: full fine-tuning, logged to W&B
python3 scripts/train.py --config configs/training_full.yaml

# optional: enable W&B in configs/wandb.yaml (enabled: true, project, entity, mode)
```

Checkpoints land in `outputs/checkpoints/<run_name>/step-<n>/` and contain the
model, the tokenizer, `training_state.json` and `optimizer_state.pt`. Upload the
directory that suits your storage; nothing here uploads automatically.

### Metrics logging

`training.report_to: ["wandb"]` (or `wandb.enabled: true`) routes metrics through
`lifemate.tracking.WandbTracker`, which **redacts before sending**: keys matching
`api_key|token|secret|password|credential` become `[REDACTED]`, and PII keys
(`user_profile`, `audio_text`, `messages`, `history`, …) are replaced with
`[DROPPED_PII]`. `scripts/train.py` never calls `wandb.init` itself, so there is
no unredacted path.

When W&B is disabled, uninstalled, or offline, metrics fall back to
`<output_dir>/<run_name>/metrics.jsonl` (same redaction, one warning) — they are
never silently dropped. `--smoke` uses a null tracker.

Each step directory also gets `training_state.json` (step, epoch, seeds, RNG
state, tuning mode, dataset manifest hash), `optimizer_state.pt`, and the saved
model/adapter.

### Interruption and crash safety

* `SIGINT`/`SIGTERM` set a stop flag; the current optimizer step finishes, an
  `interrupt-<step>` checkpoint is written, the process exits `130`.
* An unhandled exception writes `crash-<step>` metadata (best effort) and re-raises.
* A step directory is never reused: resuming or re-running against a populated
  `<output_dir>/<run_name>/step-<n>` raises instead of overwriting.

---

## 6. Fail-fast conditions

| Condition | Behaviour |
|---|---|
| `--tuning` invalid | `train.py` exits 2 via argparse (argparse's usage-error code) |
| Dataset dir has no `train.jsonl` | exit 2 + the exact `DatasetNotFound` text + the prepare command |
| Zero usable examples after masking | `LifeMateError` naming the drop reasons (e.g. `target_too_long`) |
| `target` too long for `max_seq_length` | example dropped, counted in `dropped["target_too_long"]`; the target is **never** truncated |
| Model returns no `loss` | `LifeMateError` (labels were probably not passed) |
| Full mode, any parameter frozen | `LifeMateError` (`trainable != total`) |
| LoRA target set empty | `LifeMateError` listing discovered module names |
| LoRA attached but a non-`lora_` param trainable | `LifeMateError` |
| Resume: LoRA↔full mismatch, manifest hash change, missing `training_state.json` | `LifeMateError` |
| No torch / no CUDA for a real run | exit 3 with the preflight and a pointer to `--dry-run` / `--smoke` |

---

## 7. Checking the training signal is honest

Two properties are enforced by tests rather than promised by documentation:

1. **Prompt tokens never reach the loss.** `labels` are `-100` over the prompt
   and the target verbatim afterwards (`assert_loss_masking`). With torch present,
   `test_gradient_independent_of_prompt_tokens` trains on the same target with two
   different prompts and asserts the gradients are *identical* — a stronger check
   than comparing tensors.
2. **No causality leak.** `tests/test_trajectory.py` asserts by marker string that
   prompt *k* contains observations `< k` only, that no future result is borrowed,
   and that `final_state` never appears.

Validation reports `val_loss` (teacher-forced token cross-entropy) and
`val_token_accuracy` (share of correct target tokens). Neither is the competition
metric: use `scripts/evaluate.py` for action-level scoring.

### CPU tiny-model validation

To verify the real transformers training loop without downloading Qwen weights or requiring a GPU, build a local randomly-initialized model and run CPU validation:

```bash
# 0. Build the tiny model and tokenizer in /tmp/tiny_lifemate (run once)
python3 -c "from pathlib import Path; from tests.test_tiny_e2e import make_tiny_model_and_tokenizer; make_tiny_model_and_tokenizer(Path('/tmp/tiny_lifemate'))"

# 1. Prepare SFT dataset with tiny tokenizer
python scripts/prepare_dataset.py --data-root tests/fixtures/synthetic_lifemate --split train --output /tmp/sft_tiny --tokenizer /tmp/tiny_lifemate

# 2. Full fine-tune 4 steps on CPU
python scripts/train.py --model /tmp/tiny_lifemate --dataset-dir /tmp/sft_tiny/train --output-dir /tmp/ckpt_full --max-steps 4 --seq-len 512 --config configs/training_full.yaml --device cpu

# 3. Resume from step 4 to step 6 (monotonic step counter)
python scripts/train.py --model /tmp/tiny_lifemate --dataset-dir /tmp/sft_tiny/train --output-dir /tmp/ckpt_full --resume /tmp/ckpt_full/full/step-4 --max-steps 6 --seq-len 512 --config configs/training_full.yaml --device cpu

# 4. LoRA fine-tune 4 steps on CPU (proves adapter gradient flow)
python scripts/train.py --model /tmp/tiny_lifemate --dataset-dir /tmp/sft_tiny/train --output-dir /tmp/ckpt_lora --max-steps 4 --tuning lora --device cpu

# 5. Export and merge LoRA adapter into base weights
python scripts/export_model.py --ckpt /tmp/ckpt_lora --merge-lora --output /tmp/merged
```

*Coverage limit*: A tiny random model proves that data pipelines, collators, gradient backprop, optimizer steps, checkpointing, resume, and adapter merge plumbing work end to end on CPU. It does not validate model generation quality, reasoning capabilities, or convergence on real competition data.

---

## 8. Files in this slice

```
src/lifemate/data/preprocessing.py      normalisation, duplicate hashing, feature buckets
src/lifemate/data/trajectory_builder.py causal decision points, alignment, leak guard
src/lifemate/data/split_builder.py      deterministic stratified split + leakage guard
src/lifemate/training/collator.py       prompt masking, truncation/drop policy, padding
src/lifemate/training/full_finetuning.py
src/lifemate/training/lora_finetuning.py
src/lifemate/training/memory_preflight.py
src/lifemate/training/trainer.py        own SFT loop, checkpoints, resume, interrupts
src/lifemate/training/callbacks.py      StopController, trackers
configs/{base,training_full,training_lora,inference,evaluation,wandb}.yaml
scripts/prepare_dataset.py  scripts/train.py
tests/test_trajectory.py  tests/test_training.py
```
