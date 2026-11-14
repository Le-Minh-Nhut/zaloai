# Training guide — LifeMate action policy (default: MiniCPM5-2B + LoRA)

Fine-tunes `openbmb/MiniCPM5-2B` to emit one frozen action block per turn:

```
<action>{"type": "tool_call", "name": "alarm.set", "arguments": {"time": "06:30"}}</action>
<action>{"type": "finish"}</action>
```

`Qwen/Qwen3.5-4B` remains supported: `--model Qwen/Qwen3.5-4B` or
`--config configs/training_lora_qwen.yaml`. Nothing in this guide is
Qwen-specific unless it says so.

## 0. What was measured here, and what was not

| Item | Status on this machine |
|---|---|
| SFT data construction, alignment, leakage guards, splits | measured, CPU (`tests/test_trajectory.py`) |
| Collator masking, truncation/drop policy, padding, checkpoint policy, preflight math | measured, CPU (`tests/test_training.py`) |
| `prepare_dataset.py` on the real splits with the MiniCPM tokenizer | measured (`--tokenizer` run; the figures below come from the collator's own measurement over the real records) |
| `train.py --dry-run` (default LoRA, `--tuning full`, `--seq-len 12288`) | measured, estimator output only |
| **train-only step**, real weights, seq padded to exactly **8192**, bf16, grad-ckpt, micro-batch 1, chunked CE 1024, no `PYTORCH_CUDA_ALLOC_CONF` | **measured on GPU**: OK — peak allocated **14.37 GB**, peak reserved **15.17 GB**, loss 2.911356, 3.78 s wall |
| same **train-only step** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **measured on GPU**: OK — peak allocated **14.37 GB**, peak reserved **14.88 GB**, loss 2.911356, 3.82 s wall |
| **eval-only step** (`SFTTrainer.evaluate`, chunked), seq padded to exactly 8192, no env var | 8.34 GB allocated / 9.48 GB reserved, 1.08 s (re-run: 8.34 / 9.48, 1.10 s) |
| **train + eval + save cycle** (one train step → one eval step → one epoch-end adapter save, same process), padded to 8192, no env var | **measured**: 14.37 GB allocated / **15.17 GB reserved** peak, 5.03 s total — **1.40 GB reserved headroom** on the 16.57 GB card |
| epoch-end adapter save inside that cycle | **measured**: 245 ms, 9 files written |
| seq 12288 on the same card | **measured earlier in this repo, not re-run on 2026-11-14**: OOM, "Tried to allocate 502.00 MiB; 14.02 GiB allocated, 123.88 MiB free" |
| seq 5632, chunking on vs off | **measured earlier in this repo, not re-run on 2026-11-14**: 14.98 / 15.17 GB off, 11.50 / 13.76 GB on (3.48 GB allocated saved) |
| seq 8192 with chunking OFF | **measured earlier in this repo, not re-run on 2026-11-14**: OOM — `ForCausalLMLoss` upcast the whole `[seq, vocab]` logits to fp32 (8192 × 130560 × 4 ≈ 3.98 GiB) in one allocation |
| chunked CE vs single-shot loss | **measured equivalent** (CPU): max relative deviation 8.6e-8 (all tokens supervised), 1.7e-7 (mixed `-100` masking), tolerance 1e-5 |
| chunked **eval** vs single-shot eval (chunks 0/1024/5/1) | **measured identical**: `val_loss`, `val_token_accuracy`, `val_target_tokens` all equal, relative difference 0.0 on a real tiny Llama and 0.0 on a synthetic oracle |
| adapter save + reload, greedy generation | **measured** on the real loaded model |
| QLoRA on GPU | **NOT TESTED** — `bitsandbytes` is not installed in the torch environment used here |
| A whole epoch, convergence, any leaderboard score | **not run** — measured worst case is one train step + one eval step + one save in the same process |
| `precision: float16` at seq 8192 | **measured earlier in this repo, not re-run on 2026-11-14**: OOM on the first backward ("… 14.22 GiB in use") while the identical bf16 run fits, so fp16 buys nothing on this card; `scripts/train.py` warns on any fp16-on-CUDA run and refuses to launch when the preflight verdict is `does_not_fit` |

The four `8192`-padded VRAM cases come from the sweep documented in §4; each ran
in a fresh process with the batch padded to an explicit 8192 tokens, so the
sequence length is identical across runs. Rows marked "measured earlier in this
repo" are older runs that were not re-executed on 2026-11-14.

Test suite on the current tree: `python3 tests/run_tests.py` → **516 passed / 13
skipped / 0 failed**; `.venv/bin/python -m pytest -q` → **522 / 7 / 0**;
`PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q` → **528 / 1 / 0**.
`/tmp/cudaenv` (`torch 2.15.0+cu132`, transformers 5.22.2) is the torch
interpreter of record. Docker is **NOT RUN / UNVERIFIED**.

---

## 1. Prepare the SFT dataset

`prepare_dataset.py` converts gold trajectories into *causal decision points*:
one (prompt, target action) pair per agent step.

```bash
# real data with the real MiniCPM tokenizer (exact token counts, no weights loaded)
python3 scripts/prepare_dataset.py \
    --data-root ./data --split train --output outputs/datasets \
    --tokenizer openbmb/MiniCPM5-2B --val-ratio 0.1 --seed 42 --max-seq-length 8192

# without a tokenizer the counts are chars/4 ESTIMATES and the report says so
python3 scripts/prepare_dataset.py --data-root ./data --split train --output outputs/datasets

# synthetic fixture (offline)
python3 scripts/prepare_dataset.py \
    --data-root tests/fixtures/synthetic_lifemate --split train --output /tmp/sft_synth
```

Writes `outputs/datasets/train/`:

```
train.jsonl   # {"sample_id","step_index","messages":[system,user],"target","meta"}
val.jsonl
manifest.json # counts, seed, ratio, id hashes, group hash, fit/truncation block
stats.json    # decision-point accounting + drop reasons + token-size summary
```

Prompts are rendered at preparation time and stored verbatim, so training cannot
drift from the inference-time renderer.

### Thinking is disabled in the rendered prompt, on purpose

`training/collator.py::render_prompt_text` and `models/generation.py::HFGenerator.build_prompt`
both call the chat template with `enable_thinking=False` from the model profile.
For MiniCPM5-2B that is an **officially supported mode**: the template renders an
empty, already-closed think block (`\x3cthink\x3e\n\n\x3c/think\x3e\n\n`), which is
byte-identical to what the template emits for a training assistant turn without
`reasoning_content`. Measured tail ids: `[8, 130063, 9, 130063]`.

A tokenizer that rejects the kwarg is retried without it; the bare-tokenizer
ChatML fallback appends the same closed suffix. A tokenizer that *has* a template
and raises is a hard `LifeMateError` on both sides — never a silent reformat.

### Decision-point rules

| Gold trajectory | Emitted targets |
|---|---|
| N calls, all results present, last call ≠ `sys.ask_missing_info` | `call_0 … call_{N-1}`, then `finish` |
| last call is `sys.ask_missing_info` | `…, ask_missing_info` and **no** `finish` |
| zero calls (out of scope) | a single `finish` |
| last result missing | calls only, **no** `finish`; sample flagged `visibility_limited` |

Prompt *k* contains observations of steps `0..k-1` only. It never contains step
*k*'s own observation, a later observation, or `label.final_state`
(`assert_no_label_leak` hard-fails on violation).

`visibility_limited` and `unobservable x-ref` are reported in `stats.json`;
unobservable targets are **dropped**, never repaired, and the gold value is never
injected into a prompt.

---

## 2. Train — the default local workflow

```bash
python3 scripts/train.py
```

No arguments means: `--config configs/training_lora.yaml` (MiniCPM5-2B, LoRA
r=16/alpha=32/dropout=0.05, 8192 tokens, 3 epochs), because `--config` defaults
to that file and `training.tuning_mode: lora` in it. `--tuning` defaults to the
config value; when passed, it overrides the config **and** sets
`training.quantization` explicitly (`qlora` → `nf4`, everything else → `none`).
The run never silently downgrades `full` → LoRA.

```bash
# plan only: no weights loaded, no training launched
python3 scripts/train.py --dry-run

# real run, explicit model override (Qwen3.5-4B stays one flag away)
python3 scripts/train.py --model Qwen/Qwen3.5-4B
python3 scripts/train.py --config configs/training_lora_qwen.yaml

# other useful flags (all verified in `--help`)
python3 scripts/train.py --dataset-dir outputs/datasets/train   # explicit dataset
python3 scripts/train.py --output-dir /tmp/ckpt --run-name lora-8192
python3 scripts/train.py --max-steps 20                          # short run
python3 scripts/train.py --seq-len 5632                          # alias: --max-seq-length
python3 scripts/train.py --device cuda --seed 7
python3 scripts/train.py --smoke                                 # tiny CPU model, 2 steps
python3 scripts/train.py --config configs/training_full.yaml --tuning full
```

Full fine-tuning still exists and trains **every** parameter
(`configure_full_finetuning` raises unless `trainable == total`), but it does not
fit this card: `--dry-run --tuning full` on the DEFAULT config (MiniCPM5-2B,
2,516,756,480 params, seq 8192, micro-batch 1, grad-ckpt, vocab 130560, shape
from the `minicpm5` profile fallback) prints **TOTAL 45.76 GB**
(weights 10.07 + gradients 10.07 + optimizer 20.13 + activations 2.82 + bf16
logits 2.14 + fp32 chunked logits 0.53) and `verdict: does_not_fit`. That is
estimator arithmetic, not a measured peak.

### Multimodal towers policy (Qwen3.5 variants only)

If a multimodal variant exposing `visual`, `vision_tower`, `audio_tower` is
loaded: `everything` (default) trains all parameters; `lm_only` explicitly freezes
the towers and trains only the LM. The policy is printed at startup and reflected
in the trainable count. Silent partial training is forbidden.

Exit codes: `0` ok · `1` config/training error · `2` dataset missing · `3`
environment cannot run the request (no torch/CUDA, model unloadable) · `130`
interrupted after saving `interrupt-<step>`.

---

## 3. Sequence length: 8192, and why not 12288

Measured with the real MiniCPM tokenizer over the real SFT records:

| full-sequence tokens | train | val |
|---|---|---|
| p50 | 6198 | — |
| p90 | 8435 | — |
| p95 | 9089 | — |
| p99 | 10075 | 9309 |
| max | 11037 | 9623 |

The **target alone is never the constraint** (max 165 tokens including the
terminator).

What each budget actually costs, at the 4368 train / 482 val decision points:

| `max_seq_length` | train | val | measured memory (train step, chunked CE) |
|---|---|---|---|
| 5632 | 3760 kept, 2101 truncated (48.1 %), 608 dropped (13.92 %) | 415 kept, 232 truncated, 67 dropped | 11.50 GB allocated / 13.76 GB reserved with chunking on (14.98 / 15.17 GB with it off) |
| **8192** | **4298 kept, 488 truncated (11.17 %), 70 dropped `target_value_truncated` (1.60 %)** | **477 kept, 41 truncated (8.51 %), 5 dropped (1.04 %)** | **14.37 GB allocated / 15.17 GB reserved peak for the train+eval+save cycle; 1.40 GB headroom on 16.57 GB** (train-only step: same 14.37 / 15.17 GB) |
| 12288 | 0 truncated, 0 dropped | 0 truncated, 0 dropped | **measured OOM** on this 16.57 GB card |

**0 kept prompts lose the `### <tool> (kind=...)` schema block or the
`# Công cụ khả dụng` catalog** at 8192. The 70 train / 5 val drops are
`target_value_truncated`: a deeper head cut removed the history value a
`tool_call` target copies, so the record would supervise an identifier the model
was never shown. **That is a deliberate choice at 8192, not an oversight** — the
alternative is picking a budget that OOMs.

Two different outcomes are never conflated in `manifest.json`:

* **truncated** — `kept=True`, prompt middle cut, request prefix and the tool
  catalog pinned;
* **dropped** — `kept=False`, counted by exact reason (`target_too_long`,
  `target_value_truncated`). A target is never truncated.

Qwen-era figures, which do **not** apply to MiniCPM: with the Qwen3.5-4B
tokenizer (tokenizer only; no Qwen weights were loaded), 8192 gave 47 truncated
and 0 schema loss; 4096 gave 3520 truncated with 846/3305 schema-block loss.

---

## 4. Memory reality on a 16 GB card

Measured on the local RTX 5070 Ti (16.57 GB, `sm_120`, bf16 supported,
`torch 2.15.0+cu132`, `transformers 5.22.2`), MiniCPM5-2B + LoRA bf16,
gradient checkpointing on, micro-batch 1:

| Configuration | Result |
|---|---|
| **train-only step**, seq padded to exactly 8192, chunked CE 1024, no `PYTORCH_CUDA_ALLOC_CONF` | **OK — 14.37 GB allocated / 15.17 GB reserved peak**, loss 2.911356, 3.78 s wall |
| **train-only step**, same batch, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **OK — 14.37 GB allocated / 14.88 GB reserved peak**, loss 2.911356, 3.82 s wall (0.29 GB less reserved, identical allocation) |
| **eval-only step** (`SFTTrainer.evaluate`, chunked), padded to 8192, no env var | 8.34 GB allocated / 9.48 GB reserved, 1.08 s (re-run identical) |
| **train step + eval step + epoch-end adapter save**, same process, padded to 8192, no env var | **OK — 14.37 GB allocated / 15.17 GB reserved peak**, 5.03 s total, **1.40 GB reserved headroom** on the 16.57 GB card |
| epoch-end adapter save (inside the cycle above) | 245 ms, 9 files |
| seq 8192, chunking OFF | **OOM** — `ForCausalLMLoss` upcast the whole `[seq, vocab]` logits to fp32 (8192 × 130560 × 4 ≈ 3.98 GiB in one allocation) |
| seq 12288, chunked CE | **OOM** — "Tried to allocate 502.00 MiB; 14.02 GiB allocated, 123.88 MiB free" |
| seq 5632, chunking OFF | 14.98 GB allocated / 15.17 GB reserved |
| seq 5632, chunking ON | 11.50 GB / 13.76 GB — chunking saved **3.48 GB allocated** |

The last four rows are earlier measurements in this repo; they were **not**
re-run in the 2026-11-14 sweep and their sequence lengths came from the
collator on real records, not from an explicit pad.

### Reproducing the four cases

Every case is a fresh process, loads `models/MiniCPM5-2B` through
`load_model_and_tokenizer` + `configure_lora`, builds its batch with
`build_supervised_example` and then **pads it to exactly 8192 tokens**
(`input_ids` = pad id, `attention_mask` = 0, `labels` = -100), and uses the same
longest-fitting record of the same dataset (`outputs/datasets/train`, seed 42):

```bash
# 1. train-only, no allocator override
/tmp/cudaenv/bin/python /tmp/measure_step.py train
# 2. train-only, expandable segments
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /tmp/cudaenv/bin/python /tmp/measure_step.py train_xseg
# 3. eval-only
/tmp/cudaenv/bin/python /tmp/measure_step.py eval
# 4. train -> eval -> epoch-end adapter save, one process
/tmp/cudaenv/bin/python /tmp/measure_step.py cycle
```

All four **succeeded without** `PYTORCH_CUDA_ALLOC_CONF`; case 4 therefore needed
no retry with it. `/tmp/measure_step.py` is a session-local harness (not in the
repo); it is ~200 lines around the four calls above, so re-deriving it is cheap if
the numbers are ever re-measured. The train step's loss is 2.911356 in cases 1, 2
and 4. The eval
step reports `val_loss` 0.760582 standalone and 0.55708 inside case 4 — the same
batch, but case 4 evaluates **after** the adapter has taken one optimizer step.
The earlier 0.059119 / 14.44 GB figures came from a run whose batch length was
whatever the collator produced, not an explicit pad; they are superseded and
deleted rather than kept as a second variant.

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: OPTIONAL

Verdict for the shipped default: **OPTIONAL — not recommended, not required.**
Measured delta on this card at seq 8192, train-only step:

| | peak allocated | peak reserved |
|---|---|---|
| unset | 14.37 GB | **15.17 GB** |
| `expandable_segments:True` | 14.37 GB | **14.88 GB** |

It saves 0.29 GB of *reserved* (fragmentation) memory and **nothing** of
allocated memory, and the worst case (case 4: train + eval + save) already fits
without it at 15.17 GB reserved against 16.57 GB — 1.40 GB headroom. Do **not**
put it in the documented training command; add it only if a run on a busier card
OOMs while `nvidia-smi` still shows free memory (fragmentation, not demand).

The preflight estimator (`python3 scripts/train.py --dry-run`) reports
**10.68 GB** for the shipped LoRA 8192 configuration (weights 5.03 + adapter 0.03
+ gradients 0.03 + optimizer 0.10 + activations 2.82 + **bf16 logits 2.14** +
**fp32 chunked logits 0.53**, vocab 130560, `fits`), against a measured 14.37 GB
allocated / 15.17 GB reserved — the activation heuristic
(`layers * seq * hidden * 2`, k=2) is not the same accounting as the real stepped
graph, so the total is still optimistic by **3.69 GB** here. Both logits terms are
priced because the chunked-CE path allocates both: the compute-dtype
`[seq, vocab]` head output the trainer asks for by withholding `labels`
(`logits_compute_gb`, 2.14 GB, alive through `backward()`) and the fp32 upcast of
one `[1024, vocab]` slice (`logits_chunked_gb`, 0.53 GB). The whole-sequence fp32
upcast would be 4.28 GB (`logits_unchunked_gb`, reported for contrast only).
At 12288 it prints **13.16 GB** for a configuration that in fact OOMs on this
card.
**Treat the estimator as a plan, the measured peak as the answer.**

Before a real launch, measure free VRAM immediately — free VRAM on a shared card
is not a constant:

```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Rule: free VRAM must exceed the **measured 15.17 GB reserved peak of the
train+eval+save cycle** (plus whatever margin you want); if free is below that,
wait and measure again rather than panic-reconfiguring the run.

### Why `loss_chunk_tokens` exists — and why the shipped default is executable

`training.loss_chunk_tokens` (dataclass default **1024** in
`lifemate/config.py`, written explicitly in `configs/base.yaml`; the LoRA configs
inherit the default) makes the trainer compute cross-entropy in chunks along the
sequence dimension, capping the fp32 logits temporary at `chunk × vocab × 4`
bytes — 510 MiB at 1024 × 130560 instead of the 3.98 GiB whole-tensor upcast. It
is consumed by **both** the train step (`trainer.py::chunked_causal_lm_loss`) and
the validation step (`SFTTrainer.evaluate`).

* mathematically equivalent: per-chunk **sums** are accumulated and divided once
  by the supervised-token count — dividing per chunk and averaging would
  over-weight a short trailing chunk;
* measured deviation from the single-shot loss: max 8.6e-8 with every token
  supervised, 1.7e-7 with mixed `-100` masking (tolerance 1e-5);
* **`loss_chunk_tokens: 0` restores the old single-shot path** — the model
  computes its own loss, and at seq 8192 that path OOMs on this card.

**The eval path had to be chunked too.** `SFTTrainer.evaluate` used to call the
model WITH `labels`, which re-ran `ForCausalLMLoss`'s whole-vocabulary fp32 upcast
and then made a second full `[seq, vocab]` copy via
`logits[..., :-1, :].contiguous()`. Because `eval_strategy: epoch` is the shipped
default, that would have OOM'd at the end of epoch 1 — after paying for the whole
epoch. It now:

* forwards **without** `labels`, so the model's own loss is never invoked;
* runs `F.cross_entropy(..., reduction="sum", ignore_index=-100)` per sequence
  chunk and divides by the supervised-token count once;
* does `argmax` on the slice only, never on `[seq, vocab]`.

Shift semantics are unchanged (`logits[..., :-1, :]` against `labels[..., 1:]`).
Equivalence was measured, not assumed: identical `val_loss`,
`val_token_accuracy` and `val_target_tokens` for chunk widths 0 / 1024 / 5 / 1
(relative difference 0.0 on a real tiny Llama, 0.0 on a synthetic oracle). A
model returning no logits falls back to its own loss, exactly as before.

Measured on this card at seq 8192: the **eval step alone** costs 8.34 GB allocated
/ 9.48 GB reserved in ~1.0 s; **one train step plus one eval step plus one
epoch-end adapter save in the same process** peaks at 14.37 GB allocated /
15.17 GB reserved, leaving **1.40 GB reserved headroom** on 16.57 GB; the save
inside that cycle takes 245 ms. That combination — not the train step on its
own — is why the shipped `eval_strategy: epoch` default is executable on a 16 GB
card.

**What was NOT run:** a whole epoch. The measured worst case is one train step +
one eval step + one save in the same process.

### QLoRA caveat

`--tuning qlora` (or `training.quantization: nf4`) is implemented for real:
`BitsAndBytesConfig` NF4 with double quantisation, `prepare_model_for_kbit_training`,
no `.to()` on a quantized model, and the loader refuses 4-bit on a non-CUDA
device with an actionable error. **GPU execution was NOT tested**: `bitsandbytes`
is not installed in the torch environment used here, so no QLoRA number in this
repo is a measurement. A QLoRA run is a LoRA run with a 4-bit frozen base — the
adapter artifact is the same kind.

---

## 5. Hyper-parameters (starting point, **not tuned**)

No hyper-parameter search has been run and no epoch has completed. These are
conservative defaults.

| Setting | Full | LoRA (shipped default) | Rationale |
|---|---|---|---|
| `learning_rate` | `5e-6` | `2e-4` | Full FT of a 2.5B model needs a small LR; LoRA tolerates ~40× more because only adapters move. |
| `per_device_train_batch_size` | 1 | 1 | Memory-bound at seq 8192: the measured train+eval+save cycle peaks at 14.37 GB allocated / 15.17 GB reserved on a 16.57 GB card (1.40 GB reserved headroom). |
| `gradient_accumulation_steps` | 8 | 8 | Effective batch 8 sequences. |
| `num_train_epochs` | 2 | 3 | Short action-format task. |
| `max_seq_length` | 8192 | 8192 | See §3. |
| `warmup_ratio` | 0.05 | 0.03 | Standard warmup. |
| `lr_scheduler_type` | cosine | cosine | `linear`/`constant` also supported. |
| `weight_decay` | 0.01 | 0.01 | Matrices only (biases/norms excluded). |
| `precision` | bfloat16 | bfloat16 | bf16 avoids fp16 loss-scaling failures; CPU runs use fp32. Full FT keeps fp32 master weights — AdamW's update lands in the parameter tensor, so a bf16 master quantises a 5e-6 step away. |
| `gradient_checkpointing` | true | true | ~5× smaller activation term, ~20% slower. Modeled by the estimator (`k = 2` vs `10`). |
| `gradient_clip_norm` | 1.0 | 1.0 | Guards occasional spikes in tool-call JSON. |
| `save_strategy` / `eval_strategy` | epoch | epoch | Few epochs. Executable at 8192 because `evaluate` is chunked like the train step: the train+eval+save cycle peaks at 14.37 GB allocated / 15.17 GB reserved (§4). |
| `save_total_limit` | 2 | 2 | Older `step-*` dirs pruned after each save. |
| `loss_chunk_tokens` | 1024 | 1024 | See §4. |
| LoRA `r` / `alpha` / `dropout` | — | 16 / 32 / 0.05 | Conventional starting point. |
| LoRA `target_modules` | — | `null` → profile list | MiniCPM5-2B: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — **measured** on the loaded model: 588 adapter tensors, **25,116,672 trainable params = 0.9881 %** of 2,541,873,152 total-with-adapter. |

`lora.target_modules: null` on MiniCPM resolves to the profile's 7 projections;
on Qwen it stays architecture-aware discovery (12 suffixes / 248 Linear modules
/ 32,464,896 trainable — **Qwen numbers, derived from the released `text_config`
on a meta-device build (no Qwen weights on this machine), not from MiniCPM**).
If a model exposes projections its profile does not list, training fails and says
which knob to turn instead of adapting a silent subset.

---

## 6. Checkpoints, resume, interruption

Checkpoints land in `outputs/checkpoints/<run_name>/step-<n>/` (`run_name`
defaults to the tuning mode, so a default run writes `.../lora/step-<n>`), and
contain the model or adapter, the tokenizer, `lifemate_meta.json`,
`training_state.json` and `optimizer_state.pt`. `training_state.json` is written
last, so a checkpoint without state is visibly incomplete.

### Identity validation

* `lifemate_meta.json` records `model_name` and `model_profile`; the adapter
  attach path (`loader._attach_lora_adapter`) and `export_model.py --merge-lora`
  both call `checkpoint.assert_adapter_base_matches` **before any weight is
  touched**. A MiniCPM adapter on a Qwen base (or the reverse) is a hard
  `LifeMateError`; comparison is by profile family, so `openbmb/MiniCPM5-2B` and
  the local `models/MiniCPM5-2B` count as the same base. When neither record
  exists the check is a no-op and says so (`{"recorded": False}`).
* The loader fails when the weight files do not provide every parameter
  (`missing_keys`, tied weights excluded): a truncated download would otherwise
  silently fine-tune a partly random model.

### Resume semantics — exactly what exists

```bash
# resume from ONE explicit step directory
python3 scripts/train.py --resume outputs/checkpoints/lora/step-120
# alias
python3 scripts/train.py --resume-from outputs/checkpoints/lora/step-120
```

`--resume` takes a **`step-<n>` directory you name**. There is **no auto-latest**:
`CheckpointManager.latest()` exists but no CLI path calls it. Resuming requires
`training_state.json` in that directory and refuses when:

* the recorded step is not positive;
* the checkpoint is LoRA and the run is full, or the reverse;
* the dataset manifest hash differs (the split changed);
* `config_hash` differs — it hashes `training.*` minus bookkeeping
  (`resume_from_checkpoint`, `output_dir`, `logging_steps`, `save_strategy`,
  `eval_strategy`, `save_total_limit`, `report_to`, `dataloader_num_workers`,
  `method`), `model.*` minus `device`/`tokenizer_name`, and `lora.*` only in LoRA
  mode. So **may change**: dataset mount, device, W&B names, ASR/inference paths,
  checkpoint cadence. **May not change**: lr, weight decay, epochs, batch,
  grad accumulation, warmup, scheduler, gradient checkpointing, clip norm, seed,
  `max_seq_length`, `validation_ratio`, optimizer, precision, tuning mode, model
  identity/precision, `loss_chunk_tokens` (it is not in the ignore list, so it
  moves the digest), and (in LoRA mode) LoRA hyper-parameters.

Optimizer moments and RNG state are restored, so a resumed run continues rather
than restarting from zero moments.

### Interruption and crash safety

* `SIGINT`/`SIGTERM` set a cooperative stop flag; the current optimizer step
  finishes, `interrupt-<step>` is written, the process exits `130`, and the
  printed hint is `python3 scripts/train.py --config <cfg> --resume <dir>`.
* An unhandled exception writes `crash-<step>` metadata (best effort) and re-raises.
* A populated `<output_dir>/<run_name>/step-<n>` is never overwritten.

---

## 7. Metrics logging (optional)

`training.report_to: ["wandb"]` — or `wandb.enabled: true`, see
`configs/training_lora_wandb.yaml` — routes metrics through
`lifemate.tracking.WandbTracker`, which **redacts before sending**:
`api_key|token|secret|password|credential` keys become `[REDACTED]`, PII keys
(`user_profile`, `audio_text`, `messages`, `history`, …) become `[DROPPED_PII]`.
`scripts/train.py` never calls `wandb.init` itself.

W&B is **not required**: with it disabled, uninstalled or offline, metrics fall
back to `<output_dir>/<run_name>/metrics.jsonl` (same redaction, one warning).
`--smoke` uses a null tracker.

---

## 8. Fail-fast conditions

| Condition | Behaviour |
|---|---|
| `--tuning` invalid | argparse usage error, exit 2 |
| Dataset dir has no `train.jsonl` | exit 2 + the exact `DatasetNotFound` text + the prepare command |
| Zero usable examples after masking | `LifeMateError` naming the drop reasons |
| `target` too long for `max_seq_length` | example dropped as `target_too_long`; the target is **never** truncated |
| Model returns no `loss` | `LifeMateError` (labels were probably not passed) — unless `loss_chunk_tokens > 0`, where the loss is computed from logits |
| Full mode, any parameter frozen | `LifeMateError` (`trainable != total`) |
| LoRA target set empty | `LifeMateError` listing discovered module names |
| LoRA attached but a non-`lora_` param trainable | `LifeMateError` |
| Profile lists fewer projections than the model has | `LifeMateError` naming the extras and the knob |
| `quantization: nf4` on a non-CUDA device, or `bitsandbytes` missing | `LifeMateError` with the install hint |
| Resume: LoRA↔full mismatch, manifest hash change, `config_hash` change, missing `training_state.json` | `LifeMateError` |
| Adapter attached or merged onto another family's base | `LifeMateError` before any weight is touched |
| No torch / no CUDA for a real run | exit 3 with a pointer to `--dry-run` / `--smoke` |

---

## 9. Checking the training signal is honest

1. **Prompt tokens never reach the loss.** `labels` are `-100` over the prompt
   and the target verbatim afterwards (`assert_loss_masking`), and
   `test_gradient_independent_of_prompt_tokens` asserts identical gradients for
   two different prompts with the same target.
2. **No causality leak.** `tests/test_trajectory.py` asserts by marker string that
   prompt *k* contains observations `< k` only and that `final_state` never
   appears.
3. **Prompt/target bytes equal what inference serves.** The supervised region is
   exactly the target plus the template's terminator: measured last supervised id
   **130073** (`<|im_end|>`), id 1 (`</s>`) never in the supervised region, `<s>`
   exactly once, and target-standalone ids identical to target-in-context ids.

Validation reports `val_loss` and `val_token_accuracy`; neither is the competition
metric — use `scripts/evaluate.py`.

### CPU tiny-model validation

```bash
# 0. Build a tiny random model + tokenizer in /tmp/tiny_lifemate (run once)
python3 -c "from pathlib import Path; from tests.test_tiny_e2e import make_tiny_model_and_tokenizer; make_tiny_model_and_tokenizer(Path('/tmp/tiny_lifemate'))"

# 1. Prepare SFT records with the tiny tokenizer
python3 scripts/prepare_dataset.py --data-root tests/fixtures/synthetic_lifemate --split train --output /tmp/sft_tiny --tokenizer /tmp/tiny_lifemate

# 2. Full fine-tune 4 steps on CPU
python3 scripts/train.py --model /tmp/tiny_lifemate --dataset-dir /tmp/sft_tiny/train --output-dir /tmp/ckpt_full --max-steps 4 --seq-len 512 --config configs/training_full.yaml --device cpu

# 3. Resume from step 4 to step 6
python3 scripts/train.py --model /tmp/tiny_lifemate --dataset-dir /tmp/sft_tiny/train --output-dir /tmp/ckpt_full --resume /tmp/ckpt_full/full/step-4 --max-steps 6 --seq-len 512 --config configs/training_full.yaml --device cpu

# 4. LoRA 4 steps on CPU (proves adapter gradient flow)
python3 scripts/train.py --model /tmp/tiny_lifemate --dataset-dir /tmp/sft_tiny/train --output-dir /tmp/ckpt_lora --max-steps 4 --tuning lora --device cpu

# 5. Export/merge the adapter
python3 scripts/export_model.py --ckpt /tmp/ckpt_lora --merge-lora --output /tmp/merged
```

*Coverage limit*: a tiny random model proves data pipelines, collators, gradient
backprop, optimizer steps, checkpointing, resume and adapter merge plumbing work
end to end on CPU. It says nothing about generation quality or convergence.

---

## 10. Files in this slice

```
src/lifemate/data/trajectory_builder.py causal decision points, alignment, leak guard
src/lifemate/data/split_builder.py      deterministic stratified split + leakage guard
src/lifemate/models/profiles.py         per-family prompt/terminator/LoRA/decode facts
src/lifemate/training/collator.py       masking, truncation/drop policy, padding
src/lifemate/training/full_finetuning.py
src/lifemate/training/lora_finetuning.py (configure_lora / configure_qlora)
src/lifemate/training/memory_preflight.py
src/lifemate/training/trainer.py        own SFT loop, chunked CE, checkpoints, resume
src/lifemate/training/callbacks.py      StopController, trackers
configs/{base,training_full,training_lora,training_lora_qwen,training_lora_wandb}.yaml
scripts/prepare_dataset.py  scripts/train.py
tests/test_trajectory.py  tests/test_training.py  tests/test_model_profiles.py
```
